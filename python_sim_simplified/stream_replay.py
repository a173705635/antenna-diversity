"""Sample-index fault replay and SDCP arc consumer; no automatic reacquisition.

Counters identify source samples at block boundaries, not host receive order.
The source is a bank of IF tones, not a GNSS tracking loop. A reference event
represents external admission, never inferred authenticity from clean truth.
"""
import numpy as np

from sdcp_arc import SDCPArc


def source(*, channels, sample_step, reset_rad, reported, event_epoch=8, fault_channel=-1):
    """A positive sample_step drops samples; a negative one repeats samples."""
    fs, block = 100000., 100
    epochs = np.arange(20)
    nominal = epochs[:, None]*block + np.zeros(channels, dtype=int)
    actual = nominal.copy()
    actual[event_epoch:, fault_channel] += sample_step
    reset = np.zeros_like(actual, dtype=float)
    reset[event_epoch:, fault_channel] = reset_rad
    doppler = np.array([-3000., 0., 4000.])
    # Explicit finite IF samples: changed source indices persist after the event.
    indices = actual[..., None]+np.arange(block)
    wave = np.exp(2j*np.pi*(10000.+doppler)[None, None, None, :]
                  * indices[..., None]/fs)
    local = np.exp(-2j*np.pi*10000.*(nominal[..., None]+np.arange(block))/fs)
    baseband = wave*local[..., None]*np.exp(-1j*reset[..., None, None])
    tracking = np.zeros_like(actual)
    if reported and reset_rad != 0.:
        tracking[event_epoch:, fault_channel] = 1
    observations = dict(time_s=epochs*block/fs, block_samples=block,
                        source_start=actual.copy() if reported else nominal.copy(),
                        tracking_id=tracking, phasor=baseband[:, :, 0, :],
                        quality=np.ones(actual.shape, dtype=bool))
    truth = dict(actual_start=actual, nominal_start=nominal, reset_rad=reset,
                 event_epoch=event_epoch, fs_hz=fs, doppler_hz=doppler,
                 samples=baseband)
    return observations, truth


def consume(observations, *, wavelength_m, reference_epochs):
    """Causal pairwise consumer, given only published observables and admissions.

Counter discontinuity invalidates the current arc. Continuity on the next
block does not approve a reference. Unreported events are not identifiable
from these metadata alone. No geometry or phase-jump detector is implied.
"""
    times = observations['time_s']
    counts = observations['source_start']
    channels = counts.shape[1]
    arcs = [SDCPArc(f'pair-0-{a}', .0015) for a in range(1, channels)]
    reference = [None]*(channels-1)
    rows = []
    for k, time in enumerate(times):
        continuous = (np.ones(channels, dtype=bool) if k == 0 else
                      counts[k]-counts[k-1] == observations['block_samples'])
        for a, arc in enumerate(arcs, 1):
            state = arc.step(float(time),
                             quality_valid=bool(observations['quality'][k, [0, a]].all()),
                             phase_state_valid=bool(continuous[[0, a]].all()),
                             tracking_ids=tuple(observations['tracking_id'][k, [0, a]]),
                             calibration_id='fixed-synthetic-calibration')
            pair = observations['phasor'][k, a]*np.conj(observations['phasor'][k, 0])
            if k in reference_epochs and state['state'] == 'await_reference':
                arc.accept_reference(state['arc_id'], f'ref-{k}', f'admission-{k}')
                reference[a-1] = pair.copy()
                state.update(state='reference_usable', reference_id=arc.reference_id,
                             evidence_id=arc.evidence_id)
            residual = (-np.angle(pair*np.conj(reference[a-1]))*wavelength_m/(2*np.pi)
                        if state['state'] == 'reference_usable' else None)
            rows.append(dict(epoch=k, channel=a, time_s=float(time), **state,
                             counter_continuous=bool(continuous[[0, a]].all()),
                             residual_m=None if residual is None else residual.tolist()))
    return rows
