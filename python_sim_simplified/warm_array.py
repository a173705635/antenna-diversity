"""Paired sampled-code array experiment; nominal sky/attitude and code-period aid.

Independent coherent snapshots, not a continuous carrier-tracking receiver.
Startup eligibility is observed PVT availability, not authentication.
"""
import numpy as np
from signal_recovery import observe, position_solution, FREQUENCY


def calibration(x, a, spectrum, coarse):
    _, delays, _, _ = observe(x, spectrum, a/len(a), coarse)
    wave = np.fft.ifft(spectrum*np.exp(-2j*np.pi*delays[:, None]*FREQUENCY), axis=1)
    prompt = np.linalg.lstsq(wave.T, x.T, rcond=None)[0].T
    left, singular, right = np.linalg.svd(prompt/a, full_matrices=False)
    gain = left[:, 0]
    return gain/gain[0]


def weights(x, a, startup_cov):
    m = len(a)
    eigenvalues, vectors = np.linalg.eigh(x@x.conj().T/x.shape[1])
    from scipy.linalg import eigh
    change, generalized = eigh(x@x.conj().T/x.shape[1], startup_cov)
    if change[-1] < 3.:
        return a/m, np.ones(a.shape[1], bool)
    v = startup_cov@generalized[:, -1]
    v /= np.linalg.norm(v)
    projected = a-v[:, None]*(v.conj()@a)[None, :]
    response = np.sum(a.conj()*projected, axis=0).real
    return projected/(response+.01)[None, :], response/m >= .05


def estimate(x, spectrum, w, coarse, prior):
    return observe(x, spectrum, w, coarse, prior_delay=prior)


def process(record, subset):
    a = record['nominal'][subset]
    samples = record['samples'][:, subset]
    spectrum, coarse = record['spectrum'], record['coarse']
    first = record['startup_count']
    gain = calibration(samples[0], a, spectrum, coarse)
    corrected = samples/gain[None, :, None]
    startup_cov = np.mean([x@x.conj().T/x.shape[1] for x in corrected[:first]], axis=0)
    positions, ranges, masks, qualities, weight_norms = [], [], [], [], []
    prior = None
    range_state = np.zeros(len(spectrum))
    precision = np.zeros(len(spectrum))
    previous_admitted = np.zeros(len(spectrum), bool)
    startup_power = []
    for k, x in enumerate(corrected):
        w, admitted = (a/len(a), np.ones(a.shape[1], bool)) if k < first else weights(x, a, startup_cov)
        r, delay, q, reacquired = estimate(x, spectrum, w, coarse, prior)
        wave = np.fft.ifft(spectrum*np.exp(-2j*np.pi*delay[:, None]*FREQUENCY), axis=1)
        prompt = np.sum((w.conj().T@x)*wave.conj(), axis=1)/x.shape[1]
        power = abs(prompt)**2
        if k < first:
            startup_power.append(power)
        else:
            admitted &= power >= .1*np.mean(startup_power, axis=0)
        admitted &= q >= 10.
        if k >= first:
            innovation = r-range_state
            centered = innovation-np.median(innovation)
            scale = 1.4826*np.median(abs(centered))
            admitted &= abs(centered) <= 4.5*scale
        continuous = admitted & previous_admitted
        precision = np.where(continuous, .8*precision, 0.)
        alpha = q/(q+precision)
        range_state = alpha*r+(1-alpha)*range_state
        precision += q
        previous_admitted = admitted.copy()
        r = range_state.copy()
        p, condition, residual = position_solution(record['satellites'][k], r, admitted)
        if admitted.sum() >= 4 and condition <= 1000.:
            from positioning import solve_position
            from run_gps_spp import receiver_reference
            reference, rotation = receiver_reference()
            fit = solve_position(record['satellites'][k][admitted], r[admitted], sigma_m=1/np.sqrt(q[admitted]))
            p = (fit['position_m']-reference)@rotation.T
        positions.append(p); ranges.append(r); masks.append(admitted); qualities.append(q)
        weight_norms.append(np.sum(abs(w)**2, axis=0))
        prior = delay
    positions = np.array(positions)
    startup_spread = np.max(np.linalg.norm(positions[:first, :2]-np.mean(positions[:first, :2], axis=0), axis=1))
    initialized = bool(np.all(np.isfinite(positions[:first])) and startup_spread <= 20.)
    if not initialized:
        positions[first:] = np.nan
    return dict(position=positions, ranges=np.array(ranges), admitted=np.array(masks),
                quality=np.array(qualities), weight_norm=np.array(weight_norms),
                initialized=initialized, gain=gain)
