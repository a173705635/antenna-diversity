"""Two independent receiver chains; compare tracked NCOs at common times."""
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from acquisition import acquire
from dfe import process_dfe
from gnss import generate_scenario
from presets import default_config
from rf import process_rf
from tracking import track


def compare_phases(first, second, fs_hz, samples_per_epoch):
    """Evaluate the second NCO at first-channel block starts, without extrapolation.

    An NCO is piecewise linear within a block; loop corrections can jump at its
    boundary. This uses that block's actual pre-update frequency, not linear
    interpolation across a correction. No scenario truth enters alignment.
    """
    t1 = np.asarray(first['epoch_start_samples'])
    t2 = np.asarray(second['epoch_start_samples'])
    if not len(t1) or not len(t2):
        raise ValueError('both channels require tracking epochs')
    index = np.searchsorted(t2, t1, side='right') - 1
    safe_index = np.clip(index, 0, len(t2)-1)
    valid = (index >= 0) & (t1 < t2[safe_index] + samples_per_epoch)
    times = t1[valid]
    if not len(times):
        raise ValueError('no overlapping tracking time')
    index = index[valid]
    second_phase = np.asarray(second['carrier_nco_phase_rad'])[index] + (
        2*np.pi*np.asarray(second['carrier_nco_frequency_hz'])[index]
        * (times-t2[index])/fs_hz
    )
    difference = second_phase - np.asarray(first['carrier_nco_phase_rad'])[valid]
    # Branch chosen from observations; no truth-assisted integer-cycle fixing.
    legacy_difference = np.unwrap(np.angle(np.exp(1j*difference)))
    return {'time_s': times/fs_hz, 'phase_difference_rad': legacy_difference,
            'raw_phase_difference_rad': difference}


def statistics(phase, wavelength, expected):
    error = np.angle(np.exp(1j*(phase-expected)))
    scale = wavelength/(2*np.pi)
    return {
        'epochs': len(phase),
        'max_deg': float(np.degrees(np.max(phase))),
        'max_abs_deg': float(np.degrees(np.max(abs(phase)))),
        'mean_deg': float(np.degrees(np.mean(phase))),
        'std_deg': float(np.degrees(np.std(phase))),
        'max_m': float(np.max(phase)*scale),
        'mean_m': float(np.mean(phase)*scale),
        'std_m': float(np.std(phase)*scale),
        'wrapped_error_mean_m': float(np.mean(error)*scale),
        'wrapped_error_std_m': float(np.std(error)*scale),
        'wrapped_error_rms_m': float(np.sqrt(np.mean(error**2))*scale),
        'wrapped_error_max_abs_m': float(np.max(abs(error))*scale),
    }


def run_case(thermal_noise=True, *, duration_ms=100, seed_base=42,
             multipath_paths=(), multipath_start_ms=None, receiver_config=None):
    if not isinstance(duration_ms, int) or duration_ms < 10:
        raise ValueError('duration_ms must be an integer >= 10')
    if not isinstance(seed_base, int) or seed_base < 0:
        raise ValueError('seed_base must be a nonnegative integer')
    if multipath_start_ms is not None and not 0 <= multipath_start_ms < duration_ms-10:
        raise ValueError('echo onset must leave a full 10 ms ramp within the record')
    cfg = default_config() if receiver_config is None else deepcopy(receiver_config)
    cfg.signal.duration_ms = duration_ms
    cfg.random_seed = seed_base
    cfg.tracking.max_epochs = max(cfg.tracking.max_epochs,
                                  int(np.ceil(duration_ms/cfg.tracking.coherent_ms)))
    cfg.phone.active_antennas = ('TL', 'TR')
    cfg.interferers = []
    cfg.signal.multipath_paths = list(multipath_paths)
    cfg.phone.channel_delay_ns = {name: 0. for name in cfg.phone.channel_delay_ns}
    cfg.phone.estimated_channel_delay_ns = dict(cfg.phone.channel_delay_ns)
    if not thermal_noise:
        cfg.rf.temperature_k = 0.
    antenna, truth = generate_scenario(cfg)
    if multipath_start_ms is not None:
        # Source construction only: turn generated echo components on BEFORE RF.
        # No path truth is passed to acquisition, tracking or SDCP weighting.
        progress = np.clip((truth['time_s']-multipath_start_ms*.001)/.01, 0., 1.)
        off = .5*(1.+np.cos(np.pi*progress))
        for path in truth['multipath_paths']:
            for channel in range(antenna.shape[0]):
                antenna[channel] -= off*path['component'][channel]
        del progress, off
    # Geometry alone: do not let antenna/RF errors enter the compensation truth.
    from antenna import direction_vector
    positions = cfg.phone.phase_centers_m
    baseline = np.array(positions['TR']) - np.array(positions['TL'])
    expected = -2*np.pi/cfg.signal.wavelength_m * np.dot(
        baseline, direction_vector(cfg.signal.true_direction))
    # Long records need only the array samples and this small post-hoc reference.
    del truth
    histories, diagnostics = [], []
    fs = None
    for index, name in enumerate(cfg.phone.active_antennas):
        seed = cfg.random_seed + index
        channel_rf = replace(cfg.rf,
            amplitude_error_db=float(np.broadcast_to(cfg.rf.amplitude_error_db, (2,))[index]),
            phase_error_deg=float(np.broadcast_to(cfg.rf.phase_error_deg, (2,))[index]))
        adc, rf_info = process_rf(antenna[index:index+1], channel_rf,
                                  cfg.reference_impedance_ohm, seed)
        baseband, dfe_info = process_dfe(adc, cfg.dfe, cfg.signal.fs_rf_hz,
                                        cfg.signal.if_hz)
        fs = dfe_info['fs_out_hz']
        if not np.any(baseband):
            return {'thermal_noise': thermal_noise, 'status': 'zero_dfe_output',
                    'failed_channel': name, 'adc_nonzero_samples': int(np.count_nonzero(adc)),
                    'dfe_nonzero_samples': 0,
                    'interpretation': 'No phase observable survives DFE quantization; not a tracking precision result'}
        acq = acquire(baseband[0], fs, cfg.signal.prn, cfg.signal.code_rate_hz,
                      cfg.acquisition)
        if not acq['success']:
            return {'thermal_noise': thermal_noise, 'status': 'acquisition_failed',
                    'failed_channel': name}
        history = track(baseband[0], fs, cfg.signal.prn, acq, cfg.tracking,
                        cfg.signal.carrier_hz, cfg.signal.code_rate_hz)
        if not history['success']:
            return {'thermal_noise': thermal_noise, 'status': 'tracking_failed',
                    'failed_channel': name, 'tracking_status': history['status']}
        histories.append(history)
        diagnostics.append({
            'channel': name, 'rf_noise_seed': seed,
            'acquisition_code_phase_samples': int(acq['code_phase_samples']),
            'acquisition_doppler_hz': float(acq['doppler_hz']),
            'saturation_fraction': float(rf_info['saturation_fraction'][0]),
            'tail_pll_error_rms_deg': float(np.degrees(np.sqrt(np.mean(
                history['pll_error_rad'][-30:]**2)))),
            'tail_frequency_mean_hz': float(np.mean(history['carrier_frequency_hz'][-30:])),
            'tail_frequency_std_hz': float(np.std(history['carrier_frequency_hz'][-30:])),
            'epoch_start_samples': history['epoch_start_samples'].tolist(),
            'carrier_nco_phase_rad': history['carrier_nco_phase_rad'].tolist(),
            'carrier_nco_frequency_hz': history['carrier_nco_frequency_hz'].tolist(),
            'pll_error_rad': history['pll_error_rad'].tolist(),
            'phase_lock_metric': history['phase_lock_metric'].tolist(),
        })
        del adc, baseband
    aligned = compare_phases(*histories, fs, round(fs*cfg.tracking.coherent_ms*.001))
    phase = aligned['phase_difference_rad']
    expected = float(expected)
    return {'thermal_noise': thermal_noise, 'status': 'evaluated_not_lock_certified',
            'pair': 'TR - TL', 'duration_ms': cfg.signal.duration_ms,
            'seed_base': seed_base, 'tracking_max_epochs': cfg.tracking.max_epochs,
            'multipath_start_ms': multipath_start_ms,
            'multipath_count': len(cfg.signal.multipath_paths),
            'fs_hz': fs, 'wavelength_m': cfg.signal.wavelength_m,
            'ideal_geometry_phase_deg': float(np.degrees(expected)),
            'ideal_geometry_path_m': expected*cfg.signal.wavelength_m/(2*np.pi),
            'time_s': aligned['time_s'].tolist(), 'phase_difference_rad': phase.tolist(),
            'raw_phase_difference_rad': aligned['raw_phase_difference_rad'].tolist(),
            'all_epochs': statistics(phase, cfg.signal.wavelength_m, expected),
            'last_30_epochs': statistics(phase[-30:], cfg.signal.wavelength_m, expected),
            'channels': diagnostics}


def main():
    destination = Path(__file__).resolve().parent/'results'/'dual_tracking_phase'
    destination.mkdir(parents=True, exist_ok=True)
    cases = [run_case(True), run_case(False)]
    result = {'architecture': 'TL/TR -> separate RF/ADC -> separate DFE -> separate acquisition and DLL/FLL/PLL',
              'scope': '100 ms; no interferers or multipath; zero channel delay; independent RF noise seeds 42/43; identical ideal clock/LO reference; default rectangular source/replica and default loop gains; ADC/DFE quantization retained',
              'observable': 'NCO phase at common physical time, before current-block feedback; not Prompt residual phase; phase branch retains integer-cycle ambiguity',
              'tail_policy': 'last 30 overlapping epochs fixed in advance, not a lock declaration',
              'cases': cases}
    (destination/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), constrained_layout=True)
    for case in cases:
        if case['status'] != 'evaluated_not_lock_certified':
            print(json.dumps(case, indent=2))
            continue
        time = np.array(case['time_s'])*1000
        phase = np.array(case['phase_difference_rad'])
        expected = np.radians(case['ideal_geometry_phase_deg'])
        label = 'Thermal noise ON' if case['thermal_noise'] else 'Thermal noise OFF'
        axes[0].plot(time, np.degrees(phase), label=label)
        axes[1].plot(time, np.angle(np.exp(1j*(phase-expected)))*case['wavelength_m']/(2*np.pi)*1000, label=label)
        print(json.dumps({k:case[k] for k in ('thermal_noise','all_epochs','last_30_epochs')}, indent=2))
    axes[0].set(ylabel='TR - TL tracked NCO phase (deg)')
    axes[1].set(xlabel='Time (ms)', ylabel='Wrapped error vs geometry (mm)')
    for ax in axes:
        ax.grid(alpha=.3)
        ax.legend()
    fig.savefig(destination/'phase_difference.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
