"""Paired receiver controls for GPS code bias; production defaults unchanged."""

import copy
import hashlib
import json
from pathlib import Path
import argparse

import numpy as np

from acquisition import acquire
from dfe import process_dfe
from positioning import PositioningFailure
from positioning import code_pseudorange, carrier_range, solve_position, solve_code_carrier_float
from rf import process_rf
from tracking import track
from run_gps_spp import build_scene, serializable
from audit_code_carrier_float import error_summary


def error_stats(error):
    error = np.asarray(error, dtype=float)
    return {'epochs': np.isfinite(error).sum(axis=0).tolist(),
            'mean_m': np.nanmean(error, axis=0).tolist(),
            'std_m': np.nanstd(error, axis=0).tolist(),
            'rms_m': np.sqrt(np.nanmean(error**2, axis=0)).tolist(),
            'max_abs_m': np.nanmax(np.abs(error), axis=0).tolist()}


def observe(baseband, fs, satellites, cfg, times, continuous_local):
    local_cfg = copy.deepcopy(cfg.tracking)
    local_cfg.code_replica_bandwidth_hz = cfg.dfe.cutoff_hz if continuous_local else None
    code = np.full((len(times), len(satellites)), np.nan)
    phase = np.full_like(code, np.nan)
    diagnostics = []
    for column, satellite in enumerate(satellites):
        acq = acquire(baseband[0], fs, satellite['prn'], cfg.signal.code_rate_hz, cfg.acquisition)
        diagnostic = {'prn': satellite['prn'], 'acquired': bool(acq['success']),
                      'acquisition_code_phase_samples': int(acq['code_phase_samples']),
                      'acquisition_doppler_hz': float(acq['doppler_hz'])}
        if acq['success']:
            h = track(baseband[0], fs, satellite['prn'], acq, local_cfg,
                      cfg.signal.carrier_hz, cfg.signal.code_rate_hz)
            diagnostic['tracking_status'] = h['status']
            if h['success']:
                code[:, column] = code_pseudorange(h, times, fs, local_cfg.coherent_ms*.001,
                                                  satellite['coarse_range_m'])
                phase[:, column] = carrier_range(h, times, fs, local_cfg.coherent_ms*.001,
                                                cfg.signal.wavelength_m)
                index = np.searchsorted(h['epoch_start_samples']/fs, times, side='right')-1
                safe = np.clip(index, 0, len(h['prompt'])-1)
                good = ((index >= 0) & (h['phase_lock_metric'][safe] >= .8)
                        & (abs(h['pll_error_rad'][safe]) < np.pi/2))
                code[~good, column] = np.nan
                phase[~good, column] = np.nan
        diagnostics.append(diagnostic)
    return {'pseudoranges_m': code, 'carrier_range_m': phase, 'tracking': diagnostics}


def summarize(name, observations, times, satellites, reference, clock_bias, positioning=False):
    states = np.array([[s['position_m']+t*s['velocity_mps'] for s in satellites] for t in times])
    expected = np.linalg.norm(states-reference, axis=2)+clock_bias
    errors = observations['pseudoranges_m']-expected
    result = {'name': name, 'time_s': times, **observations, 'pseudorange_error_m': errors,
              'per_prn': error_stats(errors)}
    if positioning:
        positions = np.full((len(times), 3), np.nan)
        failures = []
        for k, (s, p) in enumerate(zip(states, observations['pseudoranges_m'])):
            valid = np.isfinite(p)
            try:
                positions[k] = solve_position(s[valid], p[valid])['position_m']
            except PositioningFailure as exc:
                failures.append({'index': k, 'reason': str(exc)})
        result['spp'] = error_summary(positions, reference)
        result['spp_positions_m'] = positions
        result['spp_failures'] = failures
        # Compare identical full continuous arcs only, no artificial gap bridging.
        if np.isfinite(observations['pseudoranges_m']).all() and np.isfinite(observations['carrier_range_m']).all():
            try:
                joint = solve_code_carrier_float(states, observations['pseudoranges_m'], observations['carrier_range_m'])
                result['joint_float'] = error_summary(joint['position_m'], reference)
                result['joint_positions_m'] = joint['position_m']
                result['joint_max_carrier_residual_m'] = float(np.max(np.abs(joint['carrier_residual_m'])))
            except PositioningFailure as exc:
                result['joint_failure'] = str(exc)
    return result


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=root/'results/pseudorange_bias')
    args = parser.parse_args()
    source = root.parent/'docs/reports/data/2026-09-11-gps-spp'
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    expected_hashes = json.loads((source/'sha256.json').read_text(encoding='utf-8'))
    inputs = [source/'manifest.json']+[source/f'seed_{s}.json' for s in manifest['seeds']]
    for path in inputs:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hashes[path.name]:
            raise ValueError('archive checksum mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    satellites = [{**s, 'position_m': np.array(s['position_m']), 'velocity_mps': np.array(s['velocity_mps'])}
                  for s in manifest['satellites']]
    reference = np.array(manifest['receiver_truth_ecef_m'])
    clock_bias = manifest['clock_bias_truth_m']
    duration = manifest['duration_ms']
    # Reproduce the archived experiment, even when today's defaults change.
    archived_rates = {'fs_rf_hz': manifest['signal_config_last_prn']['fs_rf_hz'],
                      'decimation': manifest['dfe_config']['decimation']}
    times = np.arange(20, duration//10)*.01
    results = []
    def save(name, observations, positioning=False):
        result = summarize(name, observations, times, satellites, reference, clock_bias, positioning)
        (args.output/f'{name}.json').write_text(json.dumps(serializable(result), indent=2, allow_nan=False), encoding='utf-8')
        results.append(result)
        print(name, 'mean=', np.round(result['per_prn']['mean_m'], 3),
              'std=', np.round(result['per_prn']['std_m'], 3), flush=True)
    archived_errors = []
    for seed in manifest['seeds']:
        data = json.loads((source/f'seed_{seed}.json').read_text(encoding='utf-8'))
        save(f'archive_{seed}', {'pseudoranges_m': np.array(data['pseudoranges_m'], dtype=float)})
        archived_errors.append(results[-1]['pseudorange_error_m'])
    meta = {'scope': 'paired per-PRN pseudorange diagnosis, no default receiver change',
            'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.py')},
            'input_sha256': {p.name: expected_hashes[p.name] for p in inputs},
            'source_manifest': manifest,
            'cross_seed_3000_3001_correlation_per_prn': [float(np.corrcoef(archived_errors[0][:, i], archived_errors[1][:, i])[0, 1]) for i in range(len(satellites))],
            'linear_control': 'RF/ADC bypassed, DFE output quantization disabled, original FIR retained; diagnostic only',
            'local_control': 'only existing 1.8 MHz continuous replica switched on; same loop gains, no phase refinement',
            'source_control': 'existing power-normalized ideal 1.8 MHz continuous source',
            'isolated_control': 'each PRN separately through linear DFE; not a simultaneous receiver performance result'}
    print('Building rectangular eight-PRN source...', flush=True)
    scene, cfg = build_scene(satellites, duration, **archived_rates)
    for seed in manifest['seeds']:
        adc, _ = process_rf(scene, cfg.rf, cfg.reference_impedance_ohm, seed)
        bb, info = process_dfe(adc, cfg.dfe, cfg.signal.fs_rf_hz, cfg.signal.if_hz)
        del adc
        if seed == manifest['seeds'][0]:
            original = observe(bb, info['fs_out_hz'], satellites, cfg, times, False)
            archive_code = np.array(json.loads((source/f'seed_{seed}.json').read_text())['pseudoranges_m'])
            np.testing.assert_allclose(original['pseudoranges_m'], archive_code, rtol=0, atol=1e-8)
            meta['baseline_reproduction_max_difference_m'] = float(np.max(abs(original['pseudoranges_m']-archive_code)))
        save(f'full_rect_source_cont_local_{seed}', observe(bb, info['fs_out_hz'], satellites, cfg, times, True), True)
        del bb
    linear = copy.deepcopy(cfg.dfe)
    linear.output_bits = 0
    bb, info = process_dfe(scene, linear, cfg.signal.fs_rf_hz, cfg.signal.if_hz)
    del scene
    for continuous in (False, True):
        save('linear_rect_mixed_'+('cont' if continuous else 'rect'), observe(bb, info['fs_out_hz'], satellites, cfg, times, continuous))
    del bb
    isolated = {False: [], True: []}
    for satellite in satellites:
        scene, single_cfg = build_scene([satellite], duration, **archived_rates)
        bb, info = process_dfe(scene, linear, cfg.signal.fs_rf_hz, cfg.signal.if_hz)
        del scene
        for continuous in (False, True):
            isolated[continuous].append(observe(bb, info['fs_out_hz'], [satellite], single_cfg, times, continuous))
        del bb
        print(f"isolated rectangular source PRN {satellite['prn']} complete", flush=True)
    for continuous, parts in isolated.items():
        save('linear_rect_isolated_'+('cont' if continuous else 'rect'), {
            key: np.column_stack([part[key][:, 0] for part in parts]) for key in ('pseudoranges_m', 'carrier_range_m')})
    print('Building continuous eight-PRN source...', flush=True)
    scene, cfg = build_scene(satellites, duration, code_bandwidth_hz=1.8e6, **archived_rates)
    bb, info = process_dfe(scene, linear, cfg.signal.fs_rf_hz, cfg.signal.if_hz)
    del scene
    save('linear_cont_mixed_cont', observe(bb, info['fs_out_hz'], satellites, cfg, times, True))
    del bb
    parts = []
    for satellite in satellites:
        scene, single_cfg = build_scene([satellite], duration, code_bandwidth_hz=1.8e6, **archived_rates)
        bb, info = process_dfe(scene, linear, cfg.signal.fs_rf_hz, cfg.signal.if_hz)
        del scene
        parts.append(observe(bb, info['fs_out_hz'], [satellite], single_cfg, times, True))
        del bb
        print(f"isolated continuous source PRN {satellite['prn']} complete", flush=True)
    save('linear_cont_isolated_cont', {key: np.column_stack([p[key][:, 0] for p in parts]) for key in ('pseudoranges_m', 'carrier_range_m')})
    (args.output/'manifest.json').write_text(json.dumps(serializable(meta), indent=2, allow_nan=False), encoding='utf-8')
    summaries = [{key: r[key] for key in ('name', 'per_prn', 'spp', 'joint_float', 'joint_max_carrier_residual_m') if key in r} for r in results]
    (args.output/'summary.json').write_text(json.dumps(serializable(summaries), indent=2, allow_nan=False), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    selected = ['archive_3000', 'full_rect_source_cont_local_3000', 'linear_rect_mixed_cont',
                'linear_rect_isolated_cont', 'linear_cont_isolated_cont']
    for result in results:
        if result['name'] in selected:
            axes[0].plot(np.arange(1, 9), result['per_prn']['mean_m'], 'o-', label=result['name'])
            axes[1].plot(np.arange(1, 9), result['per_prn']['std_m'], 'o-')
    axes[0].set(ylabel='Mean pseudorange error (m)', title='Paired code observation controls; no multipath or interference')
    axes[1].set(xlabel='GPS PRN', ylabel='Temporal standard deviation (m)', yscale='log')
    axes[0].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.3)
    fig.savefig(args.output/'per_prn.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
