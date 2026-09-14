"""Offline float code/carrier positioning from archived single-antenna RF runs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from positioning import carrier_range, solve_code_carrier_float
from run_gps_spp import serializable


def continuous_arcs(times, code, phase):
    """Conservative first version: all satellites valid, no skipped epochs."""
    valid = np.isfinite(code).all(axis=1) & np.isfinite(phase).all(axis=1)
    indices = np.flatnonzero(valid)
    if not indices.size:
        return []
    # Input archive has nominal 10 ms epochs. Any hole starts a new bias set.
    cuts = np.flatnonzero((np.diff(indices) != 1) | (np.diff(times[indices]) > .010001))+1
    return [arc for arc in np.split(indices, cuts) if arc.size >= 2]


def error_summary(positions, truth):
    valid = np.isfinite(positions).all(axis=1)
    if not valid.any():
        return {'epochs': 0}
    error = positions[valid]-truth
    bias = error.mean(axis=0)
    norm = np.linalg.norm(error, axis=1)
    return {'epochs': int(valid.sum()), 'mean_error_ecef_m': bias,
            'mean_bias_norm_m': float(np.linalg.norm(bias)),
            'absolute_3d_rms_m': float(np.sqrt(np.mean(norm**2))),
            'absolute_3d_p95_m': float(np.quantile(norm, .95)),
            'absolute_3d_max_m': float(norm.max()),
            'demeaned_3d_rms_m': float(np.sqrt(np.mean(np.sum((error-bias)**2, axis=1))))}


def analyze(data, manifest, code_sigma_m, phase_sigma_m, controls=False):
    times = np.array(data['time_s'])
    code = np.array(data['pseudoranges_m'], dtype=float)
    histories = data['satellite_tracking']
    phase = np.column_stack([carrier_range(
        {**h, 'success': h.get('tracking_status') == 'tracking'}, times,
        data['fs_dfe_hz'], manifest['tracking_config']['coherent_ms']*.001,
        299792458./manifest['signal_config_last_prn']['carrier_hz']) for h in histories])
    states = np.array([[np.array(s['position_m'])+t*np.array(s['velocity_mps'])
                        for s in manifest['satellites']] for t in times])
    positions = np.full((len(times), 3), np.nan)
    clocks = np.full(len(times), np.nan)
    phase_residual = np.full_like(code, np.nan)
    code_residual = np.full_like(code, np.nan)
    arcs = []
    checks = {}
    for indices in continuous_arcs(times, code, phase):
        try:
            solution = solve_code_carrier_float(states[indices], code[indices], phase[indices],
                                               code_sigma_m, phase_sigma_m)
        except ValueError as exc:
            arcs.append({'indices': indices, 'status': 'solver_failed', 'reason': str(exc)})
            continue
        peak = float(np.max(np.abs(solution['carrier_residual_m'])))
        accepted = peak <= .05
        arcs.append({'indices': indices, 'status': 'accepted' if accepted else 'carrier_residual_rejected',
                     'max_abs_carrier_residual_m': peak,
                     'ambiguity_m': solution['ambiguity_m'],
                     'centered_ambiguity_m': solution['centered_ambiguity_m'],
                     'phase_origin_shift_m': solution['phase_origin_shift_m'],
                     'iterations': solution['iterations'], 'rank': solution['rank'],
                     'unknowns': solution['unknowns'], 'integer_fixed': False})
        if not accepted:
            continue
        positions[indices] = solution['position_m']
        clocks[indices] = solution['clock_bias_m']
        phase_residual[indices] = solution['carrier_residual_m']
        code_residual[indices] = solution['code_residual_m']
        if controls and not checks:
            offset = np.linspace(-123.456, 456.789, code.shape[1])
            shifted = solve_code_carrier_float(states[indices], code[indices], phase[indices]+offset,
                                               code_sigma_m, phase_sigma_m)
            checks['phase_origin'] = {
                'offset_m': offset,
                'max_position_change_m': float(np.max(np.linalg.norm(shifted['position_m']-solution['position_m'], axis=1))),
                'max_bias_shift_error_m': float(np.max(np.abs(shifted['ambiguity_m']-solution['ambiguity_m']-offset)))}
            # Controlled per-PRN constant code error consistent with a 20 m X shift.
            # Uses a receiver estimate, not truth; carrier measurements stay identical.
            approximate = np.array(data['position_ecef_m'])[indices[0]]
            displacement = np.array([20., 0., 0.])
            code_offset = (np.linalg.norm(states[indices[0]]-approximate-displacement, axis=1)
                           -np.linalg.norm(states[indices[0]]-approximate, axis=1))
            shifted_code = solve_code_carrier_float(states[indices], code[indices]+code_offset,
                                                   phase[indices], code_sigma_m, phase_sigma_m)
            difference = shifted_code['position_m']-solution['position_m']
            checks['constant_code_bias'] = {'code_offset_m': code_offset,
                'expected_position_shift_m': displacement, 'mean_position_shift_m': difference.mean(axis=0),
                'max_departure_from_mean_shift_m': float(np.max(np.linalg.norm(difference-difference.mean(axis=0), axis=1)))}
            stepped = phase[indices].copy()
            wavelength = 299792458./manifest['signal_config_last_prn']['carrier_hz']
            stepped[len(indices)//2:, 0] += wavelength
            slipped = solve_code_carrier_float(states[indices], code[indices], stepped,
                                              code_sigma_m, phase_sigma_m)
            slip_peak = float(np.max(np.abs(slipped['carrier_residual_m'])))
            checks['one_cycle_step'] = {'prn': histories[0]['prn'],
                'start_time_s': float(times[indices[len(indices)//2]]), 'step_m': wavelength,
                'max_carrier_residual_m': slip_peak, 'rejected_by_50mm_gate': slip_peak > .05,
                'max_position_change_m': float(np.max(np.linalg.norm(slipped['position_m']-solution['position_m'], axis=1)))}
    # The archived receiver reference is used only for evaluation after solving.
    truth = np.array(manifest['receiver_truth_ecef_m'])
    baseline = np.array(data['position_ecef_m'], dtype=float)
    matched = np.isfinite(positions).all(axis=1)
    summary = {'seed': data['summary']['seed'], 'total_epochs': len(times),
               'accepted_epochs': int(matched.sum()),
               'raw_spp_all': error_summary(baseline, truth),
               'raw_spp_matched': error_summary(baseline[matched], truth),
               'joint_float': error_summary(positions, truth)}
    if matched.any():
        summary['carrier_residual_rms_m'] = float(np.sqrt(np.mean(phase_residual[matched]**2)))
    return {'summary': summary, 'time_s': times, 'pseudoranges_m': code,
            'carrier_range_m': phase, 'position_ecef_m': positions, 'clock_bias_m': clocks,
            'raw_spp_position_ecef_m': baseline, 'code_residual_m': code_residual,
            'carrier_residual_m': phase_residual, 'arcs': arcs, 'controls': checks}


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=root.parent/'docs/reports/data/2026-09-11-gps-spp')
    parser.add_argument('--output', type=Path, default=root/'results/code_carrier_float')
    parser.add_argument('--code-sigma-m', type=float, default=30.)
    parser.add_argument('--phase-sigma-m', type=float, default=.01)
    args = parser.parse_args()
    if not np.isfinite([args.code_sigma_m, args.phase_sigma_m]).all() or min(args.code_sigma_m, args.phase_sigma_m) <= 0:
        parser.error('observation sigmas must be positive and finite')
    manifest_path = args.input/'manifest.json'
    source_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    files = [manifest_path]+[args.input/f'seed_{seed}.json' for seed in source_manifest['seeds']]
    input_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    if (args.input/'sha256.json').exists():
        expected = json.loads((args.input/'sha256.json').read_text(encoding='utf-8'))
        if any(expected.get(name) != digest for name, digest in input_hashes.items()):
            raise ValueError('input archive hash mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    meta = {'method': 'offline full-arc code/carrier joint float least squares',
            'state': 'independent XYZ/clock per epoch, shared real bias per satellite per arc',
            'integer_fixed': False, 'static_receiver_prior': False,
            'code_sigma_m': args.code_sigma_m, 'phase_sigma_m': args.phase_sigma_m,
            'weighting': 'engineering scales, uncalibrated temporal correlation; no confidence claim',
            'residual_gate_m': .05,
            'slip_handling': 'split invalid common epochs; reject excessive carrier residual; not complete slip detection',
            'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.py')},
            'input_sha256': input_hashes, 'source_manifest': source_manifest}
    (args.output/'manifest.json').write_text(json.dumps(serializable(meta), indent=2, allow_nan=False), encoding='utf-8')
    results = []
    for number, path in enumerate(files[1:]):
        data = json.loads(path.read_text(encoding='utf-8'))
        print(f"Solving seed {data['summary']['seed']} with floating ambiguities...", flush=True)
        result = analyze(data, source_manifest, args.code_sigma_m, args.phase_sigma_m, controls=number == 0)
        (args.output/path.name).write_text(json.dumps(serializable(result), indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps(serializable(result['summary'])), flush=True)
        results.append(result)
    (args.output/'summary.json').write_text(json.dumps(serializable([r['summary'] for r in results]), indent=2, allow_nan=False), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    truth = np.array(source_manifest['receiver_truth_ecef_m'])
    for result in results:
        for key, style, label in [('raw_spp_position_ecef_m', '--', 'Code'), ('position_ecef_m', '-', 'Code + carrier float')]:
            position = np.array(result[key])
            error = position-truth
            axes[0].plot(result['time_s'], np.linalg.norm(error, axis=1), style,
                         label=f"{label}, {result['summary']['seed']}")
            if np.isfinite(error).any():
                axes[1].plot(result['time_s'], np.linalg.norm(error-np.nanmean(error, axis=0), axis=1), style)
    axes[0].set(ylabel='Absolute 3D error (m)', title='Offline float solution: absolute error and scatter are different')
    axes[1].set(xlabel='Time (s)', ylabel='Distance from each solution mean (m)', yscale='log')
    for axis in axes:
        axis.grid(alpha=.3)
    axes[0].legend(fontsize=8, ncol=2)
    fig.savefig(args.output/'comparison.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
