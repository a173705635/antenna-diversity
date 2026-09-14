"""Phone-array direction and double-difference research audit, no new RF run."""
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
import time
import numpy as np

from audit_spatial_diversity import W, NAMES, LAYOUTS, CASES, model_observations, save_json, wilson, paired_gain
from presets import mate70_phase_centers
from spatial_spoofing import spatial_scores
from spatial_direction import (double_difference_scores, estimate_direction,
                               calibrate_channel_bias, select_common_prns)
from run_gps_spp import satellite_setup, receiver_reference


def batch_scores(seed, n, case):
    output = {}
    for chunk, start in enumerate(range(0, n, 250)):
        x = model_observations(seed+chunk, min(250, n-start), case)
        for layout, indices in LAYOUTS.items():
            for family, function in [('original', spatial_scores), ('dd', double_difference_scores)]:
                for method, values in function(x[:, :, indices], W).items():
                    output.setdefault(f'{layout}_{family}_{method}', []).append(values)
    return {key: np.concatenate(value) for key, value in output.items()}


def detection(out, n):
    calibration = batch_scores(200000, 5000, 'los')
    thresholds = {k: float(np.quantile(v, .01, method='lower')) for k, v in calibration.items()}
    np.savez_compressed(out/'dd_calibration.npz', **calibration)
    save_json(out/'dd_thresholds.json', thresholds)
    rows = []
    for i, (case, attack) in enumerate(CASES):
        seed = 300000 if case == 'spatial_matched' else 300000+i*10000
        scores = batch_scores(seed, n, case)
        np.savez_compressed(out/f'dd_{case}.npz', **scores)
        for layout in LAYOUTS:
            for method in ('global', 'cluster3'):
                decisions = [scores[f'{layout}_{family}_{method}'] < thresholds[f'{layout}_{family}_{method}']
                             for family in ('original', 'dd')]
                rows.append(dict(case=case, attack=attack, layout=layout, method=method, n=n,
                    original_hits=int(decisions[0].sum()), dd_hits=int(decisions[1].sum()),
                    original_fraction=float(decisions[0].mean()), dd_fraction=float(decisions[1].mean()),
                    dd_wilson95=wilson(int(decisions[1].sum()), n), **paired_gain(*decisions)))
        print('DD done', case, flush=True)
    save_json(out/'dd_comparison.json', rows)
    # Explicit invariance checks on common, including time-varying, channel bias.
    x = model_observations(1984, 100, 'partial_3')
    bias = np.random.default_rng(5).normal(0, .1, (100, 10, 4, 1))
    invariant = {}
    for label, fn in [('original', spatial_scores), ('dd', double_difference_scores)]:
        a, b = fn(x, W), fn(x+bias, W)
        invariant[label] = {k: float(np.max(abs(a[k]-b[k]))) for k in a}
        assert max(invariant[label].values()) < 1e-10
    save_json(out/'common_bias_invariance.json', invariant)


def direction_error(result, truth):
    candidates = result['candidates']
    # Evaluation ONLY. Never used to pick, prune or order the estimated set.
    if not candidates:
        return None
    return float(min(np.rad2deg(np.arccos(np.clip(np.dot(c['unit_vector'], truth), -1, 1))) for c in candidates))


def model_direction(out, count):
    rng = np.random.default_rng(20260914)
    p = np.array([mate70_phase_centers()[name] for name in NAMES])
    bias = np.array([0, .012, -.018, .025])
    rows, inputs = [], []
    for trial in range(count):
        az = rng.uniform(0, 2*np.pi); up = rng.uniform(np.sin(np.deg2rad(10)), .98)
        u = np.array([np.sqrt(1-up**2)*np.sin(az), np.sqrt(1-up**2)*np.cos(az), up])
        x = np.broadcast_to((-p@u)[None, :, None], (10, 4, 5)).copy()
        x += rng.normal(0, .003/np.sqrt(2), x.shape)
        half = rng.integers(-10, 11, (1, 4, 5))*W/2
        inputs.append(dict(trial=trial, truth=u.tolist(), carrier_m=x.tolist(), half_cycle_m=half.tolist()))
        for layout in ('3', '4'):
            indices = LAYOUTS[layout]
            for case, data, calibration, harmonic in [
                ('resolved_polarity', x, np.zeros(4), 1),
                ('half_cycle', x+half, np.zeros(4), 2),
                ('biased_wrong_zero', x+half+bias[None, :, None], np.zeros(4), 2),
                ('biased_calibrated', x+half+bias[None, :, None], bias, 2)]:
                estimate = estimate_direction(data[:, indices], p[indices], W,
                                              channel_bias_m=calibration[indices], harmonic=harmonic)
                rows.append(dict(trial=trial, layout=layout, case=case, result=estimate,
                                 candidate_set_min_error_deg=direction_error(estimate, u)))
        if trial%20 == 0:
            print('DOA model', trial, '/', count, flush=True)
    save_json(out/'direction_model_inputs.json', inputs)
    save_json(out/'direction_model.json', rows)
    # Two distinct source directions, plus compensating common per-channel bias,
    # yield exactly the same single-difference observations: unknown bias confound.
    u1, u2 = np.array([1., 0, 0]), np.array([0, 1., 0])
    a = -p@u1
    compensation = p@(u2-u1)
    b = -p@u2+compensation
    np.testing.assert_allclose(a, b, atol=1e-15)
    save_json(out/'unobservable_counterexample.json', dict(u1=u1.tolist(), u2=u2.tolist(),
              first_observation_m=a.tolist(), second_observation_m=b.tolist(),
              second_channel_bias_m=compensation.tolist(), same_observations=True))


def rf_direction(out, archive):
    p = np.array([mate70_phase_centers()[name] for name in NAMES])
    rx, rot = receiver_reference()
    directions = np.array([rot@(s['position_m']-rx) for s in satellite_setup()])
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    # Independent seed's LOS observation is the calibration input for both attack seeds.
    with np.load(archive/'rf_a/los_610000_observations.npz') as data:
        assert data['valid'].all()
        calibration = calibrate_channel_bias(data['carrier_m'], p, directions, W)
    save_json(out/'rf_calibration.json', {**calibration,
        'assumption': 'trusted LOS, known orientation and synthetic satellite geometry; transfer assumes stable PRN-common biases',
        'input': str(archive/'rf_a/los_610000_observations.npz')})
    rows = []
    az, el = np.deg2rad([100, 10])
    truth = np.array([np.cos(el)*np.sin(az), np.cos(el)*np.cos(az), np.sin(el)])
    thresholds = json.loads((archive/'model/thresholds.json').read_text(encoding='utf-8'))['thresholds']
    for batch in ('rf_a', 'rf_b'):
        summary = json.loads((archive/batch/'summary.json').read_text(encoding='utf-8'))['rows']
        for case in dict.fromkeys(r['case'] for r in summary):
            for seed in (610000, 620000):
                with np.load(archive/batch/f'{case}_{seed}_observations.npz') as data:
                    x, valid = data['carrier_m'], data['valid']
                for layout in ('2_short', '3', '4'):
                    indices = LAYOUTS[layout]
                    for k in range(4):
                        block = x[k*10:(k+1)*10, indices]
                        if not valid[k*10:(k+1)*10, indices].all():
                            rows.append(dict(case=case, seed=seed, layout=layout, window=k,
                                             status='invalid_observation', alarm=False))
                            continue
                        scores = spatial_scores(block, W)
                        alarm = bool(scores['global'] < thresholds[layout+'_global'] or
                                     scores['cluster3'] < thresholds[layout+'_cluster3'])
                        # OR is an analysis trigger ONLY; no fused detector performance claim.
                        if not alarm:
                            rows.append(dict(case=case, seed=seed, layout=layout, window=k,
                                             status='no_detection_trigger', alarm=False))
                            continue
                        selected = select_common_prns(block, W)
                        for mode, bias in [('ideal_zero_bias', np.zeros(4)),
                                           ('trusted_los_calibration', np.array(calibration['bias_m']))]:
                            fit = estimate_direction(block[:, :, selected], p[indices], W,
                                                     channel_bias_m=bias[indices], harmonic=2)
                            # Label contamination checked only after data-driven selection.
                            fake_count = 0 if case=='los' else int(case.split('_')[1]) if case.startswith('partial_') else 8
                            rows.append(dict(case=case, seed=seed, layout=layout, window=k, mode=mode,
                                alarm=True, status=fit['status'], selected_prns=[s+1 for s in selected],
                                all_selected_are_forged=bool(all(s<fake_count for s in selected)),
                                result=fit, candidate_set_min_error_deg=direction_error(fit, truth) if case!='los' else None))
    save_json(out/'direction_rf.json', rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--events', type=int, default=5000)
    parser.add_argument('--directions', type=int, default=100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    detection(args.output, args.events)
    model_direction(args.output, args.directions)
    rf_direction(args.output, args.archive)
    save_json(args.output/'manifest.json', dict(events_per_case=args.events, calibration_events=5000,
        direction_trials=args.directions, positions_m=mate70_phase_centers(), elapsed_s=time.monotonic()-start,
        archived_rf=str(args.archive), source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob('*.py')},
        limits=['array-frame angles; no IMU', 'planar mirror unresolved', 'no unique candidate selected using truth',
                'RF replay, not new RF capture', 'fit tolerances not probabilistic confidence',
                'grid/refinement candidate search is numerical, not certified exhaustive']))
    print('DONE', time.monotonic()-start, flush=True)


if __name__ == '__main__':
    main()
