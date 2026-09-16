"""Adaptive continuation: rounds 35--42 TD spatial candidates, 43--50 holdout."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from audit_forty_rounds import PROFILES, CASES, base_run, phase_paths
from audit_navigation_fusion import navigation_inputs, carrier_stream
from audit_navigation_iterations import BASE_CASES, metrics, aggregate, write
from audit_spatial_diversity import W
from navigation_monitors import first_evidence
from temporal_spatial_monitor import NAMES, temporal_scores


def prepare(root, previous, split):
    source = (root/'final_inputs' if split == 'final' else
              previous/'data'/('calibration' if split == 'calibration' else 'stress'))
    t = np.arange(8000)*.01
    for path in sorted(source.glob('*.npz')):
        kind, seed = path.stem.split('_'); seed = int(seed)
        ins, _, _, _, _ = navigation_inputs(kind, 'clean', seed)
        rotation = Slerp(ins['time_s'], Rotation.from_matrix(ins['body_to_nav']))(t)
        cases = ('clean',) if split == 'calibration' else BASE_CASES
        for case in cases:
            carrier = carrier_stream(t, kind, case, seed)
            by_rate = {}
            for rate in (0., .0017/9.80665, -.0017/9.80665):
                error = Rotation.from_rotvec(np.column_stack((t*0, rate*t, t*0)))
                by_rate[rate] = temporal_scores(carrier, (error*rotation).as_matrix(), W)
            for profile in PROFILES:
                rate = (.0017/9.80665 if profile in ('gyro_plus', 'combined') else
                        -.0017/9.80665 if profile == 'gyro_minus' else 0.)
                feature, active = by_rate[rate]
                folder = root/'temporal'/split/profile; folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(folder/f'{kind}_{seed}_{case}.npz', features=feature, excitation=active)
        print(f'Temporal features: {split}/{kind}/{seed}', flush=True)


def feature_path(root, split, profile, kind, seed, case):
    source_case = 'prn_bias' if case.startswith('bias_') else case
    return root/'temporal'/split/profile/f'{kind}_{seed}_{source_case}.npz'


def evaluate(root, previous, number, method, threshold, paths, split, provenance):
    out = root/f'round{number:02d}'; out.mkdir(parents=True, exist_ok=True)
    policy = json.loads((previous/'round08/frozen.json').read_text())['policy']
    spec = dict(round=number, method=NAMES[method], method_index=method, threshold=float(threshold),
                confirmation=3, excitation_deg=5., baseline_policy=policy, synthetic=True,
                hypothesis=provenance, evaluation_paths=[str(p.relative_to(root)) for p in paths],
                sources={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                         for name in ('audit_temporal_rounds.py', 'temporal_spatial_monitor.py',
                                      'audit_forty_rounds.py', 'navigation_monitors.py', 'navigation_research.py',
                                      'audit_navigation_fusion.py', 'audit_navigation_iterations.py',
                                      'audit_spatial_confirmation.py', 'spatial_spoofing.py', 'inertial.py',
                                      'presets.py', 'audit_spatial_diversity.py')})
    write(out/'frozen.json', spec)
    rows = []; controls = []
    for path in paths:
        kind, seed, case = path.stem.split('_', 2); profile = path.parent.name
        with np.load(path) as d, np.load(feature_path(root, split, profile, kind, seed, case)) as f:
            base = dict(state=d['state'], mode=d['mode'], gnss_accepted=d['accepted'])
            limit = np.full(800, threshold)
            alarm = first_evidence(f['features'][method], limit, 3, base['mode'] == 1)
            result = base if alarm is None else base_run(d, d['gp'], d['gv'], d['spatial'], policy, alert=alarm)
            meta = dict(motion=kind, seed=int(seed), case=case, profile=profile)
            drift_start = 40. if case == 'bias_delayed_accel' else 20.
            mask = (d['time_s'] >= drift_start)&(d['time_s'] <= drift_start+10)
            err = np.linalg.norm(result['state'][:, :2]-d['truth_position_m'][:, :2], axis=1)
            base_err = np.linalg.norm(base['state'][:, :2]-d['truth_position_m'][:, :2], axis=1)
            common = dict(drift_start_s=drift_start, excitation_fraction=float(f['excitation'][method//2].mean()))
            rows.append(dict(meta, **common, new_alert_s=float(d['time_s'][alarm+1]) if alarm is not None else None,
                             drift_first10_max_m=float(err[mask].max()), **metrics(d, result, case)))
            controls.append(dict(meta, **common, drift_first10_max_m=float(base_err[mask].max()),
                                 **metrics(d, base, case)))
    s = aggregate(rows); b = aggregate(controls)
    write(out/'rows.json', rows); write(out/'baseline_rows.json', controls)
    write(out/'summary.json', s); write(out/'baseline_summary.json', b)
    print(json.dumps(dict(round=number, method=NAMES[method], n=len(rows),
                          clean=s['clean']['isolated'], baseline_clean=b['clean']['isolated'],
                          accel_detected=s['bias_accel']['eligible_detected'],
                          accel_total=s['bias_accel']['trajectories'],
                          accel_over30=s['bias_accel']['over30_allowed_runs'],
                          accel_over30_baseline=b['bias_accel']['over30_allowed_runs'])), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous', type=Path, default=Path('results/ten-rounds'))
    parser.add_argument('--prepare', choices=('calibration', 'development', 'final'))
    parser.add_argument('--phase', type=int); parser.add_argument('--select', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare(args.output, args.previous, args.prepare)
        return
    if args.select:
        ranking = []
        for k in range(35, 43):
            s = json.loads((args.output/f'round{k:02d}/summary.json').read_text())
            score = [s['clean']['isolated'], sum(v['over30_allowed_runs'] for c,v in s.items() if c != 'clean'),
                     max(v['max_first10_m'] for c,v in s.items() if c != 'clean')]
            ranking.append(dict(round=k, score=score))
        winner = min(ranking, key=lambda r:r['score'])
        chosen = json.loads((args.output/f"round{winner['round']:02d}/frozen.json").read_text())
        write(args.output/'selected.json', dict(ranking=ranking, selected_round=winner['round'], policy=chosen))
        print(json.dumps(dict(selection=winner, ranking=ranking)), flush=True)
        return
    if args.phase == 4:
        files = sorted((args.output/'temporal/calibration').glob('*/*.npz'))
        values = np.array([np.load(p)['features'] for p in files])
        threshold = values.max(axis=(0, 2))
        # Strict greater-than with three confirmations gives zero added H0
        # training alerts. Inactive feature -1 cannot exceed the fitted limit.
        write(args.output/'temporal_thresholds.json', dict(names=NAMES, threshold=threshold.tolist(),
                                                         calibration_files=[str(p.relative_to(args.output)) for p in files]))
        for i in range(8):
            evaluate(args.output, args.previous, 35+i, i, threshold[i], phase_paths(args.output, 4),
                     'development', 'Fixed-offset cancellation by temporal phase differences; rotation required.')
    else:
        chosen = json.loads((args.output/'selected.json').read_text())['policy']
        for i, profile in enumerate(PROFILES):
            paths = sorted((args.output/'cache/final'/profile).glob('*.npz'))
            evaluate(args.output, args.previous, 43+i, chosen['method_index'], chosen['threshold'], paths,
                     'final', f'Frozen independent validation: {profile}; no holdout retuning.')


if __name__ == '__main__':
    main()
