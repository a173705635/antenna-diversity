"""Ten-round observable-level research driver. Each invocation freezes a round.

Prepare each split explicitly before using it. Development/stress data are for
iteration; holdout is generated only after the candidate policy has been frozen.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_navigation_fusion import navigation_inputs
from audit_spatial_confirmation import stream_scores
from navigation_research import candidate_modes, navigate

SEEDS = {'development': [740001, 740010, 740024, 720004],
         'calibration': list(range(750001, 750009)),
         'stress': list(range(760001, 760005)),
         'holdout': list(range(770001, 770025))}
BASE_CASES = ('clean', 'all_spoof', 'partial3', 'slow', 'withdraw', 'prn_bias')
STRESS_CASES = BASE_CASES+('bias_slow', 'bias_accel')
THRESHOLD = .012710679080318886


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def prepare(root, split):
    directory = root/'data'/split; directory.mkdir(parents=True, exist_ok=True)
    for kind in ('static', 'turn'):
        for seed in SEEDS[split]:
            ins, gi, gp, gv, truth = navigation_inputs(kind, 'clean', seed)
            indices = np.r_[0, gi]
            data = dict(time_s=ins['time_s'][indices], position_m=ins['position_m'][indices],
                        velocity_mps=ins['velocity_mps'][indices], gnss_position_m=gp,
                        gnss_velocity_mps=gv, truth_position_m=truth['position_m'][indices])
            for case in (('clean',) if split == 'calibration' else BASE_CASES):
                data['score_'+case] = stream_scores(kind, case, seed)
            np.savez_compressed(directory/f'{kind}_{seed}.npz', **data)
    print(f'Prepared {split}: {2*len(SEEDS[split])} paired base trajectories', flush=True)


def inputs(data, case):
    t = data['time_s'][1:]; u = np.maximum(t-20, 0)
    gp = data['gnss_position_m'].copy(); gv = data['gnss_velocity_mps'].copy()
    if case != 'clean':
        slope = .5 if case in ('slow', 'bias_slow') else 5.
        gp[:, 0] += slope*u; gv[:, 0] += slope*(t >= 20)
        if case == 'bias_accel':
            gp[:, 0] += .025*u*u-slope*u
            gv[:, 0] += .05*u-slope*(t >= 20)
        if case == 'withdraw':
            gp[t >= 30] = data['gnss_position_m'][t >= 30]
            gv[t >= 30] = data['gnss_velocity_mps'][t >= 30]
    scores = data['score_prn_bias' if case.startswith('bias_') else 'score_'+case]
    return gp, gv, scores


def run(data, case, policy):
    gp, gv, scores = inputs(data, case)
    modes = candidate_modes(scores, THRESHOLD, 3, policy['quiet_windows'])
    return navigate(data, gp, gv, modes, estimate_acceleration=policy['estimate_acceleration'],
                    velocity_limit=policy['velocity_limit'] if policy['velocity_gate'] else np.inf,
                    position_limit=policy['position_limit'] if policy['position_gate'] else np.inf,
                    mean_velocity_limit=policy['mean_velocity_limit'] if policy['mean_velocity_gate'] else np.inf)


def calibrate(root, policy):
    # Fit limits against clean trajectories with spatial pending holds enabled,
    # but innovation isolation disabled; gates cannot censor their own fit data.
    fit = dict(policy, velocity_gate=False, position_gate=False, mean_velocity_gate=False)
    norms = []
    for path in sorted((root/'data'/'calibration').glob('*.npz')):
        with np.load(path) as data:
            result = run(data, 'clean', fit)
        norms.append([float(np.linalg.norm(result['innovation'][:, 3:5], axis=1).max()),
                      float(np.linalg.norm(result['innovation'][:, :2], axis=1).max()),
                      float(result['mean_velocity_innovation'].max())])
    limits = np.max(norms, axis=0)
    return dict(velocity_limit=float(limits[0]), position_limit=float(limits[1]),
                mean_velocity_limit=float(limits[2]))


def metrics(data, result, case):
    t = data['time_s']; error = np.linalg.norm(result['state'][:, :2]-data['truth_position_m'][:, :2], axis=1)
    isolated = t[result['mode'] == 1]
    before = isolated[isolated < 20]; after = isolated[isolated >= 20]
    allowed = result['mode'] == 0
    rejection = ~result['gnss_accepted'][1:]
    starts = rejection & ~np.r_[False, rejection[:-1]]
    row = dict(first_isolation_s=float(isolated[0]) if len(isolated) else None,
               pre_attack_isolation=bool(len(before)),
               isolation_delay_s=float(after[0]-20) if len(after) and not len(before) and case != 'clean' else None,
               update_fraction=float(result['gnss_accepted'][1:].mean()),
               interruption_episodes=int(starts.sum()), rejected_update_time_s=float(rejection.sum()*.1),
               error_p95_m=float(np.quantile(error, .95)),
               allowed_over30_epochs=int(np.sum(allowed & (error >= 30))),
               first10_max_m=float(error[(t >= 20)&(t <= 30)].max()),
               first30_max_m=float(error[(t >= 20)&(t <= 50)].max()),
               first60_max_m=float(error[t >= 20].max()))
    return row


def aggregate(rows):
    out = {}
    for case in sorted({r['case'] for r in rows}):
        group = [r for r in rows if r['case'] == case]
        eligible = [r for r in group if not r['pre_attack_isolation']]
        delays = [r['isolation_delay_s'] for r in eligible if r['isolation_delay_s'] is not None]
        out[case] = dict(trajectories=len(group), premature=sum(r['pre_attack_isolation'] for r in group),
                         isolated=sum(r['first_isolation_s'] is not None for r in group),
                         eligible_detected=len(delays), eligible=len(eligible),
                         max_delay_s=max(delays) if delays else None,
                         mean_update_fraction=float(np.mean([r['update_fraction'] for r in group])),
                         mean_rejected_time_s=float(np.mean([r['rejected_update_time_s'] for r in group])),
                         interruption_episodes=sum(r['interruption_episodes'] for r in group),
                         max_first10_m=max(r['first10_max_m'] for r in group),
                         max_first30_m=max(r['first30_max_m'] for r in group),
                         max_p95_m=max(r['error_p95_m'] for r in group),
                         over30_allowed_runs=sum(r['allowed_over30_epochs'] > 0 for r in group))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=SEEDS, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--round', type=int)
    parser.add_argument('--quiet-windows', type=int, default=0)
    parser.add_argument('--estimate-acceleration', action='store_true')
    parser.add_argument('--velocity-gate', action='store_true')
    parser.add_argument('--position-gate', action='store_true')
    parser.add_argument('--mean-velocity-gate', action='store_true')
    parser.add_argument('--policy', type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.output, args.split)
        return
    out = args.output/f'round{args.round:02d}'; out.mkdir(parents=True, exist_ok=True)
    if args.policy:
        policy = json.loads(args.policy.read_text())['policy']
    else:
        policy = dict(quiet_windows=args.quiet_windows, estimate_acceleration=args.estimate_acceleration,
                      velocity_gate=args.velocity_gate, position_gate=args.position_gate,
                      mean_velocity_gate=args.mean_velocity_gate)
        policy.update(calibrate(args.output, policy))
    manifest = dict(round=args.round, split=args.split, policy=policy, threshold=THRESHOLD,
                    synthetic=True, navigation_hz=10, calibration_seeds=SEEDS['calibration'],
                    evaluation_seeds=SEEDS[args.split], covariance_is_integrity_bound=False,
                    sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                             for name in ('navigation_research.py', 'audit_navigation_iterations.py',
                                          'audit_navigation_fusion.py', 'audit_spatial_confirmation.py',
                                          'spatial_spoofing.py', 'inertial.py', 'presets.py',
                                          'audit_spatial_diversity.py')})
    write(out/'frozen.json', manifest)
    rows = []
    cases = STRESS_CASES if args.split in ('stress', 'holdout') else BASE_CASES
    for path in sorted((args.output/'data'/args.split).glob('*.npz')):
        kind, seed = path.stem.split('_')
        with np.load(path) as data:
            for case in cases:
                result = run(data, case, policy)
                rows.append(dict(motion=kind, seed=int(seed), case=case, **metrics(data, result, case)))
    write(out/'rows.json', rows); write(out/'summary.json', aggregate(rows))
    print(json.dumps(dict(round=args.round, policy=policy, summary=aggregate(rows))), flush=True)


if __name__ == '__main__':
    main()
