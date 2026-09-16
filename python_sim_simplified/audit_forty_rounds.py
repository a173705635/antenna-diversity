"""Rounds 11--50: monitor discovery, robustness, stress, ablation, holdout.

All navigation and score generation are causal; the two-pass implementation
replays only when a new first alert changes the baseline trajectory. Input
cache preparation is explicit and separate from evaluation/frozen selection.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_navigation_fusion import navigation_inputs
from audit_spatial_confirmation import stream_scores
from audit_navigation_iterations import aggregate, metrics, inputs, write, THRESHOLD, BASE_CASES, STRESS_CASES
from navigation_research import candidate_modes, navigate
from navigation_monitors import NAMES, monitor_scores, first_evidence

PROFILES = ('nominal', 'accel_plus', 'accel_minus', 'gyro_plus', 'gyro_minus',
            'initial_velocity', 'correlated_pvt', 'combined')
CASES = STRESS_CASES+('bias_creep', 'bias_gentle_accel', 'bias_delayed_accel', 'bias_jerk')


def perturb(data, profile, seed):
    d = {k:data[k].copy() for k in data.files}; t = d['time_s']
    a = {'accel_plus':.01, 'accel_minus':-.01, 'combined':.01}.get(profile, 0.)
    j = {'gyro_plus':.0017, 'gyro_minus':-.0017, 'combined':.0017}.get(profile, 0.)
    v = {'initial_velocity':.4, 'combined':.2}.get(profile, 0.)
    d['position_m'][:, 0] += v*t+.5*a*t*t+j*t**3/6
    d['velocity_mps'][:, 0] += v+a*t+j*t*t/2
    if profile in ('correlated_pvt', 'combined'):
        rng = np.random.default_rng(seed+910000)
        noise = rng.normal(size=(len(t)-1, 6))
        for k in range(1, len(noise)):
            noise[k] = .8*noise[k-1]+.6*noise[k]
        d['gnss_position_m'] += noise[:, :3]*3.
        d['gnss_velocity_mps'] += noise[:, 3:]*.15
    return d


def observations(d, case):
    if case in STRESS_CASES:
        return inputs(d, case)
    gp = d['gnss_position_m'].copy(); gv = d['gnss_velocity_mps'].copy()
    t = d['time_s'][1:]; u = np.maximum(t-(40 if case == 'bias_delayed_accel' else 20), 0.)
    if case == 'bias_creep':
        gp[:, 0] += .1*u; gv[:, 0] += .1*(t >= 20)
    elif case in ('bias_gentle_accel', 'bias_delayed_accel'):
        a = .005 if case == 'bias_gentle_accel' else .05
        gp[:, 0] += .5*a*u*u; gv[:, 0] += a*u
    elif case == 'bias_jerk':
        gp[:, 0] += .001*u**3/6; gv[:, 0] += .001*u*u/2
    # Carrier takeover remains at 20 s; delayed PVT drift starts at 40 s.
    return gp, gv, d['score_prn_bias']


def base_run(d, gp, gv, scores, policy, fit=False, alert=None):
    modes = candidate_modes(scores, THRESHOLD, 3, 1)
    if alert is not None:
        modes[alert] = 1
    return navigate(d, gp, gv, modes, estimate_acceleration=True,
                    velocity_limit=np.inf if fit else policy['velocity_limit'], position_limit=np.inf,
                    mean_velocity_limit=np.inf if fit else policy['mean_velocity_limit'])


def prepare_cache(root, old, split):
    source = old/'data'/('calibration' if split == 'calibration' else 'stress')
    if split == 'final':
        source = root/'final_inputs'; source.mkdir(parents=True, exist_ok=True)
        for kind in ('static', 'turn'):
            for seed in range(780001, 780009):
                ins, gi, gp, gv, truth = navigation_inputs(kind, 'clean', seed)
                ix = np.r_[0, gi]
                d = dict(time_s=ins['time_s'][ix], position_m=ins['position_m'][ix],
                         velocity_mps=ins['velocity_mps'][ix], gnss_position_m=gp,
                         gnss_velocity_mps=gv, truth_position_m=truth['position_m'][ix])
                for case in BASE_CASES:
                    d['score_'+case] = stream_scores(kind, case, seed)
                np.savez_compressed(source/f'{kind}_{seed}.npz', **d)
    policy = json.loads((old/'round08/frozen.json').read_text())['policy']
    for profile in PROFILES:
        folder = root/'cache'/split/profile; folder.mkdir(parents=True, exist_ok=True)
        for path in sorted(source.glob('*.npz')):
            kind, seed = path.stem.split('_')
            with np.load(path) as data:
                d = perturb(data, profile, int(seed))
            cases = ('clean',) if split == 'calibration' else CASES
            # Non-nominal/combined development profiles only need H0 robustness.
            if split == 'development' and profile not in ('nominal', 'combined'):
                cases = ('clean',)
            for case in cases:
                gp, gv, spatial = observations(d, case)
                base = base_run(d, gp, gv, spatial, policy, fit=split == 'calibration')
                feature = monitor_scores(d, gp, gv, base)
                np.savez_compressed(folder/f'{kind}_{seed}_{case}.npz',
                                    time_s=d['time_s'], position_m=d['position_m'], velocity_mps=d['velocity_mps'],
                                    truth_position_m=d['truth_position_m'], gp=gp, gv=gv, spatial=spatial,
                                    state=base['state'], mode=base['mode'], accepted=base['gnss_accepted'],
                                    features=feature)
        print(f'Prepared {split}/{profile}', flush=True)


def limits(root, robust, envelope):
    folders = PROFILES if robust else ('nominal',)
    x = np.array([np.load(p)['features'] for profile in folders
                  for p in sorted((root/'cache/calibration'/profile).glob('*.npz'))])
    maximum = x.max(axis=0)
    return (np.maximum.accumulate(maximum, axis=1)*1.25 if envelope
            else np.repeat(maximum.max(axis=1)[:, None], 800, axis=1))


def evaluate(root, old, number, method, limit, confirmation, paths, hypothesis):
    folder = root/f'round{number:02d}'; folder.mkdir(parents=True, exist_ok=True)
    policy = json.loads((old/'round08/frozen.json').read_text())['policy']
    spec = dict(round=number, method=NAMES[method], method_index=method, limit=limit.tolist(),
                confirmation=confirmation, hypothesis=hypothesis, baseline_policy=policy,
                synthetic=True, reference_epoch_s=10., evaluation_paths=[str(p.relative_to(root)) for p in paths],
                sources={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                         for n in ('audit_forty_rounds.py', 'navigation_monitors.py', 'navigation_research.py',
                                   'audit_navigation_iterations.py', 'audit_navigation_fusion.py',
                                   'audit_spatial_confirmation.py', 'spatial_spoofing.py', 'inertial.py',
                                   'presets.py', 'audit_spatial_diversity.py')})
    write(folder/'frozen.json', spec)
    rows = []; baseline_rows = []
    for path in paths:
        kind, seed, case = path.stem.split('_', 2)
        profile = path.parent.name
        with np.load(path) as d:
            base = dict(state=d['state'], mode=d['mode'], gnss_accepted=d['accepted'])
            alarm = first_evidence(d['features'][method], limit, confirmation, base['mode'] == 1)
            result = base if alarm is None else base_run(d, d['gp'], d['gv'], d['spatial'], policy, alert=alarm)
            meta = dict(motion=kind, seed=int(seed), case=case, profile=profile)
            rows.append(dict(meta, new_alert_s=float(d['time_s'][alarm+1]) if alarm is not None else None,
                             **metrics(d, result, case)))
            baseline_rows.append(dict(meta, **metrics(d, base, case)))
    summary = aggregate(rows); baseline_summary = aggregate(baseline_rows)
    write(folder/'rows.json', rows); write(folder/'baseline_rows.json', baseline_rows)
    write(folder/'summary.json', summary); write(folder/'baseline_summary.json', baseline_summary)
    print(json.dumps(dict(round=number, method=NAMES[method], n=len(rows),
                          clean_isolated=summary['clean']['isolated'],
                          baseline_clean=baseline_summary['clean']['isolated'],
                          accel_detected=summary['bias_accel']['eligible_detected'],
                          accel_total=summary['bias_accel']['trajectories'],
                          accel_over30=summary['bias_accel']['over30_allowed_runs'],
                          accel_10m=summary['bias_accel']['max_first10_m'])), flush=True)


def phase_paths(root, phase):
    directory = root/'cache/development'
    nominal = sorted((directory/'nominal').glob('*.npz'))
    if phase == 1:
        return [p for p in nominal if p.stem.split('_', 2)[2] in STRESS_CASES]
    h0 = [p for profile in PROFILES[1:] for p in sorted((directory/profile).glob('*_clean.npz'))]
    if phase == 2:
        return [p for p in nominal if p.stem.split('_', 2)[2] in STRESS_CASES]+h0
    stress = nominal+sorted((directory/'combined').glob('*.npz'))
    if phase == 3:
        return stress
    return stress+[p for p in h0 if p.parent.name != 'combined']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous', type=Path, default=Path('results/ten-rounds'))
    parser.add_argument('--prepare', choices=('calibration', 'development', 'final'))
    parser.add_argument('--phase', type=int)
    parser.add_argument('--select', action='store_true')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        prepare_cache(args.output, args.previous, args.prepare)
        return
    if args.select:
        candidates = []
        for k in range(35, 43):
            s = json.loads((args.output/f'round{k:02d}/summary.json').read_text())
            # Explicit development ranking: clean false isolations first, then
            # attack runs exceeding 30 m while updates remain allowed, then
            # the worst first-ten-second attack error. Not a release criterion.
            score = [s['clean']['isolated'], sum(v['over30_allowed_runs'] for c,v in s.items() if c != 'clean'),
                     max(v['max_first10_m'] for c,v in s.items() if c != 'clean')]
            candidates.append(dict(round=k, score=score))
        winner = min(candidates, key=lambda c:c['score'])
        chosen = json.loads((args.output/f"round{winner['round']:02d}/frozen.json").read_text())
        write(args.output/'selected.json', dict(ranking=candidates, selected_round=winner['round'], policy=chosen))
        print(json.dumps(dict(selection=winner, ranking=candidates)), flush=True)
        return
    if args.phase < 5:
        threshold = limits(args.output, robust=args.phase >= 2, envelope=args.phase == 4)
        paths = phase_paths(args.output, args.phase)
        hypotheses = {1:'Nominal H0 scalar envelope: can additional causal memory reveal smooth drift?',
                      2:'Broader H0 scalar envelope: does gain survive ordinary sensor/model errors?',
                      3:'Frozen robust limits: stress lower slope, smaller acceleration, delayed drift and jerk.',
                      4:'Cumulative age-dependent H0 envelope plus 25 percent margin and 3-sample confirmation.'}
        for method in range(8):
            evaluate(args.output, args.previous, 11+(args.phase-1)*8+method, method, threshold[method],
                     3 if args.phase == 4 else 1, paths, hypotheses[args.phase])
    else:
        chosen = json.loads((args.output/'selected.json').read_text())['policy']
        for i, profile in enumerate(PROFILES):
            paths = sorted((args.output/'cache/final'/profile).glob('*.npz'))
            evaluate(args.output, args.previous, 43+i, chosen['method_index'], np.array(chosen['limit']),
                     chosen['confirmation'], paths, f'Frozen independent holdout under {profile}; no retuning.')


if __name__ == '__main__':
    main()
