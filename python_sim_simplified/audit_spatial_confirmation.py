"""Train on continuous H0 only, freeze, then evaluate temporal confirmation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from audit_navigation_fusion import carrier_stream, experiment
from audit_spatial_diversity import W, model_observations
from spatial_spoofing import spatial_scores, calibrate_threshold, confirm_windows


def stream_scores(kind, case, seed):
    carrier = carrier_stream(np.arange(8000)*.01, kind, case, seed)
    return spatial_scores(carrier.reshape(800, 10, 4, 8), W)['cluster3']


def evaluate(scores, threshold, count):
    raw = [dict(decision_s=float(.091+k*.1), state=int(s < threshold))
           for k, s in enumerate(scores)]
    rows = confirm_windows(raw, count)
    alarm = np.array([r['state'] == 1 for r in rows])
    time = np.array([r['decision_s'] for r in rows])
    first = time[alarm]; pre = first[first < 20]; post = first[first >= 20]
    starts = alarm & ~np.r_[False, alarm[:-1]]
    return dict(first_alert_s=float(first[0]) if len(first) else None,
                pre_attack_alert_s=float(pre[0]) if len(pre) else None,
                post_attack_delay_s=float(post[0]-20) if len(post) else None,
                alert_episodes=int(starts.sum()), alert_windows=int(alarm.sum()),
                longest_raw_run=max(r['evidence_run'] for r in rows),
                latched_isolation_s=float(80-first[0]) if len(first) else 0.)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--navigation', action='store_true')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    original = calibrate_threshold(spatial_scores(model_observations(710001, 2000, 'los'), W)['cluster3'])
    train = np.array([stream_scores(kind, 'clean', seed)
                      for kind in ('static', 'turn') for seed in range(730001, 730025)])
    # Smallest rolling three-window maximum: strict '<' yields zero training
    # confirmations. This is an empirical operating point, not a false-rate bound.
    threshold = float(min(original, sliding_window_view(train, 3, axis=1).max(axis=-1).min()))
    policies = {'original': [original, 1], 'three_only': [original, 3],
                'three_calibrated': [threshold, 3]}
    frozen = dict(source_kind='synthetic_continuous_carrier', policies=policies,
                  training_seeds=[730001, 730024], holdout_seeds=[740001, 740024],
                  training_trajectories=48, duration_s=80, confirmation_count_chosen_before_data=3,
                  threshold_rule='min(original, min_training_rolling_three_window_maximum)',
                  sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                           for name in ('audit_spatial_confirmation.py', 'audit_navigation_fusion.py',
                                        'navigation_fusion.py', 'inertial.py', 'spatial_spoofing.py',
                                        'audit_spatial_diversity.py', 'presets.py')})
    (args.output/'frozen.json').write_text(json.dumps(frozen, indent=2), encoding='utf-8')
    np.savez_compressed(args.output/'training_scores.npz', scores=train)
    print(json.dumps(dict(frozen=policies)), flush=True)
    rows = []; saved = {}
    for kind in ('static', 'turn'):
        for seed in range(740001, 740025):
            for case in ('clean', 'all_spoof', 'partial3', 'slow', 'withdraw', 'prn_bias'):
                scores = stream_scores(kind, case, seed)
                saved[f'{kind}_{seed}_{case}'] = scores
                for policy, (limit, count) in policies.items():
                    rows.append(dict(motion=kind, seed=seed, case=case, policy=policy,
                                     **evaluate(scores, limit, count)))
    np.savez_compressed(args.output/'holdout_scores.npz', **saved)
    (args.output/'detection.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print('Holdout complete: 288 trajectories, 864 policy rows', flush=True)
    if args.navigation:
        nav = []
        # Historical seeds are paired regression, not the independent holdout.
        for kind in ('static', 'turn'):
            for seed in range(720001, 720005):
                for case in ('clean', 'all_spoof', 'partial3', 'slow', 'withdraw', 'prn_bias'):
                    result, _ = experiment(kind, case, seed, threshold, confirmation_count=3)
                    nav.extend(result)
        (args.output/'navigation.json').write_text(json.dumps(nav, indent=2), encoding='utf-8')
        print('Navigation regression complete: 96 rows', flush=True)


if __name__ == '__main__':
    main()
