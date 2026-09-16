"""Final paired baseline, error-estimator diagnosis and ten-round figures."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_navigation_iterations import STRESS_CASES, aggregate, metrics, run, write


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    root = parser.parse_args().output
    baseline = json.loads((root/'round01/frozen.json').read_text())['policy']
    candidate = json.loads((root/'round08/frozen.json').read_text())['policy']
    rows = []
    for path in sorted((root/'data/holdout').glob('*.npz')):
        kind, seed = path.stem.split('_')
        with np.load(path) as data:
            for case in STRESS_CASES:
                result = run(data, case, baseline)
                rows.append(dict(motion=kind, seed=int(seed), case=case, **metrics(data, result, case)))
    write(root/'round10/baseline_rows.json', rows)
    write(root/'round10/baseline_summary.json', aggregate(rows))
    rounds = [json.loads((root/f'round{k:02d}/summary.json').read_text()) for k in range(1, 11)]
    write(root/'all_rounds.json', rounds)
    # Preselected first stress seed. Compare estimator states, not true biases.
    with np.load(root/'data/stress/static_760001.npz') as data:
        clean = run(data, 'clean', candidate); attacked = run(data, 'bias_accel', candidate)
        t = data['time_s']
        err = np.linalg.norm(attacked['state'][:, :2]-data['truth_position_m'][:, :2], axis=1)
        diagnostic = dict(seed=760001, motion='static', case='bias_accel',
                          clean_final_effective_acceleration=clean['state'][-1, 6:].tolist(),
                          attacked_final_effective_acceleration=attacked['state'][-1, 6:].tolist(),
                          difference=(attacked['state'][-1, 6:]-clean['state'][-1, 6:]).tolist(),
                          final_error_m=float(err[-1]), isolation=bool(np.any(attacked['mode'] == 1)))
        write(root/'acceleration_absorption.json', diagnostic)
        np.savez_compressed(root/'acceleration_absorption.npz', time_s=t,
                            clean_state=clean['state'], attacked_state=attacked['state'], error_m=err)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    k = np.arange(1, 9)
    axes[0].plot(k, [r['partial3']['max_first10_m'] for r in rounds[:8]], 'o-', label='partial3')
    axes[0].plot(k, [r['prn_bias']['max_first10_m'] for r in rounds[:8]], 's-', label='PRN bias / fast drag')
    axes[0].axhline(20, ls='--', color='gray'); axes[0].set(xlabel='Development round', ylabel='Worst first 10 s error (m)')
    axes[0].legend()
    axes[1].bar(k, [r['clean']['mean_rejected_time_s']/80*3600 for r in rounds[:8]])
    axes[1].set_yscale('symlog', linthresh=1)
    axes[1].axhline(5, ls='--', color='gray')
    axes[1].set(xlabel='Development round', ylabel='Rejected GNSS update time (s / hour equivalent)')
    fig.tight_layout(); fig.savefig(root/'rounds.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(t, clean['state'][:, 6], label='clean estimate')
    axes[0].plot(t, attacked['state'][:, 6], label='attacked estimate')
    axes[0].set(xlabel='Time (s)', ylabel='Effective x acceleration error (m/s2)'); axes[0].legend()
    axes[1].plot(t, err); axes[1].axhline(30, ls='--', color='gray')
    axes[1].set(xlabel='Time (s)', ylabel='Horizontal position error (m)')
    fig.tight_layout(); fig.savefig(root/'absorption.png', dpi=170); plt.close(fig)
    write(root/'analysis_manifest.json', dict(
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        data_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted((root/'data').glob('*/*.npz'))},
        primary_rows=sum(len(json.loads((root/f'round{k:02d}/rows.json').read_text())) for k in range(1, 11)),
        holdout_baseline_rows=len(rows)))
    print(json.dumps(dict(baseline=aggregate(rows), diagnosis=diagnostic)), flush=True)


if __name__ == '__main__':
    main()
