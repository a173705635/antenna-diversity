"""Reproducible forty-round summaries, paired diagnostics and provenance."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_forty_rounds import base_run
from audit_navigation_iterations import aggregate, write
from audit_temporal_rounds import feature_path
from navigation_monitors import first_evidence


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    root = parser.parse_args().output
    table = []; final = []; controls = []; count = 0
    for k in range(11, 51):
        folder = root/f'round{k:02d}'
        f = json.loads((folder/'frozen.json').read_text())
        r = json.loads((folder/'rows.json').read_text()); b = json.loads((folder/'baseline_rows.json').read_text())
        s = aggregate(r); bs = aggregate(b); count += len(r)
        table.append(dict(round=k, method=f['method'], rows=len(r),
                          clean_candidate=s['clean']['isolated'], clean_baseline=bs['clean']['isolated'],
                          accel_total=s['bias_accel']['trajectories'],
                          accel_over30=s['bias_accel']['over30_allowed_runs'],
                          accel_over30_baseline=bs['bias_accel']['over30_allowed_runs'],
                          new_alerts=sum(x['new_alert_s'] is not None for x in r),
                          summary=s, baseline_summary=bs))
        if k >= 43:
            final.extend(r); controls.extend(b)
    write(root/'all_rounds.json', table)
    write(root/'final_summary.json', dict(candidate=aggregate(final), baseline=aggregate(controls),
        nominal={motion:dict(candidate=aggregate([r for r in final if r['profile']=='nominal' and r['motion']==motion]),
                             baseline=aggregate([r for r in controls if r['profile']=='nominal' and r['motion']==motion]))
                 for motion in ('static', 'turn')}))
    md = ['| Round | Method | Runs | Clean isolation B/C | Bias accel >30 m while allowed B/C | New alerts |',
          '|---|---|---:|---:|---:|---:|']
    for r in table:
        md.append(f"| {r['round']} | {r['method']} | {r['rows']} | {r['clean_baseline']}/{r['clean_candidate']} | "
                  f"{r['accel_over30_baseline']}/{r['accel_over30']} ({r['accel_total']} runs) | {r['new_alerts']} |")
    (root/'round_table.md').write_text('\n'.join(md)+'\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 4))
    k = np.array([r['round'] for r in table]); width=.4
    ax.bar(k-width/2, [r['accel_over30_baseline']/r['accel_total'] for r in table], width, label='Baseline')
    ax.bar(k+width/2, [r['accel_over30']/r['accel_total'] for r in table], width, label='Candidate')
    for edge in (18.5, 26.5, 34.5, 42.5):
        ax.axvline(edge, color='gray', ls=':')
    ax.set(xlabel='Round (different cohorts across phases)', ylabel='Unsafe update fraction\n(bias-acceleration case)',
           xticks=k[::2], ylim=(0, 1.08)); ax.legend()
    fig.tight_layout(); fig.savefig(root/'rounds.png', dpi=160); plt.close(fig)
    selected = json.loads((root/'selected.json').read_text())['policy']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4)); diagnostic = []
    for ax, motion in zip(axes, ('static', 'turn')):
        path = root/f'cache/final/nominal/{motion}_780001_bias_accel.npz'
        with np.load(path) as d, np.load(feature_path(root, 'final', 'nominal', motion, 780001, 'bias_accel')) as f:
            alarm = first_evidence(f['features'][selected['method_index']], np.full(800, selected['threshold']),
                                   3, d['mode'] == 1)
            result = (dict(state=d['state']) if alarm is None else
                      base_run(d, d['gp'], d['gv'], d['spatial'], selected['baseline_policy'], alert=alarm))
            error = np.linalg.norm(result['state'][:, :2]-d['truth_position_m'][:, :2], axis=1)
            be = np.linalg.norm(d['state'][:, :2]-d['truth_position_m'][:, :2], axis=1)
            ax.plot(d['time_s'], be, label='Baseline'); ax.plot(d['time_s'], error, label='Candidate')
            ax.axhline(30, ls='--', color='gray'); ax.axvline(20, ls=':', color='gray')
            alert_s = float(d['time_s'][alarm+1]) if alarm is not None else None
            if alert_s is not None:
                ax.axvline(alert_s, ls='--', color='orange')
            ax.set(title=motion, xlabel='Time (s)', ylabel='Horizontal error (m)'); ax.legend()
            diagnostic.append(dict(motion=motion, seed=780001, alert_s=alert_s,
                                   final_error_m=float(error[-1]), baseline_final_error_m=float(be[-1])))
    fig.tight_layout(); fig.savefig(root/'static_turn.png', dpi=170); plt.close(fig)
    write(root/'diagnostic.json', diagnostic)
    inputs = sorted((root/'cache').glob('*/*/*.npz'))+sorted((root/'temporal').glob('*/*/*.npz'))
    write(root/'manifest.json', dict(primary_runs=count, paired_baseline_runs=count,
                                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                    input_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}))
    print(json.dumps(dict(primary_runs=count, paired_baseline_runs=count, inputs=len(inputs), diagnostic=diagnostic)), flush=True)


if __name__ == '__main__':
    main()
