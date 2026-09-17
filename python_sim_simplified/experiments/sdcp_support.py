"""Shared SDCP experiment serialization and source verification."""
import hashlib
import json

import numpy as np

from core.provenance import source_hashes


def statistics(x):
    return {'epochs': len(x), 'mean_mm': float(np.mean(x)*1000),
            'std_mm': float(np.std(x)*1000),
            'rms_mm': float(np.sqrt(np.mean(x*x))*1000),
            'max_abs_mm': float(np.max(abs(x))*1000)}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False,
        default=lambda v: v.tolist() if isinstance(v, np.ndarray) else str(v)), encoding='utf-8')


def verify_sources(root, expected):
    """Verify either a preserved snapshot or the current maintained sources."""
    if (root/'code').is_dir():
        actual = {name: hashlib.sha256((root/'code'/name).read_bytes()).hexdigest()
                  for name in expected}
        mode = 'snapshot'
    else:
        actual = source_hashes()
        mode = 'current_source'
    assert actual == expected, f'{mode} source hashes differ from the recorded run'
    return mode


def plot_case_scores(plan, traces, limit, ylabel, title, output):
    """One scenario per row and one independent noise seed per column."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(plan['cases']), len(plan['test_seeds']),
                             figsize=(15, 2*len(plan['cases'])),
                             sharex=True, sharey='row', squeeze=False,
                             layout='constrained')
    for row, case in enumerate(plan['cases']):
        for col, seed in enumerate(plan['test_seeds']):
            ax = axes[row, col]
            t, score, alarm = traces[case['name'], seed]
            ax.plot(t, score*1000, color=f'C{col}', lw=1)
            ax.scatter(t[alarm], score[alarm]*1000, color='red', s=7, zorder=3)
            ax.axhline(limit*1000, color='black', ls='--', lw=.7)
            ax.axhline(-limit*1000, color='black', ls='--', lw=.7)
            onset = case.get('onset_s', .4 if case['name']=='drift' else 0.)
            if onset > 0:
                ax.axvline(onset, color='gray', ls=':', lw=.8)
            ax.set_title(f"{case['name']} | seed {seed}", fontsize=9)
            ax.grid(alpha=.2)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if row == len(plan['cases'])-1:
                ax.set_xlabel('Time (s)')
    fig.suptitle(title+'\nColumns: independent noise seeds. Dashed: thresholds. Red dots: confirmed alarms.')
    fig.savefig(output, dpi=160)
    plt.close(fig)
