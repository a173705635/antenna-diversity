"""Post-freeze descriptive breakdowns; no model selection or tuning."""
import json
from itertools import groupby
import numpy as np
from research_warm_array import ROOT, SUBSETS, save
from signal_recovery import sky, steering
from presets import mate70_phase_centers


def main():
    rows = json.loads((ROOT/'holdout.json').read_text(encoding='utf-8'))
    breakdown = []
    for subset in SUBSETS:
        for kind in ('normal','attack','weak','near','absent','drift'):
            selected = [r['scores']['candidate'] for r in rows if r['epochs']==64 and r['subset']==subset and r['kind']==kind]
            totals = {key:sum(r[key] for r in selected) for key in ('opportunities','solved','good20','error30','error50')}
            missing = [[e is None for e in r['errors_m']] for r in selected]
            longest = max([0]+[len(list(run)) for flags in missing for flag,run in groupby(flags) if flag])
            breakdown.append(dict(subset=subset, kind=kind, **totals,
                                  unavailable=totals['opportunities']-totals['solved'],
                                  longest_rejected_snapshots=longest))
    geometry = []
    positions = np.array(list(mate70_phase_centers().values()))
    for seed in (112001,112013,112037):
        directions = sky(seed,8)
        for subset, indices in SUBSETS.items():
            a = steering(directions,positions[indices],0.)
            v = steering(np.array([[np.cos(.7),np.sin(.7),0.]]),positions[indices],0.)[:,0]
            retained = 1-abs(v.conj()@a)**2/len(indices)**2
            geometry.append(dict(seed=seed,subset=subset,retained=retained.tolist(),
                                 minimum=float(retained.min()),median=float(np.median(retained)),
                                 noise_penalty_db=(-10*np.log10(retained)).tolist()))
    save(ROOT/'diagnosis.json',dict(breakdown=breakdown,ideal_geometry_only=geometry,
        caveat='Geometry uses generator directions only for offline diagnosis; not receiver inputs or measured suppression proof.',
        ignored_reacquisition='observe reports local/global peak switching; no source reauthentication or continuous carrier tracking implemented',
        timing='20ms independent coherent snapshots; initialization ends at60ms, attack begins60ms, gap0. Labels do not establish continuous tracking.',
        covariance='quality is relative correlation evidence, not calibrated range variance or protection level'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=list(SUBSETS)
    totals=[{key:sum(r[key] for r in breakdown if r['subset']==s and r['kind']!='normal')
             for key in ('opportunities','solved','good20','error30','error50')} for s in labels]
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    x=np.arange(4)
    axes[0].bar(x-.18,[r['good20']/r['opportunities']*100 for r in totals],.36,label='<=20m / all epochs')
    axes[0].bar(x+.18,[r['solved']/r['opportunities']*100 for r in totals],.36,label='Any position / all epochs')
    axes[0].set_ylim(0,100);axes[0].set_ylabel('Percent');axes[0].legend(fontsize=8)
    axes[1].bar(x-.18,[r['error30'] for r in totals],.36,label='>=30m')
    axes[1].bar(x+.18,[r['error50'] for r in totals],.36,label='>=50m')
    axes[1].set_ylabel('Epoch count');axes[1].legend()
    for ax in axes:ax.set_xticks(x,labels,rotation=15);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Frozen holdout: attacks after initialization (sampled-code snapshots)')
    fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    main()
