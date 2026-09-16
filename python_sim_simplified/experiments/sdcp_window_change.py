"""Exploratory adjacent-window change test, without wrapping or position solution."""
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from experiments.paths import RESULTS

OUTPUT = RESULTS/'sdcp_multipath'


def change_score(r):
    ends = np.arange(399,len(r),10)
    score = np.array([np.mean(r[j-199:j+1])-np.mean(r[j-399:j-199]) for j in ends])
    return ends, score


def alarm(score,limit):
    return np.array([j>=2 and all(abs(score[j-2:j+1])>limit) for j in range(len(score))])


def main():
    out=OUTPUT
    plan=json.loads((out/'plan.json').read_text(encoding='utf-8'))
    maxima=[]
    for seed in plan['calibration_seeds']:
        with np.load(out/f'cal_{seed}_decision.npz') as d:
            _,s=change_score(d['residual_m'])
            maxima.append(float(np.max(abs(s))))
    limit=1.25*max(maxima)
    supplement=out/'window_change'
    if supplement.exists():
        shutil.rmtree(supplement)
    supplement.mkdir()
    (supplement/'threshold.json').write_text(json.dumps(dict(threshold_m=limit,
        calibration_max_abs_change_m=maxima,margin=1.25,
        design='Exploratory amendment after primary calibration exposed half-cycle offsets.',
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2),encoding='utf-8')
    rows=[]
    fig,axs=plt.subplots(4,4,figsize=(18,13),sharex=True)
    for ax,c in zip(axs.flat,plan['cases']):
        name=c['name']
        for seed in plan['test_seeds']:
            with np.load(out/f'{seed}_{name}_decision.npz') as d:
                t=d['time_s']
                ends,s=change_score(d['residual_m'])
            dt=t[ends]
            a=alarm(s,limit)
            onset=c.get('onset_s',.4 if name=='drift' else 0.)
            event=dt>=onset
            fired=dt[a & event]
            row=dict(case=name,seed=seed,detected=bool(len(fired)),
                first_alarm_s=float(fired[0]) if len(fired) else None,
                delay_s=float(fired[0]-onset) if len(fired) else None,
                event_alarm_count=int(np.sum(a & event)),event_decisions=int(np.sum(event)),
                pre_event_alarm_count=int(np.sum(a & ~event)),
                max_abs_score_mm=float(np.max(abs(s))*1000),
                tail_score_mean_mm=float(np.mean(s[dt>=1.1])*1000))
            rows.append(row)
            np.savez_compressed(supplement/f'{seed}_{name}.npz',time_s=dt,score_m=s,alarm=a)
            ax.plot(dt,s*1000,lw=1,label=str(seed))
        ax.axhline(limit*1000,color='black',ls='--',lw=.7)
        ax.axhline(-limit*1000,color='black',ls='--',lw=.7)
        ax.set_title(name,fontsize=10)
        ax.set_ylabel('Adjacent mean change (mm)')
        ax.grid(alpha=.2)
    for ax in axs[-1]:
        ax.set_xlabel('Time (s)')
    axs.flat[0].legend(fontsize=7)
    fig.suptitle('Exploratory change detector: two adjacent 200 ms windows')
    fig.tight_layout()
    fig.savefig(supplement/'all_cases.png',dpi=160)
    plt.close(fig)
    (supplement/'results.json').write_text(json.dumps(rows,indent=2,allow_nan=False),encoding='utf-8')
    lines=['| 场景 | 报警记录数 | 报警判定数/判定数 | 首次报警延迟 ms | 最大绝对变化 mm |',
        '|---|---:|---:|---|---|']
    for c in plan['cases']:
        rs=[r for r in rows if r['case']==c['name']]
        delays=' / '.join('—' if r['delay_s'] is None else f"{r['delay_s']*1000:.0f}" for r in rs)
        maximum=' / '.join(f"{r['max_abs_score_mm']:.3f}" for r in rs)
        lines.append(f"| {c['name']} | {sum(r['detected'] for r in rs)}/3 | {sum(r['event_alarm_count'] for r in rs)}/{sum(r['event_decisions'] for r in rs)} | {delays} | {maximum} |")
    (supplement/'result_table.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Threshold mm:',limit*1000)
    print('\n'.join(lines))


if __name__=='__main__':
    main()
