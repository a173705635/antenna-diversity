"""Independently re-score all saved recovery traces and produce report artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from audit_recovery_fifty import save


def stats(errors):
    e=np.concatenate(errors);valid=np.isfinite(e)
    return dict(opportunities=len(e),solved=int(valid.sum()),good20=int((e<=20).sum()),
                error30=int((e>=30).sum()),error50=int((e>=50).sum()),
                p95_m=float(np.quantile(e[valid],.95)) if valid.any() else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    out=parser.parse_args().output
    rows=[];rechecks=0;paired_runs=0;policy_runs=0;opportunities=0
    for number in range(1,51):
        directory=out/f'round{number:02d}'
        summary=json.loads((directory/'summary.json').read_text())
        frozen=json.loads((directory/'frozen.json').read_text())
        case=summary['case'];method=summary['method']
        methods=['single',method] if method!='refined' else ['single','gated',method]
        strata={}
        for attacked in (False,True):
            errors={m:[] for m in methods};active={m:[] for m in methods};after={m:[] for m in methods}
            for seed in frozen['seeds']:
                z=np.load(directory/f'{seed}_{int(attacked)}.npz')
                t=z['time_s'];post=t>=10
                attack_mask=post & (t<case['attack_end'])
                withdrawal=post & (t>=case['attack_end'])
                for m in methods:
                    e=np.linalg.norm((z[m+'_position']-z['truth_position'])[:,:2],axis=1)
                    np.testing.assert_allclose(e,z[m+'_error'],equal_nan=True)
                    errors[m].append(e[post]);active[m].append(e[attack_mask]);after[m].append(e[withdrawal])
                    rechecks+=len(e);policy_runs+=1
                paired_runs+=1;opportunities+=int(post.sum())
            key='attack' if attacked else 'normal'
            strata[key]={m:stats(errors[m]) for m in methods}
            strata[key+'_active']={m:stats(active[m]) for m in methods}
            if case['attack_end']<=case['duration']:
                strata[key+'_after_withdrawal']={m:stats(after[m]) for m in methods}
            for m,stored in ((method,'candidate_'+key),('single','baseline_'+key)):
                for field in ('opportunities','solved','good20'):
                    assert strata[key][m][field]==summary[stored][field],(number,m,field)
        rows.append(dict(round=number,label=summary['label'],case=case,**strata))
    save(out/'rescored.json',rows)
    lines=['|轮次|问题|方法|正常≤20m|攻击组≤20m|单天线≤20m|候选≥30m|候选P95(m)|',
           '|---:|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        m=r['case']['method'];a=r['attack'][m];b=r['attack']['single'];h=r['normal'][m]
        p95='无解' if a['p95_m'] is None else f"{a['p95_m']:.2f}"
        lines.append(f"|{r['round']}|{r['label']}|{m}|{h['good20']}/{h['opportunities']}|{a['good20']}/{a['opportunities']}|{b['good20']}/{b['opportunities']}|{a['error30']}|{p95}|")
    (out/'round_table.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(2,1,figsize=(13,7),sharex=True,layout='constrained')
    x=np.arange(1,51)
    def fraction(group,field,method):
        return [100*r[group][r['case']['method'] if method=='candidate' else method][field]/r[group][method if method!='candidate' else r['case']['method']]['opportunities'] for r in rows]
    axes[0].plot(x,fraction('attack','good20','single'),color='.6',label='Single antenna')
    axes[0].plot(x,fraction('attack','good20','candidate'),'o-',ms=3,label='Candidate')
    axes[0].plot(x,fraction('normal','good20','candidate'),'--',label='Candidate, normal')
    axes[0].set(ylabel='Sampled epochs within 20 m (%)',ylim=(-3,103));axes[0].legend(ncol=3,fontsize=9)
    axes[1].bar(x,fraction('attack','error30','candidate'),color='#c64b39')
    axes[1].set(ylabel='All epochs: solved error >=30 m (%)',xlabel='Research round',ylim=(0,103))
    for ax in axes:
        for boundary in (10.5,20.5,30.5,40.5):ax.axvline(boundary,color='.7',ls=':')
        ax.grid(axis='y',alpha=.2)
    fig.savefig(out/'rounds.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for ax,number,title in zip(axes.flat,(41,43,45,46),('8 satellites, moving','16 satellites, moving','8 satellites, static, 30 min','Combined errors')):
        z=np.load(out/f'round{number:02d}/830001_1.npz')
        for method,label in (('single','Single'),('gated','Initial projection'),('refined','Refined')):
            ax.plot(z['time_s'],z[method+'_error'],label=label,lw=1)
        ax.axhline(20,color='black',ls='--',lw=.8);ax.axhline(30,color='#c64b39',ls=':',lw=.8)
        ax.set(title=title,xlabel='Snapshot time (s)',ylabel='Horizontal error (m)',yscale='symlog')
        ax.set_ylim(0,1.25*max(np.nanmax(z[m+'_error']) for m in ('single','gated','refined')))
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.savefig(out/'holdout_traces.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
    for number,count in ((41,8),(42,12),(43,16),(44,24)):
        data=[np.load(out/f'round{number:02d}/{seed}_1.npz') for seed in (830001,830002,830003,830004)]
        worst=np.max(np.stack([z['refined_error'] for z in data]),axis=0)
        ax.plot(data[0]['time_s'],worst,label=f'{count} satellites',lw=1)
    ax.axhline(20,color='black',ls='--',lw=.8);ax.axhline(30,color='#c64b39',ls=':',lw=.8)
    ax.set(xlabel='Snapshot time (s)',ylabel='Horizontal error (m)',yscale='symlog',
           title='Worst of four held-out seeds at each epoch: rare failures retained')
    ax.set_ylim(0,20000)
    ax.legend(ncol=4,fontsize=8);ax.grid(alpha=.2)
    fig.savefig(out/'outliers.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    for method,label in (('single','Single'),('gated','Raw covariance projection'),('refined_coherent','Coherent + 2 refinements')):
        z=np.load(out/f'continuous-final/1_{method}.npz')
        ax.plot(z['time_s'],np.linalg.norm(z['position'][:,:2],axis=1),label=label)
    ax.axhline(20,color='black',ls='--',lw=.8);ax.axhline(30,color='#c64b39',ls=':',lw=.8)
    ax.set(xlabel='Observation time (s); attack starts at 0.15 s',ylabel='Horizontal error (m)',title='Independent 1 s continuous DLL/PLL experiment, one seed')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(out/'continuous.png',dpi=180);plt.close(fig)
    source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    archive=out/'final_source';archive.mkdir(exist_ok=True)
    for p in Path(__file__).parent.glob('*.py'):(archive/p.name).write_bytes(p.read_bytes())
    data_hashes={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}
    save(out/'manifest.json',dict(rounds=50,paired_runs=paired_runs,policy_runs=policy_runs,
         paired_post10_opportunities=opportunities,rescored_policy_epochs=rechecks,
         source_sha256=source,data_sha256=data_hashes,
         note='Paired runs include normal and attack inputs; repeated seeds/epochs are not independent trials. Final source reproduces early policies; early frozen hashes record development versions.'))
    print(json.dumps(dict(rounds=50,paired_runs=paired_runs,policy_runs=policy_runs,
                         opportunities=opportunities,rescored_policy_epochs=rechecks)),flush=True)


if __name__=='__main__':main()
