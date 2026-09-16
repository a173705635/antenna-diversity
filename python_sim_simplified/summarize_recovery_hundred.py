"""Re-score all 50 rounds, retaining unavailable outputs and continuous warmup."""
import hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_recovery_fifty import save
from audit_recovery_hundred import metrics


def main():
    out=Path('results/recovery-hundred');rows=[];checked=0;comparisons=0
    for number in range(51,101):
        directory=out/f'round{number:03d}'
        original=json.loads((directory/'summary.json').read_text())
        continuous=81<=number<=90
        methods=('single','raw_covariance','fold','despread','admitted') if continuous else ('baseline','candidate')
        totals={m+suffix:{k:0 for k in ('opportunities','solved','good20','error30','error50')}
                for m in methods for suffix in ('_normal','_attack')}
        for path in sorted(directory.glob('*.npz')):
            if path.stem.endswith('_separation'):continue
            z=np.load(path);flag=int(path.stem.split('_')[-1]);t=z['time_s']
            mask=np.ones(len(t),bool) if continuous else t>=10
            for method in methods:
                result=metrics(z[method+'_position'],z['truth_position'],mask)
                for key in totals[method+('_attack' if flag else '_normal')]:
                    totals[method+('_attack' if flag else '_normal')][key]+=result[key]
                checked+=int(mask.sum())
            comparisons+=1
        for name,values in totals.items():
            for key,value in values.items():assert value==original[name][key],(number,name,key)
        full={}
        if continuous:
            seeds=json.loads((directory/'frozen.json').read_text())['seeds']
            for name,values in totals.items():
                full[name]=dict(values,opportunities=84*len(seeds),unrecorded_warmup_epochs=20*len(seeds))
        rows.append(dict(round=number,label=original['label'],continuous=continuous,**totals,full_attack_clock=full))
    save(out/'rescored.json',rows)
    final=[]
    root=out/'continuous-holdout'
    methods=('single','raw_covariance','fold','despread','admitted','final')
    for bits in (False,True):
        for attack in (False,True):
            totals={m:{k:0 for k in ('opportunities','solved','good20','error30','error50')} for m in methods}
            for seed in (890001,890002,890003):
                z=np.load(root/f'{int(bits)}_{seed}_{int(attack)}.npz')
                for m in methods:
                    result=metrics(z[m+'_position'],z['truth_position'],np.ones(len(z['time_s']),bool))
                    for k in totals[m]:totals[m][k]+=result[k]
            final.append(dict(data_bits=bits,attack=attack,conditional_window=totals,
                full_attack_clock={m:dict(v,opportunities=252,unrecorded_warmup_epochs=60) for m,v in totals.items()}))
    save(out/'continuous_holdout_summary.json',final)
    lines=['|轮次|研究条件|候选正常≤20m|候选攻击≤20m|对照攻击≤20m|候选攻击≥30m|候选攻击≥50m|',
           '|---:|---|---:|---:|---:|---:|---:|']
    for r in rows:
        candidate='despread' if r['continuous'] else 'candidate'
        baseline='raw_covariance' if r['continuous'] else 'baseline'
        a=r[candidate+'_attack'];h=r[candidate+'_normal'];b=r[baseline+'_attack']
        lines.append(f"|{r['round']}|{r['label']}|{h['good20']}/{h['opportunities']}|{a['good20']}/{a['opportunities']}|{b['good20']}/{b['opportunities']}|{a['error30']}|{a['error50']}|")
    (out/'round_table.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(2,1,figsize=(13,7),sharex=True,layout='constrained')
    x=np.arange(51,101)
    for group,label,style in (('candidate_attack','Candidate / despread','o-'),('baseline_attack','Raw / raw covariance','--'),('candidate_normal','Normal candidate','-')):
        values=[]
        for r in rows:
            name=group
            if r['continuous']:name=name.replace('candidate','despread').replace('baseline','raw_covariance')
            v=r[name];values.append(100*v['good20']/v['opportunities'])
        axes[0].plot(x,values,style,label=label,ms=3)
    axes[0].set(ylabel='Recorded epochs within 20 m (%)',ylim=(-2,103));axes[0].legend(fontsize=8,ncol=3)
    axes[1].bar(x,[r['despread_attack' if r['continuous'] else 'candidate_attack']['error30'] for r in rows])
    axes[1].set(ylabel='Output errors >=30 m (count)',xlabel='Round; 81-90 are continuous experiments')
    for ax in axes:
        for boundary in (60.5,70.5,80.5,90.5):ax.axvline(boundary,color='.6',ls=':')
        ax.grid(axis='y',alpha=.2)
    fig.savefig(out/'rounds.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,attack in zip(axes,(False,True)):
        for method,label in (('baseline','Raw position'),('candidate','Admitted position')):
            data=[r[method+('_attack' if attack else '_normal')] for r in rows[-10:]]
            ax.plot(np.arange(91,101),[100*d['solved']/d['opportunities'] for d in data],'o-',label=label)
        ax.set(title='Attack' if attack else 'Normal',xlabel='Independent holdout round',ylabel='Available epochs (%)',ylim=(-2,103))
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(out/'availability_cost.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    names=['raw_covariance','fold','despread','admitted','final'];labels=['Raw covariance','Fold ms','Despread','Residual check','Final admission']
    for i,bits in enumerate((False,True)):
        r=next(r for r in final if r['data_bits']==bits and r['attack'])
        ax.bar(np.arange(5)+i*.35,[100*r['full_attack_clock'][m]['good20']/252 for m in names],width=.35,label='Unknown data bits' if bits else 'Constant data bits')
    ax.set(xticks=np.arange(5)+.175,xticklabels=labels,ylabel='All scheduled attack epochs within 20 m (%)',ylim=(0,100),title='Independent continuous holdout; initial 200 ms counted unavailable')
    ax.legend();ax.grid(axis='y',alpha=.2);fig.savefig(out/'continuous_holdout.png',dpi=170);plt.close(fig)
    source=out/'final_source';source.mkdir(exist_ok=True)
    for p in Path('.').glob('*.py'):(source/p.name).write_bytes(p.read_bytes())
    hashes={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}
    save(out/'manifest.json',dict(rounds=50,numbered_input_comparisons=comparisons,scored_policy_epochs=checked,
        data_sha256=hashes,continuous_extra_trials=12,
        scope='Cached inputs and repeated seeds are not independent trials; continuous warmup retained separately.'))
    print(json.dumps(dict(rounds=50,comparisons=comparisons,scored_policy_epochs=checked,continuous_holdout=final)),flush=True)


if __name__=='__main__':main()
