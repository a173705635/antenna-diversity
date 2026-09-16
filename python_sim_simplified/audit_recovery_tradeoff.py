"""Fresh paired holdout of a development Pareto alternative, outside numbered rounds."""
import hashlib,json
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
from continuous_improvement import Case,experiment
from continuous_admission import Gate,admit
from audit_recovery_two_hundred import SOURCES,score
from audit_recovery_fifty import save


def main():
    out=Path('results/recovery-two-hundred');root=out/'tradeoff-holdout';root.mkdir(exist_ok=False)
    choices=json.loads((out/'development_selection.json').read_text())
    feasible=[v for v in choices if v['normal']['solved']>=.95*v['normal']['opportunities']
              and v['attack']['solved']>=.5*v['attack']['opportunities']]
    alternate=min(feasible,key=lambda v:(v['normal']['error30']+v['attack']['error30'],
        v['normal']['error50']+v['attack']['error50'],-v['normal']['good20'],-v['attack']['good20']))
    primary=json.loads((out/'selected.json').read_text())
    changes=[{},dict(cn0=40),dict(phase_deg=5),dict(phase_deg=10),dict(angle=15),dict(angle=0),
             dict(genuine=False),dict(antennas=3),dict(duration=2.),dict(attack_start=0.)]
    save(root/'frozen.json',dict(primary=primary,alternate=alternate,changes=changes,seeds=[960001,960002],
        rationale='Secondary objective minimizes all >=30m failures before >=50m severity; selected on development data only. New seeds; primary 161-200 unchanged.',
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (*SOURCES,'audit_recovery_tradeoff.py')}))
    methods=list(dict.fromkeys([primary['method'],alternate['method']]))
    rows=[]
    for index,change in enumerate(changes):
        case=replace(Case(),**change)
        for seed in (960001,960002):
            for attack in (False,True):
                z,record=experiment(case,seed,attack,methods)
                for name,policy in (('primary',primary),('alternate',alternate)):
                    m=policy['method'];result=admit(z['satellite_positions'],z[m+'_ranges'],z[m+'_lock'],
                                                 z[m+'_cn0'],z[m+'_gain'],Gate(**policy['gate']))
                    z[name+'_position']=result['position'];z[name+'_scores']=result['scores']
                key=f'{index:02d}_{seed}_{int(attack)}'
                np.savez_compressed(root/f'{key}.npz',**z);save(root/f'{key}.json',record)
                rows.append(dict(case=index,seed=seed,attack=attack,**{m:score(z[m+'_position'],z['truth_position'])
                    for m in methods+['primary','alternate']}));save(root/'rows.json',rows)
        print('tradeoff independent case '+str(index),flush=True)
    summary={}
    for attack in (False,True):
        for m in methods+['primary','alternate']:
            summary[m+('_attack' if attack else '_normal')]={k:sum(r[m][k] for r in rows if r['attack']==attack)
                for k in ('opportunities','solved','good20','error30','error50')}
    save(root/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
