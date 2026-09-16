"""Predeclared frozen validation; no validation case counts as an iteration."""
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from research_hundred import load
from research_next_navigation import ROOT
from validate_research_hundred import source_closure
from audit_recovery_fifty import save
from audit_recovery_two_hundred import score

CASES=[(83101,'normal_low',dict(cn0=35.),False),
       (83102,'cold_low',dict(cn0=35.,attack_start=0.),True),
       (83103,'phase_cold',dict(phase_deg=20.,attack_start=0.),True),
       (83104,'absent_cold',dict(genuine=False,attack_start=0.),True),
       (83105,'two',dict(antennas=2),True),
       (83106,'moving',dict(duration=1.,velocity_east=2.),True),
       (83107,'normal',{},False)]

def freeze():
    save(ROOT/'navigation_freeze.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),cases=CASES,
        observation_seeds=[84101,84117,84139],versions=[197,200],
        purpose='197 availability research comparator; 200 unresolved-conflict memory. Neither authenticates common bias. No selection after holdout.',
        sources=source_closure(['validate_research_next.py','navigation_next.py','navigation_next_cases.py',
            'audit_array_navigation.py',str(ROOT/'iteration197/navigation_next.py'),str(ROOT/'iteration200/navigation_next.py'),
            'results/research-hundred/iteration036/audit_array_navigation.py'])))

def run():
    from continuous_improvement import Case
    from audit_array_navigation import receiver
    from navigation_hundred import candidate as baseline
    from navigation_next_cases import generate,KINDS
    frozen=json.loads((ROOT/'navigation_freeze.json').read_text())
    for name,digest in frozen['sources'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    models={str(n):load(ROOT/f'iteration{n}/navigation_next.py').candidate for n in frozen['versions']}
    capture=load(Path('results/research-hundred/iteration036/audit_array_navigation.py')).receiver
    folder=ROOT/'navigation_validation';folder.mkdir();rows=[]
    def evaluate(pair,truth,seed,kind,domain):
        positions={'baseline':baseline(pair['joint'])}
        if not pair['joint']['joint_usable']:positions['baseline'][:]=np.nan
        positions.update({name:model(pair) for name,model in models.items()})
        np.savez_compressed(folder/f'{seed}_{kind}_positions.npz',**positions)
        row=dict(seed=seed,kind=kind,domain=domain,scores={name:score(p,truth) for name,p in positions.items()})
        rows.append(row);save(folder/f'{seed}_{kind}.json',row);print(json.dumps(row),flush=True)
    for seed,kind,changes,attacked in frozen['cases']:
        case=Case(**(dict(duration=.4)|changes));array,meta=capture(case,seed,attacked);joint,jmeta=receiver(case,seed,attacked)
        for name,z in (('capture',array),('joint',joint)):np.savez_compressed(folder/f'{seed}_{name}.npz',**z)
        save(folder/f'{seed}_metadata.json',dict(capture=meta,joint=jmeta))
        truth=joint.pop('truth_position');array.pop('truth_position')
        evaluate(dict(capture=array,joint=joint),truth,seed,kind,'RF-derived')
    for seed in frozen['observation_seeds']:
        for kind in KINDS:
            pair,truth=generate(seed,kind);evaluate(pair,truth,seed,kind,'observation')
    totals={domain:{name:{key:sum(r['scores'][name][key] for r in rows if r['domain']==domain)
        for key in ('opportunities','solved','good20','error30','error50')} for name in ('baseline','197','200')}
        for domain in ('RF-derived','observation')}
    save(ROOT/'navigation_validation.json',dict(rows=rows,totals=totals,freeze=frozen))
    print(json.dumps(totals),flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
