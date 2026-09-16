"""Frozen navigation holdouts, separate from the hundred source-change rounds."""
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from audit_recovery_fifty import save
from audit_recovery_two_hundred import score
from research_hundred import load
from validate_research_hundred import source_closure
ROOT=Path('results/research-third-hundred')
CASES=[(101001,'normal_low',dict(cn0=35.),False),(101002,'cold_low',dict(cn0=35.,attack_start=0.),True),
       (101003,'phase_cold',dict(phase_deg=20.,attack_start=0.),True),(101004,'absent',dict(genuine=False,attack_start=0.),True),
       (101005,'two',dict(antennas=2),True),(101006,'moving',dict(duration=1.,velocity_east=2.),True)]

def freeze():
    save(ROOT/'navigation_freeze.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),cases=CASES,seeds=[102001,102013,102037],
      versions=[217,225],meaning='217 robust correlated-phase GLS comparator;225 final covariance/route model. Both declared before holdout, no winner selected from holdout.',
      sources=source_closure(['validate_research_third.py','navigation_third.py','navigation_third_cases.py','audit_array_navigation.py',
          str(ROOT/'iteration217/navigation_third.py'),str(ROOT/'iteration225/navigation_third.py'),str(ROOT/'baseline_navigation.py'),
          'results/research-hundred/iteration036/audit_array_navigation.py'])))

def run():
    from continuous_improvement import Case
    from audit_array_navigation import receiver
    from navigation_third_cases import generate,KINDS
    frozen=json.loads((ROOT/'navigation_freeze.json').read_text())
    for name,digest in frozen['sources'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    models={'baseline200':load(ROOT/'baseline_navigation.py').candidate}
    models.update({str(n):load(ROOT/f'iteration{n}/navigation_third.py').candidate for n in frozen['versions']})
    capture=load(Path('results/research-hundred/iteration036/audit_array_navigation.py')).receiver
    folder=ROOT/'navigation_validation';folder.mkdir();rows=[]
    def evaluate(pair,truth,seed,kind,domain):
        positions={name:f(pair) for name,f in models.items()}
        np.savez_compressed(folder/f'{seed}_{kind}_positions.npz',**positions)
        row=dict(seed=seed,kind=kind,domain=domain,scores={name:score(p,truth) for name,p in positions.items()},
                 unobserved_time_s=float(np.sum(np.maximum(np.diff(pair['joint']['time_s'])-.01,0))))
        rows.append(row);save(folder/f'{seed}_{kind}.json',row);print(json.dumps(row),flush=True)
    for seed,kind,changes,attacked in frozen['cases']:
        case=Case(**(dict(duration=.4)|changes));array,meta=capture(case,seed,attacked);joint,jmeta=receiver(case,seed,attacked)
        for route,z in (('capture',array),('joint',joint)):np.savez_compressed(folder/f'{seed}_{route}.npz',**z)
        save(folder/f'{seed}_metadata.json',dict(capture=meta,joint=jmeta))
        truth=joint.pop('truth_position');array.pop('truth_position')
        evaluate(dict(capture=array,joint=joint),truth,seed,kind,'RF-derived')
    for seed in frozen['seeds']:
        for kind in KINDS:
            pair,truth=generate(seed,kind);evaluate(pair,truth,seed,kind,'observation')
    totals={domain:{name:{key:sum(r['scores'][name][key] for r in rows if r['domain']==domain)
       for key in ('opportunities','solved','good20','error30','error50')} for name in models} for domain in ('RF-derived','observation')}
    save(ROOT/'navigation_validation.json',dict(totals=totals,rows=rows,freeze=frozen));print(json.dumps(totals),flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
