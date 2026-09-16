"""Development comparison of solution-separation admission, before final holdout."""
from pathlib import Path
import json
import numpy as np
from audit_recovery_fifty import save
from audit_recovery_hundred import metrics
from recovery_admission import Policy,admit
from solution_separation import separation_scores


def main():
    out=Path('results/recovery-hundred')
    policy=Policy(**json.loads((out/'selected.json').read_text())['policy'])
    records=[]
    for name in ('dev42','dev44','dev46'):
        for path in sorted((out/'inputs'/name).glob('*.npz')):
            z=np.load(path);result=admit(z['time_s'],z['satellites'],z['refined_ranges'],z['refined_valid'],policy)
            scores=separation_scores(z['satellites'],z['refined_ranges'],result['selected'],result['position'])
            records.append((path.stem.endswith('_1'),z['time_s'],z['truth_position'],result['position'],scores))
    rows=[]
    for limit in (3.,5.,10.,15.,20.,30.,50.,100.):
        totals={flag:{k:0 for k in ('opportunities','solved','good20','error30','error50')} for flag in (False,True)}
        for flag,t,truth,position,score in records:
            p=position.copy();p[score>limit]=np.nan
            m=metrics(p,truth,t>=10)
            for k in totals[flag]:totals[flag][k]+=m[k]
        rows.append(dict(limit_m=limit,normal=totals[False],attack=totals[True]))
    # Explicit development objective includes normal severe errors as well as attack errors.
    chosen=min(rows,key=lambda r:(r['normal']['error50']+r['attack']['error50'],
        r['normal']['error30']+r['attack']['error30'],-r['normal']['good20'],-r['attack']['good20']))
    save(out/'separation_development.json',rows)
    save(out/'final_policy.json',dict(policy=json.loads((out/'selected.json').read_text())['policy'],
        separation_m=chosen['limit_m'],minimum_stars=6,development_results=chosen,
        scope='Additional development comparison; no held-out data used. Separation is not authentication.'))
    print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
