"""Broader old-data regression, not independent RF validation."""
import hashlib,json
from pathlib import Path
import numpy as np
from navigation_candidate import candidate
from carrier_admission import Rule,evaluate
from audit_recovery_two_hundred import score
from audit_recovery_fifty import save

ROOT=Path('results/research-twenty')
rows=[]
folder=ROOT/'old_replay';folder.mkdir()
for number in (*range(201,241),*range(321,401)):
    for path in sorted(Path(f'results/recovery-four-hundred/round{number:03d}').glob('*.npz')):
        with np.load(path) as archive:z={k:archive[k] for k in archive.files}
        p=candidate(z)
        old=evaluate(z,'calibrated',Rule(window=5,mature=1,divergence=np.inf,spatial=1.))['position']
        metrics=score(p,z['truth_position']);e2=np.sum((p[:,:2]-z['truth_position'][:,:2])**2,axis=1)
        assert metrics['good20']==int(np.sum(e2<=400))
        assert metrics['error30']==int(np.sum(e2>=900))
        assert metrics['error50']==int(np.sum(e2>=2500))
        np.savez_compressed(folder/(path.stem+'.npz'),position=p,old=old)
        rows.append(dict(number=number,attack=path.stem.endswith('_1'),path=str(path),
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         old=score(old,z['truth_position']),proposal=metrics))
totals={}
for label,subset in (('all',rows),('previous_holdout',[r for r in rows if r['number']>=321])):
    totals[label]={str(flag):{method:{key:sum(r[method][key] for r in subset if r['attack']==flag)
        for key in ('opportunities','solved','good20','error30','error50')}
        for method in ('old','proposal')} for flag in (False,True)}
save(ROOT/'old_replay.json',dict(rows=rows,totals=totals))
print(json.dumps(totals),flush=True)
