"""Post-freeze bug-fix replay; reuses RF unchanged and never claims holdout selection."""
import hashlib,sys,time
from single_source_twenty import ROOT,read,save,digest,evaluate


subset=sys.argv[1];frozen=read(ROOT/'freeze.json');fix=read(ROOT/'post_freeze_correction.json')
cases=frozen['cases'][:]
cases += [dict(seed=s,kind='warm_weak',duration=2.) for s in read(ROOT/'warm_weak_freeze.json')['seeds']]
cases += [dict(seed=s,kind='nominal',duration=2.) for s in read(ROOT/'nominal_freeze.json')['seeds']]
assert digest(ROOT/'audit_corrected.py')==fix['corrected_sha256']
rows=[]
for case in cases:
    folder=ROOT/'inputs'/f"{case['seed']}_{case['kind']}_{case['duration']}";meta=read(folder/'metadata.json')
    original_id=hashlib.sha256((frozen['selected_sha256']+meta['samples_sha256']+meta['observations_sha256']+subset).encode()).hexdigest()
    # Wait for the known ongoing frozen RF job, then reuse its completed observation stream.
    while not (ROOT/'evaluations'/original_id/'score.json').exists():time.sleep(1)
    before,bid=evaluate(ROOT/'selected.py',folder,subset)
    after,aid=evaluate(ROOT/'audit_corrected.py',folder,subset)
    rows.append(dict(case=case,subset=subset,score=after,identity=aid,before=before,before_id=bid))
    save(ROOT/f'corrective_{subset}.json',rows)
    print(case,subset,{k:after[k] for k in ('solved','good20','error30','error50')},flush=True)
