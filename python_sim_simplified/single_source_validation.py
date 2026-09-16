"""Frozen holdout creation/evaluation. Never selects or tunes a receiver."""
import sys,shutil
from pathlib import Path
from datetime import datetime,timezone
from single_source_twenty import ROOT,save,read,digest,create_input,evaluate


def freeze():
    cases=[dict(seed=seed,kind=kind,duration=2.) for seed in (114001,114013,114037)
           for kind in ('normal','drift','weak','absent','return','near','cluster','drag','release')]
    cases += [dict(seed=114037,kind=kind,duration=4.) for kind in ('drift','weak','return')]
    shutil.copy2(ROOT/'round020/receiver.py',ROOT/'selected.py')
    dependencies={str(p):digest(p) for p in Path('.').glob('*.py')}
    save(ROOT/'freeze.json',dict(utc=datetime.now(timezone.utc).isoformat(),selected_round=20,
         selected_sha256=digest(ROOT/'selected.py'),dependencies=dependencies,cases=cases,
         subsets=['two','three','four'],no_tuning_on_holdout=True,
         baseline_comparison=dict(seed=114001,kinds=['normal','drift','weak'],duration=2.),
         cluster_geometry='three satellites at source direction and +/-0.5 degrees'))


def inputs():
    for case in read(ROOT/'freeze.json')['cases']:
        folder=create_input(**case)
        print(str(folder),flush=True)
    save(ROOT/'inputs_complete.json',dict(count=len(read(ROOT/'freeze.json')['cases'])))


def validate(subset):
    plan=read(ROOT/'freeze.json');assert digest(ROOT/'selected.py')==plan['selected_sha256']
    rows=[]
    for case in plan['cases']:
        folder=ROOT/'inputs'/f"{case['seed']}_{case['kind']}_{case['duration']}"
        score,identity=evaluate(ROOT/'selected.py',folder,subset)
        row=dict(case=case,subset=subset,score=score,identity=identity);rows.append(row)
        save(ROOT/f'holdout_{subset}.json',rows)
        print(case,subset,{k:score[k] for k in ('solved','good20','error30','error50','initialized_before_attack')},flush=True)
    comparison=[]
    for kind in plan['baseline_comparison']['kinds']:
        folder=ROOT/'inputs'/f'114001_{kind}_2.0'
        score,identity=evaluate(ROOT/'baseline.py',folder,subset)
        comparison.append(dict(kind=kind,subset=subset,score=score,identity=identity))
        save(ROOT/f'holdout_baseline_{subset}.json',comparison)
        print('baseline',kind,subset,{k:score[k] for k in ('solved','good20','error30','error50')},flush=True)


if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    elif sys.argv[1]=='inputs':inputs()
    else:validate(sys.argv[1])
