"""Frozen, unselected holdouts; validation runs do not increase iteration count."""
import ast,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from research_hundred import ROOT,load
from audit_recovery_fifty import save
from audit_recovery_two_hundred import score


RF_CASES=[(43001,'normal_low',dict(cn0=35.),False),
          (43002,'cold_low',dict(cn0=35.,attack_start=0.),True),
          (43003,'phase_cold',dict(phase_deg=20.,attack_start=0.),True),
          (43004,'absent_cold',dict(genuine=False,attack_start=0.),True),
          (43005,'two',dict(antennas=2),True),
          (43006,'moving_2s',dict(duration=2.,velocity_east=2.),True),
          (43007,'cold_2s',dict(duration=2.,phase_deg=20.,attack_start=0.),True),
          (43008,'normal',{},False)]


def source_closure(names):
    found={};pending=[Path(name) for name in names]
    while pending:
        p=pending.pop()
        if str(p) in found:continue
        found[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        for node in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            modules=([node.module] if isinstance(node,ast.ImportFrom) and node.module else
                     [a.name for a in node.names] if isinstance(node,ast.Import) else [])
            pending.extend(q for name in modules if (q:=Path(name.split('.')[0]+'.py')).is_file())
    return found


def freeze():
    save(ROOT/'rf_freeze.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),
        sources=source_closure(['audit_array_navigation.py','navigation_hundred.py',
            str(ROOT/'baseline_navigation.py'),str(ROOT/'iteration036/audit_array_navigation.py')]),
        cases=RF_CASES,meaning='Declared RF holdouts frozen before first execution; no threshold selection on results.'))


def rf():
    from continuous_improvement import Case,experiment
    from audit_array_navigation import receiver
    from navigation_hundred import candidate
    frozen=json.loads((ROOT/'rf_freeze.json').read_text())
    for name,digest in frozen['sources'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    baseline_nav=load(ROOT/'baseline_navigation.py').candidate
    captured=load(ROOT/'iteration036/audit_array_navigation.py').receiver
    folder=ROOT/'rf_validation';folder.mkdir();rows=[]
    for seed,kind,changes,attacked in RF_CASES:
        case=Case(**(dict(duration=.6)|changes))
        raw,metadata=experiment(case,seed,attacked,['raw','calibrated'],extended=True)
        array,_=captured(case,seed,attacked);joint,joint_meta=receiver(case,seed,attacked)
        positions=dict(baseline=baseline_nav(raw),navigation=candidate(raw),capture=candidate(array),joint=candidate(joint))
        if not joint['joint_usable']:positions['joint'][:]=np.nan
        for name,z in (('baseline',raw),('capture',array),('joint',joint)):
            np.savez_compressed(folder/f'{seed}_{name}.npz',**z)
        np.savez_compressed(folder/f'{seed}_positions.npz',**positions)
        row=dict(seed=seed,kind=kind,metadata=metadata,joint_calibration=joint_meta['joint_calibration'],
            scores={name:score(position,raw['truth_position']) for name,position in positions.items()})
        rows.append(row);save(folder/f'{seed}.json',row);print(json.dumps(row),flush=True)
    save(ROOT/'rf_validation.json',dict(rows=rows,freeze=frozen))


def lag():
    from audit_lag_support import KINDS,evaluate
    frozen=dict(time_utc=datetime.now(timezone.utc).isoformat(),seeds=[32051,32052],
        sources=source_closure(['lag_source_support.py','audit_lag_support.py']))
    save(ROOT/'lag_freeze.json',frozen)
    rows=[]
    for kind in KINDS:
        rows.extend(evaluate(seed,kind) for seed in frozen['seeds'])
        print('lag holdout',kind,flush=True)
    save(ROOT/'lag_validation.json',dict(rows=rows,freeze=frozen))


if __name__=='__main__':globals()[sys.argv[1]]()
