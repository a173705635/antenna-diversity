"""Rounds 81-90: continuous multi-Doppler and unknown-data-bit stress."""
import argparse
from dataclasses import asdict,replace
import hashlib,json
from pathlib import Path
import numpy as np

from audit_recovery_fifty import save
from audit_recovery_hundred import metrics
from doppler_recovery import ContinuousCase,experiment
from recovery_admission import Policy


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rounds',type=int,nargs='+');p.add_argument('--seeds',type=int,nargs='+',default=[870001,870002])
    args=p.parse_args();out=args.output
    policy=Policy(**json.loads((out/'selected.json').read_text())['policy'])
    changes=[dict(doppler_scale=0.),dict(),dict(doppler_scale=.2),dict(doppler_scale=1.5),
             dict(data_bits=True),dict(cn0=35.),dict(doppler_rate=100.),dict(phase_deg=10.),
             dict(angle=0.),dict(genuine=False)]
    for i,change in enumerate(changes):
        number=81+i
        if args.rounds and number not in args.rounds:continue
        case=replace(ContinuousCase(),**change)
        directory=out/f'round{number:03d}';directory.mkdir(exist_ok=False)
        save(directory/'frozen.json',dict(round=number,case=asdict(case),policy=asdict(policy),seeds=args.seeds,
            source_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in
                           ('doppler_recovery.py','audit_doppler_recovery.py','recovery_admission.py')}))
        rows=[]
        for seed in args.seeds:
            for attack in (False,True):
                z,record=experiment(case,seed,attack,policy)
                np.savez_compressed(directory/f'{seed}_{int(attack)}.npz',**z)
                save(directory/f'{seed}_{int(attack)}_tracking.json',record)
                methods=('single','raw_covariance','fold','despread','admitted')
                rows.append(dict(seed=seed,attacked=attack,**{m:metrics(z[m+'_position'],z['truth_position'],
                    np.ones(len(z['time_s']),bool)) for m in methods}))
        totals={}
        for attack in (False,True):
            for method in methods:
                selected=[r[method] for r in rows if r['attacked']==attack]
                totals[method+('_attack' if attack else '_normal')]={k:sum(r[k] for r in selected)
                    for k in ('opportunities','solved','good20','error30','error50')}
        save(directory/'rows.json',rows);save(directory/'summary.json',dict(round=number,label=str(change),**totals))
        print(json.dumps(dict(round=number,**totals)),flush=True)


if __name__=='__main__':main()
