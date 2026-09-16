"""Supplementary independent continuous holdout after both candidates are frozen."""
from dataclasses import asdict
import hashlib,json
from pathlib import Path
import numpy as np
from audit_recovery_fifty import save
from audit_recovery_hundred import metrics
from doppler_recovery import ContinuousCase,experiment
from recovery_admission import Policy
from solution_separation import separation_scores


def main():
    out=Path('results/recovery-hundred');final=json.loads((out/'final_policy.json').read_text())
    policy=Policy(**final['policy']);root=out/'continuous-holdout';root.mkdir(exist_ok=False)
    save(root/'frozen.json',dict(final_policy=final,seeds=[890001,890002,890003],
        source_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in
             ('doppler_recovery.py','recovery_admission.py','solution_separation.py','audit_recovery_holdout.py')}))
    rows=[]
    for bits in (False,True):
        case=ContinuousCase(data_bits=bits)
        for seed in (890001,890002,890003):
            for attack in (False,True):
                z,record=experiment(case,seed,attack,policy)
                scores=separation_scores(z['satellite_positions'],z['despread_ranges'],
                                         z['admitted_selected'],z['admitted_position'])
                z['final_position']=z['admitted_position'].copy()
                z['final_position'][scores>final['separation_m']]=np.nan
                z['separation']=scores
                key=f'{int(bits)}_{seed}_{int(attack)}'
                np.savez_compressed(root/f'{key}.npz',**z);save(root/f'{key}_tracking.json',record)
                rows.append(dict(bits=bits,seed=seed,attacked=attack,**{m:metrics(z[m+'_position'],z['truth_position'],
                    np.ones(len(z['time_s']),bool)) for m in ('single','raw_covariance','fold','despread','admitted','final')}))
                save(root/'rows.json',rows);print('continuous holdout '+key,flush=True)
    print('continuous independent holdout complete',flush=True)


if __name__=='__main__':main()
