"""Paired additional development method; preserve the original four methods."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from continuous_improvement import Case,experiment
from audit_recovery_two_hundred import score,SOURCES
from audit_recovery_fifty import save


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--rounds',type=int,nargs='+',required=True)
    args=p.parse_args()
    for number in args.rounds:
        root=args.output/f'round{number:03d}';f=json.loads((root/'frozen.json').read_text());case=Case(**f['case'])
        rows=json.loads((root/'rows.json').read_text());summary=json.loads((root/'summary.json').read_text())
        save(root/'calibration_frozen.json',dict(method='calibrated',same_inputs=True,
            source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in SOURCES}))
        for row in rows:
            z,metadata=experiment(case,row['seed'],row['attack'],['calibrated'])
            path=root/f"{row['seed']}_{int(row['attack'])}.npz"
            with np.load(path) as old:combined={k:old[k] for k in old.files}
            for key in ('time_s','satellite_positions','truth_position','raw_ratio','startup_prns'):
                np.testing.assert_array_equal(combined[key],z[key])
            combined.update({k:v for k,v in z.items() if k.startswith('calibrated_') or k=='startup_phase_estimate'})
            np.savez_compressed(path,**combined)
            row['calibrated']=score(z['calibrated_position'],z['truth_position'])
        for attack in (False,True):
            values=[r['calibrated'] for r in rows if r['attack']==attack]
            summary['calibrated'+('_attack' if attack else '_normal')]={k:sum(v[k] for v in values)
                for k in ('opportunities','solved','good20','error30','error50')}
        save(root/'rows.json',rows);save(root/'summary.json',summary)
        print('calibration comparison '+str(number),flush=True)


if __name__=='__main__':main()
