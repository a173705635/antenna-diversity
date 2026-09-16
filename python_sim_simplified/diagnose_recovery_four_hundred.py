"""Revisit three archived failures without treating their seeds as new validation."""
import argparse,json
from pathlib import Path
import numpy as np
from continuous_improvement import Case,experiment
from carrier_admission import rule_from_record,evaluate
from audit_recovery_four_hundred import ROOT,hashes,old_policies
from audit_recovery_two_hundred import score
from audit_recovery_fifty import save


def prepare():
    out=ROOT/'archived-failure-replay';out.mkdir(exist_ok=False)
    for number,seed in ((171,920171),(185,940185),(190,950190)):
        old=Path('results/recovery-two-hundred')/f'round{number:03d}'/f'{seed}_1.npz'
        metadata=json.loads(old.with_suffix('.json').read_text())
        z,record=experiment(Case(**metadata['case']),seed,True,['raw','calibrated'],extended=True)
        with np.load(old) as previous:
            for key in ('raw_ranges','raw_position','raw_lock','raw_cn0','raw_gain','time_s','satellite_positions'):
                np.testing.assert_allclose(z[key],previous[key],rtol=0,atol=1e-8,equal_nan=True)
        old_policies(z);np.savez_compressed(out/f'{number}.npz',**z)
        save(out/f'{number}.json',dict(**record,old_input=str(old),original_raw_reproduced=True,sha256=hashes()))
        print('archived failure reproduced '+str(number),flush=True)


def analyze():
    selected=json.loads((ROOT/'selected.json').read_text());rows=[]
    for path in sorted((ROOT/'archived-failure-replay').glob('*.npz')):
        z=np.load(path);result=evaluate(z,selected['method'],rule_from_record(selected['rule']))
        positions={m:z[m+'_position'] for m in ('raw','previous','alternate')}
        positions['candidate']=result['position'];row=dict(round=int(path.stem))
        row.update({m:score(p,z['truth_position']) for m,p in positions.items()})
        if int(path.stem)==185:
            k=int(np.argmin(abs(z['time_s']-.17)))
            row['previous_108m_epoch']={m:p[k].tolist() for m,p in positions.items()}
            row['ages']=result['ages'][k].tolist();row['scores']=result['scores'][k].tolist()
        rows.append(row)
    save(ROOT/'archived-failure-replay'/'comparison.json',dict(selected=selected,rows=rows))
    print(json.dumps(rows),flush=True)


def calibration_ablation():
    """Post-hoc same-IQ failure attribution; not an additional validation seed."""
    path=ROOT/'round350'/'1100350_1.npz';metadata=json.loads(path.with_suffix('.json').read_text())
    out=ROOT/'cold-calibration-ablation';out.mkdir(exist_ok=False)
    z,record=experiment(Case(**metadata['case']),metadata['seed'],True,['short'],extended=True)
    with np.load(path) as original:
        for key in ('time_s','satellite_positions','truth_position','raw_ratio','startup_prns'):
            np.testing.assert_array_equal(z[key],original[key])
        original_calibrated=score(original['calibrated_position'],original['truth_position'])
        original_candidate=score(original['candidate_position'],original['truth_position'])
    selected=json.loads((ROOT/'selected.json').read_text())
    result=evaluate(z,'short',rule_from_record(selected['rule']))
    z.update({'candidate_'+k:v for k,v in result.items()})
    np.savez_compressed(out/'same_input.npz',**z)
    save(out/'frozen.json',dict(**record,input=str(path),sha256=hashes(),
        scope='Post-hoc attribution only. Short without startup calibration; no holdout retuning.'))
    result=dict(original_calibrated=original_calibrated,original_candidate=original_candidate,
        short=score(z['short_position'],z['truth_position']),
        short_with_frozen_rule=score(z['candidate_position'],z['truth_position']))
    save(out/'comparison.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--calibration-ablation',action='store_true');args=parser.parse_args()
    if args.prepare:prepare()
    elif args.calibration_ablation:calibration_ablation()
    else:analyze()
