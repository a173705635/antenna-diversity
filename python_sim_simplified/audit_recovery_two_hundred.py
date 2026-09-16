"""Numbered rounds 101-200; paired continuous IQ, frozen admission holdout."""
import argparse,hashlib,json,time
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
from continuous_improvement import Case,experiment
from continuous_admission import Gate,admit
from audit_recovery_fifty import save
from audit_recovery_hundred import metrics

SOURCES=('continuous_improvement.py','continuous_admission.py','audit_recovery_two_hundred.py',
         'tracking.py','positioning.py','doppler_recovery.py','signal_recovery.py','code_replica.py','acquisition.py')


def score(position,truth):
    result=metrics(position,truth,np.ones(len(truth),bool))
    good=np.isfinite(position).all(axis=1)
    error=np.linalg.norm((position-truth)[:,:2],axis=1)
    result['p95_output_m']=float(np.percentile(error[good],95)) if good.any() else None
    result['max_output_m']=float(np.max(error[good])) if good.any() else None
    runs=np.diff(np.flatnonzero(np.r_[True,good,True]))-1
    result['longest_unavailable_s']=float(np.max(runs)*.01)
    return result


def cases():
    changes=[{},dict(short_ms=2),dict(short_ms=5),dict(short_ms=10),dict(data_bits=False),
       dict(covariance_ms=10),dict(covariance_ms=40),dict(doppler_scale=.2),dict(doppler_scale=1.5),
       dict(doppler_rate=100),dict(cn0=40),dict(cn0=35),dict(phase_deg=5),dict(phase_deg=10),
       dict(amplitude_db=2),dict(angle=15),dict(angle=0),dict(spoof_db=3),dict(spoof_db=20),dict(genuine=False)]
    second=[dict(sic_radius=.25),dict(sic_radius=1),dict(fake_east=20),dict(fake_east=300),dict(fake_east=1000),
       dict(data_bits=False),dict(spoof_db=0),dict(spoof_db=5),dict(spoof_db=15),dict(spoof_db=20),
       dict(cn0=40),dict(cn0=35),dict(phase_deg=3),dict(phase_deg=10),dict(amplitude_db=1),
       dict(angle=5),dict(angle=30),dict(antennas=2),dict(antennas=3),dict(attack_end=.4)]
    return changes+second


def run_round(out,number):
    directory=out/f'round{number:03d}';directory.mkdir(exist_ok=False)
    if number<=140:
        change=cases()[number-101];case=replace(Case(duration=.8),**change)
        methods=['raw','short','mvdr','sic'];seeds=[910000+number];final=None
    else:
        final=json.loads((out/'selected.json').read_text());methods=list(dict.fromkeys(['raw',final['method']]))
        if number<=180:
            change=cases()[number-161];case=replace(Case(),**change);seeds=[920000+number,930000+number]
        else:
            change=[dict(duration=2.),dict(duration=3.),dict(duration=4.,data_bits=False),
                dict(duration=2.,doppler_rate=50),dict(duration=2.,attack_end=1.),
                dict(duration=2.,cn0=35),dict(duration=2.,phase_deg=10),dict(duration=2.,genuine=False),
                dict(duration=2.,angle=0),dict(duration=2.,attack_start=0.),
                dict(antennas=2),dict(antennas=3),dict(stars=12),dict(stars=16),
                dict(cn0=30),dict(angle=2),dict(angle=10),dict(fake_east=10000),
                dict(phase_deg=20,amplitude_db=3),dict(doppler_scale=2.)][number-181]
            case=replace(Case(),**change);seeds=[940000+number,950000+number]
    save(directory/'frozen.json',dict(round=number,case=asdict(case),seeds=seeds,methods=methods,selected=final,
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in SOURCES}))
    rows=[];start=time.time()
    for seed in seeds:
        for attack in (False,True):
            z,record=experiment(case,seed,attack,methods)
            if final:
                method=final['method'];candidate=admit(z['satellite_positions'],z[method+'_ranges'],
                    z[method+'_lock'],z[method+'_cn0'],z[method+'_gain'],Gate(**final['gate']))
                z['candidate_position']=candidate['position'];z['candidate_selected']=candidate['selected']
                z['candidate_scores']=candidate['scores']
            measured=methods+(['candidate'] if final else [])
            row=dict(seed=seed,attack=attack,**{m:score(z[m+'_position'],z['truth_position']) for m in measured})
            key=f'{seed}_{int(attack)}';np.savez_compressed(directory/f'{key}.npz',**z)
            save(directory/f'{key}.json',record);rows.append(row);save(directory/'rows.json',rows)
    totals={}
    for attack in (False,True):
        for method in measured:
            values=[r[method] for r in rows if r['attack']==attack]
            totals[method+('_attack' if attack else '_normal')]={k:sum(v[k] for v in values)
                for k in ('opportunities','solved','good20','error30','error50')}
    save(directory/'summary.json',dict(round=number,case=asdict(case),seconds=time.time()-start,**totals))
    print(json.dumps(dict(round=number,seconds=round(time.time()-start),**totals)),flush=True)


def calibrate(out,numbers):
    records=[]
    for number in range(101,141):
        for path in sorted((out/f'round{number:03d}').glob('*.npz')):
            records.append((path,np.load(path)))
    # Twenty predefined rules, evaluated on all paired development records.
    changes=[{},dict(min_lock=0),dict(min_lock=.8),dict(min_cn0=20),dict(min_cn0=30),
       dict(min_gain=.01),dict(min_gain=.1),dict(sigma45=3),dict(sigma45=10),dict(sigma45=20),
       dict(residual_z=2),dict(residual_z=5),dict(separation_z=3),dict(separation_z=10),
       dict(horizontal_sigma=10),dict(horizontal_sigma=30),dict(horizontal_sigma=100),
       dict(sigma45=10,horizontal_sigma=30),dict(min_lock=0,min_cn0=20,horizontal_sigma=100),
       dict(min_lock=.8,min_cn0=30,min_gain=.1,residual_z=2,separation_z=3)]
    choices=[]
    for i,change in enumerate(changes):
        number=141+i
        if number not in numbers:continue
        directory=out/f'round{number:03d}';directory.mkdir(exist_ok=False)
        gate=replace(Gate(),**change);rows=[]
        for method in ('raw','short','mvdr','sic','calibrated'):
            totals={flag:{k:0 for k in ('opportunities','solved','good20','error30','error50')} for flag in (False,True)}
            for path,z in records:
                result=admit(z['satellite_positions'],z[method+'_ranges'],z[method+'_lock'],
                             z[method+'_cn0'],z[method+'_gain'],gate)
                result_score=score(result['position'],z['truth_position']);flag=path.stem.endswith('_1')
                for k in totals[flag]:totals[flag][k]+=result_score[k]
                rows.append(dict(input=str(path),method=method,attack=flag,**result_score))
            choices.append(dict(round=number,method=method,gate=asdict(gate),normal=totals[False],attack=totals[True]))
        save(directory/'frozen.json',dict(gate=asdict(gate),source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in SOURCES}))
        save(directory/'rows.json',rows);save(directory/'summary.json',choices[-5:])
        print('gate round '+str(number),flush=True)


def select(out):
    choices=[row for number in range(141,161)
             for row in json.loads((out/f'round{number:03d}'/'summary.json').read_text())]
    # Predeclared research compromise, not product acceptance: retain normal >=95%, attack >=50%.
    feasible=[v for v in choices if v['normal']['solved']>=.95*v['normal']['opportunities']
              and v['attack']['solved']>=.5*v['attack']['opportunities']]
    pool=feasible if feasible else choices
    chosen=min(pool,key=lambda v:(v['normal']['error50']+v['attack']['error50'],
        v['normal']['error30']+v['attack']['error30'],-v['normal']['good20'],-v['attack']['good20']))
    chosen['feasible_development_rules']=len(feasible)
    chosen['selection_rule']='Normal availability >=95%, attack >=50% if feasible; then minimize >=50, >=30 errors, maximize good20. No claim of integrity.'
    save(out/'development_selection.json',choices);save(out/'selected.json',chosen)
    print(json.dumps(chosen),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rounds',type=int,nargs='+');p.add_argument('--calibrate',action='store_true')
    p.add_argument('--select',action='store_true')
    args=p.parse_args();args.output.mkdir(exist_ok=True)
    if args.select:select(args.output)
    elif args.calibrate:calibrate(args.output,args.rounds or list(range(141,161)))
    else:
        for number in args.rounds:run_round(args.output,number)


if __name__=='__main__':main()
