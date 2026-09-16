"""Rounds 201-400: fresh IQ, causal carrier/spatial admission, frozen validation."""
import argparse,hashlib,json,time
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
from continuous_improvement import Case,experiment
from continuous_admission import Gate,admit
from carrier_admission import Rule,evaluate,smooth,rule_from_record
from audit_recovery_two_hundred import score,cases,SOURCES
from audit_recovery_fifty import save

ROOT=Path('results/recovery-four-hundred')
FILES=(*SOURCES,'carrier_admission.py','audit_recovery_four_hundred.py','RECOVERY_FOUR_HUNDRED_PLAN.md')
KEYS=('opportunities','solved','good20','error30','error50')


def hashes():
    return {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in FILES}


def configurations():
    changes=cases()
    # Replace SIC-only variations with hypotheses relevant to this campaign.
    changes[20]=dict(velocity_east=2.)
    changes[21]=dict(velocity_east=15.)
    changes[25]=dict(fake_east=5.,fake_rate=20.)
    changes[26]=dict(fake_east=0.,fake_rate=50.)
    changes[29]=dict(attack_start=0.)
    return changes


def boundaries():
    return [dict(duration=d,**change) for d in (2.,4.) for change in (
        {},dict(genuine=False),dict(attack_start=0.),dict(phase_deg=10),dict(cn0=35),
        dict(angle=0),dict(antennas=2),dict(stars=16),dict(velocity_east=15.),
        dict(fake_east=0.,fake_rate=20.))]+[
        dict(duration=8.),dict(duration=8.,velocity_east=2.),
        dict(duration=8.,fake_east=0.,fake_rate=10.),dict(duration=8.,genuine=False),
        dict(duration=2.,attack_end=1.),dict(duration=2.,phase_deg=20,amplitude_db=3),
        dict(duration=2.,antennas=3),dict(duration=2.,stars=12),
        dict(duration=2.,cn0=30),dict(duration=2.,angle=2),
        dict(duration=2.,angle=10),dict(duration=2.,fake_east=10000),
        dict(duration=2.,fake_east=20),dict(duration=2.,fake_east=5),
        dict(duration=2.,doppler_rate=100),dict(duration=2.,doppler_scale=2.),
        dict(duration=2.,spoof_db=0),dict(duration=2.,spoof_db=20),
        dict(duration=2.,velocity_east=30.),dict(duration=2.,attack_start=0.,genuine=False)]


def rules():
    return [Rule(window=w,spatial=s,mature=m,divergence=d,consensus=c,weighted=weight)
        for w in (1,5,20,50,100) for s in (-np.inf,1.,3.)
        for m,d,c,weight in ((1,np.inf,np.inf,True),(3,50.,np.inf,True),
                             (3,20.,20.,True),(3,50.,20.,False))]


def totals(rows,methods):
    return {m+('_attack' if flag else '_normal'):
        {k:sum(r[m][k] for r in rows if r['attack']==flag) for k in KEYS}
        for flag in (False,True) for m in methods}


def old_policies(z):
    for name,method,gate in (('previous','raw',Gate(min_lock=.8)),
                             ('alternate','calibrated',Gate(min_gain=.01))):
        result=admit(z['satellite_positions'],z[method+'_ranges'],z[method+'_lock'],
                     z[method+'_cn0'],z[method+'_gain'],gate)
        z[name+'_position']=result['position']


def rf_round(number):
    root=ROOT/f'round{number:03d}';root.mkdir(exist_ok=False)
    dev=number<241
    change=configurations()[number-201 if dev else number-321] if number<361 else boundaries()[number-361]
    case=replace(Case(duration=.8 if dev else 1.),**change)
    selected=None if dev else json.loads((ROOT/'selected.json').read_text())
    seed=1100000+number
    save(root/'frozen.json',dict(round=number,case=asdict(case),seed=seed,selected=selected,sha256=hashes()))
    rows=[];start=time.time()
    for attack in (False,True):
        z,record=experiment(case,seed,attack,['raw','calibrated'],extended=True)
        old_policies(z);methods=['raw','calibrated','previous','alternate']
        if selected:
            result=evaluate(z,selected['method'],rule_from_record(selected['rule']))
            z.update({'candidate_'+k:v for k,v in result.items()});methods+=['candidate']
        key=f'{seed}_{int(attack)}'
        np.savez_compressed(root/(key+'.npz'),**z);save(root/(key+'.json'),record)
        rows.append(dict(attack=attack,seed=seed,**{m:score(z[m+'_position'],z['truth_position']) for m in methods}))
        save(root/'rows.json',rows)
    summary=dict(round=number,seconds=time.time()-start,**totals(rows,methods))
    save(root/'summary.json',summary);print(json.dumps(summary),flush=True)


def development(numbers):
    records=[]
    for n in range(201,241):
        for p in sorted((ROOT/f'round{n:03d}').glob('*.npz')):
            with np.load(p) as archive:records.append((p,{k:archive[k] for k in archive.files}))
    assert len(records)==80
    for number in numbers:
        rule=rules()[number-241];root=ROOT/f'round{number:03d}';root.mkdir(exist_ok=False)
        save(root/'frozen.json',dict(round=number,rule=asdict(rule),sha256=hashes(),
            inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p,z in records}))
        rows=[]
        for path,z in records:
            row=dict(input=str(path),attack=path.stem.endswith('_1'))
            for method in ('raw','calibrated'):
                result=evaluate(z,method,rule);row[method]=score(result['position'],z['truth_position'])
            rows.append(row)
        summary=dict(round=number,rule=asdict(rule),**totals(rows,['raw','calibrated']))
        save(root/'rows.json',rows);save(root/'summary.json',summary)
        print(json.dumps(summary),flush=True)


def select():
    choices=[]
    for number in range(241,301):
        v=json.loads((ROOT/f'round{number:03d}'/'summary.json').read_text())
        for m in ('raw','calibrated'):
            choices.append(dict(round=number,method=m,rule=v['rule'],normal=v[m+'_normal'],attack=v[m+'_attack']))
    feasible=[v for v in choices if v['normal']['solved']>=.95*v['normal']['opportunities']
        and v['attack']['solved']>=.5*v['attack']['opportunities']]
    chosen=min(feasible,key=lambda v:(v['normal']['error30']+v['attack']['error30'],
        v['normal']['error50']+v['attack']['error50'],-v['attack']['good20'],-v['normal']['good20']))
    save(ROOT/'development_selection.json',choices)
    save(ROOT/'selected.json',dict(**chosen,feasible=len(feasible),sha256=hashes()))
    print(json.dumps(chosen),flush=True)


def falsification(number):
    root=ROOT/f'round{number:03d}';root.mkdir(exist_ok=False)
    index=number-301;kind=index%10;seed=1200000+number
    rng=np.random.default_rng(seed);t=np.arange(1000)*.01;s=8
    truth=2e7+np.arange(s)*1000+3*t[:,None]
    code=truth+rng.normal(0,3,truth.shape);carrier=truth+np.arange(s)*789.
    mask=np.ones_like(code,bool);active=t>=3.
    names=['nominal','common_clock','receiver_motion','coherent_false_step','coherent_slow_pull',
           'code_only_step','carrier_cycle_slip','outage','missing_time','constant_code_bias']
    if kind==1:code[active]+=500.;carrier[active]+=500.;truth[active]+=500.
    if kind==2:
        movement=15*np.sin(t[:,None])*np.linspace(-1,1,s)
        truth+=movement;code+=movement;carrier+=movement
    if kind==3:code[active]+=100.;carrier[active]+=100.
    if kind==4:
        drag=np.maximum(t[:,None]-3,0)*10
        code+=drag;carrier+=drag
    if kind==5:code[active,0]+=100.
    if kind==6:carrier[active,0]+=.1903*50
    if kind==7:mask[(t>=3)&(t<5)]=False
    if kind==8:t[500:]+=.5
    if kind==9:code+=100.
    rule=Rule(window=20 if index<10 else 100,mature=3)
    result,ages=smooth(t,code,carrier,mask,rule)
    prefix,_=smooth(t[:600],code[:600],carrier[:600],mask[:600],rule)
    np.testing.assert_array_equal(prefix,result[:600])
    np.savez_compressed(root/'observations.npz',time=t,code=code,carrier=carrier,eligible=mask,
        truth=truth,result=result,ages=ages)
    error=result-truth;valid=np.isfinite(error)
    save(root/'frozen.json',dict(round=number,name=names[kind],seed=seed,rule=asdict(rule),sha256=hashes()))
    save(root/'summary.json',dict(round=number,name=names[kind],domain='pseudorange, not horizontal position',
        rms_m=float(np.sqrt(np.mean(error[valid]**2))),max_m=float(np.max(abs(error[valid]))),
        error30=int((abs(error)>=30).sum()),opportunities=error.size,solved=int(valid.sum()),prefix_equal=True))
    print('falsification '+str(number)+' '+names[kind],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--rounds',type=int,nargs='+');p.add_argument('--select',action='store_true')
    args=p.parse_args();ROOT.mkdir(exist_ok=True)
    if args.select:select()
    elif args.rounds[0] in range(241,301):development(args.rounds)
    else:
        for n in args.rounds:
            if n in range(301,321):falsification(n)
            else:rf_round(n)


if __name__=='__main__':main()
