"""Rounds 51-100: admission, observation stress, and independent validation."""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import numpy as np

from audit_recovery_fifty import Case,run_case,save
from recovery_admission import Policy,admit
from run_gps_spp import receiver_reference


def metrics(p,truth,mask):
    error=np.linalg.norm((p-truth)[:,:2],axis=1)[mask];valid=np.isfinite(error)
    return dict(opportunities=len(error),solved=int(valid.sum()),good20=int((error<=20).sum()),
        error30=int((error>=30).sum()),error50=int((error>=50).sum()),
        p95_m=float(np.quantile(error[valid],.95)) if valid.any() else None,
        maximum_m=float(error[valid].max()) if valid.any() else None)


def prepare(out,name,case,seeds):
    directory=out/'inputs'/name;directory.mkdir(parents=True,exist_ok=False)
    save(directory/'frozen.json',dict(case=asdict(case),seeds=seeds))
    for seed in seeds:
        for attacked in (False,True):
            _,z,_=run_case(case,seed,attacked)
            np.savez_compressed(directory/f'{seed}_{int(attacked)}.npz',**z)
    print('prepared '+name,flush=True)


def policies():
    base=Policy(confirmations=1,use_continuity=False)
    return [replace(base,residual_m=15,exclusions=0,minimum_stars=4,geometry_limit=1000),
            replace(base,residual_m=10,exclusions=0,minimum_stars=4,geometry_limit=1000),
            replace(base,residual_m=15,exclusions=1),replace(base,residual_m=15),
            base,replace(base,residual_m=5),replace(base,use_continuity=True),
            replace(base,use_continuity=True,range_rate_mps=10),
            replace(base,use_continuity=True,confirmations=2),
            replace(base,use_continuity=True,confirmations=3)]


def evaluate(out,number,label,inputs,policy,perturb='none',separation_m=None):
    directory=out/f'round{number:03d}';directory.mkdir(exist_ok=False)
    save(directory/'frozen.json',dict(round=number,label=label,inputs=inputs,policy=asdict(policy),perturb=perturb,separation_m=separation_m,
        source_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in ('recovery_admission.py','audit_recovery_hundred.py')}))
    rows=[]
    for name in inputs:
        for path in sorted((out/'inputs'/name).glob('*.npz')):
            seed,flag=map(int,path.stem.split('_'));attack=bool(flag)
            z=np.load(path);t=z['time_s'];sat=z['satellites'];truth=z['truth_position']
            ranges=z['refined_ranges'].copy();valid=z['refined_valid'].copy()
            active=t>=10
            if attack:
                if perturb=='single_jump':ranges[active,0]+=22000
                if perturb=='double_bias':ranges[active,:2]+=np.array([200.,-200.])
                if perturb=='clock':ranges[active]+=1000
                if perturb in ('outage','outage_forged'):valid[(t>=20)&(t<40)]=False
                if perturb=='period':ranges[active,0]+=299792.458
                if perturb in ('coherent_slow','coherent_fast','outage_forged','unearned_start'):
                    reference,rotation=receiver_reference()
                    delta=.1*np.maximum(t-10,0) if perturb=='coherent_slow' else 5*np.maximum(t-10,0)
                    if perturb in ('outage_forged','unearned_start'):delta=np.full(len(t),200.)
                    false=truth.copy();false[:,0]+=delta
                    forged=np.linalg.norm(sat-(reference+false@rotation)[:,None,:],axis=2)+75000
                    selection=t>=40 if perturb=='outage_forged' else active
                    ranges[selection]=forged[selection]
                if perturb in ('cold_start','unearned_start'):
                    t=t[active];sat=sat[active];truth=truth[active];ranges=ranges[active];valid=valid[active]
                    active=np.ones(len(t),bool)
            current=replace(policy,admitted_start=False) if attack and perturb=='cold_start' else policy
            result=admit(t,sat,ranges,valid,current)
            residual_only=result['position'].copy()
            if separation_m is not None:
                from solution_separation import separation_scores
                separation=separation_scores(sat,ranges,result['selected'],result['position'])
                result['position'][separation>separation_m]=np.nan
            # Baseline is the same raw code observations, including stress injections.
            from signal_recovery import position_solution
            raw=np.array([position_solution(s,r,v)[0] for s,r,v in zip(sat,ranges,valid)])
            row=dict(input=name,seed=seed,attacked=attack,candidate=metrics(result['position'],truth,active),
                     baseline=metrics(raw,truth,active),residual_only=metrics(residual_only,truth,active))
            rows.append(row)
            np.savez_compressed(directory/f'{name}_{seed}_{flag}.npz',time_s=t,truth_position=truth,
                candidate_position=result['position'],baseline_position=raw,selected=result['selected'],residual=result['residual'])
            if separation_m is not None:
                np.savez_compressed(directory/f'{name}_{seed}_{flag}_separation.npz',score=separation,position=residual_only)
    totals={}
    for group in (False,True):
        for method in ('candidate','baseline','residual_only'):
            selected=[r[method] for r in rows if r['attacked']==group]
            totals[method+('_attack' if group else '_normal')]={k:sum(r[k] for r in selected)
                for k in ('opportunities','solved','good20','error30','error50')}
    save(directory/'rows.json',rows);save(directory/'summary.json',dict(round=number,label=label,**totals))
    print(json.dumps(dict(round=number,label=label,**totals)),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',type=int,required=True);args=p.parse_args();out=args.output
    out.mkdir(exist_ok=True)
    if args.phase==1:
        for old in (42,44,46):
            c=Case(**json.loads((Path('results/recovery-fifty')/f'round{old}'/'frozen.json').read_text())['case'])
            prepare(out,f'dev{old}',c,[830003,830004])
        for i,policy in enumerate(policies()):evaluate(out,51+i,f'admission_{i+1}',['dev42','dev44','dev46'],policy)
        ranking=[]
        for i in range(51,61):
            r=json.loads((out/f'round{i:03d}'/'summary.json').read_text())
            a=r['candidate_attack'];h=r['candidate_normal']
            ranking.append(((a['error50'],a['error30'],-h['good20'],-a['good20']),i))
        chosen=min(ranking)[1]
        save(out/'selected.json',dict(round=chosen,policy=json.loads((out/f'round{chosen:03d}'/'frozen.json').read_text())['policy'],
             selection_rule='min attack error50, error30; max normal good20, attack good20',development_seeds=[830003,830004]))
    else:
        policy=Policy(**json.loads((out/'selected.json').read_text())['policy'])
        base=Case(method='refined',duration=120,step=1)
        if args.phase==2:
            changes=[dict(antennas=2),dict(antennas=3),dict(satellites=8),dict(satellites=12),dict(satellites=16),
                     dict(satellites=24),dict(cn0=30),dict(genuine=False),dict(clustered=True),dict(spoof_fraction=.25)]
            for i,change in enumerate(changes):
                name=f'scale{i}';prepare(out,name,replace(base,**change),[860001,860002])
                evaluate(out,61+i,name,[name],policy)
        if args.phase==3:
            prepare(out,'measurement_stress',base,[860001,860002])
            for i,perturb in enumerate(('single_jump','double_bias','coherent_slow','coherent_fast','clock',
                                       'outage','outage_forged','period','cold_start','unearned_start')):
                evaluate(out,71+i,perturb,['measurement_stress'],policy,perturb)
        if args.phase==5:
            final=json.loads((out/'final_policy.json').read_text());policy=Policy(**final['policy'])
            save(out/'holdout_frozen.json',dict(policy=asdict(policy),separation_m=final['separation_m'],seeds=[880001,880002,880003]))
            changes=[dict(satellites=8),dict(satellites=12),dict(satellites=16),dict(satellites=24),
                dict(motion=False,duration=1800,step=5),
                dict(phase_deg=10.,amplitude_db=2.,geometry_mm=3.,yaw_error_deg=10.,phase_drift_deg_s=.3,multipath=.3,clip=5.),
                dict(genuine=False),dict(angle=0),dict(cn0=35),dict(attack_end=150)]
            for i,change in enumerate(changes):
                name=f'holdout{i}';prepare(out,name,replace(base,duration=300,step=2,**change) if i!=4
                    else replace(base,**change),[880001,880002,880003])
                evaluate(out,91+i,name,[name],policy,separation_m=final['separation_m'])


if __name__=='__main__':main()
