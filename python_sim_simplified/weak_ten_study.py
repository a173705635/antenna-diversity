"""Sequential warm-start weak-signal research; immutable model/RF identities."""
import sys,hashlib,shutil
from pathlib import Path
from dataclasses import asdict
from datetime import datetime,timezone
from types import FunctionType
import numpy as np
import single_source_twenty as campaign
from single_source_twenty import save,read,digest,load,input_record
from satellite_count_generator import generate
from continuous_improvement import Case
from signal_recovery import FS,sky

ROOT=Path('results/weak-ten')
PRIOR=Path('results/spatial-ten')
OLD=Path('results/single-source-twenty')


def propose(number,problem,hypothesis):
    folder=ROOT/f'round{number:02d}';folder.mkdir(parents=True)
    save(folder/'proposal.json',dict(number=number,problem=problem,hypothesis=hypothesis,utc=datetime.now(timezone.utc).isoformat()))


def finish(number,decision,evidence,next_problem):
    save(ROOT/f'round{number:02d}/conclusion.json',dict(decision=decision,evidence=evidence,next_problem=next_problem))


def create(seed,kind,cn0=35.,duration=2.,stars=24):
    folder=ROOT/'inputs'/f'{seed}_{kind}_{cn0:g}_{stars}_{duration:g}';folder.mkdir(parents=True)
    case=Case(stars=stars,duration=duration,cn0=45.,phase_deg=10.,amplitude_db=1.,velocity_east=2.,attack_start=.3,
              genuine=kind!='absent',angle=0. if kind=='cluster' else 60.)
    if kind=='drag':case.fake_east=0.;case.fake_rate=50.
    if kind=='release':case.attack_end=.8
    generator=generate
    if kind=='cluster':
        def clustered(seed,count):
            u=sky(seed,count);tangent=np.cross(u[0],[0.,0.,1.]);tangent/=np.linalg.norm(tangent)
            for i,angle in ((1,.5),(2,-.5)):u[i]=u[0]*np.cos(np.radians(angle))+tangent*np.sin(np.radians(angle))
            return u
        generator=FunctionType(generate.__code__,dict(generate.__globals__,sky=clustered))
    attacked=kind!='control'
    x,a,codes,times,satellites,radii=generator(case,seed,attacked)
    begin=round(.3*FS);shape=(4,x.shape[1]-begin);rng=np.random.default_rng(seed+10000000)
    extra=np.sqrt(FS*(10**(-cn0/10)-10**(-45/10))/2)
    x[:,begin:]+=extra*(rng.normal(size=shape)+1j*rng.normal(size=shape))
    if kind=='burst':
        left,right=round(.6*FS),round(.8*FS);shape=(4,right-left)
        burst_rng=np.random.default_rng(seed+20000000)
        scale=np.sqrt(FS*(10**(-20/10)-10**(-cn0/10))/2)
        x[:,left:right]+=scale*(burst_rng.normal(size=shape)+1j*burst_rng.normal(size=shape))
    np.save(folder/'samples.npy',x)
    np.savez(folder/'observations.npz',nominal=a,codes=np.array(codes),times=times,satellites=satellites,
             coarse=np.round((radii+75000)/10000)*10000)
    save(folder/'metadata.json',dict(seed=seed,kind=kind,case=asdict(case),attacked=attacked,cn0_after_db_hz=cn0,
         noise_change_time_s=.3,extra_noise_seed=seed+10000000,physical_spoof_emitters=int(attacked),
         samples_sha256=digest(folder/'samples.npy'),observations_sha256=digest(folder/'observations.npz'),
         burst=dict(start=.6,end=.8,cn0=20.,seed=seed+20000000) if kind=='burst' else None))
    print('created',folder,flush=True)
    return folder


def evaluate(model,folder,subset):
    meta=read(folder/'metadata.json');source=digest(model)
    identity=hashlib.sha256((source+meta['samples_sha256']+meta['observations_sha256']+subset).encode()).hexdigest()
    cached=PRIOR/'evaluations'/identity/'score.json'
    if cached.is_file():return read(cached),identity,str(PRIOR)
    campaign.ROOT=ROOT
    score,identity=campaign.evaluate(model,folder,subset)
    return score,identity,str(ROOT)


def run(number,model,folder,subset,parent=None):
    row=dict(input=str(folder),subset=subset)
    for label,path in [('before',ROOT/'baseline.py'),('after',Path(model))]+([('parent',Path(parent))] if parent else []):
        score,identity,base=evaluate(path,folder,subset)
        row[label]=score;row[label+'_id']=identity;row[label+'_root']=base
        print(number,folder.name,subset,label,{k:score[k] for k in ('solved','good20','error30','error50')},flush=True)
    save(ROOT/f'round{number:02d}'/f'{folder.name}_{subset}.json',row)


if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='create':create(int(sys.argv[2]),sys.argv[3],float(sys.argv[4]),float(sys.argv[5]))
    elif mode=='run':run(int(sys.argv[2]),sys.argv[3],Path(sys.argv[4]),sys.argv[5],sys.argv[6] if len(sys.argv)>6 else None)
