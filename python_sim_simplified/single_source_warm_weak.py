"""Supplement: independent clean-start then low-C/N0 attack, no model tuning."""
import sys
from dataclasses import asdict
from datetime import datetime,timezone
import numpy as np
from single_source_twenty import ROOT,save,read,digest,evaluate
from continuous_improvement import Case,generate
from signal_recovery import FS


def inputs():
    seeds=[114101,114113,114137]
    save(ROOT/'warm_weak_freeze.json',dict(utc=datetime.now(timezone.utc).isoformat(),seeds=seeds,
         source_sha256=digest(ROOT/'selected.py'),duration=2.,subsets=['two','three','four'],
         cn0_before=45.,cn0_after=35.,change_time=.3,
         reason='Separate failed low-SNR startup from warm-start low-SNR attack; no receiver changes'))
    for seed in seeds:
        folder=ROOT/'inputs'/f'{seed}_warm_weak_2.0';folder.mkdir(parents=True)
        case=Case(duration=2.,cn0=45.,phase_deg=10.,amplitude_db=1.,velocity_east=2.,attack_start=.3)
        x,a,codes,times,satellites,radii=generate(case,seed,True)
        begin=round(.3*FS);shape=(4,x.shape[1]-begin);rng=np.random.default_rng(seed+10000000)
        extra=np.sqrt(FS*(10**(-35/10)-10**(-45/10))/2)
        x[:,begin:]+=extra*(rng.normal(size=shape)+1j*rng.normal(size=shape))
        np.save(folder/'samples.npy',x)
        np.savez(folder/'observations.npz',nominal=a,codes=np.array(codes),times=times,satellites=satellites,
                 coarse=np.round((radii+75000)/10000)*10000)
        save(folder/'metadata.json',dict(seed=seed,kind='warm_weak',case=asdict(case),attacked=True,
             samples_sha256=digest(folder/'samples.npy'),observations_sha256=digest(folder/'observations.npz'),
             physical_spoof_emitters=1,cn0_after_db_hz=35.,noise_change_time_s=.3,
             extra_noise_seed=seed+10000000))
        print(folder,flush=True)


def validate(subset):
    plan=read(ROOT/'warm_weak_freeze.json');assert digest(ROOT/'selected.py')==plan['source_sha256']
    rows=[]
    for seed in plan['seeds']:
        score,identity=evaluate(ROOT/'selected.py',ROOT/'inputs'/f'{seed}_warm_weak_2.0',subset)
        rows.append(dict(case=dict(seed=seed,kind='warm_weak',duration=2.),subset=subset,score=score,identity=identity))
        save(ROOT/f'warm_weak_{subset}.json',rows)
        print(seed,subset,{k:score[k] for k in ('solved','good20','error30','error50','initialized_before_attack')},flush=True)


if __name__=='__main__':
    if sys.argv[1]=='inputs':inputs()
    else:validate(sys.argv[1])
