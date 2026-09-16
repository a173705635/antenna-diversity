"""Predeclared new 24-PRN RF records, single physical source, trusted startup."""
from dataclasses import asdict
from pathlib import Path
from types import FunctionType
import numpy as np
from satellite_count_generator import generate
from continuous_improvement import Case
from signal_recovery import sky
from spatial_ten_study import ROOT,save,digest


def create(seed,kind,duration):
    folder=ROOT/'inputs'/f'{seed}_{kind}_{duration}';folder.mkdir(parents=True)
    case=Case(stars=24,duration=duration,cn0=45.,phase_deg=10.,amplitude_db=1.,velocity_east=2.,attack_start=.3,
              angle=0. if kind=='cluster' else 60.)
    generator=generate
    if kind=='cluster':
        def clustered(seed,count):
            u=sky(seed,count);tangent=np.cross(u[0],[0.,0.,1.]);tangent/=np.linalg.norm(tangent)
            for i,angle in ((1,.5),(2,-.5)):
                u[i]=u[0]*np.cos(np.radians(angle))+tangent*np.sin(np.radians(angle))
            return u
        generator=FunctionType(generate.__code__,dict(generate.__globals__,sky=clustered))
    x,a,codes,times,satellites,radii=generator(case,seed,kind!='normal')
    np.save(folder/'samples.npy',x)
    np.savez(folder/'observations.npz',nominal=a,codes=np.array(codes),times=times,satellites=satellites,
             coarse=np.round((radii+75000)/10000)*10000)
    save(folder/'metadata.json',dict(seed=seed,kind=kind,case=asdict(case),attacked=kind!='normal',
         samples_sha256=digest(folder/'samples.npy'),observations_sha256=digest(folder/'observations.npz'),
         physical_spoof_emitters=int(kind!='normal'),generator_sha256=digest(Path('satellite_count_generator.py'))))
    print('created',folder,flush=True)


if __name__=='__main__':
    for kind in ('normal','cluster'):create(115901,kind,2.)
    create(115921,'cluster',6.)
