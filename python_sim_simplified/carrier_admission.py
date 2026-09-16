"""Causal code/carrier smoothing and measured spatial evidence; no authentication claim."""
from dataclasses import dataclass
import numpy as np
from code_replica import _sample_code
from continuous_admission import Gate,admit
from signal_recovery import position_solution


def spatial_evidence(x,code,history,times,desired,fs):
    """1 ms signed-bit-insensitive prompt power outside observed common source.

    Uses only samples before each output epoch. A positive score alone cannot
    distinguish genuine code lock from a nearby, correlated counterfeit code.
    """
    starts=history['epoch_start_samples']/fs
    result=np.full(len(times),np.nan);unit=round(fs*.001);m=len(desired)
    for k,t in enumerate(times):
        end=round(t*fs);begin=end-20*unit
        block=x[:,begin:end];cov=block@block.conj().T/block.shape[1]
        eig,vec=np.linalg.eigh(cov);v=vec[:,-1]
        q=desired-v*np.vdot(v,desired) if eig[-1]/eig[-2]>=3 else desired
        sample_times=np.arange(begin,end)/fs
        idx=np.searchsorted(starts,sample_times,side='right')-1
        dt=sample_times-starts[idx]
        phase=history['code_nco_phase_chips'][idx]+history['code_nco_rate_hz'][idx]*dt
        carrier=history['carrier_nco_phase_rad'][idx]+2*np.pi*history['carrier_nco_frequency_hz'][idx]*dt
        local=_sample_code(code,phase)*np.exp(1j*carrier)
        prompt=((q.conj()@block)*local.conj()).reshape(20,unit).mean(axis=1)
        noise=eig[0]*np.vdot(q,q).real*np.mean(abs(local)**2)/unit
        result[k]=np.mean(abs(prompt)**2)/noise-1.
    return result


@dataclass
class Rule:
    window: int=20
    mature: int=3
    divergence: float=50.
    spatial: float=-np.inf
    consensus: float=np.inf
    weighted: bool=True


def rule_from_record(record):
    """The archive writer encodes non-finite disabled thresholds as JSON null."""
    return Rule(**{k:((-np.inf if k=='spatial' else np.inf) if v is None else v)
                   for k,v in record.items()})


def smooth(time,ranges,carrier,eligible,rule):
    """Hatch recursion on uninterrupted observed arcs; rejected epochs break arcs."""
    n,s=ranges.shape;out=np.full_like(ranges,np.nan);ages=np.zeros((n,s),int)
    state=np.zeros(s);previous=np.zeros(s);age=np.zeros(s,int)
    for k,t in enumerate(time):
        valid=eligible[k]&np.isfinite(ranges[k])&np.isfinite(carrier[k])
        prediction=state+carrier[k]-previous
        continuous=valid&(age>0)&(abs(ranges[k]-prediction)<=rule.divergence)
        if k and t-time[k-1]>.015:continuous[:]=False
        age=np.where(valid,np.where(continuous,age+1,1),0)
        current=np.flatnonzero(valid);weight=1/np.minimum(age[current],rule.window)
        state[current]=np.where(continuous[current],
            weight*ranges[k,current]+(1-weight)*prediction[current],ranges[k,current])
        accepted=valid&(age>=rule.mature);out[k,accepted]=state[accepted]
        ages[k]=age;previous=carrier[k].copy()
    return out,ages


def evaluate(z,method,rule):
    gate=Gate(min_gain=.01,min_lock=.5)
    r=z[method+'_ranges'];cn=z[method+'_cn0'];lock=z[method+'_lock'];gain=z[method+'_gain']
    eligible=(lock>=gate.min_lock)&(cn>=gate.min_cn0)&(gain>=gate.min_gain)
    eligible&=z[method+'_spatial']>=rule.spatial
    ranges,ages=smooth(z['time_s'],r,z[method+'_carrier'],eligible,rule)
    quality=cn if rule.weighted else np.full_like(cn,45.)
    weighted=admit(z['satellite_positions'],ranges,lock,quality,gain,gate)
    unweighted=np.array([position_solution(s,p,np.isfinite(p))[0]
        for s,p in zip(z['satellite_positions'],ranges)])
    separation=np.linalg.norm((weighted['position']-unweighted)[:,:2],axis=1)
    position=weighted['position'].copy();position[separation>rule.consensus]=np.nan
    return dict(position=position,ranges=ranges,ages=ages,selected=weighted['selected'],
                scores=weighted['scores'],consensus_m=separation)
