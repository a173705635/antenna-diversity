"""Continuous sampled RF -> acquisition -> tracking -> aided PVT research.

No attack schedule or true position is supplied. Nominal attitude, ephemeris,
code-period aiding and synchronized samples remain explicit assumptions.
"""
from dataclasses import replace
import numpy as np
from .acquisition import acquire
from .array_acquisition import acquire_array
from .tracking import track
from .presets import default_config
from .positioning import code_pseudorange, carrier_range, solve_position
from .single_source_scene import local_signal
from .recovery_geometry import FS, WAVELENGTH, position_solution
from .code_replica import _sample_code
from .array_processing import prompt_projector
from .gps_simulation import receiver_reference
from .mechanism_phase import phase_prediction
from .mechanism_smoothing import smooth_history


def coherent_cn0(prompt,width):
    """Estimate (E[I虏]-E[Q虏])/(2 E[Q虏] T); phase alignment is required.

    Nonpositive coherent power is an unobservable signal estimate, represented
    as NaN. This is not an acquisition metric or an independent phase test.
    """
    count=np.minimum(np.arange(len(prompt))+1,width)
    i=np.convolve(prompt.real**2,np.ones(width),'full')[:len(prompt)]/count
    q=np.convolve(prompt.imag**2,np.ones(width),'full')[:len(prompt)]/count
    # Fitted initial phase leaves no single-sample noise degree of freedom.
    measured=(count>=2)&(i>q)
    result=np.full(len(prompt),np.nan)
    result[measured]=10*np.log10((i[measured]-q[measured])/(2*q[measured]*.001))
    return result


def receive_measured(signal,prn,acquisition_cfg,tracking_cfg):
    window=round(FS*.001*acquisition_cfg.noncoherent_ms*acquisition_cfg.coherent_ms)
    cfg=replace(tracking_cfg,acquisition_ms=acquisition_cfg.noncoherent_ms*acquisition_cfg.coherent_ms)
    cursor=0;segments=[]
    while cursor+window+4092<=len(signal):
        acq=acquire(signal[cursor:cursor+window],FS,prn,1.023e6,acquisition_cfg)
        if not acq['success']:
            cursor+=window
            continue
        h=track(signal[cursor:],FS,prn,acq,cfg,1575.42e6,1.023e6)
        if not h['success']:break
        h['cn0_db_hz']=coherent_cn0(h['prompt'],100)
        snr=10**(h['cn0_db_hz']/10)*.001
        expected=snr/(1+snr)
        good=(h['phase_lock_metric']>=.8*expected)&(h['cn0_db_hz']>=25)&(abs(h['pll_error_rad'])<np.pi/2)
        bad=0;end=None
        for k in range(100,len(good)):
            bad=0 if good[k] else bad+1
            if bad>=5:
                end=k+1
                break
        if end is not None:
            h={key:value[:end] if isinstance(value,np.ndarray) else value for key,value in h.items()}
        h['epoch_start_samples']+=cursor
        segments.append(dict(history=h,tracking_id=f'tracking-{len(segments)}'))
        cursor=int(h['epoch_start_samples'][-1])+2046
        if end is None:break
    return dict(segments=segments)


def spatial_match(block,local,desired):
    prompts=(block*local.conj()).reshape(len(block),-1,2046).mean(axis=2)
    covariance=prompts@prompts.conj().T
    return float(np.vdot(desired,covariance@desired).real/(len(desired)*np.trace(covariance).real))


def calibrate(block, local, desired):
    m,n=block.shape;chunks=n//2046
    z=np.einsum('mkn,skn->msk',block.reshape(m,chunks,2046),local.conj().reshape(len(local),chunks,2046))/2046
    cross=np.sum(z*z[0].conj()[None,:,:]*(desired[0]/desired)[:,:,None],axis=1)
    time=(np.arange(chunks)+.5)*.001
    angles=np.unwrap(np.angle(cross),axis=1).T
    rate,phase=np.polyfit(time,angles,1)
    residual=angles-(time[:,None]*rate+phase)
    error=np.sqrt(np.sum(residual**2,axis=0)/(chunks-2)/np.sum((time-time.mean())**2))
    rate=np.where(abs(rate)>3*error,rate,0.)
    phase=angles.mean(axis=0)-time.mean()*rate
    return np.exp(1j*phase),rate



def residual_trigger(prompts,desired,noise=1.):
    """Score residual reduction per added complex parameter and noise power.

    Noise is measured raw sample power / coherent sample count, so unresolved
    signals inflate this estimate. Threshold 3 is a research setting, not a Pfa.
    """
    fitted=desired*np.sum(desired.conj()*prompts,axis=0)/len(desired)
    projector,_=prompt_projector(prompts,desired,6)
    projected=projector@desired
    amplitude=np.sum(projected.conj()*prompts,axis=0)/(np.sum(desired.conj()*projected,axis=0).real+.01*len(desired))
    residual=projector@(prompts-desired*amplitude)
    reduction=np.sum(abs(prompts-fitted)**2)-np.sum(abs(residual)**2)
    return projector,float(reduction/(noise*(prompts.shape[1]+len(desired)-1)))


def source_evidence(block,local,desired):
    """At least two residual dimensions and a nonzero desired vector required.

    On-direction energy is compared with mean orthogonal energy. White noise
    gives a ratio near one; threshold three is an uncalibrated research choice.
    """
    prompts=(block*local.conj()).reshape(len(block),-1,2046).mean(axis=2)
    covariance=prompts@prompts.conj().T
    on=np.vdot(desired,covariance@desired).real/np.vdot(desired,desired).real
    off=(np.trace(covariance).real-on)/(len(block)-1)
    return float(on/off)


def beam(block, a, local, startup_cov, prns):
    m,n=block.shape;chunks=n//2046
    prompts=np.einsum('mkn,skn->msk',block.reshape(m,chunks,2046),local.conj().reshape(len(local),chunks,2046))/2046
    noise=float(np.mean(abs(block)**2)/2046)
    p,ratio=residual_trigger(prompts.reshape(m,-1),np.repeat(a[:,prns],chunks,axis=1),noise)
    if ratio < 3.:
        return a/len(a), np.ones(a.shape[1]), ratio
    projected = p@a
    response = np.sum(a.conj()*projected, axis=0).real
    return projected/(response+.01)[None, :], response/len(a), ratio


def frontend(record, subset):
    x=record['samples'][subset]; a=record['nominal'][subset]
    codes=record['codes']; times=record['times']; stars=len(codes); count=len(times)
    cfg=default_config(); cfg.acquisition.doppler_bins_hz=np.arange(-4000.,4001.,250.)
    cfg.acquisition.noncoherent_ms=100
    cfg.tracking.cn0_window_epochs=100
    cfg.tracking.max_epochs=round(x.shape[1]/FS*1000)
    cfg.tracking.code_replica_bandwidth_hz=.9e6;cfg.tracking.refine_initial_phase=True
    cfg.tracking.carrier_discriminator='costas'
    startup=round(.1*FS); width=round(.02*FS)
    templates=[]; prns=[]
    for i in range(stars):
        acq=acquire_array(x[:,:startup],FS,i+1,1.023e6,cfg.acquisition)
        if acq['success']:
            h=track(x[0,:startup],FS,i+1,acq,cfg.tracking,1575.42e6,1.023e6)
            if h['success']:
                prns.append(i)
                templates.append((h['epoch_start_samples'][-1]/FS,h['code_nco_phase_chips'][-1],
                                  h['code_nco_rate_hz'][-1],h['carrier_nco_frequency_hz'][-1]))
    # At least four actually acquired/tracked PRNs are required for startup.
    ranges=np.full((count,stars),np.nan); lock=ranges.copy(); cn0=ranges.copy(); carrier=ranges.copy();support=ranges.copy();arc=np.full((count,stars),-1,int)
    if len(prns)<4:
        return dict(time=times,available_time=times+.02,satellites=record['satellites'],
                    ranges=ranges,lock=lock,cn0=cn0,carrier=carrier,gain=np.zeros_like(ranges),
                    startup_prns=np.array(prns),calibration=np.full(len(a),np.nan),ratio=np.zeros(count),support=support,arc=arc)
    t0=np.arange(startup)/FS
    local0=np.stack([local_signal(codes[i],template,t0) for i,template in zip(prns,templates)])
    gain,rate=calibrate(x[:,:startup],local0,a[:,prns])
    calibrated=x/(gain[:,None]*np.exp(1j*rate[:,None]*np.arange(x.shape[1])[None,:]/FS))
    startup_cov=calibrated[:,:startup]@calibrated[:,:startup].conj().T/startup
    stream=np.empty((stars,x.shape[1]),complex); gains=[];ratios=[];bases=[]
    for start in range(0,x.shape[1],width):
        block=calibrated[:,start:start+width]
        t=np.arange(start,start+len(block[0]))/FS
        local=np.stack([local_signal(codes[i],template,t) for i,template in zip(prns,templates)])
        # Templates only exist for acquired PRNs; beam steering still covers all.
        if start < startup:
            w=a/len(a); retained=np.ones(stars); ratio=0.
        else:
            w,retained,ratio=beam(block,a,local,startup_cov,prns)
        if ratio>=3.:
            _,vectors=np.linalg.eigh(w@w.conj().T);basis=vectors[:,1:]
        else:basis=np.eye(len(a))
        bases.append(basis)
        stream[:,start:start+block.shape[1]]=w.conj().T@block
        gains.append(retained);ratios.append(ratio)
    for i,signal in enumerate(stream):
        reception=receive_measured(signal,i+1,cfg.acquisition,cfg.tracking)
        for identity,segment in enumerate(reception['segments']):
                h=segment['history'];starts=h['epoch_start_samples']/FS
                selected=(times>=starts[0]+.02)&(times<starts[-1]+.001)
                rr=code_pseudorange(h,times,FS,.001,record['coarse'][i])
                ranges[selected,i]=rr[selected]
                carrier[selected,i]=carrier_range(h,times,FS,.001,WAVELENGTH)[selected]
                idx=np.searchsorted(starts+.001,times[selected],side='right')-1
                lock[selected,i]=h['phase_lock_metric'][idx];cn0[selected,i]=h['cn0_db_hz'][idx];arc[selected,i]=identity
                for k in np.flatnonzero(selected):
                    t=times[k]
                    end=round(t*FS);begin=end-width;sample_time=np.arange(begin,end)/FS
                    hi=np.searchsorted(starts,sample_time,side='right')-1
                    dt=sample_time-starts[hi]
                    code_phase=h['code_nco_phase_chips'][hi]+h['code_nco_rate_hz'][hi]*dt
                    phase=h['carrier_nco_phase_rad'][hi]+2*np.pi*h['carrier_nco_frequency_hz'][hi]*dt
                    local=_sample_code(codes[i],code_phase)*np.exp(1j*phase)
                    basis=bases[end//width-1];desired=basis.conj().T@a[:,i]
                    # A residual 1-D array or nulled genuine steering provides no directional evidence.
                    if basis.shape[1]>=2 and np.vdot(desired,desired).real/len(a)>=.05:
                        support[k,i]=source_evidence(basis.conj().T@calibrated[:,begin:end],local,desired)
    indices=(times/.02).astype(int)
    return dict(time=times,available_time=times+.02,satellites=record['satellites'],
                ranges=ranges,lock=lock,cn0=cn0,carrier=carrier,gain=np.array(gains)[indices],
                startup_prns=np.array(prns),calibration=gain,calibration_rate=rate,ratio=np.array(ratios)[indices],support=support,arc=arc)


def navigate(z):
    out=np.full((len(z['time']),3),np.nan); candidates=out.copy(); initialized=None; stable=[];last=None;isolated=False;recovery_count=0
    valid=np.isfinite(z['ranges'])&(z['lock']>=.5)&(z['cn0']>=25)&(z['gain']>=.05)&(z['support']>=3.)
    ranges,ages=smooth_history(z,valid)
    valid &= np.isfinite(ranges)
    anchor=-1000
    for k in range(len(out)):
        separation=np.inf
        phase_ok=False
        p,condition,residual=position_solution(z['satellites'][k],ranges[k],valid[k])
        if valid[k].sum()>=4 and condition<=1000:
            reference,rotation=receiver_reference()
            sigma=10**((45-z['cn0'][k,valid[k]])/20)
            fit=solve_position(z['satellites'][k,valid[k]],ranges[k,valid[k]],sigma_m=sigma)
            p=(fit['position_m']-reference)@rotation.T
            fresh=p.copy()
            if valid[k].sum()>=6:
                indices=np.flatnonzero(valid[k]);separation=0.
                for drop in indices:
                    retained=indices[indices!=drop]
                    alternate=solve_position(z['satellites'][k,retained],ranges[k,retained],sigma_m=10**((45-z['cn0'][k,retained])/20))
                    delta=(alternate['position_m']-fit['position_m'])@rotation.T
                    separation=max(separation,float(np.linalg.norm(delta[:2])))
            prediction=phase_prediction(z,valid,out,k)
            phase_ok=prediction is not None
            if phase_ok:p=.1*p+.9*prediction if separation<=20. else prediction
        if separation>20. and not phase_ok:p=np.full(3,np.nan)
        if separation<=20.:anchor=k
        last=p.copy() if np.all(np.isfinite(p)) else None
        candidates[k]=p
        if initialized is None:
            if np.all(np.isfinite(p)):
                stable.append(p)
                if len(stable)>=3:
                    window=np.array(stable[-3:])
                    if np.max(np.linalg.norm(window[:,:2]-window[:,:2].mean(axis=0),axis=1))<=20:
                        initialized=k
            else:stable=[]
        if initialized is not None:
            isolated |= not np.all(np.isfinite(p))
            if isolated:
                confirmed=(valid[k]&(z['support'][k]>=3.)).sum()>=6 and np.all(np.isfinite(p)) and separation<=20.
                recovery_count=recovery_count+1 if confirmed else 0
                if recovery_count>=10:
                    isolated=False;p=fresh.copy();last=p.copy();recovery_count=0
            if not isolated and (separation<=20. or phase_ok and k-anchor<=20):out[k]=p
    return dict(position=out,candidate_position=candidates,selected=valid,
                initialization_time=None if initialized is None else float(z['available_time'][initialized]))
