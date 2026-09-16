"""Explicit research copy of continuous receiver; 100ms measured startup aid, no known bits.
Source baseline continuous_improvement.experiment, modifications confined to this research copy.
"""
from dataclasses import asdict
import numpy as np
from continuous_improvement import (Case,generate,local_signal,short_projector,cancel_replicas,
    FS,WAVELENGTH,C,array_weights,position_solution,default_config,code_pseudorange,carrier_range,track,prompt_projector)
from array_acquisition import acquire_array
from cold_next import calibrate,align_prompts,complex_calibrate


def prompts(block,local):
    m,n=block.shape;s=len(local);chunks=n//2046
    return np.einsum('mkn,skn->msk',block.reshape(m,chunks,2046),local.conj().reshape(s,chunks,2046))/2046


def robust_projector(z,a):
    """Equal-PRN residual influence after eliminating the nominal sky component."""
    m,s,k=z.shape;flat=z.reshape(m,-1);desired=np.repeat(a,k,axis=1)
    p,ratio=prompt_projector(flat,desired,2)
    projected=p@desired
    coeff=np.sum(projected.conj()*flat,axis=0)/(np.sum(desired.conj()*projected,axis=0).real+.01*m)
    remaining=(flat-desired*coeff).reshape(m,s,k)
    energy=np.mean(abs(remaining)**2,axis=(0,2))
    weights=np.sqrt(np.median(energy)/(np.median(energy)+energy))
    weighted=(remaining*weights[None,:,None]).reshape(m,-1)
    _,vectors=np.linalg.eigh(weighted@weighted.conj().T);v=vectors[:,-1]
    return np.eye(m)-np.outer(v,v.conj()),ratio


def experiment(case,seed,attacked,methods,*,extended=False,revision=141):
    x,a,codes,evaluation,satellites,radii=generate(case,seed,attacked)
    original=x
    diagnostic={'usable':False,'template_updates':[]};history=[]
    def acquire(signal,fs,prn,rate,config):
        raw=original[:,:len(signal)] if np.shares_memory(signal,original) else signal[None,:]
        return acquire_array(raw,fs,prn,rate,config)
    def startup_phase(block,local,desired):
        z=prompts(block,local)
        g,info=calibrate(z,desired,140 if revision>=141 else 125)
        if revision>=150:g,info=complex_calibrate(z,desired)
        diagnostic.update(info)
        return 1/g.conj() if revision>=150 else np.exp(1j*np.angle(g))
    def persistent_projector(block,local,desired,short_samples):
        z=prompts(block,local);history.append(z)
        joined=np.concatenate(history,axis=2)
        if revision>=144:
            aligned=align_prompts(joined,'reference' if revision>=147 else 'sign')
            joined=aligned.reshape(*aligned.shape[:2],-1,10).mean(axis=3)
        if revision>=145:return robust_projector(joined,desired)
        return prompt_projector(joined.reshape(len(z),-1),np.repeat(desired,joined.shape[2],axis=1),2)
    m,n=x.shape;cfg=default_config()
    cfg.acquisition.doppler_bins_hz=np.arange(-4000.,4001.,250.)
    cfg.tracking.max_epochs=round(case.duration*1000)
    cfg.tracking.code_replica_bandwidth_hz=.9e6;cfg.tracking.refine_initial_phase=True
    cfg.tracking.carrier_discriminator='costas'
    startup=round(.1*FS);templates=[];prns=[]
    for i in range(case.stars):
        acq=acquire(x[0,:startup],FS,i+1,1.023e6,cfg.acquisition)
        if acq['success']:
            h=track(x[0,:startup],FS,i+1,acq,cfg.tracking,1575.42e6,1.023e6)
            if h['success']:
                prns.append(i);templates.append((h['epoch_start_samples'][-1]/FS,h['code_nco_phase_chips'][-1],
                    h['code_nco_rate_hz'][-1],h['carrier_nco_frequency_hz'][-1]))
    phase_estimate=np.ones(m,complex)
    if 'calibrated' in methods and len(prns)>=3:
        t0=np.arange(startup)/FS
        locals0=np.stack([local_signal(codes[i],temp,t0) for i,temp in zip(prns,templates)])
        phase_estimate=startup_phase(x[:,:startup],locals0,a[:,prns])
        if revision>=143:
            initial_projector=persistent_projector(x[:,:startup]*phase_estimate.conj()[:,None],locals0,a[:,prns],2046)
    width=round(case.covariance_ms*.001*FS)
    streams={method:np.empty((case.stars,n),complex) for method in methods}
    gains={method:[] for method in methods};ratios=[]
    for start in range(0,n,width):
        block=x[:,start:start+width];t=np.arange(start,start+block.shape[1])/FS
        if revision>=146 and start>=2*startup and start%startup==0:
            for j,i in enumerate(prns):
                observed=streams['calibrated'][i,start-startup:start]
                acq=acquire(observed,FS,i+1,1.023e6,cfg.acquisition)
                if acq['success']:
                    h=track(observed,FS,i+1,acq,cfg.tracking,1575.42e6,1.023e6)
                    if h['success']:
                        templates[j]=((start-startup+h['epoch_start_samples'][-1])/FS,
                            h['code_nco_phase_chips'][-1],h['code_nco_rate_hz'][-1],h['carrier_nco_frequency_hz'][-1])
                        diagnostic['template_updates'].append([float(start/FS),int(i+1)])
        _,_,info=array_weights(block,a,'gated');v=info['dominant'];ratio=info['eigen_ratio']
        projections={};beam=a/m
        if start>=startup and len(prns)>=3:
            local=np.stack([local_signal(codes[i],temp,t) for i,temp in zip(prns,templates)])
            if 'despread' in methods:
                projections['despread']=prompt_projector(block@local.conj().T/block.shape[1],a[:,prns],2)
            if 'short' in methods:
                projections['short']=short_projector(block,local,a[:,prns],round(case.short_ms*.001*FS))
        for method in methods:
            if method=='calibrated':
                calibrated=phase_estimate.conj()[:,None]*block
                w=beam;g=np.ones(case.stars)
                if (start>=startup or revision>=143) and len(prns)>=3:
                    projector=persistent_projector if revision>=142 else short_projector
                    if start<startup:p,r=initial_projector
                    else:p,r=projector(calibrated,local,a[:,prns],round(case.short_ms*.001*FS))
                    if revision>=148 and diagnostic['selected_model']=='sky':r=0.
                    if r>=3.:
                        projected=p@a;g=np.sum(a.conj()*projected,axis=0).real/m
                        w=projected/(m*g+case.loading)[None,:]
                y=w.conj().T@calibrated
            elif method=='single':
                w=np.zeros_like(a);w[0]=1.;y=w.conj().T@block;g=np.ones(case.stars)
            elif method=='sic':
                cleaned=block
                if start>=startup and ratio>=3.:
                    cleaned,_=cancel_replicas(block,t,codes,templates,prns,a,v,case.sic_radius,case.min_gain)
                y=beam.conj().T@cleaned;g=np.ones(case.stars)
            elif method=='mvdr':
                w,_,_=array_weights(block,a,'mvdr',loading=case.loading)
                y=w.conj().T@block;g=1/(m*np.sum(abs(w)**2,axis=0))
            else:
                p=np.eye(m)-np.outer(v,v.conj());r=ratio
                if method in ('short','despread'):
                    p,r=projections[method] if method in projections else (np.eye(m),0.)
                w=beam;g=np.ones(case.stars)
                if r>=3.:
                    projected=p@a;g=np.sum(a.conj()*projected,axis=0).real/m
                    # Loaded distortionless normalization; loading is a defined beam model.
                    w=projected/(m*g+case.loading)[None,:]
                y=w.conj().T@block
            streams[method][:,start:start+block.shape[1]]=y;gains[method].append(g)
        ratios.append(ratio)
    traces=dict(time_s=evaluation,satellite_positions=satellites,truth_position=np.zeros((len(evaluation),3)),
                raw_ratio=np.array(ratios),startup_prns=np.array(prns),startup_phase_estimate=phase_estimate)
    traces['truth_position'][:,0]=case.velocity_east*evaluation
    block_index=(evaluation/(case.covariance_ms*.001)).astype(int)
    for method in methods:
        ranges=np.full((len(evaluation),case.stars),np.nan)
        lock=np.full_like(ranges,np.nan);cn0=lock.copy();dll=lock.copy()
        carrier=lock.copy();spatial=lock.copy()
        if extended:
            from carrier_admission import spatial_evidence
            observed=x*phase_estimate.conj()[:,None] if method=='calibrated' else x
        for i in range(case.stars):
            signal=streams[method][i]
            if revision>=149 and method=='calibrated' and len(prns)>=3 and diagnostic['selected_model']!='sky':
                residual_array=initial_projector[0]@(x[:,:startup]*phase_estimate.conj()[:,None])
                acq=acquire_array(residual_array,FS,i+1,1.023e6,cfg.acquisition)
            else:acq=acquire(signal[:startup],FS,i+1,1.023e6,cfg.acquisition)
            if acq['success']:
                h=track(signal,FS,i+1,acq,cfg.tracking,1575.42e6,1.023e6)
                if h['success']:
                    ranges[:,i]=code_pseudorange(h,evaluation,FS,.001,round((radii[i]+75000)/10000)*10000)
                    # Last completed tracking epoch; no future interpolation of quality.
                    ends=h['epoch_start_samples']/FS+.001
                    idx=np.searchsorted(ends,evaluation,side='right')-1
                    lock[:,i]=h['phase_lock_metric'][idx];cn0[:,i]=h['cn0_db_hz'][idx]
                    dll[:,i]=h['dll_error'][idx]
                    if extended:
                        carrier[:,i]=carrier_range(h,evaluation,FS,.001,WAVELENGTH)
                        spatial[:,i]=spatial_evidence(observed,codes[i],h,evaluation,a[:,i],FS)
        traces[method+'_ranges']=ranges;traces[method+'_lock']=lock;traces[method+'_cn0']=cn0
        traces[method+'_dll']=dll;traces[method+'_gain']=np.array(gains[method])[block_index]
        traces[method+'_position']=np.array([position_solution(s,r,np.isfinite(r))[0] for s,r in zip(satellites,ranges)])
        if extended:
            traces[method+'_carrier']=carrier;traces[method+'_spatial']=spatial
        del streams[method]
    traces['joint_usable']=np.array(diagnostic['usable'])
    traces['cold_revision']=np.array(revision)
    return traces,dict(joint_calibration=diagnostic,case=asdict(case),seed=seed,attacked=attacked,methods=methods,
                       assumed_trusted_startup_s=.1,attack_start_s=case.attack_start,buffer_latency_s=case.covariance_ms*.001)
