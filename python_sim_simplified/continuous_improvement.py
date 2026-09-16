"""Continuous array recovery experiments; trusted startup and attitude assumed."""
from dataclasses import dataclass,asdict
import numpy as np
from acquisition import acquire
from code_replica import bandlimited_code_replica,_sample_code
from gnss import ca_code
from positioning import code_pseudorange,carrier_range
from presets import default_config,mate70_phase_centers
from doppler_recovery import prompt_projector
from signal_recovery import FS,WAVELENGTH,C,sky,steering,array_weights,position_solution
from run_gps_spp import receiver_reference
from tracking import track


@dataclass
class Case:
    duration: float=1.
    stars: int=8
    antennas: int=4
    cn0: float=45.
    data_bits: bool=True
    doppler_scale: float=1.
    doppler_rate: float=0.
    spoof_db: float=10.
    fake_east: float=100.
    phase_deg: float=0.
    amplitude_db: float=0.
    angle: float=60.
    genuine: bool=True
    attack_end: float=1000.
    attack_start: float=.15
    short_ms: int=1
    covariance_ms: int=20
    loading: float=.01
    min_gain: float=.05
    sic_radius: float=.5
    velocity_east: float=0.
    fake_rate: float=0.


def generate(case,seed,attacked):
    rng=np.random.default_rng(seed);u=sky(seed,case.stars)
    antennas=np.array(list(mate70_phase_centers().values()))[:case.antennas]
    a=steering(u,antennas,0.)
    tangent=np.cross(u[0],[0,0,1.]);tangent/=np.linalg.norm(tangent)
    source=u[0]*np.cos(np.radians(case.angle))+tangent*np.sin(np.radians(case.angle))
    v=steering(source[None],antennas,0.)[:,0]
    gain=np.exp(1j*np.radians(case.phase_deg*rng.normal(size=case.antennas)))
    gain*=10**(case.amplitude_db*rng.normal(size=case.antennas)/20)
    n=round(FS*case.duration);t=np.arange(n)/FS
    radii=21e6+np.arange(case.stars)*17000.
    fd=case.doppler_scale*np.linspace(-1500.,2000.,case.stars)
    rates=case.doppler_rate*np.linspace(-1.,1.,case.stars)
    noise=(rng.normal(size=(case.antennas,n))+1j*rng.normal(size=(case.antennas,n)))/np.sqrt(2)
    x=np.sqrt(FS/10**(case.cn0/10))*noise;del noise
    attack=(t>=case.attack_start)&(t<case.attack_end)&attacked
    codes=[]
    for i in range(case.stars):
        code=bandlimited_code_replica(ca_code(i+1),1.023e6,.9e6);codes.append(code)
        distance=radii[i]-WAVELENGTH*(fd[i]*t+.5*rates[i]*t*t)
        rho=distance+75000
        forged=np.sqrt(distance**2-2*distance*u[i,0]*case.fake_east+case.fake_east**2)+75000
        if case.velocity_east or case.fake_rate:
            east=case.velocity_east*t
            false_east=east+case.fake_east+case.fake_rate*np.maximum(t-case.attack_start,0.)
            rho=np.sqrt(distance**2-2*distance*u[i,0]*east+east**2)+75000
            forged=np.sqrt(distance**2-2*distance*u[i,0]*false_east+false_east**2)+75000
        bits=rng.choice([-1.,1.],size=round(case.duration/.02)+10)
        for ranges,vector,scale in ((rho,a[:,i],np.where(attack&~np.array(case.genuine),0.,1.)),
                                   (forged,v,attack*np.sqrt(10**(case.spoof_db/10)))):
            signal=_sample_code(code,(t-ranges/C)*1.023e6)*np.exp(-2j*np.pi*ranges/WAVELENGTH)
            if case.data_bits:signal*=bits[(np.floor((t-ranges/C)/.02).astype(int)+5)%len(bits)]
            x+=(gain*vector)[:,None]*(signal*scale)
    reference,rotation=receiver_reference()
    evaluation=np.arange(.15,case.duration-.01,.01)
    distance=radii-WAVELENGTH*(evaluation[:,None]*fd+.5*evaluation[:,None]**2*rates)
    satellites=reference+(u[None,:,:]*distance[:,:,None])@rotation
    return x,a,codes,evaluation,satellites,radii


def local_signal(code,template,t,offset=0.):
    epoch,phase,rate,f=template
    return _sample_code(code,phase+rate*(t-epoch)+offset)*np.exp(2j*np.pi*f*t)


def short_projector(block,local,desired,short_samples):
    """Stack independently phased short prompts; do not average signed bits."""
    m,n=block.shape;s=len(local);chunks=n//short_samples
    prompts=np.einsum('mkn,skn->msk',block.reshape(m,chunks,short_samples),
                      local.conj().reshape(s,chunks,short_samples))/short_samples
    return prompt_projector(prompts.reshape(m,s*chunks),np.repeat(desired,chunks,axis=1),2)


def startup_phase(block,local,desired):
    """Relative channel phase from signed-bit-insensitive startup cross prompts."""
    m,n=block.shape;s=len(local);chunks=n//2046
    z=np.einsum('mkn,skn->msk',block.reshape(m,chunks,2046),local.conj().reshape(s,chunks,2046))/2046
    cross=np.sum(z*z[0].conj()[None,:,:]*(desired[0]/desired)[:,:,None],axis=(1,2))
    return np.exp(1j*np.angle(cross))


def cancel_replicas(block,t,codes,templates,prns,desired,source,radius,min_gain):
    """Spatially isolate each PRN, estimate delay and 1-ms amplitudes, then cancel.

    Source is a measured unit vector. No transmitter states or data bits supplied.
    Limited local delay search is an explicit model assumption, not reacquisition.
    """
    m,n=block.shape;replica=np.zeros(n,complex);used=np.zeros(desired.shape[1],bool)
    unit=2046;chunks=n//unit
    for i,template in zip(prns,templates):
        a=desired[:,i];q=source-a*np.vdot(a,source)/m
        response=np.vdot(source,q).real
        if response<min_gain:continue
        isolated=q.conj()@block/response
        center=0.
        for width in (radius,radius/4):
            offsets=center+np.linspace(-width,width,9)
            local=np.stack([local_signal(codes[i],template,t,d) for d in offsets])
            pieces=local.reshape(9,chunks,unit)
            energy=np.sum(abs(pieces)**2,axis=2)
            corr=np.sum(isolated.reshape(1,chunks,unit)*pieces.conj(),axis=2)
            best=np.argmax(np.sum(abs(corr)**2/energy,axis=1));center=offsets[best]
        coeff=corr[best]/energy[best]
        replica+=(coeff[:,None]*pieces[best]).reshape(n);used[i]=True
    return block-source[:,None]*replica,used


def experiment(case,seed,attacked,methods,*,extended=False):
    x,a,codes,evaluation,satellites,radii=generate(case,seed,attacked)
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
    width=round(case.covariance_ms*.001*FS)
    streams={method:np.empty((case.stars,n),complex) for method in methods}
    gains={method:[] for method in methods};ratios=[]
    for start in range(0,n,width):
        block=x[:,start:start+width];t=np.arange(start,start+block.shape[1])/FS
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
                if start>=startup and len(prns)>=3:
                    p,r=short_projector(calibrated,local,a[:,prns],round(case.short_ms*.001*FS))
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
            acq=acquire(signal[:startup],FS,i+1,1.023e6,cfg.acquisition)
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
    return traces,dict(case=asdict(case),seed=seed,attacked=attacked,methods=methods,
                       assumed_trusted_startup_s=.1,attack_start_s=case.attack_start,buffer_latency_s=case.covariance_ms*.001)
