"""Continuous multi-Doppler recovery using measured, pre-attack tracking templates."""
from dataclasses import dataclass,asdict
import numpy as np

from acquisition import acquire
from code_replica import bandlimited_code_replica,_sample_code
from gnss import ca_code
from positioning import code_pseudorange
from presets import default_config,mate70_phase_centers
from recovery_admission import admit
from signal_recovery import FS,WAVELENGTH,C,sky,steering,array_weights,position_solution
from run_gps_spp import receiver_reference
from tracking import track


@dataclass
class ContinuousCase:
    duration: float = 1.
    cn0: float = 45.
    doppler_scale: float = 1.
    doppler_rate: float = 0.
    data_bits: bool = False
    spoof_db: float = 10.
    phase_deg: float = 0.
    angle: float = 60.
    genuine: bool = True
    iterations: int = 2


def prompt_projector(prompts,desired,iterations):
    """Fit Z_s = a_s alpha_s + v beta_s; no emitter truth supplied."""
    m=len(prompts)
    eigen,vectors=np.linalg.eigh(prompts@prompts.conj().T)
    ratio=float(eigen[-1]/np.mean(eigen[:-1]));v=vectors[:,-1]
    for _ in range(iterations):
        projection=np.eye(m)-np.outer(v,v.conj())
        projected=projection@desired
        amplitude=np.sum(projected.conj()*prompts,axis=0)/(np.sum(desired.conj()*projected,axis=0).real+.01*m)
        remaining=prompts-desired*amplitude
        _,vectors=np.linalg.eigh(remaining@remaining.conj().T)
        v=vectors[:,-1]
    return np.eye(m)-np.outer(v,v.conj()),ratio


def experiment(case,seed,attacked,policy):
    rng=np.random.default_rng(seed);u=sky(seed,8)
    antennas=np.array(list(mate70_phase_centers().values()));a=steering(u,antennas,0.)
    tangent=np.cross(u[0],[0,0,1.]);tangent/=np.linalg.norm(tangent)
    source=u[0]*np.cos(np.radians(case.angle))+tangent*np.sin(np.radians(case.angle))
    v=steering(source[None],antennas,0.)[:,0]
    gain=np.exp(1j*np.radians(case.phase_deg*rng.normal(size=4)))
    count=round(FS*case.duration);time=np.arange(count)/FS
    radii=21e6+np.arange(8)*17000.
    fd=case.doppler_scale*np.arange(-1500.,2001.,500.)
    rates=case.doppler_rate*np.linspace(-1.,1.,8)
    clean=np.zeros((4,count),complex);fake=np.zeros_like(clean);codes=[]
    for i in range(8):
        code=bandlimited_code_replica(ca_code(i+1),1.023e6,.9e6);codes.append(code)
        distance=radii[i]-WAVELENGTH*(fd[i]*time+.5*rates[i]*time**2)
        genuine=distance+75000
        # Static false position; the same satellite ephemeris determines forged code/carrier.
        forged=np.sqrt(distance**2-2*distance*u[i,0]*100+10000)+75000
        bits=rng.choice([-1.,1.],size=round(case.duration/.02)+10)
        for rho,vector,target in ((genuine,a[:,i],clean),(forged,v,fake)):
            signal=_sample_code(code,(time-rho/C)*1.023e6)*np.exp(-2j*np.pi*rho/WAVELENGTH)
            if case.data_bits:signal*=bits[(np.floor((time-rho/C)/.02).astype(int)+5)%len(bits)]
            target+=vector[:,None]*signal
    noise=(rng.normal(size=clean.shape)+1j*rng.normal(size=clean.shape))/np.sqrt(2)
    x=gain[:,None]*clean+np.sqrt(FS/10**(case.cn0/10))*noise
    if attacked:
        x[:,time>=.15]+=gain[:,None]*np.sqrt(10**(case.spoof_db/10))*fake[:,time>=.15]
        if not case.genuine:x[:,time>=.15]-=gain[:,None]*clean[:,time>=.15]
    del clean,fake,noise
    cfg=default_config();cfg.acquisition.doppler_bins_hz=np.arange(-3000.,3001.,250.)
    cfg.tracking.max_epochs=round(case.duration*1000)
    cfg.tracking.code_replica_bandwidth_hz=.9e6;cfg.tracking.refine_initial_phase=True
    cfg.tracking.carrier_discriminator='costas'
    # Startup measurements only; no synthetic Doppler or range is used as a template.
    templates=[];prns=[];startup=[]
    for i in range(8):
        acq=acquire(x[0,:round(.1*FS)],FS,i+1,1.023e6,cfg.acquisition)
        if acq['success']:
            h=track(x[0,:round(.1*FS)],FS,i+1,acq,cfg.tracking,1575.42e6,1.023e6)
            if h['success']:
                templates.append((h['epoch_start_samples'][-1]/FS,h['code_nco_phase_chips'][-1],
                                  h['code_nco_rate_hz'][-1],h['carrier_nco_frequency_hz'][-1]))
                prns.append(i)
        startup.append(dict(prn=i+1,acquired=bool(acq['success']),estimated_doppler_hz=float(acq['doppler_hz'])))
    width=round(.02*FS);weights={m:[] for m in ('raw_covariance','fold','despread')};ratios=[]
    for start in range(0,count,width):
        block=x[:,start:start+width];times=time[start:start+width]
        w,_,info=array_weights(block,a,'gated');weights['raw_covariance'].append(w)
        folded=block.reshape(4,20,2046).mean(axis=1)
        w,_,_=array_weights(folded,a,'gated');weights['fold'].append(w)
        w=a/4;ratio=0.
        if start>=round(.1*FS) and len(prns)>=3:
            local=np.stack([_sample_code(codes[i],phase+rate*(times-epoch))*np.exp(2j*np.pi*f*times)
                            for i,(epoch,phase,rate,f) in zip(prns,templates)])
            prompts=block@local.conj().T/width
            projector,ratio=prompt_projector(prompts,a[:,prns],case.iterations)
            if ratio>=3.:w=projector@a
        weights['despread'].append(w);ratios.append(ratio)
    evaluation=np.arange(.35,case.duration-.01,.01)
    reference,rotation=receiver_reference()
    distance=radii[None,:]-WAVELENGTH*(evaluation[:,None]*fd+.5*evaluation[:,None]**2*rates)
    satellites=reference+(u[None,:,:]*distance[:,:,None])@rotation
    truth=np.zeros((len(evaluation),3));traces={};tracking_records={}
    for method in ('single',*weights):
        ranges=np.full((len(evaluation),8),np.nan)
        records=[]
        for i in range(8):
            signal=x[0] if method=='single' else np.concatenate([
                w[:,i].conj()@x[:,k*width:(k+1)*width] for k,w in enumerate(weights[method])])
            acq=acquire(signal,FS,i+1,1.023e6,cfg.acquisition)
            record=dict(prn=i+1,acquired=bool(acq['success']))
            if acq['success']:
                h=track(signal,FS,i+1,acq,cfg.tracking,1575.42e6,1.023e6)
                ranges[:,i]=code_pseudorange(h,evaluation,FS,.001,round((radii[i]+75000)/10000)*10000)
                record['tracking_success']=bool(h['success'])
            records.append(record)
        valid=np.isfinite(ranges)
        position=np.array([position_solution(s,r,v)[0] for s,r,v in zip(satellites,ranges,valid)])
        traces[method+'_position']=position
        traces[method+'_ranges']=ranges
        tracking_records[method]=records
        if method=='despread':
            admitted=admit(evaluation,satellites,ranges,valid,policy)
            traces['admitted_position']=admitted['position']
            traces['admitted_selected']=admitted['selected']
    traces.update(time_s=evaluation,truth_position=truth,despread_ratio=np.array(ratios),
                  satellite_positions=satellites)
    return traces,dict(startup=startup,tracking=tracking_records,case=asdict(case),
                      template_end_s=.1,attack_start_s=.15,minimum_latency_s=.02)
