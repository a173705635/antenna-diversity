"""Continuous single-emitter RF scene with explicit startup/attitude assumptions."""
from dataclasses import dataclass
import numpy as np
from ..code_replica import bandlimited_code_replica, _sample_code
from ..gnss import ca_code
from ..presets import mate70_phase_centers
from .recovery_geometry import FS, WAVELENGTH, C, sky, steering
from ..gps_simulation import receiver_reference


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
    nav_bit_epoch_s: float=0.


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
            if case.data_bits:signal*=bits[(np.floor((t-ranges/C-case.nav_bit_epoch_s)/.02).astype(int)+5)%len(bits)]
            x+=(gain*vector)[:,None]*(signal*scale)
    reference,rotation=receiver_reference()
    evaluation=np.arange(.15,case.duration-.01,.01)
    distance=radii-WAVELENGTH*(evaluation[:,None]*fd+.5*evaluation[:,None]**2*rates)
    satellites=reference+(u[None,:,:]*distance[:,:,None])@rotation
    return x,a,codes,evaluation,satellites,radii


def local_signal(code,template,t,offset=0.):
    epoch,phase,rate,f=template
    return _sample_code(code,phase+rate*(t-epoch)+offset)*np.exp(2j*np.pi*f*t)
