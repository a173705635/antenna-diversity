"""Paired 8/24-star RF; original eight retain geometry, signals and motion."""
import numpy as np
from code_replica import bandlimited_code_replica, _sample_code
from gnss import ca_code
from presets import mate70_phase_centers
from signal_recovery import FS, WAVELENGTH, C, sky, steering
from run_gps_spp import receiver_reference

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
    fd=case.doppler_scale*np.r_[np.linspace(-1500.,2000.,8),np.linspace(-1450.,1950.,16)][:case.stars]
    rates=case.doppler_rate*np.r_[np.linspace(-1.,1.,8),np.linspace(-.95,.95,16)][:case.stars]
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
