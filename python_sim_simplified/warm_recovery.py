"""Warm-start correlator experiment. Truth exists only in generate/score.

Observed initialization is tracking/calibration evidence, not RF acquisition/PVT.
"""
import numpy as np
from scipy.optimize import minimize

POSITIONS = np.array([[0,0,0],[.0759,0,0],[0,.1609,0],[.0759,.1609,0]])
LAG = np.linspace(-1.2,1.2,49)
GRID = np.linspace(-.6,.9,151)
CHIP_M = 293.052256
SCENARIOS = ('nominal','weak','near','absent','drift','colored','multipath','motion','init_weak')

def pulse(t):
    return np.sinc(t)*np.exp(-.35*np.asarray(t)**2)

def generate(seed, scenario):
    rng=np.random.default_rng(seed)
    dirs=rng.normal(size=(6,3));dirs[:,2]=abs(dirs[:,2]);dirs/=np.linalg.norm(dirs,axis=1)[:,None]
    a=np.exp(2j*np.pi/.1902936728*(POSITIONS@dirs.T)).T
    u=dirs[0]+np.array([.015,-.01,0]) if scenario=='near' else np.array([.7,-.3,.64])
    u/=np.linalg.norm(u);b=np.exp(2j*np.pi/.1902936728*(POSITIONS@u))
    gain=np.exp(rng.normal(0,.15,4)+1j*rng.normal(0,.3,4))
    tau=rng.uniform(-.18,.18,6); phase=np.exp(1j*rng.uniform(-np.pi,np.pi,6))
    fake=tau+rng.uniform(.3,.6,6)
    cov=.65**abs(np.arange(4)[:,None]-np.arange(4)[None,:]) if scenario=='colored' else np.eye(4)
    lc=np.linalg.cholesky(.45*pulse(LAG[:,None]-LAG[None,:])+.55*np.eye(len(LAG)))
    cc=np.linalg.cholesky(cov)
    def noise(n):
        z=(rng.normal(size=(n,6,4,len(LAG)))+1j*rng.normal(size=(n,6,4,len(LAG))))/np.sqrt(2)
        return .065*np.einsum('ij,npjl,kl->npik',cc,z,lc)
    init_amp=.05 if scenario=='init_weak' else 1.
    initial=init_amp*gain[None,:,None]*a[:,:,None]*phase[:,None,None]*pulse(LAG[None,None,:]-tau[:,None,None])
    init=initial[None,:,:,:]+noise(12)
    amp=0. if scenario=='absent' else (.08 if scenario in ('weak','near') else .35)
    times=np.arange(8);delay=tau[None,:]+(.012*times[:,None] if scenario=='motion' else 0.)
    attack=amp*a[None,:,:,None]*phase[None,:,None,None]*pulse(LAG[None,None,None,:]-delay[:,:,None,None])
    counterfeit=2*np.exp(1j*rng.uniform(-np.pi,np.pi,6))[:,None,None]*b[None,:,None]*pulse(LAG[None,None,:]-fake[:,None,None])
    attack=attack+counterfeit[None,:,:,:]
    if scenario=='multipath':
        attack+=.4*amp*a[None,:,:,None]*phase[None,:,None,None]*pulse(LAG[None,None,None,:]-delay[:,:,None,None]-.16)
    drift=np.exp(1j*.035*times[:,None]*np.arange(4)[None,:]) if scenario=='drift' else np.ones((8,4))
    attack=attack*gain[None,None,:,None]*drift[:,None,:,None]
    attack+=noise(8)
    return dict(init=init,z=attack,noise=noise(40),lag=LAG.copy(),dt=.02,attack_start_s=.32),dict(delay=delay,exists=amp>0,scenario=scenario)

def subset(obs,m):
    return {**obs,**{k:obs[k][:,:,:m,:].copy() for k in ('init','z','noise')}}

def fit_rows(y,grid,lag,whitener=None):
    h=pulse(lag[None,:]-grid[:,None])
    if whitener is not None:y=y@whitener;h=h@whitener
    energy=np.sum(abs(h)**2,axis=1)
    profiles=abs(y@h.conj().T)**2/energy
    index=np.argmax(profiles,axis=-1)
    delay=grid[index];amp=np.sum(y*h[index].conj(),axis=-1)/energy[index]
    return delay,amp,profiles

def estimate(obs,methods=()):
    methods=set(methods);init=obs['init'];z=obs['z'].copy();noise=obs['noise'];lag=obs['lag'];m=z.shape[2]
    training=init.mean(axis=0) if 'init_average' in methods else init[-1]
    init_delay,_,_=fit_rows(training[:,0,:],GRID,lag)
    h=pulse(lag[None,:]-init_delay[:,None])
    a=np.sum(training*h[:,None,:],axis=-1)/np.sum(h*h,axis=-1)[:,None]
    if 'rank1_init' in methods:
        a=np.array([np.linalg.svd(training[p],full_matrices=False)[0][:,0] for p in range(6)])
    a/=a[:,0,None]
    init_tracks=np.array([fit_rows(row[:,0,:],GRID,lag)[0] for row in init])
    stability=np.std(init_tracks,axis=0)*CHIP_M
    init_ok=stability<8
    init_completed_s=float(len(init)*obs['dt']) if np.all(init_ok) else None
    space=np.eye(m,dtype=complex)
    if 'spatial_whiten' in methods:
        nc=noise.transpose(2,0,1,3).reshape(m,-1);cov=nc@nc.conj().T/nc.shape[1]
        space=np.linalg.inv(np.linalg.cholesky(cov));z=np.einsum('ij,tpjl->tpil',space,z);a=a@space.T
    def direction(frames):
        x=frames.transpose(2,0,1,3).reshape(m,-1)
        b=np.linalg.svd(x,full_matrices=False)[0][:,0]
        if 'residual_direction' in methods:
            proj=np.eye(m)[None,:,:]-a[:,:,None]*a[:,None,:].conj()/np.sum(abs(a)**2,axis=1)[:,None,None]
            residual=np.einsum('pij,tpjl->tpil',proj,frames)
            b=np.linalg.svd(residual.transpose(2,0,1,3).reshape(m,-1),full_matrices=False)[0][:,0]
        if 'projected_fit' in methods:
            proj=np.eye(m)[None,:,:]-a[:,:,None]*a[:,None,:].conj()/np.sum(abs(a)**2,axis=1)[:,None,None]
            residual=np.einsum('pij,tpjl->tpil',proj,frames)
            def loss(x):
                v=np.r_[1.,x[:m-1]+1j*x[m-1:]];pb=proj@v
                c=np.einsum('pm,tpml->tpl',pb.conj(),residual)/np.sum(abs(pb)**2,axis=1)[None,:,None]
                return np.sum(abs(residual-pb[None,:,:,None]*c[:,:,None,:])**2)
            b=b/b[0];opt=minimize(loss,np.r_[b[1:].real,b[1:].imag],method='BFGS',options={'maxiter':25})
            b=np.r_[1.,opt.x[:m-1]+1j*opt.x[m-1:]];b/=np.linalg.norm(b)
        return b
    b=direction(z[:4] if 'crossfit_direction' in methods else z)
    projected=a-(a@b.conj())[:,None]*b
    weights=projected/np.sum(abs(projected)**2,axis=1)[:,None]
    y=np.einsum('pm,tpml->tpl',weights.conj(),z)
    if 'block_direction' in methods:
        for t in range(len(z)):
            bt=direction(z[t:t+1]);pr=a-(a@bt.conj())[:,None]*bt
            wt=pr/np.sum(abs(pr)**2,axis=1)[:,None];y[t]=np.einsum('pm,pml->pl',wt.conj(),z[t])
    if 'phase_align' in methods:
        phase=np.angle(np.sum(y[0].conj()[None,:,:]*y,axis=-1));y*=np.exp(-1j*phase[:,:,None])
    lagw=None
    if 'lag_whiten' in methods:
        flat=noise.reshape(-1,len(lag));cov=(flat.conj().T@flat).real/len(flat)
        lagw=np.linalg.inv(np.linalg.cholesky(cov)).T
    mean=np.median(y.real,axis=0)+1j*np.median(y.imag,axis=0) if 'robust_average' in methods else y.mean(axis=0)
    delay,amp,profiles=fit_rows(mean,GRID,lag,lagw)
    if 'peak_baseline' not in methods:delay=lag[np.argmax(abs(mean),axis=-1)]
    if 'subgrid' in methods:
        ix=np.argmax(profiles,axis=-1);ii=np.clip(ix,1,len(GRID)-2)
        left=profiles[np.arange(6),ii-1];mid=profiles[np.arange(6),ii];right=profiles[np.arange(6),ii+1]
        delay=GRID[ii]+.005*(left-right)/(left-2*mid+right)
    frame_delay=np.array([fit_rows(row,GRID,lag,lagw)[0] for row in y])
    if 'motion_fit' in methods:
        t=np.arange(len(y));coef=np.linalg.lstsq(np.column_stack((np.ones(len(y)),t)),frame_delay,rcond=None)[0]
        delay=coef[0]+coef[1]*t[-1]
    if 'joint_fit' in methods:
        for p in range(6):
            row=z[:,p].mean(axis=0).ravel()
            fake=fit_rows((b.conj()@z[:,p].mean(axis=0))[None,:],GRID,lag)[0][0]
            def design(t):return np.column_stack(((a[p,:,None]*pulse(lag-t[0])).ravel(),(b[:,None]*pulse(lag-t[1])).ravel()))
            def objective(t):
                mat=design(t);c=np.linalg.lstsq(mat,row,rcond=None)[0];return np.sum(abs(row-mat@c)**2)
            opt=minimize(objective,[delay[p],fake],method='Nelder-Mead',options={'maxiter':60})
            delay[p]=opt.x[0];amp[p]=np.linalg.lstsq(design(opt.x),row,rcond=None)[0][0]
    fitted=amp[:,None]*pulse(lag[None,:]-delay[:,None])
    nbeam=np.einsum('pm,npml->npl',weights.conj()@space,noise)
    variance=np.mean(abs(nbeam)**2,axis=(0,2))/len(y)
    evidence=abs(amp)**2*np.sum(abs(pulse(lag[None,:]-delay[:,None]))**2,axis=1)/variance
    derivative=(pulse(lag[None,:]-delay[:,None]-.0001)-pulse(lag[None,:]-delay[:,None]+.0001))/.0002
    sigma=CHIP_M*np.sqrt(variance/(2*abs(amp)**2*np.sum(derivative**2,axis=1)))
    residual=np.mean(abs(mean-fitted)**2,axis=1)/variance
    accepted=np.ones(6,bool)
    if 'init_gate' in methods:accepted &= init_ok
    if 'presence_gate' in methods:accepted &= evidence>2*np.log(len(GRID)*6/.01)
    if 'precision_gate' in methods:accepted &= 3*sigma<20
    if 'shape_gate' in methods:accepted &= residual<2
    if 'temporal_gate' in methods:accepted &= np.std(frame_delay,axis=0)*CHIP_M<20
    if 'calibration_uncertainty' in methods:
        sigma=np.sqrt(sigma**2+stability**2/len(init));accepted &= 3*sigma<20
    if 'ambiguity_gate' in methods:
        distant=abs(GRID[None,:]-delay[:,None])>.1
        alt=np.max(np.where(distant,profiles,-np.inf),axis=1)
        accepted &= (profiles.max(axis=1)-alt)/variance>9
    if 'continuity_gate' in methods:accepted &= abs(delay-init_delay)*CHIP_M<30
    if 'holdout_gate' in methods:
        d1,c1,_=fit_rows(y[4:].mean(axis=0),GRID,lag,lagw)
        accepted &= (abs(d1-delay)*CHIP_M<20)&(abs(c1)**2*np.sum(h*h,axis=1)/(2*variance)>15)
    if 'projection_gate' in methods:accepted &= np.sum(abs(projected)**2,axis=1)/np.sum(abs(a)**2,axis=1)>.08
    if 'innovation_subtract' in methods:
        predicted=init_delay[:,None]+np.zeros((6,len(lag)))
        leakage=np.sum(mean*pulse(lag[None,:]-predicted),axis=1)/np.sum(pulse(lag[None,:]-predicted)**2,axis=1)
        corrected=mean-leakage[:,None]*pulse(lag[None,:]-init_delay[:,None])
        accepted &= np.mean(abs(corrected)**2,axis=1)/variance<3
    if 'split_consistency' in methods:
        d0=fit_rows(y[::2].mean(axis=0),GRID,lag,lagw)[0];d1=fit_rows(y[1::2].mean(axis=0),GRID,lag,lagw)[0]
        accepted &= abs(d0-d1)*CHIP_M<15
    if 'persistence_gate' in methods:
        ev=[]
        for row in y:
            dd,aa,_=fit_rows(row,GRID,lag,lagw);ev.append((abs(dd-delay)*CHIP_M<20)&(abs(aa)**2*np.sum(h*h,axis=1)/(variance*len(y))>9))
        accepted &= np.sum(ev,axis=0)>=6
    return dict(delay=delay,accepted=accepted,sigma_m=sigma,evidence=evidence,init_completed_s=init_completed_s,init_ok=init_ok,attack_start_s=obs['attack_start_s'],gap_s=None if init_completed_s is None else obs['attack_start_s']-init_completed_s)
