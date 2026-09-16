"""Assisted correlator-domain single-spoofer recovery, not RF/PVT.

Receiver inputs: nominal sky steering, lag bank, two coherent blocks,
independent noise-only calibration and measured common-injection gain samples.
Truth and scenario are deliberately outside the receiver interface.
"""
import numpy as np
from scipy.optimize import minimize, minimize_scalar

POSITIONS = np.array([[0,0,0],[.0759,0,0],[0,.1609,0],[.0759,.1609,0]])
LAG = np.linspace(-1.6, 1.6, 41)

def shape(t, delay, bandwidth=1.):
    f = np.linspace(0, 1.2*bandwidth, 65)
    p = np.sinc(f)**2
    p[[0,-1]] *= .5
    p /= p.sum()
    return np.cos(2*np.pi*(np.asarray(t)-delay)[...,None]*f) @ p

def generate(seed, scenario):
    rng = np.random.default_rng(seed)
    m = {'two':2, 'three':3}.get(scenario,4)
    dirs = rng.normal(size=(6,3)); dirs[:,2] = abs(dirs[:,2]); dirs /= np.linalg.norm(dirs,axis=1)[:,None]
    a = np.exp(2j*np.pi/.1902936728*(POSITIONS[:m]@dirs.T)).T
    u = np.array([.7,-.3,.64]); u /= np.linalg.norm(u)
    b = np.exp(2j*np.pi/.1902936728*(POSITIONS[:m]@u))
    gain = np.exp(rng.normal(0,.15,m)+1j*rng.normal(0,.3,m)); gain /= gain[0]
    tau = rng.uniform(-.22,.22,6)
    fake = tau + rng.uniform(.32,.75,6)
    if scenario == 'overlap': fake = tau + .025
    amp = .16 if scenario in ('weak','overlap','two','three') else .5
    if scenario in ('absent','noise'): amp = 0.
    alpha = amp*np.exp(1j*rng.uniform(-np.pi,np.pi,6))
    beta = 2*np.exp(1j*rng.uniform(-np.pi,np.pi,6))
    if scenario == 'noise': beta *= 0
    bw = .78 if scenario == 'bandwidth' else 1.
    wave = alpha[:,None]*shape(LAG[None,:],tau[:,None],bw)
    counterfeit = beta[:,None]*shape(LAG[None,:],fake[:,None],bw)
    clean = a[:,:,None]*wave[:,None,:]
    reflected = .35*alpha[:,None]*shape(LAG[None,:],tau[:,None]+.18,bw) if scenario == 'multipath' else np.zeros_like(wave)
    signal = gain[None,:,None]*(clean+b[None,:,None]*counterfeit[:,None,:]+a[:,:,None]*reflected[:,None,:])
    lagcov = .65*shape(LAG[:,None],LAG[None,:])+.35*np.eye(len(LAG))
    channelcov = .7**abs(np.arange(m)[:,None]-np.arange(m)[None,:]) if scenario == 'colored' else np.eye(m)
    lc = np.linalg.cholesky(lagcov); cc = np.linalg.cholesky(channelcov)
    def noise(n):
        v = (rng.normal(size=(n,m,len(LAG)))+1j*rng.normal(size=(n,m,len(LAG))))/np.sqrt(2)
        return .09*np.einsum('ab,nbt,st->nas',cc,v,lc)
    z = np.stack([signal, signal])*np.exp(1j*np.array([0,.45 if scenario == 'drift' else 0]))[:,None,None,None]+noise(12).reshape(2,6,m,len(LAG))
    calibration = gain[None,:]+.025*(rng.normal(size=(64,m))+1j*rng.normal(size=(64,m)))
    if scenario == 'cal_outlier': calibration[:8] += 2+3j
    obs = dict(z=z,a=a,lag=LAG.copy(),noise=noise(160),calibration=calibration)
    return obs, dict(wave=wave,delay=tau,alpha=alpha,exists=bool(amp))

def prepare(obs):
    return obs['z'].copy(), obs['a'].copy(), obs['noise'].copy()

def common_direction(z,a,noise):
    return np.linalg.svd(z.reshape(-1,z.shape[-2],z.shape[-1]).transpose(1,0,2).reshape(z.shape[-2],-1),full_matrices=False)[0][:,0]

def beam(z,a,b,noise):
    projected=a-(a@b.conj())[:,None]*b
    weights=projected/np.sum(abs(projected)**2,axis=1)[:,None]
    return np.einsum('pm,bpml->bpl',weights.conj(),z)

def temporal(y, lag):
    return y.mean(axis=0)

def recover(y,z,a,b,noise,lag):
    delay = lag[np.argmax(abs(y),axis=1)]
    h = template(lag[None,:],delay[:,None])
    amp = np.sum(h*y,axis=1)/np.sum(h*h,axis=1)
    return dict(wave=y,delay=delay,amp=amp,accepted=np.ones(len(a),bool))

def estimate(obs):
    z,a,noise = prepare(obs)
    b = common_direction(z,a,noise)
    y = temporal(beam(z,a,b,noise),obs['lag'])
    return recover(y,z,a,b,noise,obs['lag'])

def common_direction(z,a,noise):
    residual=z-a[None,:,:,None]*np.einsum('pm,bpml->bpl',a.conj(),z)[:,:,None,:]/a.shape[1]
    matrix=residual.transpose(2,0,1,3).reshape(a.shape[1],-1)
    return np.linalg.svd(matrix,full_matrices=False)[0][:,0]
def common_direction(z,a,noise):
    m=a.shape[1]
    projections=np.eye(m)[None,:,:]-a[:,:,None]*a[:,None,:].conj()/m
    residual=np.einsum('pmn,bpnl->bpml',projections,z)
    initial=np.linalg.svd(z.transpose(2,0,1,3).reshape(m,-1),full_matrices=False)[0][:,0]
    initial=initial/initial[0]
    def objective(x):
        direction=np.r_[1.,x[:m-1]+1j*x[m-1:]]
        pb=np.einsum('pmn,n->pm',projections,direction)
        amplitude=np.einsum('pm,bpml->bpl',pb.conj(),residual)/np.sum(abs(pb)**2,axis=1)[None,:,None]
        return float(np.sum(abs(residual-pb[None,:,:,None]*amplitude[:,:,None,:])**2))
    fit=minimize(objective,np.r_[initial[1:].real,initial[1:].imag],method='BFGS',options={'maxiter':40})
    direction=np.r_[1.,fit.x[:m-1]+1j*fit.x[m-1:]]
    return direction/np.linalg.norm(direction)
def prepare(obs):
    gain=obs['calibration'].mean(axis=0)
    return obs['z']/gain[None,None,:,None],obs['a'].copy(),obs['noise'].copy()
def prepare(obs):
    gain=obs['calibration'].mean(axis=0)
    return obs['z']/gain[None,None,:,None],obs['a'].copy(),obs['noise']/gain[None,:,None]

def beam(z,a,b,noise):
    covariance=np.einsum('nml,nkl->mk',noise,noise.conj())/(noise.shape[0]*noise.shape[2])
    weights=[]
    for sky in a:
        constraints=np.column_stack((sky,b))
        solved=np.linalg.solve(covariance,constraints)
        weights.append(solved@np.linalg.solve(constraints.conj().T@solved,np.array([1.,0.])))
    return np.einsum('pm,bpml->bpl',np.array(weights).conj(),z)
def recover(y,z,a,b,noise,lag):
    delay=lag[np.argmax(abs(y),axis=1)]
    h=template(lag[None,:],delay[:,None])
    amp=np.sum(h*y,axis=1)/np.sum(h*h,axis=1)
    return dict(wave=amp[:,None]*h,delay=delay,amp=amp,accepted=np.ones(len(a),bool))
def fit_single(row,lag,whiten):
    def profile(t):
        h=template(lag,t)@whiten
        c=np.vdot(h,row@whiten)/np.vdot(h,h)
        return float(np.sum(abs(row@whiten-c*h)**2))
    grid=np.linspace(-1.2,1.2,25)
    start=grid[np.argmin([profile(t) for t in grid])]
    delay=minimize_scalar(profile,bounds=(start-.1,start+.1),method='bounded').x
    h=template(lag,delay);hw=h@whiten
    amp=np.vdot(hw,row@whiten)/np.vdot(hw,hw)
    return delay,amp,h

def recover(y,z,a,b,noise,lag):
    fitted=[fit_single(row,lag,np.eye(len(lag))) for row in y]
    delay=np.array([v[0] for v in fitted]);amp=np.array([v[1] for v in fitted]);h=np.array([v[2] for v in fitted])
    return dict(wave=amp[:,None]*h,delay=delay,amp=amp,accepted=np.ones(len(a),bool))
def lag_whitener(noise):
    covariance=np.einsum('nml,nmk->lk',noise.conj(),noise).real/(noise.shape[0]*noise.shape[1])
    return np.linalg.inv(np.linalg.cholesky(covariance)).T

def recover(y,z,a,b,noise,lag):
    whiten=lag_whitener(noise)
    fitted=[fit_single(row,lag,whiten) for row in y]
    delay=np.array([v[0] for v in fitted]);amp=np.array([v[1] for v in fitted]);h=np.array([v[2] for v in fitted])
    return dict(wave=amp[:,None]*h,delay=delay,amp=amp,accepted=np.ones(len(a),bool))
def lag_whitener(noise):
    covariance=np.einsum('nml,nmk->lk',noise.conj(),noise).real/(noise.shape[0]*noise.shape[1])
    stationary=np.array([np.diag(covariance,k).mean() for k in range(noise.shape[-1])])
    covariance=stationary[abs(np.arange(len(stationary))[:,None]-np.arange(len(stationary))[None,:])]
    return np.linalg.inv(np.linalg.cholesky(covariance)).T
def fit_pair(row,sky,b,lag,whiten,start):
    def design(t):
        return np.column_stack(((sky[:,None]*template(lag,t[0])[None,:])@whiten,(b[:,None]*template(lag,t[1])[None,:])@whiten)).reshape(len(sky),2,len(lag)).transpose(0,2,1).reshape(-1,2)
    observed=(row@whiten).ravel()
    def objective(t):
        matrix=design(t);c=np.linalg.lstsq(matrix,observed,rcond=None)[0]
        return float(np.sum(abs(observed-matrix@c)**2))
    fit=minimize(objective,start,method='Nelder-Mead',options={'maxiter':90,'xatol':1e-6})
    matrix=design(fit.x);c=np.linalg.lstsq(matrix,observed,rcond=None)[0]
    return fit.x,c,matrix,observed

def recover(y,z,a,b,noise,lag):
    whiten=lag_whitener(noise);wave=[];delays=[];amplitudes=[]
    for p,row in enumerate(z.mean(axis=0)):
        start=fit_single(y[p],lag,whiten)[0]
        fake=fit_single(b.conj()@row,lag,whiten)[0]
        delay,c,matrix,observed=fit_pair(row,a[p],b,lag,whiten,[start,fake])
        wave.append(c[0]*template(lag,delay[0]));delays.append(delay[0]);amplitudes.append(c[0])
    return dict(wave=np.array(wave),delay=np.array(delays),amp=np.array(amplitudes),accepted=np.ones(len(a),bool))
def spatial_whitener(noise):
    covariance=np.einsum('nml,nkl->mk',noise,noise.conj())/(noise.shape[0]*noise.shape[2])
    covariance/=np.trace(covariance).real/len(covariance)
    return np.linalg.inv(np.linalg.cholesky(covariance))

def recover(y,z,a,b,noise,lag):
    whiten=lag_whitener(noise);space=spatial_whitener(noise);wave=[];delays=[];amplitudes=[]
    for p,row in enumerate(z.mean(axis=0)):
        start=fit_single(y[p],lag,whiten)[0];fake=fit_single(b.conj()@row,lag,whiten)[0]
        delay,c,matrix,observed=fit_pair(space@row,space@a[p],space@b,lag,whiten,[start,fake])
        wave.append(c[0]*template(lag,delay[0]));delays.append(delay[0]);amplitudes.append(c[0])
    return dict(wave=np.array(wave),delay=np.array(delays),amp=np.array(amplitudes),accepted=np.ones(len(a),bool))
def prepare(obs):
    calibration=obs['calibration']
    gain=np.median(calibration.real,axis=0)+1j*np.median(calibration.imag,axis=0)
    return obs['z']/gain[None,None,:,None],obs['a'].copy(),obs['noise']/gain[None,:,None]
_prepare_gain=prepare
def prepare(obs):
    z,a,noise=_prepare_gain(obs)
    phase=np.angle(np.vdot(z[0],z[1]))
    z[1]*=np.exp(-1j*phase)
    return z,a,noise
def conditional_score(matrix,observed,c,variance):
    true=matrix[:,0];fake=matrix[:,1]
    orthogonal=true-fake*np.vdot(fake,true)/np.vdot(fake,fake)
    information=np.vdot(orthogonal,orthogonal).real/variance
    return abs(c[0])**2*information,information

def recover(y,z,a,b,noise,lag):
    whiten=lag_whitener(noise);space=spatial_whitener(noise);wave=[];delays=[];amplitudes=[];accepted=[];scores=[]
    variance=float(np.mean(abs(np.einsum('mn,bnl->bml',space,noise)@whiten)**2)/len(z))
    for p,row in enumerate(z.mean(axis=0)):
        start=fit_single(y[p],lag,whiten)[0];fake=fit_single(b.conj()@row,lag,whiten)[0]
        delay,c,matrix,observed=fit_pair(space@row,space@a[p],space@b,lag,whiten,[start,fake])
        score,information=conditional_score(matrix,observed,c,variance)
        accepted.append(score>np.log(matrix.size)*3)
        wave.append(c[0]*template(lag,delay[0]));delays.append(delay[0]);amplitudes.append(c[0]);scores.append(score)
    return dict(wave=np.array(wave),delay=np.array(delays),amp=np.array(amplitudes),accepted=np.array(accepted),scores=np.array(scores))
_recover_evidence=recover
def recover(y,z,a,b,noise,lag):
    result=_recover_evidence(y,z,a,b,noise,lag)
    result['wave']*=result['accepted'][:,None]
    result['amp']*=result['accepted']
    return result
_recover_presence=recover
def recover(y,z,a,b,noise,lag):
    result=_recover_presence(y,z,a,b,noise,lag)
    whiten=lag_whitener(noise);space=spatial_whitener(noise)
    variance=float(np.mean(abs(np.einsum('mn,bnl->bml',space,noise)@whiten)**2)/len(z))
    sigma=[];residual=[]
    for p,row in enumerate(z.mean(axis=0)):
        fake=fit_single(b.conj()@row,lag,whiten)[0]
        delay,c,matrix,observed=fit_pair(space@row,space@a[p],space@b,lag,whiten,[result['delay'][p],fake])
        dtrue=(template(lag,delay[0]+1e-4)-template(lag,delay[0]-1e-4))/2e-4
        dfake=(template(lag,delay[1]+1e-4)-template(lag,delay[1]-1e-4))/2e-4
        derivative=((space@a[p])[:,None]*dtrue[None,:]@whiten).ravel()*c[0]
        nuisance=np.column_stack((matrix,((space@b)[:,None]*dfake[None,:]@whiten).ravel()*c[1]))
        derivative-=nuisance@np.linalg.lstsq(nuisance,derivative,rcond=None)[0]
        sigma.append(np.sqrt(variance/(2*np.vdot(derivative,derivative).real))*293.052256)
        residual.append(float(np.sum(abs(observed-matrix@c)**2)/variance))
    result['sigma_m']=np.array(sigma);result['residual']=np.array(residual)
    result['accepted'] &= result['sigma_m']*3<20
    return result
_recover_precision=recover
def recover(y,z,a,b,noise,lag):
    result=_recover_precision(y,z,a,b,noise,lag)
    from scipy.stats import chi2
    result['accepted'] &= 2*result['residual']<chi2.ppf(.999,2*a.shape[1]*len(lag)-6)
    return result
def prepare(obs):
    return _prepare_gain(obs)

def estimate(obs):
    z,a,noise=prepare(obs)
    b=common_direction(z[:1],a,noise)
    y=temporal(beam(z[:1],a,b,noise),obs['lag'])
    result=recover(y,z[:1],a,b,noise,obs['lag'])
    held=beam(z[1:],a,b,noise)[0]
    projected_noise=beam(noise[:,None,:,:],a,b,noise)
    whiten=lag_whitener(noise)
    h=template(obs['lag'][None,:],result['delay'][:,None])@whiten
    coefficient=np.sum(h.conj()*(held@whiten),axis=1)/np.sum(abs(h)**2,axis=1)
    variance=np.mean(abs(projected_noise@whiten)**2,axis=(0,2))
    score=abs(coefficient)**2*np.sum(abs(h)**2,axis=1)/variance
    result['accepted'] &= score>3*np.log(2*len(obs['lag']))
    result['validation_score']=score
    return result
_estimate_hold=estimate
def estimate(obs):
    validated=_estimate_hold(obs)
    z,a,noise=prepare(obs)
    z[1]*=np.exp(-1j*np.angle(np.vdot(z[0],z[1])))
    b=common_direction(z,a,noise)
    result=recover(temporal(beam(z,a,b,noise),obs['lag']),z,a,b,noise,obs['lag'])
    result['accepted'] &= validated['accepted']
    result['validation_score']=validated['validation_score']
    return result
_recover_gof=recover
def recover(y,z,a,b,noise,lag):
    result=_recover_gof(y,z,a,b,noise,lag)
    shrink=np.maximum(0.,1.-1./result['scores'])
    result['wave']*=shrink[:,None];result['amp']*=shrink
    return result
RECEIVER_BW=1.
def template(t,delay):
    return shape(t,delay,RECEIVER_BW)

_estimate_fixed_bandwidth=estimate
def estimate(obs):
    global RECEIVER_BW
    RECEIVER_BW=1.
    z,a,noise=prepare(obs);b=common_direction(z[:1],a,noise)
    rows=np.einsum('m,pml->pl',b.conj(),z[0]);lag=obs['lag'];whiten=lag_whitener(noise)
    starts=[fit_single(row,lag,whiten)[0] for row in rows]
    def objective(parameters):
        h=shape(lag[None,:],parameters[:-1,None],parameters[-1])@whiten
        observed=rows@whiten
        c=np.sum(h.conj()*observed,axis=1)/np.sum(abs(h)**2,axis=1)
        return float(np.sum(abs(observed-c[:,None]*h)**2))
    fit=minimize(objective,np.r_[starts,1.],bounds=[(-2,2)]*len(a)+[(.6,1.3)],method='L-BFGS-B')
    RECEIVER_BW=float(fit.x[-1])
    result=_estimate_fixed_bandwidth(obs);result['bandwidth']=RECEIVER_BW
    return result

_fit_two=fit_pair
def fit_pair(row,sky,b,lag,whiten,start):
    def design(t):
        columns=[sky[:,None]*template(lag,t[0])[None,:],b[:,None]*template(lag,t[1])[None,:],sky[:,None]*template(lag,t[0]+.18)[None,:]]
        return np.column_stack([(v@whiten).ravel() for v in columns])
    observed=(row@whiten).ravel()
    def objective(t):
        matrix=design(t);c=np.linalg.lstsq(matrix,observed,rcond=None)[0]
        return float(np.sum(abs(observed-matrix@c)**2))
    fit=minimize(objective,start,method='Nelder-Mead',options={'maxiter':90,'xatol':1e-6})
    matrix=design(fit.x);c=np.linalg.lstsq(matrix,observed,rcond=None)[0]
    return fit.x,c,matrix,observed

def conditional_score(matrix,observed,c,variance):
    true=matrix[:,0];nuisance=matrix[:,1:]
    orthogonal=true-nuisance@np.linalg.lstsq(nuisance,true,rcond=None)[0]
    information=np.vdot(orthogonal,orthogonal).real/variance
    return abs(c[0])**2*information,information
_fit_three=fit_pair
def fit_pair(row,sky,b,lag,whiten,start):
    two=_fit_two(row,sky,b,lag,whiten,start)
    three=_fit_three(row,sky,b,lag,whiten,start)
    def residual(fit):
        return float(np.sum(abs(fit[3]-fit[2]@fit[1])**2))
    # Unit complex variance is the single-block calibration scale; conservative for two-block means.
    return three if residual(two)-residual(three)>2*np.log(two[2].shape[0]) else two
_recover_shrink=recover
def recover(y,z,a,b,noise,lag):
    result=_recover_shrink(y,z,a,b,noise,lag)
    whiten=lag_whitener(noise)
    projected_noise=beam(noise[:,None,:,:],a,b,noise)
    variance=np.mean(abs(projected_noise@whiten)**2,axis=(0,2))/len(z)
    margins=[]
    for p,row in enumerate(y@whiten):
        def cost(delay):
            h=template(lag,delay)@whiten
            amplitude=np.vdot(h,row)/np.vdot(h,h)
            return float(np.sum(abs(row-amplitude*h)**2))
        center=result['delay'][p]
        margin=(min(cost(center-20/293.052256),cost(center+20/293.052256))-cost(center))/variance[p]
        margins.append(margin)
    result['boundary_margin']=np.array(margins)
    result['accepted'] &= result['boundary_margin']>9
    return result
_recover_boundary=recover
def recover(y,z,a,b,noise,lag):
    result=_recover_boundary(y,z,a,b,noise,lag)
    whiten=lag_whitener(noise);space=spatial_whitener(noise)
    reconstructed=[]
    for p,row in enumerate(z.mean(axis=0)):
        fake=fit_single(b.conj()@row,lag,whiten)[0]
        delay,c,matrix,observed=fit_pair(space@row,space@a[p],space@b,lag,whiten,[result['delay'][p],fake])
        cancellation=b[:,None]*c[1]*template(lag,delay[1])[None,:]
        # SIC followed by authentic beam retains residual noise, unlike template synthesis.
        reconstructed.append(a[p].conj()@(row-cancellation)/np.vdot(a[p],a[p]))
    result['wave']=np.array(reconstructed)
    return result
