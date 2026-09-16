"""Measured startup prompt research; no truth or transmitted symbols in estimator."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from calibration_candidate import calibrate as previous_calibrate


def align_prompts(z, method):
    out=np.empty_like(z);time=np.arange(z.shape[2])*.001
    for s in range(z.shape[1]):
        u,_,_=np.linalg.svd(z[:,s],full_matrices=False)
        scalar=u[:,0].conj()@z[:,s]
        if method in ('svd','reference'):
            phase=np.angle(scalar)+(np.angle(u[0,0]) if method=='reference' else 0.)
        elif method=='sign':
            axis=.5*np.angle(np.sum(scalar**2))
            phase=np.angle(np.where((scalar*np.exp(-1j*axis)).real>=0,1.,-1.))
        else:
            objective=lambda f:-abs(np.sum(scalar**2*np.exp(-4j*np.pi*f*time)))**2
            grid=np.arange(-30.,31.)
            center=grid[np.argmin([objective(f) for f in grid])]
            frequency=minimize_scalar(objective,bounds=(center-1,center+1),method='bounded').x
            rotation=np.exp(-2j*np.pi*frequency*time)
            axis=.5*np.angle(np.sum((scalar*rotation)**2))
            signs=np.where((scalar*rotation*np.exp(-1j*axis)).real>=0,1.,-1.)
            phase=2*np.pi*frequency*time+np.angle(signs)
        out[:,s]=z[:,s]*np.exp(-1j*phase)
    return out


def calibrate(z,a,revision=125):
    if revision==125:return previous_calibrate(z,a)
    if revision>=131:return joint_fit(z,a,revision)
    method={126:'svd',127:'sign'}.get(revision,'square')
    aligned=align_prompts(z,method)
    if revision==126:
        z=aligned.mean(axis=2,keepdims=True)
    elif revision==127:
        z=aligned.reshape(*aligned.shape[:2],-1,10).mean(axis=3)
    else:
        z=aligned.reshape(*aligned.shape[:2],-1,10).mean(axis=3)
    g,info=previous_calibrate(z,a)
    if revision==129:
        g=np.exp(1j*np.angle(g))
    if revision==130:
        # Explicit nominal-phase prior; changes the admissible hardware model.
        g=np.exp(.5j*np.angle(g))
    return g,info


def joint_fit(z,a,revision):
    if revision>=136:return supported_fit(z,a,revision)
    m,s,k=z.shape
    aligned=align_prompts(z,'sign');y=aligned.mean(axis=2)
    g0,old=previous_calibrate(z,a)
    _,vectors=np.linalg.eigh(y@y.conj().T);v0=vectors[:,-1]/vectors[0,-1]
    sigma=np.sqrt(np.mean(abs(aligned-y[:,:,None])**2,axis=(0,2))/k)
    weights=np.ones(s) if revision==131 else 1/sigma
    def unpack(x):
        g=np.exp(1j*np.r_[0.,x[:m-1]])
        v=np.r_[1.,x[m-1:2*(m-1)]+1j*x[2*(m-1):]];v/=np.linalg.norm(v)
        return g,v
    def residual(x):
        g,v=unpack(x);remaining=[]
        for j in range(s):
            basis=np.column_stack((g*a[:,j],v))
            r=y[:,j]-basis@np.linalg.lstsq(basis,y[:,j],rcond=None)[0]
            remaining.append(r*weights[j])
        r=np.array(remaining).ravel();return np.r_[r.real,r.imag]
    start=np.r_[np.angle(g0[1:]),v0[1:].real,v0[1:].imag]
    fit=least_squares(residual,start,max_nfev=100,loss='soft_l1' if revision>=134 else 'linear')
    if revision>=133:
        nominal=start.copy();nominal[:m-1]=0
        other=least_squares(residual,nominal,max_nfev=100,loss='soft_l1' if revision>=134 else 'linear')
        if other.cost<fit.cost:fit=other
    g,v=unpack(fit.x);singular=np.linalg.svd(fit.jac,compute_uv=False)
    usable=old['usable']
    if revision>=135:usable=m>=3 and singular[-1]/singular[0]>.005
    return g,dict(usable=bool(usable),cost=float(fit.cost),jacobian_ratio=float(singular[-1]/singular[0]),
                  source_real=v.real.tolist(),source_imag=v.imag.tolist())


def supported_fit(z,a,revision):
    m,s,k=z.shape
    train=z if revision==136 else z[:,:,:k//2]
    test=z if revision==136 else z[:,:,k//2:]
    g,info=joint_fit(train,a,134)
    v=np.array(info['source_real'])+1j*np.array(info['source_imag'])
    aligned=align_prompts(test,'sign');y=aligned.mean(axis=2)
    noise=np.mean(abs(aligned-y[:,:,None])**2,axis=(0,2))/test.shape[2]
    support=[];power=[];two_residual=[]
    for j in range(s):
        basis=np.column_stack((g*a[:,j],v));inverse=np.linalg.pinv(basis)
        coeff=inverse@y[:,j]
        snr=abs(coeff[0])**2/(noise[j]*np.sum(abs(inverse[0])**2))
        power.append(float(snr));support.append(snr>9.)
        two_residual.extend((y[:,j]-basis@coeff)/np.sqrt(noise[j]))
    usable=m>=3 and sum(support)>=4
    if revision>=138:
        second,detail=joint_fit(test,a,134)
        disagreement=float(np.max(abs(np.degrees(np.angle(g/second)))))
        usable=usable and disagreement<10.
        info['split_disagreement_deg']=disagreement
    if revision>=139:
        def clean_residual(phase):
            desired=np.exp(1j*np.r_[0.,phase])[:,None]*a
            coefficient=np.sum(desired.conj()*y,axis=0)/m
            r=((y-desired*coefficient)/np.sqrt(noise)[None,:]).ravel()
            return np.r_[r.real,r.imag]
        clean=least_squares(clean_residual,np.zeros(m-1))
        count=2*m*s
        clean_bic=2*clean.cost+(m-1+2*s)*np.log(count)
        two_bic=np.sum(abs(np.array(two_residual))**2)+(3*(m-1)+4*s)*np.log(count)
        info.update(clean_bic=float(clean_bic),two_bic=float(two_bic))
        if clean_bic<two_bic:
            g=np.exp(1j*np.r_[0.,clean.x]);usable=m>=3
            info['selected_model']='sky'
        else:info['selected_model']='sky_common'
        if revision>=140:
            white=y/np.sqrt(noise)[None,:];u,sv,vh=np.linalg.svd(white,full_matrices=False)
            common_bic=float(np.sum(sv[1:]**2)+(2*(m-1)+2*s)*np.log(count))
            info['common_bic']=common_bic
            if common_bic<min(clean_bic,two_bic):usable=False;info['selected_model']='common_only'
    info.update(usable=bool(usable),sky_snr=power,supported_prns=int(sum(support)))
    return g,info


def complex_calibrate(z,a):
    """Relative complex diagonal gains, with reference gain fixed to one."""
    m,s,k=z.shape;aligned=align_prompts(z,'reference');y=aligned.mean(axis=2)
    initial,old=previous_calibrate(z,a)
    _,vectors=np.linalg.eigh(y@y.conj().T);v0=vectors[:,-1]/vectors[0,-1]
    def unpack(x):
        g=np.r_[1.,x[:m-1]+1j*x[m-1:2*(m-1)]]
        v=np.r_[1.,x[2*(m-1):3*(m-1)]+1j*x[3*(m-1):]];v/=np.linalg.norm(v)
        return g,v
    def residual(x):
        g,v=unpack(x);r=[]
        for j in range(s):
            basis=np.column_stack((g*a[:,j],v))
            r.extend(y[:,j]-basis@np.linalg.lstsq(basis,y[:,j],rcond=None)[0])
        r=np.array(r);return np.r_[r.real,r.imag]
    fit=least_squares(residual,np.r_[initial[1:].real,initial[1:].imag,v0[1:].real,v0[1:].imag],max_nfev=200)
    g,v=unpack(fit.x)
    noise=np.mean(abs(aligned-y[:,:,None])**2,axis=(0,2))/k
    supported=[]
    for j in range(s):
        inverse=np.linalg.pinv(np.column_stack((g*a[:,j],v)));alpha=(inverse@y[:,j])[0]
        supported.append(float(abs(alpha)**2/(noise[j]*np.sum(abs(inverse[0])**2))))
    count=2*m*s
    common=np.linalg.svd(y,compute_uv=False)
    # Fixed sample-mean noise model; this is a conditional score, not calibrated authentication.
    two_bic=2*fit.cost/np.mean(noise)+(4*(m-1)+4*s)*np.log(count)
    common_bic=np.sum(common[1:]**2)/np.mean(noise)+(2*(m-1)+2*s)*np.log(count)
    def clean_residual(x):
        gain=np.r_[1.,x[:m-1]+1j*x[m-1:]];desired=gain[:,None]*a
        alpha=np.sum(desired.conj()*y,axis=0)/np.sum(abs(desired)**2,axis=0)
        r=(y-desired*alpha).ravel();return np.r_[r.real,r.imag]
    clean=least_squares(clean_residual,np.r_[np.ones(m-1),np.zeros(m-1)])
    clean_bic=2*clean.cost/np.mean(noise)+(2*(m-1)+2*s)*np.log(count)
    info=dict(usable=bool(m>=3 and sum(q>9 for q in supported)>=4 and two_bic<common_bic),
              selected_model='sky_common' if two_bic<common_bic else 'common_only',
              sky_snr=supported,supported_prns=int(sum(q>9 for q in supported)),
              source_real=v.real.tolist(),source_imag=v.imag.tolist(),two_bic=float(two_bic),common_bic=float(common_bic),
              gain_real=g.real.tolist(),gain_imag=g.imag.tolist())
    info['clean_bic']=float(clean_bic)
    if clean_bic<min(two_bic,common_bic):
        g=np.r_[1.,clean.x[:m-1]+1j*clean.x[m-1:]]
        info.update(selected_model='sky',usable=bool(m>=3),gain_real=g.real.tolist(),gain_imag=g.imag.tolist())
    return g,info
