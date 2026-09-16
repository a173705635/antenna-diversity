"""Conditional calibrated navigation; measured support is not authentication."""
import numpy as np
from navigation_hundred import candidate as navigate,carrier_steps as screen_steps
from continuous_admission import admit,Gate
from scipy.stats import chi2
from scipy.optimize import minimize_scalar
from itertools import combinations


def weighted_screen(z):
    from run_gps_spp import receiver_reference
    ref,rot=receiver_reference();c=z['calibrated_carrier'];cn=z['calibrated_cn0'];lock=z['calibrated_lock']
    steps=np.full((len(c),3),np.nan);slips=np.zeros_like(c,bool)
    factor=(299792458/1575.42e6/(2*np.pi))**2/(2*.001)
    for k in range(1,len(c)):
        ids=np.flatnonzero(np.isfinite(c[k])&np.isfinite(c[k-1])&(lock[k]>=.5)&(lock[k-1]>=.5)&np.isfinite(cn[k])&np.isfinite(cn[k-1]))
        if len(ids)<6:continue
        states=z['satellite_positions'][k,ids]-ref
        h=np.column_stack((-states/np.linalg.norm(states,axis=1)[:,None]@rot.T,np.ones(len(ids))))
        orbit=np.linalg.norm(states,axis=1)-np.linalg.norm(z['satellite_positions'][k-1,ids]-ref,axis=1)
        y=c[k,ids]-c[k-1,ids]-orbit;var=factor*(10**(-cn[k,ids]/10)+10**(-cn[k-1,ids]/10))
        candidates=[]
        for omitted in [()]+list(combinations(range(len(ids)),1))+list(combinations(range(len(ids)),2)):
            keep=np.ones(len(ids),bool);keep[list(omitted)]=False
            if keep.sum()<6:continue
            a=h[keep]/np.sqrt(var[keep,None]);b=y[keep]/np.sqrt(var[keep])
            fit=np.linalg.lstsq(a,b,rcond=None)[0];cost=np.sum((b-a@fit)**2)
            if cost<=chi2.ppf(.999,keep.sum()-4):
                candidates.append((omitted,fit))
                if not omitted:break
        if candidates:
            minimum=min(len(item[0]) for item in candidates)
            candidates=[item for item in candidates if len(item[0])==minimum]
        if len(candidates)==1:
            omitted,fit=candidates[0];steps[k]=fit[:3];slips[k,ids[list(omitted)]]=True
        else:slips[k,ids]=True
    return steps,slips


def carrier_steps(z):
    from run_gps_spp import receiver_reference
    steps,slips=weighted_screen(z);ref,rot=receiver_reference()
    c=z['calibrated_carrier'];lock=z['calibrated_lock'];cn=z['calibrated_cn0']
    covs=np.full((len(c),3,3),np.nan)
    coefficients=np.zeros((len(c),3,c.shape[1]));noise=np.zeros_like(c)
    history=np.zeros((len(c),len(c),3,c.shape[1]))
    factor=(299792458/1575.42e6/(2*np.pi))**2/(2*.001)
    for k in range(1,len(c)):
        if z['time_s'][k]-z['time_s'][k-1]>.015:
            steps[k]=np.nan;coefficients[:]=0.;continue
        if not np.isfinite(steps[k]).all():coefficients[:]=0.;continue
        valid=np.isfinite(c[k])&np.isfinite(c[k-1])&(lock[k]>=.5)&(lock[k-1]>=.5)&np.isfinite(cn[k])&np.isfinite(cn[k-1])&~slips[k]
        if valid.sum()<6:steps[k]=np.nan;coefficients[:]=0.;continue
        states=z['satellite_positions'][k,valid]-ref
        h=np.column_stack((-states/np.linalg.norm(states,axis=1)[:,None]@rot.T,np.ones(valid.sum())))
        orbit=np.linalg.norm(states,axis=1)-np.linalg.norm(z['satellite_positions'][k-1,valid]-ref,axis=1)
        y=c[k,valid]-c[k-1,valid]-orbit
        var=factor*(10**(-cn[k,valid]/10)+10**(-cn[k-1,valid]/10))
        info=np.linalg.inv(h.T@(h/var[:,None]));mapping=(info@h.T/var)[:3]
        fit=info@(h.T@(y/var))
        base_range=np.linalg.norm(states,axis=1)
        for _ in range(4):
            shifted=states-fit[:3]@rot
            jacobian=np.column_stack((-shifted/np.linalg.norm(shifted,axis=1)[:,None]@rot.T,np.ones(valid.sum())))
            residual=y-(np.linalg.norm(shifted,axis=1)-base_range+fit[3])
            fit+=np.linalg.lstsq(jacobian/np.sqrt(var[:,None]),residual/np.sqrt(var),rcond=None)[0]
        info=np.linalg.inv(jacobian.T@(jacobian/var[:,None]));mapping=(info@jacobian.T/var)[:3]
        steps[k]=fit[:3]
        coefficients[k][:,valid]+=mapping;coefficients[k-1][:,valid]-=mapping
        noise[k,valid]=factor*10**(-cn[k,valid]/10);noise[k-1,valid]=factor*10**(-cn[k-1,valid]/10)
        covs[k]=np.einsum('tis,ts,tjs->ij',coefficients[:k+1],noise[:k+1],coefficients[:k+1])
        history[k]=coefficients
    return steps,slips,covs,history,noise


def covariance(z,k,kind,position=(0.,0.,0.)):
    from run_gps_spp import receiver_reference
    ref,rot=receiver_reference();ref=ref+np.asarray(position)@rot;cn=z['calibrated_cn0'][k]
    valid=np.isfinite(z['calibrated_'+kind][k])&np.isfinite(cn)&(z['calibrated_lock'][k]>=.5)
    if kind=='ranges':valid&=(cn>=25.)&(z['calibrated_gain'][k]>=.01)
    state=z['satellite_positions'][k,valid]-ref
    h=np.column_stack((-state/np.linalg.norm(state,axis=1)[:,None]@rot.T,np.ones(valid.sum())))
    sigma=5.*10**((45-cn[valid])/20) if kind=='ranges' else 299792458/1575.42e6/(2*np.pi)/np.sqrt(2*.001*10**(cn[valid]/10))
    return np.linalg.inv((h/sigma[:,None]).T@(h/sigma[:,None]))[:3,:3]


def code_position(z):
    fields=('satellite_positions','calibrated_ranges','calibrated_lock','calibrated_cn0','calibrated_gain')
    data=[z[name] for name in fields];gate=Gate(min_gain=.01)
    fit=admit(*data,gate);result=fit['position'];selected=fit['selected'].copy()
    for k in np.flatnonzero(~np.isfinite(result).all(axis=1)):
        choices=[]
        for count in (1,2):
            for omitted in combinations(range(z['calibrated_ranges'].shape[1]),count):
                sample=[value[k:k+1].copy() for value in data];sample[4][0,list(omitted)]=0.
                fit=admit(*sample,gate);trial=fit['position'][0]
                if np.isfinite(trial).all():choices.append((trial,fit['selected'][0]))
            if choices:break
        if len(choices)==1:result[k],selected[k]=choices[0]
    return result,selected


def calibrated(z,diagnostics=False):
    observed=dict(z)
    snr=z['calibrated_spatial'];positive=snr>0
    equivalent=np.full_like(snr,-np.inf)
    equivalent[positive]=10*np.log10(snr[positive]/.001)
    observed['calibrated_cn0']=np.minimum(z['calibrated_cn0'],equivalent)
    steps,slips,phase_cov,history,noise=carrier_steps(observed)
    observed['calibrated_carrier']=z['calibrated_carrier'].copy()
    observed['calibrated_carrier'][slips]=np.nan
    absolute,selected=code_position(observed)
    observed['calibrated_gain']=np.where(selected,observed['calibrated_gain'],0.)
    out=np.full_like(absolute,np.nan);posterior=np.full((len(out),3,3),np.nan);origin=np.zeros(3);displacement=np.zeros(3)
    for k in range(len(out)):
        available=np.isfinite(absolute[k]).all()
        if k==0 or not np.isfinite(out[k-1]).all():
            out[k]=absolute[k];displacement=np.zeros(3);anchor=k
            values=[];coeffs=[];code_cov=[]
            if available:
                origin=absolute[k].copy();values=[origin.copy()]
                coeffs=[np.zeros_like(history[k])];code_cov=[covariance(observed,k,'ranges',absolute[k])]
                weights=np.eye(3);g=np.zeros((3,noise.size));code_sigma=code_cov[0];posterior[k]=code_sigma
            continue
        if not np.isfinite(steps[k]).all():continue
        displacement+=steps[k];coefficient=history[k]-history[anchor]
        if available:
            values.append(absolute[k]-displacement);coeffs.append(coefficient.copy());code_cov.append(covariance(observed,k,'ranges',absolute[k]))
            values=values[-20:];coeffs=coeffs[-20:];code_cov=code_cov[-20:]
            m=len(values);g=np.concatenate([c.transpose(1,0,2).reshape(3,-1) for c in coeffs])
            phase_sigma=(g*noise.ravel())@g.T;sigma=phase_sigma.copy()
            for j,r in enumerate(code_cov):
                residual=values[j]-origin
                distance=np.sqrt(residual@np.linalg.solve(r+sigma[3*j:3*j+3,3*j:3*j+3],residual))
                sigma[3*j:3*j+3,3*j:3*j+3]+=r*max(1.,distance/3.)
            design=np.tile(np.eye(3),(m,1));inverse_design=np.linalg.solve(sigma,design)
            weights=np.linalg.solve(design.T@inverse_design,inverse_design.T)
            origin=weights@np.concatenate(values);code_sigma=sigma-phase_sigma
        out[k]=origin+displacement
        influence=coefficient.transpose(1,0,2).reshape(3,-1)-weights@g
        posterior[k]=weights@code_sigma@weights.T+(influence*noise.ravel())@influence.T
    return (out,posterior) if diagnostics else out


def capture_position(z,diagnostics=False):
    raw=dict(z)
    for field in ('ranges','carrier','lock','cn0','gain','spatial'):raw['calibrated_'+field]=z['raw_'+field]
    first,p1=calibrated(z,True);second,p2=calibrated(raw,True)
    agreed=np.isfinite(first).all(axis=1)&np.isfinite(second).all(axis=1)&(np.linalg.norm((first-second)[:,:2],axis=1)<=20.)
    out=(first+second)/2;out[~agreed]=np.nan
    cov=(p1+p2)/2;cov[~agreed]=np.nan
    return (out,cov) if diagnostics else out


def candidate(observed):
    joint=observed['joint'];capture=observed['capture']
    first,p1=calibrated(joint,True);second,p2=capture_position(capture,True)
    usable=np.isfinite(first).all(axis=1)&joint['joint_usable']
    out=np.where(usable[:,None],first,second)
    both=usable&np.isfinite(second).all(axis=1)
    for k in np.flatnonzero(both):
        i1=np.linalg.inv(p1[k]);i2=np.linalg.inv(p2[k])
        objective=lambda weight:np.trace(np.linalg.inv(weight*i1+(1-weight)*i2))
        optimum=minimize_scalar(objective,bounds=(0.,1.),method='bounded').x
        weight=min((0.,optimum,1.),key=objective)
        covariance_ci=np.linalg.inv(weight*i1+(1-weight)*i2)
        out[k]=covariance_ci@(weight*i1@first[k]+(1-weight)*i2@second[k])
    conflict=np.zeros(len(first),bool)
    for k in np.flatnonzero(both):
        delta=(first[k]-second[k])[:2];variance=2*(p1[k,:2,:2]+p2[k,:2,:2])
        conflict[k]=delta@np.linalg.solve(variance,delta)>chi2.ppf(.999,2)
    out[np.maximum.accumulate(conflict)]=np.nan
    return out
