"""Conditional calibrated navigation; measured support is not authentication."""
import numpy as np
from navigation_hundred import candidate as navigate,carrier_steps as screen_steps
from continuous_admission import admit,Gate


def carrier_steps(z):
    from run_gps_spp import receiver_reference
    steps,slips=screen_steps(z);ref,rot=receiver_reference()
    c=z['calibrated_carrier'];lock=z['calibrated_lock'];cn=z['calibrated_cn0']
    anchor=0;previous=np.zeros(3);lost=np.zeros(c.shape[1],bool)
    for k in range(1,len(steps)):
        if not np.isfinite(steps[k]).all():
            anchor=k;previous=np.zeros(3);lost[:]=False;continue
        lost|=~np.isfinite(c[k])|~np.isfinite(c[k-1])|slips[k]|~(lock[k]>=.5)|~(lock[k-1]>=.5)
        valid=~lost&np.isfinite(c[anchor])&(lock[k]>=.5)&(lock[anchor]>=.5)&np.isfinite(cn[k])&np.isfinite(cn[anchor])
        if valid.sum()<6:
            steps[k]=np.nan;anchor=k;previous=np.zeros(3);lost[:]=False;continue
        states=z['satellite_positions'][k,valid]-ref
        h=np.column_stack((-states/np.linalg.norm(states,axis=1)[:,None]@rot.T,np.ones(valid.sum())))
        orbit=np.linalg.norm(states,axis=1)-np.linalg.norm(z['satellite_positions'][anchor,valid]-ref,axis=1)
        y=c[k,valid]-c[anchor,valid]-orbit
        sigma=np.sqrt(10**(-cn[k,valid]/10)+10**(-cn[anchor,valid]/10))
        relative=np.linalg.lstsq(h/sigma[:,None],y/sigma,rcond=None)[0][:3]
        steps[k]=relative-previous;previous=relative
    return steps,slips


def covariance(z,k,kind):
    from run_gps_spp import receiver_reference
    ref,rot=receiver_reference();cn=z['calibrated_cn0'][k]
    valid=np.isfinite(z['calibrated_'+kind][k])&np.isfinite(cn)&(z['calibrated_lock'][k]>=.5)
    state=z['satellite_positions'][k,valid]-ref
    h=np.column_stack((-state/np.linalg.norm(state,axis=1)[:,None]@rot.T,np.ones(valid.sum())))
    sigma=5.*10**((45-cn[valid])/20) if kind=='ranges' else 299792458/1575.42e6/(2*np.pi)/np.sqrt(2*.001*10**(cn[valid]/10))
    return np.linalg.inv((h/sigma[:,None]).T@(h/sigma[:,None]))[:3,:3]


def calibrated(z):
    observed=dict(z)
    snr=z['calibrated_spatial'];positive=snr>0
    equivalent=np.full_like(snr,-np.inf)
    equivalent[positive]=10*np.log10(snr[positive]/.001)
    observed['calibrated_cn0']=np.minimum(z['calibrated_cn0'],equivalent)
    steps,slips=carrier_steps(observed)
    observed['calibrated_carrier']=z['calibrated_carrier'].copy()
    observed['calibrated_carrier'][slips]=np.nan
    absolute=admit(observed['satellite_positions'],observed['calibrated_ranges'],observed['calibrated_lock'],observed['calibrated_cn0'],observed['calibrated_gain'],Gate(min_gain=.01))['position']
    out=np.full_like(absolute,np.nan);p=np.eye(3);origin=np.zeros(3);displacement=np.zeros(3)
    for k in range(len(out)):
        available=np.isfinite(absolute[k]).all()
        if k==0 or not np.isfinite(out[k-1]).all():
            out[k]=absolute[k];displacement=np.zeros(3)
            if available:
                origin=absolute[k].copy();p=covariance(observed,k,'ranges')+covariance(observed,k,'carrier')
            continue
        if not np.isfinite(steps[k]).all():continue
        displacement+=steps[k];q=covariance(observed,k,'carrier')
        if available:
            r=covariance(observed,k,'ranges')+q
            innovation=absolute[k]-displacement-origin
            distance=np.sqrt(innovation@np.linalg.solve(p+r,innovation))
            scale=max(1.,distance/3.)
            r*=scale;gain=np.linalg.solve((p+r).T,p.T).T
            origin+=gain@(absolute[k]-displacement-origin);p=(np.eye(3)-gain)@p
        out[k]=origin+displacement
    return out


def candidate(observed):
    joint=observed['joint'];capture=observed['capture']
    first=calibrated(joint);second=navigate(capture)
    usable=np.isfinite(first).all(axis=1)&joint['joint_usable']
    out=np.where(usable[:,None],first,second)
    both=usable&np.isfinite(second).all(axis=1)
    out[both]=(first[both]+second[both])/2
    conflict=both&(np.linalg.norm((first-second)[:,:2],axis=1)>20.)
    out[np.maximum.accumulate(conflict)]=np.nan
    return out
