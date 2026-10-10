"""Measured carrier displacement from a recently published position."""
import numpy as np


def fit_carrier_displacement(current_satellites,previous_satellites,carrier_change,
                             return_residual=False):
    """Fit local displacement and clock change from >=5 same-arc carriers."""
    from .gps_simulation import receiver_reference
    reference,rotation=receiver_reference()
    states=current_satellites-reference;previous=previous_satellites-reference
    h=np.column_stack((-(states/np.linalg.norm(states,axis=1)[:,None])@rotation.T,np.ones(len(states))))
    if np.linalg.cond(h)>1000:
        return (None,np.full(len(states),np.nan)) if return_residual else None
    measured=carrier_change-(np.linalg.norm(states,axis=1)-np.linalg.norm(previous,axis=1))
    displacement=np.linalg.lstsq(h,measured,rcond=None)[0]
    residual=h@displacement-measured
    variance=max(.02**2,float(residual@residual)/(len(states)-4))
    covariance=variance*np.linalg.inv(h.T@h)
    if np.sqrt(np.linalg.eigvalsh(covariance[:2,:2])[-1])<=.5 and np.max(abs(residual))<=.1:
        return (displacement,residual) if return_residual else displacement
    return (None,residual) if return_residual else None


def phase_prediction(z,valid,positions,k,return_anchor=False):
    for j in range(k-1,max(-1,k-3),-1):
        if z['time'][k]-z['time'][j]>.025 or not np.isfinite(positions[j]).all():continue
        common=valid[k]&valid[j]&np.isfinite(z['carrier'][k])&np.isfinite(z['carrier'][j])&(z['arc'][k]==z['arc'][j])
        if common.sum()<5:continue
        displacement=fit_carrier_displacement(
            z['satellites'][k,common],z['satellites'][j,common],
            z['carrier'][k,common]-z['carrier'][j,common])
        if displacement is not None:
            prediction=positions[j]+displacement[:3]
            return (prediction,j) if return_anchor else prediction
    return (None,-1) if return_anchor else None


def phase_prediction_leave_one_out(z,valid,positions,k):
    """Use one excluded PRN only after both recent full fits have failed.

    Every passing leave-one-out fit must implicate the same PRN. The original
    geometry, uncertainty, and residual gates remain in the carrier fit.
    """
    stars=valid.shape[1]
    details=dict(lag_epochs=0,excluded_prn=0,
                 full_common_count=np.zeros(2,int),
                 full_residual_m=np.full((2,stars),np.nan),
                 candidate_common_count=0,
                 candidate_residual_m=np.full(stars,np.nan))
    eligible=[]
    for lag in (1,2):
        j=k-lag
        if j<0 or z['time'][k]-z['time'][j]>.025 or not np.isfinite(positions[j]).all():
            continue
        common=valid[k]&valid[j]&np.isfinite(z['carrier'][k])&np.isfinite(z['carrier'][j])&(z['arc'][k]==z['arc'][j])
        indices=np.flatnonzero(common)
        details['full_common_count'][lag-1]=len(indices)
        if len(indices)<5:continue
        displacement,residual=fit_carrier_displacement(
            z['satellites'][k,common],z['satellites'][j,common],
            z['carrier'][k,common]-z['carrier'][j,common],return_residual=True)
        details['full_residual_m'][lag-1,common]=residual
        if displacement is not None:
            details['lag_epochs']=lag
            return positions[j]+displacement[:3],details
        if len(indices)>=6:eligible.append((lag,j,indices))

    passing=[]
    for lag,j,indices in eligible:
        for excluded in indices:
            retained=indices[indices!=excluded]
            displacement,residual=fit_carrier_displacement(
                z['satellites'][k,retained],z['satellites'][j,retained],
                z['carrier'][k,retained]-z['carrier'][j,retained],
                return_residual=True)
            if displacement is not None:
                passing.append((excluded,lag,positions[j]+displacement[:3],
                                retained,residual))
    if passing and len({candidate[0] for candidate in passing})==1:
        excluded,lag,prediction,retained,residual=passing[0]
        details['lag_epochs']=lag
        details['excluded_prn']=int(excluded+1)
        details['candidate_common_count']=len(retained)
        details['candidate_residual_m'][retained]=residual
        return prediction,details
    return None,details


def phase_prediction_long_anchor(z,valid,positions,k,short_excluded_prn):
    """Offer one exact-time 1 s carrier bridge from this trajectory's anchor.

    A passing full long fit or ambiguous leave-one-out keeps the short bridge.
    The only accepted long fit excludes one uniquely implicated PRN.
    """
    stars=valid.shape[1]
    details=dict(anchor_index=-1,elapsed_s=np.nan,excluded_prn=0,
                 reason='no_anchor',full_common_count=0,
                 full_residual_m=np.full(stars,np.nan),
                 candidate_common_count=0,
                 candidate_residual_m=np.full(stars,np.nan),
                 candidate_prediction_m=np.full(3,np.nan))
    if k==0:return None,details
    target=z['time'][k]-1.
    j=int(np.argmin(abs(z['time'][:k]-target)))
    elapsed=float(z['time'][k]-z['time'][j])
    if abs(elapsed-1.)>1e-9 or not np.isfinite(positions[j]).all():
        return None,details
    details['anchor_index']=j
    details['elapsed_s']=elapsed
    common=(valid[k]&valid[j]&np.isfinite(z['carrier'][k])
            &np.isfinite(z['carrier'][j])&(z['arc'][k]==z['arc'][j]))
    indices=np.flatnonzero(common)
    details['full_common_count']=len(indices)
    if len(indices)<6:
        details['reason']='too_few'
        return None,details
    displacement,residual=fit_carrier_displacement(
        z['satellites'][k,indices],z['satellites'][j,indices],
        z['carrier'][k,indices]-z['carrier'][j,indices],return_residual=True)
    details['full_residual_m'][indices]=residual
    if displacement is not None:
        details['reason']='full_pass'
        return None,details
    passing=[]
    for excluded in indices:
        retained=indices[indices!=excluded]
        candidate,candidate_residual=fit_carrier_displacement(
            z['satellites'][k,retained],z['satellites'][j,retained],
            z['carrier'][k,retained]-z['carrier'][j,retained],
            return_residual=True)
        if candidate is not None:
            passing.append((excluded,retained,candidate,candidate_residual))
    if not passing:
        details['reason']='no_candidate'
        return None,details
    if len({candidate[0] for candidate in passing})!=1:
        details['reason']='ambiguous'
        return None,details
    excluded,retained,candidate,residual=passing[0]
    details['excluded_prn']=int(excluded+1)
    details['candidate_common_count']=len(retained)
    details['candidate_residual_m'][retained]=residual
    details['candidate_prediction_m']=positions[j]+candidate[:3]
    if short_excluded_prn not in (0,details['excluded_prn']):
        details['reason']='short_conflict'
        return None,details
    details['reason']='used'
    return details['candidate_prediction_m'],details
