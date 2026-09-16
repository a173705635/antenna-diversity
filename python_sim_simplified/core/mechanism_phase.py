"""Measured carrier displacement from a recently published position."""
import numpy as np


def phase_prediction(z,valid,positions,k):
    from .gps_simulation import receiver_reference
    reference,rotation=receiver_reference()
    for j in range(k-1,max(-1,k-3),-1):
        if z['time'][k]-z['time'][j]>.025 or not np.isfinite(positions[j]).all():continue
        common=valid[k]&valid[j]&np.isfinite(z['carrier'][k])&np.isfinite(z['carrier'][j])&(z['arc'][k]==z['arc'][j])
        if common.sum()<5:continue
        states=z['satellites'][k,common]-reference;previous=z['satellites'][j,common]-reference
        h=np.column_stack((-(states/np.linalg.norm(states,axis=1)[:,None])@rotation.T,np.ones(common.sum())))
        if np.linalg.cond(h)>1000:continue
        measured=z['carrier'][k,common]-z['carrier'][j,common]-(np.linalg.norm(states,axis=1)-np.linalg.norm(previous,axis=1))
        displacement=np.linalg.lstsq(h,measured,rcond=None)[0]
        residual=h@displacement-measured
        variance=max(.02**2,float(residual@residual)/(common.sum()-4))
        covariance=variance*np.linalg.inv(h.T@h)
        if np.sqrt(np.linalg.eigvalsh(covariance[:2,:2])[-1])<=.5 and np.max(abs(residual))<=.1:
            return positions[j]+displacement[:3]
    return None
