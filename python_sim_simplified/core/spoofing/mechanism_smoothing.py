"""Short eligibility gaps versus actual receiver tracking arcs."""
import numpy as np
from .mechanism_phase import fit_carrier_displacement


def smooth_history(z, eligible, history_gap_s=.025, diagnostics=False):
    ranges=z['ranges'];carrier=z['carrier'];n,s=ranges.shape
    out=np.full_like(ranges,np.nan);ages=np.zeros((n,s),int)
    state=np.zeros(s);previous=np.zeros(s);age=np.zeros(s,int)
    last_time=np.full(s,-np.inf);last_arc=np.full(s,-1,int)
    if history_gap_s>.025:
        last_index=np.full(s,-1,int)
        if diagnostics:
            extensions=np.zeros((n,s),bool);extension_gap=np.full((n,s),np.nan)
    for k,t in enumerate(z['time']):
        valid=eligible[k]&np.isfinite(ranges[k])&np.isfinite(carrier[k])
        prediction=state+carrier[k]-previous
        # At 10 ms output spacing this retains exactly one rejected epoch.
        continuous=valid&(age>0)&(t-last_time<=.025)&(z['arc'][k]==last_arc)&(abs(ranges[k]-prediction)<=50.)
        if history_gap_s>.025:
            long=valid&(age>0)&(t-last_time>.025)&(t-last_time<=history_gap_s)&(z['arc'][k]==last_arc)&(abs(ranges[k]-prediction)<=50.)
            for j in np.unique(last_index[long]):
                targets=long&(last_index==j)
                common=valid&eligible[j]&np.isfinite(carrier[j])&(z['arc'][k]==z['arc'][j])
                if common.sum()<5:continue
                displacement=fit_carrier_displacement(
                    z['satellites'][k,common],z['satellites'][j,common],
                    carrier[k,common]-carrier[j,common])
                if displacement is not None:
                    continuous[targets]=True
                    if diagnostics:
                        extensions[k,targets]=True
                        extension_gap[k,targets]=t-z['time'][j]
        age[valid]=np.where(continuous[valid],age[valid]+1,1)
        weight=1/np.minimum(age[valid],20)
        state[valid]=np.where(continuous[valid],weight*ranges[k,valid]+(1-weight)*prediction[valid],ranges[k,valid])
        out[k,valid]=state[valid];ages[k,valid]=age[valid]
        previous[valid]=carrier[k,valid];last_time[valid]=t;last_arc[valid]=z['arc'][k,valid]
        if history_gap_s>.025:last_index[valid]=k
    if diagnostics and history_gap_s>.025:
        return out,ages,dict(history_extended=extensions,history_extension_gap_s=extension_gap)
    return out,ages
