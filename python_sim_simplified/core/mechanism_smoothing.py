"""Short eligibility gaps versus actual receiver tracking arcs."""
import numpy as np


def smooth_history(z, eligible):
    ranges=z['ranges'];carrier=z['carrier'];n,s=ranges.shape
    out=np.full_like(ranges,np.nan);ages=np.zeros((n,s),int)
    state=np.zeros(s);previous=np.zeros(s);age=np.zeros(s,int)
    last_time=np.full(s,-np.inf);last_arc=np.full(s,-1,int)
    for k,t in enumerate(z['time']):
        valid=eligible[k]&np.isfinite(ranges[k])&np.isfinite(carrier[k])
        prediction=state+carrier[k]-previous
        # At 10 ms output spacing this retains exactly one rejected epoch.
        continuous=valid&(age>0)&(t-last_time<=.025)&(z['arc'][k]==last_arc)&(abs(ranges[k]-prediction)<=50.)
        age[valid]=np.where(continuous[valid],age[valid]+1,1)
        weight=1/np.minimum(age[valid],20)
        state[valid]=np.where(continuous[valid],weight*ranges[k,valid]+(1-weight)*prediction[valid],ranges[k,valid])
        out[k,valid]=state[valid];ages[k,valid]=age[valid]
        previous[valid]=carrier[k,valid];last_time[valid]=t;last_arc[valid]=z['arc'][k,valid]
    return out,ages
