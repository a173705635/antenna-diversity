"""Research proposals; truth is consumed only by the external scorer."""
import numpy as np
from carrier_admission import Rule,evaluate
from run_gps_spp import receiver_reference


def carrier_steps(z):
    """Observed motion increments and per-star discontinuities that terminate smoothing arcs."""
    carrier=z['calibrated_carrier'];lock=z['calibrated_lock'];ref,rot=receiver_reference()
    steps=np.full((len(carrier),3),np.nan);slips=np.zeros_like(carrier,bool)
    for k in range(1,len(carrier)):
        valid=np.isfinite(carrier[k])&np.isfinite(carrier[k-1])&(lock[k]>=.5)&(lock[k-1]>=.5)
        if valid.sum()<6:continue
        u=z['satellite_positions'][k,valid]-ref
        h=np.column_stack((-u/np.linalg.norm(u,axis=1)[:,None]@rot.T,np.ones(valid.sum())))
        # Satellite motion is known ephemeris, not receiver truth.
        orbit=np.linalg.norm(z['satellite_positions'][k,valid]-ref,axis=1)-np.linalg.norm(z['satellite_positions'][k-1,valid]-ref,axis=1)
        innovation=carrier[k,valid]-carrier[k-1,valid]-orbit
        step=np.linalg.lstsq(h,innovation,rcond=None)[0]
        # Iteration 09: carrier increments must fit one receiver displacement/clock.
        residual=innovation-h@step
        if np.sqrt(np.mean(residual**2))>.1:
            # Iteration 10: uniquely identifiable single-star slip is excluded for this increment.
            alternatives=[]
            for j in range(len(innovation)):
                keep=np.arange(len(innovation))!=j
                if keep.sum()<6:continue
                trial=np.linalg.lstsq(h[keep],innovation[keep],rcond=None)[0]
                if np.sqrt(np.mean((innovation[keep]-h[keep]@trial)**2))<=.1:
                    alternatives.append((j,trial))
            if len(alternatives)!=1:
                slips[k,valid]=True;continue
            j,step=alternatives[0];slips[k,np.flatnonzero(valid)[j]]=True
        steps[k]=step[:3]
    return steps,slips


def candidate(z):
    steps,slips=carrier_steps(z)
    # Integration correction: a repaired displacement does not repair the Hatch state.
    observed=dict(z);observed['calibrated_carrier']=z['calibrated_carrier'].copy()
    observed['calibrated_carrier'][slips]=np.nan
    rule=Rule(window=5,mature=1,divergence=np.inf)
    absolute=evaluate(observed,'calibrated',rule)['position']
    raw=evaluate(z,'raw',rule)['position']
    agreed=np.isfinite(raw).all(axis=1)&(np.linalg.norm((raw-absolute)[:,:2],axis=1)<=20.)
    absolute=(absolute+raw)/2;absolute[~agreed]=np.nan
    out=np.full_like(absolute,np.nan)
    for k in range(len(out)):
        if k==0 or not np.isfinite(out[k-1]).all():
            out[k]=absolute[k];continue
        if not np.isfinite(steps[k]).all():continue
        out[k]=out[k-1]+steps[k]
        # Iteration 07: complementary absolute correction, 0.5 s time constant.
        if np.isfinite(absolute[k]).all():
            out[k]+=(1-np.exp(-(z['time_s'][k]-z['time_s'][k-1])/.5))*(absolute[k]-out[k])
    return out
