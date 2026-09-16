"""Warm-start observation-domain navigation. No RF/array gain claim."""
import numpy as np


def estimate(z):
    h=z["design"]; code=z["code"]; phase=z["phase"]; lock=z["lock"]; cn=z["cn0"]
    n=len(code); out=np.full((n,3),np.nan); states=np.full((n,4),np.nan)
    initialized=None; origin=None; anchor=None; alarm=False; displacement=np.zeros(4)
    history=[]; previous=None; smooth=None; last_output=None; gaps=0
    cmc_anchor=None; cmc_count=0; rate_history=[]; recovered=0
    for k in range(n):
        valid=lock[k]&np.isfinite(code[k])
        # VALID
        if valid.sum()<5:
            gaps+=1
            # GAP
            continue
        a=h[valid]; y=code[k,valid]; weights=np.ones(len(y))
        # WEIGHTS
        fit=np.linalg.lstsq(a*weights[:,None],y*weights,rcond=None)[0]
        residual=y-a@fit
        for _ in range(3):
            robust=np.minimum(1.,8./np.maximum(np.abs(y-a@fit),1e-12))
            fit=np.linalg.lstsq(a*np.sqrt(robust[:,None]),y*np.sqrt(robust),rcond=None)[0]
        residual=y-a@fit
        # FIT
        states[k]=fit
        if initialized is None:
            history.append(fit)
            if len(history)<30:continue
            recent=np.array(history[-30:])
            if np.max(np.std(recent[:,:2],axis=0))>12:continue
            initialized=k;origin=np.mean(recent,axis=0);anchor=phase[k].copy();cmc_anchor=code[k]-phase[k]
            smooth=origin.copy(); previous=fit.copy()
        step=None
        if k>0:
            both=valid&lock[k-1]&np.isfinite(phase[k])&np.isfinite(phase[k-1])
            if both.sum()>=5:
                b=h[both]; delta=phase[k,both]-phase[k-1,both]
                step=np.linalg.lstsq(b,delta,rcond=None)[0]
                phase_residual=delta-b@step
                phase_keep=np.abs(phase_residual)<.15
                if phase_keep.sum()>=5:
                    step=np.linalg.lstsq(b[phase_keep],delta[phase_keep],rcond=None)[0]
                else:step=None
                # STEP
        candidate=fit.copy()
        if step is not None:
            smooth=.98*(smooth+step)+.02*fit
            candidate=smooth.copy()
        else:
            if last_output is not None:
                smooth=.9*(last_output+warm_velocity*(z["time"][k]-z["time"][k-1]))+.1*fit
                candidate=smooth.copy()
            else:smooth=fit.copy()
        # NAVIGATE
        cmc=(code[k]-phase[k])-cmc_anchor
        cmc_fit=np.linalg.lstsq(h[valid],cmc[valid],rcond=None)[0]
        for _ in range(3):
            w=np.minimum(1.,6./np.maximum(np.abs(cmc[valid]-h[valid]@cmc_fit),1e-12))
            cmc_fit=np.linalg.lstsq(h[valid]*np.sqrt(w[:,None]),cmc[valid]*np.sqrt(w),rcond=None)[0]
        code_sigma=3.*10**((45-np.median(cn[k,valid]))/20)
        cmc_sigma=code_sigma*np.sqrt(np.trace(np.linalg.inv(h[valid].T@h[valid])[:2,:2]))
        cmc_count=cmc_count+1 if np.linalg.norm(cmc_fit[:2])>18+2*cmc_sigma else 0
        if cmc_count>=3:alarm=True
        if step is not None and np.linalg.norm(step[:2])>8:alarm=True
        if k==initialized:
            warm_velocity=np.linalg.lstsq(h,(phase[k]-phase[k-20])/(z["time"][k]-z["time"][k-20]),rcond=None)[0]
        if step is not None:
            velocity=step/(z["time"][k]-z["time"][k-1])
            rate_history.append(velocity[:2]);rate_history=rate_history[-3:]
            if len(rate_history)==3:
                mean_velocity=np.mean(rate_history,axis=0)
                if np.linalg.norm(mean_velocity-warm_velocity[:2])>1.:alarm=True
                else:warm_velocity[:2]=.98*warm_velocity[:2]+.02*mean_velocity
        # SCREEN
        if not alarm:out[k]=candidate[:3]
        previous=fit.copy();last_output=candidate.copy();gaps=0
    return {"position":out,"initialized_epoch":initialized,"states":states}
