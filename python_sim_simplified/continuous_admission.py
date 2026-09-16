"""Quality-weighted research admission, not an authenticated protection bound."""
from dataclasses import dataclass
import numpy as np
from positioning import solve_position
from run_gps_spp import receiver_reference


@dataclass
class Gate:
    min_gain: float=.05
    min_lock: float=.5
    min_cn0: float=25.
    sigma45: float=5.
    residual_z: float=3.
    separation_z: float=5.
    horizontal_sigma: float=20.


def admit(satellites,ranges,lock,cn0,gain,gate):
    n,s=ranges.shape;p=np.full((n,3),np.nan);selected=np.zeros((n,s),bool)
    scores=np.full((n,3),np.nan);ref,rot=receiver_reference()
    for k in range(n):
        valid=np.isfinite(ranges[k])&(lock[k]>=gate.min_lock)&(cn0[k]>=gate.min_cn0)&(gain[k]>=gate.min_gain)
        selected[k]=valid
        if valid.sum()<6:continue
        states=satellites[k,valid];rho=ranges[k,valid]
        sigma=gate.sigma45*10**((45-cn0[k,valid])/20)
        u=states-ref;h=np.column_stack((-u/np.linalg.norm(u,axis=1)[:,None],np.ones(len(rho))))
        if np.linalg.cond(h/sigma[:,None])>1000:continue
        fit=solve_position(states,rho,sigma_m=sigma)
        cov=np.linalg.inv((h/sigma[:,None]).T@(h/sigma[:,None]))
        horizontal=rot[:2]@cov[:3,:3]@rot[:2].T
        hs=np.sqrt(np.linalg.eigvalsh(horizontal)[-1])
        z=np.sqrt(np.sum((fit['residual_m']/sigma)**2)/(len(rho)-4))
        # Linearized deleted-observation shift and covariance, not truth error.
        influence=cov@h.T/sigma**2
        leverage=np.sum(h*influence.T,axis=1)
        delta=influence*fit['residual_m']/(1-leverage)
        delta_h=rot[:2]@delta[:3]
        variance=np.sum((rot[:2]@influence[:3])**2,axis=0)*sigma**2/(1-leverage)
        separation=np.max(np.linalg.norm(delta_h,axis=0)/np.sqrt(variance))
        scores[k]=z,separation,hs
        if z<=gate.residual_z and separation<=gate.separation_z and hs<=gate.horizontal_sigma:
            p[k]=(fit['position_m']-ref)@rot.T
    return dict(position=p,selected=selected,scores=scores)
