"""Measurement admission experiments, not an authenticated integrity bound."""
from dataclasses import dataclass
import numpy as np

from positioning import solve_position
from run_gps_spp import receiver_reference


@dataclass
class Policy:
    residual_m: float = 10.
    exclusions: int = 2
    minimum_stars: int = 6
    range_rate_mps: float = 30.
    range_offset_m: float = 20.
    confirmations: int = 2
    geometry_limit: float = 30.
    use_continuity: bool = True
    admitted_start: bool = True


def fit_ranges(satellites, ranges, eligible, policy):
    mask=eligible.copy();reference,rotation=receiver_reference()
    for iteration in range(policy.exclusions+1):
        if mask.sum()<policy.minimum_stars:
            return np.full(3,np.nan),mask,np.nan
        u=satellites[mask]-reference
        design=np.column_stack((-u/np.linalg.norm(u,axis=1)[:,None],np.ones(mask.sum())))
        if np.linalg.cond(design)>policy.geometry_limit:
            return np.full(3,np.nan),mask,np.nan
        solution=solve_position(satellites[mask],ranges[mask])
        residual=solution['residual_m'];rms=np.sqrt(np.mean(residual**2))
        if rms<=policy.residual_m:
            return (solution['position_m']-reference)@rotation.T,mask,rms
        if iteration<policy.exclusions:
            # Leverage-adjusted residual, using an independent QR projection.
            q,_=np.linalg.qr(design,mode='reduced')
            statistic=abs(residual)/np.sqrt(1-np.sum(q*q,axis=1))
            mask[np.flatnonzero(mask)[np.argmax(statistic)]]=False
    return np.full(3,np.nan),mask,rms


def admit(time_s, satellites, ranges, eligible, policy):
    """Consume observed ranges only. References advance on admitted outputs only."""
    n,s=ranges.shape
    position=np.full((n,3),np.nan);selected=np.zeros((n,s),bool)
    residual=np.full(n,np.nan);streak=np.zeros(s,int)
    reference_range=np.zeros(s);reference_time=np.zeros(s)
    reference_sat=np.zeros((s,3));known=np.zeros(s,bool)
    reference,rotation=receiver_reference()
    last_position=reference.copy()
    for k,t in enumerate(time_s):
        mask=eligible[k].copy()
        if policy.use_continuity and known.any():
            movement=(np.linalg.norm(satellites[k]-last_position,axis=1)
                      -np.linalg.norm(reference_sat-last_position,axis=1))
            delta=ranges[k]-reference_range-movement
            shared=mask & known
            if shared.sum()>=policy.minimum_stars:
                clock=np.median(delta[shared])
                accepted=(abs(delta-clock)<=policy.range_offset_m+policy.range_rate_mps*(t-reference_time))
                mask &= ~known | accepted
        p,mask,rms=fit_ranges(satellites[k],ranges[k],mask,policy)
        residual[k]=rms
        if np.isfinite(p).all():
            streak=np.where(mask,streak+1,0)
            mature=mask & (streak>=policy.confirmations)
            p,mature,rms=fit_ranges(satellites[k],ranges[k],mature,policy)
            if policy.admitted_start and np.isfinite(p).all():
                position[k]=p;selected[k]=mature;residual[k]=rms
                reference_range[mature]=ranges[k,mature];reference_time[mature]=t
                reference_sat[mature]=satellites[k,mature];known[mature]=True
                last_position=reference+p@rotation
        else:
            streak[:]=0
    return dict(position=position,selected=selected,residual=residual)
