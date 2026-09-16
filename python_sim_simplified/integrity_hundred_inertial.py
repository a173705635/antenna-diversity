"""Selected 081--100 bounded-inertial integrity model; not an RF authenticator."""
import numpy as np
from integrity_hundred_inertial_model import make_case, score


def receive(o):
    if o['hypotheses']:
        runs = [receive(dict(o, initial_position=p, hypotheses=[])) for p in o['hypotheses']]
        pos = np.array([r['position'] for r in runs])
        rad = np.array([r['bound'] for r in runs])
        viable = np.array([r['feasible'] for r in runs])
        lo = np.min(np.where(viable[:, :, None], pos-rad[:, :, None], np.inf), axis=0)
        hi = np.max(np.where(viable[:, :, None], pos+rad[:, :, None], -np.inf), axis=0)
        any_viable = np.any(viable, axis=0)
        p = np.full_like(lo, np.nan); radius = np.full(len(o['time']), np.inf)
        p[any_viable] = (lo[any_viable]+hi[any_viable])/2
        radius[any_viable] = np.linalg.norm((hi[any_viable]-lo[any_viable])/2, axis=1)
        return dict(position=p, bound=radius, normal=any_viable & (radius <= 20.),
                    accepted=np.any([r['accepted'] for r in runs], axis=0),
                    feasible=any_viable)
    # HYPOTHESES
    t = o['time']
    p = o['initial_position'].copy()
    v = o['initial_velocity'].copy()
    rp = o['initial_position_radius'].copy() if o['initial_position_trusted'] else np.full(2, np.inf)
    rv = o['initial_velocity_radius'].copy()
    p += v*o['time_delay']
    rp += rv*abs(o['time_delay']) + (np.abs(v)+rv)*o['time_delay_radius']
    # INITIAL_ALIGNMENT
    positions = []; bounds = []; normal = []; admitted = []; feasibility = []
    feasible = True
    for k in range(len(t)):
        a_bound = o['accel_bound'].copy() + 9.81*(o['tilt_bound']+t[k]*o['gyro_bound']) + t[k]*o['bias_drift_bound']
        a = o['acceleration'][k].copy()
        if o['bias_certificate'] is not None:
            bias, r = o['bias_certificate']
            a -= bias
            a_bound += r+.002-o['accel_bound']
        if o['attitude_certificate'] is not None:
            tilt, r = o['attitude_certificate']
            a -= 9.81*tilt
            a_bound += 9.81*(r-o['tilt_bound'])
        # ACCEL_CORRECTIONS
        if k:
            h = t[k]-t[k-1]
            p += v*h+a*h*h/2
            v += a*h
            rp += rv*h+a_bound*h*h/2
            rv += a_bound*h
        if k in o['absolute_certificates']:
            z, r = o['absolute_certificates'][k]
            lo = np.maximum(p-rp, z-r); hi = np.minimum(p+rp, z+r)
            feasible = feasible and bool(np.all(lo <= hi))
            if feasible:
                p = (lo+hi)/2; rp = (hi-lo)/2
        if k in o['velocity_certificates']:
            v, rv = (x.copy() for x in o['velocity_certificates'][k])
        if o['nonholonomic_certificate']:
            v[1] = 0.; rv[1] = .01
        g = o['gnss'][k]-o['lever_arm']
        available = bool(np.isfinite(g).all())
        # GNSS arc changes carry no absolute trust.
        output = p.copy()
        accepted = available and np.linalg.norm(g-p) <= 20.
        # Keep the minimax independent center; accepted is a diagnostic only.
        bound = np.linalg.norm(rp+o['lever_radius']+np.abs(output-p))
        is_normal = bool(bound <= 20. and feasible)
        output = output+o['lever_arm']
        positions.append(output); bounds.append(bound); normal.append(is_normal)
        admitted.append(accepted); feasibility.append(feasible)
    return dict(position=np.array(positions), bound=np.array(bounds),
                normal=np.array(normal), accepted=np.array(admitted), feasible=np.array(feasibility))
