"""Phone differential carrier/gyro observation-domain research, no navigation."""
import numpy as np
from scipy.spatial.transform import Rotation


def estimate(o):
    t = o['time']
    rate = o['gyro'].copy()
    rate -= o['bias_readings'].mean(axis=0)
    rate /= o['scale_readings'].mean(axis=0)
    aligned_time = o['gyro_time']-o['clock_offset_readings'].mean()
    rate = np.column_stack([np.interp(t,aligned_time,rate[:,j]) for j in range(3)])
    rotations = np.repeat(np.eye(3)[None], len(t), axis=0)
    for k in range(1,len(t)):
        rotations[k] = rotations[k-1] @ Rotation.from_rotvec((rate[k-1]+rate[k])*(t[k]-t[k-1])/2).as_matrix()
    baselines = o['measured_baselines'].copy()
    x = np.einsum('tij,bj->tbi', rotations, baselines) * o['wave_number']
    x = np.concatenate((x,x*(t-t.mean())[:,None,None]),axis=2)
    y = o['phase'].copy()
    y = np.unwrap(y,axis=0)
    increment_support = o['wave_number']*np.max(np.linalg.norm(baselines,axis=1))*np.linalg.norm(rate[1:],axis=1)*np.diff(t)+2*o['phase_noise_bound']
    breaks = np.flatnonzero(np.any(np.abs(np.diff(y,axis=0))>increment_support[:,None,None],axis=(1,2)))+1
    edges = np.r_[0,breaks,len(t)]
    tau = t-t.mean()
    nuisance = np.column_stack([((np.arange(len(t))>=start)&(np.arange(len(t))<stop)).astype(float) for start,stop in zip(edges[:-1],edges[1:])]+[tau])
    x -= (nuisance @ np.linalg.lstsq(nuisance,x.reshape(len(t),-1),rcond=None)[0]).reshape(x.shape)
    y -= (nuisance @ np.linalg.lstsq(nuisance,y.reshape(len(t),-1),rcond=None)[0]).reshape(y.shape)
    a = x.reshape(-1, 6)
    b = y.reshape(-1, y.shape[-1])
    u = np.linalg.lstsq(a, b, rcond=None)[0]
    u[:3] = u[:3] / np.linalg.norm(u[:3],axis=0,keepdims=True)
    u[3:] -= u[:3]*np.sum(u[:3]*u[3:],axis=0,keepdims=True)
    static = np.linalg.lstsq(a[:,:3],b,rcond=None)[0]
    static /= np.linalg.norm(static,axis=0,keepdims=True)
    sigma2 = o['phase_noise_bound']**2/3
    static_score = np.sum((b-a[:,:3]@static)**2)/sigma2+2*b.shape[1]*np.log(b.size)
    moving_score = np.sum((b-a@u)**2)/sigma2+4*b.shape[1]*np.log(b.size)
    selected_model = 'moving'
    if static_score < moving_score:
        a = a[:,:3]
        u = static
        selected_model = 'static'
    residual = b - a @ u
    contrast = b-b.mean(axis=1,keepdims=True)
    relative_directions = np.linalg.lstsq(a,contrast,rcond=None)[0]
    predicted = a @ relative_directions
    consensus = np.median(predicted,axis=1,keepdims=True)
    spread = float(np.sqrt(np.median(np.mean((predicted-consensus)**2,axis=0))))
    rms = float(np.sqrt(np.mean(residual**2)))
    singular = np.linalg.svd(a, compute_uv=False)
    status = 'common' if spread < .12 else 'diverse'
    if singular[-1] <= .5:
        status = 'indeterminate'
    observable = singular[-1] > .5 and rms < .08 and status == 'common'
    mean = np.median(u[:3],axis=1)
    direction = mean / np.linalg.norm(mean)
    angular_support = o['gyro_error_bound']*(t[-1]-t[0])+o['timestamp_bound']*np.max(np.linalg.norm(rate,axis=1))
    phase_support = o['phase_noise_bound']+o['wave_number']*np.max(np.linalg.norm(baselines,axis=1))*angular_support
    phase_support += o['wave_number']*o['phase_center_bound']
    radius = float(np.degrees(np.arcsin(np.clip(np.sqrt(len(a))*phase_support/singular[-1],0,1))))
    declared_domain = o['gyro_certified'] and o['bounded_phase_noise']
    if not declared_domain:
        status = 'indeterminate'
    observable = observable and declared_domain and radius < 30.
    return {'status': status, 'spread': spread, 'rms': rms, 'selected_model': selected_model,
            'singular_min': float(singular[-1]), 'direction': direction.tolist(),
            'direction_available': bool(observable), 'direction_radius_deg': radius,
            'direction_frame': 'initial_body', 'support_is_certificate': False,
            'absolute_position_available': False}
