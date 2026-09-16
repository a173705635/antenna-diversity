"""W4 local Cartesian INS baseline. SI units; R maps body into navigation frame.

Inputs are midpoint specific force (gravity NOT removed) and body angular rate.
Caller supplies initial state, bias estimates, gravity and the IMU lever arm.
Flat nonrotating navigation frame: Earth rotation/curvature are not modeled.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def propagate(time_edges_s, force_mps2, rate_rad_s, *, position_m, velocity_mps,
              body_to_nav, accel_bias_mps2, gyro_bias_rad_s, gravity_mps2,
              lever_arm_m, previous_rate_rad_s):
    p=np.asarray(position_m,float).copy();v=np.asarray(velocity_mps,float).copy()
    rotation=Rotation.from_matrix(body_to_nav)
    previous=np.asarray(previous_rate_rad_s)-gyro_bias_rad_s
    dt=np.diff(time_edges_s);old_dt=dt[0]
    positions=[p.copy()];velocities=[v.copy()];attitudes=[rotation.as_matrix()]
    for h,f,w in zip(dt,force_mps2,rate_rad_s):
        omega=w-gyro_bias_rad_s
        alpha=(omega-previous)/((h+old_dt)/2)
        force=(f-accel_bias_mps2-np.cross(alpha,lever_arm_m)
               -np.cross(omega,np.cross(omega,lever_arm_m)))
        half=rotation*Rotation.from_rotvec(omega*h/2)
        acceleration=half.apply(force)+gravity_mps2
        p=p+v*h+acceleration*h*h/2;v=v+acceleration*h
        rotation=rotation*Rotation.from_rotvec(omega*h)
        positions.append(p.copy());velocities.append(v.copy());attitudes.append(rotation.as_matrix())
        previous=omega;old_dt=h
    return dict(time_s=np.asarray(time_edges_s),position_m=np.array(positions),
                velocity_mps=np.array(velocities),body_to_nav=np.array(attitudes))


def trajectory(time_s, kind):
    """Analytic generator/evaluator only; never called by propagate."""
    t=np.asarray(time_s);zero=np.zeros_like(t)
    if kind=='turn':
        yaw=.12*t+.2*np.sin(.4*t);w=.12+.08*np.cos(.4*t);alpha=-.032*np.sin(.4*t)
        radius=40.
        p=np.column_stack((radius*np.sin(yaw),radius*(1-np.cos(yaw)),zero))
        v=np.column_stack((radius*w*np.cos(yaw),radius*w*np.sin(yaw),zero))
        a=np.column_stack((radius*(alpha*np.cos(yaw)-w*w*np.sin(yaw)),
                           radius*(alpha*np.sin(yaw)+w*w*np.cos(yaw)),zero))
    else:
        speed={'static':0.,'straight':10.}[kind]
        yaw=zero;w=zero;alpha=zero
        p=np.column_stack((speed*t,zero,zero));v=np.column_stack((zero+speed,zero,zero))
        a=np.zeros_like(p)
    rotation=Rotation.from_rotvec(np.column_stack((zero,zero,yaw)))
    return dict(position_m=p,velocity_mps=v,acceleration_mps2=a,rotation=rotation,
                rate_rad_s=np.column_stack((zero,zero,w)),alpha_rad_s2=np.column_stack((zero,zero,alpha)))


def measurements(time_edges_s, kind, *, lever_arm_m, gravity_mps2, time_offset_s,
                 bias, axis_matrix, noise_covariance, correlation, seed):
    """Correlated six-axis errors; positive offset samples an earlier physical time."""
    t=(np.asarray(time_edges_s[:-1])+time_edges_s[1:])/2-time_offset_s
    state=trajectory(t,kind);w=state['rate_rad_s']
    force=(state['rotation'].inv().apply(state['acceleration_mps2']-gravity_mps2)
           +np.cross(state['alpha_rad_s2'],lever_arm_m)+np.cross(w,np.cross(w,lever_arm_m)))
    clean=np.column_stack((force,w))
    rng=np.random.default_rng(seed)
    noise=rng.multivariate_normal(np.zeros(6),noise_covariance,len(t))
    for i in range(1,len(noise)):
        noise[i]=correlation*noise[i-1]+np.sqrt(1-correlation**2)*noise[i]
    y=clean@np.asarray(axis_matrix).T+bias+noise
    return dict(sample_time_s=t,time_labels_s=t+time_offset_s,
                force_mps2=y[:,:3],rate_rad_s=y[:,3:])
