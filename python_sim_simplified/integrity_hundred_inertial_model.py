"""Declared-domain 2-D integrity experiments; not RF or phone validation.

Truth is used only here to synthesize observations and by score(). Independent
certificates are explicit scenario inputs, never inferred from GNSS agreement.
"""
import numpy as np


NAMES = ['normal', 'step', 'creep', 'outage', 'velocity', 'tilt', 'gyro',
         'unknown', 'auth', 'zero_velocity', 'bias_calibration', 'timing',
         'lever', 'mounted', 'handheld', 'calibrated', 'ambiguous', 'resolved',
         'drift']


def make_case(name, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(0., 120.5, .5)
    h = .5
    stationary = name in ['zero_velocity', 'bias_calibration']
    mounted = name == 'mounted'
    v0 = np.array([0., 0.]) if stationary else np.array([8., 0. if mounted else 1.])
    acceleration = np.zeros((len(t), 2))
    if not stationary:
        acceleration[:, 0] = .025*np.sin(.13*t)
        if not mounted:
            acceleration[:, 1] = .018*np.cos(.09*t)
    truth = [np.zeros(2)]
    velocity = [v0.copy()]
    for a in acceleration[1:]:
        truth.append(truth[-1] + velocity[-1]*h + a*h*h/2)
        velocity.append(velocity[-1]+a*h)
    truth = np.array(truth)
    velocity = np.array(velocity)
    bias = np.array([.018, -.012])
    tilt = np.array([.006, -.003]) if name in ['tilt', 'calibrated'] else np.zeros(2)
    gyro = np.array([.0002, -.0001]) if name == 'gyro' else np.zeros(2)
    drift = np.array([.0007, -.0003]) if name == 'drift' else np.zeros(2)
    noise = rng.uniform(-.002, .002, acceleration.shape)
    measured = acceleration + bias + 9.81*(tilt+t[:, None]*gyro) + t[:, None]*drift + noise
    delay = .7 if name == 'timing' else 0.
    lever = np.array([2., -1.]) if name == 'lever' else np.zeros(2)
    # Initial values are measured certificates, with deliberately nonzero errors.
    p0 = np.array([1.5, -.8]) - v0*delay
    velocity_error = np.array([.35, -.2]) if name == 'velocity' else np.array([.025, -.015])
    p_radius = np.array([2., 2.])
    v_radius = np.abs(velocity_error)+.01
    gnss = truth+lever+rng.uniform(-1., 1., truth.shape)
    if name in ['step', 'unknown']:
        gnss[t >= (0. if name == 'unknown' else 10.), 0] += 200.
    if name == 'creep':
        gnss[:, 0] += .8*t
    if name == 'outage':
        gnss[(t >= 10.) & (t <= 13.)] = np.nan
        gnss[t > 13., 0] += 200.
    auth_p = {}
    if name in ['auth', 'resolved']:
        for k in range(40, len(t), 40):
            auth_p[k] = (truth[k] + rng.uniform(-.5, .5, 2), np.ones(2))
    auth_v = {}
    if name in ['zero_velocity', 'auth']:
        for k in range(10, len(t), 10):
            auth_v[k] = (velocity[k]+rng.uniform(-.005, .005, 2), np.full(2, .01))
    bias_certificate = None
    if name == 'bias_calibration':
        bias_certificate = (bias+np.array([.0003, -.0002]), np.full(2, .0005))
    attitude_certificate = None
    if name == 'calibrated':
        attitude_certificate = (tilt+np.array([.0001, -.0001]), np.full(2, .0002))
    hypotheses = []
    if name in ['ambiguous', 'resolved']:
        hypotheses = [p0.copy(), p0+np.array([200., 0.])]
    observed = dict(time=t, acceleration=measured, gnss=gnss,
                    initial_position=p0, initial_velocity=v0+velocity_error,
                    initial_position_radius=p_radius, initial_velocity_radius=v_radius,
                    initial_position_trusted=name != 'unknown',
                    accel_bound=np.full(2, .022), tilt_bound=np.abs(tilt),
                    gyro_bound=np.abs(gyro), bias_drift_bound=np.abs(drift),
                    absolute_certificates=auth_p, velocity_certificates=auth_v,
                    bias_certificate=bias_certificate, attitude_certificate=attitude_certificate,
                    time_delay=delay, time_delay_radius=.03 if delay else 0.,
                    lever_arm=lever, lever_radius=np.full(2, .1) if name == 'lever' else np.zeros(2),
                    nonholonomic_certificate=mounted, hypotheses=hypotheses)
    return observed, truth+lever


def score(result, truth):
    error = np.linalg.norm(result['position']-truth, axis=1)
    normal = result['normal']
    return dict(epochs=len(error), all_epochs_le20=bool(np.all(normal & (error <= 20))),
                normal_output_rate=float(normal.mean()), protected_epochs=int((~normal).sum()),
                error_le20_epochs=int((normal & (error <= 20)).sum()),
                normal_error_ge30=int((normal & (error >= 30)).sum()),
                normal_error_ge50=int((normal & (error >= 50)).sum()),
                latent_error_ge30=int((error >= 30).sum()),
                latent_error_ge50=int((error >= 50).sum()),
                max_latent_error_m=float(np.max(error)),
                max_normal_error_m=float(np.max(error[normal])) if normal.any() else None,
                bound_violations=int((error > result['bound']+1e-8).sum()),
                gnss_admission_rate=float(result['accepted'].mean()))
