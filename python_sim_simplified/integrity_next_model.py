"""600 s 3-D synthetic measurements. No hardware certificate is claimed.

Independent bench records use a separate RNG and zero-acceleration fixture.
Only score() sees trajectory truth; receiver sees observations and declarations.
"""
import numpy as np

NAMES = ['bounded', 'unknown_origin', 'empirical_only', 'gaussian', 'drift',
         'holdout_fault', 'temperature', 'velocity', 'stationary', 'false_stop',
         'initial_delay', 'jitter', 'expired', 'delayed_anchor', 'bias_identify',
         'tilt3d', 'attitude', 'gyro', 'lever', 'lever_uncertain', 'scale',
         'clipping', 'conflicting_anchor', 'anchor_recovery', 'circular_trust',
         'real_phone_unavailable']

def make_case(name, seed):
    rng = np.random.default_rng(seed)
    bench = np.random.default_rng(seed + 900000)
    t = np.arange(601., dtype=float)
    a = np.column_stack((.05*np.sin(t*.07), .04*np.cos(t*.03), .02*np.sin(t*.11)))
    v0 = np.array([8., 1., .2])
    if name in ['stationary', 'false_stop']:
        a[:] = 0
        v0 = np.zeros(3) if name == 'stationary' else np.array([2., 0., 0.])
    p = np.zeros((len(t), 3)); v = p.copy(); v[0] = v0
    for k in range(1, len(t)):
        p[k] = p[k-1]+v[k-1]+a[k]/2
        v[k] = v[k-1]+a[k]
    bias = np.array([.018, -.012, .009])
    train = bias + bench.uniform(-.002, .002, (100, 3))
    holdout = bias + bench.uniform(-.002, .002, (40, 3))
    temp = 20+10*np.sin(t*.003)
    coeff = np.array([.003, -.002, .001])
    drift = np.array([.0001, -.00006, .00004]) if name == 'drift' else np.zeros(3)
    tilt = np.array([.025, -.017, .012]) if name in ['tilt3d','attitude','gyro'] else np.zeros(3)
    gyro = np.array([.00008, -.00005, .00003]) if name == 'gyro' else np.zeros(3)
    leakage = 9.81*np.sin(tilt[None,:]+t[:,None]*gyro)
    noise = rng.normal(0, .002, a.shape) if name == 'gaussian' else rng.uniform(-.002,.002,a.shape)
    measured = a+bias+t[:,None]*drift+leakage+noise
    if name == 'holdout_fault':
        measured += .2
        holdout += .2
    if name == 'temperature':
        measured += (temp-20)[:,None]*coeff
    if name == 'scale':
        measured += a*.15
    if name == 'clipping':
        a[100:105,0] = 3.
        for k in range(1,len(t)):
            p[k] = p[k-1]+v[k-1]+a[k]/2
            v[k] = v[k-1]+a[k]
        measured[100:105,0] = .1
    delay = 2. if name in ['initial_delay','jitter'] else 0.
    initial = np.array([1.,-.5,.3])-v0*delay
    jitter = .4 if name == 'jitter' else 0.
    initial -= v0*jitter
    lever = np.column_stack((2*np.cos(t*.03),2*np.sin(t*.03), np.full(len(t),.5))) if name in ['lever','lever_uncertain'] else np.zeros_like(p)
    lever_meas = lever.copy()
    if name == 'lever_uncertain':
        lever_meas += np.array([.3,-.2,.1])
    anchors = {}
    if name in ['delayed_anchor','conflicting_anchor','anchor_recovery','circular_trust']:
        for k in range(20,len(t),20):
            lag = 3 if name == 'delayed_anchor' else 0
            z = p[k-lag]+rng.uniform(-.5,.5,3)
            if name == 'conflicting_anchor' and k == 40: z += 200
            if name == 'circular_trust': z += 200
            anchors[k] = dict(z=z, radius=np.ones(3), lag=lag,
                              origin='gnss' if name == 'circular_trust' else 'independent_synthetic')
    velocities = {}
    if name in ['velocity','bias_identify','delayed_anchor']:
        for k in range(10,len(t),10): velocities[k] = (v[k]+rng.uniform(-.005,.005,3), np.full(3,.01))
    gnss = p+lever+rng.uniform(-1,1,p.shape)
    gnss[t>=10,0] += 200
    o = dict(time=t, acceleration=measured, gnss=gnss, initial_position=initial,
             initial_velocity=v0+np.array([.025,-.015,.01]), initial_radius=np.full(3,2.),
             velocity_radius=np.full(3,.04), origin_trusted=name not in ['unknown_origin','real_phone_unavailable'],
             training=train, holdout=holdout, reference_radius=.001, noise_bound=.002,
             envelope_origin='empirical' if name in ['empirical_only','real_phone_unavailable'] else 'synthetic_declared_domain',
             noise_model='gaussian' if name == 'gaussian' else 'bounded',
             drift_bound=np.full(3,.00015), drift_enabled=name=='drift',
             temperature=temp, temperature_coefficient=coeff if name=='temperature' else np.zeros(3),
             temperature_radius=.0001 if name=='temperature' else 0.,
             velocities=velocities, stationary_claim=name in ['stationary','false_stop'],
             stationary_origin='fixture' if name=='stationary' else 'accelerometer_only',
             initial_delay=delay, time_radius=.5 if name=='jitter' else 0.,
             envelope_expiry=40 if name=='expired' else 1000,
             anchors=anchors, tilt_bound=np.full(3,.03) if np.any(tilt) else np.zeros(3),
             attitude=(tilt+np.array([.0002,-.0001,.0001]),np.full(3,.0005)) if name in ['attitude','gyro'] else None,
             gyro_bound=np.full(3,.0001) if name=='gyro' else np.zeros(3),
             lever=lever_meas, lever_radius=np.full(3,.4) if name=='lever_uncertain' else np.zeros(3),
             scale_bound=.2 if name=='scale' else 0., saturated=np.any(np.abs(measured)>=.099,axis=1) if name=='clipping' else np.zeros(len(t),bool))
    if name=='anchor_recovery': o['envelope_origin']='empirical'
    if name=='unknown_origin': p += np.array([200.,0.,0.])
    return o, p+lever

def score(r, truth):
    error = np.linalg.norm(r['position']-truth,axis=1)
    normal = r['normal']
    return dict(epochs=len(error), normal_output_rate=float(normal.mean()),
                full_success=bool(np.all(normal & (error<=20))),
                normal_ge30=int(np.sum(normal & (error>=30))), normal_ge50=int(np.sum(normal & (error>=50))),
                bound_violations=int(np.sum(error>r['bound']+1e-7)),
                max_latent_error=float(error.max()), max_normal_error=float(error[normal].max()) if normal.any() else None)
