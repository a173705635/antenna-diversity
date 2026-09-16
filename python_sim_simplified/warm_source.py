"""Warm-start source classification in carrier-wiped spatial observation domain.

Not RF, acquisition, tracking or PVT. Four synchronous antenna samples are
generated once and subset by the caller. Nominal physical aperture is 7 x 14 cm.
"""
import numpy as np

SCENARIOS = ('clean', 'phase_drift', 'gain_drift', 'low_signal',
             'reflection', 'near_direction', 'motion', 'reacquisition')


def unit(a):
    return a / np.linalg.norm(a, axis=0)


def make_case(seed, scenario, attack):
    rng = np.random.default_rng(seed)
    positions = np.array([[0, 0], [.07, 0], [0, .14], [.07, .14]])
    az = rng.uniform(-np.pi, np.pi, 8)
    el = rng.uniform(.2, 1.3, 8)
    sky = np.array([np.cos(az)*np.cos(el), np.sin(az)*np.cos(el)])
    a = np.exp(2j*np.pi/.1903 * positions @ sky)
    carrier = np.exp(1j*rng.uniform(-np.pi, np.pi, 8))
    calibration = np.exp(.08*rng.normal(size=4)+.15j*rng.normal(size=4))
    a = a * carrier * calibration[:, None]
    noise = lambda shape: (rng.normal(size=shape)+1j*rng.normal(size=shape))/np.sqrt(2)
    sigma = .65 if scenario == 'low_signal' else .16
    warm = a[None] + sigma*noise((12, 4, 8))
    source_direction = sky[:, 0] if scenario == 'near_direction' else np.array([.35, -.4])
    b = np.exp(2j*np.pi/.1903 * positions @ source_direction)*calibration
    post = a.copy()
    if scenario == 'motion':
        post = post*np.exp(1j*np.arange(4)[:, None]*np.linspace(-.25,.25,8))
    if scenario == 'reflection':
        # A benign common rebroadcast/reflection is an intentional counterexample.
        post = .25*post + .9*b[:, None]*carrier
    if attack:
        alpha = .65 if scenario == 'near_direction' else .9
        post = (1-alpha)*post + alpha*b[:, None]*carrier
    if scenario == 'phase_drift':
        post = post*np.exp(1j*np.array([0,.9,-.7,.45]))[:, None]
    if scenario == 'gain_drift':
        post = post*np.array([1,1.8,.55,1.3])[:, None]
    if scenario == 'reacquisition':
        post = post*np.exp(1j*rng.uniform(-np.pi,np.pi,8))
    current = post[None] + sigma*noise((6,4,8))
    return dict(warm=warm, current=current, attack=attack, scenario=scenario,
                attack_time=1.6, interval=.1, domain='carrier-wiped spatial observations')


def initialize(warm):
    template = warm[:8].mean(axis=0)
    validation = warm[8:].mean(axis=0)
    mismatch = float(np.mean(1-abs(np.sum(unit(template).conj()*unit(validation),axis=0))**2))
    complete = mismatch < .06
    return dict(template=template, mismatch=mismatch, complete=bool(complete),
                time=1.2 if complete else None)


def features(warm, current):
    a = warm[:8].mean(axis=0)
    b = current.mean(axis=0)
    return a, b, unit(a), unit(b)


def score(warm, current):
    a, b, u, v = features(warm, current)
    ratio = b/a
    left, singular, right = np.linalg.svd(ratio,full_matrices=False)
    fit_sky = a*(left[:,:1]*singular[0]@right[:1])
    ls, ss, rs = np.linalg.svd(b,full_matrices=False)
    fit_common = ls[:,:1]*ss[0]@rs[:1]
    noise = np.mean(abs(warm[:8]-a)**2)
    return float((np.sum(abs(b-fit_sky)**2)-np.sum(abs(b-fit_common)**2))/noise)
