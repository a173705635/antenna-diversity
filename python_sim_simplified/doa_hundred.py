"""061--080: synthetic single-emitter DOA research; no RF/IMU or PVT claims.

Convention matches signal_recovery: a=exp(+j*2pi/lambda*p.u), x-east
azimuth. The source complex amplitude is nuisance. All simulations share
fixed input draws within each comparison, and retain planar mirror solutions.
"""
import hashlib
import itertools
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from presets import mate70_phase_centers

P = np.array(list(mate70_phase_centers().values()))
LAM = 299792458 / 1575.42e6
K = 2*np.pi/LAM
ROOT = Path(__file__).resolve().parent / 'results/research-hundred'


def unit(az, el):
    a, e = np.deg2rad([az, el])
    return np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])


def angle(a, b):
    return float(np.rad2deg(np.arccos(np.clip(np.dot(a, b), -1, 1))))


def lift(q):
    z = np.sqrt(1-np.dot(q, q))
    return np.array([[*q, z], [*q, -z]])


def aliases(phase, positions=P, harmonic=1):
    b = positions[1:, :2]-positions[0, :2]
    scale = K*harmonic
    candidates = []
    for n in itertools.product(range(-2, 3), repeat=len(b)):
        y = (phase+2*np.pi*np.array(n))/scale
        q = np.linalg.lstsq(b, y, rcond=None)[0]
        if q@q <= 1 and np.linalg.norm(b@q-y) < 1e-8:
            if not any(np.linalg.norm(q-v) < 1e-7 for v in candidates):
                candidates.append(q)
    return np.array(candidates)


def rng():
    return np.random.default_rng(61080)


def phase_samples(u, count=200, sigma=.02):
    error = rng().normal(0, sigma, (count, 4))
    return K*((P-P[0])@u)[None, :]+error-error[:, :1]


def serial(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x).__name__)


def run(number, hypothesis, change, comparison):
    folder = ROOT / f'iteration{number:03d}'
    folder.mkdir(parents=True, exist_ok=False)
    source = Path(__file__)
    shutil.copyfile(source, folder / source.name)
    proposal = dict(iteration=number, hypothesis=hypothesis, actual_change=change,
                    comparison=comparison, scope='synthetic single-emitter DOA, no navigation inference',
                    no_measured_rf_or_imu=True)
    (folder/'proposal.json').write_text(json.dumps(proposal, indent=2), encoding='utf8')
    result = globals()[f'round{number:03d}']()
    result['source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    (folder/'results.json').write_text(json.dumps(result, indent=2, default=serial), encoding='utf8')
    print(json.dumps(dict(iteration=number, **result), default=serial))


def round061():
    u = unit(67, 25)
    phi = np.angle(np.exp(1j*K*((P[1:]-P[0])@u)))
    naive = np.linalg.lstsq(K*(P[1:, :2]-P[0, :2]), phi, rcond=None)[0]
    qs = aliases(phi)
    return dict(truth_xy=u[:2], principal_phase_xy=naive, integer_candidates=qs,
                naive_xy_error=np.linalg.norm(naive-u[:2]),
                candidate_truth_error=np.min(np.linalg.norm(qs-u[:2], axis=1)),
                candidate_count=len(qs), observed_phase=phi)


def round062():
    u = unit(25, -42)
    phi = np.angle(np.exp(1j*K*((P[1:]-P[0])@u)))
    candidates = np.concatenate([lift(q) for q in aliases(phi)])
    predicted = np.exp(1j*K*P@candidates.T)
    target = np.exp(1j*K*P@u)
    losses = np.sum(abs(predicted/predicted[:1]-target[:, None]/target[0])**2, axis=0)
    upper = candidates[candidates[:, 2] >= 0]
    return dict(candidate_count=len(candidates), equal_phase_losses=losses,
                forced_upper_error_deg=min(angle(c, u) for c in upper),
                candidate_set_error_deg=min(angle(c, u) for c in candidates),
                status='unresolved_mirror_and_possible_wrap_alias')


def round063():
    u = unit(35, 30)
    b = P[1]-P[0]
    qx = np.angle(np.exp(1j*K*b@u))/(K*b[0])
    t = np.linspace(0, 2*np.pi, 721)
    circle = np.column_stack((np.full(len(t), qx), np.sqrt(1-qx*qx)*np.cos(t),
                              np.sqrt(1-qx*qx)*np.sin(t)))
    return dict(array_count=2, tangent_information_rank=1,
                candidate_circle_phase_error=float(np.max(abs(np.exp(1j*K*circle@b)-np.exp(1j*K*u@b)))),
                candidates_angular_diameter_deg=angle(circle[0], circle[360]),
                status='continuous_circle_not_point_doa')


def round064():
    u = unit(30, 45)
    y = phase_samples(u, 2000, .08)
    metrics = {}
    for n in [3, 4]:
        b = K*(P[1:n, :2]-P[0, :2])
        q = y[:, 1:n]@np.linalg.pinv(b).T
        metrics[str(n)] = dict(xy_rmse=np.sqrt(np.mean(np.sum((q-u[:2])**2, axis=1))),
                              geometry_rank=np.linalg.matrix_rank(b),
                              noiseless_alias_count=len(aliases(np.angle(np.exp(1j*y[0, 1:n]*0+1j*K*(P[1:n]-P[0])@u)), P[:n])))
    return dict(metrics=metrics, premise='Locally resolved phase branch, identical first three channels',
                finding='Fourth rectangle corner improves variance but adds no new noiseless lattice constraint')


def round065():
    u = unit(30, 45)
    y = phase_samples(u, 2000, .08)[:, 1:]
    b = K*(P[1:, :2]-P[0, :2])
    covariance = .08**2*(np.eye(3)+np.ones((3, 3)))
    w = np.linalg.inv(covariance)
    ols = np.linalg.pinv(b)
    gls = np.linalg.solve(b.T@w@b, b.T@w)
    def assess(g):
        q = y@g.T
        return dict(xy_rmse=np.sqrt(np.mean(np.sum((q-u[:2])**2, axis=1))),
                    predicted_covariance=g@covariance@g.T,
                    empirical_covariance=np.cov((q-u[:2]).T))
    return dict(ordinary=assess(ols), correlated_reference_gls=assess(gls),
                covariance_model='Independent equal-variance channel noise creates common reference covariance')


def complex_fit(v, phase_only=False):
    target = v/abs(v) if phase_only else v
    def residual(q):
        a = np.exp(1j*K*P[:, :2]@q)
        alpha = np.vdot(a, target)/len(a)
        e = target-alpha*a
        return np.r_[e.real, e.imag]
    fit = least_squares(residual, [.3, .3])
    return fit.x, np.linalg.norm(fit.fun)


def round066():
    u = unit(35, 65)
    amplitude = np.array([1, .3, 1.8, .6])
    v = amplitude*np.exp(1j*K*P@u)
    raw, raw_loss = complex_fit(v)
    phase, phase_loss = complex_fit(v, True)
    return dict(raw_xy_error=np.linalg.norm(raw-u[:2]), raw_manifold_residual=raw_loss,
                phase_xy_error=np.linalg.norm(phase-u[:2]), phase_manifold_residual=phase_loss,
                channel_amplitudes=amplitude, premise='Positive amplitude mismatch only, noiseless, same local branch')


def round067():
    u, reference = unit(35, 65), unit(10, 50)
    gain = np.array([1, .8, 1.2, .9])*np.exp(1j*np.array([0, .2, -.15, .3]))
    generator = rng()
    noise = (generator.normal(size=(2, 200, 4))+1j*generator.normal(size=(2, 200, 4)))*.03
    # Reference direction is an explicit simulated external calibration input.
    cal = gain*np.exp(1j*K*P@reference)+noise[0]
    estimate = cal.mean(axis=0)/np.exp(1j*K*P@reference)
    v = gain*np.exp(1j*K*P@u)+noise[1].mean(axis=0)
    raw, _ = complex_fit(v, True)
    corrected, _ = complex_fit(v/estimate, True)
    return dict(uncalibrated_xy_error=np.linalg.norm(raw-u[:2]),
                calibrated_xy_error=np.linalg.norm(corrected-u[:2]),
                estimated_gain_error=np.linalg.norm(estimate-gain),
                premise='Independent synthetic known-reference calibration; not RF measured or attacker authenticated')


def round068():
    u, alternate = unit(35, 65), unit(140, -20)
    a, aa = np.exp(1j*K*P@u), np.exp(1j*K*P@alternate)
    alternative_gain = a/aa
    return dict(distinct_direction_deg=angle(u, alternate),
                equal_observation_max_difference=np.max(abs(a-alternative_gain*aa)),
                original_gain=np.ones(4), alternate_gain_phase=np.angle(alternative_gain),
                status='arbitrary_unknown_channel_phase_makes_doa_unidentifiable',
                algebra='g2_m=g1_m*exp(j*k*p_m.(u1-u2))')


def round069():
    u = unit(35, 65)
    b = K*(P[1:, :2]-P[0, :2])
    bias = np.array([.10, -.12, .04])
    observed = b@u[:2]+bias
    qpoint = np.linalg.lstsq(b, observed, rcond=None)[0]
    grid = np.arange(-1, 1.0001, .004)
    x, y = np.meshgrid(grid, grid)
    qs = np.column_stack([x.ravel(), y.ravel()])
    qs = qs[np.sum(qs*qs, axis=1) <= 1]
    error = np.angle(np.exp(1j*(observed[None, :]-qs@b.T)))
    retained = qs[np.all(abs(error) <= .15, axis=1)]
    truth_residual = abs(np.angle(np.exp(1j*(observed-b@u[:2]))))
    return dict(point_xy_error=np.linalg.norm(qpoint-u[:2]),
                set_count=len(retained), set_xy_bounds=[retained.min(axis=0), retained.max(axis=0)],
                truth_within_continuous_bound=bool(np.all(truth_residual<=.15)),
                closest_grid_distance=np.min(np.linalg.norm(retained-u[:2], axis=1)),
                premise='Relative calibration residual bounded by 0.15 rad, explicitly assumed; deterministic set, not 95 percent confidence')


def round070():
    u = unit(35, 65)
    generator = rng()
    noise = (generator.normal(size=(300, 32, 4))+1j*generator.normal(size=(300, 32, 4)))*.8
    z = .15*np.exp(1j*K*P@u)+noise
    coherent, phase = [], []
    for row in z:
        qc, _ = complex_fit(row.mean(axis=0))
        qp, _ = complex_fit((row/abs(row)).mean(axis=0))
        coherent.append(np.linalg.norm(qc-u[:2]))
        phase.append(np.linalg.norm(qp-u[:2]))
    return dict(coherent_xy_median=np.median(coherent), phase_first_xy_median=np.median(phase),
                coherent_xy_p95=np.quantile(coherent, .95), phase_first_xy_p95=np.quantile(phase, .95),
                coherent_outside_unit_disk_interpretation='Do not interpret unconstrained local q as a physical point',
                premise='32 coherent fixed-phase post-correlation complex samples; single-source symbol phase already aligned',
                cases=300)


def round071():
    u = unit(20, 2)
    sigma = .02
    qhat = u[:2]+rng().normal(0, sigma, (2000, 2))
    coverage = np.sum((qhat-u[:2])**2, axis=1)/sigma**2 <= 5.991464547
    # Intersect one measured spatial-frequency confidence ellipse with disk,
    # then lift BOTH signs; near horizon the angle transform is nonlinear.
    measured = u[:2]+np.array([-.008, .003])
    axis = np.arange(-1, 1.0001, .002)
    x, y = np.meshgrid(axis, axis)
    q = np.column_stack((x.ravel(), y.ravel()))
    selected = q[(np.sum(q*q, axis=1)<=1) & (np.sum((q-measured)**2, axis=1)<=5.991464547*sigma**2)]
    elevation = np.rad2deg(np.arccos(np.linalg.norm(selected, axis=1)))
    return dict(unphysical_unconstrained_fraction=np.mean(np.sum(qhat*qhat, axis=1)>1),
                spatial_confidence_coverage=np.mean(coverage),
                local_linear_elevation_sigma_deg=np.rad2deg(sigma/np.sin(np.deg2rad(2))),
                physical_elevation_absolute_bounds_deg=[elevation.min(), elevation.max()],
                mirror_signs=[-1, 1], premise='Known Gaussian spatial-frequency covariance, local phase branch only; 95 percent disk-intersection confidence')


def round072():
    u = unit(35, 65)
    generator = rng()
    noise = (generator.normal(size=(4, 1000))+1j*generator.normal(size=(4, 1000)))/np.sqrt(2)
    scale = np.array([1, 1, 1, 5])
    independent_noise = (generator.normal(size=(4, 1000))+1j*generator.normal(size=(4, 1000)))/np.sqrt(2)
    estimated_variance = np.mean(abs(independent_noise*scale[:, None])**2, axis=1)
    waveform = np.exp(1j*generator.uniform(-np.pi, np.pi, 1000))
    x = np.exp(1j*K*P@u)[:, None]*waveform+noise*scale[:, None]
    covariance = x@x.conj().T/1000
    raw_vector = np.linalg.eigh(covariance)[1][:, -1]
    d = np.sqrt(estimated_variance)
    white_vector = d*np.linalg.eigh(covariance/d[:, None]/d[None, :])[1][:, -1]
    raw, _ = complex_fit(raw_vector, True)
    corrected, _ = complex_fit(white_vector, True)
    return dict(raw_xy_error=np.linalg.norm(raw-u[:2]), whitened_xy_error=np.linalg.norm(corrected-u[:2]),
                estimated_noise_variance=estimated_variance,
                premise='Noise-only independent simulated segment supplies whitening; gain recalibrated coordinate map is reversed before fitting')


def round073():
    u = unit(35, 65)
    b = K*(P[1:, :2]-P[0, :2])
    y = phase_samples(u, 120, .02)[:, 1:]
    y[::8, 0] += 1.3
    y[::11, 2] -= 1.1
    mean = np.linalg.lstsq(b, y.mean(axis=0), rcond=None)[0]
    robust = least_squares(lambda q: (y-b@q).ravel(), mean, loss='huber', f_scale=.04).x
    return dict(mean_xy_error=np.linalg.norm(mean-u[:2]), huber_xy_error=np.linalg.norm(robust-u[:2]),
                premise='Sparse channel-specific phase transients; local unwrapped branch. Huber scale fixed from assumed 0.02 rad noise, not tuned on errors',
                warning='Persistent bias or coherent reflection is not sparse and remains a model error')


def round074():
    u = unit(35, 65)
    b = K*(P[1:, :2]-P[0, :2])
    phase = phase_samples(u, 500, .10)[:, 1:]
    q = phase@np.linalg.pinv(b).T
    window = 20
    averaged = np.array([q[t-window+1:t+1].mean(axis=0) for t in range(window-1, len(q))])
    return dict(instant_xy_rmse=np.sqrt(np.mean(np.sum((q[window-1:]-u[:2])**2, axis=1))),
                constant_state_xy_rmse=np.sqrt(np.mean(np.sum((averaged-u[:2])**2, axis=1))),
                window=window, premise='Constant source in array frame and stable calibration, causal 20-epoch state fit; phase branch known')


def round075():
    n, window = 160, 20
    qtrue = np.column_stack((.2+.003*np.arange(n), .25-.001*np.arange(n)))
    error = rng().normal(0, .04, (n, 4))
    b = K*(P[1:, :2]-P[0, :2])
    y = qtrue@b.T+error[:, 1:]-error[:, :1]
    q = y@np.linalg.pinv(b).T
    time = np.arange(-window+1, 1)
    design = np.column_stack((np.ones(window), time))
    static, moving = [], []
    for t in range(window-1, n):
        history = q[t-window+1:t+1]
        static.append(history.mean(axis=0))
        moving.append(np.linalg.lstsq(design, history, rcond=None)[0][0])
    truth = qtrue[window-1:]
    return dict(static_xy_rmse=np.sqrt(np.mean(np.sum((static-truth)**2, axis=1))),
                velocity_xy_rmse=np.sqrt(np.mean(np.sum((moving-truth)**2, axis=1))),
                static_mean_bias=np.mean(static-truth, axis=0),
                velocity_mean_bias=np.mean(moving-truth, axis=0),
                premise='Single moving emitter; causal local linear spatial-frequency trajectory, same 20 epochs and noise')


def round076():
    u = unit(67, 25)
    yaw = np.array([0, 25, 55, 80])
    actual = Rotation.from_euler('z', yaw[:, None], degrees=True).as_matrix()
    measured = Rotation.from_euler('z', (yaw+np.array([0, .3, -.2, .4]))[:, None], degrees=True).as_matrix()
    baseline = P[1:]-P[0]
    btrue = K*np.einsum('tij,mj->tmi', actual, baseline)
    bfit = K*np.einsum('tij,mj->tmi', measured, baseline)
    z = np.exp(1j*btrue@u)
    starts = aliases(np.angle(z[0]))
    estimates, loss = [], []
    for start in starts:
        def residual(q):
            e = np.exp(1j*bfit[:, :, :2]@q)-z
            return np.r_[e.real.ravel(), e.imag.ravel()]
        fit = least_squares(residual, start)
        estimates.append(fit.x)
        loss.append(np.linalg.norm(fit.fun))
    best = estimates[int(np.argmin(loss))]
    candidates = lift(best)
    return dict(initial_alias_count=len(starts), fitted_xy=estimates, losses=loss,
                best_xy_error=np.linalg.norm(best-u[:2]),
                stacked_geometry_rank=np.linalg.matrix_rank(bfit.reshape(-1, 3)),
                mirror_observation_difference=np.max(abs(np.exp(1j*bfit@candidates[0])-np.exp(1j*bfit@candidates[1]))),
                premise='Externally supplied simulated yaw readings with 0 to 0.4 degree error; static world source and stable channel calibration')


def pose_geometry():
    return Rotation.from_euler('xyz', [[0, 0, 0], [25, 0, 20], [0, 30, 50], [-20, 10, 80]], degrees=True).as_matrix()


def sphere_fit(z, geometry, starts):
    outputs = []
    for az, el in starts:
        def residual(ae):
            e = np.exp(1j*geometry@unit(*ae))-z
            return np.r_[e.real.ravel(), e.imag.ravel()]
        fit = least_squares(residual, [az, el])
        outputs.append((np.linalg.norm(fit.fun), unit(*fit.x)))
    return sorted(outputs, key=lambda item: item[0])


def round077():
    u = unit(67, -25)
    actual = pose_geometry()
    errors = rng().normal(0, np.deg2rad(.25), (4, 3))
    measured = Rotation.from_rotvec(errors).as_matrix()@actual
    b = P[1:]-P[0]
    btrue = K*np.einsum('tij,mj->tmi', actual, b)
    bfit = K*np.einsum('tij,mj->tmi', measured, b)
    z = np.exp(1j*btrue@u)
    fits = sphere_fit(z, bfit, itertools.product(range(0, 360, 60), [-45, 45]))
    best = fits[0][1]
    mirror = u*np.array([1, 1, -1])
    return dict(best_error_deg=angle(best, u), best_residual=fits[0][0],
                mirror_residual=np.linalg.norm(np.exp(1j*bfit@mirror)-z),
                stacked_geometry_rank=np.linalg.matrix_rank(bfit.reshape(-1, 3)),
                premise='Synthetic multi-axis pose observations with 0.25 degree component error, stationary world source; not measured IMU',
                optimizer='12 sphere starts, no global uniqueness proof')


def round078():
    u = unit(67, -25)
    actual = pose_geometry()
    btrue = K*np.einsum('tij,mj->tmi', actual, P[1:]-P[0])
    generator = rng()
    phase_sigma, pose_sigma = .01, np.deg2rad(1.)
    covered_naive, covered_pose, errors = [], [], []
    for _ in range(500):
        measured = Rotation.from_rotvec(generator.normal(0, pose_sigma, (4, 3))).as_matrix()@actual
        b = K*np.einsum('tij,mj->tmi', measured, P[1:]-P[0])
        y = (btrue@u+generator.normal(0, phase_sigma, (4, 3))).ravel()
        a = b.reshape(-1, 3)
        estimate = np.linalg.lstsq(a, y, rcond=None)[0]
        estimate /= np.linalg.norm(estimate)
        tangent = np.linalg.svd(estimate[None])[2][1:].T
        # Jacobian of normalized unconstrained LS, the actual estimator used.
        linear = tangent.T@np.linalg.pinv(a)
        covariance = phase_sigma**2*np.eye(12)
        augmented = covariance.copy()
        for t in range(4):
            h = np.cross(b[t], estimate)
            augmented[t*3:t*3+3, t*3:t*3+3] += pose_sigma**2*h@h.T
        e = tangent.T@(u-estimate)
        naive_cov, pose_cov = linear@covariance@linear.T, linear@augmented@linear.T
        covered_naive.append(e@np.linalg.solve(naive_cov, e) <= 5.991464547)
        covered_pose.append(e@np.linalg.solve(pose_cov, e) <= 5.991464547)
        errors.append(angle(estimate, u))
    return dict(phase_only_95_coverage=np.mean(covered_naive), pose_augmented_95_coverage=np.mean(covered_pose),
                median_error_deg=np.median(errors), p95_error_deg=np.quantile(errors, .95),
                cases=500, premise='Known independent Gaussian pose covariance supplied explicitly; local phase branch; covariance Jacobian evaluated at estimated direction, not truth')


def round079():
    u = unit(67, 25)
    baseline = P[1:]-P[0]
    phase_l1 = np.angle(np.exp(1j*K*baseline@u))
    qs = aliases(phase_l1)
    k5 = 2*np.pi/(299792458/1176.45e6)
    phase_l5 = np.exp(1j*k5*baseline@u)
    losses = np.linalg.norm(np.exp(1j*k5*baseline[:, :2]@qs.T)-phase_l5[:, None], axis=0)
    accepted = qs[losses<1e-8]
    return dict(l1_xy_candidates=qs, l5_residuals=losses, dual_xy_candidates=accepted,
                dual_full_sphere_count=2*len(accepted),
                premise='Same far-field source at L1/L5; separately resolved channel phase calibration and symbol polarity on both bands; noiseless identifiability experiment',
                mirror_remains=True)


def round080():
    u = unit(35, 65)
    reflection = unit(150, 30)
    direct = np.exp(1j*K*P@u)
    reflected = direct+.7*np.exp(.6j)*np.exp(1j*K*P@reflection)
    q, residual = complex_fit(reflected, True)
    planar_mirror = u*np.array([1, 1, -1])
    mirror_mix = direct+.7*np.exp(.6j)*np.exp(1j*K*P@planar_mirror)
    qm, mirror_residual = complex_fit(mirror_mix, True)
    return dict(generic_reflection_xy_error=np.linalg.norm(q-u[:2]),
                generic_reflection_phase_residual=residual,
                mirror_reflection_xy_error=np.linalg.norm(qm-u[:2]),
                mirror_reflection_phase_residual=mirror_residual,
                mirror_mix_rank1_identity_error=np.max(abs(mirror_mix-(1+.7*np.exp(.6j))*direct)),
                premise='ONE emitter with coherent direct and reflected paths, static narrowband model',
                conclusion='Residual can reveal some manifold mismatch, but zero residual does not certify direct-path source localization or source authenticity')
