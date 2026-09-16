"""Assisted, stop-and-hop C/A IQ recovery research, not a continuous RF receiver.

Each observation is an equivalent coherent average of repeated 1 ms codes with
constant geometry/carrier and +1 data. No unknown Doppler or data-bit decoding.
Truth belongs to synthesize/evaluation, never observe or array_weights.
"""
import numpy as np
from scipy.optimize import minimize_scalar

from gnss import ca_code
from positioning import solve_position
from presets import mate70_phase_centers
from run_gps_spp import receiver_reference

C = 299792458.
FS = 2.046e6
N = 2046
WAVELENGTH = C / 1575.42e6
FREQUENCY = np.fft.fftfreq(N)


def codes(count):
    samples = np.stack([np.repeat(ca_code(i + 1), 2) for i in range(count)])
    spectrum = np.fft.fft(samples, axis=1)
    spectrum[:, abs(FREQUENCY) > .45] = 0
    return spectrum / np.sqrt(np.mean(abs(np.fft.ifft(spectrum, axis=1))**2, axis=1))[:, None]


def sky(seed, count, clustered=False):
    rng = np.random.default_rng(seed)
    az = rng.uniform(0, 2*np.pi, 24)[:count]
    el = np.arcsin(rng.uniform(np.sin(np.radians(15)), .97, 24)[:count])
    if clustered:
        az = .5 + .1 * az
        el = .5 + .1 * el
    return np.column_stack((np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)))


def yaw_matrix(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def steering(directions, positions, yaw):
    return np.exp(2j*np.pi/WAVELENGTH * (positions @ yaw_matrix(yaw).T @ directions.T))


def synthesize(spectrum, directions, position, fake_position, source, positions,
               yaw, gain, noise, cn0, coherent_s, spoof_db, spoof_count,
               genuine=True, multipath=0., clip=0.):
    """Generate common-source forgery of PRNs, with geometrically consistent ranges."""
    count = len(directions)
    ranges0 = 21e6 + np.arange(count)*17000.
    reference, rotation = receiver_reference()
    satellites = reference + (directions * ranges0[:, None]) @ rotation
    ranges = np.linalg.norm(directions*ranges0[:, None]-position, axis=1) + 75000.
    forged = np.linalg.norm(directions*ranges0[:, None]-fake_position, axis=1) + 75000.
    def waveform(rho):
        return np.fft.ifft(spectrum * np.exp(-2j*np.pi * (rho*FS/C)[:, None] * FREQUENCY), axis=1) * np.exp(-2j*np.pi*rho[:, None]/WAVELENGTH)
    actual = steering(directions, positions, yaw)
    source_vector = steering(source[None], positions, yaw)[:, 0]
    clean = actual @ waveform(ranges) * genuine
    if multipath:
        reflected = directions.copy(); reflected[:, 2] *= -1
        clean += multipath * steering(reflected, positions, yaw) @ waveform(ranges + 30.) * genuine
    counterfeit = np.sqrt(10**(spoof_db/10)) * source_vector[:, None] * waveform(forged)[:spoof_count].sum(axis=0)
    # Noise is added before ADC clipping, common exact realization for all policies.
    x = gain[:, None] * (clean + counterfeit) + np.sqrt(N/(10**(cn0/10)*coherent_s)) * noise
    fraction = 0.
    if clip:
        fraction = float(np.mean((abs(x.real)>clip) | (abs(x.imag)>clip)))
        x = np.clip(x.real, -clip, clip) + 1j*np.clip(x.imag, -clip, clip)
    return x, satellites, dict(ranges=ranges, actual=actual*gain[:, None],
                              source_vector=source_vector*gain, clip_fraction=fraction)


def array_weights(x, desired, method, loading=.01, ratio_limit=3., min_gain=.05):
    """Sample-only dominant direction; ephemeris/estimated-pose desired responses."""
    antennas = len(x)
    count = desired.shape[1]
    covariance = x @ x.conj().T / x.shape[1]
    eigenvalues, vectors = np.linalg.eigh(covariance)
    ratio = float(eigenvalues[-1] / np.mean(eigenvalues[:-1]))
    dominant = vectors[:, -1]
    projection = np.eye(antennas) - np.outer(dominant, dominant.conj())
    usable = np.ones(count, bool)
    if method == 'single':
        weights = np.zeros((antennas, count), complex); weights[0] = 1
    elif method == 'beam':
        weights = desired / antennas
    elif method == 'mvdr':
        loaded = covariance + loading*np.trace(covariance).real/antennas*np.eye(antennas)
        solved = np.linalg.solve(loaded, desired)
        weights = solved / np.sum(desired.conj()*solved, axis=0)
    elif method == 'blind':
        reference = np.eye(antennas)[:, 0]
        weights = np.tile((projection @ reference)[:, None], (1, count))
    else:
        projected = projection @ desired
        response = np.sum(desired.conj()*projected, axis=0).real
        # Model decision: irrecoverable near-null directions are excluded, not repaired.
        usable = response / antennas >= min_gain
        weights = projected
        if method == 'gated' and ratio < ratio_limit:
            weights = desired / antennas
            usable[:] = True
    return weights, usable, dict(eigen_ratio=ratio, dominant=dominant)


def observe(x, spectrum, weights, coarse_ranges, prior_delay=None, search_radius=2.):
    """FFT acquisition + local bandlimited delay ML; prior enables code tracking.

    Acquisition and tracking operate on the same measured correlation function.
    The prior is the preceding observed delay, never a source delay. Full search
    reacquires when the local peak is < 1/2 of the global measured peak.
    """
    transformed = np.fft.fft(x, axis=1)
    cross = (weights.conj().T @ transformed) * spectrum.conj()
    correlation = np.fft.ifft(cross, axis=1)
    power = abs(correlation)**2
    peaks = np.argmax(power, axis=1)
    reacquired = np.ones(len(peaks), bool)
    if prior_delay is not None:
        offsets = np.arange(-int(search_radius), int(search_radius)+1)
        local_indices = (np.rint(prior_delay).astype(int)[:, None]+offsets) % N
        local_power = np.take_along_axis(power, local_indices, axis=1)
        local_best = np.argmax(local_power, axis=1)
        local_peaks = local_indices[np.arange(len(peaks)), local_best]
        use_local = power[np.arange(len(peaks)), local_peaks] >= .5*power[np.arange(len(peaks)), peaks]
        peaks = np.where(use_local, local_peaks, peaks)
        reacquired = ~use_local
    delays = np.empty(len(peaks)); quality = np.empty(len(peaks))
    for s, peak in enumerate(peaks):
        # Only +/- one sample interpolation; the global search must locate its basin.
        def objective(delta):
            return -abs(np.sum(cross[s]*np.exp(2j*np.pi*FREQUENCY*(peak+delta))))**2
        fit = minimize_scalar(objective, bounds=(-1., 1.), method='bounded', options={'xatol':1e-5})
        delays[s] = (peak+fit.x) % N
        quality[s] = power[s, peak]/np.median(power[s])
    fractional = delays*C/FS
    ranges = fractional + np.rint((coarse_ranges-fractional)/(C*.001))*(C*.001)
    return ranges, delays, quality, reacquired


def position_solution(satellites, ranges, valid):
    """Explicit observability outcomes are research results, not default fixes."""
    if valid.sum() < 4:
        return np.full(3, np.nan), np.nan, np.nan
    reference, rotation = receiver_reference()
    # Rank/geometry eligibility is part of the estimator's measurement admission.
    direction = satellites[valid]-reference
    design = np.column_stack((-direction/np.linalg.norm(direction, axis=1)[:, None], np.ones(valid.sum())))
    condition = np.linalg.cond(design)
    if condition > 1000.:
        return np.full(3, np.nan), condition, np.nan
    result = solve_position(satellites[valid], ranges[valid])
    return (result['position_m']-reference) @ rotation.T, condition, float(np.sqrt(np.mean(result['residual_m']**2)))


def refine_source(x, spectrum, desired, coarse, weights, iterations, min_gain):
    """Alternate fitting genuine-code templates and sample-only residual direction.

    Delays and complex amplitudes are estimated from the current IQ. A fixed
    ridge of 1e-3 of the mean Gram diagonal regularizes the template fit; this is
    an estimator definition, not a singular-matrix exception fallback.
    """
    antennas = len(x)
    for _ in range(iterations):
        _, delays, _, _ = observe(x, spectrum, weights, coarse)
        wave = np.fft.ifft(spectrum*np.exp(-2j*np.pi*delays[:,None]*FREQUENCY),axis=1)
        # Fit only in the current interference-null subspace. Otherwise the
        # strong forgery would be absorbed into the supposed genuine templates.
        _, _, right = np.linalg.svd(weights.conj().T, full_matrices=False)
        # Weights have rank antennas-1 for projection; construct that subspace.
        basis = right[:antennas-1].conj().T
        projected_x = basis.conj().T @ x
        projected_a = basis.conj().T @ desired
        gram = (projected_a.conj().T @ projected_a) * (wave.conj() @ wave.T)
        rhs = np.sum(projected_a.conj()*(projected_x @ wave.conj().T),axis=0)
        amplitude = np.linalg.solve(gram+1e-3*np.trace(gram).real/len(gram)*np.eye(len(gram)),rhs)
        residual = x - desired @ (amplitude[:,None]*wave)
        _, vectors = np.linalg.eigh(residual @ residual.conj().T / N)
        v = vectors[:,-1]
        projector = np.eye(antennas)-np.outer(v,v.conj())
        weights = projector @ desired
    retained = np.sum(desired.conj()*weights,axis=0).real/antennas
    return weights, retained >= min_gain
