"""Research-only planar-array direction candidates and inter-PRN differences.

Angles are in the ARRAY frame (x east, y north, z up only if installed so).
Known channel calibration is an explicit input; unknown biases cannot be
estimated jointly with an unknown common-source direction from these data.
"""
from itertools import combinations
import numpy as np
from scipy.ndimage import minimum_filter
from scipy.optimize import least_squares


def double_difference_scores(carrier_m, wavelength_m):
    """Mean 1-cos of doubled inter-PRN double differences, all or best 3 PRNs.

    Same scale [0,1] as the previous score is NOT implied (range is [0,2]).
    Shared baselines and PRN references are correlated. Calibrate separately.
    """
    x = np.asarray(carrier_m, float)
    if x.ndim < 3 or x.shape[-3] < 1 or not 2 <= x.shape[-2] <= 4 or x.shape[-1] < 3:
        raise ValueError('epochs x 2-4 antennas x >=3 PRNs required')
    if not np.isfinite(x).all() or not np.isfinite(wavelength_m) or wavelength_m <= 0:
        raise ValueError('finite phase and positive wavelength required')
    pairs = np.array(list(combinations(range(x.shape[-2]), 2)))
    triples = np.array(list(combinations(range(x.shape[-1]), 3)))
    z = np.exp(4j*np.pi*(x[..., pairs[:, 1], :]-x[..., pairs[:, 0], :])/wavelength_m)
    n = x.shape[-1]
    # Exactly equal to averaging explicit 1-Re(z_s conj(z_r)) over all s<r.
    global_score = (n/(n-1)*(1-abs(z.mean(axis=-1))**2)).mean(axis=(-2, -1))
    subset = (1.5*(1-abs(z[..., triples].mean(axis=-1))**2)).mean(axis=(-3, -2))
    return {'global': np.maximum(global_score, 0), 'cluster3': np.maximum(subset.min(axis=-1), 0)}


def select_common_prns(carrier_m, wavelength_m):
    """Data-only best triple under the original circular-dispersion score."""
    x = np.asarray(carrier_m, float)
    if x.ndim != 3 or x.shape[2] < 3 or not np.isfinite(x).all():
        raise ValueError('finite epoch x antenna x PRN block required')
    triples = np.array(list(combinations(range(x.shape[2]), 3)))
    pairs = np.array(list(combinations(range(x.shape[1]), 2)))
    z = np.exp(4j*np.pi*(x[:, pairs[:, 1]]-x[:, pairs[:, 0]])/wavelength_m)
    scores = (1-abs(z[..., triples].mean(axis=-1))).mean(axis=(0, 1))
    return triples[int(scores.argmin())].tolist()


def calibrate_channel_bias(carrier_m, positions_m, known_directions, wavelength_m, harmonic=2):
    """Relative channel bias from independently trusted LOS directions.

    Only PRN-common channel bias is calibrated. This does not authenticate the
    calibration segment, handle arbitrary per-arc PRN biases, or track drift.
    Returns bias modulo wavelength/harmonic, adequate for that harmonic only.
    """
    x, p, u = map(lambda v: np.asarray(v, float), (carrier_m, positions_m, known_directions))
    if x.ndim != 3 or p.shape != (x.shape[1], 3) or u.shape != (x.shape[2], 3):
        raise ValueError('calibration dimensions do not match')
    if not all(np.isfinite(v).all() for v in (x, p, u)) or harmonic not in (1, 2) or wavelength_m <= 0:
        raise ValueError('invalid calibration values')
    predicted = -(p-p[0]) @ u.T
    observed = x-x[:, :1]
    z = np.exp(2j*np.pi*harmonic*(observed-predicted[None])/wavelength_m)
    mean = z.mean(axis=(0, 2))
    return {'bias_m': (np.angle(mean)*wavelength_m/(2*np.pi*harmonic)).tolist(),
            'concentration': abs(mean).tolist()}


def estimate_direction(carrier_m, positions_m, wavelength_m, *, channel_bias_m=None,
                       harmonic=2, fit_limit_m=.008, ambiguity_slack_m=.002):
    """Fit candidate directions, never use truth or a hidden hemisphere prior.

    Full upper-hemisphere grid (5 degrees), local refinement, then planar
    mirrors. Fit/slack limits are engineering tolerances, NOT probabilities.
    harmonic=2 tolerates half-cycle branches but retains extra aliases.
    harmonic=1 requires externally resolved half-cycle polarity.
    Candidate search is numerical; not a proof of exhaustive identifiability.
    """
    x, p = np.asarray(carrier_m, float), np.asarray(positions_m, float)
    if x.ndim != 3 or x.shape[0] < 1 or x.shape[2] < 1 or not 2 <= x.shape[1] <= 4 or p.shape != (x.shape[1], 3):
        raise ValueError('epoch x 2-4 antennas x PRNs and antenna x xyz required')
    if harmonic not in (1, 2) or not np.isfinite([wavelength_m, fit_limit_m, ambiguity_slack_m]).all() or wavelength_m <= 0 or fit_limit_m <= 0 or ambiguity_slack_m < 0:
        raise ValueError('invalid fit settings')
    def empty(status):
        return {'status': status, 'candidates': [], 'harmonic': harmonic}
    if not np.isfinite(x).all():
        return empty('invalid_observation')
    if not np.isfinite(p).all() or not np.allclose(p[:, 2], p[0, 2]):
        raise ValueError('this estimator supports planar xy arrays only')
    if channel_bias_m is None:
        return empty('uncalibrated')
    bias = np.asarray(channel_bias_m, float)
    if bias.shape != (len(p),) or not np.isfinite(bias).all():
        raise ValueError('finite per-channel calibrated biases required')
    if np.linalg.matrix_rank((p-p[0])[:, :2]) < 2:
        return empty('rank_deficient')
    pairs = np.array(list(combinations(range(len(p)), 2)))
    b = p[pairs[:, 1]]-p[pairs[:, 0]]
    corrected = x-bias[None, :, None]
    d = corrected[:, pairs[:, 1]]-corrected[:, pairs[:, 0]]
    scale = 2*np.pi*harmonic/wavelength_m
    z = np.exp(1j*scale*d)
    target = z.mean(axis=(0, 2))
    def unit(az, el):
        az, el = np.deg2rad(az), np.deg2rad(el)
        return np.stack([np.cos(el)*np.sin(az), np.cos(el)*np.cos(az), np.sin(el)], axis=-1)
    az, el = np.meshgrid(np.arange(0., 360., 5.), np.arange(0., 91., 5.), indexing='ij')
    vectors = unit(az, el)
    predicted = np.exp(-1j*scale*np.einsum('bc,ijc->ijb', b, vectors))
    loss = (abs(predicted-target)**2).mean(axis=-1)
    minima = np.argwhere(loss <= minimum_filter(loss, size=3, mode=('wrap', 'nearest')) + 1e-14)
    def residual(params):
        difference = np.exp(-1j*scale*(b @ unit(*params)))-target
        return np.r_[difference.real, difference.imag]
    fits = []
    for i, j in minima:
        optimum = least_squares(residual, [az[i, j], el[i, j]], bounds=([-np.inf, 0], [np.inf, 90]),
                                ftol=1e-11, xtol=1e-11, gtol=1e-11, max_nfev=200)
        u = unit(*optimum.x)
        if any(np.linalg.norm(u-r['unit_vector']) < 1e-3 for r in fits):
            continue
        pred = np.exp(-1j*scale*(b @ u))
        phase_error_m = np.angle(z*np.conj(pred[None, :, None]))/scale
        fits.append({'azimuth_deg': float(optimum.x[0] % 360), 'elevation_deg': float(optimum.x[1]),
                     'unit_vector': u.tolist(), 'rms_m': float(np.sqrt(np.mean(phase_error_m**2)))})
    fits.sort(key=lambda r: r['rms_m'])
    best = fits[0]['rms_m']
    accepted = [r for r in fits if r['rms_m'] <= min(fit_limit_m, best+ambiguity_slack_m)]
    candidates = []
    for r in accepted:
        candidates.append(r)
        if r['elevation_deg'] > .001:
            candidates.append({**r, 'elevation_deg': -r['elevation_deg'],
                               'unit_vector': [r['unit_vector'][0], r['unit_vector'][1], -r['unit_vector'][2]]})
    return {'status': 'rejected_fit' if not accepted else 'single_upper_candidate' if len(accepted)==1 else 'ambiguous',
            'candidates': candidates, 'best_rms_m': best, 'upper_candidate_count': len(accepted),
            'harmonic': harmonic, 'fit_limit_m': fit_limit_m, 'ambiguity_slack_m': ambiguity_slack_m,
            'calibration_assumed_valid': True, 'planar_mirror_unresolved': bool(accepted)}
