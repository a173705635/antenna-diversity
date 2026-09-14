"""Single-frequency dual-antenna SDCP weighting (A1 process 400 front end).

Geometry uses this repository's negative spatial-phase convention. Reference
selection and sigmoid scale are engineering choices, not patent constants.
"""
import numpy as np
from scipy.special import expit


def geometry_free_sdcp(time_l1_s, a_l1_m, b_l1_m,
                       time_l5_s, a_l5_m, b_l5_m):
    """Raw (B-A)L1 - (B-A)L5, in range metres, columns in identical PRN order.

    Common geometry cancels, but ambiguities and hardware biases remain.
    This is neither a calibrated multipath residual nor a detector. No fitting,
    wrapping or interpolation is performed; callers must select a valid arc.
    """
    t, u = np.asarray(time_l1_s, float), np.asarray(time_l5_s, float)
    values = [np.asarray(v, float) for v in (a_l1_m, b_l1_m, a_l5_m, b_l5_m)]
    if (t.ndim != 1 or not t.size or not np.isfinite(t).all()
            or np.any(np.diff(t) <= 0) or not np.array_equal(t, u)
            or values[0].ndim != 2 or values[0].shape[0] != t.size
            or values[0].shape[1] == 0
            or any(v.shape != values[0].shape or not np.isfinite(v).all()
                   for v in values)):
        raise ValueError('finite same-PRN matrices and exactly aligned increasing times required')
    a1, b1, a5, b5 = values
    return (b1-a1)-(b5-a5)


def geometry_offset_m(baseline_m, azimuth_deg, elevation_deg):
    baseline = np.asarray(baseline_m, dtype=float)
    if baseline.shape != (3,) or not np.all(np.isfinite(baseline)):
        raise ValueError('finite three-component baseline required')
    if not np.all(np.isfinite([azimuth_deg, elevation_deg])):
        raise ValueError('finite direction required')
    az, el = np.radians([azimuth_deg, elevation_deg])
    return float(-baseline @ np.array([np.cos(el)*np.cos(az),
                                      np.cos(el)*np.sin(az), np.sin(el)]))


def modified_sdcp(time_s, phase_difference_rad, wavelength_m, geometry_m,
                  reference_start_s=.2, reference_stop_s=.4):
    time = np.asarray(time_s, dtype=float)
    phase = np.asarray(phase_difference_rad, dtype=float)
    geometry = np.asarray(geometry_m, dtype=float)
    if (time.ndim != 1 or phase.shape != time.shape or not time.size or
            np.any(np.diff(time) <= 0) or not np.all(np.isfinite(time)) or
            not np.all(np.isfinite(phase)) or not np.all(np.isfinite(geometry)) or
            not np.isfinite(wavelength_m) or wavelength_m <= 0 or
            geometry.shape not in ((), time.shape)):
        raise ValueError('finite aligned observations, geometry and positive wavelength required')
    mask = (time >= reference_start_s) & (time < reference_stop_s)
    if mask.sum() < 2:
        raise ValueError('reference interval requires at least two observations')
    sdcp = phase*wavelength_m/(2*np.pi)
    compensated = sdcp-geometry
    offset = float(np.median(compensated[mask]))
    # Do not wrap residuals or re-fit an offset on the evaluation window.
    return {'sdcp_m': sdcp, 'geometry_compensated_m': compensated,
            'reference_offset_m': offset, 'residual_m': compensated-offset,
            'reference_epochs': int(mask.sum())}


def observation_weights(residual_m, threshold_m):
    residual = np.asarray(residual_m, dtype=float)
    if (residual.ndim != 1 or not np.all(np.isfinite(residual)) or
            not np.isfinite(threshold_m) or threshold_m <= 0):
        raise ValueError('finite residual vector and positive threshold required')
    magnitude = abs(residual)
    soft = expit((threshold_m-magnitude)/(threshold_m/4))
    hard = (magnitude <= threshold_m).astype(float)
    return {'soft_pair': np.column_stack([soft, soft]),
            'hard_pair': np.column_stack([hard, hard])}


def calibrated_sdcp(sdcp_m, geometry_m, wavelength_m, channel_bias_m):
    """Wrapped residual with externally calibrated bias; no temporal fitting.

    Requires integer-cycle coherent phases. Half cycles and unknown hardware
    phase are not resolved here. This is not an integer ambiguity fix.
    """
    values = [np.asarray(v, dtype=float) for v in (sdcp_m, geometry_m, channel_bias_m)]
    if (not all(np.isfinite(v).all() for v in values) or
            not np.isfinite(wavelength_m) or wavelength_m <= 0):
        raise ValueError('finite observations/calibration and positive wavelength required')
    compensated = values[0]-values[1]-values[2]
    return (compensated+wavelength_m/2) % wavelength_m-wavelength_m/2
