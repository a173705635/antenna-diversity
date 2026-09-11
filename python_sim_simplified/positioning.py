"""Assisted code observations and single-epoch GPS least-squares positioning."""

import numpy as np

from config import SPEED_OF_LIGHT_MPS


def _sigmas(value, shape):
    """Broadcast observation standard deviations; +inf explicitly removes a row."""
    try:
        sigma = np.broadcast_to(np.asarray(value, dtype=float), shape)
    except ValueError as exc:
        raise ValueError('sigmas must broadcast to observation shape') from exc
    if np.isnan(sigma).any() or np.any(sigma <= 0):
        raise ValueError('sigmas must be positive, with +inf for excluded observations')
    return sigma


def solve_position(satellite_positions_m, pseudoranges_m, *, sigma_m=1.):
    """Estimate ECEF position and clock range from corrected code observations.

    Satellite coordinates must already be in the reception-time ECEF frame.
    Satellite clock/atmosphere corrections belong upstream of this solver.
    No reference receiver position is supplied or used for initialization.
    """
    satellites = np.asarray(satellite_positions_m, dtype=float)
    ranges = np.asarray(pseudoranges_m, dtype=float)
    if (ranges.ndim != 1 or satellites.shape != (ranges.size, 3)
            or ranges.size < 4 or not np.isfinite(satellites).all()
            or not np.isfinite(ranges).all() or np.any(ranges <= 0)):
        raise ValueError('require at least four finite satellite positions and positive ranges')
    state = np.zeros(4)
    sigma = _sigmas(sigma_m, ranges.shape)
    for iteration in range(20):
        delta = state[:3]-satellites
        distance = np.linalg.norm(delta, axis=1)
        if np.any(distance < 1.):
            raise ValueError('receiver iterate coincides with a satellite')
        design = np.column_stack((delta/distance[:, None], np.ones(ranges.size)))
        residual = ranges-distance-state[3]
        update, _, rank, singular = np.linalg.lstsq(design/sigma[:, None], residual/sigma, rcond=None)
        if rank < 4 or singular[0]/singular[-1] > 1e8:
            raise ValueError('insufficient satellite geometry')
        state += update
        if not np.isfinite(state).all():
            raise ValueError('nonfinite position iterate')
        if np.linalg.norm(update) < 1e-5:
            residual = ranges-np.linalg.norm(state[:3]-satellites, axis=1)-state[3]
            return {'position_m': state[:3], 'clock_bias_m': float(state[3]),
                    'residual_m': residual, 'iterations': iteration+1,
                    'geometry_condition': float(singular[0]/singular[-1])}
    raise ValueError('position did not converge within 20 iterations')


def code_pseudorange(history, times_s, fs_hz, coherent_s, coarse_range_m,
                    *, code_rate_hz=1.023e6, code_length=1023):
    """Evaluate code NCO within its block and choose an aided 1 ms range period.

    The receiver nominal code epoch is t=0. Coarse range is external timing
    assistance, accurate to < half a code period (~150 km), not a fine correction.
    Missing tracking times return NaN and must be excluded before positioning.
    """
    times = np.asarray(times_s, dtype=float)
    if (times.ndim != 1 or not np.isfinite(times).all() or fs_hz <= 0
            or coherent_s <= 0 or not np.isfinite([fs_hz, coherent_s, coarse_range_m]).all()
            or coarse_range_m <= 0):
        raise ValueError('invalid observation time, sample rate or timing aid')
    result = np.full(times.shape, np.nan)
    if not history['success']:
        return result
    starts = np.asarray(history['epoch_start_samples'])/fs_hz
    phase = np.asarray(history['code_nco_phase_chips'])
    rate = np.asarray(history['code_nco_rate_hz'])
    if (starts.ndim != 1 or phase.shape != starts.shape or rate.shape != starts.shape
            or not np.isfinite(starts).all() or np.any(np.diff(starts) <= 0)):
        raise ValueError('invalid code NCO history')
    if starts.size == 0:
        return result
    index = np.searchsorted(starts, times, side='right')-1
    safe = np.clip(index, 0, starts.size-1)
    valid = (index >= 0) & (times < starts[safe]+coherent_s)
    local_phase = phase[safe]+rate[safe]*(times-starts[safe])
    if not np.isfinite([code_rate_hz, code_length]).all() or min(code_rate_hz, code_length) <= 0:
        raise ValueError('positive finite code rate and length required')
    period_m = SPEED_OF_LIGHT_MPS*code_length/code_rate_hz
    fractional = np.mod(times*code_rate_hz-local_phase, code_length)*SPEED_OF_LIGHT_MPS/code_rate_hz
    aided = fractional + np.rint((coarse_range_m-fractional)/period_m)*period_m
    result[valid] = aided[valid]
    return result


def carrier_range(history, times_s, fs_hz, coherent_s, wavelength_m):
    """NCO phase in range-sign convention, with unknown per-arc phase origin.

    Do not unwrap the accumulated NCO again or round it to whole wavelengths.
    The source uses exp(-j*2*pi*range/lambda), hence the minus sign here.
    """
    times = np.asarray(times_s, dtype=float)
    if (times.ndim != 1 or not np.isfinite(times).all()
            or not np.isfinite([fs_hz, coherent_s, wavelength_m]).all()
            or min(fs_hz, coherent_s, wavelength_m) <= 0):
        raise ValueError('invalid carrier observation timing or wavelength')
    result = np.full(times.shape, np.nan)
    if not history['success']:
        return result
    starts = np.asarray(history['epoch_start_samples'])/fs_hz
    phase = np.asarray(history['carrier_nco_phase_rad'])
    frequency = np.asarray(history['carrier_nco_frequency_hz'])
    if (starts.ndim != 1 or phase.shape != starts.shape or frequency.shape != starts.shape
            or not np.isfinite(starts).all() or np.any(np.diff(starts) <= 0)):
        raise ValueError('invalid carrier NCO history')
    if not starts.size:
        return result
    index = np.searchsorted(starts, times, side='right')-1
    safe = np.clip(index, 0, starts.size-1)
    valid = (index >= 0) & (times < starts[safe]+coherent_s)
    accumulated = phase[safe]+2*np.pi*frequency[safe]*(times-starts[safe])
    result[valid] = -accumulated[valid]*wavelength_m/(2*np.pi)
    return result


def solve_code_carrier_float(satellite_positions_m, code_m, carrier_m,
                             code_sigma_m=30., carrier_sigma_m=.01):
    """Offline joint float solution on one finite, continuous tracking arc.

    Each epoch has independent XYZ/clock; each satellite has one shared real
    phase bias. No static-position prior, integer fixing or truth input. Caller
    must split loss-of-lock/gaps and handle slips; this is not a slip detector.
    Coordinates must be in reception ECEF, with other corrections upstream.
    """
    from scipy.linalg import lstsq

    states = np.asarray(satellite_positions_m, dtype=float)
    code = np.asarray(code_m, dtype=float)
    raw_carrier = np.asarray(carrier_m, dtype=float)
    if (code.ndim != 2 or raw_carrier.shape != code.shape
            or states.shape != (*code.shape, 3) or code.shape[0] < 2 or code.shape[1] < 4
            or not all(np.isfinite(x).all() for x in (states, code, raw_carrier))
            or np.any(code <= 0)):
        raise ValueError('require >=2 epochs, >=4 satellites, finite continuous observations and positive sigmas')
    epochs, satellites = code.shape
    code_sigma = _sigmas(code_sigma_m, code.shape)
    phase_sigma = _sigmas(carrier_sigma_m, code.shape)
    cs, ls = code_sigma.ravel(), phase_sigma.ravel()
    # Observed code sets a numerical phase origin, not a true ambiguity value.
    phase_origin = code[0]-raw_carrier[0]
    carrier = raw_carrier+phase_origin
    estimates = [solve_position(s, p, sigma_m=w) for s, p, w in zip(states, code, code_sigma)]
    position = np.array([v['position_m'] for v in estimates])
    clock = np.array([v['clock_bias_m'] for v in estimates])
    bias = np.mean(carrier-code, axis=0)
    count = epochs*satellites
    epoch_index = np.repeat(np.arange(epochs), satellites)
    satellite_index = np.tile(np.arange(satellites), epochs)
    row = np.arange(count)
    for iteration in range(12):
        delta = position[:, None, :]-states
        distance = np.linalg.norm(delta, axis=2)
        if np.any(distance < 1.):
            raise ValueError('receiver iterate coincides with satellite')
        geometry = np.concatenate((delta/distance[:, :, None], np.ones((*code.shape, 1))), axis=2)
        code_residual = code-distance-clock[:, None]
        phase_residual = carrier-distance-clock[:, None]-bias
        design = np.zeros((2*count, 4*epochs+satellites))
        for axis in range(4):
            design[row, 4*epoch_index+axis] = geometry[:, :, axis].ravel()/cs
            design[count+row, 4*epoch_index+axis] = geometry[:, :, axis].ravel()/ls
        design[count+row, 4*epochs+satellite_index] = 1/ls
        residual = np.concatenate((code_residual.ravel()/cs,
                                   phase_residual.ravel()/ls))
        update, _, rank, _ = lstsq(design, residual, cond=1e-12, lapack_driver='gelsy')
        if rank != design.shape[1] or not np.isfinite(update).all():
            raise ValueError('rank deficient joint position/ambiguity system')
        step = update[:4*epochs].reshape(epochs, 4)
        position += step[:, :3]
        clock += step[:, 3]
        bias += update[4*epochs:]
        if np.max(np.abs(update)) < 1e-4:
            distance = np.linalg.norm(position[:, None, :]-states, axis=2)
            return {'position_m': position, 'clock_bias_m': clock,
                    'ambiguity_m': bias-phase_origin,
                    'centered_ambiguity_m': bias, 'phase_origin_shift_m': phase_origin,
                    'code_residual_m': code-distance-clock[:, None],
                    'carrier_residual_m': carrier-distance-clock[:, None]-bias,
                    'iterations': iteration+1, 'rank': rank,
                    'unknowns': design.shape[1], 'integer_fixed': False}
    raise ValueError('joint float solution did not converge')
