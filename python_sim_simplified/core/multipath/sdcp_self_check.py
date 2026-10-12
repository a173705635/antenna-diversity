"""Conservative causal SDCP quality gate; no truth input or offset refitting."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SelfCheckConfig:
    window: int = 50
    block: int = 10
    confirmation: int = 10
    minimum_satellites: int = 6
    noise_k: float = 3.
    threshold_cap: float = 2.
    geometry_limit_m: float = .005
    jump_m: float = .04
    epoch_s: float = .01


def circular_mean(x, wavelength, axis=0):
    return np.angle(np.mean(np.exp(2j*np.pi*x/wavelength), axis=axis))*wavelength/(2*np.pi)


def detect(time_s, residual_m, wavelength_m, threshold_floor_m, *,
           valid, geometry_uncertainty_m, channel_skew_s, phase_reset_valid=False,
           config=SelfCheckConfig()):
    """Return 0=no anomaly, 1=suspected multipath, -1=unreliable.

    valid must include caller's lock and half-cycle validity. Unreported geometry
    errors and integer-cycle jumps are not identifiable from wrapped SDCP alone.
    Noise estimate uses sub-block means; not a certified standard error or CFAR.
    """
    t = np.asarray(time_s, float)
    r = np.asarray(residual_m, float)
    c = config
    if (r.ndim != 2 or t.shape != (len(r),) or not np.isfinite(t).all()
            or np.any(np.diff(t) <= 0) or not np.isfinite(wavelength_m)
            or wavelength_m <= 0 or not np.isfinite(threshold_floor_m)
            or threshold_floor_m <= 0 or c.window < 2*c.block
            or c.block < 1 or c.window % c.block or c.confirmation < 1):
        raise ValueError('invalid aligned observations or configuration')
    valid = np.broadcast_to(np.asarray(valid, bool), r.shape).copy()
    uncertainty = np.broadcast_to(np.asarray(geometry_uncertainty_m, float), r.shape)
    skew = np.broadcast_to(np.asarray(channel_skew_s, float), t.shape)
    quality = valid & np.isfinite(r) & np.isfinite(uncertainty) & (uncertainty >= 0)
    quality &= uncertainty <= c.geometry_limit_m
    quality &= (np.isfinite(skew) & (abs(skew) <= 1e-6))[:, None]
    gaps = np.r_[False, abs(np.diff(t)-c.epoch_s) > 1e-6]
    quality[gaps] = False
    delta = (np.diff(r, axis=0)+wavelength_m/2) % wavelength_m-wavelength_m/2
    quality[1:] &= np.isfinite(delta)
    # A discontinuity cannot be repaired by merely waiting out a window.
    # Reset requires external phase-branch validation, not an anomaly label.
    reset = np.broadcast_to(np.asarray(phase_reset_valid, bool), r.shape)
    quarantined = np.zeros(r.shape[1], bool)
    for k in range(1, len(r)):
        quarantined |= np.isfinite(delta[k-1]) & (abs(delta[k-1]) > c.jump_m)
        quarantined[reset[k]] = False
        quality[k] &= ~quarantined
    shape = r.shape
    state = np.full(shape, -1, int)
    reason = np.full(shape, 'warmup', dtype='<U24')
    score = np.full(shape, np.nan)
    threshold = np.full(shape, np.nan)
    fixed = np.zeros(shape, bool)
    fixed_confirmed = np.zeros(shape, bool)
    fixed_streak = np.zeros(shape[1], int)
    streak = np.zeros(shape[1], int)
    for k in range(c.window-1, len(r)):
        sl = slice(k-c.window+1, k+1)
        usable = quality[sl].all(axis=0)
        reason[k] = 'quality'
        safe = np.where(np.isfinite(r[sl]), r[sl], 0.)
        mean = circular_mean(safe, wavelength_m)
        blocks = safe.reshape(-1, c.block, shape[1])
        bm = circular_mean(blocks, wavelength_m, axis=1)
        centered = (bm-mean+wavelength_m/2) % wavelength_m-wavelength_m/2
        noise = np.std(centered, axis=0, ddof=1)/np.sqrt(len(bm))
        candidate = threshold_floor_m+c.noise_k*noise+np.max(uncertainty[sl], axis=0)
        score[k] = abs(mean)
        threshold[k] = candidate
        fixed[k] = abs(mean) > threshold_floor_m
        fixed_streak = np.where(fixed[k], fixed_streak+1, 0)
        fixed_confirmed[k] = fixed_streak >= c.confirmation
        noisy = candidate > c.threshold_cap*threshold_floor_m
        reason[k, usable & noisy] = 'noise_limit'
        usable &= ~noisy
        if usable.sum() < c.minimum_satellites:
            reason[k, usable] = 'few_satellites'
            streak[:] = 0
            continue
        alarm = abs(mean) > candidate
        if np.sum(alarm & usable) >= np.ceil(usable.sum()/2):
            reason[k, usable] = 'common_ambiguity'
            streak[:] = 0
            continue
        streak = np.where(usable & alarm, streak+1, 0)
        state[k, usable] = (streak[usable] >= c.confirmation).astype(int)
        reason[k, usable] = np.where(alarm[usable], 'confirming', 'clear')
        reason[k, state[k] == 1] = 'persistent_anomaly'
    return dict(state=state, reason=reason, score_m=score, threshold_m=threshold,
                fixed_alarm=fixed, fixed_confirmed=fixed_confirmed)
