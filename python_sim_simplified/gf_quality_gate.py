"""Conditional GF anomaly gate. Does not identify multipath or verify branches.

Bounds must cover residual calibration error AND other non-multipath errors
for the statistic supplied (instantaneous or externally windowed). Unknown
bounds/arc validity yield unknown. A low result never proves no multipath.
"""
import numpy as np


def classify(residual_m, threshold_m, *, nuisance_bound_m, valid, branch_verified):
    """States: -1 unknown, 0 no above-threshold evidence, 1 suspected anomaly.

    residual_m is branch-corrected GF minus an EXTERNAL calibration estimate.
    No mean fitting, rounding, smoothing, confidence or rate claim is made here.
    branch_verified includes externally checked continuous/half-cycle-valid arcs.
    """
    r = np.asarray(residual_m, float)
    b = np.broadcast_to(np.asarray(nuisance_bound_m, float), r.shape)
    flags = []
    for flag in (valid, branch_verified):
        a = np.asarray(flag)
        flags.append(np.broadcast_to(a, r.shape))
    usable = np.isfinite(r) & np.isfinite(b) & (b >= 0) & flags[0] & flags[1]
    lower = np.full(r.shape, np.nan)
    upper = np.full(r.shape, np.nan)
    lower[usable] = np.maximum(0., abs(r[usable])-b[usable])
    upper[usable] = abs(r[usable])+b[usable]
    state = np.full(r.shape, -1, dtype=int)
    reason = np.full(r.shape, 'unverified_inputs', dtype='<U24')
    reason[usable] = 'overlaps_threshold'
    low = usable & (upper <= threshold_m)
    high = usable & (lower > threshold_m)
    state[low], reason[low] = 0, 'no_exceedance_evidence'
    state[high], reason[high] = 1, 'suspected_anomaly'
    return {'state':state, 'reason':reason, 'lower_m':lower, 'upper_m':upper}
