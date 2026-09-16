"""Eight causal diagnostics for GNSS/INS consistency, not authentication.

At 10 Hz, epoch 100 is an explicit reference checkpoint. Fixed references do
not refresh from later GNSS; rolling references do and can be contaminated.
The nine-state input is a provisional causal filter run; an alert can reject
that epoch's provisional update by replaying the identical preceding prefix.
"""
import numpy as np

NAMES = ('mean_v5', 'mean_v10', 'cusum_v', 'bias_change', 'bias_rate',
         'fixed_v', 'fixed_p', 'rolling_v')


def monitor_scores(inertial, position, velocity, provisional):
    t = inertial['time_s']; ip = inertial['position_m']; iv = inertial['velocity_mps']
    x = provisional['state']; r = provisional['innovation'][:, 3:5]
    scores = np.zeros((8, len(position))); cumulative = np.vstack((np.zeros(2), np.cumsum(r, axis=0)))
    plus = np.zeros(2); minus = np.zeros(2); anchor = 100
    for k in range(anchor+1, len(t)):
        j = k-1; dt = t[k]-t[k-1]
        scores[0, j] = np.linalg.norm((cumulative[k]-cumulative[k-50])/50)
        scores[1, j] = np.linalg.norm((cumulative[k]-cumulative[k-100])/100)
        plus = np.maximum(0, plus+(r[j]-.02)*dt)
        minus = np.maximum(0, minus+(-r[j]-.02)*dt)
        scores[2, j] = max(np.linalg.norm(plus), np.linalg.norm(minus))
        scores[3, j] = np.linalg.norm(x[k, 6:8]-x[anchor, 6:8])
        scores[4, j] = np.linalg.norm(x[k, 6:8]-x[k-100, 6:8])/(t[k]-t[k-100])
        age = t[k]-t[anchor]
        v = x[anchor, 3:6]+iv[k]-iv[anchor]-x[anchor, 6:]*age
        p = (x[anchor, :3]+x[anchor, 3:6]*age+ip[k]-ip[anchor]-iv[anchor]*age
             -.5*x[anchor, 6:]*age**2)
        scores[5, j] = np.linalg.norm(velocity[j, :2]-v[:2])
        scores[6, j] = np.linalg.norm(position[j, :2]-p[:2])
        a = k-100; age = t[k]-t[a]
        v = x[a, 3:6]+iv[k]-iv[a]-x[a, 6:]*age
        scores[7, j] = np.linalg.norm(velocity[j, :2]-v[:2])
    return scores


def first_evidence(score, limit, confirmation, already_isolated):
    """Return zero-based observation index; None is an observed non-detection."""
    run = 0
    for k in range(100, len(score)):
        run = run+1 if score[k] > limit[k] else 0
        if run >= confirmation:
            return k if not already_isolated[k+1] else None
        if already_isolated[k+1]:
            return None
    return None
