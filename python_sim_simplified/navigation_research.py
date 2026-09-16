"""Explicit research policies at PVT epochs; no truth or attack-time inputs.

Optional extra states estimate effective navigation-frame acceleration error,
NOT physical attitude, accelerometer bias and gyro bias separately. Covariance
is a filter tuning model, not an integrity bound. All samples are synchronous.
"""
import numpy as np


def candidate_modes(scores, threshold, confirmation, quiet_windows):
    """Complete 100 ms windows; -1 pending hold, 1 evidence, 0 allow updates.

    quiet_windows=0 allows updates before confirmation. Other settings hold
    immediately on raw evidence and release a *pending* hold after quiet windows.
    A confirmed navigation isolation is never released here.
    """
    run = 0; quiet = 0; pending = False; modes = []
    for score in scores:
        raw = score < threshold
        run = run+1 if raw else 0
        quiet = 0 if raw else quiet+1
        if raw:
            pending = True
        elif quiet >= quiet_windows:
            pending = False
        modes.append(1 if run >= confirmation else -1 if pending and quiet_windows else 0)
    return np.array(modes)


def navigate(inertial, position, velocity, spatial_modes, *, estimate_acceleration,
             velocity_limit, position_limit, mean_velocity_limit, admitted=True):
    """One GNSS observation per inertial edge after edge 0, decision already due.

    Gate norms use horizontal innovations BEFORE the current update. Limits are
    explicit calibrated values; infinity disables that particular research test.
    Optional 1 s mean uses the last ten pre-update velocity innovations.
    """
    t = inertial['time_s']; ip = inertial['position_m']; iv = inertial['velocity_mps']
    n = 9 if estimate_acceleration else 6
    x = np.zeros(n); x[:6] = np.r_[ip[0], iv[0]]
    p = np.diag([9.]*3+[.04]*3+([.05**2]*3 if estimate_acceleration else []))
    h = np.eye(6, n); r = np.diag([9.]*3+[.04]*3)
    states = [x.copy()]; modes = [-1]; innovations = []; monitor = []
    accepted = [False]; isolated = False; reasons = ['startup']
    for k, dt in enumerate(np.diff(t), 1):
        f = np.eye(n); f[:3, 3:6] = np.eye(3)*dt
        if estimate_acceleration:
            f[:3, 6:] = -np.eye(3)*dt**2/2
            f[3:6, 6:] = -np.eye(3)*dt
        u = np.zeros(n); u[:6] = np.r_[ip[k]-ip[k-1]-iv[k-1]*dt, iv[k]-iv[k-1]]
        x = f@x+u
        q = np.zeros((n, n)); a = np.eye(3)*.02**2
        q[:6, :6] = np.block([[a*dt**3/3, a*dt**2/2], [a*dt**2/2, a*dt]])
        if estimate_acceleration:
            # Continuous acceleration-error random walk, including cross terms.
            b = np.eye(3)*.0005**2
            q += np.block([[b*dt**5/20, b*dt**4/8, -b*dt**3/6],
                           [b*dt**4/8, b*dt**3/3, -b*dt**2/2],
                           [-b*dt**3/6, -b*dt**2/2, b*dt]])
        p = f@p@f.T+q
        innovation = np.r_[position[k-1], velocity[k-1]]-h@x
        innovations.append(innovation.copy())
        # No partial one-second window is credited at startup.
        mean_v = np.linalg.norm(np.mean(np.array(innovations[-10:])[:, 3:5], axis=0)) if k >= 10 else 0.
        monitor.append(mean_v)
        abnormal = (np.linalg.norm(innovation[3:5]) > velocity_limit or
                    np.linalg.norm(innovation[:2]) > position_limit or mean_v > mean_velocity_limit)
        isolated |= spatial_modes[k-1] == 1 or abnormal
        mode = 1 if isolated else int(spatial_modes[k-1])
        if not admitted:
            mode = -1
        if mode == 0:
            gain = np.linalg.solve(h@p@h.T+r, (p@h.T).T).T
            x += gain@innovation
            residual = np.eye(n)-gain@h
            p = residual@p@residual.T+gain@r@gain.T
        states.append(x.copy()); modes.append(mode); accepted.append(mode == 0)
        reasons.append('innovation' if abnormal else 'spatial' if spatial_modes[k-1] == 1
                       else 'latched' if isolated else 'pending' if mode == -1 else 'update')
    return dict(time_s=t, state=np.array(states), mode=np.array(modes),
                gnss_accepted=np.array(accepted), innovation=np.array(innovations),
                mean_velocity_innovation=np.array(monitor), reason=reasons)
