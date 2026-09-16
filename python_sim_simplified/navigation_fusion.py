"""Observable-level six-state GNSS/INS baseline, not an integrity monitor.

Attitude and bias estimates belong to the supplied inertial trajectory; GNSS
updates correct position/velocity only. No truth or attack schedule is consumed.
Independent admission is explicit. Spatial evidence latches GNSS isolation for
this run; a quiet window does not grant readmission. Unknown blocks skip updates.
"""
import numpy as np


def fuse(inertial, gnss_indices, gnss_position_m, gnss_velocity_mps, windows, *,
         initial_covariance, gnss_covariance, acceleration_noise_intensity,
         admitted, use_spatial):
    time = inertial['time_s']
    p = inertial['position_m']; v = inertial['velocity_mps']
    state = np.r_[p[0], v[0]]; covariance = initial_covariance.copy()
    output = [state.copy()]; covariances = [covariance.copy()]
    modes = [-1]; accepted = np.zeros(len(time), bool)
    decision = -1; isolated = False; wi = 0; gi = 0
    for k, dt in enumerate(np.diff(time), 1):
        transition = np.eye(6); transition[:3, 3:] = np.eye(3)*dt
        state = transition@state + np.r_[p[k]-p[k-1]-v[k-1]*dt, v[k]-v[k-1]]
        q = acceleration_noise_intensity
        process = np.block([[q*dt**3/3, q*dt**2/2], [q*dt**2/2, q*dt]])
        covariance = transition@covariance@transition.T + process
        while wi < len(windows) and windows[wi]['decision_s'] <= time[k]:
            decision = windows[wi]['state']
            isolated |= decision == 1
            wi += 1
        # These are experiment modes, not authenticated NORMAL/DEGRADED labels.
        mode = (1 if isolated else decision) if use_spatial else 0
        if not admitted:
            mode = -1
        if gi < len(gnss_indices) and gnss_indices[gi] == k:
            if mode == 0:
                observation = np.r_[gnss_position_m[gi], gnss_velocity_mps[gi]]
                gain = np.linalg.solve(covariance+gnss_covariance, covariance.T).T
                state += gain@(observation-state)
                residual = np.eye(6)-gain
                covariance = residual@covariance@residual.T + gain@gnss_covariance@gain.T
                accepted[k] = True
            gi += 1
        output.append(state.copy()); covariances.append(covariance.copy()); modes.append(mode)
    return dict(time_s=time, position_m=np.array(output)[:, :3],
                velocity_mps=np.array(output)[:, 3:], covariance=np.array(covariances),
                mode=np.array(modes), gnss_accepted=accepted)
