"""Reproducible synthetic carrier + imposed PVT attack + IMU closed loop.

PVT displacement is an explicit assumed attack effect, not an RF/PVT solution.
Fixed sky/source in navigation frame; array rotates with analytic vehicle motion.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from audit_spatial_diversity import W, NAMES, model_observations
from inertial import measurements, propagate, trajectory
from navigation_fusion import fuse
from presets import mate70_phase_centers
from spatial_spoofing import spatial_scores, calibrate_threshold, detect_windows, confirm_windows


def carrier_stream(time, kind, case, seed):
    rng = np.random.default_rng(seed)
    az = rng.uniform(0, 2*np.pi, 8)
    up = rng.uniform(np.sin(np.deg2rad(15)), 1, 8)
    sky = np.column_stack((np.sqrt(1-up**2)*np.cos(az), np.sqrt(1-up**2)*np.sin(az), up))
    source = rng.normal(size=3); source /= np.linalg.norm(source)
    positions = np.array([mate70_phase_centers()[name] for name in NAMES])
    positions -= positions[0]
    rotation = trajectory(time, kind)['rotation'].as_matrix()
    antenna_nav = np.einsum('tij,aj->tai', rotation, positions)
    genuine = -np.einsum('tai,si->tas', antenna_nav, sky)
    fake = -np.einsum('tai,i->ta', antenna_nav, source)
    active = time >= 20
    if case == 'clean':
        active[:] = False
    if case == 'withdraw':
        active &= time < 30
    phase = genuine.copy()
    count = 3 if case == 'partial3' else 8
    phase[active, :, :count] = fake[active, :, None]
    if case == 'prn_bias':
        phase[active] += rng.normal(0, .02, (4, 8))
    noise = rng.normal(size=phase.shape)+1j*rng.normal(size=phase.shape)
    for k in range(1, len(time)):
        noise[k] = .7*noise[k-1]+np.sqrt(1-.7**2)*noise[k]
    z = np.exp(-2j*np.pi*phase/W)+noise*(.003/np.sqrt(2))*2*np.pi/W
    return -np.angle(z)*W/(2*np.pi)


def navigation_inputs(kind, case, seed):
    """Synthetic generator; truth is returned separately from consumer inputs."""
    time = np.arange(4001)*.02; gi = np.arange(5, len(time), 5)
    truth = trajectory(time, kind); rng = np.random.default_rng(seed+50000)
    gravity = np.array([0., 0., -9.80665]); lever = np.array([.07, .03, .01])
    imu = measurements(time, kind, lever_arm_m=lever, gravity_mps2=gravity,
                       time_offset_s=0., bias=np.array([.002, -.001, .001, 0., np.deg2rad(.005), 0.]),
                       axis_matrix=np.eye(6), noise_covariance=np.diag([.008]*3+[8e-5]*3)**2,
                       correlation=.7, seed=seed+60000)
    initial = trajectory(np.array([-.01]), kind)
    inertial = propagate(time, imu['force_mps2'], imu['rate_rad_s'],
                         position_m=truth['position_m'][0]+rng.normal(0, 3, 3),
                         velocity_mps=truth['velocity_mps'][0]+rng.normal(0, .2, 3),
                         body_to_nav=(truth['rotation'][0]*Rotation.from_euler('y', .2, degrees=True)).as_matrix(),
                         accel_bias_mps2=np.zeros(3), gyro_bias_rad_s=np.zeros(3),
                         gravity_mps2=gravity, lever_arm_m=lever,
                         previous_rate_rad_s=initial['rate_rad_s'][0]+[0, np.deg2rad(.005), 0])
    gt = time[gi]; drag = np.zeros(len(gt)); speed = np.zeros(len(gt))
    active = gt >= 20
    if case != 'clean':
        slope = .5 if case == 'slow' else 5.
        drag[active] = (gt[active]-20)*slope
        speed[active] = slope
    if case == 'withdraw':
        drag[gt >= 30] = 0; speed[gt >= 30] = 0
    gp = truth['position_m'][gi]+rng.normal(0, 3, (len(gi), 3))
    gv = truth['velocity_mps'][gi]+rng.normal(0, .2, (len(gi), 3))
    gp[:, 0] += drag; gv[:, 0] += speed
    return inertial, gi, gp, gv, truth


def experiment(kind, case, seed, threshold, *, confirmation_count=1):
    inertial, gi, gp, gv, truth = navigation_inputs(kind, case, seed)
    time = inertial['time_s']
    ct = np.arange(8000)*.01
    carrier = carrier_stream(ct, kind, case, seed)
    windows = detect_windows(ct, carrier, np.ones_like(carrier, bool), threshold, W)
    windows = confirm_windows(windows, confirmation_count)
    rows = []; curves = {}
    alarms = np.array([w['decision_s'] for w in windows if w['state'] == 1])
    before = alarms[alarms < 20]; after = alarms[alarms >= 20]
    for gated in (False, True):
        result = fuse(inertial, gi, gp, gv, windows,
                      initial_covariance=np.diag([3]*3+[.2]*3)**2,
                      gnss_covariance=np.diag([3]*3+[.2]*3)**2,
                      acceleration_noise_intensity=np.eye(3)*.02**2,
                      admitted=True, use_spatial=gated)
        error = np.linalg.norm(result['position_m'][:, :2]-truth['position_m'][:, :2], axis=1)
        row = dict(motion=kind, case=case, seed=seed, spatial_gate=gated,
                   first_pre_attack_alert_s=float(before[0]) if len(before) else None,
                   first_post_attack_evidence_delay_s=float(after[0]-20) if len(after) and case != 'clean' else None,
                   pre_attack_isolation=bool(gated and len(before)),
                   gnss_update_fraction=float(result['gnss_accepted'][gi].mean()),
                   error_p95_m=float(np.quantile(error, .95)),
                   max_error_while_updates_allowed_m=float(error[result['mode'] == 0].max()),
                   allowed_epochs_over30m=int(np.sum((result['mode'] == 0)&(error >= 30))),
                   error_at_attack_m=float(error[1000]))
        for seconds in (10, 30, 60):
            segment = error[(time >= 20)&(time <= 20+seconds)]
            row[f'attack_first{seconds}s_max_m'] = float(segment.max())
            row[f'attack_plus{seconds}s_m'] = float(segment[-1])
        rows.append(row); curves['gated' if gated else 'ungated'] = error
    return rows, curves


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    calibration = model_observations(710001, 2000, 'los')
    threshold = calibrate_threshold(spatial_scores(calibration, W)['cluster3'])
    rows = []; example = {}
    for kind in ('static', 'turn'):
        for seed in range(720001, 720005):
            for case in ('clean', 'all_spoof', 'partial3', 'slow', 'withdraw', 'prn_bias'):
                result, curves = experiment(kind, case, seed, threshold)
                rows.extend(result)
                if kind == 'turn' and seed == 720001:
                    example[case] = curves
    summary = dict(source_kind='synthetic_observables_and_imposed_pvt', threshold=threshold,
                   calibration_events=2000, independent_trajectory_seeds=4, rows=rows,
                   admitted_initialization=True, attack_start_s=20, duration_s=80,
                   covariance_is_integrity_bound=False,
                   sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                            for name in ('audit_navigation_fusion.py', 'navigation_fusion.py', 'inertial.py',
                                         'spatial_spoofing.py', 'audit_spatial_diversity.py', 'presets.py')})
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for ax, (case, curves) in zip(axes.flat, example.items()):
        for label, values in curves.items():
            ax.plot(np.arange(len(values))*.02, values, label=label)
        ax.axvline(20, color='gray', ls=':'); ax.axhline(20, color='gray', ls='--')
        ax.set(title=case, xlabel='Time (s)', ylabel='Horizontal error (m)'); ax.legend()
    fig.tight_layout(); fig.savefig(args.output/'navigation.png', dpi=160); plt.close(fig)
    print(json.dumps(dict(output=str(args.output), runs=len(rows), threshold=threshold)))


if __name__ == '__main__':
    main()
