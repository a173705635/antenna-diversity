"""Single-antenna, eight-PRN RF-to-position experiment with external timing aid.

ponytail: short synthetic receive-frame satellite trajectories; add broadcast
ephemeris/TOW decoding when autonomous navigation becomes the experiment.
"""


from dataclasses import replace

import numpy as np

from .acquisition import acquire
from .antenna import build_antenna_input, process_antenna
from .config import SPEED_OF_LIGHT_MPS
from .dfe import process_dfe
from .feedline import process_feedline
from .gnss import generate_signal_sources
from .positioning import PositioningFailure
from .positioning import code_pseudorange, solve_position
from .presets import default_config
from .rf import process_rf
from .tracking import track


def receiver_reference():
    """Simulation/evaluation only: WGS84 31 N, 121 E, 30 m; ECEF-to-ENU matrix."""
    lat, lon = np.radians([31., 121.])
    e2 = 6.69437999014e-3
    radius = 6378137./np.sqrt(1-e2*np.sin(lat)**2)
    position = np.array([(radius+30)*np.cos(lat)*np.cos(lon),
                         (radius+30)*np.cos(lat)*np.sin(lon),
                         (radius*(1-e2)+30)*np.sin(lat)])
    rotation = np.array([[-np.sin(lon), np.cos(lon), 0.],
                         [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
                         [np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)]])
    return position, rotation


def satellite_setup():
    """Construct synthetic states and independent coarse timing-aid fields.

    Positions are already expressed in the reception ECEF frame. No additional
    Sagnac rotation should be applied to these particular synthetic coordinates.
    """
    receiver, rotation = receiver_reference()
    wavelength = SPEED_OF_LIGHT_MPS/1575.42e6
    result = []
    for prn, (azimuth, elevation, doppler) in enumerate(zip(
            [10, 55, 100, 145, 190, 235, 280, 325],
            [25, 60, 35, 75, 20, 50, 40, 65],
            [-1500, -1000, -500, 0, 500, 1000, 1500, 2000]), start=1):
        az, el = np.radians([azimuth, elevation])
        enu = np.array([np.cos(el)*np.sin(az), np.cos(el)*np.cos(az), np.sin(el)])
        direction = rotation.T @ enu
        projection = receiver @ direction
        distance = -projection+np.sqrt(projection**2+26560000.**2-receiver @ receiver)
        pseudorange = distance+75000.
        result.append({'prn': prn, 'position_m': receiver+distance*direction,
                       'velocity_mps': -doppler*wavelength*direction,
                       'doppler_hz': doppler, 'range_at_zero_m': pseudorange,
                       'coarse_range_m': float(np.round(pseudorange/10000.)*10000.)})
    return result


def build_scene(satellites, duration_ms, *, code_bandwidth_hz=None,
                fs_rf_hz=None, decimation=None, signal_kind='L1CA'):
    """在阵元等效输入参考面汇总各 PRN；公共天线与馈线稍后各处理一次。"""
    cfg = default_config()
    if signal_kind not in ('L1CA', 'L5Q'):
        raise ValueError('unsupported signal kind')
    cfg.signal.signal_kind = cfg.tracking.signal_kind = signal_kind
    if signal_kind == 'L5Q':
        # NH码尚未同步，L5捕获仍用1 ms相干块；非相干次数继承公共配置。
        cfg.acquisition.coherent_ms = 1
        cfg.signal.carrier_hz = 1176.45e6
        cfg.signal.code_rate_hz = 10.23e6
        cfg.dfe.decimation = 1
        cfg.dfe.cutoff_hz = 10.23e6
        cfg.rf.noise_bandwidth_hz = 20.46e6
    if fs_rf_hz is not None:
        cfg.signal.fs_rf_hz = fs_rf_hz
    if decimation is not None:
        cfg.dfe.decimation = decimation
    cfg.signal.duration_ms = duration_ms
    cfg.signal.code_bandwidth_hz = code_bandwidth_hz
    cfg.tracking.max_epochs = duration_ms
    cfg.phone.active_antennas = ('TL',)
    cfg.phone.phase_centers_m['TL'] = (0., 0., 0.)
    cfg.phone.assumed_phase_centers_m['TL'] = (0., 0., 0.)
    cfg.interferers = []
    scene = np.zeros((1, round(duration_ms*.001*cfg.signal.fs_rf_hz)), dtype=complex)
    for satellite in satellites:
        cfg.signal.prn = satellite['prn']
        cfg.signal.doppler_hz = satellite['doppler_hz']*cfg.signal.carrier_hz/1575.42e6
        # Full delay retains the transmitted NH epoch; L1 periodic lookup is unchanged.
        delay = satellite['range_at_zero_m']/SPEED_OF_LIGHT_MPS*cfg.signal.code_rate_hz
        cfg.signal.code_delay_chips = delay if signal_kind == 'L5Q' else delay % 1023.
        cfg.signal.initial_phase_rad = float((-2*np.pi*satellite['range_at_zero_m']/cfg.signal.wavelength_m) % (2*np.pi))
        signal_sources = generate_signal_sources(cfg)
        component, truth = build_antenna_input(signal_sources, cfg)
        scene += component
        del signal_sources, component, truth
    return scene, cfg


def run_record(seed, satellites, scene, cfg):
    """所有 PRN 干净源汇总后仅加一次三段噪声，再形成共享 DFE 流。"""
    record_cfg = replace(cfg, random_seed=seed)
    antenna_output_v, antenna_info = process_antenna(scene, record_cfg)
    rf_input_v, _ = process_feedline(
        antenna_output_v, cfg.feedline, antenna_info,
        cfg.reference_impedance_ohm, cfg.signal.fs_rf_hz, cfg.signal.if_hz,
        cfg.phone.active_antennas, seed,
    )
    del antenna_output_v
    adc, rf_info = process_rf(
        rf_input_v, cfg.rf, cfg.reference_impedance_ohm, seed,
        fs_hz=cfg.signal.fs_rf_hz, if_hz=cfg.signal.if_hz,
        channel_ids=cfg.phone.active_antennas,
    )
    del rf_input_v
    baseband, dfe_info = process_dfe(adc, cfg.dfe, cfg.signal.fs_rf_hz, cfg.signal.if_hz)
    del adc
    fs = dfe_info['fs_out_hz']
    times = np.arange(20, cfg.signal.duration_ms//10)*.01
    ranges = np.full((times.size, len(satellites)), np.nan)
    records = []
    for column, satellite in enumerate(satellites):
        prn = satellite['prn']
        acq = acquire(baseband[0], fs, prn, cfg.signal.code_rate_hz, cfg.acquisition,
                      signal_kind=cfg.signal.signal_kind)
        entry = {'prn': prn, 'acquired_flag': bool(acq['success_flag']),
                 'acquisition_metric': float(acq['metric']),
                 'acquisition_doppler_hz': float(acq['doppler_hz']),
                 'acquisition_code_phase_samples': int(acq['code_phase_samples'])}
        if acq['success_flag']:
            history = track(baseband[0], fs, prn, acq, cfg.tracking,
                            cfg.signal.carrier_hz, cfg.signal.code_rate_hz)
            entry['tracking_status'] = history['status']
            if history['success_flag']:
                ranges[:, column] = code_pseudorange(history, times, fs,
                    cfg.tracking.coherent_ms*.001, satellite['coarse_range_m'],
                    code_rate_hz=cfg.signal.code_rate_hz,
                    code_length=round(cfg.signal.code_rate_hz*.001))
                if 'nh_sync' in history:
                    entry['nh_sync'] = history['nh_sync']
                # A basic current-epoch quality screen, not a cycle-slip certification.
                index = np.searchsorted(history['epoch_start_samples']/fs, times, side='right')-1
                safe = np.clip(index, 0, len(history['prompt'])-1)
                quality_flag = ((index >= 0) & (history['phase_lock_metric'][safe] >= .8)
                           & (abs(history['pll_error_rad'][safe]) < np.pi/2))
                ranges[~quality_flag, column] = np.nan
                for key in ('epoch_start_samples', 'code_nco_phase_chips', 'code_nco_rate_hz',
                            'carrier_nco_phase_rad', 'carrier_nco_frequency_hz',
                            'pll_error_rad', 'phase_lock_metric', 'cn0_db_hz'):
                    entry[key] = history[key].tolist()
        records.append(entry)
        print(f"seed={seed} PRN={prn} acquired={entry['acquired_flag']} usable={np.isfinite(ranges[:, column]).sum()}/{times.size}", flush=True)
    positions = np.full((times.size, 3), np.nan)
    clocks = np.full(times.size, np.nan)
    conditions = np.full(times.size, np.nan)
    residuals = np.full_like(ranges, np.nan)
    failures = []
    for epoch, time in enumerate(times):
        states = np.array([s['position_m']+s['velocity_mps']*time for s in satellites])
        valid_flag = np.isfinite(ranges[epoch])
        try:
            solution = solve_position(states[valid_flag], ranges[epoch, valid_flag])
            positions[epoch] = solution['position_m']
            clocks[epoch] = solution['clock_bias_m']
            conditions[epoch] = solution['geometry_condition']
            residuals[epoch, valid_flag] = solution['residual_m']
        except PositioningFailure as exc:
            failures.append({'time_s': float(time), 'reason': str(exc)})
    # Truth is introduced only here, after every receiver/position solution.
    reference, rotation = receiver_reference()
    enu = (positions-reference) @ rotation.T
    valid_flag = np.isfinite(positions).all(axis=1)
    summary = {'seed': seed, 'epochs': len(times), 'solved_epochs': int(valid_flag.sum()),
               'availability': float(np.mean(valid_flag)),
               'acquired_satellites': sum(r['acquired_flag'] for r in records)}
    if valid_flag.any():
        error = enu[valid_flag]
        norm = np.linalg.norm(error, axis=1)
        summary.update({'enu_mean_m': np.mean(error, axis=0).tolist(),
                        'enu_std_m': np.std(error, axis=0).tolist(),
                        'horizontal_rms_m': float(np.sqrt(np.mean(np.sum(error[:, :2]**2, axis=1)))),
                        'position_rms_m': float(np.sqrt(np.mean(norm**2))),
                        'position_p95_m': float(np.quantile(norm, .95)),
                        'position_max_m': float(norm.max()),
                        'clock_error_mean_m': float(np.mean(clocks[valid_flag]-75000.)),
                        'used_satellites_min': int(np.min(np.isfinite(ranges[valid_flag]).sum(axis=1)))})
    return {'summary': summary, 'time_s': times, 'pseudoranges_m': ranges,
            'position_ecef_m': positions, 'clock_bias_m': clocks,
            'enu_error_m': enu, 'geometry_condition': conditions, 'residual_m': residuals,
            'failures': failures, 'satellite_tracking': records,
            'rf_saturation_fraction': rf_info['saturation_fraction'].tolist(),
            'fs_dfe_hz': fs}
