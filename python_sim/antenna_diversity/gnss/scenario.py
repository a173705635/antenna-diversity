"""Dual-antenna GPS L1 C/A and continuous-wave interference scene."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from antenna_diversity.array.steering import steering_vector
from antenna_diversity.config import SimulationConfig

from .ca_code import ca_code


def generate_scenario(
    cfg: SimulationConfig,
) -> tuple[NDArray[np.complex128], dict[str, Any]]:
    """Generate complex voltages at every antenna before RF noise and gain."""

    cfg.validate()
    n_antennas = cfg.array.num_elements
    n_samples = round(cfg.signal.fs_rf_hz * cfg.signal.duration_ms * 1e-3)
    if n_samples <= 0:
        raise ValueError("signal duration must produce at least one sample")

    code = ca_code(cfg.signal.prn)
    n = np.arange(n_samples, dtype=np.float64)
    time_s = n / cfg.signal.fs_rf_hz
    received_code_rate_hz = cfg.signal.code_rate_hz * (
        1.0 + cfg.signal.doppler_hz / cfg.signal.carrier_hz
    )
    code_phase = np.mod(
        n * received_code_rate_hz / cfg.signal.fs_rf_hz
        - cfg.signal.code_delay_chips,
        1023.0,
    )
    sampled_code = code[np.floor(code_phase).astype(np.int64)]

    signal_power_w = 10.0 ** ((cfg.signal.power_dbm - 30.0) / 10.0)
    signal_amplitude_v = np.sqrt(signal_power_w * cfg.signal.resistance_ohm)
    signal_carrier = np.exp(
        1j
        * (
            2.0
            * np.pi
            * (cfg.signal.if_hz + cfg.signal.doppler_hz)
            * time_s
            + cfg.signal.initial_phase_rad
        )
    )
    desired_waveform = signal_amplitude_v * sampled_code * signal_carrier

    desired_steering = steering_vector(
        cfg.array.positions_m,
        cfg.array.desired_angle_deg,
        cfg.signal.wavelength_m,
    )
    amplitude_errors_db = np.asarray(
        cfg.array.channel_amplitude_error_db, dtype=np.float64
    ).reshape(-1)
    phase_errors_deg = np.asarray(
        cfg.array.channel_phase_error_deg, dtype=np.float64
    ).reshape(-1)
    channel_response = 10.0 ** (amplitude_errors_db / 20.0) * np.exp(
        1j * np.deg2rad(phase_errors_deg)
    )
    if channel_response.size != n_antennas:
        raise ValueError("channel error arrays must contain one value per antenna")
    desired_steering_effective = channel_response * desired_steering
    desired_component = (
        desired_steering_effective[:, np.newaxis]
        * desired_waveform[np.newaxis, :]
    )

    if cfg.jammer.enable:
        jammer_power_w = 10.0 ** ((cfg.jammer.power_dbm - 30.0) / 10.0)
        jammer_amplitude_v = np.sqrt(
            jammer_power_w * cfg.signal.resistance_ohm
        )
        jammer_waveform = jammer_amplitude_v * np.exp(
            1j
            * (
                2.0
                * np.pi
                * (cfg.signal.if_hz + cfg.jammer.offset_hz)
                * time_s
                + cfg.jammer.initial_phase_rad
            )
        )
        jammer_steering = steering_vector(
            cfg.array.positions_m,
            cfg.jammer.angle_deg,
            cfg.signal.wavelength_m,
        )
        jammer_steering_effective = channel_response * jammer_steering
        jammer_component = (
            jammer_steering_effective[:, np.newaxis]
            * jammer_waveform[np.newaxis, :]
        )
    else:
        jammer_steering = np.zeros(n_antennas, dtype=np.complex128)
        jammer_steering_effective = np.zeros(n_antennas, dtype=np.complex128)
        jammer_component = np.zeros(
            (n_antennas, n_samples), dtype=np.complex128
        )

    x_antenna_v = desired_component + jammer_component
    truth: dict[str, Any] = {
        "time_s": time_s,
        "code": code,
        "desired_waveform": desired_waveform,
        "desired_component": desired_component,
        "jammer_component": jammer_component,
        "desired_steering_nominal": desired_steering,
        "desired_steering_effective": desired_steering_effective,
        "jammer_steering_nominal": jammer_steering,
        "jammer_steering_effective": jammer_steering_effective,
        "channel_response": channel_response,
        "doppler_hz": cfg.signal.doppler_hz,
        "code_rate_hz": received_code_rate_hz,
        "code_delay_chips": cfg.signal.code_delay_chips,
        "code_delay_rf_samples": (
            cfg.signal.code_delay_chips
            * cfg.signal.fs_rf_hz
            / cfg.signal.code_rate_hz
        ),
        "fs_rf_hz": cfg.signal.fs_rf_hz,
        "if_hz": cfg.signal.if_hz,
    }
    return x_antenna_v, truth
