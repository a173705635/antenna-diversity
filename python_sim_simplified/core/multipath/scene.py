"""Multipath RF scene construction."""

from dataclasses import dataclass

import numpy as np

from ..antenna import AntennaModel, antenna_response
from ..channel_delay import apply_channel_delays, delay_phase
from ..config import MultipathPath, SimulationConfig
from ..models import AntennaDiagnostics, MultipathTruth


@dataclass
class _MultipathScene:
    truth: MultipathTruth
    diagnostics: AntennaDiagnostics


def _multipath_scene(
    path: MultipathPath,
    cfg: SimulationConfig,
    antenna_model: AntennaModel,
    desired_waveform: np.ndarray,
    time_s: np.ndarray,
    true_delays_s: np.ndarray,
) -> _MultipathScene:
    excess_delay_s = path.excess_delay_chips / cfg.signal.code_rate_hz
    total_delays_s = true_delays_s + excess_delay_s
    relative_voltage = 10.0 ** (path.relative_power_db / 20.0)
    path_waveform = (
        desired_waveform
        * relative_voltage
        * np.exp(1j * path.reflection_phase_rad)
        * np.exp(1j * 2.0 * np.pi * path.doppler_offset_hz * time_s)
    )
    response_base, diagnostics = antenna_response(
        antenna_model,
        path.direction,
        path.polarization,
    )
    component = apply_channel_delays(
        response_base[:, None] * path_waveform[None, :],
        total_delays_s,
        cfg.signal.fs_rf_hz,
        cfg.signal.carrier_hz,
        cfg.signal.if_hz,
    )
    path_rf_frequency_hz = (
        cfg.signal.carrier_hz
        + cfg.signal.doppler_hz
        + path.doppler_offset_hz
    )
    path_delay_phase = delay_phase(total_delays_s, path_rf_frequency_hz)
    truth: MultipathTruth = {
        "component": component,
        "steering_true": response_base * path_delay_phase,
        "delay_phase_true": path_delay_phase,
        "excess_delay_chips": path.excess_delay_chips,
        "excess_delay_s": excess_delay_s,
        "relative_power_db": path.relative_power_db,
        "reflection_phase_rad": path.reflection_phase_rad,
        "doppler_offset_hz": path.doppler_offset_hz,
        "direction": path.direction,
    }
    return _MultipathScene(truth, diagnostics)
