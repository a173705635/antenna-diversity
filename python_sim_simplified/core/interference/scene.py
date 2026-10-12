"""Interferer RF scene construction."""

from dataclasses import dataclass

import numpy as np

from ..antenna import AntennaModel, antenna_response
from ..channel_delay import apply_channel_delays, delay_phase
from ..config import CWInterferer, Interferer, SimulationConfig, WidebandInterferer
from ..models import AntennaDiagnostics
from .interference import generate_wideband_waveform


@dataclass
class _InterfererScene:
    response: np.ndarray
    component: np.ndarray
    delay_phase: np.ndarray
    diagnostics: AntennaDiagnostics


def _interferer_scene(
    interferer: Interferer,
    cfg: SimulationConfig,
    antenna_model: AntennaModel,
    time_s: np.ndarray,
    true_delays_s: np.ndarray,
) -> _InterfererScene:
    response_base, diagnostics = antenna_response(
        antenna_model,
        interferer.direction,
        interferer.polarization,
    )
    if isinstance(interferer, CWInterferer):
        center_offset_hz = interferer.offset_hz
        amplitude_v = np.sqrt(
            10.0 ** ((interferer.power_dbm - 30.0) / 10.0)
            * cfg.reference_impedance_ohm
        )
        waveform = amplitude_v * np.exp(
            1j
            * (
                2.0
                * np.pi
                * (cfg.signal.if_hz + center_offset_hz)
                * time_s
                + interferer.initial_phase_rad
            )
        )
    elif isinstance(interferer, WidebandInterferer):
        center_offset_hz = interferer.center_offset_hz
        waveform = generate_wideband_waveform(
            interferer,
            time_s.size,
            cfg.signal.fs_rf_hz,
            cfg.signal.if_hz,
            cfg.reference_impedance_ohm,
        )
    first = round(interferer.start_ms * 1e-3 * cfg.signal.fs_rf_hz)
    stop = (
        time_s.size
        if interferer.stop_ms is None
        else round(interferer.stop_ms * 1e-3 * cfg.signal.fs_rf_hz)
    )
    gate = np.zeros(time_s.size)
    gate[first:stop] = 1.0
    interferer_delay_phase = delay_phase(
        true_delays_s,
        cfg.signal.carrier_hz + center_offset_hz,
    )
    response = (
        response_base
        if np.all(true_delays_s == 0.0)
        else response_base * interferer_delay_phase
    )
    component = apply_channel_delays(
        response_base[:, None] * (waveform * gate)[None, :],
        true_delays_s,
        cfg.signal.fs_rf_hz,
        cfg.signal.carrier_hz,
        cfg.signal.if_hz,
    )
    return _InterfererScene(
        response,
        component,
        interferer_delay_phase,
        diagnostics,
    )
