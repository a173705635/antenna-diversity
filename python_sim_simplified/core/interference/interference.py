"""Band-limited interference waveform generation.

Experiments choose finite positive rates/power, an odd FIR length and an occupied band inside Nyquist.
"""

import numpy as np
from scipy.signal import firwin, lfilter

from ..config import CWInterferer, Interferer, SimulationConfig, WidebandInterferer
from ..models import InterfererSignal



def generate_wideband_waveform(
    interferer: WidebandInterferer,
    num_samples: int,
    fs_hz: float,
    if_hz: float,
    impedance_ohm: float,
) -> np.ndarray:
    """Generate one steady-state complex waveform at the configured total power."""

    warmup = interferer.filter_taps - 1
    rng = np.random.default_rng(interferer.random_seed)
    white = (
        rng.standard_normal(num_samples + warmup)
        + 1j * rng.standard_normal(num_samples + warmup)
    ) / np.sqrt(2.0)
    coefficients = firwin(
        interferer.filter_taps,
        interferer.bandwidth_hz / 2.0,
        fs=fs_hz,
    )
    filtered = lfilter(coefficients, [1.0], white)[warmup:]
    filtered /= np.sqrt(np.mean(np.abs(filtered) ** 2))

    power_w = 10.0 ** ((interferer.power_dbm - 30.0) / 10.0)
    amplitude_v = np.sqrt(power_w * impedance_ohm)
    n = np.arange(num_samples)
    carrier = np.exp(
        1j
        * 2.0
        * np.pi
        * (if_hz + interferer.center_offset_hz)
        * n
        / fs_hz
    )
    return amplitude_v * filtered * carrier


def generate_interferer_signal(
    interferer: Interferer,
    cfg: SimulationConfig,
    time_s: np.ndarray,
) -> InterfererSignal:
    """生成单个 CW/宽带干扰的一维参考电压，并按 RF 样本边界门控。"""
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
    return {
        "interferer": interferer,
        "waveform_v": waveform * gate,
        "center_offset_hz": center_offset_hz,
    }
