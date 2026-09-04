"""Band-limited interference waveform generation."""

import numpy as np
from scipy.signal import firwin, lfilter

from config import WidebandInterferer


def _validate_wideband(
    interferer: WidebandInterferer,
    num_samples: int,
    fs_hz: float,
    if_hz: float,
    impedance_ohm: float,
) -> None:
    if not isinstance(num_samples, (int, np.integer)) or num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")
    if not np.isfinite(fs_hz) or fs_hz <= 0.0:
        raise ValueError("fs_hz must be finite and positive")
    if not np.isfinite(if_hz):
        raise ValueError("if_hz must be finite")
    if not np.isfinite(impedance_ohm) or impedance_ohm <= 0.0:
        raise ValueError("impedance_ohm must be finite and positive")
    if not np.isfinite(interferer.power_dbm):
        raise ValueError("power_dbm must be finite")
    if (
        not np.isfinite(interferer.bandwidth_hz)
        or interferer.bandwidth_hz <= 0.0
        or interferer.bandwidth_hz >= fs_hz
    ):
        raise ValueError("bandwidth_hz must be between zero and fs_hz")
    if (
        isinstance(interferer.filter_taps, (bool, np.bool_))
        or not isinstance(interferer.filter_taps, (int, np.integer))
        or interferer.filter_taps < 3
        or interferer.filter_taps % 2 == 0
    ):
        raise ValueError("filter_taps must be an odd integer of at least three")
    if (
        isinstance(interferer.random_seed, (bool, np.bool_))
        or not isinstance(interferer.random_seed, (int, np.integer))
        or interferer.random_seed < 0
    ):
        raise ValueError("random_seed must be a nonnegative integer")
    center_hz = if_hz + interferer.center_offset_hz
    lower_hz = center_hz - interferer.bandwidth_hz / 2.0
    upper_hz = center_hz + interferer.bandwidth_hz / 2.0
    if not -fs_hz / 2.0 < lower_hz < upper_hz < fs_hz / 2.0:
        raise ValueError("wideband occupied band must remain inside Nyquist")


def generate_wideband_waveform(
    interferer: WidebandInterferer,
    num_samples: int,
    fs_hz: float,
    if_hz: float,
    impedance_ohm: float,
) -> np.ndarray:
    """Generate one steady-state complex waveform at the configured total power."""

    _validate_wideband(
        interferer,
        num_samples,
        fs_hz,
        if_hz,
        impedance_ohm,
    )
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
