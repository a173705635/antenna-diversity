"""Band-limited interference waveform generation.

Experiments choose finite positive rates/power, an odd FIR length and an occupied band inside Nyquist.
"""

import numpy as np
from scipy.signal import firwin, lfilter

from config import WidebandInterferer



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
