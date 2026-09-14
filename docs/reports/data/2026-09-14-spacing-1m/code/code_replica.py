"""Shared periodic continuous-code reference for source and local replica."""

import numpy as np


def bandlimited_code_replica(
    code: np.ndarray, code_rate_hz: float, bandwidth_hz: float,
    samples_per_chip: int = 64,
) -> np.ndarray:
    """Periodic rectangular chips passed through an ideal zero-phase lowpass.

    Fourier coefficients include the sinc and half-chip phase of a pulse on
    [0, 1). This is a research reference, not an exact replica of the DFE FIR.
    The dense lookup table supports continuous fractional-chip interpolation.
    """
    if (not np.isfinite(code_rate_hz) or code_rate_hz <= 0
            or not np.isfinite(bandwidth_hz) or bandwidth_hz <= 0
            or samples_per_chip < 2
            or bandwidth_hz >= samples_per_chip * code_rate_hz / 2):
        raise ValueError("positive finite rates and sufficient replica sampling required")
    length = code.size
    count = length * samples_per_chip
    harmonics = np.rint(np.fft.fftfreq(count) * count).astype(int)
    cycles_per_chip = harmonics / length
    spectrum = (
        np.fft.fft(code)[harmonics % length] / length
        * np.sinc(cycles_per_chip)
        * np.exp(-1j * np.pi * cycles_per_chip)
    )
    spectrum[np.abs(cycles_per_chip) > bandwidth_hz / code_rate_hz] = 0
    return np.fft.ifft(spectrum * count).real


def _sample_code(
    code: np.ndarray, phase_chips: np.ndarray, code_length: int = 1023,
) -> np.ndarray:
    phase = np.mod(phase_chips, float(code_length))
    if code.size == code_length:
        return code[np.floor(phase).astype(int)]
    position = phase * (code.size / code_length)
    left = np.floor(position).astype(int)
    fraction = position - left
    return (code[left % code.size] * (1 - fraction)
            + code[(left + 1) % code.size] * fraction)
