"""Fractional sequence delay shared by the simplified GNSS sources."""

import numpy as np


DEFAULT_FRACTIONAL_DELAY_TAPS = 33


def fractional_delay(
    signal: np.ndarray,
    delay_samples: float,
    num_taps: int = DEFAULT_FRACTIONAL_DELAY_TAPS,
) -> np.ndarray:
    """Delay one finite record while treating samples outside it as zero."""

    samples = np.asarray(signal, dtype=np.complex128)
    if delay_samples == 0.0:
        return samples.copy()
    if not np.isfinite(delay_samples) or delay_samples < 0.0:
        raise ValueError("delay_samples must be finite and nonnegative")
    if num_taps < 3 or num_taps % 2 == 0:
        raise ValueError("num_taps must be an odd integer of at least three")

    nearest_integer = round(delay_samples)
    if abs(delay_samples - nearest_integer) <= 1e-12:
        delay_samples = float(nearest_integer)

    integer_delay = int(np.floor(delay_samples))
    fraction = delay_samples - integer_delay
    output = np.zeros(samples.shape, dtype=np.complex128)
    if integer_delay >= samples.size:
        return output
    if fraction == 0.0:
        output[integer_delay:] = samples[: samples.size - integer_delay]
        return output

    half = num_taps // 2
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.sinc(offsets - fraction) * np.hamming(num_taps)
    kernel /= np.sum(kernel)
    full = np.convolve(samples, kernel, mode="full")
    interpolated = full[half : half + samples.size]
    output[integer_delay:] = interpolated[: samples.size - integer_delay]
    return output


def delay_phase(delays_s: np.ndarray, source_rf_frequency_hz: float) -> np.ndarray:
    """Return the narrowband RF phase caused by each true-time delay."""

    return np.exp(
        -1j
        * 2.0
        * np.pi
        * source_rf_frequency_hz
        * np.asarray(delays_s, dtype=np.float64)
    )


def apply_channel_delays(
    component: np.ndarray,
    delays_s: np.ndarray,
    fs_hz: float,
    carrier_hz: float,
    if_hz: float,
) -> np.ndarray:
    """Delay every IF channel and restore phase removed by downconversion."""

    values = np.asarray(component, dtype=np.complex128)
    delays = np.asarray(delays_s, dtype=np.float64)
    if values.ndim != 2 or delays.shape != (values.shape[0],):
        raise ValueError("component and delays_s must describe the same channels")
    if np.all(delays == 0.0):
        return values.copy()

    result = np.empty(values.shape, dtype=np.complex128)
    residual_frequency_hz = carrier_hz - if_hz
    for channel, delay_s in enumerate(delays):
        result[channel] = fractional_delay(
            values[channel], delay_s * fs_hz
        ) * np.exp(-1j * 2.0 * np.pi * residual_frequency_hz * delay_s)
    return result
