"""Noncoherent FFT parallel-code-phase GPS acquisition."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from antenna_diversity.config import AcquisitionConfig
from antenna_diversity.gnss.ca_code import ca_code


def acquire(
    x: ArrayLike,
    fs_hz: float,
    prn: int,
    cfg: AcquisitionConfig,
) -> dict[str, Any]:
    """Search configured Doppler bins and every code phase."""

    samples = np.asarray(x, dtype=np.complex128)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("x must be a nonempty one-dimensional vector")
    if not np.isscalar(fs_hz) or not np.isfinite(fs_hz) or fs_hz <= 0:
        raise ValueError("fs_hz must be a positive finite scalar")

    samples_per_ms_exact = fs_hz * 1e-3
    if not np.isclose(
        samples_per_ms_exact, round(samples_per_ms_exact), atol=1e-9
    ):
        raise ValueError("sample rate must produce an integer number of samples per millisecond")
    samples_per_ms = round(samples_per_ms_exact)
    coherent_samples = samples_per_ms * cfg.coherent_ms
    required_samples = coherent_samples * cfg.noncoherent_ms
    if samples.size < required_samples:
        raise ValueError(
            f"acquisition needs {required_samples} samples but received {samples.size}"
        )

    code = ca_code(prn)
    n = np.arange(coherent_samples, dtype=np.float64)
    code_phase = np.mod(n * cfg.code_rate_hz / fs_hz, 1023.0)
    local_code = code[np.floor(code_phase).astype(np.int64)]
    local_code_spectrum_conjugate = np.conj(np.fft.fft(local_code))
    doppler_bins = np.asarray(cfg.doppler_bins_hz, dtype=np.float64).reshape(-1)
    if doppler_bins.size == 0 or not np.all(np.isfinite(doppler_bins)):
        raise ValueError("Doppler bins must be nonempty and finite")
    search_power = np.zeros(
        (doppler_bins.size, coherent_samples), dtype=np.float64
    )

    for bin_index, doppler_hz in enumerate(doppler_bins):
        carrier_wipeoff = np.exp(-1j * 2.0 * np.pi * doppler_hz * n / fs_hz)
        accumulated_power = np.zeros(coherent_samples, dtype=np.float64)
        for block_index in range(cfg.noncoherent_ms):
            first = block_index * coherent_samples
            block = samples[first : first + coherent_samples]
            correlation = np.fft.ifft(
                np.fft.fft(block * carrier_wipeoff)
                * local_code_spectrum_conjugate
            )
            accumulated_power += np.abs(correlation) ** 2
        search_power[bin_index] = accumulated_power

    best_bin_index, best_code_index = np.unravel_index(
        int(np.argmax(search_power)), search_power.shape
    )
    peak = float(search_power[best_bin_index, best_code_index])
    best_row = search_power[best_bin_index].copy()
    indices = np.arange(coherent_samples)
    direct_distance = np.abs(indices - best_code_index)
    circular_distance = np.minimum(
        direct_distance, coherent_samples - direct_distance
    )
    exclusion_samples = max(
        1, round(cfg.exclusion_chips * fs_hz / cfg.code_rate_hz)
    )
    best_row[circular_distance <= exclusion_samples] = -np.inf
    second_peak_index = int(np.argmax(best_row))
    second_peak = float(best_row[second_peak_index])
    if not np.isfinite(second_peak) or second_peak <= 0:
        second_peak = float(np.finfo(float).eps)
        second_peak_code_phase_samples = float("nan")
        second_peak_distance_samples = float("nan")
    else:
        second_peak_code_phase_samples = second_peak_index
        direct_second_distance = abs(second_peak_index - best_code_index)
        second_peak_distance_samples = min(
            direct_second_distance,
            coherent_samples - direct_second_distance,
        )

    metric = peak / second_peak
    return {
        "success": bool(metric >= cfg.threshold),
        "prn": int(prn),
        "doppler_hz": float(doppler_bins[best_bin_index]),
        "code_phase_samples": int(best_code_index),
        "peak": peak,
        "second_peak": second_peak,
        "second_peak_code_phase_samples": second_peak_code_phase_samples,
        "second_peak_distance_samples": second_peak_distance_samples,
        "exclusion_samples": exclusion_samples,
        "metric": float(metric),
        "doppler_bins_hz": doppler_bins,
        "search_power": search_power,
        "samples_per_ms": samples_per_ms,
        "coherent_samples": coherent_samples,
    }
