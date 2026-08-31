"""Noncoherent FFT search over Doppler and C/A-code phase."""

import numpy as np

from gnss import ca_code


def acquire(x, fs_hz, prn, cfg):
    samples_per_ms = round(fs_hz * 1e-3)
    coherent_samples = samples_per_ms * cfg.coherent_ms
    n = np.arange(coherent_samples)
    code_phase = np.mod(n * cfg.code_rate_hz / fs_hz, 1023.0)
    local_code = ca_code(prn)[np.floor(code_phase).astype(int)]
    code_spectrum = np.conj(np.fft.fft(local_code))
    search_power = np.zeros((len(cfg.doppler_bins_hz), coherent_samples))

    for row, doppler_hz in enumerate(cfg.doppler_bins_hz):
        wipeoff = np.exp(-1j * 2.0 * np.pi * doppler_hz * n / fs_hz)
        for block_index in range(cfg.noncoherent_ms):
            first = block_index * coherent_samples
            block = x[first : first + coherent_samples]
            correlation = np.fft.ifft(
                np.fft.fft(block * wipeoff) * code_spectrum
            )
            search_power[row] += np.abs(correlation) ** 2

    best_row, best_code = np.unravel_index(
        np.argmax(search_power), search_power.shape
    )
    peak = search_power[best_row, best_code]
    row_without_peak = search_power[best_row].copy()
    indices = np.arange(coherent_samples)
    distance = np.abs(indices - best_code)
    circular_distance = np.minimum(distance, coherent_samples - distance)
    exclusion_samples = max(
        1, round(cfg.exclusion_chips * fs_hz / cfg.code_rate_hz)
    )
    row_without_peak[circular_distance <= exclusion_samples] = -np.inf
    second_code = int(np.argmax(row_without_peak))
    second_peak = row_without_peak[second_code]
    metric = peak / second_peak
    return {
        "success": bool(metric >= cfg.threshold),
        "prn": prn,
        "doppler_hz": float(cfg.doppler_bins_hz[best_row]),
        "code_phase_samples": int(best_code),
        "peak": float(peak),
        "second_peak": float(second_peak),
        "second_peak_code_phase_samples": second_code,
        "exclusion_samples": exclusion_samples,
        "metric": float(metric),
        "doppler_bins_hz": cfg.doppler_bins_hz,
        "search_power": search_power,
        "samples_per_ms": samples_per_ms,
        "coherent_samples": coherent_samples,
    }
