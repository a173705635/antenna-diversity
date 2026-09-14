"""Dual-antenna common-source evidence, not signal authentication.

Circular dispersion is invariant to common line bias and half-cycle ambiguity.
It deliberately does not identify individual spoofed PRNs or recover position.
"""
import numpy as np


def spatial_dispersion(difference_m, wavelength_m):
    d = np.asarray(difference_m, float)
    if (d.ndim < 1 or d.shape[-1] < 2 or not np.isfinite(d).all()
            or not np.isfinite(wavelength_m) or wavelength_m <= 0):
        raise ValueError('finite multi-PRN differences and positive wavelength required')
    return np.clip(1 - np.abs(np.mean(np.exp(4j*np.pi*d/wavelength_m), axis=-1)), 0, 1)


def window_evidence(times_s, a_m, b_m, valid, threshold, wavelength_m,
                    window_epochs=10, step_s=.01):
    """Nonoverlapping complete fixed-PRN blocks; -1 unknown, 0 no evidence, 1 alert.

    No truth, calibration interval, fitted per-PRN constants or attitude input.
    A block containing missing data or irregular timing is unknown in full.
    """
    t, a, b, v = (np.asarray(x) for x in (times_s, a_m, b_m, valid))
    if (t.ndim != 1 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0)
            or a.ndim != 2 or a.shape != b.shape or v.shape != a.shape
            or a.shape[0] != len(t) or a.shape[1] != 8 or v.dtype.kind != 'b'
            or not np.isfinite(threshold) or not 0 < threshold < 1
            or not isinstance(window_epochs, int) or window_epochs < 1
            or not np.isfinite(step_s) or step_s <= 0
            or not np.isfinite(wavelength_m) or wavelength_m <= 0):
        raise ValueError('aligned eight-PRN matrices, boolean validity and valid parameters required')
    result = []
    for start in range(0, len(t), window_epochs):
        end = min(start+window_epochs, len(t))
        good = (end-start == window_epochs and v[start:end].all()
                and np.isfinite(a[start:end]).all() and np.isfinite(b[start:end]).all()
                and np.allclose(np.diff(t[start:end]), step_s, rtol=0, atol=1e-9))
        score = float(spatial_dispersion(b[start:end]-a[start:end], wavelength_m).mean()) if good else None
        result.append({'start_s': float(t[start]), 'decision_s': float(t[end-1]),
                       'score': score, 'state': int(score < threshold) if good else -1,
                       'start_index': start, 'end_index': end})
    return result
