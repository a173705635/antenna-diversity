"""Toolbox-independent windowed-sinc decimation-filter design."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def design_decimator(
    fs_in_hz: float,
    decimation: int,
    cutoff_hz: float,
    num_taps: int,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Design the same odd-length Hamming-windowed sinc used by MATLAB."""

    if not np.isscalar(fs_in_hz) or not np.isfinite(fs_in_hz) or fs_in_hz <= 0:
        raise ValueError("fs_in_hz must be a positive finite scalar")
    if not isinstance(decimation, (int, np.integer)) or decimation < 1:
        raise ValueError("decimation must be a positive integer")
    fs_out_hz = fs_in_hz / decimation
    if not np.isscalar(cutoff_hz) or not 0 < cutoff_hz < fs_out_hz / 2:
        raise ValueError("cutoff_hz must lie below the output Nyquist frequency")
    if (
        not isinstance(num_taps, (int, np.integer))
        or num_taps < 3
        or num_taps % 2 == 0
    ):
        raise ValueError("num_taps must be an odd integer of at least three")

    group_delay = (num_taps - 1) // 2
    m = np.arange(-group_delay, group_delay + 1, dtype=np.float64)
    u = 2.0 * cutoff_hz / fs_in_hz * m
    h = (2.0 * cutoff_hz / fs_in_hz) * np.sinc(u)
    window = 0.54 - 0.46 * np.cos(
        2.0 * np.pi * np.arange(num_taps) / (num_taps - 1)
    )
    h *= window
    h /= np.sum(h)

    info: dict[str, Any] = {
        "fs_in_hz": fs_in_hz,
        "fs_out_hz": fs_out_hz,
        "decimation": decimation,
        "cutoff_hz": cutoff_hz,
        "num_taps": num_taps,
        "group_delay_input_samples": group_delay,
    }
    return h, info
