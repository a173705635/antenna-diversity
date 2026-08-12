"""Uniform-linear-array steering-vector convention."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def steering_vector(
    positions_m: ArrayLike,
    angle_deg: float,
    wavelength_m: float,
) -> NDArray[np.complex128]:
    """Return the broadside-referenced response at each array position."""

    positions = np.asarray(positions_m, dtype=np.float64).reshape(-1)
    if positions.size == 0 or not np.all(np.isfinite(positions)):
        raise ValueError("positions_m must be a nonempty finite vector")
    if not np.isscalar(angle_deg) or not np.isfinite(angle_deg):
        raise ValueError("angle_deg must be a finite scalar")
    if not np.isscalar(wavelength_m) or not np.isfinite(wavelength_m) or wavelength_m <= 0:
        raise ValueError("wavelength_m must be a positive finite scalar")

    return np.exp(
        -1j
        * 2
        * np.pi
        * positions
        * np.sin(np.deg2rad(angle_deg))
        / wavelength_m
    )
