"""Three-dimensional array manifold and element-pattern interfaces."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from antenna_diversity.config import Direction


_SPEED_OF_LIGHT_MPS = 299_792_458.0


class ElementPattern(Protocol):
    def complex_gain(
        self,
        element_name: str,
        direction: Direction,
        frequency_hz: float,
    ) -> complex: ...


@dataclass(frozen=True)
class IsotropicElementPattern:
    def complex_gain(
        self,
        element_name: str,
        direction: Direction,
        frequency_hz: float,
    ) -> complex:
        return 1.0 + 0.0j


def direction_unit_vector(direction: Direction) -> NDArray[np.float64]:
    """Return the phone-frame unit vector for a validated direction."""

    direction.validate()
    azimuth = np.deg2rad(direction.azimuth_deg)
    elevation = np.deg2rad(direction.elevation_deg)
    return np.asarray(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ],
        dtype=np.float64,
    )


def array_manifold(
    positions_m: ArrayLike,
    direction: Direction,
    frequency_hz: float,
    element_names: Sequence[str] | None = None,
    pattern: ElementPattern | None = None,
) -> NDArray[np.complex128]:
    """Return element-pattern-weighted spatial responses for 3-D positions."""

    positions = np.asarray(positions_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1:] != (3,):
        raise ValueError("positions_m must have shape (M, 3)")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions_m must contain finite values")
    if (
        not np.isscalar(frequency_hz)
        or not np.isfinite(frequency_hz)
        or frequency_hz <= 0
    ):
        raise ValueError("frequency_hz must be a positive finite scalar")

    unit = direction_unit_vector(direction)
    names = (
        tuple(str(index) for index in range(positions.shape[0]))
        if element_names is None
        else tuple(element_names)
    )
    if len(names) != positions.shape[0]:
        raise ValueError("element_names must contain one name per position")

    spatial = np.exp(
        -1j
        * 2.0
        * np.pi
        * frequency_hz
        / _SPEED_OF_LIGHT_MPS
        * (positions @ unit)
    )
    selected_pattern = IsotropicElementPattern() if pattern is None else pattern
    gains = np.asarray(
        [selected_pattern.complex_gain(name, direction, frequency_hz) for name in names],
        dtype=np.complex128,
    )
    return gains * spatial
