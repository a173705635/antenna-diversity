"""Mate 70 antenna phase-center geometry helpers."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


CORNER_ORDER = ("TL", "TR", "BL", "BR")
_PHONE_WIDTH_M = 0.0759
_PHONE_LENGTH_M = 0.1609


def mate70_phase_centers(
    inset_x_m: float = 0.0,
    inset_y_m: float = 0.0,
    z_offsets_m: Mapping[str, float] | None = None,
) -> dict[str, tuple[float, float, float]]:
    """Return Mate 70 corner phase centers in metres.

    The coordinate origin is at the centre of the phone outline.  Positive
    ``y`` points toward the top of the phone and positive ``x`` to its right.
    """

    _validate_inset("inset_x_m", inset_x_m, _PHONE_WIDTH_M / 2.0)
    _validate_inset("inset_y_m", inset_y_m, _PHONE_LENGTH_M / 2.0)

    z = {name: 0.0 for name in CORNER_ORDER}
    if z_offsets_m is not None:
        unknown_names = set(z_offsets_m) - set(CORNER_ORDER)
        if unknown_names:
            raise ValueError("z_offsets_m must use known corner names")
        for name, offset in z_offsets_m.items():
            if not np.isfinite(offset):
                raise ValueError("z_offsets_m values must be finite")
            z[name] = float(offset)

    half_width = _PHONE_WIDTH_M / 2.0 - inset_x_m
    half_length = _PHONE_LENGTH_M / 2.0 - inset_y_m
    return {
        "TL": (-half_width, half_length, z["TL"]),
        "TR": (half_width, half_length, z["TR"]),
        "BL": (-half_width, -half_length, z["BL"]),
        "BR": (half_width, -half_length, z["BR"]),
    }


def select_phase_centers(
    phase_centers_m: Mapping[str, Sequence[float]],
    corner_names: Sequence[str],
) -> NDArray[np.float64]:
    """Return requested finite 3-D phase centers in the supplied order."""

    requested_names = tuple(corner_names)
    if any(name not in CORNER_ORDER for name in requested_names):
        raise ValueError("corner_names must use known corner names")
    if set(phase_centers_m) != set(CORNER_ORDER):
        raise ValueError("phase_centers_m must use known corner names")

    positions = np.asarray([phase_centers_m[name] for name in requested_names], dtype=float)
    if positions.ndim != 2 or positions.shape[1:] != (3,) or not np.all(np.isfinite(positions)):
        raise ValueError("phase centers must be finite 3-vectors")
    return positions


def active_corner_combinations() -> tuple[tuple[str, ...], ...]:
    """Return every stable-order two-to-four-corner active array subset."""

    return tuple(
        combination
        for size in (2, 3, 4)
        for combination in itertools.combinations(CORNER_ORDER, size)
    )


def maximum_baseline_m(positions_m: Sequence[Sequence[float]]) -> float:
    """Return the longest pairwise distance among finite 3-D positions."""

    positions = np.asarray(positions_m, dtype=float)
    if positions.ndim != 2 or positions.shape[0] < 2 or positions.shape[1:] != (3,):
        raise ValueError("positions_m must contain at least two 3-D positions")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions_m must contain finite 3-D positions")

    pairwise_distances = np.linalg.norm(positions[:, np.newaxis] - positions, axis=-1)
    return float(np.max(pairwise_distances))


def _validate_inset(name: str, inset_m: float, maximum_m: float) -> None:
    if not np.isfinite(inset_m) or not 0.0 <= inset_m <= maximum_m:
        raise ValueError(f"{name} must be finite, nonnegative, and not invert the outline")
