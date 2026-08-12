"""Single-channel, equal-gain, and loaded-MVDR combining."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from antenna_diversity.config import DiversityConfig


def combine(
    x: ArrayLike,
    desired_steering: ArrayLike,
    mode: str,
    cfg: DiversityConfig,
) -> tuple[NDArray[np.complex128], dict[str, Any]]:
    """Combine antennas under a unit-response constraint for the target."""

    channels = np.asarray(x, dtype=np.complex128)
    if channels.ndim != 2 or channels.size == 0:
        raise ValueError("x must be a nonempty antennas-by-samples matrix")
    a = np.asarray(desired_steering, dtype=np.complex128).reshape(-1)
    n_antennas = channels.shape[0]
    if a.size != n_antennas:
        raise ValueError("desired_steering must contain one value per antenna")
    if np.linalg.norm(a) == 0:
        raise ValueError("desired_steering must be nonzero")

    covariance = channels @ channels.conj().T / channels.shape[1]
    loading = (
        cfg.diagonal_loading_factor
        * float(np.trace(covariance).real)
        / n_antennas
    )
    if not np.isfinite(loading) or loading <= 0:
        loading = float(np.spacing(max(1.0, np.linalg.norm(covariance, "fro"))))
    loaded_covariance = covariance + loading * np.eye(n_antennas)

    normalized_mode = str(mode).lower()
    if normalized_mode == "single":
        if abs(a[0]) < np.finfo(float).eps:
            raise ValueError("first antenna has zero desired response")
        weights = np.zeros(n_antennas, dtype=np.complex128)
        weights[0] = 1.0 / np.conj(a[0])
    elif normalized_mode == "egc":
        weights = a / np.vdot(a, a)
    elif normalized_mode == "mvdr":
        solved = np.linalg.solve(loaded_covariance, a)
        denominator = np.vdot(a, solved)
        if abs(denominator) < np.finfo(float).eps:
            raise ValueError("loaded covariance cannot support desired constraint")
        weights = solved / denominator
    else:
        raise ValueError(f"unknown diversity mode: {mode}")

    combined = weights.conj() @ channels
    info: dict[str, Any] = {
        "mode": normalized_mode,
        "weights": weights,
        "desired_response": np.vdot(weights, a),
        "weight_norm": float(np.linalg.norm(weights)),
        "condition_number": float(np.linalg.cond(loaded_covariance)),
        "diagonal_loading": loading,
        "sample_covariance": covariance,
        "loaded_covariance": loaded_covariance,
    }
    return combined, info
