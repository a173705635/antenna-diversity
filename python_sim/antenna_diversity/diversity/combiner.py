"""Single-channel, equal-gain, and loaded-MVDR combining."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from antenna_diversity.config import DiversityConfig


def compute_weights(
    covariance: ArrayLike,
    desired_steering: ArrayLike,
    mode: str,
    cfg: DiversityConfig,
) -> tuple[NDArray[np.complex128], dict[str, Any]]:
    """Compute fixed combiner weights under a unit target constraint."""

    covariance_matrix = np.asarray(covariance, dtype=np.complex128)
    if (
        covariance_matrix.ndim != 2
        or covariance_matrix.shape[0] == 0
        or covariance_matrix.shape[0] != covariance_matrix.shape[1]
    ):
        raise ValueError("covariance must be a nonempty square matrix")
    if not np.all(np.isfinite(covariance_matrix)):
        raise ValueError("covariance must contain finite values")

    a = np.asarray(desired_steering, dtype=np.complex128).reshape(-1)
    n_antennas = covariance_matrix.shape[0]
    if a.size != n_antennas:
        raise ValueError("desired_steering must contain one value per antenna")
    if not np.all(np.isfinite(a)):
        raise ValueError("desired_steering must contain finite values")
    if not np.any(a != 0):
        raise ValueError("desired_steering must be nonzero")

    with np.errstate(over="ignore", invalid="ignore"):
        loading = (
            cfg.diagonal_loading_factor
            * float(np.trace(covariance_matrix).real)
            / n_antennas
        )
    if not np.isfinite(loading) or loading <= 0:
        covariance_scale = max(
            1.0,
            float(
                np.max(
                    np.maximum(
                        np.abs(covariance_matrix.real),
                        np.abs(covariance_matrix.imag),
                    )
                )
            ),
        )
        loading = float(np.spacing(covariance_scale))
    if not np.isfinite(loading) or loading <= 0:
        raise ValueError("diagonal loading must be finite and positive")
    with np.errstate(over="ignore", invalid="ignore"):
        loaded_covariance = covariance_matrix + loading * np.eye(n_antennas)
    if not np.all(np.isfinite(loaded_covariance)):
        raise ValueError("loaded covariance must contain finite values")

    normalized_mode = str(mode).lower()
    if normalized_mode == "single":
        if abs(a[0]) < np.finfo(float).eps:
            raise ValueError("first antenna has zero desired response")
        weights = np.zeros(n_antennas, dtype=np.complex128)
        weights[0] = 1.0 / np.conj(a[0])
    elif normalized_mode == "egc":
        denominator = np.vdot(a, a)
        if not np.isfinite(denominator) or denominator == 0:
            raise ValueError("desired constraint cannot be normalized")
        weights = a / denominator
    elif normalized_mode == "mvdr":
        solved = np.linalg.solve(loaded_covariance, a)
        if not np.all(np.isfinite(solved)):
            raise ValueError("loaded covariance solve returned nonfinite values")
        denominator = np.vdot(a, solved)
        if not np.isfinite(denominator) or denominator == 0:
            raise ValueError("loaded covariance cannot support desired constraint")
        weights = solved / denominator
    else:
        raise ValueError(f"unknown diversity mode: {mode}")

    if not np.all(np.isfinite(weights)):
        raise ValueError("combiner weights must contain finite values")
    desired_response = np.vdot(weights, a)
    if not np.isfinite(desired_response):
        raise ValueError("desired response must be finite")
    info: dict[str, Any] = {
        "constraint_steering": a,
        "desired_response": desired_response,
        "diagonal_loading": loading,
        "loaded_covariance": loaded_covariance,
    }
    return weights, info


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
    if not np.all(np.isfinite(channels)):
        raise ValueError("x must contain finite values")
    a = np.asarray(desired_steering, dtype=np.complex128).reshape(-1)
    n_antennas = channels.shape[0]
    if a.size != n_antennas:
        raise ValueError("desired_steering must contain one value per antenna")

    covariance = channels @ channels.conj().T / channels.shape[1]
    weights, info = compute_weights(covariance, a, mode, cfg)

    combined = weights.conj() @ channels
    info.update({
        "mode": str(mode).lower(),
        "weights": weights,
        "weight_norm": float(np.linalg.norm(weights)),
        "condition_number": float(np.linalg.cond(info["loaded_covariance"])),
        "sample_covariance": covariance,
    })
    return combined, info
