"""Sample-driven blind detection and projection nulling."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from antenna_diversity.config import BlindNullConfig


def _is_finite_real_scalar(value: object) -> bool:
    """Return whether *value* is one finite, non-complex numeric scalar."""

    candidate = np.asarray(value)
    return (
        candidate.ndim == 0
        and np.issubdtype(candidate.dtype, np.number)
        and not np.issubdtype(candidate.dtype, np.complexfloating)
        and bool(np.isfinite(candidate))
    )


def _finite_channels(x: ArrayLike) -> NDArray[np.complex128]:
    """Validate and copy a channels-by-samples matrix."""

    samples = np.asarray(x)
    if samples.ndim != 2:
        raise ValueError("x must be a two-dimensional channels-by-samples matrix")
    if not 2 <= samples.shape[0] <= 4:
        raise ValueError("x must contain at least two and at most four channels")
    if samples.shape[1] == 0:
        raise ValueError("x must contain nonempty sample vectors")
    try:
        channels = np.array(samples, dtype=np.complex128, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError("x must contain finite numeric values") from error
    if not np.all(np.isfinite(channels)):
        raise ValueError("x must contain finite values")
    return channels


def _validate_blind_config(cfg: BlindNullConfig, channels: int) -> None:
    """Validate blind-nulling configuration for a concrete channel count."""

    if not isinstance(cfg, BlindNullConfig):
        raise ValueError("blind-null configuration must be a BlindNullConfig")
    if (
        isinstance(cfg.detection_threshold_db, (bool, np.bool_))
        or not np.isscalar(cfg.detection_threshold_db)
        or not _is_finite_real_scalar(cfg.detection_threshold_db)
    ):
        raise ValueError("detection threshold must be a finite real scalar")
    if (
        isinstance(cfg.reference_channel, (bool, np.bool_))
        or not isinstance(cfg.reference_channel, (int, np.integer))
        or not 0 <= cfg.reference_channel < channels
    ):
        raise ValueError("reference channel must be a valid channel index")
    if (
        isinstance(cfg.min_reference_projection_power, (bool, np.bool_))
        or not np.isscalar(cfg.min_reference_projection_power)
        or not _is_finite_real_scalar(cfg.min_reference_projection_power)
        or not 0.0 < cfg.min_reference_projection_power <= 1.0
    ):
        raise ValueError("projection power must lie in (0, 1]")


def _condition_number(covariance: NDArray[np.complex128]) -> float:
    """Return the covariance condition number without emitting warnings."""

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return float(np.linalg.cond(covariance))


def _base_info(
    covariance: NDArray[np.complex128],
    snapshots: int,
    cfg: BlindNullConfig,
    channel_powers: NDArray[np.float64],
    correlation: NDArray[np.complex128],
    eigenvalues: NDArray[np.float64],
    score_db: float,
    maximum_pairwise_coherence: float,
    interference_vector: NDArray[np.complex128],
    reference_projection_power: float | None,
) -> dict[str, Any]:
    """Build diagnostics shared by the bypass and nulling paths."""

    return {
        "mode": "blind_null",
        "sample_covariance": covariance.copy(),
        "condition_number": _condition_number(covariance),
        "covariance_snapshots": snapshots,
        "channel_powers": channel_powers.copy(),
        "normalized_correlation": correlation.copy(),
        "correlation_eigenvalues": eigenvalues.copy(),
        "detection_score_db": score_db,
        "detection_threshold_db": float(cfg.detection_threshold_db),
        "maximum_pairwise_coherence": maximum_pairwise_coherence,
        "reference_channel": int(cfg.reference_channel),
        "reference_projection_power": reference_projection_power,
        "estimated_interference_vector": interference_vector.copy(),
    }


def _finalize_info(
    info: dict[str, Any],
    weights: NDArray[np.complex128],
    interference_detected: bool,
    nulling_enabled: bool,
    bypass_reason: str | None,
) -> dict[str, Any]:
    """Add combiner-specific diagnostics after choosing weights."""

    finalized = info.copy()
    finalized.update({
        "weights": weights.copy(),
        "weight_norm": float(np.linalg.norm(weights)),
        "weights_are_fixed": True,
        "interference_detected": interference_detected,
        "nulling_enabled": nulling_enabled,
        "bypass_reason": bypass_reason,
        "estimated_interference_response": np.vdot(
            weights, finalized["estimated_interference_vector"]
        ),
        "reference_response": weights[
            int(finalized["reference_channel"])
        ].conj(),
    })
    return finalized


def _bypass(
    channels: NDArray[np.complex128],
    cfg: BlindNullConfig,
    info: dict[str, Any],
    reason: str,
    interference_detected: bool,
) -> tuple[NDArray[np.complex128], dict[str, Any]]:
    """Return the selected reference channel and its unit basis weights."""

    weights = np.zeros(channels.shape[0], dtype=np.complex128)
    weights[cfg.reference_channel] = 1.0
    return (
        channels[cfg.reference_channel].copy(),
        _finalize_info(
            info,
            weights,
            interference_detected=interference_detected,
            nulling_enabled=False,
            bypass_reason=reason,
        ),
    )


def combine_blind_null(
    x: ArrayLike,
    cfg: BlindNullConfig,
) -> tuple[NDArray[np.complex128], dict[str, Any]]:
    """Detect one correlated interference mode and project it from a reference.

    The weights use only the provided DFE samples and the configured reference
    channel.  No desired-target steering vector participates in this API.
    """

    channels = _finite_channels(x)
    _validate_blind_config(cfg, channels.shape[0])
    snapshots = channels.shape[1]
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            covariance = channels @ channels.conj().T / snapshots
    except FloatingPointError as error:
        raise ValueError("sample covariance must contain finite values") from error
    covariance = (covariance + covariance.conj().T) / 2.0
    if not np.all(np.isfinite(covariance)):
        raise ValueError("sample covariance must contain finite values")
    channel_powers = np.maximum(np.real(np.diag(covariance)), 0.0)
    zero_correlation = np.zeros_like(covariance)
    zero_eigenvalues = np.zeros(channels.shape[0], dtype=float)
    zero_vector = np.zeros(channels.shape[0], dtype=np.complex128)

    if np.all(channel_powers == 0.0):
        info = _base_info(
            covariance,
            snapshots,
            cfg,
            channel_powers,
            zero_correlation,
            zero_eigenvalues,
            -np.inf,
            0.0,
            zero_vector,
            None,
        )
        return _bypass(channels, cfg, info, "zero_input_power", False)

    scales = np.sqrt(channel_powers)
    if np.any(scales == 0.0) or not np.all(np.isfinite(scales)):
        info = _base_info(
            covariance,
            snapshots,
            cfg,
            channel_powers,
            zero_correlation,
            zero_eigenvalues,
            -np.inf,
            0.0,
            zero_vector,
            None,
        )
        return _bypass(channels, cfg, info, "degenerate_interference_vector", False)

    correlation = covariance / np.outer(scales, scales)
    correlation = (correlation + correlation.conj().T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    residual = float(np.mean(eigenvalues[:-1]))
    if residual == 0.0:
        score_db = np.inf if eigenvalues[-1] > 0.0 else -np.inf
    else:
        score_db = float(10.0 * np.log10(eigenvalues[-1] / residual))
    coherence = np.abs(correlation).copy()
    np.fill_diagonal(coherence, 0.0)
    maximum_pairwise_coherence = float(np.max(coherence))

    mapped = scales * eigenvectors[:, -1]
    vector_norm = float(np.linalg.norm(mapped))
    if not np.isfinite(vector_norm) or vector_norm == 0.0:
        info = _base_info(
            covariance,
            snapshots,
            cfg,
            channel_powers,
            correlation,
            eigenvalues,
            score_db,
            maximum_pairwise_coherence,
            zero_vector,
            None,
        )
        return _bypass(channels, cfg, info, "degenerate_interference_vector", False)
    interference_vector = mapped / vector_norm

    info = _base_info(
        covariance,
        snapshots,
        cfg,
        channel_powers,
        correlation,
        eigenvalues,
        score_db,
        maximum_pairwise_coherence,
        interference_vector,
        None,
    )
    interference_detected = score_db >= cfg.detection_threshold_db
    if not interference_detected:
        return _bypass(channels, cfg, info, "no_interference_detected", False)

    q = np.eye(channels.shape[0], dtype=np.complex128)[:, cfg.reference_channel]
    projector = np.eye(channels.shape[0], dtype=np.complex128) - np.outer(
        interference_vector, interference_vector.conj()
    )
    projected_reference = projector @ q
    gamma = float(np.vdot(q, projected_reference).real)
    info["reference_projection_power"] = gamma
    if gamma < cfg.min_reference_projection_power:
        return _bypass(
            channels,
            cfg,
            info,
            "unsafe_reference_projection",
            True,
        )

    weights = projected_reference / gamma
    if not np.all(np.isfinite(weights)):
        return _bypass(
            channels,
            cfg,
            info,
            "degenerate_interference_vector",
            True,
        )
    combined = weights.conj() @ channels
    if not np.all(np.isfinite(combined)):
        raise ValueError("blind-null output must contain finite values")
    return combined.copy(), _finalize_info(
        info,
        weights,
        interference_detected=True,
        nulling_enabled=True,
        bypass_reason=None,
    )
