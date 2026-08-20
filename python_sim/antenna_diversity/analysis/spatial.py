"""Purely analytical spatial-array covariance and response metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from antenna_diversity.array.phone_model import PhoneAntennaModel
from antenna_diversity.config import (
    Direction,
    PhoneArrayConfig,
    SimulationConfig,
)


_COVARIANCE_HERMITIAN_RTOL = 1e-9
_COVARIANCE_HERMITIAN_ATOL = 1e-12


def analytic_covariance(
    desired_steering: ArrayLike,
    signal_power_w: float,
    interferer_steering: Sequence[ArrayLike],
    interferer_powers_w: Sequence[float],
    noise_power_per_element_w: float,
) -> NDArray[np.complex128]:
    """Return the exact signal, interferer, and white-noise covariance sum."""

    desired = _finite_vector("desired_steering", desired_steering)
    signal_power = _nonnegative_power("signal_power_w", signal_power_w)
    noise_power = _nonnegative_power(
        "noise_power_per_element_w", noise_power_per_element_w
    )
    interferers, interferer_powers = _interferers_and_powers(
        interferer_steering,
        interferer_powers_w,
        desired.size,
    )

    with np.errstate(over="ignore", invalid="ignore"):
        covariance = signal_power * np.outer(desired, desired.conj())
        for steering, power in zip(interferers, interferer_powers, strict=True):
            covariance += power * np.outer(steering, steering.conj())
        covariance += noise_power * np.eye(desired.size)
    if not np.all(np.isfinite(covariance)):
        raise ValueError("analytical covariance must contain finite values")
    return np.asarray(covariance, dtype=np.complex128)


def evaluate_spatial_case(
    weights,
    desired_steering,
    signal_power_w,
    interferer_steering,
    interferer_powers_w,
    noise_power_per_element_w,
    covariance=None,
    diagonal_loading=None,
    covariance_snapshots=None,
    weights_are_fixed=True,
) -> dict[str, Any]:
    """Evaluate the exact output-power decomposition for fixed weights."""

    w = _finite_vector("weights", weights)
    desired = _finite_vector("desired_steering", desired_steering)
    if desired.size != w.size:
        raise ValueError("desired_steering must contain one value per weight")
    signal_power = _nonnegative_power("signal_power_w", signal_power_w)
    noise_power = _nonnegative_power(
        "noise_power_per_element_w", noise_power_per_element_w
    )
    interferers, interferer_powers = _interferers_and_powers(
        interferer_steering,
        interferer_powers_w,
        w.size,
    )

    target_response = np.vdot(w, desired)
    jammer_responses = tuple(
        np.vdot(w, steering) for steering in interferers
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        output_signal_power_w = signal_power * abs(target_response) ** 2
        output_interference_power_w = sum(
            power * abs(response) ** 2
            for power, response in zip(
                interferer_powers, jammer_responses, strict=True
            )
        )
        white_noise_gain = float(np.vdot(w, w).real)
        output_noise_power_w = noise_power * white_noise_gain

    powers = (
        output_signal_power_w,
        output_interference_power_w,
        output_noise_power_w,
        white_noise_gain,
    )
    if not all(np.isfinite(value) and value >= 0.0 for value in powers):
        raise ValueError("spatial output powers must be finite and nonnegative")

    denominator = max(
        output_interference_power_w + output_noise_power_w,
        np.finfo(float).eps,
    )
    with np.errstate(divide="ignore"):
        output_sinr_db = float(
            10.0 * np.log10(output_signal_power_w / denominator)
        )
    if np.isnan(output_sinr_db) or np.isposinf(output_sinr_db):
        raise ValueError("output SINR must not be NaN or positive infinity")

    covariance_rank: int | None = None
    covariance_condition_number: float | None = None
    if covariance is not None:
        covariance_matrix = np.asarray(covariance, dtype=np.complex128)
        if covariance_matrix.shape != (w.size, w.size):
            raise ValueError("covariance must be square with one row per weight")
        if not np.all(np.isfinite(covariance_matrix)):
            raise ValueError("covariance must contain finite values")
        if not np.allclose(
            covariance_matrix,
            covariance_matrix.conj().T,
            rtol=_COVARIANCE_HERMITIAN_RTOL,
            atol=_COVARIANCE_HERMITIAN_ATOL,
        ):
            raise ValueError("covariance must be Hermitian within tolerance")
        covariance_rank = int(np.linalg.matrix_rank(covariance_matrix))
        covariance_condition_number = float(np.linalg.cond(covariance_matrix))

    loading: float | None = None
    if diagonal_loading is not None:
        loading = _nonnegative_power("diagonal_loading", diagonal_loading)

    snapshots: int | None = None
    if covariance_snapshots is not None:
        if (
            not isinstance(covariance_snapshots, (int, np.integer))
            or isinstance(covariance_snapshots, (bool, np.bool_))
            or covariance_snapshots < 0
        ):
            raise ValueError("covariance_snapshots must be a nonnegative integer")
        snapshots = int(covariance_snapshots)

    if not isinstance(weights_are_fixed, (bool, np.bool_)):
        raise ValueError("weights_are_fixed must be boolean")

    target_gain = float(abs(target_response))
    interferer_gains = [float(abs(response)) for response in jammer_responses]
    return {
        "target_direction_gain": target_gain,
        "interferer_direction_gains": interferer_gains,
        "worst_interferer_gain": max(interferer_gains, default=0.0),
        "output_signal_power_w": float(output_signal_power_w),
        "output_interference_power_w": float(output_interference_power_w),
        "output_noise_power_w": float(output_noise_power_w),
        "output_sinr_db": output_sinr_db,
        "weight_norm": float(np.linalg.norm(w)),
        "white_noise_gain": white_noise_gain,
        "covariance_rank": covariance_rank,
        "covariance_condition_number": covariance_condition_number,
        "diagonal_loading": loading,
        "covariance_snapshots": snapshots,
        "weights_are_fixed": bool(weights_are_fixed),
    }


def response_grid_db(
    cfg: SimulationConfig,
    weights: ArrayLike,
    azimuths_deg: ArrayLike,
    elevations_deg: ArrayLike,
    model_view: str = "true",
) -> NDArray[np.float64]:
    """Return a phone-array response grid normalized to 0 dB."""

    if not isinstance(cfg.array, PhoneArrayConfig) or cfg.phone is None:
        raise ValueError("response_grid_db requires a phone-array configuration")

    w = _finite_vector("weights", weights)
    if w.size != cfg.array.num_elements:
        raise ValueError("weights must contain one value per active antenna")
    if not np.any(w != 0.0):
        raise ValueError("weights must be nonzero")
    azimuths = _direction_grid("azimuths_deg", azimuths_deg)
    elevations = _direction_grid("elevations_deg", elevations_deg)
    if np.any((elevations < 0.0) | (elevations > 90.0)):
        raise ValueError("elevations_deg must lie in [0, 90]")

    view = str(model_view).lower()
    if view not in {"true", "assumed"}:
        raise ValueError("model_view must be true or assumed")
    model = PhoneAntennaModel.from_config(cfg)
    polarization = (
        cfg.signal.true_polarization
        if view == "true"
        else cfg.signal.assumed_polarization
    )
    estimated = view == "assumed"
    response = np.empty((elevations.size, azimuths.size), dtype=float)
    for elevation_index, elevation in enumerate(elevations):
        for azimuth_index, azimuth in enumerate(azimuths):
            manifold = model.effective_manifold(
                Direction(float(azimuth), float(elevation)),
                polarization,
                cfg.signal.carrier_hz,
                estimated=estimated,
            )
            response[elevation_index, azimuth_index] = abs(np.vdot(w, manifold))

    return _normalized_response_db(response, floor_db=-80.0)


def _normalized_response_db(
    response: NDArray[np.float64], floor_db: float
) -> NDArray[np.float64]:
    peak = float(np.max(response))
    if not np.isfinite(peak):
        raise ValueError("response grid must have a finite maximum")
    if peak == 0.0:
        return np.full(response.shape, floor_db)
    with np.errstate(divide="ignore"):
        response_db = 20.0 * np.log10(response / peak)
    return np.maximum(response_db, floor_db)


def _finite_vector(name: str, values: ArrayLike) -> NDArray[np.complex128]:
    vector = np.asarray(values, dtype=np.complex128)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain finite values")
    return vector


def _nonnegative_power(name: str, value: float) -> float:
    value_array = np.asarray(value)
    if value_array.ndim != 0 or np.iscomplexobj(value_array):
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    numeric_value = float(value_array)
    if not np.isfinite(numeric_value) or numeric_value < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    return numeric_value


def _interferers_and_powers(
    interferer_steering: Sequence[ArrayLike],
    interferer_powers_w: Sequence[float],
    vector_size: int,
) -> tuple[tuple[NDArray[np.complex128], ...], tuple[float, ...]]:
    interferers = tuple(interferer_steering)
    powers = tuple(interferer_powers_w)
    if len(interferers) != len(powers):
        raise ValueError(
            "interferer_steering and interferer_powers_w must have equal lengths"
        )
    vectors = tuple(
        _finite_vector("interferer steering", steering)
        for steering in interferers
    )
    if any(vector.size != vector_size for vector in vectors):
        raise ValueError("each interferer steering vector must match the array size")
    validated_powers = tuple(
        _nonnegative_power("interferer power", power) for power in powers
    )
    return vectors, validated_powers


def _direction_grid(name: str, values: ArrayLike) -> NDArray[np.float64]:
    grid = np.asarray(values, dtype=float)
    if grid.ndim != 1 or grid.size == 0 or not np.all(np.isfinite(grid)):
        raise ValueError(f"{name} must be a nonempty finite one-dimensional grid")
    return grid
