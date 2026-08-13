"""Purely analytical phone-array combination and parameter scans."""

from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from antenna_diversity.analysis.spatial import (
    analytic_covariance,
    evaluate_spatial_case,
)
from antenna_diversity.array.geometry import (
    active_corner_combinations,
    maximum_baseline_m,
    select_phase_centers,
)
from antenna_diversity.array.manifold import IsotropicElementPattern, array_manifold
from antenna_diversity.config import (
    CWInterferer,
    Direction,
    PhoneArrayConfig,
    SimulationConfig,
)
from antenna_diversity.diversity.combiner import compute_weights
from antenna_diversity.models import (
    SpatialCombinationResult,
    SpatialEvaluationResult,
    SpatialGridScanResult,
    SpatialParameterScanResult,
)
from antenna_diversity.rf.cascade import cascade_metrics


_BOLTZMANN_J_PER_K = 1.380649e-23
_SPATIAL_METRIC_NAMES = (
    "target_direction_gain",
    "interferer_direction_gains",
    "worst_interferer_gain",
    "output_signal_power_w",
    "output_interference_power_w",
    "output_noise_power_w",
    "output_sinr_db",
    "weight_norm",
    "white_noise_gain",
    "covariance_rank",
    "covariance_condition_number",
    "diagonal_loading",
    "covariance_snapshots",
    "weights_are_fixed",
)


def evaluate_configuration_spatially(
    cfg: SimulationConfig, mode: str
) -> SpatialEvaluationResult:
    """Evaluate one copied phone configuration without generating samples."""

    case = copy.deepcopy(cfg)
    case.validate()
    if not isinstance(case.array, PhoneArrayConfig) or case.phone is None:
        raise ValueError("spatial experiments require a phone-array configuration")
    if case.array.pattern_model != "isotropic":
        raise ValueError("spatial experiments support only the isotropic pattern")

    names = case.array.active_antennas
    true_positions = select_phase_centers(case.phone.phase_centers_m, names)
    assumed_positions = select_phase_centers(
        case.phone.assumed_phase_centers_m, names
    )
    pattern = IsotropicElementPattern()
    frequency_hz = case.signal.carrier_hz
    true_desired = array_manifold(
        true_positions,
        case.signal.true_direction,
        frequency_hz,
        element_names=names,
        pattern=pattern,
    ) * case.array.channel_response(estimated=False)
    assumed_desired = array_manifold(
        assumed_positions,
        case.signal.assumed_direction,
        frequency_hz,
        element_names=names,
        pattern=pattern,
    ) * case.array.channel_response(estimated=True)

    enabled_interferers = tuple(
        interferer for interferer in case.interferers if interferer.enable
    )
    interferer_steering = tuple(
        array_manifold(
            true_positions,
            interferer.direction,
            frequency_hz,
            element_names=names,
            pattern=pattern,
        )
        * case.array.channel_response(estimated=False)
        for interferer in enabled_interferers
    )
    signal_power_w = _dbm_to_watts("signal.power_dbm", case.signal.power_dbm)
    interferer_powers_w = tuple(
        _dbm_to_watts("interferer.power_dbm", interferer.power_dbm)
        for interferer in enabled_interferers
    )
    noise_factor = cascade_metrics(case.rf.stages)["noise_factor"]
    noise_power_w = (
        _BOLTZMANN_J_PER_K
        * case.rf.temperature_k
        * case.rf.noise_bandwidth_hz
        * noise_factor
    )
    covariance = analytic_covariance(
        true_desired,
        signal_power_w,
        interferer_steering,
        interferer_powers_w,
        noise_power_w,
    )
    weights, weight_info = compute_weights(
        covariance, assumed_desired, mode, case.diversity
    )
    metrics = evaluate_spatial_case(
        weights,
        true_desired,
        signal_power_w,
        interferer_steering,
        interferer_powers_w,
        noise_power_w,
        covariance=covariance,
        diagonal_loading=weight_info["diagonal_loading"],
        covariance_snapshots=None,
        weights_are_fixed=True,
    )
    return SpatialEvaluationResult(str(mode).lower(), metrics)


def compare_corner_combinations(
    cfg: SimulationConfig, mode: str
) -> list[SpatialCombinationResult]:
    """Evaluate all eleven two-to-four-corner subsets in stable order."""

    records = []
    for combination in active_corner_combinations():
        case = copy.deepcopy(cfg)
        _require_phone(case)
        case.array.active_antennas = combination
        positions = select_phase_centers(case.phone.phase_centers_m, combination)
        records.append(
            SpatialCombinationResult(
                active_antennas=combination,
                positions_m=tuple(tuple(float(value) for value in row) for row in positions),
                maximum_baseline_m=maximum_baseline_m(positions),
                mode=str(mode).lower(),
                metrics=evaluate_configuration_spatially(case, mode).metrics,
            )
        )
    return records


def scan_interferer_directions(
    cfg: SimulationConfig,
    interferer_index: int,
    azimuths_deg: ArrayLike,
    elevations_deg: ArrayLike,
    mode: str,
) -> SpatialGridScanResult:
    """Scan one interferer's true direction over an elevation-by-azimuth grid."""

    _validate_interferer_index(cfg, interferer_index, require_enabled=True)
    azimuths, elevations = _direction_grid(azimuths_deg, elevations_deg)
    selected_gain_index = sum(
        interferer.enable for interferer in cfg.interferers[:interferer_index]
    )

    def evaluate(azimuth: float, elevation: float) -> SpatialEvaluationResult:
        case = copy.deepcopy(cfg)
        interferers = list(case.interferers)
        interferers[interferer_index] = _replace_interferer(
            interferers[interferer_index], Direction(azimuth, elevation)
        )
        case.interferers = tuple(interferers)
        return evaluate_configuration_spatially(case, mode)

    metrics = _scan_grid(azimuths, elevations, evaluate)
    gain = np.asarray(metrics["interferer_direction_gains"], dtype=float)[
        ..., selected_gain_index
    ]
    return SpatialGridScanResult(azimuths, elevations, str(mode).lower(), metrics, gain)


def scan_assumed_target_directions(
    cfg: SimulationConfig,
    azimuths_deg: ArrayLike,
    elevations_deg: ArrayLike,
    mode: str,
) -> SpatialGridScanResult:
    """Scan assumed target direction while preserving the true direction."""

    azimuths, elevations = _direction_grid(azimuths_deg, elevations_deg)

    def evaluate(azimuth: float, elevation: float) -> SpatialEvaluationResult:
        case = copy.deepcopy(cfg)
        case.signal.assumed_direction = Direction(azimuth, elevation)
        return evaluate_configuration_spatially(case, mode)

    metrics = _scan_grid(azimuths, elevations, evaluate)
    jammer_gains = np.asarray(metrics["interferer_direction_gains"], dtype=float)
    gain = (
        np.max(jammer_gains, axis=-1)
        if jammer_gains.shape[-1] > 0
        else np.zeros((elevations.size, azimuths.size), dtype=float)
    )
    return SpatialGridScanResult(azimuths, elevations, str(mode).lower(), metrics, gain)


def scan_interferer_powers(
    cfg: SimulationConfig,
    interferer_index: int,
    powers_dbm: ArrayLike,
    mode: str,
) -> SpatialParameterScanResult:
    """Scan one interferer's input power in caller-supplied order."""

    _validate_interferer_index(cfg, interferer_index, require_enabled=True)
    powers = _finite_vector("powers_dbm", powers_dbm)

    def evaluate(power: float) -> SpatialEvaluationResult:
        case = copy.deepcopy(cfg)
        interferers = list(case.interferers)
        original = interferers[interferer_index]
        interferers[interferer_index] = CWInterferer(
            direction=original.direction,
            enable=original.enable,
            power_dbm=power,
            offset_hz=original.offset_hz,
            initial_phase_rad=original.initial_phase_rad,
            start_ms=original.start_ms,
            stop_ms=original.stop_ms,
        )
        case.interferers = tuple(interferers)
        return evaluate_configuration_spatially(case, mode)

    return _parameter_result("powers_dbm", powers, mode, evaluate)


def scan_diagonal_loading(
    cfg: SimulationConfig, factors: ArrayLike, mode: str
) -> SpatialParameterScanResult:
    """Scan positive diagonal-loading factors in caller-supplied order."""

    values = _finite_vector("factors", factors)
    if np.any(values <= 0.0):
        raise ValueError("factors must contain positive finite values")

    def evaluate(factor: float) -> SpatialEvaluationResult:
        case = copy.deepcopy(cfg)
        case.diversity.diagonal_loading_factor = factor
        return evaluate_configuration_spatially(case, mode)

    return _parameter_result("diagonal_loading_factor", values, mode, evaluate)


def scan_position_errors(
    cfg: SimulationConfig,
    corner_name: str,
    xyz_errors_m: ArrayLike,
    mode: str,
) -> SpatialParameterScanResult:
    """Add assumed phase-center errors without moving the true coordinate."""

    _require_phone(cfg)
    if not isinstance(corner_name, str) or corner_name not in cfg.phone.phase_centers_m:
        raise ValueError("corner_name must be a known phone corner")
    raw_errors = np.asarray(xyz_errors_m)
    if np.iscomplexobj(raw_errors):
        raise ValueError("xyz_errors_m must have finite real shape (N, 3)")
    try:
        errors = np.asarray(xyz_errors_m, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "xyz_errors_m must have finite real shape (N, 3)"
        ) from error
    if (
        errors.ndim != 2
        or errors.shape[0] == 0
        or errors.shape[1:] != (3,)
        or not np.all(np.isfinite(errors))
    ):
        raise ValueError("xyz_errors_m must have finite real shape (N, 3)")

    def evaluate(error: NDArray[np.float64]) -> SpatialEvaluationResult:
        case = copy.deepcopy(cfg)
        assumed = np.asarray(
            case.phone.assumed_phase_centers_m[corner_name], dtype=float
        )
        case.phone.assumed_phase_centers_m[corner_name] = tuple(assumed + error)
        return evaluate_configuration_spatially(case, mode)

    return _parameter_result("xyz_errors_m", errors, mode, evaluate)


def _scan_grid(
    azimuths: NDArray[np.float64],
    elevations: NDArray[np.float64],
    evaluate: Callable[[float, float], SpatialEvaluationResult],
) -> dict[str, NDArray]:
    cases = [
        evaluate(float(azimuth), float(elevation))
        for elevation in elevations
        for azimuth in azimuths
    ]
    return _collect_metrics(cases, (elevations.size, azimuths.size))


def _parameter_result(
    name: str,
    values: NDArray,
    mode: str,
    evaluate: Callable[[object], SpatialEvaluationResult],
) -> SpatialParameterScanResult:
    metrics = _collect_metrics([evaluate(value) for value in values], (len(values),))
    return SpatialParameterScanResult(name, values.copy(), str(mode).lower(), metrics)


def _collect_metrics(
    cases: list[SpatialEvaluationResult], shape: tuple[int, ...]
) -> dict[str, NDArray]:
    expected = set(_SPATIAL_METRIC_NAMES)
    for case in cases:
        if set(case.metrics) != expected:
            raise ValueError("spatial metric schema contains unknown or missing fields")

    metrics: dict[str, NDArray] = {}
    for name in _SPATIAL_METRIC_NAMES:
        values = [case.metrics[name] for case in cases]
        first_shape = np.asarray(values[0]).shape
        if any(np.asarray(value).shape != first_shape for value in values):
            raise ValueError(f"spatial metric {name} has inconsistent shape")
        collected = np.asarray(values).reshape(shape + first_shape)
        nonnull = [value for value in collected.flat if value is not None]
        if nonnull and not np.all(np.isfinite(np.asarray(nonnull))):
            raise ValueError("scan metrics must contain only finite values or None")
        metrics[name] = collected
    return metrics


def _direction_grid(
    azimuths_deg: ArrayLike, elevations_deg: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    azimuths = _finite_vector("azimuths_deg", azimuths_deg)
    elevations = _finite_vector("elevations_deg", elevations_deg)
    if np.any((elevations < 0.0) | (elevations > 90.0)):
        raise ValueError("elevations_deg must lie in [0, 90]")
    return azimuths, elevations


def _finite_vector(name: str, values: ArrayLike) -> NDArray[np.float64]:
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be a nonempty finite real one-dimensional array")
    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must be a nonempty finite real one-dimensional array"
        ) from error
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a nonempty finite real one-dimensional array")
    return vector


def _validate_interferer_index(
    cfg: SimulationConfig, index: int, *, require_enabled: bool = False
) -> None:
    if (
        not isinstance(index, (int, np.integer))
        or isinstance(index, (bool, np.bool_))
        or not 0 <= int(index) < len(cfg.interferers)
    ):
        raise ValueError("interferer_index must select an existing interferer")
    if require_enabled and not cfg.interferers[int(index)].enable:
        raise ValueError("interferer_index must select an enabled interferer")


def _require_phone(cfg: SimulationConfig) -> None:
    if not isinstance(cfg.array, PhoneArrayConfig) or cfg.phone is None:
        raise ValueError("spatial experiments require a phone-array configuration")


def _replace_interferer(interferer: CWInterferer, direction: Direction) -> CWInterferer:
    return CWInterferer(
        direction=direction,
        enable=interferer.enable,
        power_dbm=interferer.power_dbm,
        offset_hz=interferer.offset_hz,
        initial_phase_rad=interferer.initial_phase_rad,
        start_ms=interferer.start_ms,
        stop_ms=interferer.stop_ms,
    )


def _dbm_to_watts(name: str, power_dbm: float) -> float:
    value = np.asarray(power_dbm)
    if value.ndim != 0 or np.iscomplexobj(value) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar")
    with np.errstate(over="ignore"):
        power_w = float(1e-3 * 10.0 ** (float(value) / 10.0))
    if not np.isfinite(power_w) or power_w <= 0.0:
        raise ValueError(f"{name} must convert to finite positive watts")
    return power_w
