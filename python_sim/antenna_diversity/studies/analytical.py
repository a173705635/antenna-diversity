"""Ideal, purely analytical spatial-study baseline records."""

from __future__ import annotations

import copy
from collections.abc import Iterable

import numpy as np

from antenna_diversity.array.geometry import CORNER_ORDER, maximum_baseline_m, select_phase_centers
from antenna_diversity.config import Direction, PhoneArrayConfig
from antenna_diversity.experiments.spatial_scan import (
    evaluate_configuration_spatially,
    scan_interferer_directions,
    scan_interferer_powers,
)

from .models import ResolvedStudyConfig, StageRecords, strict_snapshot
from .scenarios import build_multi_jammer_cases, sample_combined_errors, stable_case_id


def run_baseline(resolved: ResolvedStudyConfig) -> StageRecords:
    """Run no-sample ideal references plus direction and power scans."""

    reference_power_dbm = float(resolved.simulation.interferers[0].power_dbm)
    references = [
        _reference_row(resolved, _single_case(corner))
        for corner in resolved.study.single_antennas
    ]
    references.extend(
        _reference_row(resolved, _array_case(resolved, combination, mode))
        for combination in resolved.study.combinations
        for mode in resolved.study.spatial_modes
    )
    direction_rows = []
    power_rows = []
    for combination in resolved.study.combinations:
        for mode in resolved.study.spatial_modes:
            case = _array_case(resolved, combination, mode)
            direction_rows.extend(_direction_rows(resolved, case, reference_power_dbm))
            power_rows.extend(_power_rows(resolved, case, reference_power_dbm))

    return StageRecords(
        name="baseline",
        tables={
            "reference": tuple(references),
            "direction": tuple(direction_rows),
            "power": tuple(power_rows),
        },
    )


def run_single_factor_studies(resolved: ResolvedStudyConfig) -> StageRecords:
    """Evaluate one assumed-state error at a time, with matched EGC/MVDR cases."""

    _validate_single_factor_axes(resolved)
    rows = []
    for combination in resolved.study.combinations:
        baseline = _matched_metrics(_case_configuration(resolved, combination))
        for factor, values, variants in _single_factor_variants(resolved, combination):
            for value, corner, axis, simulation in variants:
                metrics = _matched_metrics(simulation)
                is_zero = value == 0.0
                rows.append(_robustness_row(
                    stage="single_factor", combination=combination, metrics=metrics,
                    parameters={"factor": factor, "value": value, "corner": corner, "axis": axis},
                    factor=factor, value=value,
                    severity=_severity(factor, value, values, resolved.simulation.diversity.diagonal_loading_factor),
                    corner=corner, axis=axis,
                    true_geometry_unchanged=True, true_state_unchanged=True,
                    zero_matches_baseline=(not is_zero or _matched_metrics_match(metrics, baseline)),
                    default_loading=(factor != "loading" or value == resolved.simulation.diversity.diagonal_loading_factor),
                ))
    return StageRecords(name="single_factor", tables={"single_factor": tuple(rows)})


def run_combined_error_studies(resolved: ResolvedStudyConfig) -> StageRecords:
    """Evaluate the exact deterministic realization for every level and subarray."""

    rows = []
    for combination in resolved.study.combinations:
        for level in resolved.study.error_levels:
            for realization in sample_combined_errors(resolved, level.name, combination):
                simulation = _case_configuration(resolved, combination)
                _apply_combined_realization(simulation, realization)
                metrics = _matched_metrics(simulation)
                realization_snapshot = realization.to_dict()
                rows.append(_robustness_row(
                    stage="combined_error", combination=combination, metrics=metrics,
                    parameters={"level": level.name, "sample_index": realization.sample_index,
                                "child_seed": realization.child_seed, "realization": realization_snapshot},
                    level=level.name, sample_index=realization.sample_index,
                    child_seed=realization.child_seed, realization=realization_snapshot,
                    true_geometry_unchanged=True, true_state_unchanged=True,
                ))
    return StageRecords(name="combined_error", tables={"combined_error": tuple(rows)})


def run_multi_jammer_studies(resolved: ResolvedStudyConfig) -> StageRecords:
    """Evaluate every template exactly, including underdetermined jammer cases."""

    rows = []
    for combination in resolved.study.combinations:
        for mode in resolved.study.spatial_modes:
            cases = build_multi_jammer_cases(resolved, resolved.simulation, combination, mode)
            for case, template in zip(cases, resolved.study.multi_jammer_templates, strict=True):
                simulation = _case_configuration(resolved, combination)
                simulation.interferers = tuple(copy.deepcopy(template.interferers))
                metrics = evaluate_configuration_spatially(simulation, mode).metrics
                enabled = [interferer for interferer in simulation.interferers if interferer.enable]
                gains = list(np.asarray(metrics["interferer_direction_gains"], dtype=float))
                jammer_rows = [
                    {**strict_snapshot(interferer), "direction_gain": float(gain)}
                    for interferer, gain in zip(enabled, gains, strict=True)
                ]
                rows.append(strict_snapshot({
                    "stage": "multi_jammer", "case_id": case.case_id,
                    "active_antennas": list(combination), "num_antennas": len(combination),
                    "mode": mode, "template": case.parameters["template"], "child_seed": case.seed,
                    "interferers": strict_snapshot(simulation.interferers),
                    "enabled_interferers": jammer_rows, "enabled_interferer_count": len(enabled),
                    "output_sinr_db": metrics["output_sinr_db"],
                    "worst_interferer_gain": metrics["worst_interferer_gain"],
                    "covariance_rank": metrics["covariance_rank"],
                    "covariance_condition_number": metrics["covariance_condition_number"],
                    "metrics": strict_snapshot(metrics),
                }))  # type: ignore[arg-type]
    return StageRecords(name="multi_jammer", tables={"multi_jammer": tuple(rows)})


def _single_factor_variants(resolved: ResolvedStudyConfig, combination: tuple[str, ...]):
    """Yield factor metadata and independently copied one-error cases."""

    for factor, values in (
        ("direction_azimuth", resolved.study.direction_errors_deg),
        ("direction_elevation", resolved.study.direction_errors_deg),
        ("position", resolved.study.position_errors_m),
        ("amplitude", resolved.study.amplitude_errors_db),
        ("phase", resolved.study.phase_errors_deg),
        ("loading", resolved.study.loading_factors),
    ):
        variants = []
        for raw_value in values:
            value = float(raw_value)
            if factor == "position":
                for corner in combination:
                    for axis_index, axis in enumerate(("x", "y", "z")):
                        simulation = _case_configuration(resolved, combination)
                        assumed = list(simulation.phone.assumed_phase_centers_m[corner])  # type: ignore[union-attr]
                        assumed[axis_index] += value
                        simulation.phone.assumed_phase_centers_m[corner] = tuple(assumed)  # type: ignore[union-attr]
                        variants.append((value, corner, axis, simulation))
            else:
                simulation = _case_configuration(resolved, combination)
                if factor == "direction_azimuth":
                    assumed = simulation.signal.assumed_direction
                    simulation.signal.assumed_direction = Direction(assumed.azimuth_deg + value, assumed.elevation_deg)  # type: ignore[union-attr]
                elif factor == "direction_elevation":
                    assumed = simulation.signal.assumed_direction
                    simulation.signal.assumed_direction = Direction(assumed.azimuth_deg, assumed.elevation_deg + value)  # type: ignore[union-attr]
                elif factor == "amplitude":
                    for corner in combination:
                        estimated = _case_configuration(resolved, combination)
                        estimated.array.estimated_channel_amplitude_error_db[corner] = value  # type: ignore[union-attr]
                        variants.append((value, corner, None, estimated))
                    continue
                elif factor == "phase":
                    for corner in combination:
                        estimated = _case_configuration(resolved, combination)
                        estimated.array.estimated_channel_phase_error_deg[corner] = value  # type: ignore[union-attr]
                        variants.append((value, corner, None, estimated))
                    continue
                else:
                    simulation.diversity.diagonal_loading_factor = value
                variants.append((value, None, None, simulation))
        yield factor, values, variants


def _validate_single_factor_axes(resolved: ResolvedStudyConfig) -> None:
    """Keep every single-error sweep anchored to its ideal configuration."""

    non_loading_axes = (
        resolved.study.direction_errors_deg,
        resolved.study.position_errors_m,
        resolved.study.amplitude_errors_db,
        resolved.study.phase_errors_deg,
    )
    if any(not np.any(np.asarray(axis) == 0.0) for axis in non_loading_axes):
        raise ValueError("single-factor non-loading axes must include zero")
    if not np.any(np.asarray(resolved.study.loading_factors) == resolved.simulation.diversity.diagonal_loading_factor):
        raise ValueError("single-factor loading axis must include the default loading factor")


def _case_configuration(resolved: ResolvedStudyConfig, combination: tuple[str, ...]):
    simulation = copy.deepcopy(resolved.simulation)
    if not isinstance(simulation.array, PhoneArrayConfig) or simulation.phone is None:
        raise ValueError("robustness studies require a phone-array configuration")
    simulation.array.active_antennas = tuple(combination)
    return simulation


def _matched_metrics(simulation):
    """Evaluate EGC and MVDR from distinct deep copies of the same case."""

    return {
        mode: evaluate_configuration_spatially(copy.deepcopy(simulation), mode).metrics
        for mode in ("egc", "mvdr")
    }


def _apply_combined_realization(simulation, realization) -> None:
    assumed = simulation.signal.assumed_direction
    simulation.signal.assumed_direction = Direction(
        assumed.azimuth_deg + realization.azimuth_error_deg,
        assumed.elevation_deg + realization.elevation_error_deg,
    )
    for corner, error in realization.position_errors_m.items():
        base = np.asarray(simulation.phone.assumed_phase_centers_m[corner], dtype=float)
        simulation.phone.assumed_phase_centers_m[corner] = tuple(base + np.asarray(error, dtype=float))
    for corner, value in realization.amplitude_errors_db.items():
        simulation.array.estimated_channel_amplitude_error_db[corner] = value
    for corner, value in realization.phase_errors_deg.items():
        simulation.array.estimated_channel_phase_error_deg[corner] = value


def _robustness_row(*, stage, combination, metrics, parameters, **fields):
    """Capture matched outputs and diagnostics in a strict, detached record."""

    pair_id = stable_case_id(stage, combination, "matched", parameters, None)
    row = {
        "stage": stage, "case_id": pair_id, "mvdr_case_id": pair_id, "egc_case_id": pair_id,
        "active_antennas": list(combination), "num_antennas": len(combination),
        "signed_value": fields.get("value"),
        "absolute_normalized_severity": fields.get("severity"),
        "egc_output_sinr_db": metrics["egc"]["output_sinr_db"],
        "mvdr_output_sinr_db": metrics["mvdr"]["output_sinr_db"],
        "mvdr_minus_egc_sinr_db": metrics["mvdr"]["output_sinr_db"] - metrics["egc"]["output_sinr_db"],
        "egc_metrics": strict_snapshot(metrics["egc"]), "mvdr_metrics": strict_snapshot(metrics["mvdr"]),
        "mvdr_metric_record": strict_snapshot(metrics["mvdr"]),
        **fields,
    }
    return strict_snapshot(row)  # type: ignore[return-value]


def _severity(factor: str, value: float, values, default_loading: float) -> float:
    if factor == "loading":
        distances = np.abs(np.log10(np.asarray(values, dtype=float) / default_loading))
        maximum = float(np.max(distances))
        raw = abs(float(np.log10(value / default_loading)))
    else:
        maximum = float(np.max(np.abs(values)))
        raw = abs(value)
    return 0.0 if maximum == 0.0 else raw / maximum


def _matched_metrics_match(first, second) -> bool:
    return all(
        np.isclose(first[mode]["output_sinr_db"], second[mode]["output_sinr_db"], rtol=0.0, atol=1e-12)
        for mode in ("egc", "mvdr")
    )


def _reference_row(resolved: ResolvedStudyConfig, case: dict[str, object]) -> dict[str, object]:
    simulation = _simulation_for_case(resolved, case, no_interferers=True)
    metrics = evaluate_configuration_spatially(simulation, str(case["mode"])).metrics
    return _row(
        table="reference",
        case=case,
        metrics=metrics,
        reference_interferer_power_dbm=float(resolved.simulation.interferers[0].power_dbm),
    )


def _direction_rows(
    resolved: ResolvedStudyConfig, case: dict[str, object], reference_power_dbm: float
) -> Iterable[dict[str, object]]:
    simulation = _simulation_for_case(resolved, case)
    target = resolved.simulation.signal.true_direction
    if target is None:
        raise ValueError("direction rows require a true target direction")
    scan = scan_interferer_directions(
        simulation,
        interferer_index=0,
        azimuths_deg=resolved.study.azimuths_deg,
        elevations_deg=resolved.study.elevations_deg,
        mode=str(case["mode"]),
    )
    for elevation_index, elevation_deg in enumerate(scan.elevations_deg):
        for azimuth_index, azimuth_deg in enumerate(scan.azimuths_deg):
            metrics = {
                name: values[elevation_index, azimuth_index]
                for name, values in scan.metrics.items()
            }
            metrics["interferer_gain"] = scan.interferer_gain[elevation_index, azimuth_index]
            yield _row(
                table="direction",
                case=case,
                metrics=metrics,
                reference_interferer_power_dbm=reference_power_dbm,
                elevation_index=elevation_index,
                azimuth_index=azimuth_index,
                elevation_deg=float(elevation_deg),
                azimuth_deg=float(azimuth_deg),
                target_azimuth_deg=float(target.azimuth_deg),
                target_elevation_deg=float(target.elevation_deg),
            )


def _power_rows(
    resolved: ResolvedStudyConfig, case: dict[str, object], reference_power_dbm: float
) -> Iterable[dict[str, object]]:
    simulation = _simulation_for_case(resolved, case)
    scan = scan_interferer_powers(
        simulation,
        interferer_index=0,
        powers_dbm=resolved.study.powers_dbm,
        mode=str(case["mode"]),
    )
    for power_index, power_dbm in enumerate(scan.parameter_values):
        metrics = {name: values[power_index] for name, values in scan.metrics.items()}
        yield _row(
            table="power",
            case=case,
            metrics=metrics,
            reference_interferer_power_dbm=reference_power_dbm,
            power_index=power_index,
            power_dbm=float(power_dbm),
            azimuth_deg=float(simulation.interferers[0].direction.azimuth_deg),
            elevation_deg=float(simulation.interferers[0].direction.elevation_deg),
        )


def _single_case(corner: str) -> dict[str, object]:
    companion = next(name for name in CORNER_ORDER if name != corner)
    return {
        "active_antennas": [corner],
        "simulation_active_antennas": [corner, companion],
        "num_antennas": 1,
        "mode": "single",
        "display_label": f"SINGLE({corner})",
        "maximum_baseline_m": 0.0,
    }


def _array_case(
    resolved: ResolvedStudyConfig, combination: tuple[str, ...], mode: str
) -> dict[str, object]:
    active_antennas = list(combination)
    return {
        "active_antennas": active_antennas,
        "simulation_active_antennas": active_antennas,
        "num_antennas": len(active_antennas),
        "mode": mode,
        "display_label": f"{mode.upper()}({'+'.join(combination)})",
        "maximum_baseline_m": float(maximum_baseline_m(select_phase_centers(
            resolved.simulation.phone.phase_centers_m, combination,
        ))),
    }


def _simulation_for_case(
    resolved: ResolvedStudyConfig,
    case: dict[str, object],
    *,
    no_interferers: bool = False,
):
    simulation = copy.deepcopy(resolved.simulation)
    simulation.array.active_antennas = tuple(case["simulation_active_antennas"])
    if no_interferers:
        simulation.interferers = ()
    return simulation


def _row(
    *,
    table: str,
    case: dict[str, object],
    metrics: dict[str, object],
    reference_interferer_power_dbm: float,
    **axes: object,
) -> dict[str, object]:
    parameters = {
        "table": table,
        "reference_interferer_power_dbm": reference_interferer_power_dbm,
        **axes,
    }
    row = {
        "stage": "baseline",
        "case_id": stable_case_id(
            f"baseline_{table}",
            tuple(case["simulation_active_antennas"]),
            str(case["mode"]),
            {"public_case": case, **parameters},
            None,
        ),
        **case,
        "reference_interferer_power_dbm": reference_interferer_power_dbm,
        "reference_power_dbm": reference_interferer_power_dbm,
        **axes,
        **metrics,
    }
    return strict_snapshot(row)  # type: ignore[return-value]
