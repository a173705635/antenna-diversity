"""Ideal, purely analytical spatial-study baseline records."""

from __future__ import annotations

import copy
from collections.abc import Iterable

from antenna_diversity.array.geometry import CORNER_ORDER, maximum_baseline_m, select_phase_centers
from antenna_diversity.experiments.spatial_scan import (
    evaluate_configuration_spatially,
    scan_interferer_directions,
    scan_interferer_powers,
)

from .models import ResolvedStudyConfig, StageRecords, strict_snapshot
from .scenarios import stable_case_id


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
