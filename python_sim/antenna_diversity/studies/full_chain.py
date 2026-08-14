"""Compact sample-domain validation for selected spatial-study cases."""

from __future__ import annotations

import copy
from collections.abc import Mapping

import numpy as np

from antenna_diversity.config import CWInterferer, Direction, PhoneArrayConfig
from antenna_diversity.experiments.spatial_scan import evaluate_configuration_spatially
from antenna_diversity.pipeline import run_end_to_end

from .models import FullChainRecords, ResolvedStudyConfig, SelectionResult, StudyCase, strict_snapshot
from .ranking import validate_interferer_snapshots
from .scenarios import derive_child_seed


def run_selected_full_chain(
    resolved: ResolvedStudyConfig, selection: SelectionResult
) -> FullChainRecords:
    """Run each selected case and retain only strict-JSON summary records."""

    metric_rows: list[dict[str, object]] = []
    cn0_rows: list[dict[str, object]] = []
    seen_case_ids: set[str] = set()
    for case in selection.cases:
        if case.case_id in seen_case_ids:
            continue
        seen_case_ids.add(case.case_id)
        for run_index in range(resolved.study.full_chain_run_count):
            seed = derive_child_seed(
                resolved.study.seed, f"full-chain:{case.case_id}", run_index
            )
            simulation = _reconstruct_case(resolved, case, seed)
            analytic_sinr = {
                mode: float(
                    evaluate_configuration_spatially(copy.deepcopy(simulation), mode)
                    .metrics["output_sinr_db"]
                )
                for mode in simulation.diversity.modes
            }
            results = run_end_to_end(simulation)
            metrics_by_mode = {str(row["mode"]): row for row in results.metrics}
            for mode, branch in results.branches.items():
                metric = metrics_by_mode[mode]
                tracking = branch.tracking
                row = _metric_row(case, run_index, seed, mode, metric, tracking, analytic_sinr[mode], results.config.signal.doppler_hz)
                metric_rows.append(row)
                cn0_rows.extend(_cn0_rows(case, run_index, seed, mode, tracking, results.config.dfe.fs_out_hz))
    return FullChainRecords(
        metrics=tuple(strict_snapshot(row) for row in metric_rows),  # type: ignore[arg-type]
        cn0=tuple(strict_snapshot(row) for row in cn0_rows),  # type: ignore[arg-type]
    )


def _reconstruct_case(
    resolved: ResolvedStudyConfig, case: StudyCase, seed: int,
):
    simulation = copy.deepcopy(resolved.simulation)
    if not isinstance(simulation.array, PhoneArrayConfig) or simulation.phone is None:
        raise ValueError("full-chain validation requires a phone-array configuration")
    simulation.array.active_antennas = tuple(case.active_antennas)
    simulation.diversity.modes = tuple(resolved.study.full_chain_modes)
    simulation.random_seed = seed
    simulation.plot.enable = False
    simulation.plot.show = False
    if case.stage in {"representative_direction", "multi_jammer", "full_chain"}:
        simulation.interferers = _interferers(case.parameters["interferers"])
    elif case.stage == "single_factor":
        _apply_single_factor(simulation, case.parameters["single_factor"])
    elif case.stage == "combined_error":
        _apply_combined_realization(simulation, case.parameters["realization"])
    else:
        raise ValueError(f"unsupported selected case stage: {case.stage}")
    simulation.validate()
    return simulation


def _interferers(value: object) -> tuple[CWInterferer, ...]:
    validate_interferer_snapshots(value)
    built = []
    for raw in value:  # type: ignore[union-attr]
        direction = raw["direction"]
        built.append(CWInterferer(
            direction=Direction(direction["azimuth_deg"], direction["elevation_deg"]),
            enable=raw["enable"], power_dbm=raw["power_dbm"],
            offset_hz=raw["offset_hz"], initial_phase_rad=raw["initial_phase_rad"],
            start_ms=raw["start_ms"], stop_ms=raw["stop_ms"],
        ))
    return tuple(built)


def _apply_single_factor(simulation, raw: object) -> None:
    if not isinstance(raw, Mapping):
        raise ValueError("selected case single_factor is invalid")
    factor = str(raw["factor"])
    value = float(raw["value"])
    corner = raw["corner"]
    axis = raw["axis"]
    if factor == "direction_azimuth":
        assumed = simulation.signal.assumed_direction
        simulation.signal.assumed_direction = Direction(assumed.azimuth_deg + value, assumed.elevation_deg)
    elif factor == "direction_elevation":
        assumed = simulation.signal.assumed_direction
        simulation.signal.assumed_direction = Direction(assumed.azimuth_deg, assumed.elevation_deg + value)
    elif factor == "position":
        base = np.asarray(simulation.phone.assumed_phase_centers_m[str(corner)], dtype=float)
        index = {"x": 0, "y": 1, "z": 2}[str(axis)]
        base[index] += value
        simulation.phone.assumed_phase_centers_m[str(corner)] = tuple(base)
    elif factor == "amplitude":
        simulation.array.estimated_channel_amplitude_error_db[str(corner)] = value
    elif factor == "phase":
        simulation.array.estimated_channel_phase_error_deg[str(corner)] = value
    elif factor == "loading":
        simulation.diversity.diagonal_loading_factor = value
    else:
        raise ValueError("selected case single_factor has unknown factor")


def _apply_combined_realization(simulation, raw: object) -> None:
    if not isinstance(raw, Mapping):
        raise ValueError("selected case combined realization is invalid")
    assumed = simulation.signal.assumed_direction
    simulation.signal.assumed_direction = Direction(
        assumed.azimuth_deg + float(raw["azimuth_error_deg"]),
        assumed.elevation_deg + float(raw["elevation_error_deg"]),
    )
    for corner, error in dict(raw["position_errors_m"]).items():
        base = np.asarray(simulation.phone.assumed_phase_centers_m[str(corner)], dtype=float)
        simulation.phone.assumed_phase_centers_m[str(corner)] = tuple(base + np.asarray(error, dtype=float))
    for corner, value in dict(raw["amplitude_errors_db"]).items():
        simulation.array.estimated_channel_amplitude_error_db[str(corner)] = float(value)
    for corner, value in dict(raw["phase_errors_deg"]).items():
        simulation.array.estimated_channel_phase_error_deg[str(corner)] = float(value)


def _metric_row(case, run_index, seed, mode, metric, tracking, analytic_sinr, target_doppler_hz):
    final_doppler = metric["final_doppler_hz"]
    return {
        "case_id": case.case_id, "stage": case.stage,
        "role": list(case.selection_reasons), "run_index": run_index,
        "seed": seed, "active_antennas": list(case.active_antennas), "mode": mode,
        "acquired": bool(metric["acquired"]),
        "acquisition_metric": float(metric["acquisition_metric"]),
        "tracking": bool(metric["tracking"]),
        "status": str(tracking.get("status", "tracking" if metric["tracking"] else "acquisition_failed")),
        "reason": None if metric["tracking"] else str(tracking.get("status", "acquisition_failed")),
        "final_cn0_db_hz": None if metric["final_cn0_db_hz"] is None else float(metric["final_cn0_db_hz"]),
        "final_doppler_error_hz": None if final_doppler is None else float(final_doppler) - float(target_doppler_hz),
        "target_direction_gain": float(metric["target_direction_gain"]),
        "interferer_direction_gains": [float(value) for value in metric["interferer_direction_gains"]],
        "worst_interferer_gain": float(metric["worst_interferer_gain"]),
        "analytic_output_sinr_db": analytic_sinr,
    }


def _cn0_rows(case, run_index, seed, mode, tracking, fs_out_hz):
    starts = np.asarray(tracking.get("epoch_start_samples", ()), dtype=float)
    cn0 = np.asarray(tracking.get("cn0_db_hz", ()), dtype=float)
    if starts.ndim != 1 or cn0.ndim != 1 or starts.shape != cn0.shape:
        raise ValueError("tracking C/N0 and epoch times must be equal one-dimensional arrays")
    return [
        {
            "case_id": case.case_id, "stage": case.stage,
            "role": list(case.selection_reasons), "run_index": run_index,
            "seed": seed, "active_antennas": list(case.active_antennas),
            "mode": mode, "time_ms": float(start / fs_out_hz * 1000.0),
            "cn0_db_hz": float(value),
        }
        for start, value in zip(starts, cn0, strict=True)
        if np.isfinite(start) and np.isfinite(value)
    ]
