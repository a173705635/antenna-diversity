import copy
import json
from dataclasses import replace

import numpy as np
import pytest

from antenna_diversity.config import CWInterferer, Direction
from antenna_diversity.experiments.spatial_scan import evaluate_configuration_spatially
from antenna_diversity.studies.analytical import (
    run_baseline,
    run_combined_error_studies,
    run_multi_jammer_studies,
    run_single_factor_studies,
)
from antenna_diversity.studies.config import resolve_study_config
from antenna_diversity.studies.models import InterfererTemplate, ResolvedStudyConfig


def test_baseline_emits_single_corner_references_and_requested_subarrays():
    resolved = resolve_study_config(
        preset="quick",
        cli_overrides={
            "combinations": "TL+BR,TL+TR+BR",
            "modes": ["egc", "mvdr"],
            "single_antennas": ["TL", "BR"],
        },
    )
    before = copy.deepcopy(resolved.to_dict())

    result = run_baseline(resolved)

    labels = {row["display_label"] for row in result.tables["reference"]}
    assert {"SINGLE(TL)", "SINGLE(BR)", "EGC(TL+BR)", "MVDR(TL+BR)"} <= labels
    assert {tuple(row["active_antennas"]) for row in result.tables["direction"]} == {
        ("TL", "BR"), ("TL", "TR", "BR"),
    }
    assert resolved.to_dict() == before
    json.dumps(result.to_dict(), allow_nan=False)


def test_baseline_direction_rows_have_exact_quick_grid_size():
    resolved = resolve_study_config(
        preset="quick", cli_overrides={"combinations": "TL+BR", "modes": ["mvdr"]}
    )

    result = run_baseline(resolved)

    assert len(result.tables["direction"]) == 12 * 7
    assert all(np.isfinite(row["output_sinr_db"]) for row in result.tables["direction"])
    assert all(np.isfinite(row["interferer_gain"]) for row in result.tables["direction"])


def test_baseline_orders_direction_axes_cases_and_never_runs_receiver(monkeypatch):
    import antenna_diversity.pipeline as pipeline

    monkeypatch.setattr(
        pipeline,
        "run_end_to_end",
        lambda *_args, **_kwargs: pytest.fail("receiver pipeline must not run"),
    )
    resolved = resolve_study_config(
        preset="quick",
        cli_overrides={
            "combinations": "TL+BR,TL+TR+BR",
            "modes": ["egc", "mvdr"],
            "azimuths_deg": [5.0, 15.0],
            "elevations_deg": [20.0, 40.0],
        },
    )

    result = run_baseline(resolved)

    direction = result.tables["direction"]
    assert [
        (row["active_antennas"], row["mode"])
        for row in direction[::4]
    ] == [
        (["TL", "BR"], "egc"),
        (["TL", "BR"], "mvdr"),
        (["TL", "TR", "BR"], "egc"),
        (["TL", "TR", "BR"], "mvdr"),
    ]
    assert [
        (row["elevation_index"], row["azimuth_index"], row["elevation_deg"], row["azimuth_deg"])
        for row in direction[:4]
    ] == [
        (0, 0, 20.0, 5.0),
        (0, 1, 20.0, 15.0),
        (1, 0, 40.0, 5.0),
        (1, 1, 40.0, 15.0),
    ]
    all_rows = tuple(row for table in result.tables.values() for row in table)
    assert len({row["case_id"] for row in all_rows}) == len(all_rows)
    assert all(row["num_antennas"] == 1 for row in result.tables["reference"][:4])
    assert all(len(row["active_antennas"]) == 1 for row in result.tables["reference"][:4])


def test_single_result_ignores_its_zero_weight_companion_channel():
    resolved = resolve_study_config(preset="quick")
    results = []
    for companion in ("TR", "BL", "BR"):
        simulation = resolved.simulation
        simulation.array.active_antennas = ("TL", companion)
        results.append(evaluate_configuration_spatially(simulation, "single").metrics)

    for metrics in results[1:]:
        assert metrics["output_sinr_db"] == pytest.approx(results[0]["output_sinr_db"])
        assert metrics["output_interference_power_w"] == pytest.approx(
            results[0]["output_interference_power_w"]
        )


def _robustness_config():
    return resolve_study_config(
        preset="quick",
        cli_overrides={
            "combinations": "TL+BR",
            "direction_errors_deg": [-1.0, 0.0, 1.0],
            "position_errors_m": [-0.001, 0.0, 0.001],
            "amplitude_errors_db": [-0.5, 0.0, 0.5],
            "phase_errors_deg": [-5.0, 0.0, 5.0],
            "loading_factors": [1e-3, 1e-2, 1e-1],
            "combined_error_samples_per_level": 2,
        },
    )


def test_single_factor_changes_assumed_values_only_and_keeps_zero_baseline():
    resolved = _robustness_config()
    before = copy.deepcopy(resolved.to_dict())

    records = run_single_factor_studies(resolved).tables["single_factor"]

    assert {row["factor"] for row in records} == {
        "direction_azimuth", "direction_elevation", "position",
        "amplitude", "phase", "loading",
    }
    position = [row for row in records if row["factor"] == "position"]
    assert {(row["corner"], row["axis"]) for row in position} == {
        (corner, axis) for corner in ("TL", "BR") for axis in ("x", "y", "z")
    }
    assert all(row["true_geometry_unchanged"] for row in position)
    assert all(row["true_state_unchanged"] for row in records)
    assert all(row["mvdr_case_id"] == row["egc_case_id"] for row in records)
    assert all(
        row["mvdr_minus_egc_sinr_db"] == pytest.approx(
            row["mvdr_output_sinr_db"] - row["egc_output_sinr_db"]
        )
        for row in records
    )
    zero_rows = [row for row in records if row["factor"] != "loading" and row["value"] == 0.0]
    assert zero_rows
    assert all(row["zero_matches_baseline"] for row in zero_rows)
    assert all(row["default_loading"] for row in records if row["factor"] == "loading" and row["value"] == 1e-2)
    assert resolved.to_dict() == before


def test_single_factor_loading_severity_is_normalized_log_distance_from_default():
    resolved = _robustness_config()

    rows = run_single_factor_studies(resolved).tables["single_factor"]
    severity = {
        row["value"]: row["absolute_normalized_severity"]
        for row in rows if row["factor"] == "loading"
    }

    assert severity == {1e-3: pytest.approx(1.0), 1e-2: pytest.approx(0.0), 1e-1: pytest.approx(1.0)}


def test_combined_error_studies_copy_saved_realizations_and_are_deterministic():
    resolved = _robustness_config()

    first = run_combined_error_studies(resolved)
    second = run_combined_error_studies(resolved)
    rows = first.tables["combined_error"]

    assert first.to_dict() == second.to_dict()
    assert len(rows) == 3 * resolved.study.combined_error_samples_per_level * len(resolved.study.combinations)
    assert all(row["mvdr_case_id"] == row["egc_case_id"] for row in rows)
    assert all(row["realization"]["child_seed"] == row["child_seed"] for row in rows)
    assert all(row["true_state_unchanged"] for row in rows)
    assert all(set(row["realization"]["position_errors_m"]) == {"TL", "BR"} for row in rows)


def test_multi_jammer_studies_preserve_enabled_pressure_cases():
    resolved = _robustness_config()

    rows = run_multi_jammer_studies(resolved).tables["multi_jammer"]

    assert rows
    assert {row["mode"] for row in rows} == {"egc", "mvdr"}
    assert all(row["enabled_interferer_count"] == len(row["enabled_interferers"]) for row in rows)
    stress = [row for row in rows if row["template"] == "three_jammer_stress"]
    assert stress and all(row["enabled_interferer_count"] == 3 for row in stress)
    assert all(np.isfinite(row["worst_interferer_gain"]) for row in rows)


def test_multi_jammer_studies_keep_duplicate_named_template_definitions_distinct():
    base = _robustness_config()
    templates = (
        InterfererTemplate("duplicate", (CWInterferer(Direction(25.0, 10.0), power_dbm=-81.0),)),
        InterfererTemplate("duplicate", (CWInterferer(Direction(260.0, 35.0), power_dbm=-103.0),)),
    )
    resolved = ResolvedStudyConfig.from_parts(
        replace(base.study, multi_jammer_templates=templates), base.simulation
    )

    rows = run_multi_jammer_studies(resolved).tables["multi_jammer"]
    definitions = {
        (row["interferers"][0]["direction"]["azimuth_deg"], row["interferers"][0]["power_dbm"])
        for row in rows
    }

    assert definitions == {(25.0, -81.0), (260.0, -103.0)}
