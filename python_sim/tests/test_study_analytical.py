import copy
import json

import numpy as np
import pytest

from antenna_diversity.experiments.spatial_scan import evaluate_configuration_spatially
from antenna_diversity.studies.analytical import run_baseline
from antenna_diversity.studies.config import resolve_study_config


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
