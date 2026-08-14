import json
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from antenna_diversity.studies.models import SelectionResult, StudyCase
from antenna_diversity.studies.full_chain import run_selected_full_chain
from tests.study_helpers import resolved_for_test


with warnings.catch_warnings():
    warnings.simplefilter("ignore", pytest.PytestUnknownMarkWarning)
    integration = pytest.mark.integration


def one_selected_case() -> SelectionResult:
    case = StudyCase(
        case_id="selected-001",
        stage="full_chain",
        active_antennas=("TL", "BR"),
        mode="mvdr",
        seed=None,
        parameters={
            "interferers": [{
                "direction": {"azimuth_deg": 120.0, "elevation_deg": 10.0},
                "enable": True, "power_dbm": -90.0, "offset_hz": 250e3,
                "initial_phase_rad": 0.8, "start_ms": 0.0, "stop_ms": None,
            }],
        },
        selection_reasons=("best",),
    )
    return SelectionResult((case,), {})


def controlled_results(cfg):
    modes = cfg.diversity.modes
    branches = {}
    metrics = []
    for mode in modes:
        acquired = mode != "single"
        tracking = acquired
        track = {
            "success": tracking,
            "status": "tracking" if tracking else "acquisition_failed",
            "epoch_start_samples": np.array([0, 4092]) if tracking else np.array([]),
            "cn0_db_hz": np.array([36.0, 37.0]) if tracking else np.array([]),
        }
        branches[mode] = SimpleNamespace(tracking=track)
        metrics.append({
            "mode": mode, "acquired": acquired, "acquisition_metric": 3.0,
            "tracking": tracking, "final_doppler_hz": 1500.0 if tracking else None,
            "final_cn0_db_hz": 37.0 if tracking else None,
            "target_direction_gain": 1.0,
            "interferer_direction_gains": [0.01],
            "worst_interferer_gain": 0.01,
        })
    return SimpleNamespace(config=cfg, branches=branches, metrics=metrics)


def controlled_results_with_nonfinite_epoch(cfg):
    results = controlled_results(cfg)
    for branch in results.branches.values():
        if branch.tracking["success"]:
            branch.tracking["epoch_start_samples"] = np.array([np.nan, 4092.0])
    return results


def test_full_chain_records_keep_failures_and_never_store_sample_arrays(monkeypatch):
    monkeypatch.setattr(
        "antenna_diversity.studies.full_chain.run_end_to_end",
        controlled_results,
    )
    records = run_selected_full_chain(
        resolved_for_test(combinations=(("TL", "BR"),), full_chain_run_count=1),
        one_selected_case(),
    )

    failed = next(row for row in records.metrics if row["mode"] == "single")
    assert failed["acquired"] is False
    assert failed["tracking"] is False
    assert failed["final_cn0_db_hz"] is None
    assert failed["status"] == "acquisition_failed"
    assert failed["role"] == ["best"]
    assert failed["reason"] == "acquisition_failed"
    first_cn0 = records.cn0[0]
    assert first_cn0["stage"] == "full_chain"
    assert first_cn0["role"] == ["best"]
    assert first_cn0["active_antennas"] == ["TL", "BR"]
    assert "scene_input_v" not in json.dumps(records.to_dict())
    json.dumps(records.to_dict(), allow_nan=False)


def test_full_chain_uses_explicit_analytic_sinr_name(monkeypatch):
    monkeypatch.setattr(
        "antenna_diversity.studies.full_chain.run_end_to_end",
        controlled_results,
    )
    records = run_selected_full_chain(
        resolved_for_test(combinations=(("TL", "BR"),), full_chain_run_count=1),
        one_selected_case(),
    )

    assert all("analytic_output_sinr_db" in row for row in records.metrics)
    assert all("output_sinr_db" not in row for row in records.metrics)


@pytest.mark.parametrize("replacement", [
    {},
    {**one_selected_case().cases[0].parameters["interferers"][0], "enable": "false"},
    {**one_selected_case().cases[0].parameters["interferers"][0], "power_dbm": float("nan")},
    {**one_selected_case().cases[0].parameters["interferers"][0], "unexpected": 1},
])
def test_full_chain_rejects_interferers_before_coercing_field_types(monkeypatch, replacement):
    selection = one_selected_case()
    case = selection.cases[0]
    invalid = SelectionResult((StudyCase(
        case.case_id, case.stage, case.active_antennas, case.mode, case.seed,
        {"interferers": [replacement]}, case.selection_reasons,
    ),), {})
    monkeypatch.setattr(
        "antenna_diversity.studies.full_chain.run_end_to_end",
        lambda _cfg: pytest.fail("invalid input must fail before the pipeline"),
    )

    with pytest.raises(ValueError, match="interferer"):
        run_selected_full_chain(
            resolved_for_test(combinations=(("TL", "BR"),), full_chain_run_count=1),
            invalid,
        )


def test_full_chain_skips_epochs_with_nonfinite_time(monkeypatch):
    monkeypatch.setattr(
        "antenna_diversity.studies.full_chain.run_end_to_end",
        controlled_results_with_nonfinite_epoch,
    )

    records = run_selected_full_chain(
        resolved_for_test(combinations=(("TL", "BR"),), full_chain_run_count=1),
        one_selected_case(),
    )

    assert records.cn0
    assert all(np.isfinite(row["time_ms"]) for row in records.cn0)
    json.dumps(records.to_dict(), allow_nan=False)


@integration
def test_full_chain_runs_one_default_duration_case_without_raw_arrays():
    records = run_selected_full_chain(
        resolved_for_test(
            combinations=(("TL", "BR"),),
            full_chain_modes=("single", "egc", "mvdr"),
            full_chain_run_count=1,
        ),
        one_selected_case(),
    )

    assert len(records.metrics) == 3
    assert {row["mode"] for row in records.metrics} == {"single", "egc", "mvdr"}
    assert all(np.isfinite(row["cn0_db_hz"]) for row in records.cn0)
    assert "ndarray" not in repr(records.to_dict())
    json.dumps(records.to_dict(), allow_nan=False)
