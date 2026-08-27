import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from antenna_diversity.array.antenna_config import AntennaStateConfig, PhoneAntennaConfig
from antenna_diversity.array.effects import AttenuationRegion, CouplingPath, ElementOrientation
from antenna_diversity.array.patterns import ParametricPatternParameters
from antenna_diversity.array.polarization import WavePolarization
from antenna_diversity.studies.full_chain import (
    _interferers,
    _reconstruct_case,
    run_selected_full_chain,
)
from antenna_diversity.studies.models import (
    ResolvedStudyConfig,
    SelectionResult,
    StudyCase,
    strict_snapshot,
)
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
                "polarization": {"kind": "rhcp", "tilt_deg": 0.0,
                                 "az_real": 1.0, "az_imag": 0.0,
                                 "el_real": 0.0, "el_imag": 0.0},
            }],
        },
        selection_reasons=("best",),
    )
    return SelectionResult((case,), {})


def test_interferer_snapshot_reconstruction_preserves_polarization():
    snapshot = [{
        "direction": {"azimuth_deg": 120.0, "elevation_deg": 10.0},
        "enable": True, "power_dbm": -90.0, "offset_hz": 250e3,
        "initial_phase_rad": 0.8, "start_ms": 0.0, "stop_ms": None,
        "polarization": {"kind": "linear", "tilt_deg": 35.0,
                         "az_real": 1.0, "az_imag": 0.0,
                         "el_real": 0.0, "el_imag": 0.0},
    }]

    rebuilt = _interferers(snapshot)

    assert rebuilt[0].polarization.kind == "linear"
    assert rebuilt[0].polarization.tilt_deg == 35.0


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


def test_blind_full_chain_row_has_no_analytic_sinr(monkeypatch):
    analytic_calls = []

    def controlled_analytic(_simulation, mode):
        analytic_calls.append(mode)
        return SimpleNamespace(metrics={"output_sinr_db": 12.5})

    monkeypatch.setattr(
        "antenna_diversity.studies.full_chain.evaluate_configuration_spatially",
        controlled_analytic,
    )
    monkeypatch.setattr(
        "antenna_diversity.studies.full_chain.run_end_to_end",
        controlled_results,
    )
    records = run_selected_full_chain(
        resolved_for_test(
            combinations=(("TL", "BR"),),
            full_chain_modes=("mvdr", "blind_null"),
            full_chain_run_count=1,
        ),
        one_selected_case(),
    )

    assert analytic_calls == ["mvdr"]
    rows = {row["mode"]: row for row in records.metrics}
    assert isinstance(rows["mvdr"]["analytic_output_sinr_db"], float)
    assert rows["blind_null"]["analytic_output_sinr_db"] is None


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


def test_selected_case_keeps_complete_true_model_and_offsets_only_assumed_fields(
    tmp_path: Path,
):
    csv_path = tmp_path / "patterns.csv"
    csv_path.write_text("measured fixture\n", encoding="utf-8")
    csv_digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    base = resolved_for_test(combinations=(("TL", "BR"),), full_chain_run_count=1)
    simulation = base.simulation
    orientations = {
        corner: ElementOrientation(yaw_deg=float(index * 10))
        for index, corner in enumerate(("TL", "TR", "BL", "BR"))
    }
    true_state = AntennaStateConfig(
        pattern_kind="measured",
        measured_pattern_csv=csv_path,
        measured_pattern_sha256=csv_digest,
        orientations=orientations,
        attenuation_regions=(
            AttenuationRegion(("TL",), 120.0, 20.0, 15.0, 6.0, 2.0),
        ),
        coupling_paths=(CouplingPath("TL", "BR", -18.0, 25.0),),
    )
    assumed_state = AntennaStateConfig(
        pattern_kind="parametric",
        parametric_patterns={
            corner: ParametricPatternParameters(
                peak_gain_db=float(index),
                nominal_polarization=WavePolarization("linear", tilt_deg=20.0),
            )
            for index, corner in enumerate(("TL", "TR", "BL", "BR"))
        },
        orientations=orientations.copy(),
        attenuation_regions=(
            AttenuationRegion(("BR",), 220.0, 30.0, 10.0, 2.0),
        ),
        coupling_paths=(CouplingPath("BR", "TL", -24.0, -15.0),),
    )
    simulation.array.antenna_model = PhoneAntennaConfig(
        "selected measured fixture", "measured", true_state, assumed_state
    )
    simulation.signal.true_polarization = WavePolarization("rhcp")
    simulation.signal.assumed_polarization = WavePolarization("lhcp")
    simulation.array.channel_amplitude_error_db["TL"] = 7.0
    simulation.array.channel_phase_error_deg["TL"] = 70.0
    simulation.array.estimated_channel_amplitude_error_db["TL"] = 1.5
    simulation.array.estimated_channel_phase_error_deg["TL"] = 10.0
    simulation.phone.assumed_phase_centers_m["TL"] = (0.1, 0.2, 0.3)
    resolved = ResolvedStudyConfig.from_parts(base.study, simulation)
    model_before = strict_snapshot(resolved.simulation.array.antenna_model)
    true_positions_before = dict(resolved.simulation.phone.phase_centers_m)
    case = StudyCase(
        case_id="combined-rich-model",
        stage="combined_error",
        active_antennas=("TL", "BR"),
        mode="mvdr",
        seed=None,
        parameters={
            "realization": {
                "azimuth_error_deg": 2.0,
                "elevation_error_deg": -3.0,
                "position_errors_m": {"TL": [0.01, -0.02, 0.03]},
                "amplitude_errors_db": {"TL": 0.5},
                "phase_errors_deg": {"TL": -4.0},
            }
        },
    )

    rebuilt = _reconstruct_case(resolved, case, seed=123)

    assert strict_snapshot(rebuilt.array.antenna_model) == model_before
    assert rebuilt.array.antenna_model.true_state.measured_pattern_sha256 == csv_digest
    assert rebuilt.signal.true_direction == resolved.simulation.signal.true_direction
    assert rebuilt.signal.assumed_direction.azimuth_deg == 32.0
    assert rebuilt.signal.assumed_direction.elevation_deg == 57.0
    assert rebuilt.phone.phase_centers_m == true_positions_before
    assert rebuilt.phone.assumed_phase_centers_m["TL"] == pytest.approx(
        (0.11, 0.18, 0.33)
    )
    assert rebuilt.array.channel_amplitude_error_db["TL"] == 7.0
    assert rebuilt.array.channel_phase_error_deg["TL"] == 70.0
    assert rebuilt.array.estimated_channel_amplitude_error_db["TL"] == 2.0
    assert rebuilt.array.estimated_channel_phase_error_deg["TL"] == 6.0
    assert rebuilt.signal.true_polarization.kind == "rhcp"
    assert rebuilt.signal.assumed_polarization.kind == "lhcp"
    rebuilt.array.antenna_model.model_label = "mutated rebuilt copy"
    assert resolved.simulation.array.antenna_model.model_label == "selected measured fixture"


@integration
def test_full_chain_runs_one_default_duration_case_without_raw_arrays():
    records = run_selected_full_chain(
        resolved_for_test(
            combinations=(("TL", "BR"),),
            full_chain_modes=("single", "egc", "mvdr", "blind_null"),
            full_chain_run_count=1,
        ),
        one_selected_case(),
    )

    assert len(records.metrics) == 4
    assert {row["mode"] for row in records.metrics} == {
        "single", "egc", "mvdr", "blind_null",
    }
    assert all(np.isfinite(row["cn0_db_hz"]) for row in records.cn0)
    assert "ndarray" not in repr(records.to_dict())
    json.dumps(records.to_dict(), allow_nan=False)
