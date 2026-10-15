"""Behavioral checks for the reusable anti-interference validation runner."""

from copy import deepcopy

import numpy as np
import pytest

from core.config import Direction
from core.gnss import generate_scenario
from core.presets import default_config
from experiments.interference.validation_support import (
    ValidationCase, ValidationSettings, apply_dfe_phase, run_validation,
    verify_validation,
)
from experiments.interference import validation_config


def test_dfe_phase_rotates_every_sample_once_and_inverse_restores():
    samples = np.array([[1 + 2j, 3 - 4j], [2 - 1j, -3 + 0j]])
    rotated = apply_dfe_phase(samples, ("BR", "TL"), {"TL": 90})
    np.testing.assert_array_equal(rotated[0], samples[0])
    np.testing.assert_allclose(rotated[1], 1j * samples[1], atol=1e-15)
    np.testing.assert_allclose(
        apply_dfe_phase(rotated, ("BR", "TL"), {"TL": -90}), samples,
    )
    np.testing.assert_array_equal(apply_dfe_phase(samples, ("BR", "TL"), {}), samples)
    half_turn = apply_dfe_phase(samples, ("BR", "TL"), {"TL": 180})
    np.testing.assert_allclose(half_turn[1], -samples[1], atol=1e-15)


def test_selected_case_includes_reference_and_preserves_configs(tmp_path):
    base = default_config()
    base.phone.active_antennas = ("BR", "TL")
    base.interferers = []
    changed = deepcopy(base)
    changed.signal.true_direction = Direction(40, 60)
    cases = [
        ValidationCase("reference", base, scene_label="noise"),
        ValidationCase("changed", changed, scene_label="noise", reference_case="reference"),
        ValidationCase("unused", deepcopy(base), scene_label="noise"),
    ]
    settings = ValidationSettings(
        seeds=(42,), duration_ms=12, power_trim_ms=1,
        selected_cases=("changed",), output_dir=tmp_path, show_plots=False,
    )
    result = run_validation(settings, cases)
    assert {case["name"] for case in result["cases"]} == {"reference", "changed"}
    assert base.signal.true_direction.azimuth_deg == 30
    assert changed.signal.true_direction.azimuth_deg == 40
    assert all(row["antennas"] == "BR,TL" for row in result["rows"])
    assert result["rows"]


def test_verification_rejects_one_sided_cn0_and_tampered_power():
    reference = {
        "case": "base", "seed": 42, "mode": "single", "target_power": 4.0,
        "thermal_power": 1.0, "interferer_powers": [], "actual_power": 5.0,
        "linear_component_cross_power": 0.0,
        "joint_nonlinear_quantization_residual_power": 0.0,
        "combined_direct_max_abs_error": 0.0,
        "quadratic_power_relative_error": 0.0, "acquired": True,
        "tracking_success": True, "tracking_status": "ok",
        "tracking_cn0_late_median_db_hz": 40.0,
        "tracking_cn0_late_finite_count": 5,
        "acquisition_metric": 3.0, "acquired_code_error_chips": 0.0,
        "acquired_doppler_error_hz": 0.0,
        "weight_real_imag": [[1.0, 0.0]],
        "phase_reference_available": True,
        "dfe_input_relative_error": 0.0,
        "combined_relative_error": 0.0,
        "weight_relative_error": 0.0,
    }
    restored = dict(reference, case="cal", tracking_cn0_late_median_db_hz=None,
                    tracking_cn0_late_finite_count=0)
    report = verify_validation(
        [reference, restored],
        {"base": ValidationCase("base", default_config()),
         "cal": ValidationCase("cal", default_config(), reference_case="base", checks=("phase_pair",))},
    )
    assert not report["passed"]
    assert not report["checks"]["phase_pair:cal:42:single"]["passed"]


def test_default_conditions_reproduce_old_phase_grid():
    settings = validation_config.build_settings()
    cases = validation_config.build_cases()
    selected = [case for case in cases if case.name in settings.selected_cases]
    assert settings.seeds == (42, 43, 44)
    assert settings.duration_ms == 100
    assert settings.power_trim_ms == 10
    assert len(selected) == 36
    assert {case.scene_label for case in selected} == {"noise_only", "single_cw"}
    assert {case.scan_value for case in selected} == {0, 30, 60, 90, 120, 180}
    assert {case.series_label for case in selected} == {"baseline", "uncalibrated", "calibrated"}
    assert all(case.config.phone.active_antennas == ("TL", "BR") for case in selected)
    assert all(case.config.diversity.modes == ("single", "egc", "mvdr") for case in selected)


def test_short_saved_run_is_isolated_and_has_readable_plots(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from experiments.interference import validation

    plt.switch_backend("Agg")
    cfg = default_config()
    cfg.phone.active_antennas = ("TL", "BR")
    cfg.interferers = []
    cfg.diversity.modes = ("single", "egc")
    settings = ValidationSettings(
        seeds=(42,), duration_ms=12, power_trim_ms=1,
        output_dir=tmp_path / "validation", show_plots=False,
    )
    sibling = tmp_path / "other_experiment"
    sibling.mkdir()
    (sibling / "sentinel.txt").write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(validation, "build_settings", lambda: settings)
    monkeypatch.setattr(validation, "build_cases", lambda: [
        ValidationCase("control", cfg, scene_label="noise", scan_label="target azimuth",
                       scan_value=30, scan_unit="deg", checks=("ideal_noise",)),
    ])
    summary = validation.main()
    assert summary["verification"]["passed"]
    assert (sibling / "sentinel.txt").read_text(encoding="utf-8") == "preserve"
    assert all((settings.output_dir / name).is_file() for name in (
        "manifest.json", "summary.json", "summary.csv", "verification.json",
    ))
    assert len(list(settings.output_dir.glob("*.png"))) == 3


def test_short_phase_run_matches_legacy_scores_and_receiver(tmp_path, monkeypatch):
    from experiments.interference import channel_phase_validation as legacy

    monkeypatch.setattr(validation_config, "PHASE_SCAN_DEG", (0, 90, 180))
    monkeypatch.setattr(validation_config, "SEEDS", (42,))
    monkeypatch.setattr(validation_config, "DURATION_MS", 20)
    monkeypatch.setattr(validation_config, "POWER_TRIM_MS", 2)
    old_rows, old_verification = legacy.run_sweep(
        seeds=(42,), duration_ms=20, phases=(0, 90, 180), trim_ms=2,
    )
    new_result = run_validation(validation_config.build_settings(),
                                validation_config.build_cases())
    assert old_verification["passed"]
    assert new_result["verification"]["passed"]
    old_by_key = {(row["scene"], row["case"], row["mode"], row["phase_error_deg"]): row
                  for row in old_rows}
    assert len(new_result["rows"]) == len(old_rows) == 54
    for row in new_result["rows"]:
        old = old_by_key[row["scene"], row["series"], row["mode"], row["scan_value"]]
        for name in ("target_power", "thermal_power", "actual_power",
                     "acquisition_metric", "linear_in_band_sinr_db",
                     "tracking_cn0_late_median_db_hz"):
            if old[name] is None:
                assert row[name] is None
            else:
                assert row[name] == pytest.approx(old[name], rel=1e-12, abs=1e-12)
        assert row["acquired"] == old["acquired"]
        assert row["tracking_status"] == old["tracking_status"]
        np.testing.assert_allclose(row["weight_real_imag"], old["weight_real_imag"],
                                   rtol=1e-12, atol=1e-12)
        if row["scene"] == "single_cw":
            assert row["interferer_powers"][0]["power"] == pytest.approx(
                old["cw_power"], rel=1e-12,
            )


def test_saved_phase_pairs_include_csv_columns_and_failed_verification(tmp_path, monkeypatch):
    import csv
    import json
    import matplotlib.pyplot as plt
    from experiments.interference import validation

    plt.switch_backend("Agg")
    monkeypatch.setattr(validation_config, "SEEDS", (42,))
    monkeypatch.setattr(validation_config, "DURATION_MS", 12)
    monkeypatch.setattr(validation_config, "POWER_TRIM_MS", 1)
    monkeypatch.setattr(validation_config, "SELECTED_CASES", ("noise_only/phase_90/calibrated",))
    monkeypatch.setattr(validation_config, "SHOW_PLOTS", False)
    monkeypatch.setattr(validation_config, "OUTPUT_DIR", tmp_path / "phase")
    result = validation.main()
    with (tmp_path / "phase" / "summary.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    assert rows[0]["dfe_input_relative_error"] == ""
    assert float(rows[-1]["dfe_input_relative_error"]) < 1e-12
    manifest = json.loads((tmp_path / "phase" / "manifest.json").read_text(encoding="utf-8"))
    actual = manifest["actual_receiver_configs"]["noise_only/phase_90/calibrated"]["42"]
    assert actual["signal"]["duration_ms"] == 12
    assert actual["random_seed"] == 42
    stale = tmp_path / "phase" / "old_scan.png"
    stale.write_bytes(b"old generated plot")
    result["verification"] = {"passed": False, "checks": {"forced": {"status": "failed"}}}
    monkeypatch.setattr(validation, "run_validation", lambda *_: result)
    with pytest.raises(AssertionError, match="verification failed"):
        validation.main()
    assert not stale.exists()
    assert not json.loads((tmp_path / "phase" / "verification.json").read_text())["passed"]


def test_true_direction_and_element_gain_change_physical_voltage_not_assumed_steering():
    base = default_config()
    base.phone.active_antennas = ("BR", "TL")
    base.interferers = []
    base.signal.duration_ms = 2
    changed = deepcopy(base)
    changed.signal.true_direction = Direction(45, 60)
    _, base_truth = generate_scenario(base)
    _, changed_truth = generate_scenario(changed)
    assert not np.allclose(changed_truth["desired_component"],
                           base_truth["desired_component"], rtol=1e-3, atol=0)
    np.testing.assert_array_equal(changed_truth["desired_steering_assumed"],
                                  base_truth["desired_steering_assumed"])
    gain = deepcopy(base)
    gain.phone.antenna_model.true_state.amplitude_error_db["BR"] = 6.020599913279624
    _, gain_truth = generate_scenario(gain)
    voltage_ratio = np.linalg.norm(gain_truth["desired_component"][0]) / np.linalg.norm(
        base_truth["desired_component"][0]
    )
    assert voltage_ratio == pytest.approx(2.0, rel=1e-12)
    assert voltage_ratio ** 2 == pytest.approx(4.0, rel=1e-12)
    np.testing.assert_array_equal(gain_truth["desired_steering_assumed"],
                                  base_truth["desired_steering_assumed"])


@pytest.mark.parametrize("source_indexes", [(), (0,), (1,), (0, 1)])
def test_independent_budget_lists_each_enabled_interferer(source_indexes, tmp_path):
    cfg = default_config()
    cfg.phone.active_antennas = ("BR", "BL", "TR", "TL")
    cfg.diversity.modes = ("mvdr",)
    cfg.interferers = [cfg.interferers[index] for index in source_indexes]
    settings = ValidationSettings(
        seeds=(42,), duration_ms=10, power_trim_ms=1,
        output_dir=tmp_path, show_plots=False,
    )
    result = run_validation(settings, [ValidationCase("one", cfg)])
    assert result["verification"]["passed"]
    assert len(result["rows"]) == 1
    row = result["rows"][0]
    assert row["single_reference_antenna"] == "BR"
    assert len(row["interferer_powers"]) == len(source_indexes)
    assert [item["source"].split("_", 2)[-1] for item in row["interferer_powers"]] == [
        type(cfg.interferers[index]).__name__ for index in range(len(source_indexes))
    ]
    assert np.isfinite(row["thermal_snr_gain_db"])


def test_ideal_noise_check_reports_not_applicable_for_real_gain():
    cfg = default_config()
    cfg.interferers = []
    cfg.phone.antenna_model.true_state.amplitude_error_db["BR"] = 6.020599913279624
    row = {"case": "gain", "seed": 42, "mode": "egc",
           "combined_direct_max_abs_error": 0.0,
           "quadratic_power_relative_error": 0.0,
           "thermal_snr_gain_db": 0.0}
    report = verify_validation([row], {"gain": ValidationCase(
        "gain", cfg, checks=("ideal_noise",),
    )})
    check = report["checks"]["ideal_noise:gain:42:egc"]
    assert check["status"] == "not_applicable"
    assert "manifold" in check["reason"]


def test_ideal_noise_failure_cannot_be_masked_by_numpy_boolean():
    cfg = default_config()
    cfg.interferers = []
    row = {"case": "ideal", "seed": 42, "mode": "egc",
           "combined_direct_max_abs_error": 0.0,
           "quadratic_power_relative_error": 0.0,
           "thermal_snr_gain_db": -20.0}
    report = verify_validation([row], {"ideal": ValidationCase(
        "ideal", cfg, checks=("ideal_noise",),
    )})
    assert not report["passed"]
    assert report["checks"]["ideal_noise:ideal:42:egc"]["status"] == "failed"


def test_assumption_changes_reuse_samples_but_real_gain_regenerates_frontend(monkeypatch):
    from experiments.interference import validation_support as support

    cfg = default_config()
    cfg.phone.active_antennas = ("TL", "BR")
    cfg.diversity.modes = ("single",)
    cfg.interferers = []
    assumption = deepcopy(cfg)
    assumption.signal.assumed_direction = Direction(60, 60)
    gain = deepcopy(cfg)
    gain.phone.antenna_model.true_state.amplitude_error_db["BR"] = 6.020599913279624
    calls, inputs, steering = [], [], []
    frontend, receiver = support.run_frontend, support.run_receiver

    def capture_frontend(config):
        calls.append(config)
        return frontend(config)

    def capture_receiver(channels, truth, config, fs_hz):
        inputs.append(channels.copy())
        steering.append(truth["desired_steering_assumed"].copy())
        return receiver(channels, truth, config, fs_hz)

    monkeypatch.setattr(support, "run_frontend", capture_frontend)
    monkeypatch.setattr(support, "run_receiver", capture_receiver)
    result = run_validation(ValidationSettings(seeds=(42,), duration_ms=10, power_trim_ms=1), [
        ValidationCase("base", cfg), ValidationCase("assumed", assumption),
        ValidationCase("gain", gain),
    ])
    assert result["verification"]["passed"]
    assert len(calls) == 2
    np.testing.assert_array_equal(inputs[0], inputs[1])
    assert not np.array_equal(inputs[0], inputs[2])
    assert not np.allclose(steering[0], steering[1])
    np.testing.assert_array_equal(steering[0], steering[2])


def test_phase_reference_runs_before_dependent_even_when_list_is_reversed(tmp_path):
    cfg = default_config()
    cfg.phone.active_antennas = ("TL", "BR")
    cfg.diversity.modes = ("single",)
    cfg.interferers = []
    cases = [
        ValidationCase("cal", deepcopy(cfg), dfe_phase_deg={"BR": 90},
                       dfe_estimated_phase_deg={"BR": 90},
                       compensate_dfe_phase=True, reference_case="base",
                       checks=("phase_pair",)),
        ValidationCase("base", deepcopy(cfg)),
    ]
    settings = ValidationSettings((42,), 12, 1, ("cal",), tmp_path, False)
    result = run_validation(settings, cases)
    assert result["verification"]["passed"]
    assert all(row["phase_reference_available"] for row in result["rows"]
               if row["case"] == "cal")


def test_plots_keep_linear_180_degree_cancellation_and_separate_factors(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from experiments.interference.validation_support import plot_validation

    plt.switch_backend("Agg")
    captured = []
    original = plt.subplots

    def capture(*args, **kwargs):
        figure, axes = original(*args, **kwargs)
        captured.append((figure, axes))
        return figure, axes

    monkeypatch.setattr(plt, "subplots", capture)
    rows = []
    for label, unit, points in (("DFE channel phase", "deg", (0, 180)),
                                ("target true azimuth", "deg", (30, 40))):
        for point in points:
            rows.append({
                "scene": "noise", "mode": "egc", "series": "uncalibrated",
                "scan_label": label, "scan_unit": unit, "scan_value": point,
                "thermal_snr_ratio": 0.0 if point == 180 else 2.0,
                "target_power_retention": 0.0 if point == 180 else 1.0,
                "linear_in_band_sinr_db": -np.inf if point == 180 else 10.0,
                "interferer_suppression_db": {}, "acquisition_metric": 3.0,
                "acquisition_threshold": 2.5,
                "tracking_cn0_late_median_db_hz": (
                    None if point == 180 or label == "target true azimuth" else 40.0
                ),
            })
    settings = ValidationSettings(output_dir=tmp_path, show_plots=False)
    paths = plot_validation(rows, settings)
    assert len(paths) == 6
    assert len(set(paths)) == 6
    np.testing.assert_allclose(captured[0][1][0, 0].lines[0].get_ydata(), [2.0, 0.0])
    np.testing.assert_allclose(captured[1][1][1, 0].lines[0].get_ydata(), [10.0, 0.0])
    np.testing.assert_allclose(captured[3][1][0, 0].lines[0].get_xdata(), [30, 40])
    assert "target true azimuth (deg)" == captured[3][1][0, 0].get_xlabel()
    missing_axis = captured[5][1][1, 0]
    np.testing.assert_allclose(missing_axis.get_xlim(), [29.7, 40.3])
    assert len(missing_axis.get_yticks()) == 0
    assert any(text.get_text() == "No finite tracking estimate" for text in missing_axis.texts)
    for figure, _ in captured:
        plt.close(figure)
