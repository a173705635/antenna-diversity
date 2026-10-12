import numpy as np
import pytest

from core.config import SPEED_OF_LIGHT_MPS
from core.metrics import (
    acquisition_code_bias_chips,
    tracking_code_bias_chips,
)
from core.pipeline import run_end_to_end
from core.presets import default_config


def test_pipeline_runs_four_branches_and_tracks_an_antijam_branch():
    cfg = default_config()
    cfg.signal.duration_ms = 30
    cfg.tracking.max_epochs = 20

    results = run_end_to_end(cfg)

    assert set(results) == {
        "config",
        "truth",
        "scene_input_v",
        "adc_codes",
        "rf_info",
        "dfe_channels",
        "dfe_info",
        "branches",
        "multipath_detection",
        "multipath_evaluation",
        "metrics",
    }
    assert set(results["branches"]) == {
        "single", "egc", "mvdr", "blind_null"
    }
    assert results["dfe_channels"].shape[0] == 4
    assert all(
        np.all(np.isfinite(branch["combined"]))
        for branch in results["branches"].values()
    )
    assert any(
        results["branches"][mode]["tracking"]["success"]
        for mode in ("mvdr", "blind_null")
    )


def test_known_combiners_use_assumed_target_but_blind_null_does_not():
    cfg = default_config()
    cfg.signal.duration_ms = 12
    cfg.signal.assumed_direction.azimuth_deg += 10.0

    results = run_end_to_end(cfg)
    expected = results["truth"]["desired_steering_assumed"]

    np.testing.assert_allclose(
        results["branches"]["mvdr"]["combining"]["constraint_steering"],
        expected,
    )
    assert "constraint_steering" not in results["branches"]["blind_null"][
        "combining"
    ]


def test_pipeline_detects_before_appending_multipath_lcmv_branch():
    cfg = default_config()
    cfg.enable_multipath_research = True
    cfg.signal.duration_ms = 30
    cfg.tracking.max_epochs = 20

    results = run_end_to_end(cfg)

    detection = results["multipath_detection"]
    assert detection["pilot_mode"] in {"single", "egc", "mvdr", "blind_null"}
    assert detection["status"] in {"detected", "no_multipath_detected"}
    assert results["branches"]["multipath_lcmv"]["combining"]["mode"] == "multipath_lcmv"


def test_pipeline_reports_no_pilot_and_uses_mvdr_fallback():
    cfg = default_config()
    cfg.enable_multipath_research = True
    cfg.signal.duration_ms = 12
    cfg.tracking.max_epochs = 5
    cfg.acquisition.threshold = 1e9

    results = run_end_to_end(cfg)

    assert results["multipath_detection"]["status"] == "no_pilot_acquisition"
    combining = results["branches"]["multipath_lcmv"]["combining"]
    assert combining["suppression_status"] == "no_multipath_detected"


def test_code_bias_helpers_use_direct_path_and_tracking_epoch_time():
    fs_hz = 4.092e6
    truth = {
        "code_rate_hz": 1.023e6,
        "code_delay_chips": 10.0,
    }
    acquisition = {"code_phase_samples": 40}
    tracking = {
        "success": True,
        "epoch_start_samples": np.array([40]),
        "code_phase_chips": np.array([0.1]),
    }

    assert acquisition_code_bias_chips(acquisition, truth, fs_hz) == pytest.approx(0.0)
    bias = tracking_code_bias_chips(tracking, truth, fs_hz, 1)
    assert bias[0] == pytest.approx(0.1)
    assert bias[0] * SPEED_OF_LIGHT_MPS / truth["code_rate_hz"] == pytest.approx(
        29.3052256,
        rel=1e-8,
    )


def test_pipeline_metrics_include_multipath_gain_and_code_bias():
    cfg = default_config()
    cfg.enable_multipath_research = True
    cfg.signal.duration_ms = 20
    cfg.tracking.max_epochs = 10

    results = run_end_to_end(cfg)

    assert set(results["multipath_evaluation"]) >= {
        "detected_count", "missed_count", "false_alarm_count", "matches"
    }
    for record in results["metrics"]:
        assert len(record["multipath_direction_gains"]) == 1
        assert np.isfinite(record["worst_multipath_gain"])
        assert np.isfinite(record["acquisition_code_bias_chips"])
        if record["tracking"]:
            assert np.isfinite(record["tracking_code_bias_rms_chips"])
            assert np.isfinite(record["tracking_code_bias_rms_m"])


def test_resolvable_multipath_lcmv_reduces_echo_gain_and_code_bias():
    cfg = default_config()
    cfg.enable_multipath_research = True
    cfg.interferers = []
    cfg.signal.power_dbm = -95.0
    cfg.signal.duration_ms = 30
    cfg.signal.multipath_paths[0].relative_power_db = -3.0
    cfg.multipath_detection.detection_threshold_db = 3.0
    cfg.tracking.max_epochs = 20

    results = run_end_to_end(cfg)
    records = {record["mode"]: record for record in results["metrics"]}

    assert results["multipath_detection"]["candidates"][0][
        "spatial_status"
    ] == "resolvable"
    assert records["multipath_lcmv"]["worst_multipath_gain"] < 1e-6
    assert records["multipath_lcmv"]["worst_multipath_gain"] < records["egc"][
        "worst_multipath_gain"
    ]
    assert records["multipath_lcmv"]["tracking_code_bias_rms_chips"] < records[
        "egc"
    ]["tracking_code_bias_rms_chips"]
