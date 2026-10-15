import numpy as np
import pytest

from core.config import SPEED_OF_LIGHT_MPS
from core.metrics import (
    acquisition_code_bias_chips,
    tracking_code_bias_chips,
)
from core.pipeline import run_end_to_end, run_frontend, run_receiver
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


def test_pipeline_metrics_include_code_bias():
    cfg = default_config()
    cfg.signal.duration_ms = 20
    cfg.tracking.max_epochs = 10

    results = run_end_to_end(cfg)

    for record in results["metrics"]:
        assert np.isfinite(record["acquisition_code_bias_chips"])
        if record["tracking"]:
            assert np.isfinite(record["tracking_code_bias_rms_chips"])
            assert np.isfinite(record["tracking_code_bias_rms_m"])


def test_frontend_and_receiver_reproduce_complete_chain():
    cfg = default_config()
    cfg.signal.duration_ms = 12
    cfg.tracking.max_epochs = 8
    complete = run_end_to_end(cfg)
    frontend = run_frontend(cfg)
    branches = run_receiver(
        frontend["dfe_channels"], frontend["truth"], cfg,
        frontend["dfe_info"]["fs_out_hz"],
    )
    assert frontend["branches"] == {}
    assert frontend["metrics"] == []
    for name in ("scene_input_v", "adc_codes", "dfe_channels"):
        np.testing.assert_array_equal(frontend[name], complete[name])
    assert set(branches) == set(cfg.diversity.modes)
    for mode in cfg.diversity.modes:
        expected, actual = complete["branches"][mode], branches[mode]
        np.testing.assert_array_equal(actual["combined"], expected["combined"])
        np.testing.assert_array_equal(actual["combining"]["weights"], expected["combining"]["weights"])
        assert actual["acquisition"]["metric"] == expected["acquisition"]["metric"]
        assert actual["tracking"]["status"] == expected["tracking"]["status"]
