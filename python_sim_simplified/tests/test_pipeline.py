import numpy as np

from pipeline import run_end_to_end
from presets import default_config


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
        "metrics",
    }
    assert set(results["branches"]) == {
        "single", "egc", "mvdr", "blind_null", "multipath_lcmv"
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
    cfg.signal.duration_ms = 30
    cfg.tracking.max_epochs = 20

    results = run_end_to_end(cfg)

    detection = results["multipath_detection"]
    assert detection["pilot_mode"] in {"single", "egc", "mvdr", "blind_null"}
    assert detection["status"] in {"detected", "no_multipath_detected"}
    assert results["branches"]["multipath_lcmv"]["combining"]["mode"] == "multipath_lcmv"


def test_pipeline_reports_no_pilot_and_uses_mvdr_fallback():
    cfg = default_config()
    cfg.signal.duration_ms = 12
    cfg.tracking.max_epochs = 5
    cfg.acquisition.threshold = 1e9

    results = run_end_to_end(cfg)

    assert results["multipath_detection"]["status"] == "no_pilot_acquisition"
    combining = results["branches"]["multipath_lcmv"]["combining"]
    assert combining["suppression_status"] == "no_multipath_detected"
