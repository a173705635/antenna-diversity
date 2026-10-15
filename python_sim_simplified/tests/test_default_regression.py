import numpy as np
import pytest

from core.pipeline import run_end_to_end
from core.presets import default_config


def test_legacy_sample_rate_preserves_receiver_contract_with_physical_rf_noise():
    cfg = default_config()
    # Exact acquisition/CN0 snapshots describe the obsolete unfiltered RF noise model.
    cfg.signal.fs_rf_hz = 16.368e6
    cfg.dfe.decimation = 4
    results = run_end_to_end(cfg)

    assert results["scene_input_v"].shape == (4, 1_636_800)
    assert results["adc_codes"].shape == (4, 1_636_800)
    assert results["dfe_channels"].shape == (4, 409_184)
    assert results["rf_info"]["total_gain_db"] == pytest.approx(80.0)
    assert results["rf_info"]["noise_figure_db"] == pytest.approx(
        1.2350293663855842
    )
    assert results["rf_info"]["fs_hz"] == cfg.signal.fs_rf_hz
    assert results["rf_info"]["if_hz"] == cfg.signal.if_hz
    assert results["rf_info"]["channel_ids"] == cfg.phone.active_antennas
    assert results["rf_info"]["measured_noise_bandwidth_hz"] == pytest.approx(
        cfg.rf.noise_bandwidth_hz
    )
    assert results["dfe_info"]["fs_out_hz"] == pytest.approx(4_092_000.0)

    # Seed-42 ideal four-corner scene: known target protection and blind nulling
    # recover the real PRN; unchanged SINGLE/EGC cannot overcome this jammer.
    for mode in ('single', 'egc', 'mvdr', 'blind_null'):
        expected_acquired = mode in ('mvdr', 'blind_null')
        branch = results['branches'][mode]
        assert branch['acquisition']['success'] is expected_acquired
        assert branch['tracking']['success'] is expected_acquired
        if expected_acquired:
            assert abs(branch['acquisition']['doppler_hz'] - cfg.signal.doppler_hz) <= 125.0

    for record in results["metrics"]:
        assert np.isfinite(record["acquisition_metric"])
        assert np.isfinite(record["acquired_doppler_hz"])
        if record["tracking"]:
            assert abs(record['acquisition_code_bias_chips']) < 0.5
            assert np.isfinite(record["final_cn0_db_hz"])


def test_default_interference_pipeline_contract():
    cfg = default_config()
    results = run_end_to_end(cfg)

    assert not hasattr(cfg, "enable_multipath_research")
    assert not hasattr(cfg.signal, "multipath_paths")
    assert set(results["branches"]) == {
        "single",
        "egc",
        "mvdr",
        "blind_null",
    }
    assert "multipath_detection" not in results
    assert "multipath_evaluation" not in results
    assert "multipath_paths" not in results["truth"]
    for branch in results["branches"].values():
        assert np.all(np.isfinite(branch["combined"]))
    for record in results["metrics"]:
        assert np.isfinite(record["acquisition_metric"])
        assert "worst_multipath_gain" not in record
        assert np.isfinite(record["acquisition_code_bias_chips"])
