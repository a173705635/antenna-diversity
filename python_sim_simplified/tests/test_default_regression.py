import pytest

from pipeline import run_end_to_end
from presets import default_config


EXPECTED = {
    "single": (False, 1.200056624297749, -2000.0, None),
    "egc": (False, 1.088786452030734, -2250.0, None),
    "mvdr": (True, 30.344184898182302, 1500.0, 49.69009146121891),
    "blind_null": (True, 10.109661638536469, 1500.0, 45.06987725528034),
}


def test_default_simulation_matches_locked_baseline():
    results = run_end_to_end(default_config())

    assert results["scene_input_v"].shape == (4, 1_636_800)
    assert results["adc_codes"].shape == (4, 1_636_800)
    assert results["dfe_channels"].shape == (4, 409_184)
    assert results["rf_info"]["total_gain_db"] == pytest.approx(80.0)
    assert results["rf_info"]["noise_figure_db"] == pytest.approx(
        1.2350293663855842
    )
    assert results["dfe_info"]["fs_out_hz"] == pytest.approx(4_092_000.0)

    for record in results["metrics"]:
        tracked, metric, doppler_hz, cn0_db_hz = EXPECTED[record["mode"]]
        assert record["tracking"] is tracked
        assert record["acquisition_metric"] == pytest.approx(metric, rel=1e-10)
        assert record["acquired_doppler_hz"] == pytest.approx(doppler_hz)
        if cn0_db_hz is None:
            assert record["final_cn0_db_hz"] is None
        else:
            assert record["final_cn0_db_hz"] == pytest.approx(
                cn0_db_hz,
                rel=1e-10,
            )
