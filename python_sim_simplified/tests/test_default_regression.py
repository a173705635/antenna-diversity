import pytest

from pipeline import run_end_to_end
from presets import default_config


EXPECTED = {
    "single": (False, 1.0160803563227625, -1750.0, None),
    "egc": (True, 33.69005705636806, 1500.0, 50.051921894528355),
    "mvdr": (True, 34.961314521617254, 1500.0, 50.13246168366476),
    "blind_null": (True, 15.893485034538864, 1500.0, 45.123190884508226),
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
