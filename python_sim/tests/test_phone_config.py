import numpy as np
import pytest

from antenna_diversity.config import (
    CWInterferer,
    Direction,
    legacy_ula_config,
    mate70_default_config,
)


def test_mate70_config_uses_all_four_ideal_corners():
    cfg = mate70_default_config()

    assert cfg.phone.model_name == "HUAWEI Mate 70"
    assert cfg.phone.length_m == pytest.approx(0.1609)
    assert cfg.phone.width_m == pytest.approx(0.0759)
    assert cfg.array.active_antennas == ("TL", "TR", "BL", "BR")
    np.testing.assert_allclose(
        [cfg.phone.phase_centers_m[name] for name in cfg.array.active_antennas],
        [
            [-0.03795, 0.08045, 0.0],
            [0.03795, 0.08045, 0.0],
            [-0.03795, -0.08045, 0.0],
            [0.03795, -0.08045, 0.0],
        ],
    )


def test_mate70_config_disables_legacy_jammer_when_using_cw_interferer():
    cfg = mate70_default_config()

    assert not cfg.jammer.enable
    assert len(cfg.interferers) == 1
    assert cfg.interferers[0].enable


def test_phone_config_rejects_enabled_legacy_jammer():
    cfg = mate70_default_config()
    cfg.jammer.enable = True

    with pytest.raises(ValueError, match="legacy jammer.*phone"):
        cfg.validate()


def test_legacy_config_rejects_phone_interferers():
    cfg = legacy_ula_config()
    cfg.interferers = (CWInterferer(Direction(120.0, 10.0)),)

    with pytest.raises(ValueError, match="interferers.*legacy"):
        cfg.validate()


def test_phone_channel_response_uses_real_errors_in_active_antenna_order():
    cfg = mate70_default_config()
    cfg.array.active_antennas = ("BR", "TL", "TR")
    cfg.array.channel_amplitude_error_db = {
        "TL": 0.0,
        "TR": -20.0,
        "BL": 3.0,
        "BR": 20.0,
    }
    cfg.array.channel_phase_error_deg = {
        "TL": 0.0,
        "TR": 180.0,
        "BL": 45.0,
        "BR": 90.0,
    }

    response = cfg.array.channel_response(estimated=False)

    np.testing.assert_allclose(response, [10.0j, 1.0 + 0.0j, -0.1 + 0.0j], atol=1e-12)


def test_phone_channel_response_uses_estimated_errors_in_active_antenna_order():
    cfg = mate70_default_config()
    cfg.array.active_antennas = ("BR", "TL", "TR")
    cfg.array.estimated_channel_amplitude_error_db = {
        "TL": 20.0,
        "TR": 20.0 * np.log10(2.0),
        "BL": -3.0,
        "BR": 20.0 * np.log10(0.5),
    }
    cfg.array.estimated_channel_phase_error_deg = {
        "TL": 180.0,
        "TR": 0.0,
        "BL": 30.0,
        "BR": -90.0,
    }

    response = cfg.array.channel_response(estimated=True)

    np.testing.assert_allclose(response, [-0.5j, -10.0 + 0.0j, 2.0 + 0.0j], atol=1e-12)


def test_phone_config_accepts_arbitrary_unique_two_to_four_corner_subset():
    cfg = mate70_default_config()
    cfg.array.active_antennas = ("TR", "BL")
    cfg.validate()

    assert cfg.array.num_elements == 2
    np.testing.assert_allclose(
        [cfg.phone.phase_centers_m[name] for name in cfg.array.active_antennas],
        [[0.03795, 0.08045, 0.0], [-0.03795, -0.08045, 0.0]],
    )


@pytest.mark.parametrize(
    "active",
    [("TL",), ("TL", "TL"), ("TL", "XX"), ("TL", "TR", "BL", "BR", "TL")],
)
def test_phone_config_rejects_invalid_corner_selections(active):
    cfg = mate70_default_config()
    cfg.array.active_antennas = active

    with pytest.raises(ValueError, match="active_antennas"):
        cfg.validate()


def test_phone_config_rejects_consistent_noncorner_coordinate_and_error_maps():
    cfg = mate70_default_config()
    noncorners = ("XX", "YY", "ZZ", "WW")
    phase_centers = {
        name: (float(index), 0.0, 0.0)
        for index, name in enumerate(noncorners)
    }
    cfg.phone.phase_centers_m = phase_centers
    cfg.phone.assumed_phase_centers_m = phase_centers.copy()
    cfg.array.active_antennas = ("XX", "YY")
    cfg.array.channel_amplitude_error_db = {name: 0.0 for name in noncorners}
    cfg.array.channel_phase_error_deg = {name: 0.0 for name in noncorners}
    cfg.array.estimated_channel_amplitude_error_db = {name: 0.0 for name in noncorners}
    cfg.array.estimated_channel_phase_error_deg = {name: 0.0 for name in noncorners}

    with pytest.raises(ValueError, match="corner-name set"):
        cfg.validate()


@pytest.mark.parametrize("elevation", [-0.1, 90.1])
def test_direction_rejects_elevation_outside_visible_hemisphere(elevation):
    with pytest.raises(ValueError, match="elevation"):
        Direction(30.0, elevation).validate()


def test_phone_config_rejects_more_than_three_interferers():
    cfg = mate70_default_config()
    cfg.interferers = tuple(
        CWInterferer(direction=Direction(index * 30.0, 10.0), enable=False)
        for index in range(4)
    )

    with pytest.raises(ValueError, match="three"):
        cfg.validate()


@pytest.mark.parametrize("count", [0, 3])
def test_phone_config_accepts_zero_to_three_configured_interferers(count):
    cfg = mate70_default_config()
    cfg.interferers = tuple(
        CWInterferer(direction=Direction(index * 30.0, 10.0), enable=False)
        for index in range(count)
    )

    cfg.validate()


def test_phone_config_validates_each_interferer_direction():
    cfg = mate70_default_config()
    cfg.interferers = (CWInterferer(Direction(np.nan, 10.0)),)

    with pytest.raises(ValueError, match="direction.azimuth_deg"):
        cfg.validate()


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"power_dbm": np.nan}, "interferer.power_dbm"),
        ({"offset_hz": np.inf}, "interferer.offset_hz"),
        ({"initial_phase_rad": -np.inf}, "interferer.initial_phase_rad"),
        ({"start_ms": np.nan}, "interferer.start_ms"),
        ({"start_ms": -0.1}, "interferer.start_ms"),
        ({"stop_ms": np.inf}, "interferer.stop_ms"),
        ({"start_ms": 4.0, "stop_ms": 3.9}, "interferer.stop_ms"),
        ({"start_ms": 40.1}, "interferer.start_ms.*signal.duration_ms"),
        ({"stop_ms": 40.1}, "interferer.stop_ms.*signal.duration_ms"),
    ],
)
def test_phone_config_rejects_invalid_interferer_fields(overrides, match):
    cfg = mate70_default_config()
    cfg.interferers = (CWInterferer(Direction(120.0, 10.0), **overrides),)

    with pytest.raises(ValueError, match=match):
        cfg.validate()


def test_phone_config_accepts_interferer_gate_at_signal_duration_boundary():
    cfg = mate70_default_config()
    cfg.interferers = (
        CWInterferer(Direction(120.0, 10.0), start_ms=40.0),
        CWInterferer(Direction(220.0, 20.0), stop_ms=40.0),
    )

    cfg.validate()


def test_legacy_constructor_preserves_original_sampling_and_array():
    cfg = legacy_ula_config()

    assert cfg.array.num_elements == 2
    assert cfg.signal.fs_rf_hz == 16.368e6
    assert cfg.dfe.fs_out_hz == 4.092e6
    assert cfg.array.desired_angle_deg == -10.0
