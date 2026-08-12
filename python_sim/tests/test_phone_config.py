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


@pytest.mark.parametrize("elevation", [-0.1, 90.1])
def test_direction_rejects_elevation_outside_visible_hemisphere(elevation):
    with pytest.raises(ValueError, match="elevation"):
        Direction(30.0, elevation).validate()


def test_phone_config_rejects_more_than_three_interferers():
    cfg = mate70_default_config()
    cfg.interferers = tuple(
        CWInterferer(direction=Direction(index * 30.0, 10.0))
        for index in range(4)
    )

    with pytest.raises(ValueError, match="three"):
        cfg.validate()


def test_legacy_constructor_preserves_original_sampling_and_array():
    cfg = legacy_ula_config()

    assert cfg.array.num_elements == 2
    assert cfg.signal.fs_rf_hz == 16.368e6
    assert cfg.dfe.fs_out_hz == 4.092e6
    assert cfg.array.desired_angle_deg == -10.0
