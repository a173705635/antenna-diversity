import numpy as np
import pytest

from antenna_diversity.config import CWInterferer, Direction, mate70_default_config
from antenna_diversity.gnss.scenario import generate_scenario


def test_phone_scenario_generates_four_channels_and_two_interferer_components():
    cfg = mate70_default_config()
    cfg.interferers = (
        CWInterferer(Direction(120.0, 10.0), power_dbm=-90.0, offset_hz=250e3),
        CWInterferer(Direction(220.0, 20.0), power_dbm=-93.0, offset_hz=-400e3),
    )

    x, truth = generate_scenario(cfg)

    assert x.shape[0] == 4
    assert len(truth["interferer_components"]) == 2
    assert len(truth["interferer_steering_nominal"]) == 2
    np.testing.assert_allclose(
        x,
        truth["desired_component"] + sum(truth["interferer_components"]),
    )


def test_cw_start_and_stop_times_gate_only_that_interferer():
    cfg = mate70_default_config()
    cfg.signal.duration_ms = 10
    cfg.interferers = (
        CWInterferer(
            Direction(120.0, 10.0),
            power_dbm=-90.0,
            start_ms=2.0,
            stop_ms=6.0,
        ),
    )

    _, truth = generate_scenario(cfg)
    component = truth["interferer_components"][0]
    first = round(2e-3 * cfg.signal.fs_rf_hz)
    stop = round(6e-3 * cfg.signal.fs_rf_hz)

    assert np.all(component[:, :first] == 0)
    assert np.any(component[:, first:stop] != 0)
    assert np.all(component[:, stop:] == 0)


def test_true_and_assumed_target_manifolds_are_distinct_under_doa_error():
    cfg = mate70_default_config()
    cfg.signal.assumed_direction = Direction(35.0, 60.0)

    _, truth = generate_scenario(cfg)

    assert not np.allclose(
        truth["desired_steering_true"],
        truth["desired_steering_assumed"],
    )


def test_phone_scenario_supports_no_interferers():
    cfg = mate70_default_config()
    cfg.interferers = ()

    x, truth = generate_scenario(cfg)

    assert truth["interferer_components"] == []
    np.testing.assert_allclose(x, truth["desired_component"])


def test_negative_cw_start_time_is_rejected_instead_of_wrapping():
    cfg = mate70_default_config()
    cfg.interferers = (
        CWInterferer(Direction(120.0, 10.0), start_ms=-1.0),
    )

    with pytest.raises(ValueError, match="start_ms"):
        generate_scenario(cfg)


@pytest.mark.parametrize(
    "gate_times",
    [
        {"start_ms": 10.0 + 0.4 / 16_368.0},
        {"start_ms": 0.0, "stop_ms": 10.0 + 0.4 / 16_368.0},
    ],
)
def test_cw_gate_rejects_subsample_overrun_before_rounding(gate_times):
    cfg = mate70_default_config()
    cfg.signal.duration_ms = 10
    cfg.interferers = (
        CWInterferer(Direction(120.0, 10.0), **gate_times),
    )

    with pytest.raises(ValueError, match="signal duration"):
        generate_scenario(cfg)


def test_cw_gate_accepts_exact_signal_duration_boundary():
    cfg = mate70_default_config()
    cfg.signal.duration_ms = 10
    cfg.interferers = (
        CWInterferer(Direction(120.0, 10.0), start_ms=10.0),
        CWInterferer(Direction(220.0, 20.0), stop_ms=10.0),
    )

    _, truth = generate_scenario(cfg)

    assert np.all(truth["interferer_components"][0] == 0)
    assert np.all(truth["interferer_components"][1] != 0)
