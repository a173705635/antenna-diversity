import json

import copy

import numpy as np
import pytest

from antenna_diversity.array.antenna_config import (
    AntennaStateConfig,
    PhoneAntennaConfig,
)
from antenna_diversity.array.effects import AttenuationRegion
from antenna_diversity.array.phone_model import PhoneAntennaModel
from antenna_diversity.analysis.spatial import (
    analytic_covariance,
    evaluate_spatial_case,
    response_grid_db,
)
from antenna_diversity.config import (
    Direction,
    default_config,
    mate70_default_config,
)


def test_analytic_covariance_sums_signal_interferers_and_white_noise():
    desired = np.array([1.0, 1.0j])
    jammer = np.array([1.0, -1.0])
    covariance = analytic_covariance(
        desired,
        signal_power_w=2.0,
        interferer_steering=(jammer,),
        interferer_powers_w=(5.0,),
        noise_power_per_element_w=0.25,
    )

    expected = 2.0 * np.outer(desired, desired.conj())
    expected += 5.0 * np.outer(jammer, jammer.conj())
    expected += 0.25 * np.eye(2)
    np.testing.assert_allclose(covariance, expected)


def test_analytic_covariance_supports_no_interferers():
    desired = np.array([1.0, 1.0j])

    covariance = analytic_covariance(
        desired,
        signal_power_w=2.0,
        interferer_steering=(),
        interferer_powers_w=(),
        noise_power_per_element_w=0.25,
    )

    expected = 2.0 * np.outer(desired, desired.conj()) + 0.25 * np.eye(2)
    np.testing.assert_allclose(covariance, expected)


@pytest.mark.parametrize(
    ("desired", "signal_power", "interferers", "powers", "noise_power"),
    [
        ([1.0, 1.0], -1.0, (), (), 0.0),
        ([1.0, 1.0], np.inf, (), (), 0.0),
        ([1.0, 1.0], 1.0, ([1.0],), (1.0,), 0.0),
        ([1.0, 1.0], 1.0, ([1.0, 1.0],), (-1.0,), 0.0),
        ([1.0, 1.0], 1.0, ([1.0, 1.0],), (np.nan,), 0.0),
        ([1.0, 1.0], 1.0, (), (), -1.0),
    ],
)
def test_analytic_covariance_rejects_invalid_dimensions_or_powers(
    desired, signal_power, interferers, powers, noise_power
):
    with pytest.raises(ValueError):
        analytic_covariance(
            desired,
            signal_power_w=signal_power,
            interferer_steering=interferers,
            interferer_powers_w=powers,
            noise_power_per_element_w=noise_power,
        )


def test_analytic_covariance_rejects_mismatched_interferer_counts():
    with pytest.raises(ValueError):
        analytic_covariance(
            [1.0, 1.0],
            signal_power_w=1.0,
            interferer_steering=([1.0, -1.0],),
            interferer_powers_w=(),
            noise_power_per_element_w=0.0,
        )


def test_spatial_metrics_report_exact_output_power_decomposition():
    weights = np.array([0.5, 0.5])
    desired = np.array([1.0, 1.0])
    jammer = np.array([1.0, -1.0])
    metrics = evaluate_spatial_case(
        weights,
        desired,
        signal_power_w=2.0,
        interferer_steering=(jammer,),
        interferer_powers_w=(5.0,),
        noise_power_per_element_w=0.25,
    )

    assert metrics["target_direction_gain"] == pytest.approx(1.0)
    assert metrics["interferer_direction_gains"][0] == pytest.approx(0.0)
    assert metrics["output_signal_power_w"] == pytest.approx(2.0)
    assert metrics["output_interference_power_w"] == pytest.approx(0.0)
    assert metrics["output_noise_power_w"] == pytest.approx(0.125)
    assert metrics["output_sinr_db"] == pytest.approx(10.0 * np.log10(16.0))
    assert metrics["weight_norm"] == pytest.approx(np.sqrt(0.5))
    assert metrics["white_noise_gain"] == pytest.approx(0.5)
    assert metrics["worst_interferer_gain"] == pytest.approx(0.0)
    assert metrics["covariance_rank"] is None
    assert metrics["covariance_condition_number"] is None
    assert metrics["diagonal_loading"] is None
    assert metrics["covariance_snapshots"] is None
    assert metrics["weights_are_fixed"] is True
    json.dumps(metrics)


def test_spatial_metrics_report_covariance_and_snapshot_metadata():
    metrics = evaluate_spatial_case(
        np.array([1.0, 0.0]),
        np.array([1.0, 1.0]),
        signal_power_w=2.0,
        interferer_steering=(),
        interferer_powers_w=(),
        noise_power_per_element_w=0.25,
        covariance=np.diag([2.0, 1.0]),
        diagonal_loading=np.float64(0.125),
        covariance_snapshots=np.int64(32),
        weights_are_fixed=np.bool_(False),
    )

    assert metrics["interferer_direction_gains"] == []
    assert metrics["covariance_rank"] == 2
    assert isinstance(metrics["covariance_rank"], int)
    assert metrics["covariance_condition_number"] == pytest.approx(2.0)
    assert isinstance(metrics["covariance_condition_number"], float)
    assert metrics["diagonal_loading"] == pytest.approx(0.125)
    assert metrics["covariance_snapshots"] == 32
    assert metrics["weights_are_fixed"] is False
    json.dumps(metrics)


def test_spatial_metrics_reject_non_hermitian_covariance():
    with pytest.raises(ValueError, match="Hermitian"):
        evaluate_spatial_case(
            np.array([1.0, 0.0]),
            np.array([1.0, 1.0]),
            signal_power_w=1.0,
            interferer_steering=(),
            interferer_powers_w=(),
            noise_power_per_element_w=0.0,
            covariance=np.array([[1.0, 2.0], [0.0, 1.0]]),
        )


def test_spatial_metrics_accept_covariance_with_roundoff_asymmetry():
    metrics = evaluate_spatial_case(
        np.array([1.0, 0.0]),
        np.array([1.0, 1.0]),
        signal_power_w=1.0,
        interferer_steering=(),
        interferer_powers_w=(),
        noise_power_per_element_w=0.0,
        covariance=np.array(
            [[2.0, 0.5 + 0.25j], [0.5 - 0.25j + 1e-11, 1.0]]
        ),
    )

    assert metrics["covariance_rank"] == 2
    assert np.isfinite(metrics["covariance_condition_number"])


def test_spatial_metrics_use_epsilon_floor_for_zero_output_denominator():
    metrics = evaluate_spatial_case(
        np.array([1.0]),
        np.array([1.0]),
        signal_power_w=1.0,
        interferer_steering=(),
        interferer_powers_w=(),
        noise_power_per_element_w=0.0,
    )

    assert metrics["output_sinr_db"] == pytest.approx(
        10.0 * np.log10(1.0 / np.finfo(float).eps)
    )


def test_spatial_metrics_define_zero_over_zero_sinr_without_warning():
    with np.errstate(divide="raise", invalid="raise"):
        metrics = evaluate_spatial_case(
            np.array([1.0]),
            np.array([1.0]),
            signal_power_w=0.0,
            interferer_steering=(),
            interferer_powers_w=(),
            noise_power_per_element_w=0.0,
        )

    assert metrics["output_sinr_db"] == -np.inf


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weights": [1.0], "desired_steering": [1.0, 1.0]},
        {
            "interferer_steering": ([1.0, -1.0],),
            "interferer_powers_w": (),
        },
        {"signal_power_w": -1.0},
        {"noise_power_per_element_w": np.inf},
        {"weights": [np.nan, 0.0]},
    ],
)
def test_spatial_metrics_reject_invalid_inputs(kwargs):
    arguments = {
        "weights": [0.5, 0.5],
        "desired_steering": [1.0, 1.0],
        "signal_power_w": 1.0,
        "interferer_steering": (),
        "interferer_powers_w": (),
        "noise_power_per_element_w": 0.0,
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError):
        evaluate_spatial_case(**arguments)


def test_response_grid_has_elevation_by_azimuth_shape_and_marks_unit_constraint():
    cfg = mate70_default_config()
    weights = np.ones(4, dtype=complex) / 4.0
    azimuths = np.array([0.0, 90.0, 180.0])
    elevations = np.array([0.0, 45.0, 90.0])

    response = response_grid_db(cfg, weights, azimuths, elevations)

    assert response.shape == (3, 3)
    assert np.all(np.isfinite(response))
    assert np.max(response) == pytest.approx(0.0)
    assert np.min(response) >= -80.0


@pytest.mark.parametrize(
    ("azimuths", "elevations"),
    [
        ([], [0.0]),
        ([0.0], []),
        ([[0.0]], [0.0]),
        ([0.0], [[0.0]]),
        ([np.nan], [0.0]),
        ([0.0], [np.inf]),
        ([0.0], [-1.0]),
        ([0.0], [91.0]),
    ],
)
def test_response_grid_rejects_invalid_direction_grids(azimuths, elevations):
    with pytest.raises(ValueError):
        response_grid_db(
            mate70_default_config(),
            np.ones(4, dtype=complex) / 4.0,
            azimuths,
            elevations,
        )


def test_response_grid_rejects_non_phone_config_and_wrong_weight_count():
    with pytest.raises(ValueError):
        response_grid_db(default_config(), [1.0, 0.0], [0.0], [0.0])

    with pytest.raises(ValueError):
        response_grid_db(mate70_default_config(), [1.0, 0.0], [0.0], [0.0])


def test_response_grid_uses_active_phone_positions_in_configured_order():
    cfg = mate70_default_config()
    cfg.array.active_antennas = ("TL", "BR")

    response = response_grid_db(
        cfg,
        np.array([1.0, 0.0]),
        np.array([0.0, 90.0]),
        np.array([0.0, 90.0]),
    )

    np.testing.assert_allclose(response, np.zeros((2, 2)), atol=1e-12)


def test_response_grid_returns_floor_when_every_requested_point_is_exact_null():
    cfg = mate70_default_config()
    cfg.array.active_antennas = ("TL", "TR")
    cfg.phone.phase_centers_m["TL"] = (0.0, 0.0, 0.0)
    cfg.phone.phase_centers_m["TR"] = (0.0, 0.0, 0.0)

    response = response_grid_db(
        cfg,
        np.array([1.0, -1.0]),
        np.array([0.0, 90.0]),
        np.array([0.0, 45.0, 90.0]),
    )

    assert response.shape == (3, 2)
    assert np.all(np.isfinite(response))
    np.testing.assert_array_equal(response, np.full((3, 2), -80.0))


def test_response_grid_uses_selected_phone_model_view_without_normalizing_manifold():
    cfg = mate70_default_config()
    cfg.array.active_antennas = ("TL", "TR")
    true_state = AntennaStateConfig(
        attenuation_regions=(
            AttenuationRegion(("TL",), 0.0, 45.0, 1.0, 6.020599913279624),
        ),
    )
    cfg.array.antenna_model = PhoneAntennaConfig(
        model_label="grid fixture",
        model_source="ideal",
        true_state=true_state,
        assumed_state=copy.deepcopy(AntennaStateConfig()),
    )
    weights = np.array([1.0, 0.0])
    azimuths = np.array([0.0, 90.0])
    elevations = np.array([45.0])

    true_response = response_grid_db(
        cfg, weights, azimuths, elevations, model_view="true"
    )
    assumed_response = response_grid_db(
        cfg, weights, azimuths, elevations, model_view="ASSUMED"
    )

    np.testing.assert_allclose(true_response, [[-6.020599913279624, 0.0]], atol=1e-12)
    np.testing.assert_allclose(assumed_response, [[0.0, 0.0]], atol=1e-12)
    model = PhoneAntennaModel.from_config(cfg)
    open_circuit = model.open_circuit_response(
        Direction(0.0, 45.0),
        cfg.signal.true_polarization,
        cfg.signal.carrier_hz,
        estimated=False,
    )
    assert abs(open_circuit[0]) == pytest.approx(0.5)


def test_response_grid_rejects_unknown_model_view():
    with pytest.raises(ValueError, match="model_view must be true or assumed"):
        response_grid_db(
            mate70_default_config(),
            np.ones(4),
            [0.0],
            [45.0],
            model_view="nominal",
        )
