import copy
import json

import numpy as np
import pytest

import antenna_diversity.experiments.spatial_scan as spatial_scan_module
from antenna_diversity.array.antenna_config import (
    AntennaStateConfig,
    PhoneAntennaConfig,
)
from antenna_diversity.array.effects import AttenuationRegion, CouplingPath
from antenna_diversity.array.geometry import active_corner_combinations
from antenna_diversity.array.phone_model import PhoneAntennaModel
from antenna_diversity.config import CWInterferer, Direction, mate70_default_config
from antenna_diversity.experiments.spatial_scan import (
    compare_corner_combinations,
    evaluate_configuration_spatially,
    scan_assumed_target_directions,
    scan_diagonal_loading,
    scan_interferer_directions,
    scan_interferer_powers,
    scan_position_errors,
)


def test_combination_comparison_returns_all_eleven_phone_subarrays():
    records = compare_corner_combinations(mate70_default_config(), mode="mvdr")

    assert len(records) == 11
    assert tuple(record.active_antennas for record in records) == active_corner_combinations()
    assert {len(record.active_antennas) for record in records} == {2, 3, 4}
    assert len({record.active_antennas for record in records}) == 11
    json.dumps([record.to_dict() for record in records])


def test_direction_scan_returns_requested_grid_without_running_receiver(monkeypatch):
    import antenna_diversity.pipeline as pipeline

    monkeypatch.setattr(
        pipeline,
        "run_end_to_end",
        lambda *_args, **_kwargs: pytest.fail("receiver pipeline must not run"),
    )
    result = scan_interferer_directions(
        mate70_default_config(),
        interferer_index=0,
        azimuths_deg=np.array([0.0, 120.0, 240.0]),
        elevations_deg=np.array([0.0, 30.0]),
        mode="mvdr",
    )

    assert result.output_sinr_db.shape == (2, 3)
    assert result.interferer_gain.shape == (2, 3)
    assert np.all(np.isfinite(result.output_sinr_db))
    json.dumps(result.to_dict())


def test_direction_scan_copies_caller_axes_before_freezing_result():
    azimuths = np.array([0.0, 120.0])
    elevations = np.array([0.0, 30.0])
    result = scan_interferer_directions(
        mate70_default_config(),
        interferer_index=0,
        azimuths_deg=azimuths,
        elevations_deg=elevations,
        mode="mvdr",
    )
    serialized_before = result.to_dict()

    assert not np.shares_memory(result.azimuths_deg, azimuths)
    assert not np.shares_memory(result.elevations_deg, elevations)
    azimuths[:] = [-10.0, -20.0]
    elevations[:] = [80.0, 90.0]

    np.testing.assert_array_equal(result.azimuths_deg, [0.0, 120.0])
    np.testing.assert_array_equal(result.elevations_deg, [0.0, 30.0])
    assert result.to_dict() == serialized_before


def test_loading_and_power_scans_preserve_requested_parameter_order():
    cfg = mate70_default_config()

    loading = scan_diagonal_loading(
        cfg, factors=np.array([1e-4, 1e-2, 1e-1]), mode="mvdr"
    )
    powers = scan_interferer_powers(
        cfg,
        interferer_index=0,
        powers_dbm=np.array([-100.0, -90.0]),
        mode="mvdr",
    )

    np.testing.assert_array_equal(loading.parameter_values, [1e-4, 1e-2, 1e-1])
    np.testing.assert_array_equal(powers.parameter_values, [-100.0, -90.0])
    assert loading.output_sinr_db.shape == (3,)
    assert powers.output_sinr_db.shape == (2,)
    json.dumps(loading.to_dict())
    json.dumps(powers.to_dict())


def test_assumed_position_error_scan_does_not_move_the_true_phase_center():
    cfg = mate70_default_config()
    original_true = copy.deepcopy(cfg.phone.phase_centers_m)
    original_assumed = copy.deepcopy(cfg.phone.assumed_phase_centers_m)
    errors = np.array([[0.0, 0.0, 0.0], [0.002, 0.0, 0.0]])

    result = scan_position_errors(
        cfg, corner_name="TR", xyz_errors_m=errors, mode="mvdr"
    )

    assert result.output_sinr_db.shape == (2,)
    assert cfg.phone.phase_centers_m == original_true
    assert cfg.phone.assumed_phase_centers_m == original_assumed


def test_assumed_direction_scan_keeps_true_direction_and_input_unchanged():
    cfg = mate70_default_config()
    original_true = cfg.signal.true_direction
    original_assumed = cfg.signal.assumed_direction

    result = scan_assumed_target_directions(
        cfg,
        azimuths_deg=np.array([30.0, 90.0]),
        elevations_deg=np.array([60.0]),
        mode="mvdr",
    )

    assert result.output_sinr_db.shape == (1, 2)
    assert cfg.signal.true_direction == original_true == Direction(30.0, 60.0)
    assert cfg.signal.assumed_direction == original_assumed


def test_single_case_uses_only_enabled_interferers():
    cfg = mate70_default_config()
    disabled = copy.deepcopy(cfg)
    disabled.interferers = (
        type(cfg.interferers[0])(
            direction=cfg.interferers[0].direction,
            enable=False,
            power_dbm=100.0,
        ),
    )

    without_jammer = evaluate_configuration_spatially(disabled, mode="mvdr")
    disabled.interferers = ()
    empty_scene = evaluate_configuration_spatially(disabled, mode="mvdr")

    assert without_jammer.metrics["output_sinr_db"] == pytest.approx(
        empty_scene.metrics["output_sinr_db"]
    )


def test_single_case_returns_json_serializable_dataclass():
    result = evaluate_configuration_spatially(mate70_default_config(), mode="mvdr")

    assert result.mode == "mvdr"
    assert result.metrics["covariance_snapshots"] is None
    json.dumps(result.to_dict(), allow_nan=False)


def test_single_case_uses_true_effective_covariance_and_assumed_mvdr_constraint(
    monkeypatch,
):
    cfg = mate70_default_config()
    cfg.array.active_antennas = ("TL", "TR")
    true_state = AntennaStateConfig(
        attenuation_regions=(
            AttenuationRegion(
                ("TL",),
                cfg.signal.true_direction.azimuth_deg,
                cfg.signal.true_direction.elevation_deg,
                1.0,
                6.020599913279624,
            ),
        ),
        coupling_paths=(CouplingPath("TL", "TR", -6.020599913279624, 90.0),),
    )
    assumed_state = AntennaStateConfig(
        coupling_paths=(CouplingPath("TR", "TL", -12.041199826559248, 0.0),),
    )
    cfg.array.antenna_model = PhoneAntennaConfig(
        model_label="analytical integration",
        model_source="ideal",
        true_state=true_state,
        assumed_state=assumed_state,
    )
    model = PhoneAntennaModel.from_config(cfg)
    expected_true = model.effective_manifold(
        cfg.signal.true_direction,
        cfg.signal.true_polarization,
        cfg.signal.carrier_hz,
        estimated=False,
    )
    expected_assumed = model.effective_manifold(
        cfg.signal.assumed_direction,
        cfg.signal.assumed_polarization,
        cfg.signal.carrier_hz,
        estimated=True,
    )
    expected_interferers = tuple(
        model.effective_manifold(
            interferer.direction,
            interferer.polarization,
            cfg.signal.carrier_hz,
            estimated=False,
        )
        for interferer in cfg.interferers
        if interferer.enable
    )
    covariance_inputs = {}
    constraint_inputs = {}
    real_analytic_covariance = spatial_scan_module.analytic_covariance
    real_compute_weights = spatial_scan_module.compute_weights

    def capture_covariance(desired, signal_power, interferers, powers, noise_power):
        covariance_inputs["desired"] = np.array(desired, copy=True)
        covariance_inputs["interferers"] = tuple(
            np.array(value, copy=True) for value in interferers
        )
        return real_analytic_covariance(
            desired, signal_power, interferers, powers, noise_power
        )

    def capture_constraint(covariance, desired, mode, diversity):
        constraint_inputs["desired"] = np.array(desired, copy=True)
        return real_compute_weights(covariance, desired, mode, diversity)

    monkeypatch.setattr(spatial_scan_module, "analytic_covariance", capture_covariance)
    monkeypatch.setattr(spatial_scan_module, "compute_weights", capture_constraint)

    evaluate_configuration_spatially(cfg, mode="mvdr")

    np.testing.assert_allclose(covariance_inputs["desired"], expected_true)
    assert len(covariance_inputs["interferers"]) == len(expected_interferers)
    for actual, expected in zip(
        covariance_inputs["interferers"], expected_interferers, strict=True
    ):
        np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(constraint_inputs["desired"], expected_assumed)


def test_direction_scan_reports_gain_for_selected_enabled_interferer():
    cfg = mate70_default_config()
    cfg.interferers = (
        CWInterferer(direction=Direction(10.0, 10.0), power_dbm=-90.0),
        CWInterferer(direction=Direction(80.0, 20.0), power_dbm=-85.0),
    )
    expected = evaluate_configuration_spatially(cfg, mode="mvdr")

    result = scan_interferer_directions(cfg, 1, [80.0], [20.0], mode="mvdr")

    assert result.interferer_gain[0, 0] == pytest.approx(
        expected.metrics["interferer_direction_gains"][1]
    )


def test_interferer_scans_reject_disabled_selected_interferer():
    cfg = mate70_default_config()
    cfg.interferers = (
        CWInterferer(direction=Direction(120.0, 10.0), enable=False),
    )

    with pytest.raises(ValueError, match="enabled"):
        scan_interferer_directions(cfg, 0, [0.0], [0.0], mode="mvdr")
    with pytest.raises(ValueError, match="enabled"):
        scan_interferer_powers(cfg, 0, [-90.0], mode="mvdr")


def test_scan_rejects_nonfinite_output_metric():
    cfg = mate70_default_config()
    cfg.signal.power_dbm = -5000.0

    with pytest.raises(ValueError, match="finite"):
        scan_diagonal_loading(cfg, [1e-2], mode="mvdr")


def test_finite_signal_dbm_that_underflows_to_zero_is_rejected():
    cfg = mate70_default_config()
    cfg.signal.power_dbm = -5000.0

    with pytest.raises(ValueError, match="signal.power_dbm.*positive"):
        evaluate_configuration_spatially(cfg, mode="mvdr")


def test_finite_interferer_dbm_that_underflows_to_zero_is_rejected():
    cfg = mate70_default_config()
    cfg.interferers = (
        CWInterferer(direction=Direction(120.0, 10.0), power_dbm=-5000.0),
    )

    with pytest.raises(ValueError, match="interferer.power_dbm.*positive"):
        evaluate_configuration_spatially(cfg, mode="mvdr")


def test_parameter_scan_preserves_complete_spatial_metric_schema_and_none():
    single = evaluate_configuration_spatially(mate70_default_config(), mode="mvdr")
    scan = scan_diagonal_loading(mate70_default_config(), [1e-2], mode="mvdr")
    grid = scan_interferer_directions(
        mate70_default_config(), 0, [120.0], [10.0], mode="mvdr"
    )

    assert set(scan.metrics) == set(single.metrics)
    assert set(grid.metrics) == set(single.metrics)
    assert scan.metrics["covariance_snapshots"] == [None]
    assert grid.metrics["covariance_snapshots"] == [[None]]
    json.dumps(scan.to_dict(), allow_nan=False)
    json.dumps(grid.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (
            lambda cfg: scan_interferer_directions(
                cfg, 1, [0.0], [0.0], "mvdr"
            ),
            "interferer_index",
        ),
        (
            lambda cfg: scan_interferer_directions(
                cfg, 0, [[0.0]], [0.0], "mvdr"
            ),
            "azimuths_deg",
        ),
        (
            lambda cfg: scan_assumed_target_directions(
                cfg, [0.0], [91.0], "mvdr"
            ),
            "elevations_deg",
        ),
        (
            lambda cfg: scan_assumed_target_directions(
                cfg, [0.0 + 1.0j], [0.0], "mvdr"
            ),
            "azimuths_deg",
        ),
        (
            lambda cfg: scan_interferer_powers(cfg, 0, [np.nan], "mvdr"),
            "powers_dbm",
        ),
        (
            lambda cfg: scan_diagonal_loading(cfg, [0.0], "mvdr"),
            "factors",
        ),
        (
            lambda cfg: scan_position_errors(cfg, "XX", [[0.0, 0.0, 0.0]], "mvdr"),
            "corner_name",
        ),
        (
            lambda cfg: scan_position_errors(cfg, "TR", [0.0, 0.0, 0.0], "mvdr"),
            "xyz_errors_m",
        ),
        (
            lambda cfg: scan_position_errors(cfg, "TR", [], "mvdr"),
            "xyz_errors_m",
        ),
        (
            lambda cfg: scan_position_errors(
                cfg, "TR", [[0.0 + 1.0j, 0.0, 0.0]], "mvdr"
            ),
            "xyz_errors_m",
        ),
        (
            lambda cfg: scan_position_errors(
                cfg, ["TR"], [[0.0, 0.0, 0.0]], "mvdr"
            ),
            "corner_name",
        ),
    ],
)
def test_scans_reject_invalid_indices_grids_parameters_and_errors(call, match):
    with pytest.raises(ValueError, match=match):
        call(mate70_default_config())
