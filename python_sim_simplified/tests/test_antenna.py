import csv

import numpy as np

from core.antenna import (
    antenna_response,
    attenuation_voltage,
    build_antenna_model,
    interpolate_measured,
    load_measured_pattern,
    parametric_height,
)
from core.config import (
    AntennaState,
    AttenuationRegion,
    CouplingPath,
    Direction,
    ElementOrientation,
    ParametricPattern,
    SPEED_OF_LIGHT_MPS,
    WavePolarization,
)
from core.presets import default_config, measured_config, parametric_config


def test_antenna_model_owns_only_antenna_inputs():
    cfg = default_config()

    model = build_antenna_model(cfg.phone, cfg.signal.carrier_hz)

    assert not hasattr(model, "cfg")
    assert model.active_antennas == cfg.phone.active_antennas
    assert model.carrier_hz == cfg.signal.carrier_hz


def test_response_composes_open_circuit_coupling_and_channel_phase():
    cfg = default_config()
    cfg.phone.channel_phase_error_deg["TR"] = 90.0
    response, stages = antenna_response(
        build_antenna_model(cfg.phone, cfg.signal.carrier_hz),
        cfg.signal.true_direction,
        cfg.signal.true_polarization,
    )

    np.testing.assert_allclose(
        response,
        stages["channel_response"]
        * (stages["coupling_matrix"] @ stages["open_circuit_response"]),
    )
    assert np.isclose(np.angle(stages["channel_response"][1], deg=True), 90.0)


def test_parametric_model_applies_pattern_blockage_and_coupling():
    cfg = parametric_config(("TL", "TR", "BL", "BR"))
    response, stages = antenna_response(
        build_antenna_model(cfg.phone, cfg.signal.carrier_hz),
        Direction(225.0, 20.0),
        WavePolarization("rhcp"),
    )

    assert response.shape == (4,)
    assert np.all(np.abs(stages["blockage_voltage"][[2, 3]]) < 1.0)
    assert np.any(np.abs(stages["coupling_matrix"] - np.eye(4)) > 0.0)
    assert np.ptp(np.abs(stages["pattern_polarization_voltage"])) > 0.0


def test_measured_csv_interpolates_complex_jones_components(tmp_path):
    path = tmp_path / "pattern.csv"
    columns = [
        "element_name", "frequency_hz", "azimuth_deg", "elevation_deg",
        "h_az_real", "h_az_imag", "h_el_real", "h_el_imag",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for name in ("TL", "TR", "BL", "BR"):
            for elevation in (0.0, 90.0):
                for azimuth in (0.0, 180.0):
                    writer.writerow(
                        {
                            "element_name": name,
                            "frequency_hz": 1_575.42e6,
                            "azimuth_deg": azimuth,
                            "elevation_deg": elevation,
                            "h_az_real": 1.0 + azimuth / 180.0,
                            "h_az_imag": elevation / 90.0,
                            "h_el_real": 0.25,
                            "h_el_imag": -0.5,
                        }
                    )

    cfg = measured_config(path)
    grids = load_measured_pattern(path, cfg.signal.carrier_hz)
    height = interpolate_measured(grids["TL"], 90.0, 45.0)
    np.testing.assert_allclose(height, [1.5 + 0.5j, 0.25 - 0.5j])

    response, stages = antenna_response(
        build_antenna_model(cfg.phone, cfg.signal.carrier_hz),
        Direction(90.0, 45.0),
        WavePolarization("rhcp"),
    )
    assert response.shape == (4,)
    assert np.all(np.isfinite(response))
    assert stages["pattern_polarization_voltage"].shape == (4,)
    np.testing.assert_allclose(stages["pcv_path_length_mm"], np.zeros(4))
    np.testing.assert_allclose(stages["pcv_response"], np.ones(4))


def test_azimuth_gain_has_prescribed_depth_and_no_zenith_effect():
    pattern = ParametricPattern(azimuth_peak_deg=30.0, azimuth_attenuation_db=8.0)
    peak = parametric_height(pattern, 0.0, 30.0)
    opposite = parametric_height(pattern, 0.0, 210.0)
    np.testing.assert_allclose(
        np.linalg.norm(opposite) / np.linalg.norm(peak), 10**(-8 / 20)
    )
    np.testing.assert_allclose(
        parametric_height(pattern, 90.0, 30.0),
        parametric_height(pattern, 90.0, 210.0),
    )


def test_response_uses_local_azimuth_for_pattern_gain():
    cfg = default_config()
    cfg.phone.active_antennas = ("TL",)
    state = AntennaState(
        pattern_kind="parametric",
        parametric_patterns={"TL": ParametricPattern(azimuth_attenuation_db=6.0)},
    )
    cfg.phone.antenna_model.true_state = state
    model = build_antenna_model(cfg.phone, cfg.signal.carrier_hz)
    direction = Direction(90.0, 0.0)
    _, unrotated = antenna_response(model, direction, WavePolarization())
    state.orientations["TL"] = ElementOrientation(yaw_deg=90.0)
    _, rotated = antenna_response(model, direction, WavePolarization())
    ratio = abs(rotated["pattern_polarization_voltage"][0]) / abs(
        unrotated["pattern_polarization_voltage"][0]
    )
    np.testing.assert_allclose(ratio, 10**(3.0 / 20.0))


def test_positive_pcv_adds_receive_path_only_to_true_pattern():
    cfg = default_config()
    cfg.phone.active_antennas = ("TL",)
    cfg.phone.antenna_model.true_state = AntennaState(
        pattern_kind="parametric",
        parametric_patterns={
            "TL": ParametricPattern(phase_variation_mm=(10.0, 0.0, 0.0))
        },
    )
    cfg.phone.antenna_model.assumed_state = AntennaState(
        pattern_kind="parametric",
        parametric_patterns={"TL": ParametricPattern()},
    )
    model = build_antenna_model(cfg.phone, cfg.signal.carrier_hz)
    direction = Direction(0.0, 0.0)
    true_response, truth = antenna_response(model, direction, WavePolarization())
    estimated_response, estimate = antenna_response(
        model, direction, WavePolarization(), estimated=True
    )
    expected = np.exp(
        -2j * np.pi * 0.01 * cfg.signal.carrier_hz / SPEED_OF_LIGHT_MPS
    )
    np.testing.assert_allclose(true_response / estimated_response, [expected])
    np.testing.assert_allclose(truth["pcv_path_length_mm"], [10.0])
    np.testing.assert_allclose(truth["pcv_response"], [expected])
    np.testing.assert_allclose(estimate["pcv_path_length_mm"], [0.0])
    np.testing.assert_allclose(np.abs(true_response), np.abs(estimated_response))


def test_rotated_direction_pcv_is_applied_before_nontrivial_coupling():
    cfg = default_config()
    cfg.phone.active_antennas = ("TL", "TR")
    state = AntennaState(
        pattern_kind="parametric",
        parametric_patterns={"TL": ParametricPattern(), "TR": ParametricPattern()},
        orientations={"TL": ElementOrientation(yaw_deg=90), "TR": ElementOrientation()},
        coupling_paths=(
            CouplingPath("TR", "TL", -20, 40),
            CouplingPath("TL", "TR", -18, -20),
        ),
    )
    cfg.phone.antenna_model.true_state = state
    direction = Direction(40, 25)
    model = build_antenna_model(cfg.phone, cfg.signal.carrier_hz)
    _, baseline = antenna_response(model, direction, WavePolarization())
    coefficients = np.array([[2, -1, 0.75], [-1.5, 3, -0.5]])
    for name, values in zip(cfg.phone.active_antennas, coefficients):
        state.parametric_patterns[name].phase_variation_mm = tuple(values)

    # Independent Cartesian derivation: local TL axes have a 90° yaw, TR is unrotated.
    azimuth, elevation = np.deg2rad([40, 25])
    ux = np.cos(elevation) * np.cos(azimuth)
    uy = np.cos(elevation) * np.sin(azimuth)
    local_x = np.array([uy, ux])
    local_y = np.array([-ux, uy])
    paths_mm = (
        coefficients[:, 0] * local_x**2
        + coefficients[:, 1] * local_y**2
        + coefficients[:, 2] * local_x * local_y
    )
    phases = np.exp(-2j * np.pi * paths_mm * 1e-3 / cfg.signal.wavelength_m)
    coupling = np.array([
        [1, 0.1 * np.exp(1j * np.deg2rad(40))],
        [10**(-18 / 20) * np.exp(-1j * np.deg2rad(20)), 1],
    ])
    response, diagnostics = antenna_response(model, direction, WavePolarization())
    np.testing.assert_allclose(diagnostics["pcv_path_length_mm"], paths_mm)
    np.testing.assert_allclose(diagnostics["pcv_response"], phases)
    np.testing.assert_allclose(response, coupling @ (baseline["open_circuit_response"] * phases))


def test_blockage_phase_tracks_same_center_transition_and_outside_fraction():
    state = AntennaState(
        attenuation_regions=(
            AttenuationRegion(("TL",), 0.0, 0.0, 20.0, 12.0, 20.0, 60.0),
        )
    )
    for azimuth, fraction in ((0.0, 1.0), (20.0, 0.5), (40.0, 0.0)):
        voltage = attenuation_voltage(state, ("TL", "TR"), Direction(azimuth, 0.0))
        expected = 10**(-12.0 * fraction / 20.0) * np.exp(
            1j * np.deg2rad(60.0 * fraction)
        )
        np.testing.assert_allclose(voltage, [expected, 1.0])


def test_parametric_assumed_nested_values_are_independent():
    cfg = parametric_config()
    truth = cfg.phone.antenna_model.true_state
    assumed = cfg.phone.antenna_model.assumed_state
    original_gain = assumed.parametric_patterns["TL"].peak_gain_db
    original_yaw = assumed.orientations["TL"].yaw_deg
    original_tilt = (
        assumed.parametric_patterns["TL"].nominal_polarization.tilt_deg
    )
    original_coupling_phase = assumed.coupling_paths[0].phase_deg
    truth.parametric_patterns["TL"].peak_gain_db += 2.0
    truth.orientations["TL"].yaw_deg += 3.0
    truth.parametric_patterns["TL"].nominal_polarization.tilt_deg += 4.0
    truth.coupling_paths[0].phase_deg += 5.0
    assert assumed.parametric_patterns["TL"].peak_gain_db == original_gain
    assert assumed.orientations["TL"].yaw_deg == original_yaw
    assert (
        assumed.parametric_patterns["TL"].nominal_polarization.tilt_deg
        == original_tilt
    )
    assert assumed.coupling_paths[0].phase_deg == original_coupling_phase


def test_measured_loader_excludes_nearby_frequency_plane(tmp_path):
    path = tmp_path / "nearby.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "element_name", "frequency_hz", "azimuth_deg", "elevation_deg",
                "h_az_real", "h_az_imag", "h_el_real", "h_el_imag",
            ),
        )
        writer.writeheader()
        for frequency, value in ((1_575_420_000, 1.0), (1_575_420_100, 9.0)):
            writer.writerow(
                dict(
                    element_name="TL", frequency_hz=frequency,
                    azimuth_deg=0, elevation_deg=0, h_az_real=value,
                    h_az_imag=0, h_el_real=0, h_el_imag=0,
                )
            )
    grid = load_measured_pattern(path, 1_575_420_000)["TL"]
    np.testing.assert_allclose(grid.heights[0, 0], [1.0, 0.0])
