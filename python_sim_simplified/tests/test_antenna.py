import csv

import numpy as np

from antenna import (
    antenna_response,
    build_antenna_model,
    interpolate_measured,
    load_measured_pattern,
)
from config import (
    Direction,
    WavePolarization,
)
from presets import default_config, measured_config, parametric_config


def test_response_composes_open_circuit_coupling_and_channel_phase():
    cfg = default_config()
    cfg.phone.channel_phase_error_deg["TR"] = 90.0
    response, stages = antenna_response(
        build_antenna_model(cfg),
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
        build_antenna_model(cfg),
        Direction(225.0, 20.0),
        WavePolarization("rhcp"),
    )

    assert response.shape == (4,)
    assert np.all(stages["blockage_voltage"][[2, 3]] < 1.0)
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
        build_antenna_model(cfg),
        Direction(90.0, 45.0),
        WavePolarization("rhcp"),
    )
    assert response.shape == (4,)
    assert np.all(np.isfinite(response))
    assert stages["pattern_polarization_voltage"].shape == (4,)
