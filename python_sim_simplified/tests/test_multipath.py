import inspect

import numpy as np
import pytest

from core.antenna import antenna_response, build_antenna_model
from core.channel_delay import fractional_delay
from core.config import Direction
from core.gnss import ca_code
from core.multipath.multipath import detect_multipath, evaluate_multipath, project_out_los
from core.presets import default_config


def _acquisition(fs_hz):
    samples_per_ms = round(fs_hz * 1e-3)
    return {
        "success": True,
        "prn": 1,
        "doppler_hz": 0.0,
        "code_phase_samples": 0,
        "peak": 10.0,
        "second_peak": 1.0,
        "second_peak_code_phase_samples": 20,
        "exclusion_samples": 4,
        "metric": 10.0,
        "doppler_bins_hz": np.array([0.0]),
        "search_power": np.ones((1, samples_per_ms)),
        "samples_per_ms": samples_per_ms,
        "coherent_samples": samples_per_ms,
    }


def _array_code_scene(include_echo=True, same_direction=False):
    cfg = default_config()
    fs_hz = 4.092e6
    count = round(4e-3 * fs_hz)
    n = np.arange(count)
    code = ca_code(cfg.signal.prn)
    local = code[np.floor(np.mod(n * cfg.signal.code_rate_hz / fs_hz, 1023.0)).astype(int)]
    model = build_antenna_model(cfg.phone, cfg.signal.carrier_hz)
    los, _ = antenna_response(
        model,
        cfg.signal.assumed_direction,
        cfg.signal.assumed_polarization,
        estimated=True,
    )
    channels = los[:, None] * local[None, :]
    if include_echo:
        direction = cfg.signal.assumed_direction if same_direction else Direction(110.0, 15.0)
        echo, _ = antenna_response(
            model,
            direction,
            cfg.signal.assumed_polarization,
            estimated=True,
        )
        delayed = fractional_delay(local.astype(complex), 2.0)
        channels = channels + 0.7 * echo[:, None] * delayed[None, :]
    return cfg, fs_hz, model, los, channels


def test_los_projection_removes_only_the_los_spatial_component():
    los = np.array([1.0, 1.0j, -1.0, -1.0j])
    echo = np.array([1.0, -1.0, 1.0j, -1.0j])
    vectors = los[:, None] * np.array([[4.0, 2.0]]) + echo[:, None]

    residual = project_out_los(vectors, los)

    np.testing.assert_allclose(los.conj() @ residual, 0.0, atol=1e-12)
    assert np.linalg.norm(residual) > 0.0


def test_detector_interface_cannot_receive_scenario_truth():
    assert "truth" not in inspect.signature(detect_multipath).parameters


def test_detector_finds_separated_half_chip_echo_and_direction():
    cfg, fs_hz, model, _, channels = _array_code_scene()
    cfg.multipath_detection.detection_threshold_db = 3.0

    result = detect_multipath(
        channels,
        fs_hz,
        cfg.signal.prn,
        cfg.signal.code_rate_hz,
        _acquisition(fs_hz),
        cfg.acquisition,
        model,
        cfg.signal.assumed_direction,
        cfg.signal.assumed_polarization,
        cfg.multipath_detection,
        cfg.multipath_suppression.max_steering_correlation,
    )

    assert result["status"] == "detected"
    candidate = result["candidates"][0]
    assert candidate["excess_delay_chips"] == pytest.approx(0.5, abs=0.25)
    assert candidate["spatial_status"] == "resolvable"
    assert candidate["direction"].azimuth_deg == pytest.approx(110.0, abs=5.0)
    assert candidate["direction"].elevation_deg == pytest.approx(15.0, abs=5.0)


def test_los_only_input_has_no_multipath_peak():
    cfg, fs_hz, model, _, channels = _array_code_scene(include_echo=False)
    cfg.multipath_detection.detection_threshold_db = 20.0

    result = detect_multipath(
        channels,
        fs_hz,
        cfg.signal.prn,
        cfg.signal.code_rate_hz,
        _acquisition(fs_hz),
        cfg.acquisition,
        model,
        cfg.signal.assumed_direction,
        cfg.signal.assumed_polarization,
        cfg.multipath_detection,
        cfg.multipath_suppression.max_steering_correlation,
    )

    assert result["status"] == "no_multipath_detected"
    assert result["candidates"] == []


def test_same_direction_echo_is_not_reported_as_resolvable():
    cfg, fs_hz, model, _, channels = _array_code_scene(same_direction=True)

    result = detect_multipath(
        channels,
        fs_hz,
        cfg.signal.prn,
        cfg.signal.code_rate_hz,
        _acquisition(fs_hz),
        cfg.acquisition,
        model,
        cfg.signal.assumed_direction,
        cfg.signal.assumed_polarization,
        cfg.multipath_detection,
        cfg.multipath_suppression.max_steering_correlation,
    )

    assert not any(
        candidate["spatial_status"] == "resolvable"
        for candidate in result["candidates"]
    )


def test_truth_evaluation_counts_match_miss_and_false_alarm():
    detection = {
        "candidates": [
            {
                "excess_delay_chips": 0.55,
                "direction": Direction(105.0, 20.0),
            },
            {
                "excess_delay_chips": 2.8,
                "direction": Direction(15.0, 5.0),
            },
        ]
    }
    truth_paths = [
        {
            "excess_delay_chips": 0.5,
            "direction": Direction(100.0, 20.0),
        },
        {
            "excess_delay_chips": 1.5,
            "direction": Direction(200.0, 30.0),
        },
    ]

    evaluation = evaluate_multipath(detection, truth_paths, 0.5, 20.0)

    assert evaluation["detected_count"] == 1
    assert evaluation["missed_count"] == 1
    assert evaluation["false_alarm_count"] == 1
    assert evaluation["matches"][0]["delay_error_chips"] == pytest.approx(0.05)
    assert evaluation["matches"][0]["direction_error_deg"] == pytest.approx(4.7, abs=0.3)
