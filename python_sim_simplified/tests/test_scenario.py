
import numpy as np
import pytest

from core.config import CORNER_NAMES, Direction, WidebandInterferer
from core.gnss import generate_scenario
from core.presets import default_config


def test_scenario_public_keys_remain_stable():
    _, truth = generate_scenario(default_config())

    assert set(truth) == {
        "time_s",
        "code",
        "desired_waveform",
        "desired_component",
        "desired_steering_true",
        "desired_steering_assumed",
        "desired_open_circuit_response_true",
        "desired_open_circuit_response_assumed",
        "interferer_steering_effective",
        "interferer_components",
        "active_antennas",
        "positions_m",
        "channel_delay_true_ns",
        "channel_delay_estimated_ns",
        "channel_delay_true_samples",
        "channel_delay_estimated_samples",
        "desired_delay_phase_true",
        "desired_delay_phase_assumed",
        "interferer_delay_phases_true",
        "doppler_hz",
        "code_rate_hz",
        "code_delay_chips",
        "fs_rf_hz",
        "if_hz",
        "antenna_model_diagnostics",
    }


def test_no_interferers_keeps_scene_equal_to_direct_component():
    cfg = default_config()
    cfg.interferers = []

    scene, truth = generate_scenario(cfg)

    np.testing.assert_array_equal(scene, truth["desired_component"])


def test_scenario_contains_code_doppler_and_interferer_time_gate():
    cfg = default_config()
    cfg.signal.duration_ms = 8
    cfg.phone.active_antennas = ("TL", "BR")
    cfg.interferers[0].start_ms = 2.0
    cfg.interferers[0].stop_ms = 5.0

    samples, truth = generate_scenario(cfg)
    fs = cfg.signal.fs_rf_hz
    assert samples.shape == (2, round(fs * 0.008))
    assert truth["code_rate_hz"] == cfg.signal.code_rate_hz * (
        1.0 + cfg.signal.doppler_hz / cfg.signal.carrier_hz
    )
    jammer = truth["interferer_components"][0]
    assert np.all(jammer[:, : round(0.002 * fs)] == 0.0)
    assert np.any(jammer[:, round(0.002 * fs) : round(0.005 * fs)] != 0.0)
    assert np.all(jammer[:, round(0.005 * fs) :] == 0.0)


def test_channel_phase_is_present_before_rf():
    cfg = default_config()
    cfg.signal.duration_ms = 2
    cfg.interferers = []
    cfg.phone.active_antennas = ("TL", "TR")
    cfg.phone.channel_phase_error_deg["TR"] = 60.0

    _, truth = generate_scenario(cfg)
    phase = np.angle(
        truth["desired_steering_true"][1]
        / truth["desired_open_circuit_response_true"][1],
        deg=True,
    )
    assert np.isclose(phase, 60.0)


def test_estimated_delay_changes_assumed_steering_not_actual_samples():
    cfg = default_config()
    cfg.signal.duration_ms = 1
    cfg.phone.active_antennas = ("TL", "TR")
    cfg.interferers = []
    cfg.phone.channel_delay_ns["TR"] = 1e9 / cfg.signal.fs_rf_hz

    first, first_truth = generate_scenario(cfg)
    cfg.phone.estimated_channel_delay_ns["TR"] = 5.0
    second, second_truth = generate_scenario(cfg)

    np.testing.assert_allclose(first_truth["channel_delay_true_samples"], [0.0, 1.0])
    assert first_truth["desired_component"][1, 0] == 0.0j
    np.testing.assert_array_equal(second, first)
    np.testing.assert_array_equal(
        second_truth["desired_component"], first_truth["desired_component"]
    )
    assert not np.array_equal(
        second_truth["desired_steering_assumed"],
        first_truth["desired_steering_assumed"],
    )


def test_interferer_delay_phase_uses_its_rf_offset():
    cfg = default_config()
    cfg.signal.duration_ms = 1
    cfg.phone.active_antennas = ("TL", "TR")
    _, baseline_truth = generate_scenario(cfg)

    delay_ns = 10.0
    cfg.phone.channel_delay_ns["TR"] = delay_ns
    _, delayed_truth = generate_scenario(cfg)

    expected_phase = np.exp(
        -1j
        * 2.0
        * np.pi
        * (cfg.signal.carrier_hz + cfg.interferers[0].offset_hz)
        * np.array([0.0, delay_ns * 1e-9])
    )
    np.testing.assert_allclose(
        delayed_truth["interferer_delay_phases_true"][0], expected_phase
    )
    np.testing.assert_allclose(
        delayed_truth["interferer_steering_effective"][0],
        baseline_truth["interferer_steering_effective"][0] * expected_phase,
    )


def test_wideband_interferer_uses_time_gate_and_center_frequency_delay_phase():
    cfg = default_config()
    cfg.signal.duration_ms = 8
    cfg.phone.active_antennas = ("TL", "TR")
    cfg.phone.channel_delay_ns["TR"] = 10.0
    cfg.interferers = [
        WidebandInterferer(
            Direction(80.0, 10.0),
            center_offset_hz=100e3,
            bandwidth_hz=2.0e6,
            random_seed=1001,
            start_ms=2.0,
            stop_ms=5.0,
        )
    ]

    _, truth = generate_scenario(cfg)

    fs_hz = cfg.signal.fs_rf_hz
    jammer = truth["interferer_components"][0]
    start = round(0.002 * fs_hz)
    stop = round(0.005 * fs_hz)
    assert np.all(jammer[0, :start] == 0.0)
    assert np.any(jammer[:, start:stop] != 0.0)
    assert np.all(jammer[0, stop:] == 0.0)
    expected_phase = np.exp(
        -1j
        * 2.0
        * np.pi
        * (cfg.signal.carrier_hz + cfg.interferers[0].center_offset_hz)
        * np.array([0.0, 10.0e-9])
    )
    np.testing.assert_allclose(
        truth["interferer_delay_phases_true"][0],
        expected_phase,
    )


def test_default_scenario_contains_cw_then_wideband_components():
    cfg = default_config()
    cfg.signal.duration_ms = 2

    _, truth = generate_scenario(cfg)

    assert len(truth["interferer_components"]) == 2
    assert len(truth["interferer_steering_effective"]) == 2
    assert len(truth["interferer_delay_phases_true"]) == 2
