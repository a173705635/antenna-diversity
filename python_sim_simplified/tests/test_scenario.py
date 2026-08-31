import numpy as np

from config import default_config
from gnss import generate_scenario


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
