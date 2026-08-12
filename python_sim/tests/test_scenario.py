import numpy as np
import pytest

from antenna_diversity.config import default_config
from antenna_diversity.gnss.scenario import generate_scenario


def test_scenario_projects_signal_through_both_antennas():
    """Catch loss of code Doppler, array response, or antennas-by-samples shape."""
    cfg = default_config()
    cfg.jammer.enable = False

    x, truth = generate_scenario(cfg)

    expected_samples = round(cfg.signal.fs_rf_hz * cfg.signal.duration_ms * 1e-3)
    assert x.shape == (cfg.array.num_elements, expected_samples)
    expected_rate = cfg.signal.code_rate_hz * (
        1 + cfg.signal.doppler_hz / cfg.signal.carrier_hz
    )
    assert truth["code_rate_hz"] == pytest.approx(expected_rate, abs=1e-9)
    ratio = np.mean(x[1] / x[0])
    expected_ratio = (
        truth["desired_steering_effective"][1]
        / truth["desired_steering_effective"][0]
    )
    assert ratio == pytest.approx(expected_ratio, abs=1e-10)


def test_channel_amplitude_and_phase_errors_modify_each_antenna_response():
    """Catch omission of per-channel complex gain errors from scenario generation."""
    cfg = default_config()
    cfg.jammer.enable = False
    cfg.array.channel_amplitude_error_db = np.array([0.0, 6.020599913279624])
    cfg.array.channel_phase_error_deg = np.array([0.0, 90.0])

    _, truth = generate_scenario(cfg)

    np.testing.assert_allclose(truth["channel_response"], [1.0, 2.0j], atol=1e-12)
