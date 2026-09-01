import numpy as np

from acquisition import acquire
from config import AcquisitionConfig, TrackingConfig
from gnss import ca_code
from tracking import estimate_cn0_db_hz, track


def receiver_signal(
    fs_hz=4.092e6,
    duration_ms=30,
    doppler_hz=1250.0,
    delay_chips=175.0,
):
    n = np.arange(round(fs_hz * duration_ms * 1e-3))
    phase = np.mod(n * 1.023e6 / fs_hz - delay_chips, 1023.0)
    code = ca_code(1)[np.floor(phase).astype(int)]
    clean = code * np.exp(1j * 2.0 * np.pi * doppler_hz * n / fs_hz)
    rng = np.random.default_rng(11)
    noise = 0.2 * (
        rng.standard_normal(n.size) + 1j * rng.standard_normal(n.size)
    )
    return clean + noise


def test_acquisition_and_tracking_follow_valid_signal():
    fs_hz = 4.092e6
    samples = receiver_signal(fs_hz)

    acquisition = acquire(samples, fs_hz, 1, 1.023e6, AcquisitionConfig())
    tracking = track(
        samples,
        fs_hz,
        1,
        acquisition,
        TrackingConfig(max_epochs=20),
        1_575.42e6,
        1.023e6,
    )

    assert acquisition["success"]
    assert abs(acquisition["doppler_hz"] - 1250.0) <= 250.0
    assert tracking["success"]
    assert tracking["prompt"].size > 5
    assert np.isfinite(tracking["cn0_db_hz"][-1])


def test_cn0_matches_coherent_to_noise_power_ratio():
    prompt = np.array([10 + 1j, 10 - 1j, 10 + 1j, 10 - 1j])
    expected = 10.0 * np.log10(100.0 / (1.0 * 0.001))

    assert np.isclose(estimate_cn0_db_hz(prompt, 0.001), expected)
