import numpy as np
from core.config import BlindNullConfig
from core.diversity import combine_known
from core.interference.combining import combine_blind_null


def test_known_combiners_have_unit_target_response():
    channels = np.array([[1, 2, 3, 4], [1j, 2j, 1j, 3j]], dtype=complex)
    desired = np.array([1.0, np.exp(1j * 0.4)])

    for mode in ("single", "egc", "mvdr"):
        _, info = combine_known(channels, desired, mode, 1e-2)
        assert np.isclose(np.vdot(info["weights"], desired), 1.0)


def test_blind_null_uses_fixed_channel_phase_and_suppresses_correlated_mode():
    rng = np.random.default_rng(7)
    tone = np.exp(1j * 2.0 * np.pi * 0.03 * np.arange(4096))
    phase_rad = np.deg2rad(55.0)
    steering = np.array([1.0, 0.8 * np.exp(1j * phase_rad)])
    channels = steering[:, None] * tone[None, :]
    channels += 0.02 * (
        rng.standard_normal(channels.shape) + 1j * rng.standard_normal(channels.shape)
    )

    combined, info = combine_blind_null(
        channels, BlindNullConfig(detection_threshold_db=3.0)
    )

    assert np.isclose(
        np.angle(info["sample_covariance"][0, 1]), -phase_rad, atol=0.03
    )
    assert info["interference_detected"]
    assert info["nulling_enabled"]
    assert abs(np.vdot(info["weights"], steering)) < 0.05
    assert np.mean(np.abs(combined) ** 2) < 0.1 * np.mean(np.abs(channels[0]) ** 2)


def test_blind_null_returns_reference_when_detection_threshold_is_not_met():
    rng = np.random.default_rng(19)
    channels = rng.standard_normal((3, 2048)) + 1j * rng.standard_normal((3, 2048))

    combined, info = combine_blind_null(
        channels, BlindNullConfig(detection_threshold_db=20.0, reference_channel=1)
    )

    np.testing.assert_array_equal(combined, channels[1])
    assert not info["interference_detected"]
    assert not info["nulling_enabled"]
