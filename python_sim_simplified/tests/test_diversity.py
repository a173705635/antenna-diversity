import numpy as np
import pytest

from core.config import BlindNullConfig
from core.diversity import combine_blind_null, combine_known, combine_lcmv


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


def test_lcmv_preserves_los_and_nulls_resolvable_multipath():
    rng = np.random.default_rng(9)
    channels = rng.normal(size=(4, 2000)) + 1j * rng.normal(size=(4, 2000))
    los = np.array([1.0, 1.0j, -1.0, -1.0j])
    echo = np.array([1.0, -1.0, 1.0j, -1.0j])

    combined, info = combine_lcmv(channels, los, [echo], 1e-2, 0.95)

    assert combined.shape == (channels.shape[1],)
    assert np.vdot(info["weights"], los) == pytest.approx(1.0 + 0.0j, abs=1e-10)
    assert np.vdot(info["weights"], echo) == pytest.approx(0.0 + 0.0j, abs=1e-10)
    assert info["suppression_status"] == "multipath_constraints_applied"


def test_lcmv_without_echo_constraints_matches_loaded_mvdr():
    rng = np.random.default_rng(12)
    channels = rng.normal(size=(3, 1000)) + 1j * rng.normal(size=(3, 1000))
    los = np.array([1.0, np.exp(0.4j), np.exp(-0.8j)])

    expected, expected_info = combine_known(channels, los, "mvdr", 1e-2)
    actual, actual_info = combine_lcmv(channels, los, [], 1e-2, 0.95)

    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(actual_info["weights"], expected_info["weights"])
    assert actual_info["mode"] == "multipath_lcmv"
    assert actual_info["suppression_status"] == "no_multipath_detected"


def test_lcmv_rejects_constraint_collinear_with_los():
    rng = np.random.default_rng(14)
    channels = rng.normal(size=(3, 1000)) + 1j * rng.normal(size=(3, 1000))
    los = np.array([1.0, 1.0j, -1.0])
    collinear = 2.0j * los

    _, info = combine_lcmv(channels, los, [collinear], 1e-2, 0.95)

    assert info["rejected_constraint_indices"] == [0]
    assert info["suppression_status"] == "spatially_unresolvable"
    assert np.vdot(info["weights"], los) == pytest.approx(1.0 + 0.0j, abs=1e-10)
