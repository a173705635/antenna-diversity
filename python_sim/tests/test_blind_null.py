import numpy as np
import pytest

from antenna_diversity.config import BlindNullConfig
from antenna_diversity.diversity.blind_null import combine_blind_null


def complex_noise(rng, shape):
    return (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    ) / np.sqrt(2.0)


def test_independent_noise_bypasses_exact_reference_channel():
    rng = np.random.default_rng(120)
    x = complex_noise(rng, (3, 40_000))
    cfg = BlindNullConfig(detection_threshold_db=6.0, reference_channel=1)

    y, info = combine_blind_null(x, cfg)

    np.testing.assert_array_equal(y, x[1])
    np.testing.assert_array_equal(info["weights"], [0.0, 1.0, 0.0])
    assert not info["interference_detected"]
    assert not info["nulling_enabled"]
    assert info["bypass_reason"] == "no_interference_detected"


def test_zero_input_power_bypasses_exact_reference_channel():
    x = np.zeros((3, 16), dtype=np.complex128)
    cfg = BlindNullConfig(reference_channel=2)

    y, info = combine_blind_null(x, cfg)

    np.testing.assert_array_equal(y, x[2])
    np.testing.assert_array_equal(info["weights"], [0.0, 0.0, 1.0])
    assert not info["nulling_enabled"]
    assert info["bypass_reason"] == "zero_input_power"


def test_degenerate_interference_vector_bypasses_exact_reference_channel():
    x = np.array([
        [1.0 + 2.0j, -3.0 + 4.0j, 5.0 - 6.0j, -7.0 - 8.0j],
        [0.0, 0.0, 0.0, 0.0],
        [2.0 - 1.0j, 4.0 + 3.0j, -6.0 + 5.0j, 8.0 - 7.0j],
    ])
    cfg = BlindNullConfig(reference_channel=2)

    y, info = combine_blind_null(x, cfg)

    np.testing.assert_array_equal(y, x[2])
    np.testing.assert_array_equal(info["weights"], [0.0, 0.0, 1.0])
    assert not info["nulling_enabled"]
    assert info["bypass_reason"] == "degenerate_interference_vector"


@pytest.mark.parametrize("channels", [2, 3, 4])
def test_dominant_correlated_interferer_is_detected_and_nulled(channels):
    rng = np.random.default_rng(121 + channels)
    snapshots = 20_000
    steering = np.exp(1j * np.arange(channels) * 0.47)
    jammer = complex_noise(rng, snapshots)
    noise = 0.05 * complex_noise(rng, (channels, snapshots))
    x = 20.0 * steering[:, None] * jammer[None, :] + noise

    y, info = combine_blind_null(x, BlindNullConfig())

    assert info["interference_detected"]
    assert info["nulling_enabled"]
    assert info["detection_score_db"] >= 6.0
    assert np.vdot(info["weights"], np.eye(channels)[:, 0]) == pytest.approx(1.0)
    assert abs(np.vdot(info["weights"], info["estimated_interference_vector"])) < 1e-10
    assert np.mean(abs(y) ** 2) < 0.01 * np.mean(abs(x[0]) ** 2)


def test_fixed_channel_phase_offsets_are_observed_from_samples():
    rng = np.random.default_rng(130)
    phase = np.deg2rad([0.0, 37.0, -81.0, 145.0])
    steering = np.exp(1j * phase)
    jammer = complex_noise(rng, 30_000)
    x = 50.0 * steering[:, None] * jammer[None, :] + 0.02 * complex_noise(
        rng, (4, 30_000)
    )

    _, info = combine_blind_null(x, BlindNullConfig())

    assert info["nulling_enabled"]
    assert abs(np.vdot(info["weights"], steering)) < 1e-3


def test_unsafe_reference_projection_bypasses_without_large_weights():
    rng = np.random.default_rng(131)
    jammer = complex_noise(rng, 20_000)
    x = np.vstack([
        100.0 * jammer,
        0.01 * jammer + 1e-8 * complex_noise(rng, 20_000),
    ])

    y, info = combine_blind_null(
        x,
        BlindNullConfig(reference_channel=0, min_reference_projection_power=0.01),
    )

    np.testing.assert_array_equal(y, x[0])
    assert not info["nulling_enabled"]
    assert info["bypass_reason"] == "unsafe_reference_projection"
    assert np.linalg.norm(info["weights"]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("samples", "message"),
    [
        (np.ones(8, dtype=np.complex128), "two-dimensional"),
        (np.ones((1, 8), dtype=np.complex128), "at least two"),
        (np.empty((2, 0), dtype=np.complex128), "nonempty"),
        (
            np.array([[1.0 + 0j, np.nan + 0j], [1.0 + 0j, 1.0 + 0j]]),
            "finite",
        ),
    ],
)
def test_blind_null_rejects_invalid_sample_matrices(samples, message):
    with pytest.raises(ValueError, match=message):
        combine_blind_null(samples, BlindNullConfig())


def test_blind_null_rejects_out_of_range_reference_channel():
    with pytest.raises(ValueError, match="reference channel"):
        combine_blind_null(
            np.ones((2, 8), dtype=np.complex128),
            BlindNullConfig(reference_channel=2),
        )
