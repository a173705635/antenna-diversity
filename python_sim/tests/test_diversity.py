import numpy as np
import pytest

from antenna_diversity.array.steering import steering_vector
from antenna_diversity.config import default_config, mate70_default_config
from antenna_diversity.diversity.combiner import combine, compute_weights


def deterministic_array_case(cfg):
    a = steering_vector(
        cfg.array.positions_m,
        cfg.array.desired_angle_deg,
        cfg.signal.wavelength_m,
    )
    b = steering_vector(
        cfg.array.positions_m,
        cfg.jammer.angle_deg,
        cfg.signal.wavelength_m,
    )
    rng = np.random.Generator(np.random.MT19937(11))
    n_samples = 20_000
    desired = (
        rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
    ) / np.sqrt(2)
    jammer = (
        rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
    ) / np.sqrt(2)
    x = a[:, None] * desired[None, :] + 20 * b[:, None] * jammer[None, :]
    return a, b, x


@pytest.mark.parametrize("mode", ["single", "egc", "mvdr"])
def test_combiner_has_unit_desired_response(mode):
    """Catch conjugation or normalization errors in each branch's weights."""
    cfg = default_config()
    cfg.diversity.diagonal_loading_factor = 1e-4
    a, _, x = deterministic_array_case(cfg)

    y, info = combine(x, a, mode, cfg.diversity)

    assert y.shape == (x.shape[1],)
    assert info["desired_response"] == pytest.approx(1.0 + 0j, abs=1e-10)


def test_mvdr_reduces_jammer_direction_gain_by_tenfold():
    """Catch covariance, loading, or solve errors that remove the spatial null."""
    cfg = default_config()
    cfg.diversity.diagonal_loading_factor = 1e-4
    a, b, x = deterministic_array_case(cfg)

    _, single = combine(x, a, "single", cfg.diversity)
    _, mvdr = combine(x, a, "mvdr", cfg.diversity)

    single_gain = abs(np.vdot(single["weights"], b))
    mvdr_gain = abs(np.vdot(mvdr["weights"], b))
    assert mvdr_gain < 0.1 * single_gain


def test_combiner_rejects_unknown_mode():
    """Catch silent fallback to an unintended combining algorithm."""
    cfg = default_config()
    a, _, x = deterministic_array_case(cfg)

    with pytest.raises(ValueError, match="unknown diversity mode"):
        combine(x, a, "mrc", cfg.diversity)


@pytest.mark.parametrize("num_antennas", [2, 3, 4])
@pytest.mark.parametrize("mode", ["single", "egc", "mvdr"])
def test_combiner_supports_two_to_four_channels(num_antennas, mode):
    """Catch fixed-size weights or incorrect conjugation for phone subarrays."""

    rng = np.random.default_rng(7)
    desired = np.exp(1j * np.arange(num_antennas) * 0.31)
    samples = rng.standard_normal(
        (num_antennas, 4000)
    ) + 1j * rng.standard_normal((num_antennas, 4000))

    output, info = combine(
        samples,
        desired,
        mode,
        mate70_default_config().diversity,
    )

    assert output.shape == (4000,)
    assert np.vdot(info["weights"], desired) == pytest.approx(
        1.0 + 0j, abs=1e-10
    )


@pytest.mark.parametrize(
    ("covariance", "desired", "message"),
    [
        (np.ones((2, 3)), np.ones(2), "square"),
        (np.eye(2), np.ones(3), "one value per antenna"),
        (np.array([[1.0, np.nan], [0.0, 1.0]]), np.ones(2), "finite"),
        (np.eye(2), np.zeros(2), "nonzero"),
    ],
)
def test_compute_weights_rejects_invalid_constraints(
    covariance, desired, message
):
    """Catch malformed covariance or steering inputs before a linear solve."""

    with pytest.raises(ValueError, match=message):
        compute_weights(
            covariance,
            desired,
            "mvdr",
            mate70_default_config().diversity,
        )
