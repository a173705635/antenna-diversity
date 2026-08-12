import numpy as np
import pytest

from antenna_diversity.array.steering import steering_vector
from antenna_diversity.config import default_config
from antenna_diversity.diversity.combiner import combine


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
