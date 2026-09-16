import numpy as np
from scipy.spatial.transform import Rotation
from temporal_spatial_monitor import temporal_scores


def sample(n):
    t = np.arange(n)*.01
    r = Rotation.from_rotvec(np.column_stack((t*0, t*0, t*.2))).as_matrix()
    x = np.random.default_rng(11).normal(0, .02, (n, 4, 8))
    return x, r


def test_fixed_channel_prn_offsets_cancel_for_all_lags():
    x, r = sample(1000); bias = np.random.default_rng(12).normal(0, .02, (4, 8))
    a, gate = temporal_scores(x, r, .19); b, other = temporal_scores(x+bias, r, .19)
    np.testing.assert_allclose(a, b, atol=1e-14, rtol=0)
    np.testing.assert_array_equal(gate, other)


def test_stationary_scene_is_not_treated_as_spoof_evidence():
    x, r = sample(1000); r[:] = np.eye(3)
    s, gate = temporal_scores(x, r, .19)
    assert not gate.any()
    assert (s == -1).all()


def test_time_difference_and_gate_are_prefix_causal():
    x, r = sample(1000)
    whole, gate = temporal_scores(x, r, .19)
    prefix, pg = temporal_scores(x[:800], r[:800], .19)
    np.testing.assert_allclose(prefix, whole[:, :80], atol=1e-14, rtol=0)
    np.testing.assert_array_equal(pg, gate[:, :80])


def test_half_cycle_wrapping_does_not_break_cancellation_but_drift_does():
    x, r = sample(1000); wavelength = .19
    fixed = np.random.default_rng(14).normal(0, .02, (4, 8))
    wrapped = (x+fixed+wavelength/4) % (wavelength/2)-wavelength/4
    a, _ = temporal_scores(x, r, wavelength)
    b, _ = temporal_scores(wrapped, r, wavelength)
    np.testing.assert_allclose(a, b, atol=1e-14, rtol=0)
    drift = x+np.arange(1000)[:, None, None]*.01*fixed
    c, _ = temporal_scores(drift, r, wavelength)
    assert np.max(abs(c-a)) > .01
