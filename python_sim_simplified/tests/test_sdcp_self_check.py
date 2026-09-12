import numpy as np
from sdcp_self_check import detect, SelfCheckConfig


def run(r, **kw):
    return detect(np.arange(len(r))*.01, r, .19, .004, valid=kw.get('valid', True),
                  geometry_uncertainty_m=kw.get('uncertainty', 0.), channel_skew_s=0.,
                  config=SelfCheckConfig(window=20, block=10, confirmation=3))


def test_startup_static_anomaly_not_fitted_away():
    r = np.zeros((80, 8)); r[:, 0] = .025
    o = run(r)
    assert (o['state'][:19] == -1).all()
    assert (o['state'][21:, 0] == 1).all()
    assert (o['state'][21:, 1:] == 0).all()


def test_common_offset_is_unknown_not_trusted_clean():
    o = run(np.full((80, 8), .025))
    assert (o['state'] == -1).all()
    assert (o['reason'][19:] == 'common_ambiguity').all()


def test_invalid_samples_require_whole_new_window():
    r = np.zeros((80, 8)); r[35, 0] = np.nan
    o = run(r)
    # The first sample after the gap has no valid phase increment either.
    assert (o['state'][35:56, 0] == -1).all()
    assert o['state'][56, 0] == 0


def test_prefix_invariance_and_integer_wrap():
    r = np.zeros((80, 8)); r[60:, 0] = .025
    a, b = run(r), run(r[:50])
    np.testing.assert_array_equal(a['state'][:50], b['state'])
    assert (run(r+.19)['state'] == a['state']).all()


def test_geometry_uncertainty_and_lock_flags_gate():
    r = np.zeros((80, 8))
    assert (run(r, uncertainty=.01)['state'] == -1).all()
    assert (run(r, valid=False)['state'] == -1).all()


def test_half_cycle_requires_external_revalidation():
    r = np.zeros((100, 8)); r[30:, 0] = .095
    assert (run(r)['state'][30:, 0] == -1).all()
