import numpy as np
from navigation_monitors import monitor_scores, first_evidence


def fixture(n=400):
    t = np.arange(n+1)*.1; zero = np.zeros((n+1, 3))
    ins = dict(time_s=t, position_m=zero.copy(), velocity_mps=zero.copy())
    provisional = dict(state=np.zeros((n+1, 9)), innovation=np.zeros((n, 6)))
    return ins, zero[1:].copy(), zero[1:].copy(), provisional


def test_fixed_reference_detects_smooth_departure_and_is_prefix_causal():
    ins, p, v, base = fixture(); u = np.maximum(ins['time_s']-20, 0)
    p[:, 0] = .025*u[1:]**2; v[:, 0] = .05*u[1:]
    s = monitor_scores(ins, p, v, base)
    assert abs(s[5, -1]-1.) < 1e-12
    assert abs(s[6, -1]-10.) < 1e-12
    short = monitor_scores({k:a[:301] for k,a in ins.items()}, p[:300], v[:300],
                           dict(state=base['state'][:301], innovation=base['innovation'][:300]))
    np.testing.assert_array_equal(short, s[:, :300])


def test_constant_inertial_acceleration_error_is_removed_by_reference_estimate():
    ins, p, v, base = fixture(); t = ins['time_s']
    ins['position_m'][:, 0] = .02*t*t/2; ins['velocity_mps'][:, 0] = .02*t
    base['state'][:, 6] = .02
    scores = monitor_scores(ins, p, v, base)
    np.testing.assert_allclose(scores[5:], 0., atol=1e-12)


def test_confirmation_never_uses_a_future_sample_or_overrides_prior_isolation():
    score = np.zeros(200); score[110:113] = 2.; limit = np.ones(200)
    prior = np.zeros(201, bool)
    assert first_evidence(score, limit, 3, prior) == 112
    assert first_evidence(score[:112], limit[:112], 3, prior[:113]) is None
    prior[112:] = True
    assert first_evidence(score, limit, 3, prior) is None
