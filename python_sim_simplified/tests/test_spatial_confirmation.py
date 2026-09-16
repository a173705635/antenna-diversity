import numpy as np
from core.spatial_spoofing import confirm_windows, spatial_scores, detect_windows


def windows(states):
    return [dict(decision_s=.091+.1*k, state=s) for k, s in enumerate(states)]


def test_confirmation_is_causal_and_requires_consecutive_evidence():
    raw = windows([1, 1, 0, 1, 1, 1, 0])
    confirmed = confirm_windows(raw, 3)
    assert [r['state'] for r in confirmed] == [0, 0, 0, 0, 0, 1, 0]
    assert confirm_windows(raw[:5], 3) == confirmed[:5]
    assert [r['state'] for r in raw] == [1, 1, 0, 1, 1, 1, 0]


def test_unknown_and_time_gap_break_evidence_run():
    raw = windows([1, 1, -1, 1, 1, 1])
    raw[-1]['decision_s'] += .1
    result = confirm_windows(raw, 3)
    assert [r['state'] for r in result] == [0, 0, -1, 0, 0, 0]
    assert result[-1]['evidence_run'] == 1


def test_one_window_preserves_original_decisions():
    raw = windows([0, 1, -1, 1, 0])
    assert [r['state'] for r in confirm_windows(raw, 1)] == [r['state'] for r in raw]


def test_regular_stream_batch_scores_match_causal_window_consumer():
    rng = np.random.default_rng(4)
    carrier = rng.normal(0, .03, (40, 4, 8))
    batch = spatial_scores(carrier.reshape(4, 10, 4, 8), .19)['cluster3']
    rows = detect_windows(np.arange(40)*.01, carrier, np.ones_like(carrier, bool), .1, .19)
    np.testing.assert_allclose(batch, [r['score'] for r in rows], rtol=0, atol=1e-15)
    np.testing.assert_allclose([r['decision_s'] for r in rows], .091+np.arange(4)*.1)
