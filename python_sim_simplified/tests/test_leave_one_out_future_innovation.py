import numpy as np
import pytest

from tools.spoofing import leave_one_out_future_innovation as innovation


def geometry():
    a = np.array([[1., 0., 0.], [-1., 0., 0.],
                  [0., 1., 0.], [0., -1., 0.],
                  [0., 0., 1.], [0., 0., -1.],
                  [.7, .4, .5]])
    return np.column_stack((a, np.ones(len(a))))


def test_clock_and_motion_consistent_changes_are_innovation_blind():
    h = geometry()
    motion_and_clock = np.array([.03, -.02, .01, .4])
    scores = innovation.leave_one_out_scores(h, h@motion_and_clock, .01)
    assert np.nanmax(scores) < 1e-11
    np.testing.assert_allclose(scores,
        innovation.leave_one_out_scores(h, np.zeros(len(h)), .01), atol=1e-11)


def test_one_prn_outlier_has_largest_innovation_without_residual_prescreen():
    h = geometry()
    y = h@np.array([.03, -.02, .01, .4])
    y[3] += .8
    scores = innovation.leave_one_out_scores(h, y, .01)
    assert np.isfinite(scores).all()
    assert int(np.argmax(scores)) == 3
    assert scores[3] > 10.


def test_same_arc_and_finite_pair_qualification():
    z = dict(carrier=np.ones((2, 7)), arc=np.ones((2, 7), int))
    valid = np.ones((2, 7), bool)
    valid[0, 0] = False
    z['carrier'][1, 1] = np.nan
    z['arc'][1, 2] = 2
    np.testing.assert_array_equal(innovation.pair_mask(z, valid, 0, 1),
        [False, False, False, True, True, True, True])


def test_future_boundary_is_counted_before_any_score(monkeypatch):
    time = np.array([9.99, 10., 10.01, 10.02, 10.03])
    z = dict(time=time)
    selected = np.ones((5, 7), bool)
    called = []

    def current(z, selected, k):
        called.append(k)
        return np.arange(7., dtype=float), 'current_scored', 7

    def future(z, selected, k, scoreable, velocity):
        return np.ones(7), 'future_scored', 7

    monkeypatch.setattr(innovation, 'current_scores', current)
    monkeypatch.setattr(innovation, 'future_truth_error', future)
    summary, arrays = innovation.analyze_observations(z, selected, 2.)
    assert summary['candidate_opportunities'] == 4
    assert summary['primary_opportunities'] == 1
    assert summary['reasons'] == {'constant_rank': 1,
                                  'future_out_of_record': 3}
    assert summary['current_star_min_median_max'] == [7., 7., 7.]
    assert summary['scoreable_star_min_median_max'] == [7., 7., 7.]
    assert summary['future_common_min_median_max'] == [7., 7., 7.]
    assert summary['current_pair_computed'] == 1
    assert summary['future_common_computed'] == 1
    assert called == [1]
    assert arrays['reason'][4] == 'future_out_of_record'
    assert not arrays['current_pair_computed'][2:].any()
    assert not arrays['future_common_computed'][2:].any()


def test_uncomputed_future_count_is_null_but_computed_zero_is_kept(monkeypatch):
    z = dict(time=np.array([9.99, 10., 10.01, 10.02, 10.03]))
    selected = np.ones((5, 7), bool)

    def current(z, selected, k):
        return np.full(7, np.nan), 'current_fewer_than_six', 0

    monkeypatch.setattr(innovation, 'current_scores', current)
    summary, arrays = innovation.analyze_observations(z, selected, 2.)
    assert summary['current_star_min_median_max'] == [0., 0., 0.]
    assert summary['scoreable_star_min_median_max'] is None
    assert summary['future_common_min_median_max'] is None
    assert summary['current_pair_computed'] == 1
    assert summary['future_common_computed'] == 0
    assert not arrays['future_common_computed'].any()

    def scored_current(z, selected, k):
        return np.arange(7., dtype=float), 'current_scored', 7

    def empty_future(z, selected, k, scoreable, velocity):
        return np.full(7, np.nan), 'future_fewer_than_six', 0

    monkeypatch.setattr(innovation, 'current_scores', scored_current)
    monkeypatch.setattr(innovation, 'future_truth_error', empty_future)
    summary, arrays = innovation.analyze_observations(z, selected, 2.)
    assert summary['future_common_min_median_max'] == [0., 0., 0.]
    assert summary['future_common_computed'] == 1
    assert arrays['future_common_computed'][1]


def test_average_tie_ranks_and_constant_rank_are_explicit():
    current = np.array([1., 1., 2., 3., 4., 4., 5.])
    future = np.array([4., 4., 3., 2., 1., 1., 0.])
    assert innovation.rank_correlation(current, future) == pytest.approx(-1.)
    assert innovation.rank_correlation(np.ones(7), future) is None
    assert innovation.rank_correlation(current, np.ones(7)) is None


def test_current_score_is_prefix_causal_and_has_no_truth_argument(monkeypatch):
    h = geometry()
    y = h@np.array([.01, .02, -.01, .3])
    time = np.arange(7)*.01
    z = dict(time=time, carrier=np.ones((7, 7)),
             arc=np.ones((7, 7), int))
    selected = np.ones((7, 7), bool)

    def measurement(z, first, last, mask):
        assert last <= 2
        return h[mask], y[mask], .01

    monkeypatch.setattr(innovation, 'pair_measurement', measurement)
    full, status, _ = innovation.current_scores(z, selected, 2)
    prefix = {name: value[:3] for name, value in z.items()}
    early, early_status, _ = innovation.current_scores(prefix, selected[:3], 2)
    np.testing.assert_array_equal(full, early)
    assert status == early_status == 'current_scored'


@pytest.mark.local_evidence
def test_fixed_source_receipts_match_current_dependencies():
    import json
    from pathlib import Path

    root = Path(innovation.__file__).resolve().parents[1]
    stage47 = innovation.study.OUTPUT/'summary.json'
    continuity = (innovation.study.OUTPUT
                  /'single_path_carrier_continuity_summary.json')
    assert innovation.digest(stage47) == innovation.STAGE47_SUMMARY_SHA256
    assert innovation.digest(continuity) == innovation.CONTINUITY_SUMMARY_SHA256
    frozen = json.loads(stage47.read_text(encoding='utf-8'))
    receipt = json.loads(continuity.read_text(encoding='utf-8'))
    assert innovation.digest(root/'tools/single_path_carrier_continuity.py') == (
        receipt['source_sha256'])
    for source in innovation.replay.NAVIGATION_SOURCES:
        assert innovation.digest(root/source) == frozen['source_sha256'][source]


@pytest.mark.local_evidence
def test_saved_gate_three_baseline_reproduces_before_scoring():
    import json

    frozen = json.loads((innovation.study.OUTPUT/'summary.json').read_text())
    case = frozen['cases']['2026100404']['weak']
    folder = innovation.study.OUTPUT/'2026100404'/'weak'/'gate3'
    summary, arrays = innovation.analyze_path(
        folder, case['paths']['gate3']['output_sha256'], 2.)
    assert summary['baseline_reproduced']
    assert summary['candidate_opportunities'] > 0
    assert arrays['current_score_mps'].shape[1] == 24
