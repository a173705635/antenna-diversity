from types import FunctionType

import numpy as np
import pytest

from tools.spoofing import single_path_carrier_continuity as continuity


def observations(count=5):
    time = np.arange(count)*.01
    return dict(time=time, carrier=np.ones((count, 6)),
                arc=np.ones((count, 6), int))


def test_three_epoch_boundary_and_original_qualifications():
    z = observations()
    valid = np.ones((5, 6), bool)
    assert not continuity.three_epoch_mask(z, valid, 0).any()
    assert not continuity.three_epoch_mask(z, valid, 1).any()
    valid[0, 0] = False
    z['carrier'][1, 1] = np.nan
    z['arc'][0, 2] = 2
    np.testing.assert_array_equal(continuity.three_epoch_mask(z, valid, 2),
                                  [False, False, False, True, True, True])
    z['time'][3] += .001
    assert not continuity.three_epoch_mask(z, valid, 3).any()


def test_only_current_row_is_limited_and_lag_one_two_remain(monkeypatch):
    z = observations(4)
    valid = np.ones((4, 6), bool)
    valid[0, 0] = False
    seen = []

    def bridge(observations, bridge_valid, positions, k):
        seen.append(bridge_valid.copy())
        # Both old rows must remain available to the original lag search.
        return np.array([float(bridge_valid[k-1].sum()),
                         float(bridge_valid[k-2].sum()), 0.]), {}

    def fake_navigate(observations, **options):
        positions = np.zeros((len(observations['time']), 3))
        for k in range(2, len(positions)):
            phase_prediction_leave_one_out(observations, valid, positions, k)
        return dict(position=positions)

    # The fake navigate resolves this global through its own function globals.
    monkeypatch.setitem(globals(), 'phase_prediction_leave_one_out', bridge)
    monkeypatch.setattr(continuity.single_source, 'navigate', fake_navigate)
    monkeypatch.setattr(continuity.single_source,
                        'phase_prediction_leave_one_out', bridge)
    before = valid.copy()
    original_globals = fake_navigate.__globals__.copy()
    _, detail = continuity.candidate_navigate(z, np.ones(4, bool))
    np.testing.assert_array_equal(valid, before)
    assert fake_navigate.__globals__ == original_globals
    assert not seen[0][2, 0]
    assert seen[0][1].all()
    assert seen[0][0, 0] == 0
    assert seen[1][3].all()
    assert detail['called'].tolist() == [False, False, True, True]


def test_private_function_without_replacement_matches_original(monkeypatch):
    def fake_navigate(value, **options):
        return phase_prediction_leave_one_out(value)

    def bridge(value):
        return value+1

    monkeypatch.setitem(globals(), 'phase_prediction_leave_one_out', bridge)
    clone = FunctionType(fake_navigate.__code__, dict(fake_navigate.__globals__),
                         fake_navigate.__name__, fake_navigate.__defaults__)
    assert clone(3) == fake_navigate(3) == 4
    assert clone.__globals__ is not fake_navigate.__globals__


def test_original_phase_fit_retains_lag_two_fallback(monkeypatch):
    from core.spoofing import mechanism_phase

    z = observations(3)
    z['satellites'] = np.ones((3, 6, 3))
    valid = np.ones((3, 6), bool)
    positions = np.zeros((3, 3))
    calls = []

    def fit(current, previous, change, return_residual=False):
        calls.append(len(current))
        if len(calls) == 1:
            return None, np.full(len(current), np.nan)
        return np.array([.04, 0., 0., 0.]), np.zeros(len(current))

    monkeypatch.setattr(mechanism_phase, 'fit_carrier_displacement', fit)
    restricted = valid.copy()
    restricted[2] = continuity.three_epoch_mask(z, valid, 2)
    predicted, detail = mechanism_phase.phase_prediction_leave_one_out(
        z, restricted, positions, 2)
    assert calls == [6, 6]
    assert detail['lag_epochs'] == 2
    np.testing.assert_array_equal(predicted, [.04, 0., 0.])


def test_prefix_admission_is_causal():
    z = observations(7)
    valid = np.ones((7, 6), bool)
    z['arc'][5, 0] = 2
    full = [continuity.three_epoch_mask(z, valid, k)
            for k in range(len(z['time']))]
    prefix = {key: value[:5] for key, value in z.items()}
    early = [continuity.three_epoch_mask(prefix, valid[:5], k)
             for k in range(5)]
    np.testing.assert_array_equal(full[:5], early)


def test_recovery_attempts_below_ten_do_not_become_a_recovery_exit():
    time = np.arange(5)*.01
    navigation = dict(initialization_time=.02,
                      diagnostics=dict(isolated=np.array([False, True, True,
                                                          True, True]),
                                       recovery_confirmed=np.array([
                                           False, True, True, False, False])))
    events = continuity.event_times(time, time+.02, navigation)
    episode = events['isolation_episodes'][0]
    assert episode['confirmation_attempt_available_s'] == pytest.approx([.03, .04])
    assert episode['exit_index'] is None
    assert episode['unresolved_at_record_end']
    assert episode['elapsed_s'] == pytest.approx(.03)


def test_ten_confirmations_and_isolation_exit_are_reported_separately():
    time = np.arange(12)*.01
    isolated = np.array([False]+[True]*9+[False, False])
    confirmed = np.array([False]+[True]*10+[False])
    navigation = dict(initialization_time=.02,
                      diagnostics=dict(isolated=isolated,
                                       recovery_confirmed=confirmed))
    episode = continuity.event_times(time, time+.02, navigation)[
        'isolation_episodes'][0]
    assert len(episode['confirmation_attempt_available_s']) == 10
    assert episode['exit_index'] == 10
    assert episode['exit_available_s'] == pytest.approx(.12)
    assert episode['elapsed_s'] == pytest.approx(.09)
    assert not episode['unresolved_at_record_end']


def test_available_and_observation_thresholds_are_distinct():
    time = np.array([.28, .29, .30, .31])
    available = time+.02
    position = np.zeros((4, 3))
    counts = continuity.time_axis_counts(time, available, position, position)
    assert counts['score_navigation_available'] == dict(
        opportunities=4, baseline_published=4,
        candidate_published=4, common_published=4)
    assert counts['long_screen_observation'] == dict(
        opportunities=2, baseline_published=2,
        candidate_published=2, common_published=2)


@pytest.mark.local_evidence
def test_saved_gate_three_short_baseline_reproduces_every_field():
    import json

    summary = json.loads((continuity.study.OUTPUT/'summary.json').read_text())
    path = summary['cases']['2026100404']['weak']['paths']['gate3']
    folder = continuity.study.OUTPUT/'2026100404'/'weak'/'gate3'
    result, arrays = continuity.run_path(
        folder, path['output_sha256'], 'weak', 2.)
    assert result['baseline_reproduced']
    assert len(arrays['time']) == 1984
    from experiments.spoofing.single_source_long_coherent import navigation_prefix
    from experiments.spoofing.single_source_phase_state_gate import projection_eligibility
    from experiments.spoofing.single_source_temporal_trigger import navigation_input

    tracking = continuity.read_npz(folder/'tracking.npz')
    trace = continuity.read_npz(folder/'trace.npz')
    support = continuity.read_npz(folder/'support.npz')['coherent_support']
    eligible, _, _ = projection_eligibility(tracking['available_time'], trace)
    prefix = navigation_prefix(navigation_input(tracking, support), 100)
    early, bridge = continuity.candidate_navigate(prefix, eligible[:100])
    np.testing.assert_array_equal(early['position'], arrays['candidate_position'][:100])
    np.testing.assert_array_equal(bridge['admission'], arrays['bridge_admission'][:100])
