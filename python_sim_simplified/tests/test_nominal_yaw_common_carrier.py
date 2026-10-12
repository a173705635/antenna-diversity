import numpy as np
import pytest

from tools.spoofing import nominal_yaw_common_carrier as carrier


def fixture_path():
    time = np.array([.30, .31, .32, .33])
    tracking = dict(time=time, carrier=np.ones((4, 6)),
                    arc=np.ones((4, 6), int),
                    satellites=np.ones((4, 6, 3)))
    navigation = dict(short_position=np.column_stack((2*time,
                         np.zeros((4, 2)))), short_selected=np.ones((4, 6), bool))
    return tracking, navigation


def test_pair_mask_requires_two_selected_finite_same_arc():
    track, nav = fixture_path()
    nav['short_selected'][1, 0] = False
    track['carrier'][1, 1] = np.nan
    track['carrier'][0, 2] = np.nan
    track['arc'][1, 3] = 2
    np.testing.assert_array_equal(carrier.pair_mask(track, nav['short_selected'], 1),
                                  [False, False, False, False, True, True])


def test_common_intersection_changes_only_stars(monkeypatch):
    calls = []

    def fit(current, previous, change):
        calls.append(len(current))
        return np.array([.02, 0., 0., 0.])

    monkeypatch.setattr(carrier, 'fit_carrier_displacement', fit)
    tracks, navigations = {}, {}
    for yaw in (0., -2., 2.):
        tracks[yaw], navigations[yaw] = fixture_path()
    navigations[-2.]['short_selected'][:, 0] = False
    summary, arrays = carrier.compare(tracks, navigations)
    assert calls == [6]*3 + [5]*9 + [6]*3 + [5]*3
    assert summary['paths']['+0']['baseline_reconstruction_max_abs_m'] < 1e-12
    assert np.all(arrays['common_eligible_pairs'][:, 0] == 0)
    assert summary['paths']['+0']['candidate_star_min_median_max'] == [5., 5., 5.]


def test_one_paths_arc_change_removes_star_from_all_candidate_fits(monkeypatch):
    def fit(current, previous, change):
        return np.array([.02, 0., 0., 0.])

    monkeypatch.setattr(carrier, 'fit_carrier_displacement', fit)
    tracks, navigations = {}, {}
    for yaw in (0., -2., 2.):
        tracks[yaw], navigations[yaw] = fixture_path()
    tracks[-2.]['arc'][2, 0] = 2
    summary, arrays = carrier.compare(tracks, navigations)
    assert arrays['common_eligible_pairs'][1, 0]
    assert not arrays['common_eligible_pairs'][2, 0]
    assert not arrays['common_eligible_pairs'][3, 0]
    for yaw in ('+0', '-2', '+2'):
        assert arrays[f'{yaw}_candidate_stars'][2] == 5
        assert arrays[f'{yaw}_candidate_stars'][3] == 5
    assert summary['paths']['+0']['baseline_star_min_median_max'] == [6., 6., 6.]


def test_failed_fit_ends_continuous_replay_without_crossing_gap(monkeypatch):
    track, nav = fixture_path()
    calls = []

    def fit(current, previous, change):
        calls.append(len(current))
        return None

    monkeypatch.setattr(carrier, 'fit_carrier_displacement', fit)
    result = carrier.replay(track, nav,
                            carrier.pair_table(track, nav['short_selected']), 0)
    assert calls == [6]
    assert result['attempted'].tolist() == [False, True, False, False]
    assert np.isnan(result['position'][1:]).all()


@pytest.mark.local_evidence
def test_saved_baseline_reconstructs_and_inputs_are_verified():
    summary, arrays = carrier.analyze()
    assert len(summary['paths']) == 3
    assert all(path['baseline_reconstruction_max_abs_m'] <= 1e-12
               for path in summary['paths'].values())
    assert len(arrays['observation_time_s']) == 1984
