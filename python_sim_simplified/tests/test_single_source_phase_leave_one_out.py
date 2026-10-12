"""The optional phase bridge excludes one consistently implicated carrier."""
import numpy as np

from core.gps_simulation import receiver_reference
from core.spoofing.mechanism_phase import (fit_carrier_displacement,
                                  phase_prediction_leave_one_out)
from core.spoofing.recovery_geometry import sky
from core.spoofing.single_source import navigate


def phase_record(stars=8):
    reference, rotation = receiver_reference()
    directions = sky(7713, stars)
    satellites = reference + (directions * 21e6) @ rotation
    time = np.array([0., .01, .02])
    record = dict(time=time, available_time=time + .02,
                  satellites=np.broadcast_to(satellites, (3, stars, 3)).copy(),
                  carrier=np.zeros((3, stars)), arc=np.zeros((3, stars), int))
    positions = np.full((3, 3), np.nan)
    positions[:2] = [[1., 2., 3.], [1., 2., 3.]]
    valid = np.ones((3, stars), bool)
    displacement = np.array([.03, -.02, .01])
    clock = .4
    design = np.column_stack((-directions, np.ones(stars)))
    record['carrier'][2] = design @ np.r_[displacement, clock]
    return record, valid, positions, displacement


def test_single_outlier_has_unique_same_prn_across_both_real_geometry_lags():
    record, valid, positions, displacement = phase_record()
    record['carrier'][2, 4] += .8
    prediction, details = phase_prediction_leave_one_out(record, valid, positions, 2)
    np.testing.assert_allclose(prediction, positions[1] + displacement, atol=1e-8)
    assert details['lag_epochs'] == 1
    assert details['excluded_prn'] == 5
    assert details['candidate_common_count'] == 7
    assert details['full_common_count'].tolist() == [8, 8]
    assert np.isfinite(details['full_residual_m'][:, 4]).all()
    assert np.isnan(details['candidate_residual_m'][4])
    assert np.nanmax(abs(details['candidate_residual_m'])) < .1


def test_full_fit_at_older_lag_wins_before_any_recent_leave_one_out():
    record, valid, positions, displacement = phase_record()
    record['carrier'][1, 4] = .8
    # The latest full fit sees an outlier in the previous carrier; the older
    # full fit uses the original zero carrier and remains valid.
    prediction, details = phase_prediction_leave_one_out(record, valid, positions, 2)
    assert details['lag_epochs'] == 2
    assert details['excluded_prn'] == 0
    assert prediction is not None


def test_arc_change_or_missing_published_anchor_cannot_bridge():
    record, valid, positions, _ = phase_record()
    record['carrier'][2, 4] += .8
    record['arc'][2] += 1
    prediction, details = phase_prediction_leave_one_out(record, valid, positions, 2)
    assert prediction is None
    assert details['excluded_prn'] == 0
    record['arc'][2] -= 1
    positions[:2] = np.nan
    prediction, _ = phase_prediction_leave_one_out(record, valid, positions, 2)
    assert prediction is None


def test_two_plausible_exclusions_are_rejected_with_real_geometry():
    record, valid, positions, _ = phase_record(stars=6)
    record['carrier'][2, 0] -= .8
    record['carrier'][2, 1] -= .72
    passing = []
    for dropped in range(6):
        retained = np.arange(6) != dropped
        fit = fit_carrier_displacement(record['satellites'][2, retained],
                                       record['satellites'][1, retained],
                                       record['carrier'][2, retained])
        if fit is not None:
            passing.append(dropped)
    assert passing == [0, 2]
    prediction, details = phase_prediction_leave_one_out(record, valid, positions, 2)
    assert prediction is None
    assert details['excluded_prn'] == 0
    assert details['full_common_count'].tolist() == [6, 6]


def test_different_prns_across_lags_are_rejected():
    record, valid, positions, _ = phase_record()
    record['carrier'][2, 5] += .8
    record['carrier'][1, 5] += .8
    record['carrier'][1, 4] += .8
    first_only = positions.copy()
    first_only[0] = np.nan
    _, recent = phase_prediction_leave_one_out(record, valid, first_only, 2)
    assert recent['excluded_prn'] == 5
    older_only = positions.copy()
    older_only[1] = np.nan
    _, older = phase_prediction_leave_one_out(record, valid, older_only, 2)
    assert older['excluded_prn'] == 6
    prediction, both = phase_prediction_leave_one_out(record, valid, positions, 2)
    assert prediction is None
    assert both['excluded_prn'] == 0


def test_gap_excludes_older_anchor():
    record, valid, positions, _ = phase_record()
    record['carrier'][2, 4] += .8
    record['time'][0] = -.03
    prediction, details = phase_prediction_leave_one_out(record, valid, positions, 2)
    assert prediction is not None
    assert details['full_common_count'][1] == 0


def test_five_common_carriers_cannot_leave_one_out():
    record, valid, positions, _ = phase_record(stars=5)
    record['carrier'][2, 4] += 10.
    prediction, details = phase_prediction_leave_one_out(record, valid, positions, 2)
    assert prediction is None
    assert details['full_common_count'].tolist() == [5, 5]
    assert details['excluded_prn'] == 0


def test_future_carrier_cannot_change_earlier_candidate():
    record, valid, positions, _ = phase_record()
    record['carrier'][1, 4] += .8
    before, _ = phase_prediction_leave_one_out(record, valid, positions, 1)
    record['carrier'][2] += 100.
    after, _ = phase_prediction_leave_one_out(record, valid, positions, 1)
    np.testing.assert_array_equal(before, after)


def test_navigation_off_path_does_not_add_diagnostics(monkeypatch):
    from core.spoofing import single_source
    from test_single_source_code_blend import patch_geometry, record

    patch_geometry(monkeypatch)
    monkeypatch.setattr(single_source, 'phase_prediction',
                        lambda *args: np.array([10., 0., 0.]))
    baseline = navigate(record(4), diagnostics=True, code_blend=0.)
    explicit = navigate(record(4), diagnostics=True, code_blend=0.,
                        phase_leave_one_out=False)
    for key in baseline:
        if key == 'diagnostics':
            assert baseline[key].keys() == explicit[key].keys()
            for diagnostic in baseline[key]:
                np.testing.assert_array_equal(baseline[key][diagnostic],
                                              explicit[key][diagnostic])
        else:
            np.testing.assert_array_equal(baseline[key], explicit[key])


def test_navigation_enabled_records_exclusion_without_changing_admission(monkeypatch):
    from core.spoofing import single_source
    from test_single_source_code_blend import patch_geometry, record

    patch_geometry(monkeypatch)
    monkeypatch.setattr(single_source, 'phase_prediction_leave_one_out',
                        lambda z, valid, positions, k: (
                            np.array([10., 0., 0.]),
                            dict(lag_epochs=1, excluded_prn=3,
                                 full_common_count=np.array([6, 6]),
                                 full_residual_m=np.full((2, 6), .2),
                                 candidate_common_count=5,
                                 candidate_residual_m=np.zeros(6))))
    observations = record(4)
    baseline = navigate(observations, diagnostics=True, code_blend=0.)
    candidate = navigate(observations, diagnostics=True, code_blend=0.,
                         phase_leave_one_out=True)
    np.testing.assert_array_equal(candidate['selected'], baseline['selected'])
    np.testing.assert_array_equal(candidate['diagnostics']['support_count'],
                                  baseline['diagnostics']['support_count'])
    np.testing.assert_allclose(candidate['candidate_position'][:, 0], 10.)
    assert (candidate['diagnostics']['phase_excluded_prn'] == 3).all()
