"""One-second carrier anchor remains causal and rejects ambiguous exclusions."""
import numpy as np

from core import single_source
from core.mechanism_phase import phase_prediction_long_anchor
from test_single_source_code_blend import patch_geometry, record
from test_single_source_phase_leave_one_out import phase_record


def long_record(stars=8):
    short, _, _, displacement = phase_record(stars)
    n = 101
    record = dict(time=np.arange(n) * .01,
                  satellites=np.broadcast_to(short['satellites'][0],
                                             (n, stars, 3)).copy(),
                  carrier=np.zeros((n, stars)),
                  arc=np.zeros((n, stars), int))
    record['carrier'][-1] = short['carrier'][2]
    valid = np.ones((n, stars), bool)
    positions = np.full((n, 3), np.nan)
    positions[0] = [1., 2., 3.]
    return record, valid, positions, displacement


def test_real_one_second_geometry_and_clock_select_unique_prn():
    record, valid, positions, displacement = long_record()
    record['carrier'][-1, 4] += .8
    prediction, detail = phase_prediction_long_anchor(record, valid, positions,
                                                       100, short_excluded_prn=0)
    np.testing.assert_allclose(prediction, positions[0] + displacement, atol=1e-8)
    assert detail['anchor_index'] == 0
    assert detail['elapsed_s'] == 1.
    assert detail['excluded_prn'] == 5
    assert detail['reason'] == 'used'
    assert detail['full_common_count'] == 8
    assert np.nanmax(abs(detail['full_residual_m'])) > .1
    assert np.nanmax(abs(detail['candidate_residual_m'])) < 1e-8


def test_index_100_alone_is_not_a_one_second_anchor():
    for last_time in (.996, 1.004, 1.02):
        record, valid, positions, _ = long_record()
        record['time'][-1] = last_time
        record['carrier'][-1, 4] += .8
        prediction, detail = phase_prediction_long_anchor(record, valid, positions,
                                                           100, short_excluded_prn=0)
        assert prediction is None
        assert detail['reason'] == 'no_anchor'


def test_floating_point_time_rounding_still_matches_exact_one_second():
    record, valid, positions, _ = long_record()
    record['time'][-1] = 1. + 1e-15
    record['carrier'][-1, 4] += .8
    prediction, detail = phase_prediction_long_anchor(record, valid, positions,
                                                       100, short_excluded_prn=0)
    assert prediction is not None
    assert abs(detail['elapsed_s'] - 1.) < 1e-12


def test_unpublished_or_different_arc_anchor_and_five_common_fail():
    record, valid, positions, _ = long_record()
    record['carrier'][-1, 4] += .8
    positions[0] = np.nan
    assert phase_prediction_long_anchor(record, valid, positions, 100, 0)[1]['reason'] == 'no_anchor'
    positions[0] = [1., 2., 3.]
    record['arc'][-1] += 1
    assert phase_prediction_long_anchor(record, valid, positions, 100, 0)[1]['reason'] == 'too_few'
    record['arc'][-1] -= 1
    valid[-1, 5:] = False
    assert phase_prediction_long_anchor(record, valid, positions, 100, 0)[1]['reason'] == 'too_few'


def test_long_full_pass_does_not_override_short_prediction():
    record, valid, positions, _ = long_record()
    prediction, detail = phase_prediction_long_anchor(record, valid, positions,
                                                       100, short_excluded_prn=6)
    assert prediction is None
    assert detail['reason'] == 'full_pass'


def test_multiple_exclusions_and_short_prn_conflict_keep_short_path():
    record, valid, positions, _ = long_record(stars=6)
    record['carrier'][-1, 0] -= .8
    record['carrier'][-1, 1] -= .72
    prediction, detail = phase_prediction_long_anchor(record, valid, positions,
                                                       100, short_excluded_prn=0)
    assert prediction is None
    assert detail['reason'] == 'ambiguous'
    record, valid, positions, _ = long_record()
    record['carrier'][-1, 4] += .8
    prediction, detail = phase_prediction_long_anchor(record, valid, positions,
                                                       100, short_excluded_prn=6)
    assert prediction is None
    assert detail['reason'] == 'short_conflict'
    prediction, detail = phase_prediction_long_anchor(record, valid, positions,
                                                       100, short_excluded_prn=0)
    assert prediction is not None
    assert detail['reason'] == 'used'


def test_future_samples_do_not_change_prior_long_prediction():
    record, valid, positions, _ = long_record()
    record['carrier'][-1, 4] += .8
    predicted, detail = phase_prediction_long_anchor(record, valid, positions,
                                                      100, short_excluded_prn=0)
    record['time'] = np.r_[record['time'], 1.01]
    record['carrier'] = np.vstack((record['carrier'], np.full(8, 1000.)))
    record['arc'] = np.vstack((record['arc'], np.zeros(8, int)))
    record['satellites'] = np.concatenate((record['satellites'],
                                           record['satellites'][-1:]))
    valid = np.vstack((valid, np.ones(8, bool)))
    positions = np.vstack((positions, np.full(3, np.nan)))
    again, repeated = phase_prediction_long_anchor(record, valid, positions,
                                                    100, short_excluded_prn=0)
    np.testing.assert_array_equal(again, predicted)
    assert repeated['anchor_index'] == detail['anchor_index']


def test_contaminated_published_anchor_is_carried_forward_without_oracle():
    record, valid, positions, _ = long_record()
    record['carrier'][-1, 4] += .8
    clean, _ = phase_prediction_long_anchor(record, valid, positions, 100, 0)
    positions[0] += [5., -2., 0.]
    contaminated, _ = phase_prediction_long_anchor(record, valid, positions, 100, 0)
    np.testing.assert_allclose(contaminated-clean, [5., -2., 0.], atol=1e-9)


def test_navigation_long_path_uses_own_state_and_keeps_admission(monkeypatch):
    patch_geometry(monkeypatch)
    monkeypatch.setattr(single_source, 'phase_prediction_leave_one_out',
                        lambda z, valid, positions, k: (
                            np.array([10., 0., 0.]),
                            dict(lag_epochs=1, excluded_prn=0,
                                 full_common_count=np.array([6, 6]),
                                 full_residual_m=np.zeros((2, 6)),
                                 candidate_common_count=0,
                                 candidate_residual_m=np.full(6, np.nan))))
    seen_anchors = []

    def long_prediction(z, valid, positions, k, short_excluded_prn):
        seen_anchors.append(positions.copy())
        detail = dict(anchor_index=0, elapsed_s=1., excluded_prn=3,
                      reason='used' if k == 3 else 'no_anchor',
                      full_common_count=6,
                      full_residual_m=np.full(6, .2),
                      candidate_common_count=5,
                      candidate_residual_m=np.zeros(6),
                          candidate_prediction_m=np.array([11., 0., 4.]))
        return (np.array([11., 0., 4.]) if k == 3 else None), detail

    monkeypatch.setattr(single_source, 'phase_prediction_long_anchor', long_prediction)
    observations = record(5)
    baseline = single_source.navigate(observations, diagnostics=True,
                                      code_blend=0., phase_leave_one_out=True)
    candidate = single_source.navigate(observations, diagnostics=True,
                                       code_blend=0., phase_long_anchor=True)
    np.testing.assert_array_equal(candidate['selected'], baseline['selected'])
    np.testing.assert_allclose(candidate['candidate_position'][3, 0], 11.)
    np.testing.assert_allclose(baseline['candidate_position'][3, 0], 10.)
    assert candidate['diagnostics']['phase_long_path'][3] == 'long'
    np.testing.assert_allclose(candidate['diagnostics']['phase_long_jump_m'][3],
                               np.sqrt(17.))
    assert candidate['diagnostics']['phase_long_horizontal_jump_m'][3] == 1.
    np.testing.assert_array_equal(seen_anchors[4][3], candidate['position'][3])
    assert seen_anchors[4][3, 0] == 11.
    assert baseline['position'][3, 0] == 10.
