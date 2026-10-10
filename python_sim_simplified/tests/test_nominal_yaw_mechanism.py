import numpy as np
import pytest

from tools import nominal_yaw_mechanism as mechanism


def test_anchor_decomposition_keeps_vector_and_cross_term():
    time = np.array([0., .2, .3, .4])
    position = np.array([[np.nan, np.nan, np.nan],
                         [1.4, 1., 0.], [1.6, 2., 0.], [2.8, 3., 0.]])
    result, arrays = mechanism.decompose_short(time, position)

    assert result['anchor_index'] == 1
    assert result['score_count'] == 2
    assert result['anchor_error_m'] == pytest.approx(np.hypot(1., 1.))
    assert result['max_vector_identity_residual_m'] < 1e-12
    assert result['max_square_identity_residual_m2'] < 1e-12
    assert result['max_error_index'] == 3
    np.testing.assert_allclose(arrays['anchor'], [1., 1.])
    np.testing.assert_allclose(arrays['drift'][3], [1., 2.])
    assert arrays['cross'][3] == pytest.approx(6.)
    assert arrays['error_squared'][3] == pytest.approx(13.)


def test_projection_joins_last_complete_block_and_selected_prns():
    time = np.array([.30, .31, .32, .33])
    position = np.array([[.6, 0., 0.], [.62, .1, 0.],
                         [.64, .2, 0.], [.66, .3, 0.]])
    _, arrays = mechanism.decompose_short(time, position)
    latest = np.array([0, 1, 2, 2])
    raw = np.array([2., 4., 5.])
    selected = np.array([[True, True], [True, False],
                         [False, True], [True, True]])
    retained = np.array([[1., 1.], [.4, .9], [.5, .8]])

    result = mechanism.gate_state_association(time, latest, selected,
                                              raw, retained, arrays)

    assert result['epochs_by_state'] == {'inactive': 1, 'active': 3}
    assert result['selected_entries_by_state'] == {'inactive': 2, 'active': 4}
    assert result['inactive_by_prn']['1']['median'] == pytest.approx(1.)
    assert result['active_by_prn']['1']['count'] == 2
    assert result['active_by_prn']['1']['minimum'] == pytest.approx(.4)
    assert result['active_by_prn']['2']['count'] == 2
    assert result['active_by_prn']['2']['median'] == pytest.approx(.8)
    assert result['active_epoch_mean_retention'][1] == pytest.approx(.4)
    assert result['active_epoch_mean_retention'][2] == pytest.approx(.8)
    assert result['active_epoch_mean_retention'][3] == pytest.approx(.65)


def test_projection_excludes_unpublished_and_pre_score_epochs():
    time = np.array([.28, .3, .31])
    position = np.array([[.56, 0., 0.], [np.nan]*3, [.62, 0., 0.]])
    _, arrays = mechanism.decompose_short(time, position)
    result = mechanism.gate_state_association(
        time, np.array([0, 0, 0]), np.ones((3, 2), bool),
        np.array([4.]), np.array([[.2, .8]]), arrays)
    assert result['epochs_by_state'] == {'inactive': 0, 'active': 1}
    assert result['selected_entries_by_state'] == {'inactive': 0, 'active': 2}
    assert result['active_by_prn']['1']['count'] == 1


def test_evidence_window_exact_boundary_uses_previous_block():
    time = np.array([.30])
    position = np.array([[.6, 0., 0.]])
    _, arrays = mechanism.decompose_short(time, position)
    raw = np.zeros(16)
    raw[14] = 4.
    raw[15] = 2.
    retained = np.ones((16, 2))
    retained[14] = [.25, .75]
    result = mechanism.evidence_window_association(
        time, np.ones((1, 2), bool), raw, retained, arrays)
    assert result['epochs_by_state'] == {'inactive': 0, 'mixed': 0, 'active': 1}
    assert result['active_by_prn']['1']['median'] == pytest.approx(.25)
    assert result['active_fraction'][0] == pytest.approx(1.)


def test_evidence_window_half_block_weights_retention_and_activity():
    time = np.array([.31])
    position = np.array([[.62, 0., 0.]])
    _, arrays = mechanism.decompose_short(time, position)
    raw = np.zeros(16)
    raw[14] = 4.
    raw[15] = 2.
    retained = np.ones((16, 2))
    retained[14] = [.2, .8]
    retained[15] = [.6, 1.]
    result = mechanism.evidence_window_association(
        time, np.ones((1, 2), bool), raw, retained, arrays)
    assert result['epochs_by_state'] == {'inactive': 0, 'mixed': 1, 'active': 0}
    assert result['active_fraction'][0] == pytest.approx(.5)
    assert result['mixed_by_prn']['1']['median'] == pytest.approx(.4)
    assert result['mixed_by_prn']['2']['median'] == pytest.approx(.9)


def test_evidence_window_20_second_endpoint_uses_final_block():
    time = np.array([20.])
    position = np.array([[40., 0., 0.]])
    _, arrays = mechanism.decompose_short(time, position)
    raw = np.zeros(1000)
    raw[999] = 4.
    retained = np.ones((1000, 1))
    retained[999, 0] = .375
    result = mechanism.evidence_window_association(
        time, np.ones((1, 1), bool), raw, retained, arrays)
    assert result['epochs_by_state'] == {'inactive': 0, 'mixed': 0, 'active': 1}
    assert result['active_by_prn']['1']['median'] == pytest.approx(.375)


def test_constant_retention_has_no_spurious_correlation():
    time = np.array([.30, .31, .32])
    position = np.array([[.6, 0., 0.], [.62, .1, 0.], [.64, .2, 0.]])
    _, arrays = mechanism.decompose_short(time, position)
    result = mechanism.evidence_window_association(
        time, np.ones((3, 1), bool), np.zeros(17),
        np.ones((17, 1)), arrays)
    assert result['epoch_mean_retention'] == [1., 1., 1.]
    assert result['correlations_by_state']['inactive']['retention_absolute_error'] is None
    assert result['correlations_by_state']['inactive']['retention_drift'] is None


def test_no_published_drift_extrapolation():
    time = np.array([.27, .28, .6])
    position = np.array([[.54, 0., 0.], [.56, 0., 0.], [np.nan]*3])
    result, arrays = mechanism.decompose_short(time, position)
    assert result['score_count'] == 0
    assert np.isnan(arrays['drift'][2]).all()
    assert result['drift_max_m'] is None


def test_plan_sources_must_match_fixed_recovery_summary():
    plan = {'source_sha256': {'experiments/single_source_nominal_yaw.py': 'old'}}
    summary = {'frozen_source_sha256': {'experiments/single_source_nominal_yaw.py': 'old'}}
    mechanism.assert_frozen_source_identity(plan, summary)
    summary['frozen_source_sha256']['experiments/single_source_nominal_yaw.py'] = 'new'
    with pytest.raises(AssertionError):
        mechanism.assert_frozen_source_identity(plan, summary)
