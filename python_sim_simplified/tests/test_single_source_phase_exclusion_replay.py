"""Small orchestration checks for the saved-observation exclusion study."""
import hashlib
from zipfile import ZipFile

import numpy as np
import pytest

from core import single_source
from core.mechanism_phase import phase_prediction_leave_one_out
from experiments.single_source_carrier_fault import compare_fault
from experiments.single_source_phase_exclusion import (
    exclusion_geometry, fault_window_mask, fit_geometry,
    replay_pair, verify_source_snapshot,
)
from test_single_source_code_blend import patch_geometry, record
from test_single_source_phase_leave_one_out import phase_record


def saved_navigation(navigation):
    saved = {}
    for key, value in navigation.items():
        if key == 'diagnostics':
            for diagnostic, data in value.items():
                saved[f'0_diagnostic_{diagnostic}'] = data.copy()
        else:
            saved[f'0_{key}'] = np.nan if value is None else value.copy() if isinstance(value, np.ndarray) else value
    return saved


def test_replay_uses_independent_states_and_exact_off_reference(monkeypatch):
    patch_geometry(monkeypatch)
    monkeypatch.setattr(single_source, 'phase_prediction',
                        lambda *args: np.array([10., 0., 0.]))
    monkeypatch.setattr(single_source, 'phase_prediction_leave_one_out',
                        lambda z, valid, positions, k: (
                            np.array([11., 0., 0.]),
                            dict(lag_epochs=1, excluded_prn=3,
                                 full_common_count=np.array([6, 6]),
                                 full_residual_m=np.full((2, 6), .2),
                                 candidate_common_count=5,
                                 candidate_residual_m=np.zeros(6))))
    observations = record(5)
    baseline = single_source.navigate(observations, diagnostics=True,
                                      code_blend=0.)
    off, on = replay_pair(observations, saved_navigation(baseline), '0_')
    np.testing.assert_array_equal(off['position'], baseline['position'])
    np.testing.assert_array_equal(off['selected'], on['selected'])
    assert (on['diagnostics']['phase_excluded_prn'] == 3).all()
    assert not np.array_equal(off['position'], on['position'])


def test_replay_rejects_changed_saved_off_result(monkeypatch):
    patch_geometry(monkeypatch)
    monkeypatch.setattr(single_source, 'phase_prediction', lambda *args: None)
    observations = record(4)
    saved = saved_navigation(single_source.navigate(observations,
                                                    diagnostics=True, code_blend=0.))
    saved['0_diagnostic_reason'][3] = 'wrong'
    with pytest.raises(AssertionError):
        replay_pair(observations, saved, '0_')


def test_old_snapshot_checks_its_own_hashes_without_current_source(tmp_path):
    content = b'old frozen algorithm bytes'
    archive = tmp_path / 'snapshot.zip'
    with ZipFile(archive, 'w') as saved:
        saved.writestr('core/mechanism_phase.py', content)
    hashes = {'core/mechanism_phase.py': hashlib.sha256(content).hexdigest()}
    verify_source_snapshot(archive, hashes)
    with pytest.raises(AssertionError):
        verify_source_snapshot(archive, {'core/mechanism_phase.py': '0' * 64})


def test_independent_geometry_shows_full_outlier_and_clean_exclusion():
    observations, _, _, _ = phase_record()
    observations['carrier'][2, 4] += .8
    full = fit_geometry(observations, 2, 1, np.arange(8))
    retained = np.arange(8)[np.arange(8) != 4]
    candidate = fit_geometry(observations, 2, 1, retained)
    assert full['condition'] < 1000.
    assert full['max_residual_m'] > .1
    assert candidate['condition'] < 1000.
    assert candidate['horizontal_std_m'] <= .5
    assert candidate['max_residual_m'] < 1e-8


def test_used_exclusion_geometry_matches_recorded_complex_fit_residuals():
    observations, valid, positions, _ = phase_record()
    observations['carrier'][2, 4] += .8
    _, details = phase_prediction_leave_one_out(observations, valid, positions, 2)
    diagnostics = dict(phase_excluded_prn=np.array([0, 0, details['excluded_prn']]),
                       phase_lag_epochs=np.array([0, 0, details['lag_epochs']]),
                       phase_full_common_count=np.array([[0, 0], [0, 0],
                                                         details['full_common_count']]),
                       phase_full_residual_m=np.full((3, 2, 8), np.nan),
                       phase_candidate_residual_m=np.full((3, 8), np.nan))
    diagnostics['phase_full_residual_m'][2] = details['full_residual_m']
    diagnostics['phase_candidate_residual_m'][2] = details['candidate_residual_m']
    navigation = dict(position=positions, selected=valid, diagnostics=diagnostics)
    geometry = exclusion_geometry(observations, navigation)
    assert geometry['full_max_residual_m'][2, 0] > .1
    assert geometry['candidate_max_residual_m'][2] < 1e-8


def test_fault_window_scores_only_actual_target_observations():
    observations = dict(time=np.array([1.99, 2., 2.01, 2.02]),
                        arc=np.zeros((4, 14), int),
                        carrier=np.ones((4, 14)))
    observations['arc'][2, 13] = 1
    observations['carrier'][3, 13] = np.nan
    mask = fault_window_mask(observations)
    np.testing.assert_array_equal(mask, [False, True, False, False])
    off = dict(position=np.zeros((4, 3)))
    on = dict(position=np.array([[50., 0., 0.], [20., 0., 0.],
                                 [50., 0., 0.], [50., 0., 0.]]))
    comparison, _ = compare_fault(off, on, observations,
                                  {'case': {'velocity_east': 0.}}, mask)
    assert comparison['fault_window_opportunities'] == 1
    assert comparison['common_published'] == 1
    assert comparison['baseline_common']['max_error_m'] == 0.
    assert comparison['fault_common']['max_error_m'] == 20.
    assert comparison['position_displacement_m']['max_error_m'] == 20.
