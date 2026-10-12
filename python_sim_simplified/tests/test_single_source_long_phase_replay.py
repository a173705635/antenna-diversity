"""The Stage 23 replay preserves the saved Stage 22 short bridge."""
import numpy as np
import pytest

from core.spoofing import single_source
from experiments.spoofing.single_source_long_phase_replay import replay_pair
from test_single_source_code_blend import patch_geometry, record


def freeze(navigation):
    saved = {}
    for key, value in navigation.items():
        if key == 'diagnostics':
            for diagnostic, data in value.items():
                saved[f'on_diagnostic_{diagnostic}'] = data.copy()
        else:
            saved[f'on_{key}'] = np.nan if value is None else value.copy() if isinstance(value, np.ndarray) else value
    return saved


def test_saved_short_replay_exact_and_long_candidate_separate(monkeypatch):
    patch_geometry(monkeypatch)
    monkeypatch.setattr(single_source, 'phase_prediction_leave_one_out',
                        lambda z, valid, positions, k: (
                            np.array([10., 0., 0.]),
                            dict(lag_epochs=1, excluded_prn=0,
                                 full_common_count=np.array([6, 6]),
                                 full_residual_m=np.zeros((2, 6)),
                                 candidate_common_count=0,
                                 candidate_residual_m=np.full(6, np.nan))))
    monkeypatch.setattr(single_source, 'phase_prediction_long_anchor',
                        lambda z, valid, positions, k, short_excluded_prn: (
                            np.array([11., 0., 0.]) if k == 4 else None,
                            dict(anchor_index=0, elapsed_s=1., excluded_prn=3,
                                 reason='used' if k == 4 else 'no_anchor',
                                 full_common_count=6,
                                 full_residual_m=np.full(6, .2),
                                 candidate_common_count=5,
                                 candidate_residual_m=np.zeros(6),
                                 candidate_prediction_m=np.array([11., 0., 0.]))))
    observations = record(6)
    short_reference = single_source.navigate(observations, diagnostics=True,
                                             code_blend=0., phase_leave_one_out=True)
    short, long = replay_pair(observations, freeze(short_reference))
    np.testing.assert_array_equal(short['position'], short_reference['position'])
    np.testing.assert_array_equal(short['selected'], long['selected'])
    assert short['candidate_position'][4, 0] == 10.
    assert long['candidate_position'][4, 0] == 11.
    assert long['diagnostics']['phase_long_path'][4] == 'long'


def test_changed_short_reference_is_rejected(monkeypatch):
    patch_geometry(monkeypatch)
    monkeypatch.setattr(single_source, 'phase_prediction_leave_one_out',
                        lambda z, valid, positions, k: (
                            np.array([10., 0., 0.]),
                            dict(lag_epochs=1, excluded_prn=0,
                                 full_common_count=np.array([6, 6]),
                                 full_residual_m=np.zeros((2, 6)),
                                 candidate_common_count=0,
                                 candidate_residual_m=np.full(6, np.nan))))
    observations = record(4)
    saved = freeze(single_source.navigate(observations, diagnostics=True,
                                          code_blend=0., phase_leave_one_out=True))
    saved['on_diagnostic_reason'][3] = 'wrong'
    with pytest.raises(AssertionError):
        replay_pair(observations, saved)
