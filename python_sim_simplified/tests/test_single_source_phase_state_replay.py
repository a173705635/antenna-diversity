"""The saved short navigation is an exact reference, never candidate state."""
import numpy as np
import pytest

from core.single_source import navigate
from experiments.single_source_phase_state import replay_pair
from test_single_source_phase_state import fixed_code_geometry, observations


def saved_short(navigation):
    saved = {}
    for key, value in navigation.items():
        if key == 'diagnostics':
            for name, array in value.items():
                saved[f'short_diagnostic_{name}'] = array.copy()
        else:
            saved[f'short_{key}'] = np.nan if value is None else value.copy() if isinstance(value, np.ndarray) else value
    return saved


def test_replay_matches_frozen_reference_and_candidate_uses_only_own_state(monkeypatch):
    fixed_code_geometry(monkeypatch)
    record = observations()
    old = navigate(record, diagnostics=True, code_blend=0.,
                   phase_leave_one_out=True)
    baseline, candidate = replay_pair(record, saved_short(old))
    np.testing.assert_array_equal(baseline['position'], old['position'])
    assert np.isnan(baseline['position'][27]).all()
    assert np.isfinite(candidate['position'][27]).all()
    np.testing.assert_array_equal(candidate['selected'], baseline['selected'])
    assert 'phase_state_source' not in baseline['diagnostics']


def test_replay_rejects_changed_cached_navigation(monkeypatch):
    fixed_code_geometry(monkeypatch)
    record = observations(8)
    saved = saved_short(navigate(record, diagnostics=True, code_blend=0.,
                                 phase_leave_one_out=True))
    saved['short_diagnostic_reason'][3] = 'wrong'
    with pytest.raises(AssertionError):
        replay_pair(record, saved)
