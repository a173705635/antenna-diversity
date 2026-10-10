"""Only the accepted code/phase mix changes with code_blend."""
import numpy as np
import pytest

from core import single_source
from experiments.single_source_code_blend import assert_saved_navigation


def record(n):
    stars = 6
    return dict(time=np.arange(n) * .01,
                available_time=np.arange(n) * .01 + .02,
                ranges=np.ones((n, stars)), lock=np.ones((n, stars)),
                cn0=np.full((n, stars), 30.), gain=np.ones((n, stars)),
                support=np.full((n, stars), 4.),
                carrier=np.ones((n, stars)), arc=np.zeros((n, stars), int),
                satellites=np.ones((n, stars, 3)))


def patch_geometry(monkeypatch, position_solution=None, solve_position=None):
    monkeypatch.setattr(single_source, 'smooth_history',
                        lambda z, valid, **kwargs: (z['ranges'].copy(),
                                                     np.zeros_like(z['ranges'])))
    monkeypatch.setattr(single_source, 'position_solution', position_solution or
                        (lambda *args: (np.zeros(3), 1., 0.)))
    monkeypatch.setattr(single_source, 'receiver_reference',
                        lambda: (np.zeros(3), np.eye(3)))
    monkeypatch.setattr(single_source, 'solve_position', solve_position or
                        (lambda *args, **kwargs:
                         dict(position_m=np.array([2., 0., 0.]))))


def test_code_blend_zero_uses_phase_only_when_original_branch_blends(monkeypatch):
    patch_geometry(monkeypatch)
    monkeypatch.setattr(single_source, 'phase_prediction',
                        lambda z, valid, out, k: None if k == 3
                        else np.array([10., 0., 0.]))
    observations = record(5)
    default = single_source.navigate(observations, diagnostics=True)
    explicit = single_source.navigate(observations, diagnostics=True,
                                      code_blend=.1)
    candidate = single_source.navigate(observations, diagnostics=True,
                                       code_blend=0.)
    for key in ('position', 'candidate_position', 'selected'):
        np.testing.assert_array_equal(default[key], explicit[key], strict=True)
    for key in default['diagnostics']:
        np.testing.assert_array_equal(default['diagnostics'][key],
                                      explicit['diagnostics'][key], strict=True)
    assert default['initialization_time'] == explicit['initialization_time']
    np.testing.assert_allclose(default['candidate_position'][:3, 0], 9.2)
    np.testing.assert_allclose(candidate['candidate_position'][:3, 0], 10.)
    assert default['candidate_position'][3, 0] == candidate['candidate_position'][3, 0] == 2.
    np.testing.assert_array_equal(default['selected'], candidate['selected'])
    np.testing.assert_array_equal(default['diagnostics']['phase_ok'],
                                  candidate['diagnostics']['phase_ok'])


def test_code_blend_does_not_change_large_separation_branch(monkeypatch):
    def separated(satellites, ranges, **kwargs):
        east = 2. if len(satellites) == 6 else 32.
        return dict(position_m=np.array([east, 0., 0.]))

    patch_geometry(monkeypatch, solve_position=separated)
    monkeypatch.setattr(single_source, 'phase_prediction',
                        lambda *args: np.array([10., 0., 0.]))
    original = single_source.navigate(record(4), diagnostics=True, code_blend=.1)
    candidate = single_source.navigate(record(4), diagnostics=True, code_blend=0.)
    np.testing.assert_array_equal(original['candidate_position'],
                                  candidate['candidate_position'])
    np.testing.assert_allclose(candidate['candidate_position'][:, 0], 10.)
    assert (candidate['diagnostics']['code_separation_m'] > 20.).all()


def test_recovery_confirmation_uses_fresh_code_with_zero_blend(monkeypatch):
    calls = iter(range(15))
    patch_geometry(monkeypatch,
                   position_solution=lambda *args:
                   (np.zeros(3), 2000. if next(calls) == 3 else 1., 0.))
    monkeypatch.setattr(single_source, 'phase_prediction',
                        lambda *args: np.array([10., 0., 0.]))
    candidate = single_source.navigate(record(15), diagnostics=True,
                                       code_blend=0.)
    assert candidate['diagnostics']['isolated'][3]
    assert candidate['diagnostics']['recovery_confirmed'][13]
    assert not candidate['diagnostics']['isolated'][13]
    np.testing.assert_allclose(candidate['candidate_position'][12, 0], 10.)
    np.testing.assert_allclose(candidate['position'][13, 0], 2.)


def test_saved_reference_comparison_includes_diagnostics():
    navigation = dict(position=np.array([[1., 2., 3.]]),
                      initialization_time=.27,
                      diagnostics=dict(reason=np.array(['published'])))
    saved = dict(**{'100ms_coherent_position': navigation['position'].copy(),
                    '100ms_coherent_initialization_time': np.array(.27),
                    '100ms_coherent_diagnostic_reason': np.array(['published'])})
    assert_saved_navigation(navigation, saved)
    saved['100ms_coherent_diagnostic_reason'][0] = 'isolated'
    with pytest.raises(AssertionError):
        assert_saved_navigation(navigation, saved)
