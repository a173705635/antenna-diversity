"""A temporal projection gate restricts only unpublished short anchors."""
import numpy as np

from core import single_source
from test_single_source_phase_state import fixed_code_geometry, observations


def scientific_fields_equal(first, second):
    for key in ('position', 'candidate_position', 'selected'):
        np.testing.assert_array_equal(first[key], second[key], strict=True)
    assert first['initialization_time'] == second['initialization_time']
    for key, array in first['diagnostics'].items():
        np.testing.assert_array_equal(array, second['diagnostics'][key], strict=True)


def test_none_mask_matches_unrestricted_phase_state_exactly(monkeypatch):
    fixed_code_geometry(monkeypatch)
    z = observations()
    original = single_source.navigate(z, diagnostics=True, code_blend=0.,
        phase_leave_one_out=True, phase_state_continuity=True)
    explicit = single_source.navigate(z, diagnostics=True, code_blend=0.,
        phase_leave_one_out=True, phase_state_continuity=True,
        phase_state_eligible=None)
    assert original['diagnostics'].keys() == explicit['diagnostics'].keys()
    scientific_fields_equal(original, explicit)


def test_all_false_mask_keeps_original_published_only_science(monkeypatch):
    fixed_code_geometry(monkeypatch)
    z = observations()
    baseline = single_source.navigate(z, diagnostics=True, code_blend=0.,
        phase_leave_one_out=True)
    gated = single_source.navigate(z, diagnostics=True, code_blend=0.,
        phase_leave_one_out=True, phase_state_continuity=True,
        phase_state_eligible=np.zeros(len(z['time']), bool))
    scientific_fields_equal(baseline, gated)
    assert np.isnan(gated['diagnostics']['phase_state_position_m'][23]).all()
    assert gated['diagnostics']['phase_state_source'][23] == 'none'
    assert not gated['diagnostics']['phase_state_eligible'].any()


def test_true_false_true_refuses_cross_gate_internal_anchor_but_uses_published_anchor(monkeypatch):
    fixed_code_geometry(monkeypatch)
    z = observations()
    mask = np.ones(len(z['time']), bool)
    mask[24] = False
    gated = single_source.navigate(z, diagnostics=True, code_blend=0.,
        phase_leave_one_out=True, phase_state_continuity=True,
        phase_state_eligible=mask)
    d = gated['diagnostics']
    assert d['phase_ok'][23]
    assert d['phase_state_source'][23] == 'output_gate'
    assert d['phase_state_source'][24] == 'none'
    assert not d['phase_ok'][25]
    assert d['phase_state_blocked_internal_anchor_index'][25, 1] == 23
    assert np.isnan(d['phase_state_position_m'][25]).all()
    assert d['phase_ok'][22]
    assert d['phase_state_short_anchor_published'][22]


def test_false_current_epoch_does_not_read_or_write_internal_state(monkeypatch):
    fixed_code_geometry(monkeypatch)
    z = observations()
    mask = np.ones(len(z['time']), bool)
    mask[24] = False
    gated = single_source.navigate(z, diagnostics=True, code_blend=0.,
        phase_leave_one_out=True, phase_state_continuity=True,
        phase_state_eligible=mask)
    d = gated['diagnostics']
    assert d['phase_state_blocked_internal_anchor_index'][24, 0] == 23
    assert d['phase_ok'][24]
    assert d['phase_state_short_anchor_index'][24] == 22
    assert d['phase_state_short_anchor_published'][24]
    assert d['phase_state_source'][24] == 'none'


def test_gated_state_prefix_never_uses_future_mask(monkeypatch):
    fixed_code_geometry(monkeypatch)
    mask = np.ones(30, bool)
    mask[24] = False
    full = single_source.navigate(observations(), diagnostics=True,
        code_blend=0., phase_leave_one_out=True,
        phase_state_continuity=True, phase_state_eligible=mask)
    prefix = single_source.navigate(observations(27), diagnostics=True,
        code_blend=0., phase_leave_one_out=True,
        phase_state_continuity=True, phase_state_eligible=mask[:27])
    for key in ('position', 'candidate_position', 'selected'):
        np.testing.assert_array_equal(prefix[key], full[key][:27], strict=True)
    for key, array in prefix['diagnostics'].items():
        np.testing.assert_array_equal(array, full['diagnostics'][key][:27],
                                      strict=True)
