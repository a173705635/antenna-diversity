"""Five-millisecond coherent decomposition of saved output-window prompts."""
import json

import numpy as np
import pytest

from experiments.single_source_coherence_diagnosis import (
    analyze_groups, decompose_group, stratified_summary, validate_saved_sources,
)
from experiments.single_source_suppression_study import digest


CHIP = 2046


def trace(windows, *, steering=None, projector=None, dimension=3):
    """Window rows are (star, arc, start tick, available, 20 complex prompts)."""
    steering = np.array([1., 0., 0.]) if steering is None else steering
    projector = np.eye(3) if projector is None else projector
    blocks = max(row[2] + 19 for row in windows) // 20 + 1
    return dict(trace_star=np.array([row[0] for row in windows]),
                trace_arc=np.array([row[1] for row in windows]),
                trace_start_sample=np.array([row[2] * CHIP for row in windows]),
                trace_available_time=np.array([row[3] for row in windows]),
                trace_prompts=np.array([row[4] for row in windows], complex),
                trace_projectors=np.repeat(projector[None], blocks, axis=0),
                trace_dimensions=np.full(blocks, dimension),
                trace_nominal=steering[:, None])


def test_constant_complex_signal_has_five_times_single_prompt_coherent_energy():
    z = np.array([2. + 1j, 1. - .5j, .2j])
    a = np.array([1., 0., 0.])
    group = decompose_group(np.tile(z, (5, 1)), a, np.eye(3), 3)
    np.testing.assert_allclose(group['on_coh'], 5 * abs(z[0]) ** 2)
    np.testing.assert_allclose(group['on_noncoh_sum'], group['on_coh'])
    np.testing.assert_allclose(group['off_noncoh_sum'], group['off_coh'])
    np.testing.assert_allclose(group['off_res_direct'], 0., atol=1e-12)
    np.testing.assert_allclose([group['total_concentration'], group['on_concentration'],
                                group['off_concentration']], 1.)


def test_q3_total_concentration_uses_actual_projected_energy():
    prompts = np.array([[2., 1., 1.], [1., -1., 2.], [3., 2., -1.],
                        [0., 1., 3.], [2., -2., 1.]], complex)
    projector = np.eye(3)
    group = decompose_group(prompts, np.array([1., 0., 0.]), projector, 3)
    coherent = prompts.sum(axis=0) / np.sqrt(5)
    expected = np.vdot(coherent, coherent).real / np.sum(abs(prompts) ** 2)
    np.testing.assert_allclose(group['total_concentration'], expected)
    np.testing.assert_allclose(group['on_concentration'],
                               group['on_coh'] / group['on_noncoh_sum'])
    np.testing.assert_allclose(group['off_concentration'],
                               group['off_coh'] / group['off_noncoh_sum'])


def test_phase_and_spatial_basis_rotation_leave_decomposition_unchanged():
    rng = np.random.default_rng(14)
    prompts = rng.normal(size=(5, 3)) + 1j * rng.normal(size=(5, 3))
    steering = np.array([1., .3j, .4])
    basis = np.array([[1., 0.], [0., 1.], [0., 0.]])
    projector = basis @ basis.conj().T
    original = decompose_group(prompts, steering, projector, 2)
    rotation = np.array([[1., 1j], [1j, 1.]]) / np.sqrt(2)
    rotated_basis = basis @ rotation
    phase = np.exp(1j * .71)
    changed = decompose_group(prompts * phase, steering,
                              rotated_basis @ rotated_basis.conj().T, 2)
    for key in original:
        np.testing.assert_allclose(changed[key], original[key], rtol=1e-12, atol=1e-12)


def test_along_steering_sign_flips_only_reduce_coherent_on():
    a = np.array([1., 0., 0.])
    signs = np.array([1., 1., -1., 1., -1.])
    group = decompose_group(signs[:, None] * a[None, :], a, np.eye(3), 3)
    np.testing.assert_allclose(group['on_noncoh_sum'], 5.)
    np.testing.assert_allclose(group['on_coh'], .2)
    np.testing.assert_allclose([group['off_noncoh_sum'], group['off_coh'],
                                group['off_res_direct']], 0., atol=1e-12)


def test_changing_wrong_direction_component_creates_direct_off_residual():
    a = np.array([1., 0., 0.])
    prompts = np.array([[1., 0., 0.], [1., 1., 0.], [1., -1., 0.],
                        [1., 2., 0.], [1., -2., 0.]], complex)
    group = decompose_group(prompts, a, np.eye(3), 3)
    assert group['off_res_direct'] > 0
    np.testing.assert_allclose(group['off_noncoh_sum'] - group['off_coh'],
                               group['off_res_direct'], rtol=1e-12)
    np.testing.assert_allclose(group['on_noncoh_sum'] - group['on_coh'], 0., atol=1e-12)


def test_absolute_alignment_arc_gap_first_availability_and_future_prefix():
    prompt = np.tile([2., 1., 0.], (20, 1))
    prefix = trace([(0, 0, 1, .03, prompt)])
    early = analyze_groups(prefix)
    np.testing.assert_array_equal(early['start_sample'] // CHIP, [5, 10, 15])
    np.testing.assert_allclose(early['available_time'], .03)
    full = trace([(0, 0, 1, .03, prompt), (0, 0, 19, .04, prompt),
                  (0, 1, 51, .06, prompt)])
    result = analyze_groups(full)
    for key, values in early.items():
        np.testing.assert_allclose(result[key][:len(values)], values, equal_nan=True)
    starts = result['start_sample'] // CHIP
    assert 20 in starts and 25 in starts
    assert 40 not in starts and 50 not in starts and 55 in starts
    assert result['available_time'][np.flatnonzero(starts == 20)[0]] == .04
    # Repeated prompts must agree even when the later window is unavailable to earlier groups.
    full['trace_prompts'][1, 0, 0] = 9.
    with pytest.raises(AssertionError):
        analyze_groups(full)


def test_saved_source_preflight_rejects_changed_metadata(tmp_path):
    stage14 = tmp_path / 'evidence_history'
    stage14.mkdir()
    inputs = {}
    rows = {}
    for name in ('pll_weak', 'genuine_absent', 'control'):
        folder = tmp_path / name
        folder.mkdir()
        for filename in ('samples.npy', 'observations.npz', 'metadata.json'):
            (folder / filename).write_text(name + filename, encoding='utf-8')
        output = stage14 / name
        output.mkdir()
        for filename in ('tracking.npz', 'navigation.npz', 'trace.npz', 'history.npz'):
            (output / filename).write_text(name + filename, encoding='utf-8')
        inputs[name] = (output / 'tracking.npz', folder)
        rows[name] = dict(input_sha256={filename: digest(folder / filename)
                                        for filename in ('samples.npy', 'observations.npz', 'metadata.json')},
                          **{filename[:-4] + '_sha256': digest(output / filename)
                             for filename in ('tracking.npz', 'navigation.npz', 'trace.npz', 'history.npz')})
    (stage14 / 'summary.json').write_text(json.dumps({'cases': rows}), encoding='utf-8')
    validate_saved_sources(stage14, inputs)
    (inputs['genuine_absent'][1] / 'metadata.json').write_text('changed', encoding='utf-8')
    with pytest.raises(AssertionError, match='genuine_absent.*metadata.json'):
        validate_saved_sources(stage14, inputs)


def test_strata_use_sample_bounds_and_keep_empty_steady_null():
    table = dict(start_time_s=np.array([.295, .3, .495, .5]),
                 end_time_s=np.array([.3, .305, .5, .505]),
                 q=np.array([2, 2, 3, 3]),
                 total_concentration=np.array([.2, .4, .6, .8]),
                 on_concentration=np.array([.1, .3, .5, .7]),
                 off_concentration=np.array([.3, .5, .7, .9]))
    strata = stratified_summary(table)
    assert strata['pre']['2']['count'] == 1
    assert strata['transition']['2']['count'] == 1
    assert strata['transition']['3']['count'] == 1
    assert strata['steady']['3']['count'] == 1
    assert strata['steady']['2']['count'] == 0
    assert strata['steady']['2']['total_concentration']['median'] is None
    empty_steady = stratified_summary({key: values[:3] for key, values in table.items()})
    assert empty_steady['steady']['2']['count'] == 0
    assert empty_steady['steady']['3']['count'] == 0
    assert empty_steady['steady']['3']['off_concentration']['p95'] is None
