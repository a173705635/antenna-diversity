"""Causal evidence history from actual receiver output windows."""
import json
import numpy as np
import pytest

from core.spoofing import single_source
from core.spoofing.recovery_geometry import FS
from core.spoofing.single_source import split_source_evidence
from experiments.spoofing import single_source_evidence_history as evidence_history
from experiments.spoofing.single_source_evidence_history import (
    analyze_trace, prompt_energy, support_counts, support_statistics,
)


CHIP = 2046


def make_trace(windows, *, projector=None, dimension=3):
    """Each window is (k, arc, start tick, available time, 20 prompts)."""
    projector = np.eye(3) if projector is None else projector
    count = max(row[0] for row in windows) + 1
    trace = dict(time=np.arange(count) * .01, available_time=np.arange(count) * .01 + .02,
                 support_split=np.full((count, 1), np.nan),
                 split_on_energy=np.full((count, 1), np.nan),
                 split_off_energy=np.full((count, 1), np.nan),
                 trace_window_k=np.array([row[0] for row in windows]),
                 trace_star=np.zeros(len(windows), int),
                 trace_arc=np.array([row[1] for row in windows]),
                 trace_start_sample=np.array([row[2] * CHIP for row in windows]),
                 trace_available_time=np.array([row[3] for row in windows]),
                 trace_prompts=np.array([row[4] for row in windows], complex),
                 trace_projectors=np.repeat(projector[None, :, :],
                                            (max(row[2] for row in windows) + 19) // 20 + 1,
                                            axis=0),
                 trace_dimensions=np.full((max(row[2] for row in windows) + 19) // 20 + 1,
                                          dimension),
                 trace_nominal=np.array([[1.], [0.], [0.]], complex))
    on, off = prompt_energy(np.array([2., 1., 0.]), trace['trace_nominal'][:, 0],
                            projector, dimension)
    for k, *_ in windows:
        trace['split_on_energy'][k, 0] = 20 * on
        trace['split_off_energy'][k, 0] = 20 * off
        trace['support_split'][k, 0] = on / off
        trace['available_time'][k] = next(row[3] for row in windows if row[0] == k)
    return trace


def test_prompt_energy_uses_complex_projection_and_basis_is_invariant():
    z = np.array([2. + 1j, 1. - 2j, 3j])
    a = np.array([1., 1j, 0.])
    basis = np.array([[1., 0.], [0., 1.], [0., 0.]])
    projector = basis @ basis.conj().T
    on, off = prompt_energy(z, a, projector, 2)
    expected_on = abs(np.vdot(basis.conj().T @ a, basis.conj().T @ z)) ** 2 / 2
    expected_off = np.vdot(basis.conj().T @ z, basis.conj().T @ z).real - expected_on
    np.testing.assert_allclose([on, off], [expected_on, expected_off])
    rotation = np.array([[1., 1j], [1j, 1.]]) / np.sqrt(2)
    rotated = basis @ rotation
    np.testing.assert_allclose(prompt_energy(z, a, rotated @ rotated.conj().T, 2),
                               [on, off])
    assert np.isnan(prompt_energy(z, a, np.diag([1., 0., 0.]), 1)[0])
    assert np.isnan(prompt_energy(z, a, np.diag([0., 0., 1.]), 2)[0])


def test_overlap_deduplicates_and_future_window_cannot_fill_past():
    prompt = np.tile([2., 1., 0.], (20, 1))
    trace = make_trace([(0, 0, 0, .02, prompt), (1, 0, 10, .03, prompt)])
    result = analyze_trace(trace)
    assert result['sample_count'][0, 0, 0] == 20
    assert result['sample_count'][1, 0, 0] == 20
    assert np.isnan(result['ratio'][0, 0, 1:]).all()
    assert result['sample_count'][1, 0, 1] == 0
    np.testing.assert_allclose(result['ratio'][:2, 0, 0], 8.)
    np.testing.assert_allclose(result['end_time_s'][:2, 0, 0], [.02, .03])
    np.testing.assert_allclose(result['data_age_s'][:2, 0, 0], 0.)
    np.testing.assert_allclose(result['start_time_s'][:2, 0, 0], [0., .01])
    np.testing.assert_allclose(result['oldest_sample_age_s'][:2, 0, 0], .02)
    trace['trace_prompts'][1, 0, 0] = 9.
    with pytest.raises(AssertionError):
        analyze_trace(trace)


def test_contiguous_100ms_history_and_late_prompt_do_not_backfill():
    prompt = np.tile([2., 1., 0.], (20, 1))
    continuous = make_trace([(k, 0, 10 * k, .02 + .01 * k, prompt)
                             for k in range(10)])
    result = analyze_trace(continuous)
    np.testing.assert_array_equal(result['sample_count'][9, 0], [20, 50, 100])
    np.testing.assert_allclose(result['ratio'][9, 0], [8., 8., 8.])
    assert result['sample_count'][2, 0, 1] == 0
    delayed = make_trace([(0, 0, 20, .04, prompt), (1, 0, 30, .05, prompt),
                          (2, 0, 40, .06, prompt), (3, 0, 0, .20, prompt)])
    history = analyze_trace(delayed)
    assert history['sample_count'][2, 0, 1] == 0
    prefix = make_trace([(0, 0, 20, .04, prompt), (1, 0, 30, .05, prompt)])
    earlier = analyze_trace(prefix)
    for field in ('ratio', 'on', 'off', 'sample_count', 'start_time_s', 'end_time_s',
                  'data_age_s', 'oldest_sample_age_s'):
        np.testing.assert_allclose(history[field][:2], earlier[field], equal_nan=True)


def test_cross_beam_complex_prompts_match_independent_raw_sample_recomputation():
    rng = np.random.default_rng(52)
    steering = np.array([1., .5j, .25])
    prompts = steering[:, None] + .3 * (rng.normal(size=(3, 50))
                                        + 1j * rng.normal(size=(3, 50)))
    local_ticks = np.exp(1j * np.arange(50) * .37)
    local = np.repeat(local_ticks, CHIP)
    calibrated = np.repeat(prompts * local_ticks[None, :], CHIP, axis=1)
    bases = [np.eye(3, dtype=complex),
             np.array([[1., 0.], [0., 1j], [0., 0.]], complex),
             np.array([[1., 0.], [0., 1j], [0., 0.]], complex)]
    windows = [(0, 0, 0, .02, prompts[:, 0:20].T),
               (1, 0, 10, .03, prompts[:, 10:30].T),
               (2, 0, 30, .05, prompts[:, 30:50].T)]
    trace = make_trace(windows)
    trace['trace_nominal'][:, 0] = steering
    trace['trace_projectors'] = np.array([basis @ basis.conj().T for basis in bases])
    trace['trace_dimensions'] = np.array([3, 2, 2])
    for k, _, start, _, _ in windows:
        begin, end = start * CHIP, (start + 20) * CHIP
        ratio, on, off = split_source_evidence(
            calibrated[:, begin:end], local[begin:end], steering, bases,
            begin, end, 20 * CHIP, return_energy=True)
        trace['support_split'][k, 0] = ratio
        trace['split_on_energy'][k, 0] = on
        trace['split_off_energy'][k, 0] = off
    history = analyze_trace(trace)
    np.testing.assert_allclose(history['on'][:, 0, 0],
                               trace['split_on_energy'][:, 0], rtol=1e-10)
    np.testing.assert_allclose(history['off'][:, 0, 0],
                               trace['split_off_energy'][:, 0], rtol=1e-10)
    _, expected_on, expected_off = split_source_evidence(
        calibrated[:, :50 * CHIP], local[:50 * CHIP], steering, bases,
        0, 50 * CHIP, 20 * CHIP, return_energy=True)
    np.testing.assert_allclose([history['on'][2, 0, 1], history['off'][2, 0, 1]],
                               [expected_on, expected_off], rtol=1e-10)
    assert history['sample_count'][2, 0, 1] == 50


def test_arc_and_missing_tick_break_history_and_short_prefix_is_unavailable():
    prompt = np.tile([2., 1., 0.], (20, 1))
    # The final window starts after a missing 10 ms span.
    trace = make_trace([(0, 0, 0, .02, prompt), (1, 0, 10, .03, prompt),
                        (2, 1, 20, .04, prompt), (3, 1, 30, .05, prompt),
                        (4, 1, 60, .08, prompt)])
    result = analyze_trace(trace)
    assert result['sample_count'][1, 0, 1] == 0
    assert result['sample_count'][3, 0, 1] == 0
    assert result['sample_count'][4, 0, 1] == 0
    assert np.isnan(result['ratio'][:, :, 2]).all()


def test_trace_flag_preserves_receiver_fields_with_small_mock_frontend(monkeypatch):
    monkeypatch.setattr(single_source, 'acquire_array', lambda *args, **kwargs: {'success': True})
    starts = np.arange(0., .161, .001)
    history = dict(epoch_start_samples=np.rint(starts * FS).astype(int),
                   code_nco_phase_chips=np.zeros(len(starts)),
                   code_nco_rate_hz=np.zeros(len(starts)),
                   carrier_nco_phase_rad=np.zeros(len(starts)),
                   carrier_nco_frequency_hz=np.zeros(len(starts)),
                   phase_lock_metric=np.ones(len(starts)),
                   cn0_db_hz=np.ones(len(starts)))
    monkeypatch.setattr(single_source, 'track', lambda *args, **kwargs:
                        {'success': True, **history})
    monkeypatch.setattr(single_source, 'local_signal', lambda code, template, t:
                        np.ones(len(t), complex))
    monkeypatch.setattr(single_source, '_sample_code', lambda code, phase:
                        np.ones(len(phase)))
    monkeypatch.setattr(single_source, 'calibrate', lambda block, local, desired:
                        (np.ones(len(block), complex), np.zeros(len(block))))
    monkeypatch.setattr(single_source, 'beam', lambda block, a, local, cov, prns:
                        (a / len(a), np.ones(a.shape[1]), 0.))
    monkeypatch.setattr(single_source, 'receive_measured', lambda *args, **kwargs:
                        {'segments': [{'history': history}]})
    monkeypatch.setattr(single_source, 'code_pseudorange', lambda h, t, *args:
                        np.zeros(len(t)))
    monkeypatch.setattr(single_source, 'carrier_range', lambda h, t, *args:
                        np.zeros(len(t)))
    samples = np.ones((3, round(.16 * FS)), complex)
    nominal = np.array([[1., 1., 1., 1.], [0., 1., 1j, -1.],
                        [0., 0., 1., 1j]], complex)
    record = dict(samples=samples, nominal=nominal, codes=np.zeros((4, 1)),
                  times=np.array([.12, .13]), satellites=np.zeros((2, 4, 3)),
                  coarse=np.zeros(4))
    ordinary = single_source.frontend(record, [0, 1, 2], split_support=True)
    traced = single_source.frontend(record, [0, 1, 2], split_support=True,
                                    evidence_trace=True)
    for key, expected in ordinary.items():
        np.testing.assert_array_equal(traced[key], expected, err_msg=key)
    assert len(traced['trace_window_k']) == 8
    assert traced['trace_prompts'].shape == (8, 20, 3)
    history = analyze_trace(traced)
    np.testing.assert_allclose(history['on'][:, :, 0], ordinary['split_on_energy'],
                               equal_nan=True)
    np.testing.assert_allclose(history['off'][:, :, 0], ordinary['split_off_energy'],
                               equal_nan=True)


def test_support_counts_apply_original_quality_gates_and_measure_runs():
    observations = dict(ranges=np.ones((6, 2)), lock=np.ones((6, 2)),
                        cn0=np.full((6, 2), 30.), gain=np.ones((6, 2)),
                        support=np.full((6, 2), 4.))
    observations['lock'][0, 0] = 0.
    observations['cn0'][1, 0] = 20.
    observations['gain'][2, 0] = 0.
    ratio = np.full((6, 2, 3), 4.)
    ratio[3, 1, :] = 0.
    counts = support_counts(observations, ratio)
    np.testing.assert_array_equal(counts[:, 0], [1, 1, 1, 2, 2, 2])
    np.testing.assert_array_equal(counts[:, 1], [1, 1, 1, 1, 2, 2])
    statistics = support_statistics(np.array([3, 6, 6, 2, 6, 6]),
                                     np.ones(6, bool), np.array([False, True, True,
                                                                  True, True, True]))
    assert statistics['epochs_ge4'] == 4
    assert statistics['epochs_ge6'] == 4
    assert statistics['longest_ge6_s'] == .02
    assert statistics['tail50_ge6'] == 4


def test_reference_pll_schema_is_checked_before_frontend(tmp_path, monkeypatch):
    inputs = {}
    for name in evidence_history.CASES:
        folder = tmp_path / name
        folder.mkdir()
        np.save(folder / 'samples.npy', np.ones((3, 1)))
        np.savez(folder / 'observations.npz', marker=np.array([1]))
        (folder / 'metadata.json').write_text(json.dumps({
            'samples_sha256': evidence_history.digest(folder / 'samples.npy'),
            'observations_sha256': evidence_history.digest(folder / 'observations.npz')}),
            encoding='utf-8')
        tracking = folder / 'tracking.npz'
        tracking.write_bytes(b'reference')
        inputs[name] = (tracking, folder)
    pll = tmp_path / 'study' / 'pll'
    pll.mkdir(parents=True)
    (pll / 'navigation.npz').write_bytes(b'navigation')
    (pll / 'summary.json').write_text(json.dumps({
        'tracking_sha256': evidence_history.digest(inputs['pll_weak'][0]),
        'navigation_sha256': evidence_history.digest(pll / 'navigation.npz'),
        'pll': {'initialization_available_time_s': .19}}), encoding='utf-8')
    monkeypatch.setattr(evidence_history, 'RESULTS', tmp_path)
    monkeypatch.setattr(evidence_history, 'STUDY_OUTPUT', 'study')
    monkeypatch.setattr(evidence_history, 'OUTPUT', tmp_path / 'output')
    monkeypatch.setattr(evidence_history, 'CASE_INPUTS', inputs)
    monkeypatch.setattr(evidence_history, 'source_hashes', lambda: {})

    def stop_before_rf(*args, **kwargs):
        raise RuntimeError('frontend reached after preflight')

    monkeypatch.setattr(evidence_history, 'frontend', stop_before_rf)
    with pytest.raises(RuntimeError, match='frontend reached after preflight'):
        evidence_history.main()
