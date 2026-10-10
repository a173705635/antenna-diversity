import json

import numpy as np
import pytest

from core.single_source import beam as core_beam
from core.single_source import frontend as core_frontend
from experiments.single_source_temporal_trigger import navigation_input
from tools import cross_fold_prn_navigation as study


def test_fold_beam_keeps_full_raw_gate_and_one_fold_projector(monkeypatch):
    block = np.ones((3, 20 * 2046), complex)
    local = np.ones((24, 20 * 2046), complex)
    model = np.ones((3, 24), complex)
    prns = np.arange(24)
    captured = []
    monkeypatch.setattr(study, 'beam', lambda *args, **kwargs:
                        (model / 3, np.ones(24), 3.2))

    def fit(grouped, desired):
        captured.append((grouped.shape, desired.shape))
        return np.array([1., 0., 0.]), np.ones((7, 2)), -1

    monkeypatch.setattr(study.direction, 'fit_direction', fit)
    weight, gain, raw, _, failure, projector = study.fold_beam(
        block, model, local, prns, 1, 5, 3.2)
    assert captured == [((3, 48), (3, 48))]
    assert raw == 3.2 and failure == -1
    np.testing.assert_array_equal(projector, np.diag([0., 1., 1.]))
    np.testing.assert_allclose(weight, projector @ model /
                               (np.sum(model.conj() * (projector @ model),
                                       axis=0).real + .01)[None, :])
    np.testing.assert_allclose(gain, np.full(24, 2 / 3))

    monkeypatch.setattr(study, 'beam', lambda *args, **kwargs:
                        (model / 3, np.ones(24), 2.9))
    monkeypatch.setattr(study.direction, 'fit_direction', lambda *args:
                        (np.full(3, np.nan), np.zeros((7, 2)), 0))
    weight, gain, raw, _, failure, projector = study.fold_beam(
        block, model, local, prns, 1, 5, 2.9)
    assert failure == 0 and projector is None and raw == 2.9
    np.testing.assert_array_equal(weight, model / 3)
    np.testing.assert_array_equal(gain, np.ones(24))
    monkeypatch.setattr(study, 'beam', lambda *args, **kwargs:
                        (model / 3, np.ones(24), 3.2))
    with pytest.raises(AssertionError):
        study.fold_beam(block, model, local, prns, 1, 5, 3.1)
    with pytest.raises(RuntimeError, match='direction undefined'):
        study.fold_beam(block, model, local, prns, 1, 5, 3.2)


def test_two_pass_merge_restores_star_fields_without_common_projector():
    count, stars = 5, 4
    baseline = {name: np.arange(count * stars).reshape(count, stars).astype(float)
                for name in study.STAR_FIELDS}
    baseline['arc'] = baseline['arc'].astype(int)
    baseline.update(time=np.arange(count), available_time=np.arange(count) + .02,
                    satellites=np.ones((count, stars, 3)), ratio=np.arange(count),
                    startup_prns=np.arange(stars), calibration=np.ones(3),
                    calibration_rate=np.zeros(3),
                    startup_relative_amplitude=np.ones(3))
    odd = {key: value.copy() for key, value in baseline.items()}
    even = {key: value.copy() for key, value in baseline.items()}
    for key in study.STAR_FIELDS:
        odd[key][:, ::2] = -99
        even[key][:, 1::2] = -99
    merged = study.merge_star_fields(odd, even)
    for key in baseline:
        np.testing.assert_array_equal(merged[key], baseline[key])
    support_odd = {key: odd[key] for key in ('support', 'gain')}
    support_even = {key: even[key] for key in ('support', 'gain')}
    for key, value in study.merge_support(support_odd, support_even).items():
        np.testing.assert_array_equal(value, baseline[key])
    nav = navigation_input(merged, merged['support'])
    for key in study.STAR_FIELDS[:7]:
        if key in nav:
            np.testing.assert_array_equal(nav[key], baseline[key])


def test_private_frontend_skips_training_prn_and_leaves_core_globals(monkeypatch):
    monkeypatch.setattr(study, 'START_BLOCK', 5)
    monkeypatch.setattr(study, 'BLOCKS', 6)
    monkeypatch.setattr(study, 'fold_beam', lambda block, model, local, prns,
                        parity, index, raw: (
                            model / 3, np.ones(model.shape[1]), raw,
                            np.ones((7, 2)), -1, None))
    received = []
    monkeypatch.setattr(study, 'receive_measured', lambda signal, prn, *args,
                        **kwargs: received.append(prn) or {'segments': []})

    def fake_run(receiver, record, scratch, loops):
        assert receiver.__globals__['beam'] is not core_beam
        assert receiver.__globals__['receive_measured'] is not (
            core_frontend.__globals__['receive_measured'])
        model = np.ones((3, 4))
        receiver.__globals__['beam'](np.ones((3, 20 * 2046)), model,
                                     np.ones((4, 20 * 2046)), None,
                                     np.arange(4))
        for prn in range(1, 5):
            assert receiver.__globals__['receive_measured'](
                np.zeros(8), prn, None, None) == {'segments': []}
        return dict(trace_star=np.array([1, 3]),
                    trace_raw_trigger=np.array([np.nan] * 5 + [2.]),
                    trace_projectors=np.tile(np.eye(3), (6, 1, 1)),
                    trace_dimensions=np.full(6, 3))

    monkeypatch.setattr(study.oracle, 'run_frontend', fake_run)
    record = {'codes': np.arange(4)}
    observations, fit = study.private_frontend(
        record, {'raw': np.array([np.nan] * 5 + [2.])}, 1, None, [])
    assert received == [2, 4]
    assert observations['trace_star'].tolist() == [1, 3]
    assert fit['failure_step'][5] == -1
    assert core_frontend.__globals__['beam'] is core_beam


def test_pass_eligibility_needs_matching_raw_and_dimension():
    trace = dict(trace_raw_trigger=np.array([np.nan] * 5 + [3.1] * 6),
                 trace_dimensions=np.array([3] * 5 + [2] * 6),
                 trace_projectors=np.tile(np.eye(3), (11, 1, 1)))
    available = np.arange(11) * .02 + .02
    fit = {'raw': trace['trace_raw_trigger'].copy()}
    eligible, _, _ = study.verify_pass_alignment(trace, trace, fit, available)
    assert eligible.shape == available.shape
    changed = {**trace, 'trace_dimensions': trace['trace_dimensions'].copy()}
    changed['trace_dimensions'][6] = 3
    with pytest.raises(AssertionError):
        study.verify_pass_alignment(trace, changed, fit, available)


def test_completed_pass_rejects_wrong_parity_and_missing_or_damaged_output(
        tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'OUTPUT', tmp_path)
    snapshot = tmp_path / 'source_snapshot.zip'
    snapshot.write_bytes(b'source')
    folder = tmp_path / 'train_odd'
    folder.mkdir()
    for name in study.PASS_FILES:
        (folder / name).write_bytes(name.encode())
    receipt = dict(train_parity=1, source_sha256='tool',
                   source_snapshot_sha256=study.digest(snapshot),
                   input_sha256={'samples.npy': 'input'},
                   saved_fit_sha256='fit',
                   output_sha256={name: study.digest(folder / name)
                                  for name in study.PASS_FILES})
    (folder / 'receipt.json').write_text(json.dumps(receipt), encoding='utf-8')
    study.verify_completed_pass(folder, 1, receipt['input_sha256'], 'fit', 'tool')
    with pytest.raises(AssertionError):
        study.verify_completed_pass(folder, 0, receipt['input_sha256'], 'fit', 'tool')
    (folder / 'trace.npz').write_bytes(b'damaged')
    with pytest.raises(AssertionError):
        study.verify_completed_pass(folder, 1, receipt['input_sha256'], 'fit', 'tool')
    (folder / 'trace.npz').unlink()
    with pytest.raises(FileNotFoundError):
        study.verify_completed_pass(folder, 1, receipt['input_sha256'], 'fit', 'tool')


def test_completed_path_rechecks_both_pass_output_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'OUTPUT', tmp_path)
    snapshot = tmp_path / 'source_snapshot.zip'
    snapshot.write_bytes(b'source')
    saved = tmp_path / 'saved'
    saved.mkdir()
    (saved / 'fit.npz').write_bytes(b'fit')
    inputs = {'samples.npy': 'input'}
    saved_outputs = {'fit.npz': study.digest(saved / 'fit.npz')}
    monkeypatch.setattr(study, 'load_scene', lambda *args:
                        ({}, {}, saved, inputs, saved_outputs))
    monkeypatch.setattr(study, 'verified_baseline', lambda *args: ({}, None))
    folder = tmp_path / '1' / 'weak'
    folder.mkdir(parents=True)
    for name, parity in (('train_odd', 1), ('train_even', 0)):
        passed = folder / name
        passed.mkdir()
        for filename in study.PASS_FILES:
            (passed / filename).write_bytes(filename.encode())
        pass_receipt = dict(train_parity=parity, source_sha256='tool',
            source_snapshot_sha256=study.digest(snapshot),
            input_sha256=inputs, saved_fit_sha256=study.digest(saved / 'fit.npz'),
            output_sha256={filename: study.digest(passed / filename)
                           for filename in study.PASS_FILES})
        (passed / 'receipt.json').write_text(json.dumps(pass_receipt),
                                              encoding='utf-8')
    final_files = ('navigation.npz', 'merged_tracking.npz',
                   'merged_support.npz')
    for filename in final_files:
        (folder / filename).write_bytes(filename.encode())
    receipt = dict(source_sha256='tool',
                   source_snapshot_sha256=study.digest(snapshot),
                   input_sha256=inputs, saved_output_sha256=saved_outputs,
                   pass_receipt_sha256={name: study.digest(folder / name /
                       'receipt.json') for name in ('train_odd', 'train_even')},
                   output_sha256={name: study.digest(folder / name)
                                  for name in final_files}, score={'pass': False})
    (folder / 'receipt.json').write_text(json.dumps(receipt), encoding='utf-8')
    assert study.run_path(1, 'weak', {}, 'tool') == {'pass': False}
    (folder / 'train_even' / 'groups.npz').write_bytes(b'damaged')
    with pytest.raises(AssertionError):
        study.run_path(1, 'weak', {}, 'tool')


@pytest.mark.local_evidence
def test_preflight_rejects_changed_frozen_paired_helper(monkeypatch):
    original_digest = study.direction.digest

    def tampered_digest(path):
        if str(path).endswith('tools\\frozen_weak_h0_counterfactual.py'):
            return 'wrong SHA'
        return original_digest(path)

    monkeypatch.setattr(study.direction, 'digest', tampered_digest)
    with pytest.raises(AssertionError):
        study.preflight()


@pytest.mark.local_evidence
def test_saved_baseline_split_merge_reproduces_short_input():
    saved = (study.paired.stage47.OUTPUT / '2026100402' / 'weak' / 'gate3')
    baseline = study.read_npz(saved / 'tracking.npz')
    coherent = study.read_npz(saved / 'support.npz')['coherent_support']
    odd = {key: value.copy() for key, value in baseline.items()}
    even = {key: value.copy() for key, value in baseline.items()}
    for key in study.STAR_FIELDS:
        odd[key][:, ::2] = 0
        even[key][:, 1::2] = 0
    merged = study.merge_star_fields(odd, even)
    for key in baseline:
        np.testing.assert_array_equal(merged[key], baseline[key])
    odd_support = {'support': coherent.copy()}
    even_support = {'support': coherent.copy()}
    odd_support['support'][:, ::2] = 0
    even_support['support'][:, 1::2] = 0
    merged_support = study.merge_support(odd_support, even_support)['support']
    np.testing.assert_array_equal(merged_support, coherent)
    original = navigation_input(baseline, coherent)
    reproduced = navigation_input(merged, merged_support)
    for key in original:
        np.testing.assert_array_equal(reproduced[key], original[key])
