"""Causal beam-null history applied to receiver frontend blocks."""
import inspect
import json
import numpy as np
import pytest

from core.single_source import beam_history_weight, frontend
from experiments import single_source_beam_history_rf as beam_rf


def steering():
    return np.array([[1., 1., 1.], [1., 1j, -1.], [1., -1., 1j]])


def rank_two_weights(null):
    projector = np.eye(3) - np.outer(null, null.conj())
    return projector @ steering()


def test_default_frontend_signature_retains_one_block():
    signature = inspect.signature(frontend)
    assert signature.parameters['beam_history_blocks'].default == 1
    assert signature.parameters['split_support'].default is False
    assert signature.parameters['receiver_pll'].default is None


def test_constant_null_phase_gain_and_new_weight_basis():
    a = steering()
    null = np.array([1., 1j, 2.]) / np.sqrt(6.)
    history = []
    for phase in np.arange(7):
        weights, gain, count, gap = beam_history_weight(
            rank_two_weights(null * np.exp(1j * phase)), a, 4., history, 5)
        assert count == min(phase + 1, 5)
        np.testing.assert_allclose(gap, 1., atol=1e-14)
        np.testing.assert_allclose(null.conj() @ weights, 0., atol=1e-14)
        projector = np.eye(3) - np.outer(null, null.conj())
        response = np.sum(a.conj() * (projector @ a), axis=0).real
        np.testing.assert_allclose(weights, (projector @ a) / (response + .01), atol=1e-14)
        np.testing.assert_allclose(gain, response / 3, atol=1e-14)
        eigenvalues, basis = np.linalg.eigh(weights @ weights.conj().T)
        np.testing.assert_allclose(basis[:, 1:] @ basis[:, 1:].conj().T,
                                   projector, atol=1e-14)


def test_reset_and_causal_prefix():
    a = steering()
    first = np.array([1., 0., 0.])
    second = np.array([.5, np.sqrt(.75), 0.])
    directions = [first] * 4 + [second] * 4
    ratios = [4.] * 3 + [0.] + [4.] * 4

    def run(length):
        history = []
        return [beam_history_weight(rank_two_weights(null), a, ratio, history, 5)
                for null, ratio in zip(directions[:length], ratios[:length])]

    full = run(8)
    prefix = run(6)
    for actual, expected in zip(full[:6], prefix):
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        assert actual[2:] == expected[2:]
    assert full[3][2] == 0
    assert full[4][2] == 1
    assert full[7][2] == 4


def test_nontrigger_retains_weight_and_clears_history():
    a = steering()
    weights = rank_two_weights(np.array([1., 0., 0.]))
    history = [np.eye(3)]
    output = beam_history_weight(weights, a, 2., history, 5)
    assert output[0] is weights
    np.testing.assert_array_equal(output[1], np.ones(3))
    assert output[2] == 0 and np.isnan(output[3])
    assert history == []


@pytest.mark.parametrize('changed_case', ['weak', 'genuine_absent'])
def test_combined_finalize_rejects_changed_metadata_before_scoring(tmp_path, monkeypatch,
                                                                    changed_case):
    """A saved RF manifest must not label scores computed under new case metadata."""
    root = tmp_path / 'study'
    combined = root / 'beam_history_rf' / 'combined'
    combined.mkdir(parents=True)
    weak = tmp_path / 'weak'
    absent = tmp_path / 'absent'
    for folder in (weak, absent):
        folder.mkdir()
        (folder / 'metadata.json').write_text(json.dumps({
            'case': {'attack_start': .3},
            'samples_sha256': 'saved samples',
            'observations_sha256': 'saved observations'}),
                                               encoding='utf-8')
    input_hashes = {name: {'metadata.json': beam_rf.digest(folder / 'metadata.json'),
                           'samples.npy': 'saved samples',
                           'observations.npz': 'saved observations'}
                    for name, folder in (('weak', weak), ('genuine_absent', absent))}
    (combined.parent / 'summary.json').write_text('{}', encoding='utf-8')
    (root / 'pll').mkdir()
    (root / 'pll' / 'summary.json').write_text('{}', encoding='utf-8')
    selected = tmp_path / 'selected.json'
    selected.write_text(json.dumps({'selected_rows': [
        {'after_id': 'absent', 'after': {'initialization_time': .19}}]}),
        encoding='utf-8')
    (combined / 'manifest.json').write_text(json.dumps({
        'input_sha256': input_hashes,
        'rf_summary_sha256': beam_rf.digest(combined.parent / 'summary.json'),
        'pll_summary_sha256': beam_rf.digest(root / 'pll' / 'summary.json'),
        'selected_sha256': beam_rf.digest(selected)}), encoding='utf-8')
    (absent / 'metadata.json' if changed_case == 'genuine_absent'
     else weak / 'metadata.json').write_text(
         json.dumps({'case': {'attack_start': .5},
                     'samples_sha256': 'saved samples',
                     'observations_sha256': 'saved observations'}), encoding='utf-8')

    monkeypatch.setattr(beam_rf, 'ROOT', root)
    monkeypatch.setattr(beam_rf, 'DESTINATION', combined.parent)
    monkeypatch.setattr(beam_rf, 'COMBINED', combined)
    monkeypatch.setattr(beam_rf, 'INPUT', weak)
    monkeypatch.setattr(beam_rf, 'SELECTED', selected)
    monkeypatch.setitem(beam_rf.CASE_INPUTS, 'genuine_absent', (absent / 'tracking.npz', absent))
    with pytest.raises(AssertionError, match='metadata.json'):
        beam_rf.combined_finalize()
    assert not (combined / 'summary.json').exists()
