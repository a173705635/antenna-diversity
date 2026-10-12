"""The optional projection gate changes spatial weights, not the default path."""
import inspect
from pathlib import Path
from types import ModuleType
from zipfile import ZipFile

import numpy as np
import pytest

from core.spoofing import single_source
from core.spoofing.recovery_geometry import FS


STEERING = np.array([[1., 1., 1.], [1., 1j, -1.], [1., -1., 1j]])
PROJECTOR = np.diag([0., 1., 1.]).astype(complex)


@pytest.mark.parametrize('raw,project_at_two,project_at_three', [
    (1.999, False, False),
    (2., True, False),
    (2.5, True, False),
    (3., True, True),
])
def test_beam_projection_gate_boundaries(monkeypatch, raw,
                                         project_at_two, project_at_three):
    monkeypatch.setattr(single_source, 'residual_trigger',
                        lambda *args: (PROJECTOR, raw))
    block = np.ones((3, 2046), complex)
    local = np.ones((1, 2046), complex)
    projected = PROJECTOR @ STEERING
    response = np.sum(STEERING.conj() * projected, axis=0).real
    for gate, active in ((2., project_at_two), (3., project_at_three)):
        weights, gain, score = single_source.beam(
            block, STEERING, local, np.eye(3), [0],
            beam_projection_threshold=gate)
        expected_weights = (projected / (response + .01)[None, :]
                            if active else STEERING / 3.)
        expected_gain = response / 3. if active else np.ones(3)
        np.testing.assert_array_equal(weights, expected_weights)
        np.testing.assert_array_equal(gain, expected_gain)
        assert score == raw


def test_temporal_gate_uses_mean_and_preserves_current_raw():
    history = [2.8]
    mean, effective, history_only = single_source.temporal_beam_trigger(
        1.5, history, beam_projection_threshold=2.)
    assert (mean, effective, history_only) == (2.15, 2.15, True)
    assert history == [2.8, 1.5]


def test_rank_two_history_resets_at_shared_gate():
    weights = PROJECTOR @ STEERING
    history = []
    retained, gain, count, _ = single_source.beam_history_weight(
        weights, STEERING, 2., history, 5, beam_projection_threshold=2.)
    assert count == 1 and len(history) == 1
    assert retained.shape == weights.shape
    assert gain.shape == (3,)
    returned, gain, count, gap = single_source.beam_history_weight(
        weights, STEERING, 1.999, history, 5, beam_projection_threshold=2.)
    assert returned is weights and history == [] and count == 0
    np.testing.assert_array_equal(gain, np.ones(3))
    assert np.isnan(gap)


def test_default_signatures_keep_gate_three():
    for function in (single_source.temporal_beam_trigger, single_source.beam,
                     single_source.beam_history_weight, single_source.frontend):
        assert inspect.signature(function).parameters[
            'beam_projection_threshold'].default == 3.


def stub_frontend(module, monkeypatch, raw):
    """Exercise real beam/basis/stream wiring with only slow RF tracking stubbed."""
    monkeypatch.setattr(module, 'acquire_array',
                        lambda *args, **kwargs: {'success': True})
    starts = np.arange(0., .161, .001)
    history = dict(epoch_start_samples=np.rint(starts*FS).astype(int),
                   code_nco_phase_chips=np.zeros(len(starts)),
                   code_nco_rate_hz=np.zeros(len(starts)),
                   carrier_nco_phase_rad=np.zeros(len(starts)),
                   carrier_nco_frequency_hz=np.zeros(len(starts)),
                   phase_lock_metric=np.ones(len(starts)),
                   cn0_db_hz=np.full(len(starts), 30.))
    monkeypatch.setattr(module, 'track',
                        lambda *args, **kwargs: {'success': True, **history})
    monkeypatch.setattr(module, 'local_signal',
                        lambda code, template, t: np.ones(len(t), complex))
    monkeypatch.setattr(module, '_sample_code',
                        lambda code, phase: np.ones(len(phase)))
    monkeypatch.setattr(module, 'calibrate',
                        lambda block, local, desired: (np.ones(3, complex), np.zeros(3)))
    monkeypatch.setattr(module, 'startup_relative_amplitude',
                        lambda *args: (np.ones(3), None))
    streams = []

    def receive(signal, prn, *args):
        if prn == 1:
            streams.append(signal.copy())
        return {'segments': [{'history': history}]}

    monkeypatch.setattr(module, 'receive_measured', receive)
    monkeypatch.setattr(module, 'code_pseudorange',
                        lambda h, t, *args: np.zeros(len(t)))
    monkeypatch.setattr(module, 'carrier_range',
                        lambda h, t, *args: np.zeros(len(t)))
    scores = iter(raw) if isinstance(raw, tuple) else None
    monkeypatch.setattr(module, 'residual_trigger',
                        lambda *args: (PROJECTOR, next(scores) if scores else raw))
    rng = np.random.default_rng(17)
    samples = rng.normal(size=(3, round(.16*FS)))+1j*rng.normal(size=(3, round(.16*FS)))
    nominal = np.column_stack((STEERING, STEERING[:, 0]))
    record = dict(samples=samples, nominal=nominal,
                  codes=np.zeros((4, 1)), times=np.array([.10, .12]),
                  satellites=np.zeros((2, 4, 3)), coarse=np.zeros(4))
    return record, streams


def test_frontend_gate_two_changes_actual_stream_and_basis(monkeypatch):
    record, streams = stub_frontend(single_source, monkeypatch, 2.5)
    nominal = record['nominal']
    samples = record['samples']
    default = single_source.frontend(record, [0, 1, 2], evidence_trace=True,
                                     split_support=True)
    explicit = single_source.frontend(record, [0, 1, 2], evidence_trace=True,
                                     split_support=True,
                                     beam_projection_threshold=3.)
    candidate = single_source.frontend(record, [0, 1, 2], evidence_trace=True,
                                       split_support=True,
                                       beam_projection_threshold=2.)
    assert default.keys() == explicit.keys()
    for name in default:
        np.testing.assert_array_equal(default[name], explicit[name], err_msg=name)
    begin, end = round(.10*FS), round(.12*FS)
    projected = PROJECTOR @ nominal
    response = np.sum(nominal.conj()*projected, axis=0).real
    weight = projected/(response+.01)[None, :]
    np.testing.assert_array_equal(default['trace_projectors'][5], np.eye(3))
    np.testing.assert_allclose(candidate['trace_projectors'][5], PROJECTOR, atol=1e-14)
    assert default['trace_dimensions'][5] == 3
    assert candidate['trace_dimensions'][5] == 2
    np.testing.assert_allclose(streams[0][begin:end],
                               nominal[:, 0].conj()@samples[:, begin:end]/3.)
    np.testing.assert_array_equal(streams[0], streams[1])
    np.testing.assert_allclose(streams[2][begin:end],
                               weight[:, 0].conj()@samples[:, begin:end])


@pytest.mark.local_evidence
def test_default_beam_temporal_and_history_match_stage44_source(monkeypatch):
    snapshot = (Path(single_source.__file__).resolve().parents[1]/
                'results/single_source_suppression_study/low_power_amplitude_rf'/
                'source_snapshot.zip')
    with ZipFile(snapshot) as archive:
        source = archive.read('core/single_source.py').decode('utf-8')
    old = ModuleType('core._single_source_stage44')
    old.__package__ = 'core'
    exec(compile(source, str(snapshot), 'exec'), old.__dict__)
    block = np.ones((3, 2046), complex)
    local = np.ones((1, 2046), complex)
    for raw in (1.999, 2., 2.5, 3., 4.):
        for module in (old, single_source):
            monkeypatch.setattr(module, 'residual_trigger',
                                lambda *args, score=raw: (PROJECTOR, score))
        old_output = old.beam(block, STEERING, local, np.eye(3), [0])
        new_output = single_source.beam(block, STEERING, local, np.eye(3), [0])
        for expected, actual in zip(old_output, new_output):
            np.testing.assert_array_equal(expected, actual)
    for raw in ([4., 1.5, 2.8, 3.], [2.8, 1.5, 1.999, 3.]):
        old_history = []
        new_history = []
        for value in raw:
            assert old.temporal_beam_trigger(value, old_history) == (
                single_source.temporal_beam_trigger(value, new_history))
        assert old_history == new_history
    weights = PROJECTOR @ STEERING
    old_history = []
    new_history = []
    for ratio in (3., 4., 1.999, 3.):
        previous = old.beam_history_weight(
            weights, STEERING, ratio, old_history, 5)
        current = single_source.beam_history_weight(
            weights, STEERING, ratio, new_history, 5)
        for expected, actual in zip(previous, current):
            np.testing.assert_array_equal(expected, actual)
        for expected, actual in zip(old_history, new_history):
            np.testing.assert_array_equal(expected, actual)
    combinations = (
        ({}, (3.5, 2.5, 4.)),
        ({'temporal_beam_trigger_enabled': True}, (8., 1., 1.)),
        ({'beam_history_blocks': 5}, (4., 4., 1.)),
    )
    for options, scores in combinations:
        old_record, old_streams = stub_frontend(old, monkeypatch, scores)
        old_result = old.frontend(old_record, [0, 1, 2], split_support=True,
            evidence_trace=True, beam_template_carrier_aiding=True,
            beam_coherent_ms=5, beam_startup_amplitude=True, **options)
        new_record, new_streams = stub_frontend(single_source, monkeypatch, scores)
        new_result = single_source.frontend(new_record, [0, 1, 2],
            split_support=True, evidence_trace=True,
            beam_template_carrier_aiding=True, beam_coherent_ms=5,
            beam_startup_amplitude=True, **options)
        assert old_result.keys() == new_result.keys()
        for name in old_result:
            np.testing.assert_array_equal(old_result[name], new_result[name],
                                          err_msg=f'{options}: {name}')
        np.testing.assert_array_equal(old_streams[0], new_streams[0])
