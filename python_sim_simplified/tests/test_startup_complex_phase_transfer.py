import json
from pathlib import Path

import numpy as np
import pytest

from experiments.spoofing import single_source_projection_long as study
from experiments.spoofing import single_source_startup_amplitude as startup
from experiments.spoofing.single_source_suppression_study import digest
from tools.spoofing import startup_complex_phase_transfer as transfer


def test_startup_principal_direction_is_global_phase_invariant():
    direction = np.array([1., 1j, -.4+1j])
    samples = np.exp(1j*np.arange(100)*.07)
    prompts = (direction[:, None]*samples[None])[:, None, :]
    u, acquired, degenerate, _ = transfer.startup_directions(prompts, 1, [0])
    assert acquired[0] and not degenerate[0]
    weights = np.zeros((3, 3, 1), complex)
    weights[0, :, 0] = [1., 0., 0.]
    weights[1, :, 0] = [1., .2j, 0.]
    weights[2, :, 0] = [.7, 0., .2]
    indices = np.array([1, 2])
    a = transfer.predict_phase(weights, u, np.zeros(3),
                               acquired, degenerate, indices)
    b = transfer.predict_phase(weights, u*np.exp(1j*.83), np.zeros(3),
                               acquired, degenerate, indices)
    np.testing.assert_allclose(a['prediction'], b['prediction'], atol=1e-14)


def test_nonzero_rate_from_fixed_raw_physics_to_boundary(monkeypatch):
    physical = np.array([1.+.2j, .3+.8j, -.4+1j])
    calibration = np.exp(1j*np.array([.2, -.4, .6]))
    rate = np.array([1.2, -.7, .4])
    samples = np.broadcast_to(physical[:, None],
                              (3, transfer.STARTUP)).copy()
    record = dict(samples=samples, codes=[None])
    monkeypatch.setattr(transfer, 'local_signal',
                        lambda code, template, times: np.ones(len(times), complex))
    prompts = transfer.reference_prompts(record, [0], [None], calibration, rate)
    u, acquired, degenerate, _ = transfer.startup_directions(prompts, 1, [0])
    weights = np.zeros((17, 3, 1), complex)
    weights[14, :, 0] = [1., 1., 1.]
    weights[15, :, 0] = [.3, 1j, .5]
    weights[16, :, 0] = [.1j, .4, 1.]
    predicted = transfer.predict_phase(weights, u, rate, acquired, degenerate,
                                       np.array([15, 16]))
    expected = []
    for b in (15, 16):
        at_boundary = physical/calibration*np.exp(-1j*rate*b*.02)
        before = np.vdot(weights[b-1, :, 0], at_boundary)
        after = np.vdot(weights[b, :, 0], at_boundary)
        expected.append(np.angle(after*before.conjugate()))
    np.testing.assert_allclose(predicted['prediction'][:, 0], expected,
                               atol=1e-14)


def test_zero_rate_reference_prompts_equal_original(monkeypatch):
    rng = np.random.default_rng(12)
    samples = rng.normal(size=(3, transfer.STARTUP)) + 1j*rng.normal(
        size=(3, transfer.STARTUP))
    record = dict(samples=samples, codes=[None])
    local = lambda code, template, times: np.ones(len(times), complex)
    monkeypatch.setattr(transfer, 'local_signal', local)
    monkeypatch.setattr(startup, 'local_signal', local)
    calibration = np.exp(1j*np.array([.2, -.4, .6]))
    new = transfer.reference_prompts(record, [0], [None], calibration,
                                     np.zeros(3))
    old = startup.startup_prompts(record, [0], [None], calibration,
                                  np.zeros(3))
    np.testing.assert_allclose(new, old, atol=1e-15)


def test_nominal_weight_phase_is_real_but_residual_direction_can_jump():
    nominal = np.array([1., 1., 1.], complex)
    projector = np.diag([0., 1., 1.])
    weights = np.stack((nominal/3,
                        projector@nominal/(np.vdot(nominal, projector@nominal).real+.01)))
    assert np.angle(np.vdot(weights[0], nominal)) == 0
    assert np.angle(np.vdot(weights[1], nominal)) == 0
    residual = np.array([1., 1j, .2])
    true = np.angle(np.vdot(weights[1], residual)
                    * np.vdot(weights[0], residual).conjugate())
    assert abs(true) > 1e-3
    score = transfer.loss_summary(np.array([[0.]]), np.array([[true]]),
                                  np.array([[True]]), np.array([[True]]))
    assert score['prediction_loss'] == score['zero_jump_loss']


def test_missing_degenerate_and_zero_response_are_distinct():
    prompts = np.zeros((3, 2, 100), complex)
    prompts[0, 0] = 1
    prompts[0, 1, :50] = 1
    prompts[1, 1, 50:] = 1
    u, acquired, degenerate, _ = transfer.startup_directions(prompts, 3,
                                                               [0, 1])
    assert degenerate[1]
    weights = np.zeros((2, 3, 3), complex)
    weights[0, 0, 0] = 1
    weights[1, 1, 0] = 1
    predicted = transfer.predict_phase(weights, u, np.zeros(3),
                                       acquired, degenerate, np.array([1]))
    assert predicted['zero'][0, 0]
    assert predicted['ambiguous'][0, 1]
    assert predicted['missing'][0, 2]
    assert not predicted['available'].any()
    assert np.isnan(predicted['prediction']).all()
    empty = transfer.loss_summary(predicted['prediction'],
                                  np.zeros((1, 3)), predicted['available'],
                                  np.ones((1, 3), bool))
    assert empty['predicted'] == 0 and empty['prediction_loss'] is None


def test_prediction_does_not_consult_truth(monkeypatch):
    monkeypatch.setattr(transfer, 'generator_truth',
                        lambda *args: (_ for _ in ()).throw(AssertionError()))
    weights = np.ones((2, 3, 1), complex)
    predicted = transfer.predict_phase(weights, np.ones((3, 1)),
                                       np.zeros(3),
                                       np.array([True]), np.array([False]),
                                       np.array([1]))
    np.testing.assert_allclose(predicted['prediction'], 0)


def test_zero_true_response_has_no_phase_label_for_scoring():
    weights = np.zeros((2, 3, 1), complex)
    weights[0, 0, 0] = 1
    weights[1, 1, 0] = 1
    nominal = np.array([[1.], [0.], [0.]])
    _, before, after = transfer.truth_after_prediction(
        weights, nominal, np.ones(3), np.ones(3), np.zeros(3),
        np.array([1]))
    assert before[0, 0] == 1 and after[0, 0] == 0
    available = (before > transfer.RESPONSE_ZERO) & (after > transfer.RESPONSE_ZERO)
    score = transfer.loss_summary(np.zeros((1, 1)), np.zeros((1, 1)),
                                  available, np.ones((1, 1), bool))
    assert score['predicted'] == 0 and score['zero_jump_loss'] is None


@pytest.mark.local_evidence
def test_saved_startup_reproduces_calibration_and_covers_all_boundaries():
    frozen_path = study.OUTPUT/'summary.json'
    assert digest(frozen_path) == transfer.STAGE47_SUMMARY_SHA256
    frozen = json.loads(frozen_path.read_text(encoding='utf-8'))
    root = Path(transfer.__file__).resolve().parents[1]
    for name in transfer.SOURCES:
        assert digest(root/name) == frozen['source_sha256'][name]
    case = frozen['cases']['2026100404']['weak']
    folder = study.OUTPUT/'2026100404'/'weak'
    summary, arrays = transfer.analyze_path(
        folder, case, folder/'input'/'metadata.json',
        case['input_sha256']['metadata.json'])
    assert summary['boundary_count'] == 985
    assert summary['opportunity_count'] == 985*24
    assert (summary['missing']+summary['ambiguous']+summary['zero_response']
            +summary['predictable'] == summary['opportunity_count'])
    assert arrays['boundary_s'][0] == .3
