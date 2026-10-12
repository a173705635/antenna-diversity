import json
from pathlib import Path

import numpy as np
import pytest

from experiments.spoofing import single_source_projection_long as study
from experiments.spoofing.single_source_suppression_study import digest
from tools.spoofing import common_residual_phase_transfer as common
from tools.spoofing import startup_complex_phase_transfer as phase


def test_complex_ratios_have_correct_sign_and_equal_prn_covariance():
    nominal = np.array([[1, 1j], [1j, -1], [1, -1j]], complex)
    physical = np.array([1, .8*np.exp(.4j), 1.2*np.exp(-.7j)])
    power = np.array([1., 4.])
    prompts = nominal[:, :, None]*physical[:, None, None]*power[None, :, None]
    prompts = np.broadcast_to(prompts, (3, 2, 100)).copy()
    amplitude = np.array([1., .9, 1.1])
    fit = common.estimate_common_gain(prompts, nominal, np.array([0, 1]),
                                       amplitude)
    expected_cov = np.mean(power**2)*np.outer(physical, physical.conj())
    np.testing.assert_allclose(fit['covariance'], expected_cov, atol=1e-14)
    np.testing.assert_allclose(fit['phase'], [0., .4, -.7], atol=1e-14)
    np.testing.assert_allclose(fit['gain'], amplitude[:, None]
                               * np.exp(1j*fit['phase'][:, None])*nominal)


def test_reference_antenna_permutation_only_changes_global_phase():
    physical = np.array([1., .7*np.exp(.35j), 1.1*np.exp(-.55j)])
    nominal = np.ones((3, 1), complex)
    prompts = np.broadcast_to(physical[:, None, None], (3, 1, 100)).copy()
    amplitude = np.abs(physical)
    direct = common.estimate_common_gain(prompts, nominal, np.array([0]),
                                          amplitude)['gain'][:, 0]
    order = np.array([1, 0, 2])
    swapped = common.estimate_common_gain(prompts[order], nominal[order],
                                           np.array([0]), amplitude[order])[
                                               'gain'][:, 0]
    unpermuted = swapped[np.argsort(order)]
    ratio = unpermuted/direct
    np.testing.assert_allclose(ratio, np.full(3, ratio[0]), atol=1e-14)


def test_machine_zero_cross_covariance_marks_whole_path_unidentified():
    nominal = np.ones((3, 1), complex)
    prompts = np.zeros((3, 1, 100), complex)
    prompts[1] = 1
    prompts[2] = 1j
    fit = common.estimate_common_gain(prompts, nominal, np.array([0]),
                                       np.ones(3))
    assert not fit['identifiable']
    weights = np.ones((3, 3, 1), complex)
    predicted = common.predict_common(weights, fit, np.zeros(3),
                                      np.array([1, 2]))
    assert predicted['common_unidentified'].all()
    assert not predicted['available'].any()
    assert np.isnan(predicted['prediction']).all()


def test_nonzero_rate_from_fixed_physical_raw_samples_to_boundary(monkeypatch):
    physical = np.array([1.+.2j, .3+.8j, -.4+1j])
    calibration = np.exp(1j*np.array([.2, -.4, .6]))
    rate = np.array([1.2, -.7, .4])
    raw = np.broadcast_to(physical[:, None],
                          (3, phase.STARTUP)).copy()
    record = dict(samples=raw, codes=[None])
    monkeypatch.setattr(phase, 'local_signal',
                        lambda code, template, times: np.ones(len(times), complex))
    prompts = phase.reference_prompts(record, [0], [None], calibration, rate)
    true_ref = physical/calibration*np.exp(-1j*rate*phase.REFERENCE_S)
    amplitude = np.abs(true_ref)/abs(true_ref[0])
    fit = common.estimate_common_gain(prompts, np.ones((3, 1)),
                                       np.array([0]), amplitude)
    weights = np.zeros((17, 3, 1), complex)
    weights[14, :, 0] = [1., 1., 1.]
    weights[15, :, 0] = [.3, 1j, .5]
    weights[16, :, 0] = [.1j, .4, 1.]
    predicted = common.predict_common(weights, fit, rate, np.array([15, 16]))
    expected = []
    for b in (15, 16):
        spatial = physical/calibration*np.exp(-1j*rate*b*.02)
        before = np.vdot(weights[b-1, :, 0], spatial)
        after = np.vdot(weights[b, :, 0], spatial)
        expected.append(np.angle(after*before.conjugate()))
    np.testing.assert_allclose(predicted['prediction'][:, 0], expected,
                               atol=1e-14)


def test_estimator_does_not_consult_truth(monkeypatch):
    monkeypatch.setattr(common, 'generator_truth',
                        lambda *args: (_ for _ in ()).throw(AssertionError()))
    nominal = np.ones((3, 1))
    prompts = np.ones((3, 1, 100), complex)
    fit = common.estimate_common_gain(prompts, nominal, np.array([0]),
                                       np.ones(3))
    predicted = common.predict_common(np.ones((2, 3, 1), complex), fit,
                                      np.zeros(3), np.array([1]))
    np.testing.assert_allclose(predicted['prediction'], 0)


@pytest.mark.local_evidence
def test_saved_amplitude_model_is_reproduced_without_double_application():
    frozen_path = study.OUTPUT/'summary.json'
    assert digest(frozen_path) == common.STAGE47_SUMMARY_SHA256
    frozen = json.loads(frozen_path.read_text(encoding='utf-8'))
    phase_path = study.OUTPUT/'startup_complex_phase_transfer_summary.json'
    assert digest(phase_path) == common.FROZEN_PHASE_SUMMARY_SHA256
    assert digest(Path(phase.__file__)) == json.loads(
        phase_path.read_text(encoding='utf-8'))['source_sha256']
    root = Path(common.__file__).resolve().parents[1]
    for name in common.SOURCES:
        assert digest(root/name) == frozen['source_sha256'][name]
    folder = study.OUTPUT/'2026100404'/'weak'
    summary, arrays = common.analyze_path(folder,
                                          frozen['cases']['2026100404']['weak'])
    assert summary['opportunity_count'] == 985*24
    assert summary['predictable']+summary['common_unidentified']+summary[
        'zero_response'] == summary['opportunity_count']
    with np.load(folder/'input'/'observations.npz') as saved:
        nominal = saved['nominal'][:3]
    with np.load(folder/'gate3'/'trace.npz') as saved:
        trace_nominal = saved['trace_nominal']
    np.testing.assert_array_equal(trace_nominal,
                                  arrays['saved_relative_amplitude'][:, None]
                                  * nominal)
    np.testing.assert_allclose(np.abs(arrays['common_gain']/nominal),
                               np.broadcast_to(
                                   arrays['saved_relative_amplitude'][:, None],
                                   nominal.shape),
                               atol=1e-14)
