"""Small, synthetic checks for the fixed mixed-stream oracle intervention."""
import json
import numpy as np
import pytest
from types import FunctionType

from tools import oracle_phase_transport as oracle


def test_oracle_factor_uses_first_sample_reference_and_physical_sign():
    weight = np.array([1. + 0j, .4 + .3j])
    direction = np.array([.8 + .2j, .2 - .1j])
    rate = np.array([0., 41.])
    first = np.vdot(weight, direction)
    samples = np.array([0, 1, 10, oracle.WIDTH - 1])
    factor = oracle.oracle_factor(weight, direction, rate, samples, first)
    response = np.array([
        np.vdot(weight, direction * np.exp(-1j * rate * index / oracle.FS))
        for index in samples])
    expected = np.exp(-1j * (np.angle(response) - np.angle(first)))
    np.testing.assert_allclose(factor, expected, atol=2e-15)
    np.testing.assert_allclose(abs(factor), 1., atol=2e-15)
    np.testing.assert_allclose(factor[0], 1., atol=2e-15)


def test_block_boundary_rotates_whole_mixture_without_changing_ratio(monkeypatch):
    monkeypatch.setattr(oracle, 'WIDTH', 3)
    monkeypatch.setattr(oracle, 'ANTENNAS', [0])
    weights = np.array([[[1. + 0j]], [[0. + 1j]]])
    direction = np.array([[1. + 0j]])
    true = np.ones(6, complex)
    spoof = np.array([.2 + .3j] * 6)
    noise = np.array([.04 - .07j] * 6)
    signal = true + spoof + noise
    original = signal.copy()
    raw = original.copy()
    raw[3:] *= 1j
    oracle.rotate_mixed_stream(signal, 0, weights, direction, np.zeros(1),
                               raw[None], np.ones(1))
    np.testing.assert_allclose(signal[:3], original[:3])
    np.testing.assert_allclose(signal[3:], 1j * original[3:])
    np.testing.assert_allclose(abs(signal), abs(original))
    factor = signal / original
    np.testing.assert_allclose((factor * spoof) / (factor * true), spoof / true)


def test_h0_constant_weight_zero_rate_is_identity_and_zero_response_fails(monkeypatch):
    monkeypatch.setattr(oracle, 'WIDTH', 2)
    monkeypatch.setattr(oracle, 'ANTENNAS', [0])
    signal = np.array([1. + 2j, -3. + .5j, 4. - 1j, .2 + .1j])
    original = signal.copy()
    weights = np.repeat(np.array([[[.4 + .7j]]]), 2, axis=0)
    raw = (original / weights[0, 0, 0].conjugate())[None]
    oracle.rotate_mixed_stream(signal, 0, weights, np.array([[.2 + .3j]]),
                               np.zeros(1), raw, np.ones(1))
    np.testing.assert_allclose(signal, original, atol=1e-15)
    with pytest.raises(RuntimeError, match='zero true response'):
        oracle.rotate_mixed_stream(signal, 0, np.zeros_like(weights),
                                   np.array([[1. + 0j]]), np.zeros(1),
                                   raw, np.ones(1))


def test_private_receive_hook_matches_baseline_for_identity(monkeypatch, tmp_path):
    monkeypatch.setattr(oracle, 'ANTENNAS', [0])
    def fake_frontend(record, subset, **options):
        return receive_measured(record['signal'].copy(), 1, None, None,
                                fast_truncation=True)

    def fake_receive(signal, prn, acquisition, tracking, **options):
        return dict(signal=np.asarray(signal).copy(), prn=prn,
                    options=options)

    fake_frontend = FunctionType(fake_frontend.__code__,
                                 dict(fake_frontend.__globals__,
                                      receive_measured=fake_receive))
    monkeypatch.setattr(oracle, 'frontend', fake_frontend)
    monkeypatch.setattr(oracle, 'receive_measured', fake_receive)
    record = dict(signal=np.array([1. + .2j, -.4 + .7j, 2. + 3j]))
    record['samples'] = record['signal'][None]
    weights = np.ones((1, 1, 1), complex)
    baseline = oracle.run_frontend(fake_frontend, record, tmp_path/'base', [])
    candidate = oracle.private_frontend_with_oracle(
        record, weights, np.ones((1, 1), complex), np.ones(1), np.zeros(1),
        tmp_path/'oracle', [])
    np.testing.assert_array_equal(candidate['signal'], baseline['signal'])
    assert candidate['options'] == baseline['options']
    assert fake_frontend.__globals__['receive_measured'] is fake_receive


def test_saved_weights_must_reconstruct_baseline_mixed_stream(monkeypatch, tmp_path):
    monkeypatch.setattr(oracle, 'WIDTH', 2)
    monkeypatch.setattr(oracle, 'ANTENNAS', [0])
    samples = np.array([[1. + 1j, 2. - 1j, .3 + .2j, 1. - .5j]])
    weights = np.array([[[1. + 0j]], [[0. + 1j]]])
    expected = np.r_[samples[0, :2], -1j * samples[0, 2:]]
    scratch = tmp_path/'stream.dat'
    stream = np.memmap(scratch, dtype=np.complex128, mode='w+', shape=(1, 4))
    stream[0] = expected
    stream.flush()
    del stream
    record = dict(samples=samples, codes=[0])
    fit = dict(weight=weights)
    tracking = dict(calibration=np.ones(1), calibration_rate=np.zeros(1))
    oracle.verify_fit_stream(record, scratch, fit, tracking)
    stream = np.memmap(scratch, dtype=np.complex128, mode='r+', shape=(1, 4))
    stream[0, 2] += .01
    stream.flush()
    del stream
    with pytest.raises(AssertionError, match='beam block 1'):
        oracle.verify_fit_stream(record, scratch, fit, tracking)


def test_oracle_rejects_unrotated_stream_drift_before_changing_it(monkeypatch):
    monkeypatch.setattr(oracle, 'WIDTH', 2)
    monkeypatch.setattr(oracle, 'ANTENNAS', [0])
    raw = np.array([[1. + .2j, 2. - .1j]])
    signal = raw[0].copy()
    signal[0] += .01
    before = signal.copy()
    with pytest.raises(AssertionError, match='pre-oracle PRN 1 block 0'):
        oracle.rotate_mixed_stream(signal, 0, np.ones((1, 1, 1), complex),
                                   np.ones((1, 1), complex), np.zeros(1),
                                   raw, np.ones(1))
    np.testing.assert_array_equal(signal, before)


def test_oracle_frontend_fixed_quantities_reject_changes():
    tracking = {key: np.array([1.]) for key in
                ('time', 'available_time', 'satellites', 'startup_prns',
                 'calibration', 'calibration_rate', 'startup_relative_amplitude',
                 'gain', 'ratio')}
    trace = {key: np.array([1.]) for key in
             ('trace_raw_trigger', 'trace_projectors',
              'trace_dimensions', 'trace_nominal')}
    oracle.verify_oracle_frontend(tracking, trace, tracking, trace)
    changed = dict(tracking, gain=np.array([2.]))
    with pytest.raises(AssertionError, match='oracle frontend gain'):
        oracle.verify_oracle_frontend(changed, trace, tracking, trace)
    changed_trace = dict(trace, trace_projectors=np.array([2.]))
    with pytest.raises(AssertionError, match='oracle frontend trace_projectors'):
        oracle.verify_oracle_frontend(tracking, changed_trace, tracking, trace)


def test_short_replay_does_not_run_original_navigation(monkeypatch, tmp_path):
    calls = []
    def navigate(inputs, **kwargs):
        calls.append(kwargs)
        return dict(position=np.array([[1., 2., 3.]]),
                    initialization_time=0., diagnostics=dict(phase_ok=np.array([True])))
    monkeypatch.setattr(oracle, 'navigate', navigate)
    monkeypatch.setattr(oracle, 'projection_eligibility',
                        lambda available, trace: (np.array([True]),
                            np.array([0]), np.array([3])))
    monkeypatch.setattr(oracle, 'navigation_input',
                        lambda tracking, support: dict(support=support))
    monkeypatch.setattr(oracle, 'quality_support_counts',
                        lambda tracking, support: np.array([6]))
    monkeypatch.setattr(oracle, 'check_prefix', lambda *args: None)
    monkeypatch.setattr(oracle, 'score_navigation',
                        lambda *args: dict(max_relative_motion_error_m=1.))
    tracking = dict(time=np.array([.3]), available_time=np.array([.32]))
    scores, positions = oracle.replay_short_navigation(
        tracking, {}, dict(support=np.array([[4.]])),
        dict(velocity_east=2.), tmp_path)
    assert len(calls) == 1
    assert calls[0] == dict(diagnostics=True, code_blend=0.,
                            phase_leave_one_out=True,
                            phase_state_continuity=True,
                            phase_state_eligible=np.array([True]))
    assert set(scores) == {'short'} and set(positions) == {'short'}
    with np.load(tmp_path/'navigation.npz') as saved:
        assert 'short_position' in saved and not any(
            key.startswith('original_') for key in saved.files)


def test_preflight_checks_all_98_source_hashes(monkeypatch, tmp_path):
    stage = oracle.stage47.OUTPUT
    summary = json.loads((stage/'summary.json').read_text(encoding='utf-8'))
    expected = dict(summary['source_sha256'])
    assert len(expected) == 98
    last = next(reversed(expected))
    expected[last] = '0' * 64
    with pytest.raises(AssertionError, match=last.replace('.', r'\.')):
        oracle.verify_current_sources(oracle.Path(oracle.__file__).resolve().parents[1],
                                      expected)


def test_source_snapshot_is_reusable_without_rewriting(tmp_path):
    first = oracle.snapshot_extra_sources(tmp_path)
    second = oracle.snapshot_extra_sources(tmp_path)
    assert first == second


def test_fixed_screen_requires_each_seed_motion_and_no_lost_publications():
    time = np.array([.28, .30, .32, .34])
    old = np.array([[np.nan, np.nan], [.6, 0], [.64, 0], [.68, 0]])
    new = np.array([[np.nan, np.nan], [.6, 0], [.62, 0], [.64, 0]])
    good = oracle.score_pair(time, old, new, 2., 2., 1.)
    assert good['pass_rule']
    assert good['opportunities'] == 3
    assert not oracle.score_pair(time, old, new, 2., 2., 1.01)['pass_rule']
    new[2] = np.nan
    assert not oracle.score_pair(time, old, new, 2., 2., .5)['pass_rule']
    new[2] = [.62, 0]
    new[3] = [25., 0]
    assert not oracle.score_pair(time, old, new, 2., 2., .5)['pass_rule']
