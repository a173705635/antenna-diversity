"""Synthetic checks of the frozen-beam paired-H0 full-receiver boundary."""
from types import FunctionType
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from tools.spoofing import frozen_weak_h0_counterfactual as study


def test_paired_prefix_and_all_observation_fields_are_required():
    weak = {name: np.arange(4) for name in
            ('nominal', 'codes', 'times', 'satellites', 'coarse')}
    weak['samples'] = np.ones((2, 5), complex)
    h0 = {name: value.copy() for name, value in weak.items()}
    h0['samples'][:, 3:] = 2.
    study.verify_paired_records(weak, h0, 3)
    h0['samples'][0, 2] = 3.
    with pytest.raises(AssertionError, match='paired pre-attack IQ'):
        study.verify_paired_records(weak, h0, 3)
    h0['samples'][0, 2] = 1.
    h0['satellites'][0] = 99
    with pytest.raises(AssertionError, match='satellites'):
        study.verify_paired_records(weak, h0, 3)


def test_all_sample_fake_direction_budget_and_bad_direction():
    direction = np.array([1. + .2j, -.3 + .4j, .7 - .1j, 1j])
    scalar = np.array([0., 0., .3 + .5j, -.2 + .1j, 0., .8j])
    h0 = np.array([[.2 + .1j] * len(scalar)] * 4)
    weak = h0 + direction[:, None] * scalar[None, :]
    metrics = study.paired_direction_residual(
        weak, h0, direction, np.ones(4), chunk_samples=2)
    assert metrics['max_abs_iq'] < study.IQ_RESIDUAL_MAX
    assert metrics['rms_iq'] < study.IQ_RESIDUAL_MAX
    weak[2, 3] += 2e-10
    with pytest.raises(AssertionError):
        study.paired_direction_residual(
            weak, h0, direction, np.ones(4), chunk_samples=2)


def test_every_h0_stream_block_is_checked_before_receiver(monkeypatch):
    monkeypatch.setattr(study, 'WIDTH', 2)
    monkeypatch.setattr(study, 'ANTENNAS', [0])
    h0 = np.array([[1. + .2j, 2. - .1j, .3 + .4j, -.8 + .1j]])
    weights = np.array([[[1. + 0j]], [[0. + 1j]]])
    fit = dict(weight=weights)
    signal = np.r_[h0[0, :2], -1j * h0[0, 2:]]
    record = dict(samples=h0)
    study.verify_h0_stream(signal, 0, record, fit, np.ones(1), np.zeros(1))
    wrong_h0 = dict(samples=h0.copy())
    wrong_h0['samples'][0, 2] += .01
    with pytest.raises(AssertionError, match='H0 beam PRN 1 block 1'):
        study.verify_h0_stream(signal, 0, wrong_h0, fit, np.ones(1), np.zeros(1))
    wrong_fit = dict(weight=weights.copy())
    wrong_fit['weight'][0, 0, 0] += .01
    with pytest.raises(AssertionError, match='H0 beam PRN 1 block 0'):
        study.verify_h0_stream(signal, 0, record, wrong_fit,
                               np.ones(1), np.zeros(1))


def test_private_globals_keep_startup_and_map_all_beam_blocks(monkeypatch):
    calls = []
    def original_calibration(block, local, desired):
        calls.append('calibration')
        return np.array([1. + 0j]), np.array([0.])
    def original_receiver(signal, prn, acquisition, tracking, **options):
        calls.append('receiver')
        return dict(signal=np.array(signal, copy=True))
    def original_beam(*args, **kwargs):
        raise AssertionError('original beam should not refit')
    def fake_frontend(record, subset, **options):
        result = calibrate(None, None, None)
        outputs = [beam(None, None, None, None, None, coherent_ms=5)
                   for _ in range(2)]
        received = receive_measured(record['signal'], 1, None, None)
        return dict(startup=result, beam=outputs, received=received,
                    source=record['samples'][0, 0])

    fake_frontend = FunctionType(fake_frontend.__code__,
        dict(fake_frontend.__globals__, calibrate=original_calibration,
             beam=original_beam, receive_measured=original_receiver))
    monkeypatch.setattr(study, 'frontend', fake_frontend)
    monkeypatch.setattr(study, 'calibrate', original_calibration)
    monkeypatch.setattr(study, 'receive_measured', original_receiver)
    monkeypatch.setattr(study, 'verify_h0_stream',
                        lambda signal, star, record, fit, gain, rate: calls.append('verified'))
    monkeypatch.setattr(study.oracle, 'run_frontend',
        lambda receiver, record, scratch, loops: receiver(record, [0]))
    fit = dict(weight=np.arange(7).reshape(7, 1, 1).astype(complex),
               gain=np.arange(7).reshape(7, 1).astype(float),
               raw=np.arange(7).astype(float))
    tracking = dict(calibration=np.array([1. + 0j]),
                    calibration_rate=np.array([0.]))
    record = dict(samples=np.array([[3. + 0j]]), codes=[1],
                  signal=np.array([3. + 0j]))
    result = study.frozen_frontend(record, fit, tracking, None, [])
    assert [entry[0][0, 0] for entry in result['beam']] == [5., 6.]
    assert [entry[1][0] for entry in result['beam']] == [5., 6.]
    assert [entry[2] for entry in result['beam']] == [5., 6.]
    assert calls == ['calibration', 'verified', 'receiver']
    assert fake_frontend.__globals__['beam'] is original_beam
    assert result['source'] == record['samples'][0, 0]
    fit['raw'] = fit['raw'][:-1]
    with pytest.raises(AssertionError):
        study.frozen_frontend(record, fit, tracking, None, [])


def test_projector_tolerance_and_strict_gate_side():
    tracking = {key: np.array([1.]) for key in
                ('time', 'available_time', 'satellites', 'startup_prns',
                 'calibration', 'calibration_rate',
                 'startup_relative_amplitude', 'gain', 'ratio')}
    trace = dict(trace_nominal=np.array([1.]),
                 trace_raw_trigger=np.array([3.2]),
                 trace_projectors=np.array([[[1.]]]),
                 trace_dimensions=np.array([2]))
    candidate = dict(trace, trace_projectors=trace['trace_projectors'] + 5e-13)
    study.fixed_frontend_comparison(tracking, candidate, tracking, trace)
    candidate['trace_projectors'] = trace['trace_projectors'] + 2e-12
    with pytest.raises(AssertionError):
        study.fixed_frontend_comparison(tracking, candidate, tracking, trace)
    near = dict(trace, trace_raw_trigger=np.array([3. - 4e-13]))
    crossed = dict(near, trace_raw_trigger=np.array([3. + 4e-13]))
    with pytest.raises(AssertionError):
        study.fixed_frontend_comparison(tracking, crossed, tracking, near)


def test_old_oracle_snapshot_rejects_tampered_current_helper(tmp_path, monkeypatch):
    names = ('tools/oracle_phase_transport.py', 'tools/resume_projection_long.py')
    monkeypatch.setattr(study.oracle, 'EXTRA_SOURCES', names)
    root = tmp_path/'source'
    (root/'tools/spoofing').mkdir(parents=True)
    for name in names:
        (root/'tools/spoofing'/Path(name).name).write_bytes(name.encode())
    snapshot = tmp_path/'old_oracle.zip'
    with ZipFile(snapshot, 'w') as archive:
        for name in names:
            archive.writestr(name, (root/'tools/spoofing'/Path(name).name).read_bytes())
    sha = study.digest(snapshot)
    study.verify_oracle_helper_snapshot(snapshot, sha, root)
    (root/'tools/spoofing'/Path(names[0]).name).write_bytes(b'tampered current helper')
    with pytest.raises(AssertionError, match='oracle_phase_transport.py'):
        study.verify_oracle_helper_snapshot(snapshot, sha, root)
    (root/'tools/spoofing'/Path(names[0]).name).write_bytes(names[0].encode())
    with pytest.raises(AssertionError):
        study.verify_oracle_helper_snapshot(snapshot, '0' * 64, root)


def test_completed_receipt_rejects_changed_source_and_frozen_reference(tmp_path):
    seed = study.SEEDS[0]
    folder = tmp_path/'candidate'
    folder.mkdir()
    source = Path(study.__file__)
    archived = folder/'frozen_weak_h0_counterfactual.py'
    archived.write_bytes(source.read_bytes())
    saved = tmp_path/'weak'
    saved.mkdir()
    (saved/'fit.npz').write_bytes(b'frozen fit')
    outputs = {}
    for name in study.SCIENCE:
        (folder/name).write_bytes(name.encode())
        outputs[name] = study.digest(folder/name)
    inputs = {'weak': {'samples.npy': 'input hash'}}
    direction = dict(max_abs_iq=0., rms_iq=0., relative_norm=0.)
    receipt = dict(seed=seed,
                   source_sha256=study.digest(source),
                   stage47_summary_sha256=study.oracle.STAGE47_SUMMARY_SHA256,
                   oracle_summary_sha256=study.ORACLE_SUMMARY_SHA256,
                   oracle_baseline_receipt_sha256=study.ORACLE_RECEIPT_SHA256[seed],
                   weak_fit_sha256=study.digest(saved/'fit.npz'),
                   input_sha256=inputs, direction_residual=direction,
                   output_sha256=outputs)
    study.verify_completed_receipt(seed, folder, receipt, inputs, saved, direction)
    archived.write_bytes(b'different archived implementation')
    receipt['source_sha256'] = study.digest(archived)
    with pytest.raises(AssertionError):
        study.verify_completed_receipt(seed, folder, receipt, inputs,
                                       saved, direction)
    archived.write_bytes(source.read_bytes())
    receipt['source_sha256'] = study.digest(source)
    receipt['oracle_baseline_receipt_sha256'] = '0' * 64
    with pytest.raises(AssertionError):
        study.verify_completed_receipt(seed, folder, receipt, inputs,
                                       saved, direction)
