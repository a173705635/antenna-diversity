"""Replay paired H0 IQ through the frozen weak beam and original receiver.

The weak fit still contains attack history. This offline counterfactual removes
the fake RF component by using saved paired H0 IQ, not by editing carrier or
navigation observations.
"""
import json
import shutil
from pathlib import Path
from types import FunctionType
from zipfile import ZipFile

import numpy as np

from core.recovery_geometry import FS
from core.single_source import calibrate, frontend, receive_measured
from experiments.single_source_complex_response_audit import generator_truth
from experiments.single_source_coherence_diagnosis import analyze_groups
from experiments.single_source_coherent_navigation import build_coherent_support
from experiments.single_source_evidence_history import analyze_trace
from experiments.single_source_nav_bit_epoch_shift import anchor_transport
from experiments.single_source_ramp_transition import save_loops
from experiments.single_source_suppression_study import ANTENNAS, digest
from experiments import single_source_projection_long as stage47
from tools import oracle_phase_transport as oracle
from tools.resume_projection_long import read_npz
from tools.single_path_carrier_continuity import event_times


SEEDS = (2026100402, 2026100403, 2026100404)
WIDTH = round(.02 * FS)
IQ_RESIDUAL_MAX = 1e-11
FRONTEND_ATOL = 1e-12
STREAM_ATOL = 1e-8
OUTPUT = stage47.OUTPUT/'frozen_weak_h0_counterfactual'
ORACLE_SUMMARY_SHA256 = 'd7b467e77f0b7d1c8999061811fb289b6facfeb3be19fc150ad3e181e05de166'
ORACLE_RECEIPT_SHA256 = {
    2026100402: '74411d6953cd0e7719ebefe577becee414fea1f805d7d2d9304754b1e001677c',
    2026100403: 'a71cdb9c1d421ec96940fd793515ffa06139e57eeffb9072c997b3cdff842edb',
    2026100404: '40e4800c41a3ced3fa563b74f8ce913ac28583bb0a3b23a5e1d34f142c709e38',
}
INPUT_FILES = ('samples.npy', 'observations.npz', 'metadata.json')
SCIENCE = ('tracking.npz', 'trace.npz', 'loops.npz', 'history.npz',
           'groups.npz', 'support.npz', 'navigation.npz')


def preflight():
    frozen = oracle.preflight()
    assert digest(stage47.OUTPUT/'oracle_phase_transport'/'summary.json') == (
        ORACLE_SUMMARY_SHA256)
    for seed in SEEDS:
        receipt = stage47.OUTPUT/'oracle_phase_transport'/str(seed)/'receipt.json'
        assert digest(receipt) == ORACLE_RECEIPT_SHA256[seed]
        prior = json.loads(receipt.read_text(encoding='utf-8'))
        baseline = json.loads((receipt.parent/'baseline_verified.json').read_text(
            encoding='utf-8'))
        assert prior['stage47_summary_sha256'] == oracle.STAGE47_SUMMARY_SHA256
        assert prior['source_snapshot_sha256'] == baseline[
            'source_snapshot_sha256']
        verify_oracle_helper_snapshot(
            receipt.parent/'source_snapshot.zip',
            prior['source_snapshot_sha256'], Path(__file__).resolve().parents[1])
        assert baseline['input_sha256'] == frozen['cases'][str(seed)]['weak'][
            'input_sha256']
        assert Path(baseline['saved_path']).parts == (str(seed), 'weak', 'gate3')
        for name, sha in baseline['saved_output_sha256'].items():
            assert digest(stage47.OUTPUT/str(seed)/'weak'/'gate3'/name) == sha
    return frozen


def verify_oracle_helper_snapshot(snapshot, expected_sha, source_root):
    """Bind every currently imported post-Stage-47 helper to old ZIP bytes."""
    assert digest(snapshot) == expected_sha
    with ZipFile(snapshot) as archive:
        assert set(archive.namelist()) == set(oracle.EXTRA_SOURCES)
        for name in oracle.EXTRA_SOURCES:
            assert archive.read(name) == (source_root/name).read_bytes(), name


def load_pair(seed, frozen):
    """Read paired saved inputs and bind both IQ files to Stage 47 hashes."""
    parent = stage47.OUTPUT/str(seed)
    paths = {scene: parent/scene/'input' for scene in ('weak', 'h0_drop')}
    records = {}
    input_hashes = {}
    metadata = {}
    for scene, source in paths.items():
        expected = frozen['cases'][str(seed)][scene]['input_sha256']
        actual = {name: digest(source/name) for name in INPUT_FILES}
        assert actual == expected, scene
        input_hashes[scene] = actual
        metadata[scene] = json.loads((source/'metadata.json').read_text(
            encoding='utf-8'))
        records[scene] = read_npz(source/'observations.npz')
        records[scene]['samples'] = np.load(source/'samples.npy', mmap_mode='r')
    weak, h0 = records['weak'], records['h0_drop']
    verify_paired_records(weak, h0, round(.3 * FS))
    saved = parent/'weak'/'gate3'
    expected = frozen['cases'][str(seed)]['weak']['paths']['gate3'][
        'output_sha256']
    assert {name: digest(saved/name) for name in expected} == expected
    fit = read_npz(saved/'fit.npz')
    tracking = read_npz(saved/'tracking.npz')
    trace = read_npz(saved/'trace.npz')
    assert fit['weight'].shape == (1000, len(ANTENNAS), len(weak['codes']))
    assert np.array_equal(tracking['calibration_rate'], np.zeros(len(ANTENNAS)))
    return records, metadata, input_hashes, saved, fit, tracking, trace


def verify_paired_records(weak, h0, onset):
    """Assert identical geometry and raw IQ through the attack boundary."""
    for name in ('nominal', 'codes', 'times', 'satellites', 'coarse'):
        np.testing.assert_array_equal(weak[name], h0[name], err_msg=name)
    np.testing.assert_array_equal(weak['samples'][:, :onset],
                                  h0['samples'][:, :onset],
                                  err_msg='paired pre-attack IQ')


def paired_direction_residual(weak, h0, physical, fake_steering,
                              chunk_samples=1_000_000):
    """Check all samples against the common fake spatial direction."""
    direction = physical * fake_steering
    maximum = squared = difference_squared = 0.
    count = 0
    for start in range(0, weak.shape[1], chunk_samples):
        end = min(start + chunk_samples, weak.shape[1])
        difference = weak[:, start:end] - h0[:, start:end]
        scalar = difference[0] / direction[0]
        residual = difference - direction[:, None] * scalar[None, :]
        maximum = max(maximum, float(np.max(abs(residual))))
        squared += float(np.sum(abs(residual)**2))
        difference_squared += float(np.sum(abs(difference)**2))
        count += residual.size
    metrics = dict(max_abs_iq=maximum,
                   rms_iq=float(np.sqrt(squared/count)),
                   relative_norm=float(np.sqrt(squared/difference_squared)))
    assert maximum <= IQ_RESIDUAL_MAX, metrics
    return metrics


def verify_h0_stream(signal, star, record, fit, calibration, rate):
    """Independently rebuild the entire pre-receiver stream from H0 raw IQ."""
    samples = record['samples']
    for block, start in enumerate(range(0, len(signal), WIDTH)):
        end = min(start + WIDTH, len(signal))
        index = np.arange(start, end)
        calibrated = samples[ANTENNAS, start:end] / (
            calibration[:, None] *
            np.exp(1j * rate[:, None] * index[None, :] / FS))
        expected = fit['weight'][block, :, star].conj() @ calibrated
        np.testing.assert_allclose(signal[start:end], expected, rtol=0,
                                   atol=STREAM_ATOL,
                                   err_msg=f'H0 beam PRN {star + 1} block {block}')


def frozen_frontend(record, fit, weak_tracking, scratch, loops):
    """Use private globals: fixed beam history, original startup and receiver."""
    next_block = [5]
    receiver_calls = [0]

    def checked_calibration(block, local, desired):
        gain, rate = calibrate(block, local, desired)
        np.testing.assert_array_equal(gain, weak_tracking['calibration'])
        np.testing.assert_array_equal(rate, weak_tracking['calibration_rate'])
        return weak_tracking['calibration'].copy(), weak_tracking[
            'calibration_rate'].copy()

    def frozen_beam(block, model, local, startup_cov, prns,
                    trigger_history=None, coherent_ms=1,
                    beam_projection_threshold=3.):
        index = next_block[0]
        assert trigger_history is None and coherent_ms == 5
        assert beam_projection_threshold == 3. and 5 <= index < len(fit['raw'])
        next_block[0] += 1
        return (fit['weight'][index], fit['gain'][index],
                float(fit['raw'][index]))

    def checked_receive(signal, prn, acquisition, tracking, **options):
        receiver_calls[0] += 1
        verify_h0_stream(signal, prn - 1, record, fit,
                         weak_tracking['calibration'],
                         weak_tracking['calibration_rate'])
        return receive_measured(signal, prn, acquisition, tracking, **options)

    private_globals = dict(frontend.__globals__, calibrate=checked_calibration,
                           beam=frozen_beam, receive_measured=checked_receive)
    private = FunctionType(frontend.__code__, private_globals,
                           frontend.__name__, frontend.__defaults__,
                           frontend.__closure__)
    observations = oracle.run_frontend(private, record, scratch, loops)
    assert next_block[0] == len(fit['raw'])
    assert receiver_calls[0] == len(record['codes'])
    assert frontend.__globals__['calibrate'] is calibrate
    assert frontend.__globals__['receive_measured'] is receive_measured
    return observations


def fixed_frontend_comparison(tracking, trace, frozen_tracking, frozen_trace):
    """Confirm only the IQ-dependent tracking/evidence changed."""
    for key in ('time', 'available_time', 'satellites', 'startup_prns',
                'calibration', 'calibration_rate',
                'startup_relative_amplitude', 'trace_nominal'):
        actual = trace[key] if key.startswith('trace_') else tracking[key]
        expected = (frozen_trace if key.startswith('trace_') else
                    frozen_tracking)[key]
        np.testing.assert_array_equal(actual, expected, err_msg=key)
    for key in ('gain', 'ratio'):
        np.testing.assert_allclose(tracking[key], frozen_tracking[key],
                                   rtol=0, atol=FRONTEND_ATOL, err_msg=key)
    np.testing.assert_allclose(trace['trace_raw_trigger'],
                               frozen_trace['trace_raw_trigger'],
                               rtol=0, atol=FRONTEND_ATOL, equal_nan=True)
    np.testing.assert_allclose(trace['trace_projectors'],
                               frozen_trace['trace_projectors'],
                               rtol=0, atol=FRONTEND_ATOL)
    np.testing.assert_array_equal(trace['trace_dimensions'],
                                  frozen_trace['trace_dimensions'])
    raw = trace['trace_raw_trigger']
    reference = frozen_trace['trace_raw_trigger']
    np.testing.assert_array_equal(raw >= 3., reference >= 3.)
    return dict(max_gain_difference=float(np.max(abs(tracking['gain']-
                                                  frozen_tracking['gain']))),
                max_raw_difference=float(np.nanmax(abs(raw-reference))),
                max_projector_difference=float(np.max(abs(
                    trace['trace_projectors']-frozen_trace['trace_projectors']))))


def short_events(folder, tracking, position):
    saved = read_npz(folder/'navigation.npz')
    nav = dict(initialization_time=float(saved['short_initialization_time']),
               diagnostics={key[len('short_diagnostic_'):]: value
                            for key, value in saved.items()
                            if key.startswith('short_diagnostic_')})
    phase_ok = nav['diagnostics']['phase_ok']
    published = np.isfinite(position).all(axis=1)
    return dict(events=event_times(tracking['time'], tracking['available_time'], nav),
                phase_bridge_epochs=int((phase_ok & published).sum()),
                code_only_published_epochs=int((~phase_ok & published).sum()),
                code_reanchors_after_phase=int((~phase_ok & published &
                    np.r_[False, phase_ok[:-1]]).sum()))


def verify_completed_receipt(seed, folder, receipt, input_hashes, saved,
                             direction_check):
    """A completed path is reusable only with this exact current source."""
    current_source = digest(Path(__file__))
    archived_source = digest(folder/'frozen_weak_h0_counterfactual.py')
    assert current_source == archived_source == receipt['source_sha256']
    assert receipt['seed'] == seed
    assert receipt['stage47_summary_sha256'] == oracle.STAGE47_SUMMARY_SHA256
    assert receipt['oracle_summary_sha256'] == ORACLE_SUMMARY_SHA256
    assert receipt['oracle_baseline_receipt_sha256'] == ORACLE_RECEIPT_SHA256[seed]
    assert receipt['weak_fit_sha256'] == digest(saved/'fit.npz')
    assert receipt['input_sha256'] == input_hashes
    assert receipt['direction_residual'] == direction_check
    assert set(receipt['output_sha256']) == set(SCIENCE)
    for name, sha in receipt['output_sha256'].items():
        assert digest(folder/name) == sha


def run_seed(seed, frozen):
    records, metadata, input_hashes, saved, fit, old_tracking, old_trace = (
        load_pair(seed, frozen))
    weak, h0 = records['weak'], records['h0_drop']
    physical, fake_steering = generator_truth(metadata['weak'], weak['nominal'])
    direction_check = paired_direction_residual(
        weak['samples'], h0['samples'], physical, fake_steering)
    folder = OUTPUT/str(seed)
    receipt_path = folder/'receipt.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        verify_completed_receipt(seed, folder, receipt, input_hashes,
                                 saved, direction_check)
        return receipt['score']
    if folder.exists():
        raise RuntimeError(f'Partial path retained for inspection: {folder}')
    folder.mkdir(parents=True)
    source_copy = folder/'frozen_weak_h0_counterfactual.py'
    shutil.copyfile(__file__, source_copy)
    scratch = folder/'beam_stream.dat'
    loops = []
    observations = frozen_frontend(h0, fit, old_tracking, scratch, loops)
    tracking, trace = oracle.split_observations(observations)
    frozen_check = fixed_frontend_comparison(
        tracking, trace, old_tracking, old_trace)
    np.savez_compressed(folder/'tracking.npz', **tracking)
    np.savez_compressed(folder/'trace.npz', **trace)
    save_loops(folder, loops)
    assert scratch.stat().st_size == len(h0['codes']) * h0['samples'].shape[1] * 16
    scratch.unlink()
    history = analyze_trace(observations)
    groups = analyze_groups(trace)
    coherent = build_coherent_support(tracking, groups)
    np.savez_compressed(folder/'history.npz', **history)
    np.savez_compressed(folder/'groups.npz', **groups)
    np.savez_compressed(folder/'support.npz', **{
        f'coherent_{key}': value for key, value in coherent.items()})
    scores, positions = oracle.replay_short_navigation(
        tracking, trace, coherent, metadata['weak']['case'], folder)
    previous = read_npz(saved/'navigation.npz')
    baseline_position = previous['short_position']
    baseline_anchor = anchor_transport(
        dict(position=baseline_position), old_tracking['time'],
        metadata['weak']['case']['velocity_east'])
    new_anchor = anchor_transport(
        dict(position=positions['short']), tracking['time'],
        metadata['weak']['case']['velocity_east'])
    comparison = oracle.score_pair(
        tracking['time'], baseline_position, positions['short'],
        metadata['weak']['case']['velocity_east'],
        baseline_anchor['max_relative_motion_error_m'],
        scores['short']['max_relative_motion_error_m'])
    comparison.update(baseline_anchor=baseline_anchor, candidate_anchor=new_anchor,
                      baseline_arc_segments=len([key for key in
                          read_npz(saved/'loops.npz') if
                          key.endswith('_epoch_start_samples')]),
                      candidate_arc_segments=len(loops),
                      baseline_events=short_events(saved, old_tracking,
                                                   baseline_position),
                      candidate_events=short_events(folder, tracking,
                                                    positions['short']))
    receipt = dict(seed=seed, stage47_summary_sha256=oracle.STAGE47_SUMMARY_SHA256,
                   oracle_summary_sha256=ORACLE_SUMMARY_SHA256,
                   oracle_baseline_receipt_sha256=ORACLE_RECEIPT_SHA256[seed],
                   source_sha256=digest(source_copy), input_sha256=input_hashes,
                   weak_fit_sha256=digest(saved/'fit.npz'),
                   direction_residual=direction_check,
                   frozen_frontend=frozen_check,
                   output_sha256={name: digest(folder/name) for name in SCIENCE},
                   score=comparison)
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False),
                            encoding='utf-8')
    return comparison


def main():
    frozen = preflight()
    results = {}
    for seed in SEEDS:
        results[str(seed)] = run_seed(seed, frozen)
        print(seed, results[str(seed)]['pass_rule'], flush=True)
    summary = dict(stage47_summary_sha256=oracle.STAGE47_SUMMARY_SHA256,
                   oracle_summary_sha256=ORACLE_SUMMARY_SHA256,
                   all_pass=all(result['pass_rule'] for result in results.values()),
                   results=results)
    summary_path = OUTPUT/'summary.json'
    if summary_path.exists():
        assert json.loads(summary_path.read_text(encoding='utf-8')) == summary
    else:
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False),
                                encoding='utf-8')


if __name__ == '__main__':
    main()
