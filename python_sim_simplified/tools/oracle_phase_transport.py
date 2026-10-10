"""Truth-dependent, unit-phase mixed-beam replay on frozen Stage 47 weak IQ.

This is an offline mechanism isolation, not a receiver correction proposal.
The oracle touches the complete beam stream before the original receiver.
"""
import json
from pathlib import Path
from types import FunctionType
from zipfile import ZipFile, ZipInfo, ZIP_STORED

import numpy as np

from core.recovery_geometry import FS
from core.single_source import LOOP_HISTORY_FIELDS, frontend, navigate, receive_measured
from experiments.single_source_complex_response_audit import generator_truth
from experiments.single_source_coherence_diagnosis import analyze_groups
from experiments.single_source_coherent_navigation import (
    build_coherent_support, quality_support_counts,
)
from experiments.single_source_evidence_history import analyze_trace
from experiments.single_source_nav_bit_epoch_shift import anchor_transport
from experiments.single_source_phase_exclusion import verify_source_snapshot
from experiments.single_source_phase_state_gate import projection_eligibility
from experiments.single_source_projection_threshold import longest_gap
from experiments.single_source_ramp_transition import save_loops
from experiments.single_source_startup_amplitude_rf import (
    check_prefix, score_navigation,
)
from experiments.single_source_suppression_study import ANTENNAS, RECEIVER_PLL, digest
from experiments.single_source_temporal_trigger import navigation_input
from experiments import single_source_projection_long as stage47
from tools.single_path_carrier_continuity import event_times
from tools.resume_projection_long import (
    check_saved_arrays, check_saved_navigation, read_npz,
)


STAGE47_SUMMARY_SHA256 = '47a73e338bf086f8c0418f31c3db07081dea299129775be54b45d22d206e5a86'
CONTINUITY_SUMMARY_SHA256 = 'f2c6004241a8891145b5d9cb4f88d30b29377b320a57b35dc710a62392f85166'
SEEDS = (2026100402, 2026100403, 2026100404)
WIDTH = round(.02 * FS)
ZERO_RESPONSE = 1e-12
OUTPUT = stage47.OUTPUT/'oracle_phase_transport'
EXTRA_SOURCES = ('tools/oracle_phase_transport.py',
                 'tools/resume_projection_long.py',
                 'tools/single_path_carrier_continuity.py')


def oracle_factor(weight, direction, rate, sample_index, first_response):
    """Unit factor for one PRN and one original 20 ms weight block.

    ``direction`` is physical_gain / receiver_calibration times nominal steering.
    Indices are original IQ samples, so rate uses the unchanged receiver clock.
    """
    sample_index = np.asarray(sample_index)
    response = np.conj(weight) @ (
        direction[:, None] * np.exp(-1j * rate[:, None] *
                                     sample_index[None, :] / FS))
    if abs(first_response) <= ZERO_RESPONSE or np.any(abs(response) <= ZERO_RESPONSE):
        raise RuntimeError('Oracle true response is numerically zero')
    return (first_response / abs(first_response)) * (response.conj() / abs(response))


def rotate_mixed_stream(signal, star, weights, direction, rate,
                        raw_samples, calibration):
    """Verify each pre-rotation mixed sample, then rotate the entire stream."""
    first = np.vdot(weights[0, :, star], direction[:, star])
    if abs(first) <= ZERO_RESPONSE:
        raise RuntimeError(f'PRN {star + 1}: zero true response at sample 0')
    for block, start in enumerate(range(0, len(signal), WIDTH)):
        end = min(start + WIDTH, len(signal))
        index = np.arange(start, end)
        calibrated = raw_samples[ANTENNAS, start:end] / (
            calibration[:, None] *
            np.exp(1j * rate[:, None] * index[None, :] / FS))
        unrotated = weights[block, :, star].conj() @ calibrated
        np.testing.assert_allclose(signal[start:end], unrotated,
                                   rtol=0, atol=1e-8,
                                   err_msg=f'pre-oracle PRN {star + 1} block {block}')
        factor = oracle_factor(weights[block, :, star], direction[:, star],
                               rate, index, first)
        signal[start:end] *= factor


def private_frontend_with_oracle(record, weights, direction, calibration,
                                 rate, scratch, loops):
    """Replace only the private receive call; frontend globals stay untouched."""
    def receive_oracle(signal, prn, acquisition, tracking, **kwargs):
        rotate_mixed_stream(signal, prn - 1, weights, direction, rate,
                            record['samples'], calibration)
        return receive_measured(signal, prn, acquisition, tracking, **kwargs)

    private_globals = dict(frontend.__globals__)
    private_globals['receive_measured'] = receive_oracle
    private_frontend = FunctionType(frontend.__code__, private_globals,
                                    frontend.__name__, frontend.__defaults__,
                                    frontend.__closure__)
    return run_frontend(private_frontend, record, scratch, loops)


def run_frontend(receiver, record, scratch, loops):
    return receiver(record, ANTENNAS, split_support=True,
                    receiver_pll=RECEIVER_PLL, evidence_trace=True,
                    temporal_beam_trigger_enabled=False,
                    beam_template_carrier_aiding=True, beam_coherent_ms=5,
                    beam_startup_amplitude=True, fast_truncation=True,
                    beam_projection_threshold=3., stream_memmap_path=scratch,
                    loop_history=loops)


def split_observations(observations):
    tracking = {key: value for key, value in observations.items()
                if not key.startswith('trace_')}
    trace = {key: value for key, value in observations.items()
             if key.startswith('trace_')}
    return tracking, trace


def assert_saved_arrays(folder, filename, actual):
    saved = read_npz(folder/filename)
    assert saved.keys() == actual.keys(), filename
    for key, value in actual.items():
        np.testing.assert_array_equal(value, saved[key],
                                      err_msg=f'{filename}:{key}')


def verify_fit_stream(record, scratch, fit, tracking):
    """Bind archived oracle weights to every actual baseline mixed-IQ sample."""
    samples = record['samples']
    count = samples.shape[1]
    stream = np.memmap(scratch, dtype=np.complex128, mode='r',
                       shape=(len(record['codes']), count), order='C')
    gain = tracking['calibration']
    rate = tracking['calibration_rate']
    for block, start in enumerate(range(0, count, WIDTH)):
        end = min(start + WIDTH, count)
        index = np.arange(start, end)
        calibrated = samples[ANTENNAS, start:end] / (
            gain[:, None] * np.exp(1j * rate[:, None] * index[None, :] / FS))
        expected = fit['weight'][block].conj().T @ calibrated
        np.testing.assert_allclose(stream[:, start:end], expected,
                                   rtol=0, atol=1e-8, err_msg=f'beam block {block}')
    del stream


def baseline_replay(record, saved, fit, scratch):
    """Reconstruct the original path and halt before scoring on any difference."""
    loops = []
    observations = run_frontend(frontend, record, scratch, loops)
    tracking, trace = split_observations(observations)
    assert_saved_arrays(saved, 'tracking.npz', tracking)
    assert_saved_arrays(saved, 'trace.npz', trace)
    actual_loops = {
        f'prn{segment["prn"]}_arc{segment["arc"]}_{field}': segment[field]
        for segment in loops
        for field in LOOP_HISTORY_FIELDS
    }
    assert_saved_arrays(saved, 'loops.npz', actual_loops)
    verify_fit_stream(record, scratch, fit, tracking)
    coherent = check_saved_arrays(saved, tracking, trace)
    check_saved_navigation(saved, tracking, trace, coherent,
                           json.loads((saved.parent/'input'/'metadata.json').read_text(
                               encoding='utf-8'))['case'])
    return tracking


def preflight():
    """Bind the frozen 98-source experiment and all three saved inputs."""
    summary_path = stage47.OUTPUT/'summary.json'
    assert digest(summary_path) == STAGE47_SUMMARY_SHA256
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    assert digest(stage47.OUTPUT/'source_snapshot.zip') == summary['source_snapshot_sha256']
    verify_source_snapshot(stage47.OUTPUT/'source_snapshot.zip',
                           summary['source_sha256'])
    root = Path(__file__).resolve().parents[1]
    verify_current_sources(root, summary['source_sha256'])
    assert digest(root/'tools/resume_projection_long.py') == (
        summary['recovery_source_sha256'])
    continuity_path = stage47.OUTPUT/'single_path_carrier_continuity_summary.json'
    assert digest(continuity_path) == CONTINUITY_SUMMARY_SHA256
    continuity = json.loads(continuity_path.read_text(encoding='utf-8'))
    assert digest(root/'tools/single_path_carrier_continuity.py') == (
        continuity['source_sha256'])
    return summary


def verify_current_sources(root, expected):
    """Check every frozen Stage 47 source, not just direct imports here."""
    assert len(expected) == 98
    for name, sha in expected.items():
        assert digest(root/name) == sha, name


def snapshot_extra_sources(folder):
    """Save this implementation and the two post-Stage-47 helper source bytes."""
    archive = folder/'source_snapshot.zip'
    root = Path(__file__).resolve().parents[1]
    if archive.exists():
        with ZipFile(archive) as frozen:
            assert set(frozen.namelist()) == set(EXTRA_SOURCES)
            for name in EXTRA_SOURCES:
                assert frozen.read(name) == (root/name).read_bytes(), name
    else:
        with ZipFile(archive, 'x', compression=ZIP_STORED) as frozen:
            for name in EXTRA_SOURCES:
                info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                frozen.writestr(info, (root/name).read_bytes())
    return digest(archive)


def read_input(seed, summary):
    saved = stage47.OUTPUT/str(seed)/'weak'/'gate3'
    source = saved.parent/'input'
    original_hashes = summary['cases'][str(seed)]['weak']['input_sha256']
    for name, expected in original_hashes.items():
        assert digest(source/name) == expected, name
    archived_science = summary['cases'][str(seed)]['weak']['paths']['gate3'][
        'output_sha256']
    for name, expected in archived_science.items():
        assert digest(saved/name) == expected, name
    metadata = json.loads((source/'metadata.json').read_text(encoding='utf-8'))
    with np.load(source/'observations.npz') as archive:
        record = dict(archive)
    record['samples'] = np.load(source/'samples.npy', mmap_mode='r')
    fit = read_npz(saved/'fit.npz')
    tracking = read_npz(saved/'tracking.npz')
    assert fit['weight'].shape == (1000, len(ANTENNAS), len(record['codes']))
    assert np.array_equal(tracking['calibration_rate'], np.zeros(len(ANTENNAS)))
    physical, _ = generator_truth(metadata, record['nominal'])
    direction = (physical[ANTENNAS] / tracking['calibration'])[:, None] * (
        record['nominal'][ANTENNAS])
    return saved, record, metadata, fit, tracking, direction, original_hashes


def score_pair(time, original_position, oracle_position, velocity, original_motion,
               oracle_motion):
    """Fixed observation-time screen, separate from Stage 47 long_screen."""
    opportunity = time >= .3
    truth = np.column_stack((velocity * time, np.zeros(len(time))))
    old_published = np.isfinite(original_position).all(axis=1)
    new_published = np.isfinite(oracle_position).all(axis=1)
    common = opportunity & old_published & new_published

    def metrics(position, published):
        error = np.linalg.norm(position[:, :2] - truth, axis=1)
        values = error[opportunity & published]
        shared = error[common]
        return dict(published=int((opportunity & published).sum()),
                    longest_gap_epochs=longest_gap(published[opportunity]),
                    ge20=int((values >= 20.).sum()),
                    max_error_m=float(values.max()) if len(values) else None,
                    p95_error_m=float(np.percentile(values, 95)) if len(values) else None,
                    common_max_m=float(shared.max()) if len(shared) else None,
                    common_p95_m=float(np.percentile(shared, 95)) if len(shared) else None)

    old = metrics(original_position, old_published)
    new = metrics(oracle_position, new_published)
    passed = (original_motion is not None and oracle_motion is not None and
              oracle_motion <= .5 * original_motion and
              new['published'] >= old['published'] and
              new['longest_gap_epochs'] <= old['longest_gap_epochs'] and
              new['ge20'] == 0 and
              int((opportunity & old_published & ~new_published).sum()) == 0)
    return dict(opportunities=int(opportunity.sum()), common=int(common.sum()),
                baseline=old, oracle=new, baseline_motion_m=original_motion,
                oracle_motion_m=oracle_motion, pass_rule=bool(passed))


def verify_oracle_frontend(tracking, trace, baseline_tracking, baseline_trace):
    """Outputs fixed before receive_measured must match the frozen baseline."""
    for key in ('time', 'available_time', 'satellites', 'startup_prns',
                'calibration', 'calibration_rate', 'startup_relative_amplitude',
                'gain', 'ratio'):
        np.testing.assert_array_equal(tracking[key], baseline_tracking[key],
                                      err_msg=f'oracle frontend {key}')
    for key in ('trace_raw_trigger', 'trace_projectors',
                'trace_dimensions', 'trace_nominal'):
        np.testing.assert_array_equal(trace[key], baseline_trace[key],
                                      err_msg=f'oracle frontend {key}')


def replay_short_navigation(tracking, trace, coherent, case, folder):
    """Use exactly the frozen short navigation parameters; omit original mode."""
    eligible, latest, recent = projection_eligibility(
        tracking['available_time'], trace)
    support = coherent['support']
    navigation = navigate(navigation_input(tracking, support), diagnostics=True,
                          code_blend=0., phase_leave_one_out=True,
                          phase_state_continuity=True,
                          phase_state_eligible=eligible)
    check_prefix(tracking, support, eligible, navigation, 'short')
    support_count = quality_support_counts(tracking, support)
    arrays = dict(observation_time=tracking['time'],
                  available_time=tracking['available_time'],
                  phase_state_eligible=eligible,
                  latest_complete_block=latest,
                  recent_block_dimensions=recent,
                  short_support_count=support_count)
    for key, value in navigation.items():
        if key == 'diagnostics':
            for diagnostic, data in value.items():
                arrays[f'short_diagnostic_{diagnostic}'] = data
        else:
            arrays[f'short_{key}'] = np.nan if value is None else value
    np.savez_compressed(folder/'navigation.npz', **arrays)
    score = dict(**score_navigation(tracking, navigation,
                                    case['velocity_east']),
                 quality_support_ge4=int((support_count >= 4).sum()),
                 quality_support_ge6=int((support_count >= 6).sum()))
    return dict(short=score), dict(short=navigation['position'].copy())


def oracle_replay(record, metadata, fit, tracking_baseline, baseline_trace,
                  direction, folder, scratch):
    loops = []
    observations = private_frontend_with_oracle(
        record, fit['weight'], direction, tracking_baseline['calibration'],
        tracking_baseline['calibration_rate'],
        scratch, loops)
    tracking, trace = split_observations(observations)
    verify_oracle_frontend(tracking, trace, tracking_baseline, baseline_trace)
    np.savez_compressed(folder/'tracking.npz', **tracking)
    np.savez_compressed(folder/'trace.npz', **trace)
    save_loops(folder, loops)
    history = analyze_trace(observations)
    groups = analyze_groups(trace)
    coherent = build_coherent_support(tracking, groups)
    np.savez_compressed(folder/'history.npz', **history)
    np.savez_compressed(folder/'groups.npz', **groups)
    np.savez_compressed(folder/'support.npz', **{
        f'coherent_{key}': value for key, value in coherent.items()})
    scores, positions = replay_short_navigation(
        tracking, trace, coherent, metadata['case'], folder)
    return tracking, scores, positions


def run_seed(seed, summary):
    saved, record, metadata, fit, stored_tracking, direction, input_hashes = (
        read_input(seed, summary))
    folder = OUTPUT/str(seed)
    receipt_path = folder/'receipt.json'
    if receipt_path.exists():
        assert (folder/'source_snapshot.zip').exists()
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        assert receipt['stage47_summary_sha256'] == STAGE47_SUMMARY_SHA256
        assert receipt['input_sha256'] == input_hashes
        assert receipt['fit_sha256'] == digest(saved/'fit.npz')
        assert receipt['source_snapshot_sha256'] == snapshot_extra_sources(folder)
        for name, sha in receipt['output_sha256'].items():
            assert digest(folder/'oracle'/name) == sha
        return receipt['score']
    folder.mkdir(parents=True, exist_ok=True)
    source_sha = snapshot_extra_sources(folder)
    scratch = folder/'beam_stream.dat'
    baseline_path = folder/'baseline_verified.json'
    if baseline_path.exists():
        assert not scratch.exists()
        baseline_receipt = json.loads(baseline_path.read_text(encoding='utf-8'))
        assert baseline_receipt['source_snapshot_sha256'] == source_sha
        assert baseline_receipt['input_sha256'] == input_hashes
        for name, sha in baseline_receipt['saved_output_sha256'].items():
            assert digest(saved/name) == sha
        baseline = stored_tracking
    else:
        if scratch.exists() or (folder/'oracle').exists():
            raise RuntimeError(f'Partial path retained for inspection: {folder}')
        baseline = baseline_replay(record, saved, fit, scratch)
        assert scratch.stat().st_size == len(record['codes']) * record['samples'].shape[1] * 16
        scratch.unlink()
        baseline_path.write_text(json.dumps(dict(
            saved_path=str(saved.relative_to(stage47.OUTPUT)),
            saved_output_sha256={name: digest(saved/name) for name in
                ('tracking.npz', 'trace.npz', 'loops.npz', 'support.npz', 'navigation.npz')},
            source_snapshot_sha256=source_sha, input_sha256=input_hashes),
            indent=2), encoding='utf-8')
    candidate_folder = folder/'oracle'
    if candidate_folder.exists():
        raise RuntimeError(f'Partial oracle path retained for inspection: {candidate_folder}')
    candidate_folder.mkdir()
    tracking, scores, positions = oracle_replay(
        record, metadata, fit, baseline, read_npz(saved/'trace.npz'),
        direction, candidate_folder, scratch)
    assert scratch.stat().st_size == len(record['codes']) * record['samples'].shape[1] * 16
    scratch.unlink()
    old_navigation = read_npz(saved/'navigation.npz')
    old_position = old_navigation['short_position']
    old_motion = anchor_transport(dict(position=old_position),
                                  baseline['time'], metadata['case']['velocity_east'])
    comparison = score_pair(baseline['time'], old_position, positions['short'],
                            metadata['case']['velocity_east'],
                            old_motion['max_relative_motion_error_m'],
                            scores['short']['max_relative_motion_error_m'])
    diagnostic = read_npz(candidate_folder/'navigation.npz')
    baseline_navigation = dict(
        diagnostics={key[len('short_diagnostic_'):]: value
                     for key, value in old_navigation.items()
                     if key.startswith('short_diagnostic_')},
        initialization_time=float(old_navigation['short_initialization_time']))
    oracle_navigation = dict(
        diagnostics={key[len('short_diagnostic_'):]: value
                     for key, value in diagnostic.items()
                     if key.startswith('short_diagnostic_')},
        initialization_time=float(diagnostic['short_initialization_time']))
    comparison['baseline_anchor'] = old_motion
    comparison['oracle_anchor'] = anchor_transport(
        dict(position=positions['short']), tracking['time'],
        metadata['case']['velocity_east'])
    comparison['baseline_arc_segments'] = len([
        key for key in read_npz(saved/'loops.npz')
        if key.endswith('_epoch_start_samples')])
    comparison['oracle_arc_segments'] = len([
        key for key in read_npz(candidate_folder/'loops.npz')
        if key.endswith('_epoch_start_samples')])
    for label, data, current in (('baseline', baseline_navigation, baseline),
                                 ('oracle', oracle_navigation, tracking)):
        phase_ok = data['diagnostics']['phase_ok']
        published = np.isfinite(old_position if label == 'baseline'
                                else positions['short']).all(axis=1)
        comparison[f'{label}_phase_bridge_epochs'] = int((phase_ok & published).sum())
        comparison[f'{label}_code_only_published_epochs'] = int((
            ~phase_ok & published).sum())
        comparison[f'{label}_code_reanchors_after_phase'] = int((
            ~phase_ok & published & np.r_[False, phase_ok[:-1]]).sum())
        comparison[f'{label}_events'] = event_times(
            current['time'], current['available_time'], data)
    science = ('tracking.npz', 'trace.npz', 'loops.npz', 'history.npz',
               'groups.npz', 'support.npz', 'navigation.npz')
    receipt = dict(seed=seed, stage47_summary_sha256=STAGE47_SUMMARY_SHA256,
                   stage47_source_snapshot_sha256=summary['source_snapshot_sha256'],
                   source_snapshot_sha256=source_sha, input_sha256=input_hashes,
                   fit_sha256=digest(saved/'fit.npz'),
                   output_sha256={name: digest(candidate_folder/name) for name in science},
                   score=comparison)
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False),
                            encoding='utf-8')
    return comparison


def main():
    summary = preflight()
    results = {}
    for seed in SEEDS:
        results[str(seed)] = run_seed(seed, summary)
        print(seed, results[str(seed)]['pass_rule'], flush=True)
    result = dict(
        stage47_summary_sha256=STAGE47_SUMMARY_SHA256,
        all_pass=all(value['pass_rule'] for value in results.values()),
        results=results)
    summary_path = OUTPUT/'summary.json'
    if summary_path.exists():
        assert json.loads(summary_path.read_text(encoding='utf-8')) == result
    else:
        summary_path.write_text(json.dumps(result, indent=2, allow_nan=False),
                                encoding='utf-8')


if __name__ == '__main__':
    main()
