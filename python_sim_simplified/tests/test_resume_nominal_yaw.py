"""Mode isolation after a saved Stage 48 original-navigation solver failure."""
import numpy as np
import pytest
import json

from core.positioning import PositioningFailure
from tools.spoofing import resume_nominal_yaw as recovery


def test_original_solver_failure_does_not_skip_independent_short_replay(monkeypatch):
    tracking = {'time': np.array([.30, .31]),
                'available_time': np.array([.32, .33]),
                'support': np.zeros((2, 2))}
    trace = {}
    coherent = {'support': np.ones((2, 2))}
    eligible = np.array([True, True])
    monkeypatch.setattr(recovery, 'projection_eligibility',
                        lambda *args: (eligible, np.zeros(2, int), np.zeros((2, 5), int)))
    monkeypatch.setattr(recovery, 'navigation_input',
                        lambda tracking, support: {'support': support})
    calls = []

    def navigate(inputs, **options):
        mode = 'short' if options['phase_leave_one_out'] else 'original'
        calls.append(mode)
        if mode == 'original':
            raise PositioningFailure('insufficient satellite geometry')
        return {'position': np.zeros((2, 3)), 'selected': np.ones((2, 2), bool),
                'diagnostics': {}, 'initialization_time': .32}

    monkeypatch.setattr(recovery, 'navigate', navigate)
    monkeypatch.setattr(recovery, 'failure_evidence',
                        lambda *args: {'index': 1, 'observation_time_s': .31})
    monkeypatch.setattr(recovery, 'check_prefix', lambda *args: None)
    monkeypatch.setattr(recovery, 'quality_support_counts',
                        lambda *args: np.array([2, 2]))
    monkeypatch.setattr(recovery, 'score_navigation',
                        lambda *args: {'max_error_m': .5})

    arrays, modes, positions = recovery.replay_modes(
        tracking, trace, coherent, {'velocity_east': 2.})
    assert calls == ['original', 'short']
    assert modes['original']['status'] == 'solver_failed'
    assert modes['original']['failure']['index'] == 1
    assert 'published' not in modes['original']
    assert modes['short']['status'] == 'complete'
    assert positions['original'] is None
    np.testing.assert_array_equal(positions['short'], np.zeros((2, 3)))
    assert not any(key.startswith('original_') for key in arrays)
    assert 'short_position' in arrays


def test_unexpected_exception_is_not_swallowed(monkeypatch):
    monkeypatch.setattr(recovery, 'projection_eligibility',
                        lambda *args: (np.array([True]), np.zeros(1, int),
                                       np.zeros((1, 5), int)))
    monkeypatch.setattr(recovery, 'navigation_input', lambda *args: {})
    monkeypatch.setattr(recovery, 'navigate',
                        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('bug')))
    tracking = {'time': np.array([.3]), 'available_time': np.array([.32]),
                'support': np.zeros((1, 1))}
    with pytest.raises(RuntimeError, match='bug'):
        recovery.replay_modes(tracking, {}, {'support': np.ones((1, 1))},
                              {'velocity_east': 2.})


def test_completed_receipt_is_reused_without_rewriting(tmp_path, monkeypatch):
    folder = tmp_path/'yaw_plus2'
    folder.mkdir()
    receipt = dict(input_sha256={'samples.npy': 'input'},
                   source_sha256={'core/single_source.py': 'source'},
                   yaw_deg=2., diagnostics_sha256='hash',
                   output_sha256={'tracking.npz': 'science'})
    path = folder/'path_receipt.json'
    path.write_text(json.dumps(receipt))
    (folder/'physical_diagnostics.npz').write_bytes(b'old diagnostics')
    monkeypatch.setattr(recovery, 'digest', lambda *args: 'hash')
    view = {'observation': np.array([.3])}
    monkeypatch.setattr(recovery, 'recover_path',
                        lambda *args: ({'navigation': {'original': {}, 'short': {}},
                                        'output_sha256': receipt['output_sha256']}, view))
    monkeypatch.setattr(recovery, 'replay_modes',
                        lambda *args: pytest.fail('completed navigation reran'))
    result, returned = recovery.process_path(
        folder, {}, {}, 2., receipt['input_sha256'], receipt['source_sha256'],
        'recovery source')
    assert result['provenance'] == 'frozen_receipt'
    assert returned is view
    assert path.read_text() == json.dumps(receipt)
    assert (folder/'physical_diagnostics.npz').read_bytes() == b'old diagnostics'


def test_recovery_receipt_is_reused_and_partial_path_is_preserved(tmp_path,
                                                                 monkeypatch):
    completed = tmp_path/'yaw_minus2'
    completed.mkdir()
    receipt = dict(provenance='recovery_receipt',
                   input_sha256={'samples.npy': 'input'},
                   frozen_source_sha256={'core/single_source.py': 'source'},
                   recovery_source_sha256='recovery', yaw_deg=-2.,
                   frontend_sha256={name: 'hash' for name in recovery.FRONTEND_FILES},
                   navigation_file='navigation_short.npz',
                   navigation_sha256='hash', diagnostics_sha256='hash',
                   navigation_modes={'original': {'status': 'solver_failed'},
                                     'short': {'status': 'complete', 'score': {}}})
    receipt_path = completed/'recovery_receipt.json'
    receipt_path.write_text(json.dumps(receipt))
    monkeypatch.setattr(recovery, 'hashes', lambda *args: receipt['frontend_sha256'])
    monkeypatch.setattr(recovery, 'digest', lambda *args: 'hash')
    view = {'position': {'original': None}}
    monkeypatch.setattr(recovery, 'saved_view', lambda *args: view)
    monkeypatch.setattr(recovery, 'replay_modes',
                        lambda *args: pytest.fail('recovery navigation reran'))
    reused, returned = recovery.process_path(completed, {}, {}, -2.,
        receipt['input_sha256'], receipt['frozen_source_sha256'], 'recovery')
    assert reused == receipt and returned is view
    assert receipt_path.read_text() == json.dumps(receipt)

    partial = tmp_path/'yaw_plus2'
    partial.mkdir()
    (partial/'tracking.npz').write_bytes(b'partial receiver data')
    with pytest.raises(AssertionError, match='Incomplete receiver path retained'):
        recovery.process_path(partial, {}, {}, 2., receipt['input_sha256'],
                              receipt['frozen_source_sha256'], 'recovery')
    assert (partial/'tracking.npz').read_bytes() == b'partial receiver data'


def test_pending_path_uses_frozen_frontend_settings_before_mode_replay(
        tmp_path, monkeypatch):
    observed = {}

    def frontend(record, antennas, **options):
        observed.update(options)
        assert antennas == recovery.ANTENNAS
        options['stream_memmap_path'].write_bytes(b'temporary stream')
        return {'time': np.array([.3]),
                'trace_raw_trigger': np.array([np.nan])}

    monkeypatch.setattr(recovery, 'frontend', frontend)

    def save_loops(folder, loops):
        np.savez_compressed(folder/'loops.npz', empty=np.array([]))
        return {'accepted_segments': 0, 'accepted_epochs': 0}

    monkeypatch.setattr(recovery, 'save_loops', save_loops)
    monkeypatch.setattr(recovery, 'audit_blocks',
                        lambda *args: {'raw': np.array([np.nan])})
    monkeypatch.setattr(recovery, 'analyze_trace',
                        lambda *args: {'history': np.array([1])})
    monkeypatch.setattr(recovery, 'analyze_groups',
                        lambda *args: {'groups': np.array([1])})
    monkeypatch.setattr(recovery, 'build_coherent_support',
                        lambda *args: {'support': np.ones((1, 1))})
    folder = tmp_path/'scene'/'yaw_plus2'
    status, _ = recovery.run_frontend({'samples': np.ones((3, 2))}, folder)
    assert status['loop_history']['accepted_segments'] == 0
    assert observed['beam_template_carrier_aiding'] is True
    assert observed['beam_coherent_ms'] == 5
    assert observed['beam_startup_amplitude'] is True
    assert observed['fast_truncation'] is True
    assert observed['beam_projection_threshold'] == 3.
    assert not (folder/'beam_stream.dat').exists()
    assert all((folder/name).exists() for name in recovery.FRONTEND_FILES)


def test_zero_yaw_receipt_must_match_verified_stage47_summary(tmp_path,
                                                               monkeypatch):
    folder = tmp_path/'scene'
    gate = folder/'gate3'
    gate.mkdir(parents=True)
    (gate/'path_receipt.json').write_text(json.dumps({
        'output_sha256': {'tracking.npz': 'tampered'}}))
    monkeypatch.setattr(recovery, 'hashes',
                        lambda *args: {'tracking.npz': 'tampered'})
    baseline = {'cases': {str(recovery.study.SEED): {
        'h0_drop': {'paths': {'gate3': {
            'output_sha256': {'tracking.npz': 'verified'}}}}}}}
    with pytest.raises(AssertionError):
        recovery.baseline_view(folder, baseline, 'h0_drop')


def test_main_mixed_frozen_and_recovery_receipts_write_correct_manifest(
        tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, 'OUTPUT', tmp_path/'new')
    recovery.OUTPUT.mkdir()
    monkeypatch.setattr(recovery.study, 'STAGE47', tmp_path/'old')
    sources = {'experiments/single_source_nominal_yaw.py': 'source'}
    old_outputs = {name: 'old science' for name in recovery.study.SCIENCE}
    baseline = {'cases': {str(recovery.study.SEED): {
        scene: {'input_sha256': {'samples.npy': 'input'},
                'paths': {'gate3': {'output_sha256': old_outputs}}}
        for scene in recovery.study.SCENES}}}
    frozen = {'plan.json': 'plan', 'source_snapshot.zip': 'snapshot'}
    monkeypatch.setattr(recovery, 'preflight',
                        lambda: (baseline, sources, frozen))
    monkeypatch.setattr(recovery, 'snapshot_recovery_source', lambda: 'hash')
    monkeypatch.setattr(recovery, 'source_hashes', lambda: sources)
    monkeypatch.setattr(recovery, 'hashes',
                        lambda folder, names: frozen if folder == recovery.OUTPUT
                        else old_outputs)
    hashed_paths = []

    def digest(path):
        hashed_paths.append(path)
        return 'hash'

    monkeypatch.setattr(recovery, 'digest', digest)
    monkeypatch.setattr(recovery.study, 'load_input',
                        lambda *args: ({}, {'case': {'velocity_east': 2.}},
                                       {'samples.npy': 'input'}, None, None))
    monkeypatch.setattr(recovery, 'baseline_view',
                        lambda *args: ({'position': {'short': None}},
                                       {'output_sha256': old_outputs}))
    completed_modes = {mode: {'status': 'complete', 'score': {}}
                       for mode in ('short', 'original')}

    def process(folder, record, case, yaw, input_sha, sources, recovery_sha):
        if folder.parent.name == recovery.study.SCENES[0] and yaw == -2.:
            return ({'status': 'complete', 'provenance': 'frozen_receipt',
                     'output_sha256': old_outputs,
                     'navigation_modes': completed_modes}, {})
        return ({'status': 'complete', 'provenance': 'recovery_receipt',
                 'frontend_sha256': {'tracking.npz': 'new'},
                 'navigation_file': 'navigation.npz',
                 'navigation_sha256': 'nav', 'diagnostics_sha256': 'diag',
                 'navigation_modes': completed_modes}, {})

    monkeypatch.setattr(recovery, 'process_path', process)
    monkeypatch.setattr(recovery, 'score_scene', lambda *args: {
        f'{yaw:+g}': {'short': {'score': {'pass': True}}}
        for yaw in recovery.study.YAWS_DEG[1:]})
    monkeypatch.setattr(recovery, 'plot_scene', lambda *args: None)
    summary = recovery.main()
    assert summary['complete_navigation_paths'] == 6
    assert summary['short_screen_pass']
    assert recovery.OUTPUT/recovery.study.SCENES[0]/'yaw_minus2'/'path_receipt.json' in hashed_paths
    assert recovery.OUTPUT/recovery.study.SCENES[0]/'yaw_plus2'/'recovery_receipt.json' in hashed_paths


@pytest.mark.local_evidence
def test_saved_weak_minus_two_failure_and_short_replay_are_independent():
    folder = recovery.OUTPUT/'weak'/'yaw_minus2'
    original_hashes = recovery.hashes(folder, recovery.FRONTEND_FILES)
    tracking, trace, _, coherent = recovery.verify_frontend(folder)
    arrays, modes, positions = recovery.replay_modes(
        tracking, trace, coherent, {'velocity_east': 2.})
    failure = modes['original']['failure']
    assert modes['original']['status'] == 'solver_failed'
    assert positions['original'] is None
    assert failure['index'] == 407
    assert failure['selected_prns'] == [4, 5, 7, 9]
    assert failure['reference_rank'] == 4
    assert abs(failure['reference_condition']-649.9496885729169) < 1e-6
    assert failure['geocentric_iterations'][-1]['iteration'] == 6
    assert failure['geocentric_iterations'][-1]['condition'] > 2e9
    assert modes['short']['status'] == 'complete'
    assert modes['short']['score']['all']['count'] == 1974
    assert np.isfinite(positions['short'][tracking['time'] >= .3]).all(axis=1).sum() == 1969
    assert abs(modes['short']['score']['all']['max_error_m']-
               2.3291838381300987) < 1e-9
    observation_mask = tracking['time'] >= .3
    true_east = 2.*tracking['time'][observation_mask]
    observed_error = np.linalg.norm(positions['short'][observation_mask, :2]-
        np.column_stack((true_east, np.zeros(len(true_east)))), axis=1)
    assert abs(np.percentile(observed_error, 95)-2.204537056797767) < 1e-9
    assert modes['short']['score']['all']['error_ge_20'] == 0
    assert 'short_position' in arrays
    assert not any(key.startswith('original_') for key in arrays)
    assert recovery.hashes(folder, recovery.FRONTEND_FILES) == original_hashes
