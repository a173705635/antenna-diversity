"""Focused checks for the interrupted Stage 47 resume entry point."""
import json

import numpy as np

from tools import resume_projection_long as resume


def test_reconstructed_loop_count_and_missing_runtime(tmp_path):
    folder = tmp_path / 'gate3'
    folder.mkdir()
    np.savez_compressed(folder / 'loops.npz',
                        prn1_arc1_epoch_start_samples=np.arange(3),
                        prn1_arc1_phase=np.arange(3),
                        prn2_arc1_epoch_start_samples=np.arange(2))
    assert resume.loop_count(folder) == dict(accepted_segments=2,
                                              accepted_epochs=5)
    result = resume.saved_runtime(resume.loop_count(folder))
    assert result['frontend_walltime_s'] is None
    assert result['frontend_walltime_status'] == 'not_recorded_after_interruption'


def test_preflight_preserves_frozen_files(tmp_path, monkeypatch):
    output = tmp_path / 'long'
    output.mkdir()
    sources = {f'core/a{i}.py': 'frozen' for i in range(98)}
    plan = {'source_sha256': sources,
            'seeds': list(resume.study.SEEDS),
            'scenes': list(resume.study.SCENES),
            'gates': list(resume.study.GATES),
            'scene_parameters': {str(seed): {
                scene: resume.study.long_plan(seed, scene)
                for scene in resume.study.SCENES} for seed in resume.study.SEEDS}}
    (output / 'plan.json').write_text(json.dumps(plan), encoding='utf-8')
    (output / 'source_snapshot.zip').write_bytes(b'frozen')
    monkeypatch.setattr(resume, 'OUTPUT', output)
    monkeypatch.setattr(resume, 'source_hashes', lambda: plan['source_sha256'])
    monkeypatch.setattr(resume, 'verify_source_snapshot', lambda *args: None)
    monkeypatch.setattr(resume.study, 'preflight', lambda: ({}, sources))
    before = {name: (output / name).read_bytes()
              for name in ('plan.json', 'source_snapshot.zip')}
    assert resume.preflight()[0] == json.loads(json.dumps(plan))
    assert before == {name: (output / name).read_bytes() for name in before}


def test_remaining_paths_keep_all_scenes_after_no_go():
    completed = {(2026100402, scene, gate)
                 for scene in ('h0_constant', 'h0_drop', 'weak')
                 for gate in (3., 2.)}
    remaining = resume.remaining_paths(completed)
    assert len(remaining) == 18
    assert remaining[0] == (2026100402, 'genuine_absent', 3.)
    assert remaining[-1] == (2026100404, 'genuine_absent', 2.)


def test_saved_input_is_read_only_and_memory_mapped(tmp_path):
    folder = tmp_path / 'h0_constant'
    source = folder / 'input'
    source.mkdir(parents=True)
    settings = resume.study.long_plan(2026100402, 'h0_constant')
    samples = np.zeros((4, round(.3 * resume.FS) + 1), complex)
    np.save(source / 'samples.npy', samples)
    np.savez_compressed(source / 'observations.npz', codes=np.arange(2))
    case = dict(duration=20., stars=24, antennas=4, cn0=45., attack_start=.3,
                spoof_db=-5., fake_east=0., angle=5., fake_rate=3.,
                velocity_east=2., genuine=True)
    metadata = dict(seed=2026100402, scenario='no_attack_control',
                    post_cn0_db_hz=45., attacked=False, case=case,
                    pre_attack_exact_prefix_samples=round(.3 * resume.FS),
                    samples_sha256=resume.digest(source / 'samples.npy'),
                    observations_sha256=resume.digest(source / 'observations.npz'))
    (source / 'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
    before = resume.input_hashes(folder)
    record, identity, hashes, prefix, geometry = resume.load_input(
        folder, settings, None, None)
    assert isinstance(record['samples'], np.memmap)
    assert identity == metadata
    assert hashes == before == resume.input_hashes(folder)
    assert len(prefix[0]) == metadata['pre_attack_exact_prefix_samples']
    assert 'codes' in geometry


def test_scene_scoring_keeps_absolute_twenty_metre_rule():
    time = np.arange(.31, 20., .01)
    time[19] = np.nextafter(.5, 0.)
    positions = np.column_stack((2. * time, np.zeros(len(time))))
    positions[0, 0] += 20.
    score = dict(max_relative_motion_error_m=.1,
                 initial_horizontal_error_m=.1)
    view = dict(observation=time, available=time+.01,
                position={'short': positions, 'original': positions},
                navigation={'short': score, 'original': score})
    short, _ = resume.score_scene('h0_constant', {3.: view, 2.: view},
                                  {'velocity_east': 2.})
    assert short['relative']['pass']
    assert short['candidate_ge20'] == 1
    assert not short['pass']


def test_main_resumes_eighteen_paths_after_no_go(tmp_path, monkeypatch):
    output = tmp_path / 'long'
    output.mkdir()
    sources = {'frozen': 'hash'}
    monkeypatch.setattr(resume, 'OUTPUT', output)
    monkeypatch.setattr(resume, 'preflight', lambda: ({'source_sha256': sources}, {}))
    monkeypatch.setattr(resume, 'source_hashes', lambda: sources)
    monkeypatch.setattr(resume, 'digest', lambda path: 'hash')
    monkeypatch.setattr(resume, 'input_hashes', lambda folder: {'input': 'hash'})
    monkeypatch.setattr(resume, 'hashes',
                        lambda folder, names: {name: 'hash' for name in names})
    monkeypatch.setattr(resume, 'score_scene',
                        lambda *args: ({'pass': False}, {}))
    monkeypatch.setattr(resume.study, 'plot_seed',
                        lambda seed, views, folder: (folder/'overview.png').write_bytes(b'plot'))
    monkeypatch.setattr(resume, 'generate_new_record',
                        lambda settings, scene, fake_rate: (
                            {'seed': settings['seed']}, {'case': {'velocity_east': 2.}}))

    def load(folder, settings, prefix, geometry):
        return ({'seed': settings['seed']}, {'case': {'velocity_east': 2.}},
                {'input': 'hash'}, settings['seed'], settings['seed'])

    def save(folder, record, identity, prefix, geometry):
        (folder/'input').mkdir()
        (folder/'input'/'metadata.json').write_text('{}', encoding='utf-8')
        return {'input': 'hash'}, record['seed'], record['seed']

    def recovered(folder, record, case, gate, receipt=None):
        recovered_paths.append((record['seed'], folder.parent.name, gate,
                                receipt is not None))
        return receipt or {'output_sha256': {'tracking.npz': 'hash'}}, {}

    def run(record, case, gate, folder):
        run_paths.append((record['seed'], folder.parent.name, gate))
        folder.mkdir()
        return {'output_sha256': {'tracking.npz': 'hash'}}, {}

    monkeypatch.setattr(resume, 'load_input', load)
    monkeypatch.setattr(resume, 'save_new_input', save)
    monkeypatch.setattr(resume, 'recover_path', recovered)
    monkeypatch.setattr(resume.study, 'run_path', run)
    recovered_paths = []
    run_paths = []
    for seed, scene in resume.EXISTING:
        old_input = output/str(seed)/scene/'input'
        old_input.mkdir(parents=True)
        (old_input/'metadata.json').write_text('{}', encoding='utf-8')
    receipt_folder = output/'2026100402'/'genuine_absent'/'gate3'
    receipt_folder.mkdir(parents=True)
    old_receipt = {'output_sha256': {'tracking.npz': 'hash'},
                   'receipt_provenance': {'created_runtime': {'python': 'old'}}}
    resume.write_json(receipt_folder/'path_receipt.json', old_receipt)
    complete_folder = output/'2026100403'/'h0_constant'/'gate2'
    complete_folder.mkdir(parents=True)
    for name in resume.SCIENCE:
        (complete_folder/name).write_bytes(b'saved')

    result = resume.main()
    assert len(recovered_paths) == 8
    assert len(run_paths) == 16
    assert all((seed, scene) not in resume.EXISTING
               for seed, scene, gate in run_paths)
    assert (2026100402, 'genuine_absent', 3., True) in recovered_paths
    assert (2026100403, 'h0_constant', 2., False) in recovered_paths
    assert json.loads((receipt_folder/'path_receipt.json').read_text(
        encoding='utf-8')) == old_receipt
    assert json.loads((complete_folder/'path_receipt.json').read_text(
        encoding='utf-8'))['receipt_provenance']['status'] == (
            'saved_science_without_receipt')
    new_receipt = json.loads((output/'2026100404'/'weak'/'gate3'/
                              'path_receipt.json').read_text(encoding='utf-8'))
    assert new_receipt['receipt_provenance'] == dict(
        status='new_rf_path', created_by_recovery_source_sha256='hash',
        created_runtime=resume.runtime_versions())
    assert not (output/'2026100404'/'weak'/'gate3'/
                'path_receipt.json.tmp').exists()
    assert result['screen_pass'] is False
    assert len(result['cases']) == 3
    assert (output/'resume_projection_long.py').read_bytes() == resume.Path(
        resume.__file__).read_bytes()
