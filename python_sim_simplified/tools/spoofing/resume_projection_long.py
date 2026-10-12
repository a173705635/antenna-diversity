"""Resume the interrupted, frozen Stage 47 long batch from saved evidence.

The first three seed-2026100402 inputs and six receiver paths were completed
before interruption. Their original frontend durations and deleted beam streams
cannot be recovered. This entry point checks saved arrays and finishes the batch.
"""
import gc
import json
import sys
from pathlib import Path

import numpy as np
import scipy

from core.provenance import source_hashes
from core.spoofing.recovery_geometry import FS
from core.serialization import serializable
from core.spoofing.single_source import navigate
from experiments.spoofing.single_source_coherence_diagnosis import analyze_groups
from experiments.spoofing.single_source_coherent_navigation import (
    build_coherent_support, quality_support_counts,
)
from experiments.spoofing.single_source_evidence_history import analyze_trace
from experiments.spoofing.single_source_phase_exclusion import verify_source_snapshot
from experiments.spoofing.single_source_phase_state_gate import projection_eligibility
from experiments.spoofing.single_source_startup_amplitude_rf import (
    check_prefix, score_navigation,
)
from experiments.spoofing.single_source_suppression_study import digest
from experiments.spoofing.single_source_temporal_trigger import navigation_input
from experiments.spoofing.single_source_template_coherence import STARTUP, WIDTH
from experiments.spoofing.single_source_projection_threshold import SCIENCE
from experiments.spoofing.single_source_ramp_takeover_long import save_new_input
from experiments.spoofing.single_source_history_study import generate_new_record
from experiments.spoofing import single_source_projection_long as study


OUTPUT = study.OUTPUT
EXISTING = ((2026100402, 'h0_constant'), (2026100402, 'h0_drop'),
            (2026100402, 'weak'))
INPUT_FILES = ('samples.npy', 'observations.npz', 'metadata.json')


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(serializable(value), indent=2,
                                    allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def runtime_versions():
    return dict(python=sys.version.split()[0], numpy=np.__version__,
                scipy=scipy.__version__)


def snapshot_recovery_source():
    """Keep the exact recovery implementation separate from the 98-source ZIP."""
    source = Path(__file__)
    frozen = OUTPUT/'resume_projection_long.py'
    current_hash = digest(source)
    if frozen.exists():
        assert digest(frozen) == current_hash
    else:
        temporary = frozen.with_name(frozen.name + '.tmp')
        temporary.write_bytes(source.read_bytes())
        temporary.replace(frozen)
    return current_hash


def read_npz(path):
    with np.load(path) as saved:
        return dict(saved)


def hashes(folder, names):
    return {name: digest(folder/name) for name in names}


def preflight():
    """Verify prior gates and exact frozen 98-source plan without writing it."""
    references, current = study.preflight()
    plan = json.loads((OUTPUT/'plan.json').read_text(encoding='utf-8'))
    assert len(plan['source_sha256']) == 98
    assert plan['source_sha256'] == current == source_hashes()
    assert tuple(plan['seeds']) == study.SEEDS
    assert tuple(plan['scenes']) == study.SCENES
    assert tuple(plan['gates']) == study.GATES
    for seed in study.SEEDS:
        for scene in study.SCENES:
            assert plan['scene_parameters'][str(seed)][scene] == serializable(
                study.long_plan(seed, scene))
    verify_source_snapshot(OUTPUT/'source_snapshot.zip', current)
    return plan, references


def remaining_paths(completed):
    return [(seed, scene, gate) for seed in study.SEEDS
            for scene in study.SCENES for gate in study.GATES
            if (seed, scene, gate) not in completed]


def input_hashes(folder):
    return hashes(folder/'input', INPUT_FILES)


def load_input(folder, settings, first_prefix, first_geometry):
    """Memory-map archived IQ and verify its metadata and shared seed geometry."""
    source = folder/'input'
    stored = input_hashes(folder)
    metadata = json.loads((source/'metadata.json').read_text(encoding='utf-8'))
    assert metadata['samples_sha256'] == stored['samples.npy']
    assert metadata['observations_sha256'] == stored['observations.npz']
    assert metadata['seed'] == settings['seed']
    assert metadata['scenario'] == ('no_attack_control' if folder.name.startswith('h0_')
                                    else folder.name)
    assert metadata['post_cn0_db_hz'] == settings['post_0p3s_cn0_db_hz']
    assert metadata['case']['duration'] == settings['duration_s']
    assert metadata['case']['stars'] == settings['satellites']
    assert metadata['case']['antennas'] == settings['generated_antennas']
    assert metadata['case']['cn0'] == settings['startup_cn0_db_hz']
    assert metadata['case']['attack_start'] == settings['attack_start_s']
    assert metadata['case']['spoof_db'] == settings['spoof_db']
    assert metadata['case']['fake_east'] == settings['fake_east_m']
    assert metadata['case']['angle'] == settings['angle_deg']
    assert metadata['case']['fake_rate'] == settings['fake_rate_mps']
    assert metadata['case']['velocity_east'] == 2.
    assert metadata['case']['genuine'] == (folder.name != 'genuine_absent')
    assert metadata['attacked'] == (folder.name in ('weak', 'genuine_absent'))
    with np.load(source/'observations.npz') as saved:
        record = dict(saved)
    record['samples'] = np.load(source/'samples.npy', mmap_mode='r')
    onset = metadata['pre_attack_exact_prefix_samples']
    assert onset == round(settings['attack_start_s']*FS)
    if first_prefix is None:
        first_prefix = record['samples'][:, :onset].copy()
        first_geometry = {key: value.copy() for key, value in record.items()
                          if key != 'samples'}
    else:
        np.testing.assert_array_equal(record['samples'][:, :onset], first_prefix)
        for key, value in first_geometry.items():
            np.testing.assert_array_equal(record[key], value, err_msg=key)
    return record, metadata, stored, first_prefix, first_geometry


def loop_count(folder):
    loops = read_npz(folder/'loops.npz')
    epochs = [len(values) for key, values in loops.items()
              if key.endswith('_epoch_start_samples')]
    return dict(accepted_segments=len(epochs), accepted_epochs=sum(epochs))


def saved_runtime(count):
    return dict(frontend_walltime_s=None,
                frontend_walltime_status='not_recorded_after_interruption',
                loop_history=count)


def check_saved_arrays(folder, tracking, trace):
    observations = dict(tracking, **trace)
    expected = {
        'history.npz': analyze_trace(observations),
        'groups.npz': analyze_groups(trace),
    }
    coherent = build_coherent_support(tracking, expected['groups.npz'])
    expected['support.npz'] = {f'coherent_{key}': value
                               for key, value in coherent.items()}
    for filename, arrays in expected.items():
        saved = read_npz(folder/filename)
        assert saved.keys() == arrays.keys()
        for key, value in arrays.items():
            np.testing.assert_array_equal(saved[key], value, err_msg=f'{filename}:{key}')
    return coherent


def check_saved_navigation(folder, tracking, trace, coherent, case):
    """Recompute both navigations in memory; never overwrite navigation.npz."""
    eligible, latest, recent = projection_eligibility(tracking['available_time'], trace)
    expected = dict(observation_time=tracking['time'],
                    available_time=tracking['available_time'],
                    phase_state_eligible=eligible,
                    latest_complete_block=latest,
                    recent_block_dimensions=recent)
    results = {}; positions = {}; counts = {}
    for mode in ('original', 'short'):
        support = tracking['support'] if mode == 'original' else coherent['support']
        options = dict(diagnostics=True, code_blend=.1 if mode == 'original' else 0.,
                       phase_leave_one_out=mode == 'short',
                       phase_state_continuity=mode == 'short')
        if mode == 'short':
            options['phase_state_eligible'] = eligible
        navigation = navigate(navigation_input(tracking, support), **options)
        check_prefix(tracking, support, eligible, navigation, mode)
        count = quality_support_counts(tracking, support)
        expected[f'{mode}_support_count'] = count
        for key, value in navigation.items():
            if key == 'diagnostics':
                for diagnostic, data in value.items():
                    expected[f'{mode}_diagnostic_{diagnostic}'] = data
            else:
                expected[f'{mode}_{key}'] = np.nan if value is None else value
        results[mode] = dict(**score_navigation(tracking, navigation,
                                                case['velocity_east']),
                             quality_support_ge4=int((count >= 4).sum()),
                             quality_support_ge6=int((count >= 6).sum()))
        positions[mode] = navigation['position'].copy()
        counts[mode] = count
    saved = read_npz(folder/'navigation.npz')
    assert saved.keys() == expected.keys()
    for key, value in expected.items():
        np.testing.assert_array_equal(saved[key], value, err_msg=f'navigation:{key}')
    return results, positions, counts


def recover_path(folder, record, case, gate, receipt=None):
    """Reconstruct saved statistics; prior beam stream is no longer available."""
    tracking = read_npz(folder/'tracking.npz')
    trace = read_npz(folder/'trace.npz')
    fit = read_npz(folder/'fit.npz')
    coherent = check_saved_arrays(folder, tracking, trace)
    navigation, positions, counts = check_saved_navigation(
        folder, tracking, trace, coherent, case)
    np.testing.assert_allclose(trace['trace_raw_trigger'], fit['raw'],
                               rtol=0, atol=1e-9, equal_nan=True)
    np.testing.assert_allclose(trace['trace_projectors'], fit['projector'],
                               rtol=0, atol=1e-9)
    np.testing.assert_array_equal(trace['trace_dimensions'], fit['dimension'])
    index = (tracking['time']/.02).astype(int)
    np.testing.assert_allclose(tracking['gain'], fit['gain'][index],
                               rtol=0, atol=1e-9)
    attack = round(case['attack_start']*FS/WIDTH)
    projection = dict(
        after_startup=dict(triggered=int((fit['raw'][STARTUP//WIDTH:] >= gate).sum()),
                           opportunities=len(fit['raw'])-STARTUP//WIDTH),
        after_attack=dict(triggered=int((fit['raw'][attack:] >= gate).sum()),
                          opportunities=len(fit['raw'])-attack))
    saved_hashes = hashes(folder, SCIENCE)
    if receipt is None:
        result = dict(**saved_runtime(loop_count(folder)),
                      projection_blocks=projection,
                      navigation=navigation, output_sha256=saved_hashes)
    else:
        assert receipt['output_sha256'] == saved_hashes
        assert receipt['projection_blocks'] == projection
        assert receipt['navigation'] == serializable(navigation)
        assert receipt['loop_history'] == loop_count(folder)
        result = receipt
    view = dict(observation=tracking['time'].copy(),
                available=tracking['available_time'].copy(),
                position=positions, support=counts, navigation=navigation)
    return result, view


def score_scene(scene, views, case):
    baseline, candidate = views[3.], views[2.]
    np.testing.assert_array_equal(baseline['observation'], candidate['observation'])
    np.testing.assert_array_equal(baseline['available'], candidate['available'])
    time = baseline['observation']
    assert int((time >= .3).sum()) == 1969
    assert int((time >= .5).sum()) == 1949
    velocity = case['velocity_east']
    short = study.long_screen(scene, time, baseline['position']['short'],
        candidate['position']['short'], velocity,
        baseline['navigation']['short']['max_relative_motion_error_m'],
        candidate['navigation']['short']['max_relative_motion_error_m'])
    short['relative']['baseline_initial_anchor_error_m'] = (
        baseline['navigation']['short']['initial_horizontal_error_m'])
    short['relative']['candidate_initial_anchor_error_m'] = (
        candidate['navigation']['short']['initial_horizontal_error_m'])
    original = study.pair_score(scene, time, baseline['position']['original'],
        candidate['position']['original'], velocity,
        baseline['navigation']['original']['max_relative_motion_error_m'],
        candidate['navigation']['original']['max_relative_motion_error_m'])
    original.pop('pass')
    original['baseline_initial_anchor_error_m'] = (
        baseline['navigation']['original']['initial_horizontal_error_m'])
    original['candidate_initial_anchor_error_m'] = (
        candidate['navigation']['original']['initial_horizontal_error_m'])
    if scene == 'genuine_absent':
        original['candidate_after_half_published'] = int(((time >= .5) &
            np.isfinite(candidate['position']['original']).all(axis=1)).sum())
    return short, original


def main():
    """Finish all paths even when a scientific screen already fails."""
    plan, references = preflight()
    frozen_hashes = hashes(OUTPUT, ('plan.json', 'source_snapshot.zip'))
    recovery_hash = snapshot_recovery_source()
    original_inputs = {f'{seed}/{scene}': input_hashes(OUTPUT/str(seed)/scene)
                       for seed, scene in EXISTING}
    original_science = {f'{seed}/{scene}/gate{gate:g}':
        hashes(OUTPUT/str(seed)/scene/f'gate{gate:g}', SCIENCE)
        for seed, scene in EXISTING for gate in study.GATES}
    evidence = dict(input_sha256=original_inputs, output_sha256=original_science,
                    recorded_at='resume_start_before_new_rf')
    evidence_path = OUTPUT/'recovery_evidence.json'
    if evidence_path.exists():
        assert json.loads(evidence_path.read_text(encoding='utf-8')) == evidence
    else:
        write_json(evidence_path, evidence)

    cases = {}; inputs = {}; outputs = {}; figures = {}
    for seed in study.SEEDS:
        seed_folder = OUTPUT/str(seed)
        seed_folder.mkdir(exist_ok=True)
        first_prefix = first_geometry = None
        cases[str(seed)] = {}; views = {}
        for scene in study.SCENES:
            folder = seed_folder/scene
            settings = study.long_plan(seed, scene)
            if (folder/'input'/'metadata.json').exists():
                record, identity, input_sha, first_prefix, first_geometry = (
                    load_input(folder, settings, first_prefix, first_geometry))
            else:
                generated = ('no_attack_control' if scene.startswith('h0_') else scene)
                record, identity = generate_new_record(
                    settings, generated, fake_rate=settings['fake_rate_mps'])
                folder.mkdir(exist_ok=True)
                input_sha, first_prefix, first_geometry = save_new_input(
                    folder, record, identity, first_prefix, first_geometry)
            inputs[f'{seed}/{scene}'] = input_sha
            paths = {}; views[scene] = {}
            for gate in study.GATES:
                name = f'gate{gate:g}'
                path_folder = folder/name
                receipt_path = path_folder/'path_receipt.json'
                if (seed, scene) in EXISTING:
                    result, view = recover_path(path_folder, record,
                                                identity['case'], gate)
                elif receipt_path.exists():
                    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
                    result, view = recover_path(path_folder, record,
                                                identity['case'], gate, receipt)
                elif all((path_folder/name).exists() for name in SCIENCE):
                    result, view = recover_path(path_folder, record,
                                                identity['case'], gate)
                    result['receipt_provenance'] = dict(
                        status='saved_science_without_receipt',
                        recorded_by_recovery_source_sha256=recovery_hash,
                        recorded_runtime=runtime_versions())
                    write_json(receipt_path, result)
                else:
                    result, view = study.run_path(record, identity['case'],
                                                  gate, path_folder)
                    result['receipt_provenance'] = dict(
                        status='new_rf_path',
                        created_by_recovery_source_sha256=recovery_hash,
                        created_runtime=runtime_versions())
                    write_json(receipt_path, result)
                paths[name] = result
                views[scene][gate] = view
                outputs[f'{seed}/{scene}/{name}'] = result['output_sha256']
                print(seed, scene, name, 'complete', flush=True)
            short, original = score_scene(scene, views[scene], identity['case'])
            cases[str(seed)][scene] = dict(input_sha256=input_sha, paths=paths,
                                          short_screen=short,
                                          original_diagnostic=original)
            print(seed, scene, 'short_pass', short['pass'], flush=True)
            del record
            gc.collect()
        study.plot_seed(seed, views, seed_folder)
        figures[str(seed)] = digest(seed_folder/'overview.png')
        del views
        gc.collect()
    assert source_hashes() == plan['source_sha256']
    assert digest(Path(__file__)) == recovery_hash
    assert digest(OUTPUT/'resume_projection_long.py') == recovery_hash
    assert hashes(OUTPUT, ('plan.json', 'source_snapshot.zip')) == frozen_hashes
    assert {key: input_hashes(OUTPUT/key) for key in original_inputs} == original_inputs
    assert {key: hashes(OUTPUT/key, SCIENCE) for key in original_science} == original_science
    manifest = dict(source_snapshot_sha256=frozen_hashes['source_snapshot.zip'],
        plan_sha256=frozen_hashes['plan.json'],
        reference_summary_sha256={name: digest(folder/'summary.json')
            for name, folder in (('stage45', study.STAGE45),
                                 ('stage46', study.STAGE46))},
        input_sha256=inputs, output_sha256=outputs, figure_sha256=figures,
        recovery_source_sha256=recovery_hash,
        recovery_source_snapshot_sha256=digest(OUTPUT/'resume_projection_long.py'),
        recovery_evidence_sha256=digest(evidence_path),
        recovery_provenance=dict(saved_paths=len(original_science),
            saved_frontend_walltime='not_recorded_after_interruption',
            saved_beam_stream='deleted_before_interruption; saved arrays checked in memory',
            new_paths=18))
    write_json(OUTPUT/'manifest.json', manifest)
    summary = dict(scope=study.__doc__, source_sha256=plan['source_sha256'],
        source_snapshot_sha256=frozen_hashes['source_snapshot.zip'],
        plan_sha256=frozen_hashes['plan.json'],
        manifest_sha256=digest(OUTPUT/'manifest.json'), figure_sha256=figures,
        recovery_source_sha256=recovery_hash,
        recovery_source_snapshot_sha256=digest(OUTPUT/'resume_projection_long.py'),
        recovery_evidence_sha256=digest(evidence_path),
        recovery_provenance=manifest['recovery_provenance'],
        screen_pass=all(cases[str(seed)][scene]['short_screen']['pass']
                        for seed in study.SEEDS for scene in study.SCENES),
        cases=cases)
    write_json(OUTPUT/'summary.json', summary)
    return summary


if __name__ == '__main__':
    main()
