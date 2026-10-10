"""Fixed three-seed, 20 s paired projection-gate RF verification.

Each of four scenes per seed generates IQ once and reuses it for gates 3/2.
All 24 receiver paths run even when an earlier scientific screen is no-go.
Truth enters only the post-receiver publication and position scoring.
"""
import gc
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy

from core.provenance import source_hashes
from core.serialization import serializable
from experiments.paths import RESULTS
from experiments.single_source_history_study import generate_new_record
from experiments.single_source_long_coherent import source_snapshot
from experiments.single_source_phase_exclusion import verify_source_snapshot
from experiments.single_source_projection_regression import (
    OUTPUT as STAGE46, STAGE45_SUMMARY_SHA256,
)
from experiments.single_source_projection_threshold import (
    OUTPUT as STAGE45, GATES, SCENES, pair_score, run_path, scene_plan,
)
from experiments.single_source_ramp_takeover_long import save_new_input
from experiments.single_source_suppression_study import (
    ANTENNAS, OUTPUT as STUDY_OUTPUT, RECEIVER_PLL, digest,
)


OUTPUT = RESULTS/STUDY_OUTPUT/'projection_threshold_long'
SEEDS = (2026100402, 2026100403, 2026100404)
STAGE46_SUMMARY_SHA256 = '761da01ee0d8e605011f6f32f2aeff5d2b47761701aa0acf2cbdbe4bfd664374'


def long_plan(seed, scene):
    """Change only the draw and duration of the fixed Stage 45 geometry."""
    return {**scene_plan(scene), 'seed': seed, 'duration_s': 20.}


def long_screen(scene, time, baseline, candidate, velocity_east,
                baseline_motion, candidate_motion):
    """Apply the existing relative rule plus long-batch absolute gates."""
    relative = pair_score(scene, time, baseline, candidate, velocity_east,
                          baseline_motion, candidate_motion)
    serious = relative['candidate']['error_ge20']
    result = dict(relative=relative, candidate_ge20=serious)
    absolute_pass = serious == 0
    if scene == 'genuine_absent':
        published = np.isfinite(candidate).all(axis=1)
        later = int(((time >= .5) & published).sum())
        result['candidate_after_half_published'] = later
        absolute_pass = absolute_pass and later == 0
    result['pass'] = bool(relative['pass'] and absolute_pass)
    return result


def preflight():
    """Check both completed gates and every frozen source byte before RF."""
    summaries = {name: json.loads((folder/'summary.json').read_text(
        encoding='utf-8')) for name, folder in (('stage45', STAGE45),
                                                ('stage46', STAGE46))}
    assert digest(STAGE45/'summary.json') == STAGE45_SUMMARY_SHA256
    assert digest(STAGE46/'summary.json') == STAGE46_SUMMARY_SHA256
    assert summaries['stage45']['screen_pass']
    assert all(summaries['stage45']['cases'][scene]['short_screen']['pass']
               for scene in SCENES)
    assert summaries['stage46']['screen_pass']
    assert summaries['stage46']['short_screen']['pass']
    current = source_hashes()
    assert len(summaries['stage45']['source_sha256']) == 96
    assert len(summaries['stage46']['source_sha256']) == 97
    assert set(current)-set(summaries['stage46']['source_sha256']) == {
        'experiments/single_source_projection_long.py'}
    for name, folder in (('stage45', STAGE45), ('stage46', STAGE46)):
        summary = summaries[name]
        assert {path: current[path] for path in summary['source_sha256']} == (
            summary['source_sha256'])
        assert summary['source_snapshot_sha256'] == digest(folder/'source_snapshot.zip')
        verify_source_snapshot(folder/'source_snapshot.zip', summary['source_sha256'])
        assert summary['plan_sha256'] == digest(folder/'plan.json')
        assert summary['manifest_sha256'] == digest(folder/'manifest.json')
    return summaries, current


def plot_seed(seed, views, folder):
    """Keep isolated releases visible over each full 20 s record."""
    figure, axes = plt.subplots(4, 2, figsize=(12, 11), constrained_layout=True)
    for row, scene in enumerate(SCENES):
        for threshold in GATES:
            view = views[scene][threshold]
            time = view['observation']
            truth = np.column_stack((2.*time, np.zeros(len(time))))
            for mode, style in (('short', '-'), ('original', '--')):
                position = view['position'][mode]
                error = np.linalg.norm(position[:, :2]-truth, axis=1)
                label = f'gate {threshold:g} {mode}'
                axes[row, 0].plot(time, view['support'][mode],
                                  linestyle=style, linewidth=.7, label=label)
                axes[row, 1].plot(time, error, linestyle=style,
                                  linewidth=.7, marker='.', markersize=1,
                                  label=label)
        axes[row, 0].set(title=f'{seed} {scene}', ylabel='Qualified support count')
        axes[row, 1].set(ylabel='Published horizontal error (m)')
        for axis in axes[row]:
            axis.set(xlabel='Observation time (s)',
                     xlim=(0., views[scene][3.]['observation'][-1]))
            axis.grid(alpha=.25)
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].legend(fontsize=7)
    figure.savefig(folder/'overview.png', dpi=150)
    plt.close(figure)


def main():
    """Generate 12 fixed IQ scenes, then complete both gates on each."""
    summaries, sources = preflight()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_snapshot(OUTPUT/'source_snapshot.zip', sources)
    plan = dict(scope=__doc__, seeds=SEEDS, scenes=SCENES, gates=GATES,
        scene_parameters={str(seed): {scene: long_plan(seed, scene)
            for scene in SCENES} for seed in SEEDS},
        receiver=dict(processed_antennas=ANTENNAS, receiver_pll=RECEIVER_PLL,
            beam_template_carrier_aiding=True, beam_coherent_ms=5,
            beam_startup_amplitude=True, fast_truncation=True,
            navigation_modes=('original', 'short')),
        evaluation=('Stage 45 relative rules; gate 2 H0/weak observation >=0.3 s '
                    'zero errors >=20 m; absent observation >=0.5 s zero '
                    'publications and >=0.3 s zero errors >=20 m'),
        runtime=dict(python=sys.version.split()[0], numpy=np.__version__,
                     scipy=scipy.__version__), source_sha256=sources)
    (OUTPUT/'plan.json').write_text(json.dumps(serializable(plan), indent=2,
        allow_nan=False), encoding='utf-8')
    cases = {}; inputs = {}; outputs = {}; figures = {}
    for seed in SEEDS:
        seed_folder = OUTPUT/str(seed)
        seed_folder.mkdir(exist_ok=True)
        first_prefix = first_geometry = None
        cases[str(seed)] = {}; views = {}
        for scene in SCENES:
            settings = long_plan(seed, scene)
            generated_scene = ('no_attack_control' if scene.startswith('h0_')
                               else scene)
            print(seed, scene, 'IQ generation start', flush=True)
            record, identity = generate_new_record(
                settings, generated_scene, fake_rate=settings['fake_rate_mps'])
            folder = seed_folder/scene
            folder.mkdir(exist_ok=True)
            input_hashes, first_prefix, first_geometry = save_new_input(
                folder, record, identity, first_prefix, first_geometry)
            inputs[f'{seed}/{scene}'] = input_hashes
            paths = {}; views[scene] = {}
            for threshold in GATES:
                name = f'gate{threshold:g}'
                paths[name], views[scene][threshold] = run_path(
                    record, identity['case'], threshold, folder/name)
                outputs[f'{seed}/{scene}/{name}'] = paths[name]['output_sha256']
                print(seed, scene, name, 'RF, audit and navigation saved', flush=True)
            baseline = views[scene][3.]
            candidate = views[scene][2.]
            np.testing.assert_array_equal(baseline['observation'],
                                          candidate['observation'])
            np.testing.assert_array_equal(baseline['available'],
                                          candidate['available'])
            time = baseline['observation']
            assert int((time >= .3).sum()) == 1969
            assert int((time >= .5).sum()) == 1949
            velocity = identity['case']['velocity_east']
            old_motion = baseline['navigation']['short'][
                'max_relative_motion_error_m']
            new_motion = candidate['navigation']['short'][
                'max_relative_motion_error_m']
            short = long_screen(scene, time, baseline['position']['short'],
                candidate['position']['short'], velocity, old_motion, new_motion)
            short['relative']['baseline_initial_anchor_error_m'] = (
                baseline['navigation']['short']['initial_horizontal_error_m'])
            short['relative']['candidate_initial_anchor_error_m'] = (
                candidate['navigation']['short']['initial_horizontal_error_m'])
            original = pair_score(scene, time, baseline['position']['original'],
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
            cases[str(seed)][scene] = dict(input_sha256=input_hashes,
                paths=paths, short_screen=short, original_diagnostic=original)
            old = short['relative']['baseline']
            new = short['relative']['candidate']
            print(seed, scene, 'short_pass', short['pass'],
                'published', old['published'], '->', new['published'],
                'max_m', old['max_error_m'], '->', new['max_error_m'],
                'p95_m', old['p95_error_m'], '->', new['p95_error_m'],
                'candidate_ge20', short['candidate_ge20'],
                'candidate_after_half', short.get('candidate_after_half_published'),
                flush=True)
            del record
            gc.collect()
        plot_seed(seed, views, seed_folder)
        figures[str(seed)] = digest(seed_folder/'overview.png')
        del views
        gc.collect()
    assert source_hashes() == sources
    manifest = dict(source_snapshot_sha256=digest(OUTPUT/'source_snapshot.zip'),
        plan_sha256=digest(OUTPUT/'plan.json'),
        reference_summary_sha256={name: digest(folder/'summary.json')
            for name, folder in (('stage45', STAGE45), ('stage46', STAGE46))},
        input_sha256=inputs, output_sha256=outputs, figure_sha256=figures)
    (OUTPUT/'manifest.json').write_text(json.dumps(serializable(manifest),
        indent=2, allow_nan=False), encoding='utf-8')
    summary = dict(scope=__doc__, source_sha256=sources,
        source_snapshot_sha256=digest(OUTPUT/'source_snapshot.zip'),
        plan_sha256=digest(OUTPUT/'plan.json'),
        manifest_sha256=digest(OUTPUT/'manifest.json'),
        figure_sha256=figures,
        screen_pass=all(cases[str(seed)][scene]['short_screen']['pass']
                        for seed in SEEDS for scene in SCENES),
        cases=cases)
    (OUTPUT/'summary.json').write_text(json.dumps(serializable(summary),
        indent=2, allow_nan=False), encoding='utf-8')
    return summary


if __name__ == '__main__':
    main()
