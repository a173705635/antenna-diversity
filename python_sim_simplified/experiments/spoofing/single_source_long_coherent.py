"""Fixed eight-second three-scene coherent-support RF study.

Generate four-channel paired simulated IQ, process three antennas, and keep
the original and two coherent-navigation variants in separate states.
"""
import gc
import hashlib
import json
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from core.provenance import source_hashes
from core.spoofing.recovery_geometry import FS
from core.serialization import serializable
from core.spoofing.single_source import frontend, navigate
from experiments.paths import SPOOFING_RESULTS as RESULTS
from experiments.spoofing.single_source_code_blend import published_metrics
from experiments.spoofing.single_source_coherence_diagnosis import analyze_groups
from experiments.spoofing.single_source_coherent_navigation import (
    build_coherent_support, quality_support_counts, score_mode,
)
from experiments.spoofing.single_source_evidence_history import analyze_trace
from experiments.spoofing.single_source_history_study import generate_new_record
from experiments.spoofing.single_source_suppression_study import (
    ANTENNAS, OUTPUT as STUDY_OUTPUT, RECEIVER_PLL, digest,
)


OUTPUT = RESULTS / STUDY_OUTPUT / 'long_coherent'
SCENES = ('weak', 'no_attack_control', 'genuine_absent')
PLAN = dict(seed=2026100301, duration_s=8., satellites=24,
            startup_cn0_db_hz=45., post_0p3s_cn0_db_hz=30.,
            attack_start_s=.3, spoof_db=10., angle_deg=60.,
            fake_east_m=100., generated_antennas=4,
            processed_antennas=ANTENNAS, beam_history_blocks=1,
            receiver_pll=RECEIVER_PLL, scenarios=SCENES)
MODES = ('original', 'coherent_.1', 'coherent_0')


def source_snapshot(path, hashes):
    """Save and verify the exact source bytes named by source_hashes."""
    root = RESULTS.parent.parent
    with ZipFile(path, 'w', compression=ZIP_DEFLATED) as archive:
        for name in sorted(hashes):
            archive.write(root / name, arcname=name)
    with ZipFile(path) as archive:
        for name, expected in hashes.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected


def navigation_prefix(observations, count):
    epochs = len(observations['time'])
    return {name: (value[:count] if isinstance(value, np.ndarray)
                   and value.ndim and len(value) == epochs else value)
            for name, value in observations.items()}


def save_navigation(arrays, modes):
    for mode, navigation in modes.items():
        for key, value in navigation.items():
            if key == 'diagnostics':
                for diagnostic, data in value.items():
                    arrays[f'{mode}_diagnostic_{diagnostic}'] = data
            else:
                arrays[f'{mode}_{key}'] = np.nan if value is None else value


def check_prefix(observations, full, blend):
    count = len(observations['time']) // 2
    prefix = navigate(navigation_prefix(observations, count),
                      diagnostics=True, code_blend=blend)
    for key, value in prefix.items():
        if key == 'diagnostics':
            for diagnostic, data in value.items():
                np.testing.assert_array_equal(data,
                                              full[key][diagnostic][:count],
                                              err_msg=diagnostic)
        elif np.ndim(value) == 0:
            np.testing.assert_array_equal(value, full[key], err_msg=key)
        else:
            np.testing.assert_array_equal(value, full[key][:count], err_msg=key)


def comparative_metrics(tracking, navigations, case):
    """Same-epoch errors plus initial bias and relative-motion diagnostics."""
    time = tracking['time']
    available = tracking['available_time']
    truth = np.column_stack((case['velocity_east'] * time, np.zeros(len(time))))
    post = available >= case['attack_start']
    published = {mode: np.isfinite(nav['position']).all(axis=1)
                 for mode, nav in navigations.items()}
    three_way_common = post & np.logical_and.reduce(list(published.values()))
    coherent_pair_common = (post & published['coherent_.1']
                            & published['coherent_0'])
    details = {}
    relative_errors = {}
    for mode, navigation in navigations.items():
        error = np.linalg.norm(navigation['position'][:, :2] - truth, axis=1)
        initial = navigation['initialization_time']
        initial_index = None if initial is None else int(np.flatnonzero(available == initial)[0])
        first = np.flatnonzero(published[mode])
        relative = np.full(len(time), np.nan)
        if len(first):
            anchor = first[0]
            relative[published[mode]] = np.linalg.norm(
                navigation['position'][published[mode], :2]
                - navigation['position'][anchor, :2]
                - (truth[published[mode]] - truth[anchor]), axis=1)
        relative_errors[mode] = relative
        details[mode] = dict(
            three_way_common=published_metrics(error, three_way_common),
            initialization_candidate_error_m=None if initial_index is None else
                float(np.linalg.norm(navigation['candidate_position'][initial_index, :2]
                                     - truth[initial_index])),
            relative_motion_error=published_metrics(relative, post & published[mode]))
        if mode != 'original':
            details[mode]['coherent_pair_common'] = published_metrics(
                error, coherent_pair_common)

    finite = np.isfinite(tracking['ranges'])
    quality = (finite & (tracking['lock'] >= .5) & (tracking['cn0'] >= 25.)
               & (tracking['gain'] >= .05))
    steady = available >= .5
    availability = dict(steady_opportunities=int(steady.sum()),
                        steady_finite_ranges_ge4=int((steady & (finite.sum(axis=1) >= 4)).sum()),
                        steady_quality_ge4=int((steady & (quality.sum(axis=1) >= 4)).sum()))
    return details, relative_errors, availability


def main():
    sources = source_hashes()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_snapshot(OUTPUT / 'source_snapshot.zip', sources)
    start_sample = round(PLAN['attack_start_s'] * FS)
    assert (start_sample - 1) / FS < PLAN['attack_start_s'] <= start_sample / FS
    plan = dict(**PLAN, source_sha256=sources,
                source_snapshot_sha256=digest(OUTPUT / 'source_snapshot.zip'),
                first_attack_sample=start_sample,
                first_attack_sample_time_s=start_sample / FS)
    (OUTPUT / 'plan.json').write_text(json.dumps(serializable(plan), indent=2,
                                                  allow_nan=False), encoding='utf-8')
    (OUTPUT / 'manifest.json').write_text(json.dumps(serializable(dict(
        scope=__doc__, plan_sha256=digest(OUTPUT / 'plan.json'),
        source_sha256=sources, source_snapshot_sha256=plan['source_snapshot_sha256'])),
        indent=2, allow_nan=False), encoding='utf-8')

    first_prefix = None
    reports = {}
    plots = {}
    for scene in SCENES:
        folder = OUTPUT / scene
        folder.mkdir(exist_ok=True)
        print(scene, 'RF generation start', flush=True)
        record, identity = generate_new_record(PLAN, scene)
        assert identity['case']['antennas'] == PLAN['generated_antennas']
        samples = record['samples']
        if first_prefix is None:
            first_prefix = samples[:, :start_sample].copy()
        else:
            np.testing.assert_array_equal(samples[:, :start_sample], first_prefix)
        input_folder = folder / 'input'
        input_folder.mkdir(exist_ok=True)
        np.save(input_folder / 'samples.npy', samples)
        np.savez_compressed(input_folder / 'observations.npz',
                            **{key: value for key, value in record.items()
                               if key != 'samples'})
        hashes = {name: digest(input_folder / name)
                  for name in ('samples.npy', 'observations.npz')}
        metadata = dict(**identity, samples_sha256=hashes['samples.npy'],
                        observations_sha256=hashes['observations.npz'],
                        pre_attack_exact_prefix_samples=start_sample)
        (input_folder / 'metadata.json').write_text(json.dumps(serializable(metadata),
            indent=2, allow_nan=False), encoding='utf-8')
        hashes['metadata.json'] = digest(input_folder / 'metadata.json')
        del record, samples, identity
        gc.collect()
        print(scene, 'IQ saved; frontend start', flush=True)

        with np.load(input_folder / 'observations.npz') as saved:
            replay = dict(saved)
        replay['samples'] = np.load(input_folder / 'samples.npy', mmap_mode='r')
        observations = frontend(replay, ANTENNAS, split_support=True,
                                receiver_pll=RECEIVER_PLL, evidence_trace=True)
        del replay
        gc.collect()
        trace_keys = [key for key in observations if key.startswith('trace_')]
        trace = {key: observations[key] for key in trace_keys}
        tracking = {key: value for key, value in observations.items()
                    if key not in trace_keys}
        np.savez_compressed(folder / 'tracking.npz', **tracking)
        np.savez_compressed(folder / 'trace.npz', **trace)
        print(scene, 'frontend saved', flush=True)

        history = analyze_trace(observations)
        np.savez_compressed(folder / 'history.npz', **history)
        groups = analyze_groups(trace)
        np.savez_compressed(folder / 'groups.npz', **groups)
        coherent = build_coherent_support(tracking, groups)
        supports = dict(original=tracking['support'],
                        **{'coherent_.1': coherent['support'],
                           'coherent_0': coherent['support']})
        support_arrays = {f'coherent_{key}': value for key, value in coherent.items()}
        support_arrays['20ms_noncoherent'] = history['ratio'][:, :, 0]
        navigations = {}
        scores = {}
        navigation_arrays = {}
        for mode, blend in zip(MODES, (.1, .1, 0.)):
            nav_input = {**tracking, 'support': supports[mode]}
            navigation = navigate(nav_input, diagnostics=True, code_blend=blend)
            check_prefix(nav_input, navigation, blend)
            navigations[mode] = navigation
            counts = quality_support_counts(tracking, supports[mode])
            support_arrays[f'{mode}_support_count'] = counts
            scores[mode] = score_mode(navigation, tracking, metadata['case'],
                                      counts, supports[mode])
        comparison, relative_errors, availability = comparative_metrics(
            tracking, navigations, metadata['case'])
        for mode in MODES:
            scores[mode].update(comparison[mode])
            navigation_arrays[f'{mode}_relative_motion_error_m'] = relative_errors[mode]
        save_navigation(navigation_arrays, navigations)
        np.savez_compressed(folder / 'support.npz', **support_arrays)
        np.savez_compressed(folder / 'navigation.npz', **navigation_arrays)
        files = ('tracking.npz', 'trace.npz', 'history.npz', 'groups.npz',
                 'support.npz', 'navigation.npz')
        reports[scene] = dict(input_sha256=hashes,
                              output_sha256={name: digest(folder / name) for name in files},
                              final_100ms_group_count=int(
                                  (coherent['group_count'][-1] == 20).sum()),
                              trace_windows=len(trace['trace_window_k']),
                              observation_availability=availability, modes=scores)
        plots[scene] = (tracking['time'].copy(), tracking['available_time'].copy(),
                        {mode: support_arrays[f'{mode}_support_count'].copy()
                         for mode in MODES},
                        {mode: nav['position'].copy() for mode, nav in navigations.items()},
                        metadata['case'])
        print(scene, 'done', flush=True)
        del observations, trace, tracking, history, groups, coherent
        del supports, support_arrays, navigations, navigation_arrays
        gc.collect()

    fig, axes = plt.subplots(2, len(SCENES), figsize=(14, 7), sharex='col',
                             constrained_layout=True)
    for column, scene in enumerate(SCENES):
        time, available, counts, positions, case = plots[scene]
        truth = np.column_stack((case['velocity_east'] * time,
                                 np.zeros(len(time))))
        for mode in MODES:
            axes[0, column].plot(available, counts[mode], label=mode)
            error = np.linalg.norm(positions[mode][:, :2] - truth, axis=1)
            axes[1, column].plot(available, error, label=mode)
        axes[0, column].set(title=scene, ylabel='Quality-gated support count')
        axes[1, column].set(xlabel='Output available time (s)',
                            ylabel='Published error (m)')
        for axis in axes[:, column]:
            axis.grid(alpha=.3)
    axes[0, 0].legend(fontsize=7)
    fig.savefig(OUTPUT / 'overview.png', dpi=150)
    plt.close(fig)
    assert source_hashes() == sources
    (OUTPUT / 'summary.json').write_text(json.dumps(serializable(dict(
        scope=__doc__, plan_sha256=digest(OUTPUT / 'plan.json'),
        manifest_sha256=digest(OUTPUT / 'manifest.json'),
        source_snapshot_sha256=plan['source_snapshot_sha256'],
        source_sha256=sources, figure_sha256=digest(OUTPUT / 'overview.png'),
        cases=reports)), indent=2, allow_nan=False), encoding='utf-8')
    return reports


if __name__ == '__main__':
    main()
