"""Stage 44: saved -5 dB IQ, startup amplitude off/on, fast receive replay.

The two old inputs are replayed with an otherwise fixed aided-5 ms receiver.
Truth and selected-carrier diagnostics are computed only after navigation.
"""
from core.provenance import current_path_for_historical_source, ROOT
import gc
import json
import sys
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy

from core.gps_simulation import receiver_reference
from core.provenance import source_hashes
from core.spoofing.recovery_geometry import FS
from core.serialization import serializable
from core.spoofing.single_source import frontend
from experiments.paths import SPOOFING_RESULTS as RESULTS
from experiments.spoofing.single_source_coherence_diagnosis import analyze_groups
from experiments.spoofing.single_source_coherent_navigation import (
    build_coherent_support, quality_support_counts,
)
from experiments.spoofing.single_source_early_truncation import (
    OUTPUT as STAGE43, compare_arrays, near_tail_energy_check, saved_arrays,
)
from experiments.spoofing.single_source_evidence_history import analyze_trace
from experiments.spoofing.single_source_long_coherent import source_snapshot
from experiments.spoofing.single_source_phase_exclusion import verify_source_snapshot
from experiments.spoofing.single_source_ramp_low_power import OUTPUT as STAGE39
from experiments.spoofing.single_source_ramp_transition import (
    compare_paths, path_segments, save_loops, saved_record, trigger_counts,
)
from experiments.spoofing.single_source_spoof_power import fake_east_at_observation
from experiments.spoofing.single_source_startup_amplitude import OUTPUT as STAGE40
from experiments.spoofing.single_source_startup_amplitude_rf import (
    audit_alignment, replay_navigation,
)
from experiments.spoofing.single_source_suppression_study import (
    ANTENNAS, OUTPUT as STUDY_OUTPUT, RECEIVER_PLL, digest,
)


OUTPUT = RESULTS/STUDY_OUTPUT/'low_power_amplitude_rf'
SCENES = ('weak', 'genuine_absent')
PATHS = ('off', 'on')
SCIENCE = ('tracking.npz', 'trace.npz', 'history.npz', 'groups.npz',
           'support.npz', 'navigation.npz')
FIT_NAME = {'weak': 'weak_minus5', 'genuine_absent': 'absent_minus5'}
GLOBAL_NAVIGATION_MAP = dict(
    observation_time=('receiver', 'observation_time'),
    available_time=('receiver', 'available_time'),
    phase_state_eligible=('gated', 'eligible'),
    latest_complete_block=('gated', 'last_completed_block_index'),
    recent_block_dimensions=('gated', 'recent_completed_dimensions'),
)


def navigation_field_map(candidate, receiver, gated):
    """Map each current field to its same-mode Stage 39 saved field."""
    mapping = {}
    for name in candidate:
        if name in GLOBAL_NAVIGATION_MAP:
            mapping[name] = GLOBAL_NAVIGATION_MAP[name]
        elif name.startswith('original_'):
            mapping[name] = ('receiver', name)
        elif name == 'short_support_count':
            mapping[name] = ('receiver', name)
        elif name.startswith('short_'):
            mapping[name] = ('gated', 'gated_state_'+name[len('short_'):])
        else:
            raise AssertionError(f'unmapped navigation field: {name}')
    references = dict(receiver=receiver, gated=gated)
    for source, field in mapping.values():
        assert field in references[source], f'missing {source} field: {field}'
    assert len(set(mapping.values())) == len(mapping)
    return mapping


def compare_mapped_navigation(candidate, receiver, gated):
    mapping = navigation_field_map(candidate, receiver, gated)
    references = dict(receiver=receiver, gated=gated)
    for name, (source, field) in mapping.items():
        compare_arrays({name: candidate[name]}, {name: references[source][field]})
    return mapping


def check_historical_navigation_layout(candidate, receiver, gated):
    """Pin coverage to the saved 39/41 schemas; truth-only fields follow."""
    mapping = navigation_field_map(candidate, receiver, gated)
    assert len(mapping) == 74
    assert {name for name in receiver if name.startswith('original_')} == (
        {name for name in candidate if name.startswith('original_')}
        | {'original_east_minus_true_m', 'original_east_minus_fake_m'})
    assert {name for name in gated if name.startswith('gated_state_')} == (
        {'gated_state_'+name[len('short_'):] for name in candidate
         if name.startswith('short_') and name != 'short_support_count'}
        | {'gated_state_horizontal_error_m'})
    return mapping


def check_truth_fields(candidate, receiver, gated, case):
    time = candidate['observation_time']
    truth_east = case['velocity_east']*time
    fake_east = fake_east_at_observation(case, time)
    original_east = candidate['original_position'][:, 0]
    np.testing.assert_array_equal(original_east-truth_east,
                                  receiver['original_east_minus_true_m'])
    np.testing.assert_array_equal(original_east-truth_east-fake_east,
                                  receiver['original_east_minus_fake_m'])
    truth = np.column_stack((truth_east, np.zeros(len(time))))
    horizontal = np.linalg.norm(candidate['short_position'][:, :2]-truth, axis=1)
    np.testing.assert_array_equal(horizontal,
                                  gated['gated_state_horizontal_error_m'])


def preflight():
    """Bind saved IQ, Stage 39/40 evidence, and the just-frozen Stage 43 code."""
    summaries = {
        'stage39': (STAGE39, json.loads((STAGE39/'summary.json').read_text(
            encoding='utf-8'))),
        'stage40': (STAGE40, json.loads((STAGE40/'summary.json').read_text(
            encoding='utf-8'))),
        'stage43': (STAGE43, json.loads((STAGE43/'summary.json').read_text(
            encoding='utf-8'))),
    }
    current = source_hashes()
    for folder, summary in summaries.values():
        assert summary['source_snapshot_sha256'] == digest(folder/'source_snapshot.zip')
        verify_source_snapshot(folder/'source_snapshot.zip', summary['source_sha256'])
        for key, file in (('plan_sha256', 'plan.json'),
                          ('manifest_sha256', 'manifest.json')):
            assert summary[key] == digest(folder/file)
    for name, sha in summaries['stage43'][1]['source_sha256'].items():
        assert digest(current_path_for_historical_source(ROOT, name)) == sha, name
    stage39 = summaries['stage39'][1]
    stage40 = summaries['stage40'][1]
    for scene in SCENES:
        folder = STAGE39/scene
        fitted = STAGE40/FIT_NAME[scene]
        old = stage39['cases'][scene]
        fit = stage40['cases'][FIT_NAME[scene]]
        assert fit['input_sha256'] == old['input_sha256']
        for file, sha in old['input_sha256'].items():
            assert digest(folder/'input'/file) == sha
        for file, sha in old['receiver']['output_sha256'].items():
            assert digest(folder/'aided_five_ms'/file) == sha
        assert digest(folder/'navigation_replay.npz') == old['navigation'][
            'result_sha256']
        for key, file in (('fit_sha256', 'fit.npz'),
                          ('startup_sha256', 'startup.npz')):
            assert digest(fitted/file) == fit[key]
    return summaries, current


def selected_carrier_increment_error(tracking, navigation, case):
    """Posthoc 10 ms carrier increment error on consecutive selected arc pairs.

    Selection is the navigation valid mask, not evidence that a phase estimate
    actually used that PRN after leave-one-out or anchor qualification.
    Per-PRN signed sums span only selected consecutive pairs within one arc;
    they are neither complete-arc endpoint drift nor navigation contribution.
    """
    reference, rotation = receiver_reference()
    east = case['velocity_east']*tracking['time']
    enu = np.column_stack((east, np.zeros((len(east), 2))))
    receiver = reference + enu@rotation
    geometric = np.linalg.norm(tracking['satellites']-receiver[:, None, :], axis=2)
    carrier = tracking['carrier']
    arc = tracking['arc']
    result = {}
    for mode in ('original', 'short'):
        selected = navigation[f'{mode}_selected']
        pairs = (selected[1:] & selected[:-1] & (arc[1:] == arc[:-1])
                 & (arc[1:] >= 0) & np.isfinite(carrier[1:])
                 & np.isfinite(carrier[:-1]))
        innovation = np.diff(carrier, axis=0)-np.diff(geometric, axis=0)
        magnitude = np.abs(innovation[pairs])
        result[mode] = dict(pairs=int(pairs.sum()),
            meaning=('Selected consecutive same-arc carrier increments; '
                     'per-PRN sums are not complete-arc endpoint drift or '
                     'verified navigation phase contribution'),
            observed_time_range_s=[float(tracking['time'][0]),
                                   float(tracking['time'][-1])],
            median_abs_m=float(np.median(magnitude)),
            p95_abs_m=float(np.percentile(magnitude, 95)),
            max_abs_m=float(np.max(magnitude)),
            per_prn_pair_count=pairs.sum(axis=0).astype(int).tolist(),
            per_prn_signed_increment_sum_m=np.sum(
                np.where(pairs, innovation, 0.), axis=0).tolist())
    return result


def run_path(record, case, scene, path, fit, startup, folder):
    folder.mkdir(parents=True, exist_ok=True)
    scratch = folder/'beam_stream.dat'
    loops = []
    started = perf_counter()
    observations = frontend(record, ANTENNAS, split_support=True,
        receiver_pll=RECEIVER_PLL, evidence_trace=True,
        temporal_beam_trigger_enabled=False,
        beam_template_carrier_aiding=True, beam_coherent_ms=5,
        beam_startup_amplitude=path == 'on', stream_memmap_path=scratch,
        loop_history=loops, fast_truncation=True)
    elapsed_s = perf_counter()-started
    assert scratch.stat().st_size == len(record['codes'])*record['samples'].shape[1]*16
    scratch.unlink()
    tracking = {name: value for name, value in observations.items()
                if not name.startswith('trace_')}
    trace = {name: value for name, value in observations.items()
             if name.startswith('trace_')}
    np.savez_compressed(folder/'tracking.npz', **tracking)
    np.savez_compressed(folder/'trace.npz', **trace)
    loop_count = save_loops(folder, loops)
    audit_alignment(tracking, trace, fit, startup,
                    record['nominal'][ANTENNAS], mode=int(path == 'on'))

    history = analyze_trace(observations)
    groups = analyze_groups(trace)
    coherent = build_coherent_support(tracking, groups)
    support = {f'coherent_{name}': value for name, value in coherent.items()}
    for file, arrays in (('history.npz', history), ('groups.npz', groups),
                         ('support.npz', support)):
        np.savez_compressed(folder/file, **arrays)
    near_tail = near_tail_energy_check(observations, history)
    scores, positions = replay_navigation(tracking, trace, coherent, case, folder)
    navigation = saved_arrays(folder/'navigation.npz')

    if path == 'off':
        old_folder = STAGE39/scene/'aided_five_ms'
        for file, arrays in (('tracking.npz', tracking), ('trace.npz', trace),
                             ('history.npz', history), ('groups.npz', groups),
                             ('support.npz', support)):
            compare_arrays(arrays, saved_arrays(old_folder/file))
        old_navigation = saved_arrays(old_folder/'navigation.npz')
        old_gated = saved_arrays(STAGE39/scene/'navigation_replay.npz')
        check_historical_navigation_layout(navigation, old_navigation, old_gated)
        compare_mapped_navigation(navigation, old_navigation, old_gated)
        check_truth_fields(navigation, old_navigation, old_gated, case)

    segments = path_segments(tracking, folder, case)
    increment_error = selected_carrier_increment_error(tracking, navigation, case)
    view = dict(available=tracking['available_time'].copy(),
                observation=tracking['time'].copy(), positions=positions,
                support={mode: quality_support_counts(tracking,
                    tracking['support'] if mode == 'original'
                    else coherent['support']) for mode in ('original', 'short')})
    result = dict(frontend_walltime_s=elapsed_s, loop_history=loop_count,
        near_tail_recomputed_rows=near_tail, navigation=scores,
        segments=segments,
        trigger=trigger_counts(trace['trace_raw_trigger'], case['attack_start']),
        selected_carrier_increment_error=increment_error,
        output_sha256={file: digest(folder/file)
                       for file in (*SCIENCE, 'loops.npz')})
    del observations, tracking, trace, history, groups, coherent, loops
    gc.collect()
    return result, view


def plot_results(views):
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for row, scene in enumerate(SCENES):
        for path in PATHS:
            view = views[scene][path]
            truth = np.column_stack((2*view['observation'],
                                     np.zeros(len(view['observation']))))
            for mode in ('original', 'short'):
                label = f'{path} {mode}'
                error = np.linalg.norm(view['positions'][mode][:, :2]-truth, axis=1)
                axes[row, 0].plot(view['available'], view['support'][mode],
                                  linewidth=.7, label=label)
                axes[row, 1].plot(view['available'], error, linewidth=.7,
                                  marker='.', markersize=1., label=label)
        axes[row, 0].set(title=scene, ylabel='Quality-supported stars')
        axes[row, 1].set(title=scene, ylabel='Published horizontal error (m)')
        for axis in axes[row]:
            axis.set(xlabel='Available time (s)')
            axis.set_xlim(0, views[scene]['off']['available'][-1])
            axis.grid(alpha=.3)
    axes[0, 0].legend(fontsize=6)
    axes[0, 1].legend(fontsize=6)
    figure.savefig(OUTPUT/'overview.png', dpi=150)
    plt.close(figure)


def main():
    summaries, sources = preflight()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_snapshot(OUTPUT/'source_snapshot.zip', sources)
    plan = dict(scope=__doc__, scenes=SCENES, paths=PATHS,
        input='saved Stage 39 -5 dB weak/genuine-absent IQ',
        runtime=dict(python=sys.version.split()[0], numpy=np.__version__,
                     scipy=scipy.__version__),
        fast_truncation=True, receiver_pll=RECEIVER_PLL,
        beam_template_carrier_aiding=True, beam_coherent_ms=5,
        temporal_beam_trigger_enabled=False,
        beam_startup_amplitude_paths=dict(off=False, on=True),
        navigation_modes=dict(original='original support, code blend .1',
                              short='coherent support, gated continuity, code blend 0'),
        source_sha256=sources)
    (OUTPUT/'plan.json').write_text(json.dumps(serializable(plan), indent=2,
        allow_nan=False), encoding='utf-8')
    manifest = dict(source_snapshot_sha256=digest(OUTPUT/'source_snapshot.zip'),
        plan_sha256=digest(OUTPUT/'plan.json'),
        reference_summary_sha256={name: digest(folder/'summary.json')
                                  for name, (folder, _) in summaries.items()})
    (OUTPUT/'manifest.json').write_text(json.dumps(serializable(manifest),
        indent=2, allow_nan=False), encoding='utf-8')
    cases = {}; views = {}
    for scene in SCENES:
        record, identity = saved_record(STAGE39/scene)
        fit_name = FIT_NAME[scene]
        fit = saved_arrays(STAGE40/fit_name/'fit.npz')
        startup = saved_arrays(STAGE40/fit_name/'startup.npz')
        paths = {}; positions = {}; views[scene] = {}
        for path in PATHS:
            paths[path], views[scene][path] = run_path(record, identity['case'],
                scene, path, fit, startup, OUTPUT/scene/path)
            positions[path] = views[scene][path]['positions']
            print(scene, path, 'saved and audited', flush=True)
        np.testing.assert_array_equal(views[scene]['off']['observation'],
                                      views[scene]['on']['observation'])
        np.testing.assert_array_equal(views[scene]['off']['available'],
                                      views[scene]['on']['available'])
        comparison = compare_paths(dict(time=views[scene]['off']['observation'],
            available_time=views[scene]['off']['available']), positions,
            identity['case']['velocity_east'])
        cases[scene] = dict(
            input_sha256=summaries['stage39'][1]['cases'][scene]['input_sha256'],
            stage40_fit_sha256=summaries['stage40'][1]['cases'][fit_name]['fit_sha256'],
            stage40_startup_sha256=summaries['stage40'][1]['cases'][fit_name][
                'startup_sha256'], paths=paths, matched=comparison)
        del record, fit, startup, positions
        gc.collect()
    plot_results(views)
    assert source_hashes() == sources
    summary = dict(scope=__doc__, source_sha256=sources,
        source_snapshot_sha256=digest(OUTPUT/'source_snapshot.zip'),
        plan_sha256=digest(OUTPUT/'plan.json'),
        manifest_sha256=digest(OUTPUT/'manifest.json'),
        figure_sha256=digest(OUTPUT/'overview.png'), cases=cases)
    (OUTPUT/'summary.json').write_text(json.dumps(serializable(summary),
        indent=2, allow_nan=False), encoding='utf-8')
    return summary


if __name__ == '__main__':
    main()
