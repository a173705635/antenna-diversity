"""Resume Stage 48 after original navigation failed on saved weak/yaw_minus2 RF.

The frozen 99-source plan and saved receiver files are immutable. Original and
short navigation run independently; only PositioningFailure in original is a
recorded whole-mode failure. Run this entry point explicitly to finish RF.
"""
import gc
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy

from core.spoofing.mechanism_smoothing import smooth_history
from core.positioning import PositioningFailure
from core.provenance import source_hashes
from core.spoofing.recovery_geometry import FS, receiver_reference, sky
from core.serialization import serializable
from core.spoofing.single_source import frontend, navigate
from experiments.spoofing.single_source_coherence_diagnosis import analyze_groups
from experiments.spoofing.single_source_coherent_navigation import (
    build_coherent_support, quality_support_counts,
)
from experiments.spoofing.single_source_evidence_history import analyze_trace
from experiments.spoofing.single_source_long_coherent import navigation_prefix
from experiments.spoofing.single_source_phase_exclusion import verify_source_snapshot
from experiments.spoofing.single_source_phase_state_gate import projection_eligibility
from experiments.spoofing.single_source_projection_long import long_plan
from experiments.spoofing.single_source_projection_threshold import audit_blocks
from experiments.spoofing.single_source_ramp_transition import save_loops
from experiments.spoofing.single_source_startup_amplitude_rf import (
    check_prefix, score_navigation,
)
from experiments.spoofing.single_source_suppression_study import (
    ANTENNAS, RECEIVER_PLL, digest,
)
from experiments.spoofing.single_source_template_coherence import STARTUP, WIDTH
from experiments.spoofing.single_source_temporal_trigger import navigation_input
from experiments.spoofing import single_source_nominal_yaw as study
from tools.spoofing.resume_projection_long import check_saved_arrays, loop_count, recover_path


OUTPUT = study.OUTPUT
FRONTEND_FILES = ('tracking.npz', 'trace.npz', 'fit.npz', 'loops.npz',
                  'history.npz', 'groups.npz', 'support.npz')


def write_json(path, value):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(serializable(value), indent=2,
                                    allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def hashes(folder, names):
    return {name: digest(folder/name) for name in names}


def runtime():
    return dict(python=sys.version.split()[0], numpy=np.__version__,
                scipy=scipy.__version__, platform=platform.platform())


def preflight():
    baseline, sources = study.preflight()
    plan_path = OUTPUT/'plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    assert len(sources) == 99
    assert plan['source_sha256'] == sources == source_hashes()
    assert plan['seed'] == study.SEED
    assert tuple(plan['scenes']) == study.SCENES
    assert tuple(plan['yaw_deg']) == study.YAWS_DEG
    assert plan['gate'] == 3.
    assert plan['saved_path_recovery_sha256'] == digest(Path(
        sys.modules[recover_path.__module__].__file__))
    verify_source_snapshot(OUTPUT/'source_snapshot.zip', sources)
    return baseline, sources, hashes(OUTPUT, ('plan.json', 'source_snapshot.zip'))


def snapshot_recovery_source():
    source = Path(__file__)
    frozen = OUTPUT/'resume_nominal_yaw.py'
    current = digest(source)
    if frozen.exists():
        assert digest(frozen) == current
    else:
        temporary = frozen.with_name(frozen.name+'.tmp')
        temporary.write_bytes(source.read_bytes())
        temporary.replace(frozen)
    return current


def original_options():
    return dict(diagnostics=True, code_blend=.1,
                phase_leave_one_out=False, phase_state_continuity=False)


def first_failure_index(inputs):
    """Find the shortest causal original prefix that raises PositioningFailure."""
    low, high = 1, len(inputs['time'])
    while low < high:
        middle = (low+high)//2
        try:
            navigate(navigation_prefix(inputs, middle), **original_options())
        except PositioningFailure:
            high = middle
        else:
            low = middle+1
    return low-1


def geocentric_iterations(satellites, ranges):
    """Recalculate the solver's zero-state design condition for diagnostics."""
    state = np.zeros(4)
    conditions = []
    for iteration in range(1, 21):
        delta = state[:3]-satellites
        distance = np.linalg.norm(delta, axis=1)
        design = np.column_stack((delta/distance[:, None], np.ones(len(ranges))))
        residual = ranges-distance-state[3]
        update, _, rank, singular = np.linalg.lstsq(design, residual, rcond=None)
        condition = float(singular[0]/singular[-1])
        conditions.append(dict(iteration=iteration, rank=int(rank),
                               condition=condition))
        if rank < 4 or condition > 1e8:
            break
        state += update
        if np.linalg.norm(update) < 1e-5:
            break
    return conditions


def failure_evidence(tracking, error):
    """Locate the first failure and record its selected geometry, without a fix."""
    inputs = navigation_input(tracking, tracking['support'])
    index = first_failure_index(inputs)
    finite = np.isfinite(inputs['ranges'])
    valid = (finite & (inputs['lock'] >= .5) & (inputs['cn0'] >= 25)
             & (inputs['gain'] >= .05) & (inputs['support'] >= 3.))
    ranges, _ = smooth_history(inputs, valid)
    valid &= np.isfinite(ranges)
    satellites = inputs['satellites'][index, valid[index]]
    selected_ranges = ranges[index, valid[index]]
    reference, _ = receiver_reference()
    direction = satellites-reference
    design = np.column_stack((-direction/np.linalg.norm(direction, axis=1)[:, None],
                              np.ones(len(direction))))
    return dict(index=int(index),
                observation_time_s=float(inputs['time'][index]),
                available_time_s=float(inputs['available_time'][index]),
                selected_prns=(np.flatnonzero(valid[index])+1).tolist(),
                reference_rank=int(np.linalg.matrix_rank(design)),
                reference_condition=float(np.linalg.cond(design)),
                geocentric_iterations=geocentric_iterations(
                    satellites, selected_ranges),
                exception_type=type(error).__name__, exception_message=str(error))


def replay_modes(tracking, trace, coherent, case):
    """Replay unchanged original/short algorithms in separate mode states."""
    eligible, latest, recent = projection_eligibility(tracking['available_time'], trace)
    arrays = dict(observation_time=tracking['time'],
                  available_time=tracking['available_time'],
                  phase_state_eligible=eligible,
                  latest_complete_block=latest,
                  recent_block_dimensions=recent)
    modes = {}; positions = {}
    for mode in ('original', 'short'):
        support = tracking['support'] if mode == 'original' else coherent['support']
        options = original_options() if mode == 'original' else dict(
            diagnostics=True, code_blend=0., phase_leave_one_out=True,
            phase_state_continuity=True, phase_state_eligible=eligible)
        try:
            navigation = navigate(navigation_input(tracking, support), **options)
        except PositioningFailure as error:
            if mode != 'original':
                raise
            modes[mode] = dict(status='solver_failed',
                               failure=failure_evidence(tracking, error))
            positions[mode] = None
            continue
        check_prefix(tracking, support, eligible, navigation, mode)
        count = quality_support_counts(tracking, support)
        arrays[f'{mode}_support_count'] = count
        for key, value in navigation.items():
            if key == 'diagnostics':
                for diagnostic, data in value.items():
                    arrays[f'{mode}_diagnostic_{diagnostic}'] = data
            else:
                arrays[f'{mode}_{key}'] = np.nan if value is None else value
        score = dict(**score_navigation(tracking, navigation, case['velocity_east']),
                     quality_support_ge4=int((count >= 4).sum()),
                     quality_support_ge6=int((count >= 6).sum()))
        modes[mode] = dict(status='complete', score=score)
        positions[mode] = navigation['position'].copy()
    return arrays, modes, positions


def verify_frontend(folder):
    """Check saved science arrays without rewriting or claiming a stream audit."""
    tracking = study.read_npz(folder/'tracking.npz')
    trace = study.read_npz(folder/'trace.npz')
    fit = study.read_npz(folder/'fit.npz')
    coherent = check_saved_arrays(folder, tracking, trace)
    np.testing.assert_allclose(trace['trace_raw_trigger'], fit['raw'],
                               rtol=0, atol=1e-9, equal_nan=True)
    np.testing.assert_allclose(trace['trace_projectors'], fit['projector'],
                               rtol=0, atol=1e-9)
    np.testing.assert_array_equal(trace['trace_dimensions'], fit['dimension'])
    index = (tracking['time']/.02).astype(int)
    np.testing.assert_allclose(tracking['gain'], fit['gain'][index],
                               rtol=0, atol=1e-9)
    assert not (folder/'beam_stream.dat').exists()
    return tracking, trace, fit, coherent


def run_frontend(record, folder):
    """Run the frozen receiver and independent block audit, stopping before nav."""
    folder.mkdir(parents=True)
    scratch = folder/'beam_stream.dat'
    loops = []
    started = perf_counter()
    observations = frontend(record, ANTENNAS, split_support=True,
        receiver_pll=RECEIVER_PLL, evidence_trace=True,
        temporal_beam_trigger_enabled=False,
        beam_template_carrier_aiding=True, beam_coherent_ms=5,
        beam_startup_amplitude=True, fast_truncation=True,
        beam_projection_threshold=3., stream_memmap_path=scratch,
        loop_history=loops)
    elapsed = perf_counter()-started
    tracking = {name: value for name, value in observations.items()
                if not name.startswith('trace_')}
    trace = {name: value for name, value in observations.items()
             if name.startswith('trace_')}
    np.savez_compressed(folder/'tracking.npz', **tracking)
    np.savez_compressed(folder/'trace.npz', **trace)
    counts = save_loops(folder, loops)
    fit = audit_blocks(record, tracking, trace, scratch, 3.)
    np.savez_compressed(folder/'fit.npz', **fit)
    scratch.unlink()
    history = analyze_trace(observations)
    groups = analyze_groups(trace)
    coherent = build_coherent_support(tracking, groups)
    for name, arrays in (('history.npz', history), ('groups.npz', groups),
                         ('support.npz', {f'coherent_{key}': value
                                         for key, value in coherent.items()})):
        np.savez_compressed(folder/name, **arrays)
    del observations, history, groups, loops
    gc.collect()
    return dict(frontend_walltime_s=elapsed, loop_history=counts), (
        tracking, trace, fit, coherent)


def projection_blocks(fit, case):
    attack = round(case['attack_start']*FS/WIDTH)
    raw = fit['raw']
    return dict(after_startup=dict(triggered=int((raw[STARTUP//WIDTH:] >= 3.).sum()),
                                   opportunities=len(raw)-STARTUP//WIDTH),
                after_attack=dict(triggered=int((raw[attack:] >= 3.).sum()),
                                  opportunities=len(raw)-attack))


def saved_view(folder, modes):
    tracking = study.read_npz(folder/'tracking.npz')
    filename = ('navigation.npz' if modes['original']['status'] == 'complete'
                else 'navigation_short.npz')
    navigation = study.read_npz(folder/filename)
    return dict(observation=tracking['time'], available=tracking['available_time'],
                position={mode: (navigation[f'{mode}_position']
                    if modes[mode]['status'] == 'complete' else None)
                    for mode in ('original', 'short')},
                navigation={mode: (modes[mode]['score']
                    if modes[mode]['status'] == 'complete' else None)
                    for mode in ('original', 'short')})


def baseline_view(folder, baseline, scene):
    """Tie the zero-yaw receipt and files to the audited Stage 47 summary."""
    old_path = folder/'gate3'
    receipt = json.loads((old_path/'path_receipt.json').read_text(encoding='utf-8'))
    expected = baseline['cases'][str(study.SEED)][scene]['paths']['gate3'][
        'output_sha256']
    assert receipt['output_sha256'] == expected
    assert hashes(old_path, study.SCIENCE) == expected
    return study.path_view(old_path, receipt), receipt


def process_path(folder, record, case, yaw_deg, input_sha, sources, recovery_sha):
    """Reuse a completed path or finish a seven-file path without touching it."""
    original_receipt = folder/'path_receipt.json'
    recovery_receipt = folder/'recovery_receipt.json'
    if original_receipt.exists():
        receipt = json.loads(original_receipt.read_text(encoding='utf-8'))
        assert receipt['input_sha256'] == input_sha
        assert receipt['source_sha256'] == sources
        assert receipt['yaw_deg'] == yaw_deg
        result, view = recover_path(folder, record, case, 3., receipt)
        assert digest(folder/'physical_diagnostics.npz') == receipt['diagnostics_sha256']
        modes = {mode: dict(status='complete', score=result['navigation'][mode])
                 for mode in ('original', 'short')}
        return dict(status='complete', provenance='frozen_receipt',
                    receipt_sha256=digest(original_receipt),
                    output_sha256=dict(**result['output_sha256'],
                        **{'physical_diagnostics.npz': receipt['diagnostics_sha256']}),
                    navigation_modes=modes), view
    if recovery_receipt.exists():
        receipt = json.loads(recovery_receipt.read_text(encoding='utf-8'))
        assert receipt['provenance'] == 'recovery_receipt'
        assert receipt['input_sha256'] == input_sha
        assert receipt['frozen_source_sha256'] == sources
        assert receipt['recovery_source_sha256'] == recovery_sha
        assert receipt['yaw_deg'] == yaw_deg
        assert hashes(folder, FRONTEND_FILES) == receipt['frontend_sha256']
        assert digest(folder/receipt['navigation_file']) == receipt['navigation_sha256']
        assert digest(folder/'physical_diagnostics.npz') == receipt['diagnostics_sha256']
        return receipt, saved_view(folder, receipt['navigation_modes'])

    if folder.exists():
        assert all((folder/name).exists() for name in FRONTEND_FILES), (
            'Incomplete receiver path retained for manual audit', folder)
        tracking, trace, fit, coherent = verify_frontend(folder)
        frontend_status = dict(frontend_walltime_s=None,
                               frontend_walltime_status='not_recorded_after_interruption',
                               loop_history=loop_count(folder),
                               saved_beam_stream='deleted; saved arrays checked in memory')
    else:
        antennas = np.array(list(study.mate70_phase_centers().values()))
        received, physical = study.receiver_record(
            record, sky(study.SEED, 24), antennas, yaw_deg)
        assert physical is record['nominal']
        frontend_status, (tracking, trace, fit, coherent) = run_frontend(received, folder)
        del received

    frozen_frontend = hashes(folder, FRONTEND_FILES)
    arrays, modes, _ = replay_modes(tracking, trace, coherent, case)
    navigation_file = ('navigation.npz' if modes['original']['status'] == 'complete'
                       else 'navigation_short.npz')
    navigation_path = folder/navigation_file
    if navigation_path.exists():
        saved = study.read_npz(navigation_path)
        assert saved.keys() == arrays.keys()
        for key, value in arrays.items():
            np.testing.assert_array_equal(saved[key], value)
    else:
        np.savez_compressed(navigation_path, **arrays)
    diagnostics_path = folder/'physical_diagnostics.npz'
    physical = study.physical_diagnostics(record, study.SEED, folder)
    if diagnostics_path.exists():
        saved = study.read_npz(diagnostics_path)
        assert saved.keys() == physical.keys()
        for key, value in physical.items():
            np.testing.assert_array_equal(saved[key], value)
    else:
        np.savez_compressed(diagnostics_path, **physical)
    assert hashes(folder, FRONTEND_FILES) == frozen_frontend
    receipt = dict(status='navigation_incomplete' if modes['original']['status'] !=
                   'complete' else 'complete', provenance='recovery_receipt',
                   yaw_deg=yaw_deg,
                   input_sha256=input_sha, frozen_source_sha256=sources,
                   recovery_source_sha256=recovery_sha, runtime=runtime(),
                   frontend=frontend_status, frontend_sha256=frozen_frontend,
                   projection_blocks=projection_blocks(fit, case),
                   navigation_modes=modes, navigation_file=navigation_file,
                   navigation_sha256=digest(navigation_path),
                   diagnostics_sha256=digest(diagnostics_path))
    write_json(recovery_receipt, receipt)
    return receipt, saved_view(folder, modes)


def plot_scene(scene, views, folder):
    """Plot completed paths only; state failed original mode as text."""
    figure, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                constrained_layout=True)
    for yaw in study.YAWS_DEG:
        view = views[yaw]
        time = view['observation']
        truth = np.column_stack((2.*time, np.zeros(len(time))))
        for mode, style in (('short', '-'), ('original', '--')):
            position = view['position'][mode]
            if position is None:
                axes[0].text(.02, .95-.08*list(study.YAWS_DEG).index(yaw),
                             f'{yaw:+g}° original: solver failed',
                             transform=axes[0].transAxes, va='top')
                continue
            error = np.linalg.norm(position[:, :2]-truth, axis=1)
            axes[0].plot(time, error, style, linewidth=.8,
                         label=f'{yaw:+g}° {mode}')
            axes[1].plot(time, np.isfinite(position).all(axis=1).astype(int),
                         style, linewidth=.8)
    axes[0].set(ylabel='Horizontal error (m)', title=scene)
    axes[1].set(xlabel='Observation time (s)', ylabel='Published')
    for axis in axes:
        axis.set_xlim(0., 20.)
        axis.grid(alpha=.25)
    axes[0].legend(fontsize=8, ncol=3)
    figure.savefig(folder/'recovery_overview.png', dpi=150)
    plt.close(figure)


def score_scene(scene, views, case):
    baseline = views[0.]
    time = baseline['observation']
    assert int((time >= .3).sum()) == 1969
    assert int((time >= .5).sum()) == 1949
    scores = {}
    for yaw in study.YAWS_DEG[1:]:
        candidate = views[yaw]
        np.testing.assert_array_equal(time, candidate['observation'])
        np.testing.assert_array_equal(baseline['available'], candidate['available'])
        modes = {}
        for mode in ('short', 'original'):
            if candidate['position'][mode] is None:
                modes[mode] = dict(status='solver_failed', score=None)
                continue
            score = study.score_pair(scene, time, baseline['position'][mode],
                candidate['position'][mode], case['velocity_east'],
                baseline['navigation'][mode]['max_relative_motion_error_m'],
                candidate['navigation'][mode]['max_relative_motion_error_m'])
            score['baseline_initial_anchor_error_m'] = baseline['navigation'][mode][
                'initial_horizontal_error_m']
            score['candidate_initial_anchor_error_m'] = candidate['navigation'][mode][
                'initial_horizontal_error_m']
            modes[mode] = dict(status='complete', score=score)
        scores[f'{yaw:+g}'] = modes
    return scores


def main():
    """Finish the saved failed path, then three pending RF paths, serially."""
    baseline, sources, frozen_hashes = preflight()
    recovery_sha = snapshot_recovery_source()
    first_prefix = first_geometry = None
    cases = {}; inputs = {}; outputs = {}; figures = {}
    for scene in study.SCENES:
        old_folder = study.STAGE47/str(study.SEED)/scene
        record, identity, input_sha, first_prefix, first_geometry = study.load_input(
            old_folder, long_plan(study.SEED, scene), first_prefix, first_geometry)
        assert input_sha == baseline['cases'][str(study.SEED)][scene]['input_sha256']
        inputs[scene] = input_sha
        zero_view, old_receipt = baseline_view(old_folder, baseline, scene)
        views = {0.: zero_view}
        paths = {'yaw0': dict(status='reused_stage47_gate3',
                              output_sha256=old_receipt['output_sha256'])}
        for yaw in study.YAWS_DEG[1:]:
            name = f'yaw_{"plus" if yaw > 0 else "minus"}{abs(yaw):g}'
            folder = OUTPUT/scene/name
            receipt, views[yaw] = process_path(
                folder, record, identity['case'], yaw, input_sha, sources,
                recovery_sha)
            paths[name] = receipt
            outputs[f'{scene}/{name}'] = (receipt['output_sha256'] if
                receipt.get('provenance') == 'frozen_receipt' else
                dict(**receipt['frontend_sha256'],
                     **{receipt['navigation_file']: receipt['navigation_sha256'],
                        'physical_diagnostics.npz': receipt['diagnostics_sha256']}))
            print(scene, name, receipt['status'], flush=True)
            gc.collect()
        scores = score_scene(scene, views, identity['case'])
        plot_scene(scene, views, OUTPUT/scene)
        figures[scene] = digest(OUTPUT/scene/'recovery_overview.png')
        cases[scene] = dict(input_sha256=input_sha, paths=paths, scores=scores)
        del record, views
        gc.collect()
    assert source_hashes() == sources
    assert hashes(OUTPUT, ('plan.json', 'source_snapshot.zip')) == frozen_hashes
    assert digest(OUTPUT/'resume_nominal_yaw.py') == recovery_sha
    manifest = dict(frozen_plan_sha256=frozen_hashes['plan.json'],
                    frozen_source_snapshot_sha256=frozen_hashes['source_snapshot.zip'],
                    stage47_summary_sha256=digest(study.STAGE47/'summary.json'),
                    recovery_source_sha256=recovery_sha,
                    input_sha256=inputs, output_sha256=outputs,
                    figure_sha256=figures,
                    receipt_sha256={f'{scene}/{name}': digest(
                        OUTPUT/scene/name/('path_receipt.json' if
                            path['provenance'] == 'frozen_receipt' else
                            'recovery_receipt.json'))
                        for scene in study.SCENES
                        for name, path in cases[scene]['paths'].items()
                        if name != 'yaw0'})
    write_json(OUTPUT/'recovery_manifest.json', manifest)
    summary = dict(scope=__doc__, frozen_source_sha256=sources,
                   recovery_source_sha256=recovery_sha,
                   manifest_sha256=digest(OUTPUT/'recovery_manifest.json'),
                   completed_rf_paths=6,
                   complete_navigation_paths=sum(
                       paths['navigation_modes']['original']['status'] == 'complete'
                       for scene in study.SCENES
                       for key, paths in cases[scene]['paths'].items()
                       if key != 'yaw0'),
                   short_screen_pass=all(cases[scene]['scores'][f'{yaw:+g}'][
                       'short']['score']['pass'] for scene in study.SCENES
                       for yaw in study.YAWS_DEG[1:]),
                   cases=cases)
    write_json(OUTPUT/'recovery_summary.json', summary)
    return summary


if __name__ == '__main__':
    main()
