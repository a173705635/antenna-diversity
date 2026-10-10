"""Fixed 2 s gate-3/gate-2 screen on four paired single-source RF scenes.

The same generated IQ feeds both gates. Truth enters only scoring after the
receiver and independent IQ/beam audit have finished.
"""
import gc
import json
import sys
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy

from core.provenance import source_hashes
from core.recovery_geometry import FS
from core.serialization import serializable
from core.single_source import frontend
from core.single_source_scene import local_signal
from experiments.paths import RESULTS
from experiments.single_source_coherence_diagnosis import analyze_groups
from experiments.single_source_coherent_navigation import (
    build_coherent_support, quality_support_counts,
)
from experiments.single_source_evidence_history import analyze_trace
from experiments.single_source_history_study import generate_new_record
from experiments.single_source_long_coherent import source_snapshot
from experiments.single_source_ramp_takeover_long import save_new_input
from experiments.single_source_ramp_transition import save_loops
from experiments.single_source_startup_amplitude import (
    estimate_amplitude, startup_prompts,
)
from experiments.single_source_startup_amplitude_rf import replay_navigation
from experiments.single_source_suppression_study import (
    ANTENNAS, OUTPUT as STUDY_OUTPUT, RECEIVER_PLL, digest,
)
from experiments.single_source_template_coherence import (
    STARTUP, WIDTH, aided_template, extract_startup, group_five_ms,
)


SEED = 2026100401
GATES = (3., 2.)
SCENES = ('h0_constant', 'h0_drop', 'weak', 'genuine_absent')
OUTPUT = RESULTS/STUDY_OUTPUT/'projection_threshold'
SCIENCE = ('tracking.npz', 'trace.npz', 'fit.npz', 'loops.npz', 'history.npz',
           'groups.npz', 'support.npz', 'navigation.npz')


def scene_plan(scene):
    return dict(seed=SEED, duration_s=2., satellites=24,
                startup_cn0_db_hz=45.,
                post_0p3s_cn0_db_hz=45. if scene == 'h0_constant' else 30.,
                attack_start_s=.3, spoof_db=-5., angle_deg=5.,
                fake_east_m=0., fake_rate_mps=3.,
                generated_antennas=4, processed_antennas=(0, 1, 2))


def independent_fit(prompts, model, prns, noise, threshold):
    """Refit one 20 ms grouped-prompt block without receiver beam helpers."""
    antennas = len(model)
    desired = np.repeat(model[:, prns], prompts.shape[1]//len(prns), axis=1)
    nuisance = np.linalg.svd(prompts, full_matrices=False)[0][:, 0]
    for _ in range(6):
        projector = np.eye(antennas)-np.outer(nuisance, nuisance.conj())
        projected = projector @ desired
        amplitude = np.sum(projected.conj()*prompts, axis=0)/(
            np.sum(desired.conj()*projected, axis=0).real+.01*antennas)
        remainder = prompts-desired*amplitude
        nuisance = np.linalg.svd(remainder, full_matrices=False)[0][:, 0]
    fitted_projector = np.eye(antennas)-np.outer(nuisance, nuisance.conj())
    projected = fitted_projector @ desired
    amplitude = np.sum(projected.conj()*prompts, axis=0)/(
        np.sum(desired.conj()*projected, axis=0).real+.01*antennas)
    one_source = desired*np.sum(desired.conj()*prompts, axis=0)/antennas
    two_source_residual = fitted_projector @ (prompts-desired*amplitude)
    reduction = (np.sum(abs(prompts-one_source)**2)-
                 np.sum(abs(two_source_residual)**2))
    raw = float(reduction/(noise*(prompts.shape[1]+antennas-1)))
    active = raw >= threshold
    projector = fitted_projector if active else np.eye(antennas)
    response = np.sum(model.conj()*(projector@model), axis=0).real
    weight = ((projector@model)/(response+.01)[None, :] if active
              else model/antennas)
    gain = response/antennas if active else np.ones(model.shape[1])
    return dict(raw=raw, fitted_projector=fitted_projector,
                projector=projector, weight=weight, gain=gain,
                dimension=antennas-int(active))


def longest_gap(published):
    """Count missing epochs including both ends of the fixed opportunity axis."""
    edges = np.diff(np.r_[0, (~published).astype(int), 0])
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return int((ends-starts).max()) if len(starts) else 0


def pair_score(scene, time, baseline, candidate, velocity_east,
               baseline_motion, candidate_motion):
    """Apply preregistered hard screen on observation times and truth errors."""
    opportunity = time >= .3
    truth = np.column_stack((velocity_east*time, np.zeros(len(time))))
    old_published = np.isfinite(baseline).all(axis=1)
    new_published = np.isfinite(candidate).all(axis=1)
    old_error = np.linalg.norm(baseline[:, :2]-truth, axis=1)
    new_error = np.linalg.norm(candidate[:, :2]-truth, axis=1)
    common = opportunity & old_published & new_published
    added = opportunity & ~old_published & new_published
    lost = opportunity & old_published & ~new_published

    def metrics(published, error):
        values = error[opportunity & published]
        return dict(published=int((opportunity & published).sum()),
                    longest_gap_epochs=longest_gap(published[opportunity]),
                    max_error_m=float(values.max()) if len(values) else None,
                    p95_error_m=float(np.percentile(values, 95)) if len(values) else None,
                    error_ge20=int((values >= 20.).sum()))

    def common_metrics(error):
        values = error[common]
        return dict(max_error_m=float(values.max()) if len(values) else None,
                    p95_error_m=float(np.percentile(values, 95)) if len(values) else None)

    old = metrics(old_published, old_error)
    new = metrics(new_published, new_error)
    old_common = common_metrics(old_error)
    new_common = common_metrics(new_error)
    result = dict(opportunities=int(opportunity.sum()), baseline=old,
                  candidate=new, common=dict(count=int(common.sum()),
                  baseline=old_common, candidate=new_common),
                  lost=int(lost.sum()), new=int(added.sum()),
                  new_ge20=int((added & (new_error >= 20.)).sum()),
                  baseline_motion_m=baseline_motion,
                  candidate_motion_m=candidate_motion)
    if scene == 'genuine_absent':
        result['new_after_half_second'] = int((
            (time >= .5) & ~old_published & new_published).sum())
        result['pass'] = (result['new_after_half_second'] == 0 and
                          new['error_ge20'] <= old['error_ge20'])
    else:
        tolerance = 1e-6
        comparable = old['published'] > 0 and result['common']['count'] > 0
        no_worse = (comparable and result['lost'] == 0 and
                    new['longest_gap_epochs'] <= old['longest_gap_epochs'] and
                    result['new_ge20'] == 0 and
                    all(new[key] <= old[key]+tolerance for key in
                        ('max_error_m', 'p95_error_m')) and
                    all(new_common[key] <= old_common[key]+tolerance for key in
                        ('max_error_m', 'p95_error_m')))
        if scene == 'weak':
            comparable = comparable and baseline_motion is not None and candidate_motion is not None
            no_worse = no_worse and comparable and candidate_motion <= baseline_motion+tolerance
            improved = comparable and (any(
                new[key] < old[key]-tolerance for key in
                ('max_error_m', 'p95_error_m')) or
                candidate_motion < baseline_motion-tolerance)
            result['pass'] = bool(no_worse and improved)
        else:
            result['pass'] = bool(no_worse)
    return result


def audit_blocks(record, tracking, trace, scratch, threshold):
    """Refit scores and beam outputs from IQ, then compare actual stream.

    Startup acquisition/calibration is repeated; the six-step nuisance fit,
    score, gate, weights and basis dimension are computed here without the
    receiver beam or historical gate-3 fit helper.
    """
    prns, templates, calibration, rate = extract_startup(record)
    np.testing.assert_array_equal(prns, tracking['startup_prns'])
    np.testing.assert_allclose(calibration, tracking['calibration'], rtol=0, atol=1e-12)
    np.testing.assert_allclose(rate, tracking['calibration_rate'], rtol=0, atol=1e-12)
    nominal = record['nominal'][ANTENNAS]
    startup = startup_prompts(record, prns, templates, calibration, rate)
    relative_amplitude, _ = estimate_amplitude(startup, nominal, prns)
    np.testing.assert_allclose(relative_amplitude,
                               tracking['startup_relative_amplitude'], rtol=0, atol=1e-12)
    model = relative_amplitude[:, None]*nominal
    np.testing.assert_allclose(model, trace['trace_nominal'], rtol=0, atol=1e-12)
    aided = [aided_template(template) for template in templates]
    stars = len(record['codes'])
    samples = record['samples'].shape[1]
    blocks = (samples+WIDTH-1)//WIDTH
    raw = np.full(blocks, np.nan)
    projector = np.tile(np.eye(len(ANTENNAS), dtype=complex), (blocks, 1, 1))
    weight = np.empty((blocks, len(ANTENNAS), stars), complex)
    gain = np.ones((blocks, stars))
    dimension = np.full(blocks, len(ANTENNAS), int)
    stream = np.memmap(scratch, dtype=np.complex128, mode='r',
                       shape=(stars, samples), order='C')

    for index in range(blocks):
        start = index*WIDTH
        end = min(start+WIDTH, samples)
        sample_index = np.arange(start, end)
        block = record['samples'][ANTENNAS, start:end]/(
            calibration[:, None]*np.exp(1j*rate[:, None]*sample_index[None, :]/FS))
        if start < STARTUP:
            weight[index] = model/len(ANTENNAS)
        else:
            time = sample_index/FS
            local = np.stack([local_signal(record['codes'][star], template, time)
                              for star, template in zip(prns, aided)])
            milliseconds = len(sample_index)//2046
            prompts = np.einsum('mkn,skn->msk',
                block.reshape(len(ANTENNAS), milliseconds, 2046),
                local.conj().reshape(len(prns), milliseconds, 2046))/2046
            grouped = group_five_ms(prompts)
            noise = float(np.mean(abs(block)**2)/2046)
            fitted = independent_fit(grouped, model, prns, noise, threshold)
            raw[index] = fitted['raw']
            projector[index] = fitted['projector']
            weight[index] = fitted['weight']
            gain[index] = fitted['gain']
            dimension[index] = fitted['dimension']
        np.testing.assert_allclose(stream[:, start:end],
                                   weight[index].conj().T@block,
                                   rtol=0, atol=1e-9,
                                   err_msg=f'beam stream block {index}')

    np.testing.assert_allclose(trace['trace_raw_trigger'], raw,
                               rtol=0, atol=1e-9, equal_nan=True)
    np.testing.assert_allclose(trace['trace_projectors'], projector,
                               rtol=0, atol=1e-9)
    np.testing.assert_array_equal(trace['trace_dimensions'], dimension)
    indices = (tracking['time']/.02).astype(int)
    np.testing.assert_allclose(tracking['gain'], gain[indices], rtol=0, atol=1e-9)
    measured = indices >= STARTUP//WIDTH
    np.testing.assert_allclose(tracking['ratio'][measured], raw[indices[measured]],
                               rtol=0, atol=1e-9)
    del stream
    return dict(raw=raw, projector=projector, weight=weight,
                gain=gain, dimension=dimension,
                relative_amplitude=relative_amplitude)


def run_path(record, case, threshold, folder):
    """Run one actual RF receiver and the same two navigation modes."""
    folder.mkdir(parents=True, exist_ok=True)
    scratch = folder/'beam_stream.dat'
    loops = []
    started = perf_counter()
    observations = frontend(record, ANTENNAS, split_support=True,
        receiver_pll=RECEIVER_PLL, evidence_trace=True,
        temporal_beam_trigger_enabled=False,
        beam_template_carrier_aiding=True, beam_coherent_ms=5,
        beam_startup_amplitude=True, fast_truncation=True,
        beam_projection_threshold=threshold, stream_memmap_path=scratch,
        loop_history=loops)
    elapsed = perf_counter()-started
    tracking = {name: value for name, value in observations.items()
                if not name.startswith('trace_')}
    trace = {name: value for name, value in observations.items()
             if name.startswith('trace_')}
    for name, arrays in (('tracking.npz', tracking), ('trace.npz', trace)):
        np.savez_compressed(folder/name, **arrays)
    loop_count = save_loops(folder, loops)
    fit = audit_blocks(record, tracking, trace, scratch, threshold)
    np.savez_compressed(folder/'fit.npz', **fit)
    scratch.unlink()
    history = analyze_trace(observations)
    groups = analyze_groups(trace)
    coherent = build_coherent_support(tracking, groups)
    for name, arrays in (('history.npz', history), ('groups.npz', groups),
                         ('support.npz', {f'coherent_{key}': value
                                         for key, value in coherent.items()})):
        np.savez_compressed(folder/name, **arrays)
    navigation, positions = replay_navigation(tracking, trace, coherent, case, folder)
    view = dict(observation=tracking['time'].copy(),
                available=tracking['available_time'].copy(),
                position=positions,
                support={mode: quality_support_counts(tracking,
                    tracking['support'] if mode == 'original' else coherent['support'])
                    for mode in ('original', 'short')},
                navigation=navigation)
    attack_block = round(case['attack_start']*FS/WIDTH)
    result = dict(frontend_walltime_s=elapsed, loop_history=loop_count,
                  projection_blocks=dict(
                      after_startup=dict(triggered=int((fit['raw'][STARTUP//WIDTH:] >= threshold).sum()),
                                         opportunities=len(fit['raw'])-STARTUP//WIDTH),
                      after_attack=dict(triggered=int((fit['raw'][attack_block:] >= threshold).sum()),
                                        opportunities=len(fit['raw'])-attack_block)),
                  navigation=navigation,
                  output_sha256={name: digest(folder/name) for name in SCIENCE})
    del observations, tracking, trace, fit, history, groups, coherent, loops
    gc.collect()
    return result, view


def plot_results(views):
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
        axes[row, 0].set(title=scene, ylabel='Qualified support count',
                         xlabel='Observation time (s)')
        axes[row, 1].set(ylabel='Published horizontal error (m)',
                         xlabel='Observation time (s)')
        for axis in axes[row]:
            axis.grid(alpha=.25)
            axis.set_xlim(0., views[scene][3.]['observation'][-1])
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].legend(fontsize=7)
    figure.savefig(OUTPUT/'overview.png', dpi=150)
    plt.close(figure)


def main():
    """Generate four fixed IQ records and finish all eight paired RF paths."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sources = source_hashes()
    source_snapshot(OUTPUT/'source_snapshot.zip', sources)
    plan = dict(scope=__doc__, seed=SEED, scenes=SCENES, gates=GATES,
        scene_parameters={scene: scene_plan(scene) for scene in SCENES},
        runtime=dict(python=sys.version.split()[0], numpy=np.__version__,
                     scipy=scipy.__version__),
        receiver=dict(processed_antennas=ANTENNAS, receiver_pll=RECEIVER_PLL,
            beam_template_carrier_aiding=True, beam_coherent_ms=5,
            beam_startup_amplitude=True, fast_truncation=True,
            navigation_modes=('original', 'short')),
        evaluation='observation time >=0.3 s; absent new publication >=0.5 s',
        source_sha256=sources)
    (OUTPUT/'plan.json').write_text(json.dumps(serializable(plan), indent=2,
        allow_nan=False), encoding='utf-8')
    cases = {}; views = {}; input_hashes = {}; output_hashes = {}
    first_prefix = first_geometry = None
    for scene in SCENES:
        print(scene, 'IQ generation start', flush=True)
        settings = scene_plan(scene)
        generated_scene = ('no_attack_control' if scene.startswith('h0_')
                           else scene)
        record, identity = generate_new_record(
            settings, generated_scene, fake_rate=settings['fake_rate_mps'])
        folder = OUTPUT/scene
        folder.mkdir(exist_ok=True)
        hashes, first_prefix, first_geometry = save_new_input(
            folder, record, identity, first_prefix, first_geometry)
        input_hashes[scene] = hashes
        paths = {}; views[scene] = {}
        for threshold in GATES:
            name = f'gate{threshold:g}'
            paths[name], views[scene][threshold] = run_path(
                record, identity['case'], threshold, folder/name)
            output_hashes[f'{scene}/{name}'] = paths[name]['output_sha256']
            print(scene, name, 'RF, fit audit and navigation saved', flush=True)
        baseline = views[scene][3.]
        candidate = views[scene][2.]
        np.testing.assert_array_equal(baseline['observation'],
                                      candidate['observation'])
        comparison = {}
        for mode in ('short', 'original'):
            old_motion = baseline['navigation'][mode]['max_relative_motion_error_m']
            new_motion = candidate['navigation'][mode]['max_relative_motion_error_m']
            comparison[mode] = pair_score(scene, baseline['observation'],
                baseline['position'][mode], candidate['position'][mode],
                identity['case']['velocity_east'], old_motion, new_motion)
            comparison[mode]['baseline_initial_anchor_error_m'] = (
                baseline['navigation'][mode]['initial_horizontal_error_m'])
            comparison[mode]['candidate_initial_anchor_error_m'] = (
                candidate['navigation'][mode]['initial_horizontal_error_m'])
        comparison['original'].pop('pass')
        cases[scene] = dict(input_sha256=hashes, paths=paths,
                            short_screen=comparison['short'],
                            original_diagnostic=comparison['original'])
        del record
        gc.collect()
    plot_results(views)
    manifest = dict(source_snapshot_sha256=digest(OUTPUT/'source_snapshot.zip'),
        plan_sha256=digest(OUTPUT/'plan.json'), input_sha256=input_hashes,
        output_sha256=output_hashes, figure_sha256=digest(OUTPUT/'overview.png'))
    (OUTPUT/'manifest.json').write_text(json.dumps(serializable(manifest),
        indent=2, allow_nan=False), encoding='utf-8')
    assert source_hashes() == sources
    summary = dict(scope=__doc__, source_sha256=sources,
        source_snapshot_sha256=digest(OUTPUT/'source_snapshot.zip'),
        plan_sha256=digest(OUTPUT/'plan.json'),
        manifest_sha256=digest(OUTPUT/'manifest.json'),
        figure_sha256=digest(OUTPUT/'overview.png'),
        screen_pass=all(cases[scene]['short_screen']['pass'] for scene in SCENES),
        cases=cases)
    (OUTPUT/'summary.json').write_text(json.dumps(serializable(summary),
        indent=2, allow_nan=False), encoding='utf-8')
    return summary


if __name__ == '__main__':
    main()
