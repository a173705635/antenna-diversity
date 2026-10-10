"""Stage 48: fixed receiver-only nominal yaw on three saved 20 s IQ scenes.

Positive yaw rotates antenna positions counterclockwise around +z in the
existing steering convention. IQ and physical truth remain at zero yaw.
This is a sensitivity screen, not an attitude estimator or a safety claim.
"""
import gc
import json
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy

from core.presets import mate70_phase_centers
from core.provenance import source_hashes
from core.recovery_geometry import FS, sky, steering
from core.serialization import serializable
from experiments.paths import RESULTS
from experiments.single_source_long_coherent import source_snapshot
from experiments.single_source_phase_exclusion import verify_source_snapshot
from experiments.single_source_projection_threshold import SCIENCE, run_path, pair_score
from experiments.single_source_projection_long import OUTPUT as STAGE47, long_plan
from experiments.single_source_suppression_study import (
    OUTPUT as STUDY_OUTPUT, ANTENNAS, RECEIVER_PLL, digest,
)
from experiments.single_source_template_coherence import WIDTH
from tools.resume_projection_long import recover_path


OUTPUT = RESULTS/STUDY_OUTPUT/'nominal_yaw'
SEED = 2026100404
SCENES = ('h0_drop', 'weak', 'genuine_absent')
YAWS_DEG = (0., -2., 2.)
STAGE47_SUMMARY_SHA256 = '47a73e338bf086f8c0418f31c3db07081dea299129775be54b45d22d206e5a86'
INPUT_FILES = ('samples.npy', 'observations.npz', 'metadata.json')


def path_schedule():
    return [(scene, yaw) for scene in SCENES for yaw in YAWS_DEG]


def receiver_record(record, directions, antennas, yaw_deg):
    """Return a shallow record copy, changing only receiver nominal steering.

    `physical` is the untouched generated steering; it is only for offline
    diagnostics. Yaw is passed to steering in radians, with positive yaw
    rotating antenna coordinates counterclockwise about +z.
    """
    physical = record['nominal']
    received = dict(record)
    if yaw_deg:
        received['nominal'] = steering(directions, antennas,
                                       np.deg2rad(yaw_deg))
    return received, physical


def score_pair(scene, time, baseline, candidate, velocity_east,
               baseline_motion, candidate_motion):
    """Use the old opportunity/error accounting with strict non-degradation.

    The Stage 47 weak-screen improvement requirement is deliberately absent:
    yaw is an input error stress test, not a candidate suppression algorithm.
    """
    result = pair_score(scene, time, baseline, candidate, velocity_east,
                        baseline_motion, candidate_motion)
    old, new = result['baseline'], result['candidate']
    common = result['common']
    if scene == 'genuine_absent':
        published = np.isfinite(candidate).all(axis=1)
        result['candidate_after_half_published'] = int(((time >= .5) & published).sum())
        result['pass'] = bool(result['candidate_after_half_published'] == 0 and
                              new['error_ge20'] == 0)
    else:
        tolerance = 1e-6
        result['pass'] = bool(
            old['published'] > 0 and common['count'] > 0 and result['lost'] == 0
            and new['longest_gap_epochs'] <= old['longest_gap_epochs']
            and new['error_ge20'] == 0 and result['new_ge20'] == 0
            and all(new[key] <= old[key]+tolerance for key in
                    ('max_error_m', 'p95_error_m'))
            and all(common['candidate'][key] <= common['baseline'][key]+tolerance
                    for key in ('max_error_m', 'p95_error_m'))
            and (scene != 'weak' or candidate_motion <= baseline_motion+tolerance))
    return result


def read_npz(path):
    with np.load(path) as saved:
        return dict(saved)


def load_input(folder, settings, first_prefix, first_geometry):
    """Hash and memory-map the immutable Stage 47 IQ; check same-seed pairing."""
    source = folder/'input'
    hashes = {name: digest(source/name) for name in INPUT_FILES}
    metadata = json.loads((source/'metadata.json').read_text(encoding='utf-8'))
    assert metadata['samples_sha256'] == hashes['samples.npy']
    assert metadata['observations_sha256'] == hashes['observations.npz']
    assert metadata['seed'] == settings['seed']
    assert metadata['case']['duration'] == settings['duration_s']
    assert metadata['case']['stars'] == settings['satellites']
    assert metadata['post_cn0_db_hz'] == settings['post_0p3s_cn0_db_hz']
    with np.load(source/'observations.npz') as saved:
        record = dict(saved)
    record['samples'] = np.load(source/'samples.npy', mmap_mode='r')
    onset = metadata['pre_attack_exact_prefix_samples']
    if first_prefix is None:
        first_prefix = record['samples'][:, :onset].copy()
        first_geometry = {key: value.copy() for key, value in record.items()
                          if key != 'samples'}
    else:
        np.testing.assert_array_equal(record['samples'][:, :onset], first_prefix)
        for key, value in first_geometry.items():
            np.testing.assert_array_equal(record[key], value, err_msg=key)
    return record, metadata, hashes, first_prefix, first_geometry


def path_view(folder, receipt):
    """Read only saved navigation and timing for comparison with new RF."""
    tracking = read_npz(folder/'tracking.npz')
    navigation = read_npz(folder/'navigation.npz')
    return dict(observation=tracking['time'],
                available=tracking['available_time'],
                position={mode: navigation[f'{mode}_position']
                          for mode in ('short', 'original')},
                navigation=receipt['navigation'])


def physical_diagnostics(record, seed, folder):
    """Compare model and projected true star response in calibrated channels.

    The generator draws fixed per-antenna phase and amplitude gains before IQ.
    Recreate those draws separately from its seed. Evaluate the frontend's
    exp(1j*calibration_rate*t) correction at each 20 ms block midpoint. This
    point sample is a diagnostic convention, not a block-average RF response.
    Projection retention uses the actual fitted projector for that block.
    """
    tracking = read_npz(folder/'tracking.npz')
    trace = read_npz(folder/'trace.npz')
    fit = read_npz(folder/'fit.npz')
    rng = np.random.default_rng(seed)
    gain = np.exp(1j*np.deg2rad(10.*rng.normal(size=record['nominal'].shape[0])))
    gain *= 10**(rng.normal(size=len(gain))/20)
    physical = gain[ANTENNAS, None]*record['nominal'][ANTENNAS]
    block_midpoint = (np.arange(len(fit['projector']))+.5)*WIDTH/FS
    correction = (tracking['calibration'][None, :, None]*
                  np.exp(1j*tracking['calibration_rate'][None, :, None]*
                         block_midpoint[:, None, None]))
    calibrated = physical[None, :, :]/correction
    model = trace['trace_nominal']
    norm = np.linalg.norm(calibrated, axis=1)
    pre = np.linalg.norm(calibrated-model[None], axis=1)/norm
    scalar = np.sum(model.conj()[None]*calibrated, axis=1)/(
        np.sum(abs(model)**2, axis=0)[None])
    post = np.linalg.norm(calibrated-model[None]*scalar[:, None, :], axis=1)/norm
    projected = np.einsum('bij,bjk->bik', fit['projector'], calibrated)
    retained = np.sum(abs(projected)**2, axis=1)/np.sum(abs(calibrated)**2, axis=1)
    return dict(calibration=tracking['calibration'],
                calibration_rate=tracking['calibration_rate'],
                startup_relative_amplitude=tracking['startup_relative_amplitude'],
                block_midpoint_s=block_midpoint,
                mismatch_before_scalar=pre, mismatch_after_scalar=post,
                true_star_power_retained=retained)


def diagnostic_summary(physical, baseline):
    """Report calibration changes and physical mismatch without using truth online."""
    phase = np.rad2deg(np.angle(physical['calibration']/baseline['calibration']))
    amplitude = abs(physical['calibration']/baseline['calibration'])
    rate = physical['calibration_rate']-baseline['calibration_rate']
    return dict(calibration_phase_change_deg=phase.tolist(),
                calibration_amplitude_ratio=amplitude.tolist(),
                calibration_rate_change_rad_s=rate.tolist(),
                startup_relative_amplitude_change=(
                    physical['startup_relative_amplitude']-
                    baseline['startup_relative_amplitude']).tolist(),
                mismatch_before_scalar_median_by_prn=np.median(
                    physical['mismatch_before_scalar'], axis=0).tolist(),
                mismatch_before_scalar_max_by_prn=np.max(
                    physical['mismatch_before_scalar'], axis=0).tolist(),
                mismatch_after_scalar_median_by_prn=np.median(
                    physical['mismatch_after_scalar'], axis=0).tolist(),
                mismatch_after_scalar_max_by_prn=np.max(
                    physical['mismatch_after_scalar'], axis=0).tolist(),
                block_midpoint_first_last_s=[float(physical['block_midpoint_s'][0]),
                                             float(physical['block_midpoint_s'][-1])],
                true_star_power_retained_median=float(np.median(
                    physical['true_star_power_retained'])),
                true_star_power_retained_min=float(np.min(
                    physical['true_star_power_retained'])))


def preflight():
    """Verify frozen Stage 47 identity while allowing precisely this new source."""
    assert digest(STAGE47/'summary.json') == STAGE47_SUMMARY_SHA256
    baseline = json.loads((STAGE47/'summary.json').read_text(encoding='utf-8'))
    frozen = baseline['source_sha256']
    current = source_hashes()
    assert len(frozen) == 98
    assert set(current)-set(frozen) == {'experiments/single_source_nominal_yaw.py'}
    assert {name: current[name] for name in frozen} == frozen
    assert baseline['source_snapshot_sha256'] == digest(STAGE47/'source_snapshot.zip')
    verify_source_snapshot(STAGE47/'source_snapshot.zip', frozen)
    assert baseline['plan_sha256'] == digest(STAGE47/'plan.json')
    assert baseline['manifest_sha256'] == digest(STAGE47/'manifest.json')
    return baseline, current


def write_json(path, value):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(serializable(value), indent=2,
                                    allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def plot_scene(scene, views, folder):
    figure, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                constrained_layout=True)
    for yaw in YAWS_DEG:
        view = views[yaw]
        time = view['observation']
        truth = np.column_stack((2.*time, np.zeros(len(time))))
        for mode, style in (('short', '-'), ('original', '--')):
            position = view['position'][mode]
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
    figure.savefig(folder/'overview.png', dpi=150)
    plt.close(figure)


def main():
    """Run six new RF paths serially; reuse each completed receipt after interruption."""
    baseline, sources = preflight()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = dict(scope=__doc__, seed=SEED, scenes=SCENES, yaw_deg=YAWS_DEG,
                gate=3., source_sha256=sources,
                stage47_summary_sha256=STAGE47_SUMMARY_SHA256,
                saved_path_recovery_sha256=digest(Path(
                    sys.modules[recover_path.__module__].__file__)),
                scene_parameters={scene: long_plan(SEED, scene) for scene in SCENES},
                receiver=dict(processed_antennas=ANTENNAS, receiver_pll=RECEIVER_PLL,
                    beam_template_carrier_aiding=True, beam_coherent_ms=5,
                    beam_startup_amplitude=True, fast_truncation=True),
                physical_diagnostic_time='Each 20 ms fit projector uses its '
                    'block midpoint; physical channel response is divided by '
                    'calibration*exp(1j*calibration_rate*midpoint_s).',
                evaluation='Observation >=0.3 s, 1969 opportunities; no lost '
                    'short releases, no longer gap or worse max/P95 all/common '
                    'error or anchor-relative motion; absent >=0.5 s zero '
                    'releases and >=0.3 s zero >=20 m errors. Original separate.',
                runtime=dict(python=sys.version.split()[0], numpy=np.__version__,
                             scipy=scipy.__version__, platform=platform.platform()))
    plan_path = OUTPUT/'plan.json'
    if plan_path.exists():
        assert json.loads(plan_path.read_text(encoding='utf-8')) == serializable(plan)
    else:
        write_json(plan_path, plan)
    snapshot = OUTPUT/'source_snapshot.zip'
    if snapshot.exists():
        verify_source_snapshot(snapshot, sources)
    else:
        source_snapshot(snapshot, sources)

    antennas = np.array(list(mate70_phase_centers().values()))
    directions = sky(SEED, 24)
    cases = {}; inputs = {}; outputs = {}; figures = {}
    first_prefix = first_geometry = None
    for scene in SCENES:
        old_folder = STAGE47/str(SEED)/scene
        settings = long_plan(SEED, scene)
        record, identity, input_sha, first_prefix, first_geometry = load_input(
            old_folder, settings, first_prefix, first_geometry)
        assert input_sha == baseline['cases'][str(SEED)][scene]['input_sha256']
        np.testing.assert_allclose(record['nominal'],
                                   steering(directions, antennas, 0.), rtol=0, atol=1e-14)
        inputs[scene] = input_sha
        scene_folder = OUTPUT/scene
        scene_folder.mkdir(exist_ok=True)
        views = {}; paths = {}; diagnostics = {}
        old_path = old_folder/'gate3'
        old_receipt = json.loads((old_path/'path_receipt.json').read_text(encoding='utf-8'))
        assert old_receipt['output_sha256'] == baseline['cases'][str(SEED)][scene][
            'paths']['gate3']['output_sha256']
        assert {name: digest(old_path/name) for name in SCIENCE} == old_receipt['output_sha256']
        views[0.] = path_view(old_path, old_receipt)
        paths['yaw0'] = dict(status='reused_stage47_gate3',
                             source=str(old_path), output_sha256=old_receipt['output_sha256'])
        base_physical = physical_diagnostics(record, SEED, old_path)
        for yaw in (angle for scheduled_scene, angle in path_schedule()
                    if scheduled_scene == scene and angle != 0.):
            name = f'yaw_{"plus" if yaw > 0 else "minus"}{abs(yaw):g}'
            folder = scene_folder/name
            receipt_path = folder/'path_receipt.json'
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
                assert receipt['input_sha256'] == input_sha
                assert receipt['source_sha256'] == sources
                assert receipt['yaw_deg'] == yaw
                assert {file: digest(folder/file) for file in SCIENCE} == receipt['output_sha256']
                assert digest(folder/'physical_diagnostics.npz') == receipt['diagnostics_sha256']
                view = path_view(folder, receipt)
            elif all((folder/file).exists() for file in SCIENCE):
                result, view = recover_path(folder, record, identity['case'], 3.)
                physical_arrays = physical_diagnostics(record, SEED, folder)
                diagnostics_path = folder/'physical_diagnostics.npz'
                if diagnostics_path.exists():
                    saved = read_npz(diagnostics_path)
                    for key, value in physical_arrays.items():
                        np.testing.assert_array_equal(saved[key], value)
                else:
                    np.savez_compressed(diagnostics_path, **physical_arrays)
                receipt = dict(**result, yaw_deg=yaw, input_sha256=input_sha,
                               source_sha256=sources,
                               diagnostics_sha256=digest(diagnostics_path),
                               runtime=plan['runtime'],
                               provenance='recovered_saved_science_no_beam_stream')
                write_json(receipt_path, receipt)
            else:
                assert not folder.exists(), ('Unreceipted RF path retained for '
                                             'manual audit', folder)
                received, physical = receiver_record(record, directions, antennas, yaw)
                assert physical is record['nominal']
                result, view = run_path(received, identity['case'], 3., folder)
                physical_arrays = physical_diagnostics(record, SEED, folder)
                np.savez_compressed(folder/'physical_diagnostics.npz', **physical_arrays)
                receipt = dict(**result, yaw_deg=yaw, input_sha256=input_sha,
                               source_sha256=sources,
                               diagnostics_sha256=digest(folder/'physical_diagnostics.npz'),
                               runtime=plan['runtime'])
                write_json(receipt_path, receipt)
                del received
            views[yaw] = view
            paths[name] = receipt
            outputs[f'{scene}/{name}'] = receipt['output_sha256']
            physical_arrays = physical_diagnostics(record, SEED, folder)
            diagnostics[name] = diagnostic_summary(physical_arrays, base_physical)
            print(scene, name, 'RF and audit complete', flush=True)
            gc.collect()
        time = views[0.]['observation']
        assert int((time >= .3).sum()) == 1969
        assert int((time >= .5).sum()) == 1949
        scores = {}
        for yaw in YAWS_DEG[1:]:
            view = views[yaw]
            np.testing.assert_array_equal(time, view['observation'])
            np.testing.assert_array_equal(views[0.]['available'], view['available'])
            mode_scores = {}
            for mode in ('short', 'original'):
                old = views[0.]; new = view
                comparison = score_pair(scene, time, old['position'][mode],
                                        new['position'][mode],
                                        identity['case']['velocity_east'],
                                        old['navigation'][mode]['max_relative_motion_error_m'],
                                        new['navigation'][mode]['max_relative_motion_error_m'])
                comparison['baseline_initial_anchor_error_m'] = old['navigation'][mode][
                    'initial_horizontal_error_m']
                comparison['candidate_initial_anchor_error_m'] = new['navigation'][mode][
                    'initial_horizontal_error_m']
                mode_scores[mode] = comparison
            scores[f'{yaw:+g}'] = mode_scores
        plot_scene(scene, views, scene_folder)
        figures[scene] = digest(scene_folder/'overview.png')
        cases[scene] = dict(input_sha256=input_sha, paths=paths,
                            scores=scores, physical_diagnostics=diagnostics)
        del record, views, base_physical
        gc.collect()
    assert source_hashes() == sources
    manifest = dict(plan_sha256=digest(plan_path),
                    source_snapshot_sha256=digest(snapshot),
                    stage47_summary_sha256=digest(STAGE47/'summary.json'),
                    input_sha256=inputs, output_sha256=outputs,
                    figure_sha256=figures,
                    receipt_sha256={f'{scene}/{name}': digest(
                        OUTPUT/scene/name/'path_receipt.json')
                        for scene in SCENES for name in (
                            'yaw_minus2', 'yaw_plus2')})
    write_json(OUTPUT/'manifest.json', manifest)
    summary = dict(scope=__doc__, plan_sha256=digest(plan_path),
                   source_sha256=sources,
                   source_snapshot_sha256=digest(snapshot),
                   manifest_sha256=digest(OUTPUT/'manifest.json'),
                   screen_pass=all(cases[scene]['scores'][f'{yaw:+g}']['short']['pass']
                                   for scene in SCENES for yaw in YAWS_DEG[1:]),
                   cases=cases)
    write_json(OUTPUT/'summary.json', summary)
    return summary


if __name__ == '__main__':
    main()
