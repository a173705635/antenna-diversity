"""Replay one fixed carrier leave-one-out phase bridge on saved observations.

The default full-carrier fit is always tried first. Only when it fails at both
recent anchors may a unique excluded PRN carry phase; support, smoothing, and
code admission remain the saved receiver's values.
"""
import hashlib
import json
from zipfile import ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from core.provenance import source_hashes
from core.gps_simulation import receiver_reference
from core.serialization import serializable
from core.spoofing.single_source import navigate
from experiments.paths import SPOOFING_RESULTS as RESULTS
from experiments.spoofing.single_source_carrier_fault import (
    OUTPUT as STAGE20, FAULTS, FAULT_PRN, FAULT_TIME_S,
    compare_fault, inject_carrier_fault,
)
from experiments.spoofing.single_source_code_blend import (
    OUTPUT as STAGE19, assert_saved_navigation, published_metrics,
)
from experiments.spoofing.single_source_coherent_float import (
    CASES, STAGE14, STAGE16, STAGE17, preflight_sources,
)
from experiments.spoofing.single_source_coherent_navigation import (
    quality_support_counts, score_mode,
)
from experiments.spoofing.single_source_coherent_transfer import INPUT as TRANSFER_INPUT
from experiments.spoofing.single_source_float_window_study import CASE_INPUTS
from experiments.spoofing.single_source_long_coherent import (
    OUTPUT as STAGE21, SCENES, navigation_prefix, source_snapshot,
)
from experiments.spoofing.single_source_suppression_study import OUTPUT as STUDY_OUTPUT, digest


OUTPUT = RESULTS / STUDY_OUTPUT / 'phase_exclusion'
CASES22 = tuple(CASES) + tuple(f'fault_{kind}' for kind in FAULTS) + \
          tuple(f'long_{scene}' for scene in SCENES)


def verify_source_snapshot(path, hashes):
    """Check Stage 21's frozen bytes against its own historic hash map."""
    with ZipFile(path) as archive:
        assert set(archive.namelist()) == set(hashes)
        for name, expected in hashes.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected


def preflight_inputs():
    preflight_sources()
    stage19 = json.loads((STAGE19 / 'summary.json').read_text(encoding='utf-8'))
    assert stage19['manifest_sha256'] == digest(STAGE19 / 'manifest.json')
    for name in CASES:
        assert stage19['cases'][name]['result_sha256'] == digest(STAGE19 / f'{name}.npz')
    stage20 = json.loads((STAGE20 / 'summary.json').read_text(encoding='utf-8'))
    assert stage20['manifest_sha256'] == digest(STAGE20 / 'manifest.json')
    stage20_manifest = json.loads((STAGE20 / 'manifest.json').read_text(encoding='utf-8'))
    assert stage20_manifest['stage19_summary_sha256'] == digest(STAGE19 / 'summary.json')
    for kind in FAULTS:
        assert stage20['cases'][kind]['result_sha256'] == digest(STAGE20 / f'{kind}.npz')
    stage21 = json.loads((STAGE21 / 'summary.json').read_text(encoding='utf-8'))
    assert stage21['manifest_sha256'] == digest(STAGE21 / 'manifest.json')
    assert stage21['plan_sha256'] == digest(STAGE21 / 'plan.json')
    assert stage21['source_snapshot_sha256'] == digest(STAGE21 / 'source_snapshot.zip')
    verify_source_snapshot(STAGE21 / 'source_snapshot.zip', stage21['source_sha256'])
    for scene in SCENES:
        folder = STAGE21 / scene
        for name, expected in stage21['cases'][scene]['input_sha256'].items():
            assert expected == digest(folder / 'input' / name)
        for name, expected in stage21['cases'][scene]['output_sha256'].items():
            assert expected == digest(folder / name)
    return stage19, stage20, stage21


def replay_pair(observations, saved, reference_prefix):
    """Run off and on from independent initial states; match the frozen off run."""
    off = navigate(observations, diagnostics=True, code_blend=0.)
    assert_saved_navigation(off, saved, prefix=reference_prefix)
    on = navigate(observations, diagnostics=True, code_blend=0.,
                  phase_leave_one_out=True)
    np.testing.assert_array_equal(on['selected'], off['selected'])
    return off, on


def fault_window_mask(observations):
    """Use the exact targeted PRN-14 observation window from Stage 20."""
    star = FAULT_PRN-1
    return ((observations['time'] >= FAULT_TIME_S)
            & (observations['arc'][:, star] == 0)
            & np.isfinite(observations['carrier'][:, star]))


def saved_cases(stage19, stage21):
    for name in CASES:
        if name == 'transfer_weak':
            tracking_path = STAGE17 / 'tracking.npz'
            support_path = STAGE17 / 'support.npz'
            metadata_path = TRANSFER_INPUT / 'metadata.json'
        else:
            tracking_path = STAGE14 / name / 'tracking.npz'
            support_path = STAGE16 / f'{name}.npz'
            metadata_path = CASE_INPUTS[name][1] / 'metadata.json'
        recorded = stage19['cases'][name]
        assert recorded['tracking_sha256'] == digest(tracking_path)
        assert recorded['support_sha256'] == digest(support_path)
        assert recorded['metadata_sha256'] == digest(metadata_path)
        with np.load(tracking_path) as saved:
            tracking = dict(saved)
        with np.load(support_path) as saved:
            support = saved['coherent_support'].copy()
        with np.load(STAGE19 / f'{name}.npz') as saved:
            reference = dict(saved)
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        yield name, {**tracking, 'support': support}, reference, '0_', metadata, \
              dict(tracking_sha256=digest(tracking_path),
                   support_sha256=digest(support_path),
                   reference_sha256=digest(STAGE19 / f'{name}.npz'),
                   metadata_sha256=digest(metadata_path))

    with np.load(STAGE17 / 'tracking.npz') as saved:
        tracking = dict(saved)
    with np.load(STAGE17 / 'support.npz') as saved:
        support = saved['coherent_support'].copy()
    observations = {**tracking, 'support': support}
    metadata_path = TRANSFER_INPUT / 'metadata.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    for kind in FAULTS:
        changed, _ = inject_carrier_fault(observations, kind)
        with np.load(STAGE20 / f'{kind}.npz') as saved:
            reference = dict(saved)
        yield f'fault_{kind}', changed, reference, '0_', metadata, \
              dict(tracking_sha256=digest(STAGE17 / 'tracking.npz'),
                   support_sha256=digest(STAGE17 / 'support.npz'),
                   reference_sha256=digest(STAGE20 / f'{kind}.npz'),
                   metadata_sha256=digest(metadata_path))

    for scene in SCENES:
        folder = STAGE21 / scene
        with np.load(folder / 'tracking.npz') as saved:
            tracking = dict(saved)
        with np.load(folder / 'support.npz') as saved:
            support = saved['coherent_support'].copy()
        with np.load(folder / 'navigation.npz') as saved:
            reference = dict(saved)
        metadata_path = folder / 'input' / 'metadata.json'
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        yield f'long_{scene}', {**tracking, 'support': support}, reference, \
              'coherent_0_', metadata, \
              dict(tracking_sha256=stage21['cases'][scene]['output_sha256']['tracking.npz'],
                   support_sha256=stage21['cases'][scene]['output_sha256']['support.npz'],
                   reference_sha256=stage21['cases'][scene]['output_sha256']['navigation.npz'],
                   metadata_sha256=stage21['cases'][scene]['input_sha256']['metadata.json'])


def score_pair(observations, metadata, off, on):
    time = observations['time']
    available = observations['available_time']
    post = available >= metadata['case']['attack_start']
    truth = np.column_stack((metadata['case']['velocity_east'] * time,
                             np.zeros(len(time))))
    errors = {label: np.linalg.norm(nav['position'][:, :2] - truth, axis=1)
              for label, nav in (('off', off), ('on', on))}
    common = post & np.isfinite(off['position']).all(axis=1) \
                  & np.isfinite(on['position']).all(axis=1)
    counts = quality_support_counts(observations, observations['support'])
    modes = {}
    for label, nav in (('off', off), ('on', on)):
        modes[label] = score_mode(nav, observations, metadata['case'], counts,
                                  observations['support'])
        modes[label]['common_published'] = published_metrics(errors[label], common)
    excluded = on['diagnostics']['phase_excluded_prn']
    modes['on']['excluded_phase_epochs'] = int((post & (excluded > 0)).sum())
    modes['on']['excluded_prn_counts'] = {
        str(prn): int((post & (excluded == prn)).sum())
        for prn in np.unique(excluded[post]) if prn > 0}
    return modes, errors, common


def fit_geometry(observations, k, j, indices):
    """Independently inspect the original carrier-fit design and residuals."""
    reference, rotation = receiver_reference()
    current = observations['satellites'][k, indices] - reference
    previous = observations['satellites'][j, indices] - reference
    current_norm = np.linalg.norm(current, axis=1)
    previous_norm = np.linalg.norm(previous, axis=1)
    design = np.column_stack((-(current/current_norm[:, None]) @ rotation.T,
                              np.ones(len(indices))))
    measured = (observations['carrier'][k, indices]
                - observations['carrier'][j, indices]
                - (current_norm - previous_norm))
    solution = np.linalg.lstsq(design, measured, rcond=None)[0]
    residual = design @ solution - measured
    variance = max(.02**2, float(residual @ residual)/(len(indices)-4))
    covariance = variance * np.linalg.inv(design.T @ design)
    return dict(condition=float(np.linalg.cond(design)),
                horizontal_std_m=float(np.sqrt(np.linalg.eigvalsh(
                    covariance[:2, :2])[-1])),
                max_residual_m=float(np.max(abs(residual))), residual_m=residual)


def exclusion_geometry(observations, navigation):
    """Keep full and excluded-fit geometry for every actually used exclusion."""
    n, stars = observations['carrier'].shape
    arrays = dict(full_condition=np.full((n, 2), np.nan),
                  full_horizontal_std_m=np.full((n, 2), np.nan),
                  full_max_residual_m=np.full((n, 2), np.nan),
                  candidate_condition=np.full(n, np.nan),
                  candidate_horizontal_std_m=np.full(n, np.nan),
                  candidate_max_residual_m=np.full(n, np.nan))
    selected = navigation['selected']
    diagnostic = navigation['diagnostics']
    for k in np.flatnonzero(diagnostic['phase_excluded_prn'] > 0):
        for lag in (1, 2):
            j = k-lag
            if j < 0 or diagnostic['phase_full_common_count'][k, lag-1] < 5:
                continue
            common = (selected[k] & selected[j]
                      & np.isfinite(observations['carrier'][k])
                      & np.isfinite(observations['carrier'][j])
                      & (observations['arc'][k] == observations['arc'][j]))
            indices = np.flatnonzero(common)
            geometry = fit_geometry(observations, k, j, indices)
            arrays['full_condition'][k, lag-1] = geometry['condition']
            arrays['full_horizontal_std_m'][k, lag-1] = geometry['horizontal_std_m']
            arrays['full_max_residual_m'][k, lag-1] = geometry['max_residual_m']
            recorded = diagnostic['phase_full_residual_m'][k, lag-1, indices]
            if np.isfinite(recorded).all():
                np.testing.assert_allclose(geometry['residual_m'], recorded, atol=1e-8)
        lag = diagnostic['phase_lag_epochs'][k]
        j = k-lag
        excluded = diagnostic['phase_excluded_prn'][k]-1
        common = (selected[k] & selected[j]
                  & np.isfinite(observations['carrier'][k])
                  & np.isfinite(observations['carrier'][j])
                  & (observations['arc'][k] == observations['arc'][j]))
        indices = np.flatnonzero(common)
        retained = indices[indices != excluded]
        geometry = fit_geometry(observations, k, j, retained)
        for field in ('condition', 'horizontal_std_m', 'max_residual_m'):
            arrays[f'candidate_{field}'][k] = geometry[field]
        np.testing.assert_allclose(geometry['residual_m'],
                                   diagnostic['phase_candidate_residual_m'][k, retained],
                                   atol=1e-8)
    return arrays


def main():
    stage19, stage20, stage21 = preflight_inputs()
    sources = source_hashes()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_snapshot(OUTPUT / 'source_snapshot.zip', sources)
    (OUTPUT / 'manifest.json').write_text(json.dumps(serializable(dict(
        scope=__doc__, source_sha256=sources,
        source_snapshot_sha256=digest(OUTPUT / 'source_snapshot.zip'),
        stage19_summary_sha256=digest(STAGE19 / 'summary.json'),
        stage20_summary_sha256=digest(STAGE20 / 'summary.json'),
        stage21_summary_sha256=digest(STAGE21 / 'summary.json'),
        stage21_source_snapshot_sha256=digest(STAGE21 / 'source_snapshot.zip'),
        code_blend=0., phase_leave_one_out=(False, True), cases=CASES22)),
        indent=2, allow_nan=False), encoding='utf-8')
    reports = {}
    plots = {}
    for name, observations, saved, prefix, metadata, hashes in saved_cases(stage19, stage21):
        off, on = replay_pair(observations, saved, prefix)
        half = len(observations['time']) // 2
        prefix_result = navigate(navigation_prefix(observations, half),
                                 diagnostics=True, code_blend=0.,
                                 phase_leave_one_out=True)
        for key, value in prefix_result.items():
            if key == 'diagnostics':
                for diagnostic, data in value.items():
                    np.testing.assert_array_equal(data,
                                                  on[key][diagnostic][:half])
            elif key != 'initialization_time':
                np.testing.assert_array_equal(value, on[key][:half])
        modes, errors, common = score_pair(observations, metadata, off, on)
        arrays = dict(observation_time=observations['time'],
                      available_time=observations['available_time'],
                      common_published=common)
        fault_comparison = None
        if name.startswith('fault_'):
            mask = fault_window_mask(observations)
            fault_comparison, _ = compare_fault(off, on, observations, metadata, mask)
            arrays['fault_window'] = mask
        arrays.update({f'geometry_{key}': value for key, value in
                       exclusion_geometry(observations, on).items()})
        for label, navigation in (('off', off), ('on', on)):
            arrays[f'{label}_horizontal_error_m'] = errors[label]
            for key, value in navigation.items():
                if key == 'diagnostics':
                    for diagnostic, data in value.items():
                        arrays[f'{label}_diagnostic_{diagnostic}'] = data
                else:
                    arrays[f'{label}_{key}'] = np.nan if value is None else value
        path = OUTPUT / f'{name}.npz'
        np.savez_compressed(path, **arrays)
        reports[name] = dict(**hashes, result_sha256=digest(path), modes=modes)
        if fault_comparison is not None:
            reports[name]['fault_window'] = fault_comparison
        plots[name] = (observations['available_time'].copy(),
                       {label: values.copy() for label, values in errors.items()})
        print(name, 'done', flush=True)

    fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
    for axis, name in zip(axes.flat, CASES22):
        available, errors = plots[name]
        for label, values in errors.items():
            axis.plot(available, values, label=label)
        axis.axhline(20., color='gray', linestyle=':', linewidth=1)
        axis.set(title=name, xlabel='Output available time (s)',
                 ylabel='Published error (m)')
        axis.grid(alpha=.3)
    axes[0, 0].legend()
    fig.savefig(OUTPUT / 'overview.png', dpi=150)
    plt.close(fig)
    assert source_hashes() == sources
    (OUTPUT / 'summary.json').write_text(json.dumps(serializable(dict(
        scope=__doc__, source_sha256=sources,
        source_snapshot_sha256=digest(OUTPUT / 'source_snapshot.zip'),
        manifest_sha256=digest(OUTPUT / 'manifest.json'),
        figure_sha256=digest(OUTPUT / 'overview.png'), cases=reports)),
        indent=2, allow_nan=False), encoding='utf-8')
    return reports


if __name__ == '__main__':
    main()
