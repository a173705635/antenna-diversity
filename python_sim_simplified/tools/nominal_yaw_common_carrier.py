"""Offline common-star carrier replay of the saved Stage 47/48 weak paths.

This diagnostic changes only the eligible PRN list for a one-epoch carrier
fit. It neither reruns navigation nor uses truth to choose satellites.
"""
import json
from pathlib import Path

import numpy as np

from core.mechanism_phase import fit_carrier_displacement
from experiments import single_source_nominal_yaw as study
from experiments.single_source_suppression_study import digest
from tools import nominal_yaw_mechanism as mechanism


MECHANISM_SUMMARY_SHA256 = '6c1ffd94111852302a6c8fda2575a2ad1fff67e645ac9b12a8c0eca23d1d62e3'
OUTPUT_NAME = 'common_carrier_summary.json'


def pair_mask(tracking, selected, k):
    """Selected finite carriers on this path's same arc at k and k-1."""
    return (selected[k] & selected[k-1]
            & np.isfinite(tracking['carrier'][k])
            & np.isfinite(tracking['carrier'][k-1])
            & (tracking['arc'][k] == tracking['arc'][k-1]))


def pair_table(tracking, selected):
    """One row per adjacent step, including both epochs' carrier conditions."""
    pairs = np.zeros_like(selected)
    for k in range(1, len(pairs)):
        pairs[k] = pair_mask(tracking, selected, k)
    return pairs


def replay(tracking, navigation, eligible_pairs, start):
    """Accumulate east/north/up displacement from the saved start position.

    A failed fit ends this continuous trajectory. Later epochs remain missing;
    no displacement is carried across an unobserved step.
    """
    count = len(tracking['time'])
    position = np.full((count, 3), np.nan)
    stars = np.zeros(count, int)
    passed = np.zeros(count, bool)
    attempted = np.zeros(count, bool)
    position[start] = navigation['short_position'][start]
    for k in range(start+1, count):
        common = eligible_pairs[k]
        stars[k] = int(common.sum())
        if not np.isfinite(position[k-1]).all():
            continue
        attempted[k] = True
        if stars[k] < 5:
            continue
        displacement = fit_carrier_displacement(
            tracking['satellites'][k, common],
            tracking['satellites'][k-1, common],
            tracking['carrier'][k, common]-tracking['carrier'][k-1, common])
        if displacement is not None:
            position[k] = position[k-1]+displacement[:3]
            passed[k] = True
    return dict(position=position, stars=stars, passed=passed, attempted=attempted)


def horizontal_metrics(position, time, start):
    """Score positions against the trajectory only after the replay is done."""
    truth = np.column_stack((2.*time, np.zeros(len(time))))
    valid = np.isfinite(position).all(axis=1)
    error = np.linalg.norm(position[valid, :2]-truth[valid], axis=1)
    drift = ((position[valid, :2]-position[start, :2])
             -(truth[valid]-truth[start]))
    drift_norm = np.linalg.norm(drift, axis=1)
    return dict(count=int(valid.sum()),
                error_max_m=float(np.max(error)),
                error_p95_m=float(np.percentile(error, 95)),
                anchor_drift_max_m=float(np.max(drift_norm)),
                anchor_drift_p95_m=float(np.percentile(drift_norm, 95)))


def compare(tracks, navigations):
    """Replay each yaw with its own carriers, changing only the PRN set."""
    time = tracks[0.]['time']
    for yaw in study.YAWS_DEG:
        np.testing.assert_array_equal(tracks[yaw]['time'], time)
    start = int(np.flatnonzero(time >= .3)[0])
    baseline_pairs = {yaw: pair_table(tracks[yaw], navigations[yaw]['short_selected'])
                      for yaw in study.YAWS_DEG}
    shared_pairs = np.logical_and.reduce(
        [baseline_pairs[yaw] for yaw in study.YAWS_DEG])
    summary = dict(start_index=start, start_time_s=float(time[start]), paths={})
    arrays = dict(observation_time_s=time, common_eligible_pairs=shared_pairs)
    for yaw in study.YAWS_DEG:
        tracking, navigation = tracks[yaw], navigations[yaw]
        baseline = replay(tracking, navigation, baseline_pairs[yaw], start)
        candidate = replay(tracking, navigation, shared_pairs, start)
        published = navigation['short_position']
        reconstructed = np.isfinite(baseline['position']).all(axis=1)
        difference = baseline['position'][reconstructed]-published[reconstructed]
        error = float(np.max(np.abs(difference)))
        assert error <= 1e-12
        assert reconstructed[start:].all()
        delta = candidate['position'][:, :2]-baseline['position'][:, :2]
        comparable = np.isfinite(delta).all(axis=1)
        key = f'{yaw:+g}'
        summary['paths'][key] = dict(
            baseline_reconstruction_max_abs_m=error,
            baseline=horizontal_metrics(baseline['position'], time, start),
            candidate=horizontal_metrics(candidate['position'], time, start),
            candidate_minus_baseline_max_east_north_m=(
                np.max(np.abs(delta[comparable]), axis=0).tolist()),
            common_epochs=int(comparable.sum()),
            baseline_failed_steps=int((baseline['attempted'] & ~baseline['passed']).sum()),
            candidate_failed_steps=int((candidate['attempted'] & ~candidate['passed']).sum()),
            baseline_missing_steps=int((~baseline['attempted'][start+1:]).sum()),
            candidate_missing_steps=int((~candidate['attempted'][start+1:]).sum()),
            baseline_first_gap_index=int(np.flatnonzero(~baseline['passed'][start+1:])[0]+start+1)
            if not baseline['passed'][start+1:].all() else None,
            candidate_first_gap_index=int(np.flatnonzero(~candidate['passed'][start+1:])[0]+start+1)
            if not candidate['passed'][start+1:].all() else None,
            baseline_star_min_median_max=[float(f(baseline['stars'][start+1:]))
                                          for f in (np.min, np.median, np.max)],
            candidate_star_min_median_max=[float(f(candidate['stars'][start+1:]))
                                           for f in (np.min, np.median, np.max)])
        arrays[f'{key}_candidate_minus_baseline_east_north_m'] = delta
        for name, replayed in (('baseline', baseline), ('candidate', candidate)):
            for field, value in replayed.items():
                arrays[f'{key}_{name}_{field}'] = value
            arrays[f'{key}_{name}_east_north_displacement_m'] = (
                replayed['position'][:, :2]-replayed['position'][start, :2])
            arrays[f'{key}_{name}_east_north_truth_error_m'] = (
                replayed['position'][:, :2]
                - np.column_stack((2.*time, np.zeros(len(time)))))
    return summary, arrays


def analyze():
    assert digest(study.OUTPUT/'recovery_summary.json') == mechanism.STAGE48_SUMMARY_SHA256
    assert digest(study.OUTPUT/'mechanism_summary.json') == MECHANISM_SUMMARY_SHA256
    assert digest(mechanism.STAGE47/'summary.json') == study.STAGE47_SUMMARY_SHA256
    stage47 = json.loads((mechanism.STAGE47/'summary.json').read_text(encoding='utf-8'))
    stage48 = json.loads((study.OUTPUT/'recovery_summary.json').read_text(encoding='utf-8'))
    assert all(stage48['frozen_source_sha256'][name] == sha
               for name, sha in stage47['source_sha256'].items())
    for source in ('core/mechanism_phase.py', 'core/gps_simulation.py'):
        assert digest(Path(__file__).resolve().parents[1]/source) == stage47['source_sha256'][source]
    tracks, navigations, inputs = {}, {}, {}
    for yaw in study.YAWS_DEG:
        folder, nav_name = mechanism.verified_path('weak', yaw, stage47, stage48)
        tracks[yaw] = mechanism.read_npz(folder/'tracking.npz')
        navigations[yaw] = mechanism.read_npz(folder/nav_name)
        inputs[f'{yaw:+g}'] = {name: digest(folder/name)
                               for name in ('tracking.npz', nav_name)}
        assert np.all(navigations[yaw]['short_diagnostic_phase_ok'][
            tracks[yaw]['time'] >= .3])
        assert np.all(navigations[yaw]['short_diagnostic_phase_lag_epochs'][
            tracks[yaw]['time'] >= .3] == 1)
        assert np.all(navigations[yaw]['short_diagnostic_phase_excluded_prn'][
            tracks[yaw]['time'] >= .3] == 0)
        assert not np.any(navigations[yaw]['short_diagnostic_recovery_confirmed'])
        assert np.isfinite(navigations[yaw]['short_position'][tracks[yaw]['time'] >= .3]).all()
    summary, arrays = compare(tracks, navigations)
    summary.update(scope=__doc__, tool_source_sha256=digest(Path(__file__)),
                   carrier_fit_source_sha256=digest(Path(fit_carrier_displacement.__code__.co_filename)),
                   recovery_summary_sha256=mechanism.STAGE48_SUMMARY_SHA256,
                   mechanism_summary_sha256=MECHANISM_SUMMARY_SHA256,
                   stage47_summary_sha256=study.STAGE47_SUMMARY_SHA256,
                   input_sha256=inputs)
    return summary, arrays


def main():
    summary, arrays = analyze()
    destination = study.OUTPUT/OUTPUT_NAME
    assert not destination.exists()
    assert not (study.OUTPUT/'common_carrier_arrays.npz').exists()
    destination.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    np.savez_compressed(study.OUTPUT/'common_carrier_arrays.npz', **arrays)


if __name__ == '__main__':
    main()
