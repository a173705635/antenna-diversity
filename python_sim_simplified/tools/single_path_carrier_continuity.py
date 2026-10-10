"""Retrospective three-epoch carrier admission in nine saved Stage 47 paths.

Both short navigations start empty at the beginning of each saved record.
Only the candidate's phase-bridge current-row admission is changed; no RF is
run and truth is used only by the existing post-navigation long screen.
"""
import json
from pathlib import Path
from types import FunctionType

import numpy as np

from core import single_source
from experiments import single_source_projection_long as study
from experiments.single_source_coherent_navigation import quality_support_counts
from experiments.single_source_phase_state_gate import projection_eligibility
from experiments.single_source_startup_amplitude_rf import score_navigation
from experiments.single_source_suppression_study import digest
from experiments.single_source_temporal_trigger import navigation_input


STAGE47_SUMMARY_SHA256 = '47a73e338bf086f8c0418f31c3db07081dea299129775be54b45d22d206e5a86'
SCENES = ('h0_drop', 'weak', 'genuine_absent')
SCIENCE = ('tracking.npz', 'trace.npz', 'support.npz', 'navigation.npz')
NAVIGATION_SOURCES = (
    'core/single_source.py', 'core/mechanism_phase.py',
    'core/mechanism_smoothing.py', 'core/recovery_geometry.py',
    'core/gps_simulation.py', 'core/positioning.py',
    'experiments/single_source_coherent_navigation.py',
    'experiments/single_source_phase_state_gate.py',
    'experiments/single_source_temporal_trigger.py',
    'experiments/single_source_projection_threshold.py',
    'experiments/single_source_projection_long.py',
    'experiments/single_source_startup_amplitude_rf.py',
    'experiments/single_source_nav_bit_epoch_shift.py',
    'experiments/single_source_code_blend.py',
)


def read_npz(path):
    with np.load(path) as saved:
        return dict(saved)


def three_epoch_mask(z, valid, k):
    """Stars qualified at k-2:k, with uninterrupted 10 ms carrier arcs."""
    admitted = np.zeros(valid.shape[1], bool)
    if k < 2:
        return admitted
    time = z['time']
    if (abs((time[k]-time[k-1])-.01) > 1e-9
            or abs((time[k-1]-time[k-2])-.01) > 1e-9):
        return admitted
    admitted = (valid[k] & valid[k-1] & valid[k-2]
                & np.isfinite(z['carrier'][k])
                & np.isfinite(z['carrier'][k-1])
                & np.isfinite(z['carrier'][k-2])
                & (z['arc'][k] == z['arc'][k-1])
                & (z['arc'][k-1] == z['arc'][k-2]))
    return admitted


def candidate_navigate(z, eligible):
    """Bind one replacement lookup into a private copy of navigate globals."""
    admission = np.zeros_like(z['carrier'], dtype=bool)
    called = np.zeros(len(z['time']), bool)
    failed = np.zeros(len(z['time']), bool)
    original_bridge = single_source.phase_prediction_leave_one_out

    def phase_bridge(observations, valid, positions, k):
        allowed = three_epoch_mask(observations, valid, k)
        admission[k] = allowed
        called[k] = True
        bridge_valid = valid.copy()
        bridge_valid[k] = allowed
        prediction, details = original_bridge(observations, bridge_valid, positions, k)
        failed[k] = prediction is None
        return prediction, details

    private_globals = dict(single_source.navigate.__globals__)
    private_globals['phase_prediction_leave_one_out'] = phase_bridge
    navigate = FunctionType(single_source.navigate.__code__, private_globals,
                            single_source.navigate.__name__,
                            single_source.navigate.__defaults__,
                            single_source.navigate.__closure__)
    result = navigate(z, diagnostics=True, code_blend=0.,
                      phase_leave_one_out=True, phase_state_continuity=True,
                      phase_state_eligible=eligible)
    return result, dict(admission=admission, called=called, failed=failed)


def short_options(eligible):
    return dict(diagnostics=True, code_blend=0., phase_leave_one_out=True,
                phase_state_continuity=True, phase_state_eligible=eligible)


def verify_baseline(saved, baseline, support_count):
    """Compare all saved short fields, including every diagnostic array."""
    actual = {f'short_{key}': np.nan if value is None else value
              for key, value in baseline.items() if key != 'diagnostics'}
    actual.update({f'short_diagnostic_{key}': value
                   for key, value in baseline['diagnostics'].items()})
    actual['short_support_count'] = support_count
    expected = {key: value for key, value in saved.items()
                if key.startswith('short_')}
    assert actual.keys() == expected.keys()
    for key, value in actual.items():
        np.testing.assert_array_equal(value, expected[key], err_msg=key)


def event_times(time, available, navigation):
    """Pair isolation episodes; confirmations are attempts, not exits."""
    diagnostic = navigation['diagnostics']
    isolated = diagnostic['isolated']
    starts = np.flatnonzero(isolated & ~np.r_[False, isolated[:-1]])
    exits = np.flatnonzero(~isolated & np.r_[False, isolated[:-1]])
    confirmed = diagnostic['recovery_confirmed']
    episodes = []
    for start in starts:
        later = exits[exits > start]
        end = int(later[0]) if len(later) else None
        stop = len(time) if end is None else end+1
        attempts = np.flatnonzero(confirmed[start:stop])+start
        episodes.append(dict(start_index=int(start),
                             start_observation_s=float(time[start]),
                             start_available_s=float(available[start]),
                             confirmation_attempt_available_s=available[attempts].tolist(),
                             exit_index=end,
                             exit_available_s=None if end is None else float(available[end]),
                             elapsed_s=float(available[-1 if end is None else end]
                                             -available[start]),
                             unresolved_at_record_end=end is None))
    return dict(initialization_available_s=navigation['initialization_time'],
                isolation_episodes=episodes)


def time_axis_counts(time, available, baseline, candidate):
    """Expose both inherited scoring clocks around the 0.3 s boundary."""
    old = np.isfinite(baseline).all(axis=1)
    new = np.isfinite(candidate).all(axis=1)
    result = {}
    for name, opportunity in (('score_navigation_available', available >= .3),
                              ('long_screen_observation', time >= .3)):
        result[name] = dict(opportunities=int(opportunity.sum()),
                            baseline_published=int((opportunity & old).sum()),
                            candidate_published=int((opportunity & new).sum()),
                            common_published=int((opportunity & old & new).sum()))
    return result


def run_path(folder, expected_hashes, scene, velocity_east):
    input_hashes = {name: digest(folder/name) for name in SCIENCE}
    for name, sha in input_hashes.items():
        assert sha == expected_hashes[name], (folder, name)
    tracking = read_npz(folder/'tracking.npz')
    trace = read_npz(folder/'trace.npz')
    support = read_npz(folder/'support.npz')['coherent_support']
    saved = read_npz(folder/'navigation.npz')
    eligible, latest, recent = projection_eligibility(tracking['available_time'], trace)
    np.testing.assert_array_equal(eligible, saved['phase_state_eligible'])
    np.testing.assert_array_equal(latest, saved['latest_complete_block'])
    np.testing.assert_array_equal(recent, saved['recent_block_dimensions'])
    z = navigation_input(tracking, support)
    baseline = single_source.navigate(z, **short_options(eligible))
    verify_baseline(saved, baseline, quality_support_counts(tracking, support))
    candidate, bridge = candidate_navigate(z, eligible)
    baseline_score = score_navigation(tracking, baseline, velocity_east)
    candidate_score = score_navigation(tracking, candidate, velocity_east)
    time = tracking['time']
    screen = study.long_screen(
        scene, time, baseline['position'], candidate['position'], velocity_east,
        baseline_score['max_relative_motion_error_m'],
        candidate_score['max_relative_motion_error_m'])
    result = dict(input_sha256=input_hashes, baseline_reproduced=True,
                  baseline=baseline_score, candidate=candidate_score,
                  long_screen=screen,
                  evaluation_time_axes=time_axis_counts(
                      time, tracking['available_time'], baseline['position'],
                      candidate['position']),
                  bridge=dict(called=int(bridge['called'].sum()),
                              failed=int(bridge['failed'].sum()),
                              no_admission=int((bridge['called']
                                                & ~bridge['admission'].any(axis=1)).sum()),
                              failed_with_admission=int((bridge['failed']
                                                         & bridge['admission'].any(axis=1)).sum()),
                              admitted_star_min_median_max=[float(f(
                                  bridge['admission'][bridge['called']].sum(axis=1)))
                                  for f in (np.min, np.median, np.max)]),
                  baseline_events=event_times(time, tracking['available_time'], baseline),
                  candidate_events=event_times(time, tracking['available_time'], candidate))
    arrays = dict(time=time, baseline_position=baseline['position'],
                  candidate_position=candidate['position'],
                  bridge_admission=bridge['admission'],
                  bridge_called=bridge['called'], bridge_failed=bridge['failed'])
    return result, arrays


def analyze():
    assert digest(study.OUTPUT/'summary.json') == STAGE47_SUMMARY_SHA256
    frozen = json.loads((study.OUTPUT/'summary.json').read_text(encoding='utf-8'))
    root = Path(__file__).resolve().parents[1]
    for name in NAVIGATION_SOURCES:
        assert digest(root/name) == frozen['source_sha256'][name], name
    results, arrays = {}, {}
    for seed in study.SEEDS:
        results[str(seed)] = {}
        for scene in SCENES:
            path = frozen['cases'][str(seed)][scene]
            folder = study.OUTPUT/str(seed)/scene/'gate3'
            metadata_path = study.OUTPUT/str(seed)/scene/'input'/'metadata.json'
            assert digest(metadata_path) == path['input_sha256']['metadata.json']
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            result, detail = run_path(folder,
                                      path['paths']['gate3']['output_sha256'],
                                      scene, metadata['case']['velocity_east'])
            result['metadata_sha256'] = path['input_sha256']['metadata.json']
            results[str(seed)][scene] = result
            arrays[f'{seed}_{scene}'] = detail
    return dict(scope=__doc__, source_sha256=digest(Path(__file__)),
                stage47_summary_sha256=STAGE47_SUMMARY_SHA256,
                frozen_navigation_source_sha256={name: frozen['source_sha256'][name]
                                                 for name in NAVIGATION_SOURCES},
                screen_pass=all(result['long_screen']['pass']
                                for cases in results.values()
                                for result in cases.values()),
                cases=results), arrays


def main():
    summary, arrays = analyze()
    destination = study.OUTPUT/'single_path_carrier_continuity_summary.json'
    assert not destination.exists()
    destination.write_text(json.dumps(summary, indent=2, allow_nan=False),
                           encoding='utf-8')
    for key, detail in arrays.items():
        np.savez_compressed(study.OUTPUT/f'single_path_carrier_continuity_{key}.npz',
                            **detail)


if __name__ == '__main__':
    main()
