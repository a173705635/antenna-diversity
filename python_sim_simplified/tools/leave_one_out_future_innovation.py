"""Current carrier leave-one-out innovation versus a fixed future PRN error.

Nine saved Stage 47 gate-3 short navigations are reproduced in memory. Current
scores use only observations available at k; truth enters the later score.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from core.gps_simulation import receiver_reference
from core.single_source import navigate
from experiments import single_source_projection_long as study
from experiments.single_source_coherent_navigation import quality_support_counts
from experiments.single_source_phase_state_gate import projection_eligibility
from experiments.single_source_suppression_study import digest
from experiments.single_source_temporal_trigger import navigation_input
from tools import single_path_carrier_continuity as replay


STAGE47_SUMMARY_SHA256 = replay.STAGE47_SUMMARY_SHA256
CONTINUITY_SUMMARY_SHA256 = 'f2c6004241a8891145b5d9cb4f88d30b29377b320a57b35dc710a62392f85166'
CURRENT_START_S = 10.
FUTURE_START_EPOCHS = 2
FUTURE_END_EPOCHS = 3
GEOMETRY_CONDITION_LIMIT = 1000.
MIN_COMMON_STARS = 6


def pair_mask(z, selected, first, last):
    """Original short qualification and finite carriers on one same arc."""
    return (selected[first] & selected[last]
            & np.isfinite(z['carrier'][first])
            & np.isfinite(z['carrier'][last])
            & (z['arc'][first] == z['arc'][last]))


def pair_measurement(z, first, last, mask):
    """Original local H and geometry-corrected carrier change in metres."""
    reference, rotation = receiver_reference()
    current = z['satellites'][last, mask]-reference
    previous = z['satellites'][first, mask]-reference
    current_range = np.linalg.norm(current, axis=1)
    previous_range = np.linalg.norm(previous, axis=1)
    a = -(current/current_range[:, None])@rotation.T
    h = np.column_stack((a, np.ones(len(a))))
    y = (z['carrier'][last, mask]-z['carrier'][first, mask]
         -(current_range-previous_range))
    return h, y, float(z['time'][last]-z['time'][first])


def leave_one_out_scores(h, y, dt):
    """Unscreened residual innovation; only original geometry gate applies."""
    score = np.full(len(y), np.nan)
    for i in range(len(y)):
        retained = np.arange(len(y)) != i
        other = h[retained]
        if np.linalg.matrix_rank(other) < 4 or np.linalg.cond(other) > GEOMETRY_CONDITION_LIMIT:
            continue
        row = h[i]
        predicted = row@np.linalg.pinv(other)@y[retained]
        leverage = row@np.linalg.pinv(other.T@other)@row
        score[i] = abs(y[i]-predicted)/(dt*np.sqrt(1.+leverage))
    return score


def current_scores(z, selected, k):
    """Causal score and its actually scoreable PRNs at current observation k."""
    scores = np.full(selected.shape[1], np.nan)
    if k == 0 or abs((z['time'][k]-z['time'][k-1])-.01) > 1e-9:
        return scores, 'current_time_gap', 0
    common = pair_mask(z, selected, k-1, k)
    if common.sum() < MIN_COMMON_STARS:
        return scores, 'current_fewer_than_six', int(common.sum())
    h, y, dt = pair_measurement(z, k-1, k, common)
    scores[common] = leave_one_out_scores(h, y, dt)
    if np.isfinite(scores).sum() < MIN_COMMON_STARS:
        return scores, 'current_geometry', int(common.sum())
    return scores, 'current_scored', int(common.sum())


def future_truth_error(z, selected, k, scoreable, velocity_east):
    """Truth-referenced future differential carrier rate, only for scoring."""
    first, last = k+FUTURE_START_EPOCHS, k+FUTURE_END_EPOCHS
    target = np.full(selected.shape[1], np.nan)
    if last >= len(z['time']):
        return target, 'future_out_of_record', 0
    if abs((z['time'][last]-z['time'][first])-.01) > 1e-9:
        return target, 'future_time_gap', 0
    common = scoreable & pair_mask(z, selected, first, last)
    if common.sum() < MIN_COMMON_STARS:
        return target, 'future_fewer_than_six', int(common.sum())
    h, y, dt = pair_measurement(z, first, last, common)
    true_displacement = np.array([velocity_east*dt, 0., 0., 0.])
    rate = (y-h@true_displacement)/dt
    target[common] = abs(rate-rate.mean())
    return target, 'future_scored', int(common.sum())


def rank_correlation(current, future):
    """Average ranks for ties; constants have no defined correlation."""
    present = np.isfinite(current) & np.isfinite(future)
    if present.sum() < MIN_COMMON_STARS:
        return None
    x = rankdata(current[present], method='average')
    y = rankdata(future[present], method='average')
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def computed_star_summary(counts, computed):
    """Keep an uncomputed row distinct from a computed zero-star row."""
    if not computed.any():
        return None
    return [float(f(counts[computed])) for f in (np.min, np.median, np.max)]


def analyze_observations(z, selected, velocity_east):
    """Fix every opportunity before scores, then report explicit missing causes."""
    n, stars = selected.shape
    current = np.full((n, stars), np.nan)
    future = np.full_like(current, np.nan)
    rho = np.full(n, np.nan)
    reason = np.full(n, 'outside_window', dtype='<U32')
    current_count = np.zeros(n, int)
    scoreable_count = np.zeros(n, int)
    future_count = np.zeros(n, int)
    current_pair_computed = np.zeros(n, bool)
    current_score_computed = np.zeros(n, bool)
    future_common_computed = np.zeros(n, bool)
    future_truth_computed = np.zeros(n, bool)
    candidates = np.flatnonzero(z['time'] >= CURRENT_START_S)
    for k in candidates:
        if k+FUTURE_END_EPOCHS >= n:
            reason[k] = 'future_out_of_record'
            continue
        current[k], status, current_count[k] = current_scores(z, selected, k)
        current_pair_computed[k] = status != 'current_time_gap'
        current_score_computed[k] = status in ('current_geometry', 'current_scored')
        if current_score_computed[k]:
            scoreable_count[k] = int(np.isfinite(current[k]).sum())
        if status != 'current_scored':
            reason[k] = status
            continue
        future[k], status, future_count[k] = future_truth_error(
            z, selected, k, np.isfinite(current[k]), velocity_east)
        future_common_computed[k] = status in ('future_fewer_than_six',
                                               'future_scored')
        future_truth_computed[k] = status == 'future_scored'
        if status != 'future_scored':
            reason[k] = status
            continue
        result = rank_correlation(current[k], future[k])
        if result is None:
            reason[k] = 'constant_rank'
        else:
            reason[k] = 'scored'
            rho[k] = result
    valid = np.isfinite(rho[candidates])
    values = rho[candidates][valid]
    names, counts = np.unique(reason[candidates], return_counts=True)
    summary = dict(candidate_opportunities=len(candidates),
                   primary_opportunities=int((candidates+FUTURE_END_EPOCHS < n).sum()),
                   reasons={str(name): int(count) for name, count in zip(names, counts)},
                   valid_correlations=len(values),
                   median_spearman=None if not len(values) else float(np.median(values)),
                   positive_fraction=None if not len(values) else float(np.mean(values > 0.)),
                   current_star_min_median_max=computed_star_summary(
                       current_count, current_pair_computed),
                   scoreable_star_min_median_max=computed_star_summary(
                       scoreable_count, current_score_computed),
                   future_common_min_median_max=computed_star_summary(
                       future_count, future_common_computed),
                   current_pair_computed=int(current_pair_computed.sum()),
                   current_score_computed=int(current_score_computed.sum()),
                   future_common_computed=int(future_common_computed.sum()),
                   future_truth_computed=int(future_truth_computed.sum()))
    arrays = dict(observation_time=z['time'], current_score_mps=current,
                  future_truth_error_mps=future, spearman=rho, reason=reason,
                  current_pair_stars=current_count,
                  current_scoreable_stars=scoreable_count,
                  future_common_stars=future_count,
                  current_pair_computed=current_pair_computed,
                  current_score_computed=current_score_computed,
                  future_common_computed=future_common_computed,
                  future_truth_computed=future_truth_computed)
    return summary, arrays


def analyze_path(folder, hashes, velocity_east):
    inputs = {name: digest(folder/name) for name in replay.SCIENCE}
    assert inputs == {name: hashes[name] for name in replay.SCIENCE}
    tracking = replay.read_npz(folder/'tracking.npz')
    trace = replay.read_npz(folder/'trace.npz')
    support = replay.read_npz(folder/'support.npz')['coherent_support']
    saved = replay.read_npz(folder/'navigation.npz')
    eligible, latest, recent = projection_eligibility(tracking['available_time'], trace)
    np.testing.assert_array_equal(eligible, saved['phase_state_eligible'])
    np.testing.assert_array_equal(latest, saved['latest_complete_block'])
    np.testing.assert_array_equal(recent, saved['recent_block_dimensions'])
    z = navigation_input(tracking, support)
    baseline = navigate(z, **replay.short_options(eligible))
    replay.verify_baseline(saved, baseline, quality_support_counts(tracking, support))
    np.testing.assert_array_equal(baseline['selected'], saved['short_selected'])
    summary, arrays = analyze_observations(z, baseline['selected'], velocity_east)
    return dict(input_sha256=inputs, baseline_reproduced=True, **summary), arrays


def analyze():
    assert digest(study.OUTPUT/'summary.json') == STAGE47_SUMMARY_SHA256
    continuity_path = study.OUTPUT/'single_path_carrier_continuity_summary.json'
    assert digest(continuity_path) == CONTINUITY_SUMMARY_SHA256
    continuity = json.loads(continuity_path.read_text(encoding='utf-8'))
    frozen = json.loads((study.OUTPUT/'summary.json').read_text(encoding='utf-8'))
    root = Path(__file__).resolve().parents[1]
    assert digest(root/'tools/single_path_carrier_continuity.py') == continuity['source_sha256']
    for name in replay.NAVIGATION_SOURCES:
        assert digest(root/name) == frozen['source_sha256'][name]
    cases, arrays = {}, {}
    for seed in study.SEEDS:
        cases[str(seed)] = {}
        for scene in replay.SCENES:
            case = frozen['cases'][str(seed)][scene]
            folder = study.OUTPUT/str(seed)/scene/'gate3'
            metadata = study.OUTPUT/str(seed)/scene/'input'/'metadata.json'
            assert digest(metadata) == case['input_sha256']['metadata.json']
            velocity = json.loads(metadata.read_text(encoding='utf-8'))['case']['velocity_east']
            result, detail = analyze_path(
                folder, case['paths']['gate3']['output_sha256'], velocity)
            result['metadata_sha256'] = case['input_sha256']['metadata.json']
            cases[str(seed)][scene] = result
            arrays[f'{seed}_{scene}'] = detail
    weak = [cases[str(seed)]['weak'] for seed in study.SEEDS]
    supported = all(row['valid_correlations'] > 0
                    and row['median_spearman'] > 0 for row in weak)
    return dict(scope=__doc__, source_sha256=digest(Path(__file__)),
                stage47_summary_sha256=STAGE47_SUMMARY_SHA256,
                continuity_summary_sha256=CONTINUITY_SUMMARY_SHA256,
                continuity_source_sha256=continuity['source_sha256'],
                frozen_navigation_source_sha256={name: frozen['source_sha256'][name]
                                                 for name in replay.NAVIGATION_SOURCES},
                current_start_observation_s=CURRENT_START_S,
                future_start_epochs=FUTURE_START_EPOCHS,
                future_end_epochs=FUTURE_END_EPOCHS,
                weak_median_sign_rule_passed=bool(supported),
                cases=cases), arrays


def main():
    summary, arrays = analyze()
    output = study.OUTPUT
    summary_path = output/'leave_one_out_future_innovation_summary.json'
    manifest_path = output/'leave_one_out_future_innovation_manifest.json'
    assert not summary_path.exists() and not manifest_path.exists()
    paths = {name: output/f'leave_one_out_future_innovation_{name}.npz'
             for name in arrays}
    assert all(not path.exists() for path in paths.values())
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False),
                            encoding='utf-8')
    for name, path in paths.items():
        np.savez_compressed(path, **arrays[name])
    hashes = {summary_path.name: digest(summary_path)}
    hashes.update({path.name: digest(path) for path in paths.values()})
    manifest_path.write_text(json.dumps(hashes, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
