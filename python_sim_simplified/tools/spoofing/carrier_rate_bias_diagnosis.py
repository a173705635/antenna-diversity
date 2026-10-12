"""Fixed PRN carrier-rate bias: fit first half, predict second half offline.

This post-navigation mechanism check reads nine saved Stage 47 gate-3 paths.
Truth is used only after both original navigations have been reproduced.
"""
from core.provenance import current_path_for_historical_source, ROOT
import json
from pathlib import Path
from types import FunctionType

import numpy as np

from core.spoofing import single_source
from core.gps_simulation import receiver_reference
from experiments.spoofing import single_source_projection_long as study
from experiments.spoofing.single_source_coherent_navigation import quality_support_counts
from experiments.spoofing.single_source_phase_state_gate import projection_eligibility
from experiments.spoofing.single_source_suppression_study import digest
from experiments.spoofing.single_source_temporal_trigger import navigation_input
from tools.spoofing import single_path_carrier_continuity as replay


STAGE47_SUMMARY_SHA256 = replay.STAGE47_SUMMARY_SHA256
CONTINUITY_SUMMARY_SHA256 = 'f2c6004241a8891145b5d9cb4f88d30b29377b320a57b35dc710a62392f85166'
SPLIT_S = 10.
START_S = .3


def final_bridge_mask(z, valid, k, details):
    """Actual successful lag's common carriers, after unique exclusion."""
    lag = int(details['lag_epochs'])
    if lag == 0:
        return np.zeros(valid.shape[1], bool)
    j = k-lag
    mask = (valid[k] & valid[j]
            & np.isfinite(z['carrier'][k]) & np.isfinite(z['carrier'][j])
            & (z['arc'][k] == z['arc'][j]))
    excluded = int(details['excluded_prn'])
    if excluded:
        mask[excluded-1] = False
        assert mask.sum() == details['candidate_common_count']
    else:
        assert mask.sum() == details['full_common_count'][lag-1]
    return mask


def recorded_navigate(z, eligible, continuity=False):
    """Record the original bridge's final stars without mutating its globals."""
    records = []
    admission = np.zeros_like(z['carrier'], dtype=bool)
    called = np.zeros(len(z['time']), bool)
    failed = np.zeros(len(z['time']), bool)
    original = single_source.phase_prediction_leave_one_out

    def observed_bridge(observations, valid, positions, k):
        bridge_valid = valid
        if continuity:
            admission[k] = replay.three_epoch_mask(observations, valid, k)
            bridge_valid = valid.copy()
            bridge_valid[k] = admission[k]
        called[k] = True
        prediction, details = original(observations, bridge_valid, positions, k)
        failed[k] = prediction is None
        if prediction is not None:
            lag = int(details['lag_epochs'])
            j = k-lag
            records.append(dict(k=k, parent=j, lag=lag,
                                excluded_prn=int(details['excluded_prn']),
                                mask=final_bridge_mask(observations, bridge_valid,
                                                       k, details),
                                parent_position=positions[j].copy(),
                                prediction=prediction.copy()))
        return prediction, details

    private_globals = dict(single_source.navigate.__globals__)
    private_globals['phase_prediction_leave_one_out'] = observed_bridge
    navigate = FunctionType(single_source.navigate.__code__, private_globals,
                            single_source.navigate.__name__,
                            single_source.navigate.__defaults__,
                            single_source.navigate.__closure__)
    result = navigate(z, **replay.short_options(eligible))
    for record in records:
        k = record['k']
        np.testing.assert_array_equal(result['candidate_position'][k],
                                      record['prediction'])
        assert result['diagnostics']['phase_lag_epochs'][k] == record['lag']
        assert result['diagnostics']['phase_excluded_prn'][k] == record['excluded_prn']
    return result, records, dict(admission=admission, called=called, failed=failed)


def bridge_physics(z, record, velocity_east):
    """Measured b in m/s for the actual fitted stars and parent step."""
    k, j, mask = record['k'], record['parent'], record['mask']
    reference, rotation = receiver_reference()
    current = z['satellites'][k, mask]-reference
    previous = z['satellites'][j, mask]-reference
    current_range = np.linalg.norm(current, axis=1)
    previous_range = np.linalg.norm(previous, axis=1)
    a = -(current/current_range[:, None])@rotation.T
    h = np.column_stack((a, np.ones(len(a))))
    y = (z['carrier'][k, mask]-z['carrier'][j, mask]
         -(current_range-previous_range))
    dt = float(z['time'][k]-z['time'][j])
    truth_displacement = np.array([velocity_east*dt, 0., 0.])
    b = (y-a@truth_displacement)/dt
    observed_en = (record['prediction']-record['parent_position'])[:2]
    observed_en -= truth_displacement[:2]
    np.testing.assert_allclose(observed_en,
                               dt*(np.linalg.pinv(h)@b)[:2], rtol=0, atol=1e-8)
    return dict(**record, dt=dt, h=h, b=b, observed_en=observed_en)


def fit_fixed_bias(training, stars):
    """Eliminate one free common rate per bridge; require rank P-1."""
    support = np.flatnonzero(np.logical_or.reduce([step['mask']
                                                     for step in training]))
    size = len(support)
    columns = {star: index for index, star in enumerate(support)}
    design, measured = [], []
    for step in training:
        indices = np.flatnonzero(step['mask'])
        rows = np.zeros((len(indices), size))
        rows[np.arange(len(indices)), [columns[star] for star in indices]] = 1.
        design.append(rows-rows.mean(axis=0))
        measured.append(step['b']-step['b'].mean())
    matrix = np.vstack(design)
    target = np.concatenate(measured)
    rank = int(np.linalg.matrix_rank(matrix))
    report = dict(training_bridges=len(training), support_prns=(support+1).tolist(),
                  support_count=size, design_rank=rank,
                  required_rank=size-1)
    if rank != size-1:
        return dict(report, status='unidentifiable'), None
    beta = np.linalg.lstsq(matrix, target, rcond=None)[0]
    assert abs(beta.sum()) < 1e-10
    full = np.full(stars, np.nan)
    full[support] = beta
    return dict(report, status='identified',
                beta_mps=[None if np.isnan(value) else float(value)
                          for value in full],
                beta_sum_mps=float(beta.sum())), full


def evaluate_bridges(steps, beta, time):
    """Score per-bridge prediction and cumulative EN on actual parent edges."""
    validation = [step for step in steps if time[step['k']] >= SPLIT_S]
    stars = len(beta)
    empty = dict(k=np.empty(0, int), parent=np.empty(0, int),
                 mask=np.empty((0, stars), bool),
                 per_prn_en_m=np.empty((0, stars, 2)),
                 observed_en_m=np.empty((0, 2)),
                 predicted_en_m=np.empty((0, 2)),
                 cumulative_observed_en_m=np.empty((0, 2)),
                 cumulative_predicted_en_m=np.empty((0, 2)),
                 bridge_sse_predicted_m2=np.empty(0),
                 bridge_sse_zero_m2=np.empty(0),
                 k_en_ones=np.empty(0),
                 reanchored=np.empty(0, bool),
                 reanchor_reason=np.empty(0, '<U24'),
                 new_prn_k=np.empty(0, int))
    if not validation:
        return dict(status='no_validation_bridge', validation_bridges=0), empty
    supported, unknown = [], []
    for step in validation:
        if np.isfinite(beta[step['mask']]).all():
            supported.append(step)
        else:
            unknown.append(step)
    if not supported:
        empty['new_prn_k'] = np.array([step['k'] for step in unknown], int)
        return dict(status='no_supported_validation_bridge',
                    validation_bridges=len(validation),
                    new_prn_bridges=len(unknown)), empty
    predicted_sse = zero_sse = rate_sse = rate_zero_sse = 0.
    graph = {}
    roots = 0
    max_k1 = 0.
    cumulative_predicted = []
    cumulative_observed = []
    per_prn = []
    step_predicted = []
    step_observed = []
    reanchored = []
    root_reasons = []
    bridge_sse_predicted = []
    bridge_sse_zero = []
    k1_values = []
    for step in supported:
        mask = step['mask']
        local_beta = beta[mask]
        assert np.isfinite(local_beta).all()
        k_en = np.linalg.pinv(step['h'])[:2]
        k1 = float(np.max(np.abs(k_en@np.ones(len(local_beta)))))
        assert k1 < 1e-10
        max_k1 = max(max_k1, k1)
        k1_values.append(k1)
        terms = step['dt']*(k_en*local_beta)
        prediction = terms.sum(axis=1)
        np.testing.assert_allclose(prediction, step['dt']*(k_en@local_beta),
                                   rtol=0, atol=1e-12)
        observed = step['observed_en']
        star_terms = np.full((stars, 2), np.nan)
        star_terms[mask] = terms.T
        per_prn.append(star_terms)
        step_predicted.append(prediction)
        step_observed.append(observed)
        predicted_sse += float(np.sum((observed-prediction)**2))
        zero_sse += float(np.sum(observed**2))
        bridge_sse_predicted.append(float(np.sum((observed-prediction)**2)))
        bridge_sse_zero.append(float(np.sum(observed**2)))
        centered = step['b']-step['b'].mean()
        modeled = local_beta-local_beta.mean()
        rate_sse += float(np.sum((centered-modeled)**2))
        rate_zero_sse += float(np.sum(centered**2))
        parent = graph.get(step['parent'])
        new_root = (parent is None or not np.allclose(parent['position'],
                                                      step['parent_position'],
                                                      rtol=0, atol=1e-10))
        reanchored.append(new_root)
        root_reasons.append('missing_parent_bridge' if parent is None else
                            'parent_position_changed' if new_root else 'continued')
        if new_root:
            previous_prediction = previous_observed = np.zeros(2)
            roots += 1
        else:
            previous_prediction = parent['predicted']
            previous_observed = parent['observed']
        cumulative = dict(position=step['prediction'],
                          predicted=previous_prediction+prediction,
                          observed=previous_observed+observed)
        graph[step['k']] = cumulative
        cumulative_predicted.append(cumulative['predicted'])
        cumulative_observed.append(cumulative['observed'])
    predicted = np.array(cumulative_predicted)
    observed = np.array(cumulative_observed)
    residual = observed-predicted
    report = dict(status='scored', validation_bridges=len(validation),
                supported_bridges=len(supported), new_prn_bridges=len(unknown),
                lag1_bridges=sum(step['lag'] == 1 for step in supported),
                lag2_bridges=sum(step['lag'] == 2 for step in supported),
                reanchored_segments=roots,
                missing_parent_bridge_roots=root_reasons.count('missing_parent_bridge'),
                changed_parent_position_roots=root_reasons.count('parent_position_changed'),
                max_k_en_ones=max_k1,
                bridge_en_sse_predicted_m2=predicted_sse,
                bridge_en_sse_zero_m2=zero_sse,
                prn_rate_sse_predicted_m2ps2=rate_sse,
                prn_rate_sse_zero_m2ps2=rate_zero_sse,
                cumulative_en_final_observed_m=observed[-1].tolist(),
                cumulative_en_final_predicted_m=predicted[-1].tolist(),
                cumulative_en_residual_max_m=float(np.max(
                    np.linalg.norm(residual, axis=1))),
                cumulative_direction_dot_sum_m2=float(np.sum(predicted*observed)))
    detail = dict(k=np.array([step['k'] for step in supported], int),
                  parent=np.array([step['parent'] for step in supported], int),
                  mask=np.stack([step['mask'] for step in supported]),
                  per_prn_en_m=np.stack(per_prn),
                  observed_en_m=np.stack(step_observed),
                  predicted_en_m=np.stack(step_predicted),
                  cumulative_observed_en_m=observed,
                  cumulative_predicted_en_m=predicted,
                  bridge_sse_predicted_m2=np.array(bridge_sse_predicted),
                  bridge_sse_zero_m2=np.array(bridge_sse_zero),
                  k_en_ones=np.array(k1_values),
                  reanchored=np.array(reanchored, bool),
                  reanchor_reason=np.array(root_reasons, '<U24'),
                  new_prn_k=np.array([step['k'] for step in unknown], int))
    return report, detail


def analyze_path(folder, expected_hashes, formal_arrays_sha, velocity_east):
    hashes = {name: digest(folder/name) for name in replay.SCIENCE}
    assert hashes == {name: expected_hashes[name] for name in replay.SCIENCE}
    tracking = replay.read_npz(folder/'tracking.npz')
    trace = replay.read_npz(folder/'trace.npz')
    support = replay.read_npz(folder/'support.npz')['coherent_support']
    saved = replay.read_npz(folder/'navigation.npz')
    eligible, latest, recent = projection_eligibility(tracking['available_time'], trace)
    np.testing.assert_array_equal(eligible, saved['phase_state_eligible'])
    np.testing.assert_array_equal(latest, saved['latest_complete_block'])
    np.testing.assert_array_equal(recent, saved['recent_block_dimensions'])
    z = navigation_input(tracking, support)
    baseline, old_records, _ = recorded_navigate(z, eligible)
    replay.verify_baseline(saved, baseline, quality_support_counts(tracking, support))
    candidate, new_records, bridge = recorded_navigate(z, eligible, continuity=True)
    formal = replay.read_npz(formal_arrays_sha[0])
    assert digest(formal_arrays_sha[0]) == formal_arrays_sha[1]
    for key, value in (('baseline_position', baseline['position']),
                       ('candidate_position', candidate['position']),
                       ('bridge_admission', bridge['admission']),
                       ('bridge_called', bridge['called']),
                       ('bridge_failed', bridge['failed'])):
        np.testing.assert_array_equal(formal[key], value, err_msg=key)
    time = tracking['time']
    old_steps = [bridge_physics(z, record, velocity_east) for record in old_records]
    new_steps = [bridge_physics(z, record, velocity_east) for record in new_records]
    nonbridge = {}
    for name, navigation in (('baseline', baseline), ('candidate', candidate)):
        failed = (time >= SPLIT_S) & ~navigation['diagnostics']['phase_ok']
        nonbridge[f'{name}_nonbridge_epochs'] = int(failed.sum())
        nonbridge[f'{name}_code_fallback_epochs'] = int((
            failed & np.isfinite(navigation['candidate_position']).all(axis=1)).sum())
    training = [step for step in old_steps if START_S <= time[step['k']] < SPLIT_S]
    if not training:
        return dict(input_sha256=hashes, formal_arrays_sha256=formal_arrays_sha[1],
                    fit=dict(status='no_training_bridge', training_bridges=0),
                    baseline=dict(status='no_validation_bridge'),
                    candidate=dict(status='no_validation_bridge'),
                    **nonbridge), {}
    fit, beta = fit_fixed_bias(training, z['carrier'].shape[1])
    if beta is None:
        return dict(input_sha256=hashes, formal_arrays_sha256=formal_arrays_sha[1],
                    fit=fit, baseline=dict(status='unidentifiable'),
                    candidate=dict(status='unidentifiable'), **nonbridge), {}
    old_report, old_detail = evaluate_bridges(old_steps, beta, time)
    new_report, new_detail = evaluate_bridges(new_steps, beta, time)
    return dict(input_sha256=hashes, formal_arrays_sha256=formal_arrays_sha[1],
                fit=fit, baseline=old_report, candidate=new_report,
                **nonbridge), dict(
                        baseline=old_detail, candidate=new_detail)


def analyze():
    assert digest(study.OUTPUT/'summary.json') == STAGE47_SUMMARY_SHA256
    formal_path = study.OUTPUT/'single_path_carrier_continuity_summary.json'
    assert digest(formal_path) == CONTINUITY_SUMMARY_SHA256
    formal = json.loads(formal_path.read_text(encoding='utf-8'))
    frozen = json.loads((study.OUTPUT/'summary.json').read_text(encoding='utf-8'))
    root = Path(__file__).resolve().parents[2]
    continuity_source = 'tools/single_path_carrier_continuity.py'
    assert digest(current_path_for_historical_source(root,continuity_source)) == formal['source_sha256']
    for name in replay.NAVIGATION_SOURCES:
        assert digest(current_path_for_historical_source(root,name)) == frozen['source_sha256'][name]
    cases, arrays = {}, {}
    for seed in study.SEEDS:
        cases[str(seed)] = {}
        for scene in replay.SCENES:
            row = frozen['cases'][str(seed)][scene]
            folder = study.OUTPUT/str(seed)/scene/'gate3'
            metadata = study.OUTPUT/str(seed)/scene/'input'/'metadata.json'
            assert digest(metadata) == row['input_sha256']['metadata.json']
            velocity = json.loads(metadata.read_text(encoding='utf-8'))['case']['velocity_east']
            array_path = study.OUTPUT/f'single_path_carrier_continuity_{seed}_{scene}.npz'
            result, detail = analyze_path(
                folder, row['paths']['gate3']['output_sha256'],
                (array_path, digest(array_path)), velocity)
            result['metadata_sha256'] = row['input_sha256']['metadata.json']
            cases[str(seed)][scene] = result
            arrays[f'{seed}_{scene}'] = detail
    return dict(scope=__doc__, analysis_source_sha256=digest(Path(__file__)),
                stage47_summary_sha256=STAGE47_SUMMARY_SHA256,
                continuity_summary_sha256=CONTINUITY_SUMMARY_SHA256,
                continuity_source_sha256=formal['source_sha256'],
                train_observation_interval_s=[START_S, SPLIT_S],
                validation_observation_start_s=SPLIT_S, cases=cases), arrays


def main():
    summary, arrays = analyze()
    destination = study.OUTPUT/'carrier_rate_bias_diagnosis_summary.json'
    assert not destination.exists()
    destination.write_text(json.dumps(summary, indent=2, allow_nan=False),
                           encoding='utf-8')
    for name, paths in arrays.items():
        np.savez_compressed(study.OUTPUT/f'carrier_rate_bias_diagnosis_{name}.npz',
                            **{f'{mode}_{key}': value for mode, detail in paths.items()
                               for key, value in detail.items()})


if __name__ == '__main__':
    main()
