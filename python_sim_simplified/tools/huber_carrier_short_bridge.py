"""Fixed 0.02 m Huber carrier fit in the original short-bridge navigation.

Both trajectories start from empty state on nine saved Stage 47 records.
Only the leave-one-out short bridge's carrier fit is rebound in private globals.
"""
import json
from pathlib import Path
from types import FunctionType

import numpy as np
from scipy.optimize import least_squares

from core import mechanism_phase, single_source
from core.gps_simulation import receiver_reference
from experiments import single_source_projection_long as study
from experiments.single_source_coherent_navigation import quality_support_counts
from experiments.single_source_phase_state_gate import projection_eligibility
from experiments.single_source_startup_amplitude_rf import score_navigation
from experiments.single_source_suppression_study import digest
from experiments.single_source_temporal_trigger import navigation_input
from tools import single_path_carrier_continuity as replay


DELTA_M = .02
STAGE47_SUMMARY_SHA256 = replay.STAGE47_SUMMARY_SHA256
REPLAY_SUMMARY_SHA256 = 'f2c6004241a8891145b5d9cb4f88d30b29377b320a57b35dc710a62392f85166'
SCENES = replay.SCENES
SCIENCE = replay.SCIENCE
NAVIGATION_SOURCES = replay.NAVIGATION_SOURCES


def huber_objective(residual):
    absolute = np.abs(residual)
    return float(np.sum(np.where(absolute <= DELTA_M,
                                 .5*residual**2,
                                 DELTA_M*absolute-.5*DELTA_M**2)))


def huber_fit(current_satellites, previous_satellites, carrier_change,
              return_residual=False, record=None):
    """Preserve original geometry/raw gates; change only the fitted theta."""
    reference, rotation = receiver_reference()
    states = current_satellites-reference
    previous = previous_satellites-reference
    h = np.column_stack((-(states/np.linalg.norm(states, axis=1)[:, None])
                          @ rotation.T, np.ones(len(states))))
    if np.linalg.cond(h) > 1000:
        if record is not None:
            record.update(geometry_pass=False, accepted=False)
        return (None, np.full(len(states), np.nan)) if return_residual else None
    measured = carrier_change-(np.linalg.norm(states, axis=1)
                               -np.linalg.norm(previous, axis=1))
    ols = np.linalg.lstsq(h, measured, rcond=None)[0]
    initial_residual = h@ols-measured
    solution = least_squares(lambda theta: h@theta-measured, ols,
                             jac=lambda theta: h, loss='huber',
                             f_scale=DELTA_M, xtol=1e-12, ftol=1e-12,
                             gtol=1e-12, max_nfev=200)
    residual = h@solution.x-measured
    gradient = h.T@np.clip(residual, -DELTA_M, DELTA_M)
    objective = huber_objective(residual)
    initial_objective = huber_objective(initial_residual)
    if (not solution.success or objective > initial_objective+1e-12
            or np.linalg.norm(gradient, np.inf) > 1e-8):
        raise RuntimeError(
            f'Huber numerical failure: status={solution.status}, '
            f'message={solution.message}, objective={objective}, '
            f'initial={initial_objective}, gradient_inf='
            f'{np.linalg.norm(gradient, np.inf)}')
    variance = max(DELTA_M**2, float(residual@residual)/(len(states)-4))
    covariance = variance*np.linalg.inv(h.T@h)
    covariance_pass = np.sqrt(np.linalg.eigvalsh(covariance[:2, :2])[-1]) <= .5
    residual_pass = np.max(np.abs(residual)) <= .1
    accepted = bool(covariance_pass and residual_pass)
    if record is not None:
        weights = np.ones(len(residual))
        large = np.abs(residual) > DELTA_M
        weights[large] = DELTA_M/np.abs(residual[large])
        record.update(geometry_pass=True, accepted=accepted,
                      covariance_pass=bool(covariance_pass),
                      residual_pass=bool(residual_pass),
                      weights=weights, ols_delta=solution.x-ols,
                      raw_residual=residual, objective=objective,
                      gradient_inf=float(np.linalg.norm(gradient, np.inf)))
    fitted = solution.x if accepted else None
    return (fitted, residual) if return_residual else fitted


def clone_function(function, globals_):
    return FunctionType(function.__code__, globals_, function.__name__,
                        function.__defaults__, function.__closure__)


def candidate_navigate(z, eligible):
    """Rebind fit inside phase function, then phase function inside navigate."""
    current = {'k': None, 'valid': None}
    records = []
    stars = z['carrier'].shape[1]

    def wrapped_fit(current_satellites, previous_satellites, carrier_change,
                    return_residual=False):
        k = current['k']
        prn = np.array([np.flatnonzero(np.all(z['satellites'][k] == row, axis=1))[0]
                        for row in current_satellites], int)
        j = next(j for j in (k-1, k-2)
                 if np.array_equal(z['satellites'][j, prn], previous_satellites))
        common = np.zeros(stars, bool)
        common[prn] = True
        valid = current['valid']
        full = (valid[k] & valid[j]
                & np.isfinite(z['carrier'][k]) & np.isfinite(z['carrier'][j])
                & (z['arc'][k] == z['arc'][j]))
        dropped = np.flatnonzero(full & ~common)
        item = dict(epoch=k, lag=k-j, prn_mask=common,
                    excluded_prn=int(dropped[0]+1) if len(dropped) else 0,
                    chosen=False)
        try:
            result = huber_fit(current_satellites, previous_satellites,
                               carrier_change, return_residual, record=item)
        except RuntimeError as error:
            raise RuntimeError(f'epoch={k}, lag={k-j}, '
                               f'PRN={list(prn+1)}: {error}') from error
        records.append(item)
        return result

    phase_globals = dict(mechanism_phase.phase_prediction_leave_one_out.__globals__)
    phase_globals['fit_carrier_displacement'] = wrapped_fit
    phase_bridge = clone_function(mechanism_phase.phase_prediction_leave_one_out,
                                  phase_globals)

    def wrapped_phase(observations, valid, positions, k):
        current['k'] = k
        current['valid'] = valid
        start = len(records)
        prediction, details = phase_bridge(observations, valid, positions, k)
        if prediction is not None:
            lag = details['lag_epochs']
            excluded = details['excluded_prn']
            fits = records[start:]
            if excluded:
                chosen = next(item for item in fits if item['lag'] == lag
                              and item['accepted'] and item['excluded_prn'] == excluded)
            else:
                chosen = next(item for item in fits if item['lag'] == lag
                              and item['accepted'] and item['excluded_prn'] == 0)
            chosen['chosen'] = True
        return prediction, details

    navigate_globals = dict(single_source.navigate.__globals__)
    navigate_globals['phase_prediction_leave_one_out'] = wrapped_phase
    navigate = clone_function(single_source.navigate, navigate_globals)
    result = navigate(z, **replay.short_options(eligible))
    return result, records


def fit_arrays(records, stars):
    """Rectangular audit, retaining every attempted fit and final choice."""
    count = len(records)
    out = dict(epoch=np.empty(count, int), lag=np.empty(count, int),
               excluded_prn=np.zeros(count, int),
               prn_mask=np.zeros((count, stars), bool),
               chosen=np.zeros(count, bool), accepted=np.zeros(count, bool),
               geometry_pass=np.zeros(count, bool),
               covariance_pass=np.zeros(count, bool),
               residual_pass=np.zeros(count, bool),
               weights=np.full((count, stars), np.nan),
               raw_residual=np.full((count, stars), np.nan),
               ols_delta=np.full((count, 4), np.nan),
               objective=np.full(count, np.nan),
               gradient_inf=np.full(count, np.nan))
    for i, item in enumerate(records):
        prn = item['prn_mask']
        out['epoch'][i] = item['epoch']
        out['lag'][i] = item['lag']
        out['excluded_prn'][i] = item['excluded_prn']
        out['prn_mask'][i] = prn
        out['chosen'][i] = item['chosen']
        out['accepted'][i] = item['accepted']
        out['geometry_pass'][i] = item['geometry_pass']
        if item['geometry_pass']:
            out['covariance_pass'][i] = item['covariance_pass']
            out['residual_pass'][i] = item['residual_pass']
            out['weights'][i, prn] = item['weights']
            out['raw_residual'][i, prn] = item['raw_residual']
            out['ols_delta'][i] = item['ols_delta']
            out['objective'][i] = item['objective']
            out['gradient_inf'][i] = item['gradient_inf']
    return out


def run_path(folder, expected_hashes, scene, velocity_east):
    input_hashes = {name: digest(folder/name) for name in SCIENCE}
    for name in SCIENCE:
        assert input_hashes[name] == expected_hashes[name], name
    tracking = replay.read_npz(folder/'tracking.npz')
    trace = replay.read_npz(folder/'trace.npz')
    support = replay.read_npz(folder/'support.npz')['coherent_support']
    saved = replay.read_npz(folder/'navigation.npz')
    eligible, latest, recent = projection_eligibility(tracking['available_time'], trace)
    np.testing.assert_array_equal(eligible, saved['phase_state_eligible'])
    np.testing.assert_array_equal(latest, saved['latest_complete_block'])
    np.testing.assert_array_equal(recent, saved['recent_block_dimensions'])
    z = navigation_input(tracking, support)
    baseline = single_source.navigate(z, **replay.short_options(eligible))
    replay.verify_baseline(saved, baseline,
                           quality_support_counts(tracking, support))
    candidate, records = candidate_navigate(z, eligible)
    baseline_score = score_navigation(tracking, baseline, velocity_east)
    candidate_score = score_navigation(tracking, candidate, velocity_east)
    screen = study.long_screen(
        scene, tracking['time'], baseline['position'], candidate['position'],
        velocity_east, baseline_score['max_relative_motion_error_m'],
        candidate_score['max_relative_motion_error_m'])
    baseline_published = np.isfinite(baseline['position']).all(axis=1)
    candidate_published = np.isfinite(candidate['position']).all(axis=1)
    result = dict(input_sha256=input_hashes, baseline_reproduced=True,
                  baseline=baseline_score, candidate=candidate_score,
                  long_screen=screen,
                  evaluation_time_axes=replay.time_axis_counts(
                      tracking['time'], tracking['available_time'],
                      baseline['position'], candidate['position']),
                  baseline_events=replay.event_times(
                      tracking['time'], tracking['available_time'], baseline),
                  candidate_events=replay.event_times(
                      tracking['time'], tracking['available_time'], candidate),
                  baseline_code_fallback_published=int((baseline_published
                       & ~baseline['diagnostics']['phase_ok']).sum()),
                  candidate_code_fallback_published=int((candidate_published
                       & ~candidate['diagnostics']['phase_ok']).sum()),
                  fit_attempts=len(records),
                  chosen_fits=sum(item['chosen'] for item in records))
    arrays = dict(time=tracking['time'],
                  baseline_position=baseline['position'],
                  candidate_position=candidate['position'],
                  baseline_selected=baseline['selected'],
                  candidate_selected=candidate['selected'],
                  candidate_phase_ok=candidate['diagnostics']['phase_ok'],
                  candidate_phase_lag=candidate['diagnostics']['phase_lag_epochs'],
                  candidate_phase_excluded_prn=candidate['diagnostics']['phase_excluded_prn'],
                  candidate_reason=candidate['diagnostics']['reason'],
                  candidate_isolated=candidate['diagnostics']['isolated'],
                  candidate_recovery_confirmed=candidate['diagnostics']['recovery_confirmed'])
    arrays.update({f'fit_{key}': value for key, value in
                   fit_arrays(records, z['carrier'].shape[1]).items()})
    return result, arrays


def analyze():
    assert digest(study.OUTPUT/'summary.json') == STAGE47_SUMMARY_SHA256
    replay_summary_path = study.OUTPUT/'single_path_carrier_continuity_summary.json'
    assert digest(replay_summary_path) == REPLAY_SUMMARY_SHA256
    replay_summary = json.loads(replay_summary_path.read_text(encoding='utf-8'))
    assert digest(Path(replay.__file__)) == replay_summary['source_sha256']
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
                replay_summary_sha256=REPLAY_SUMMARY_SHA256,
                replay_source_sha256=replay_summary['source_sha256'],
                frozen_navigation_source_sha256={name: frozen['source_sha256'][name]
                                                 for name in NAVIGATION_SOURCES},
                screen_pass=all(result['long_screen']['pass']
                                for cases in results.values()
                                for result in cases.values()),
                cases=results), arrays


def main():
    summary, arrays = analyze()
    destination = study.OUTPUT/'huber_carrier_short_bridge_summary.json'
    assert not destination.exists()
    assert not (study.OUTPUT/'huber_carrier_short_bridge_manifest.json').exists()
    for key in arrays:
        assert not (study.OUTPUT/f'huber_carrier_short_bridge_{key}.npz').exists()
    destination.write_text(json.dumps(summary, indent=2, allow_nan=False),
                           encoding='utf-8')
    hashes = {'huber_carrier_short_bridge_summary.json': digest(destination)}
    for key, detail in arrays.items():
        path = study.OUTPUT/f'huber_carrier_short_bridge_{key}.npz'
        np.savez_compressed(path, **detail)
        hashes[path.name] = digest(path)
    (study.OUTPUT/'huber_carrier_short_bridge_manifest.json').write_text(
        json.dumps(hashes, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
