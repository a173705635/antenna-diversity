"""Offline Stage 48 short-path error decomposition and projection association.

Reads frozen Stage 47/48 arrays only. Run explicitly after review to write the
mechanism summary beside Stage 48 results; no RF or navigation is recomputed.
"""
import json
from pathlib import Path

import numpy as np

from experiments import single_source_nominal_yaw as study
from experiments.single_source_phase_exclusion import verify_source_snapshot
from experiments.single_source_projection_long import OUTPUT as STAGE47
from experiments.single_source_suppression_study import digest


STAGE48_SUMMARY_SHA256 = 'b9c0b5b3887353716520cfb5df73995c74b4ef3e7ef39b7c869c81d8780e5724'


def read_npz(path):
    with np.load(path) as saved:
        return dict(saved)


def decompose_short(time, position):
    """Split each published east/north error at its first actual fix."""
    published = np.isfinite(position).all(axis=1)
    first = np.flatnonzero(published)[0]
    truth = np.column_stack((2.*time, np.zeros(len(time))))
    error = position[:, :2]-truth
    anchor = error[first]
    drift = error-anchor
    error[~published] = np.nan
    drift[~published] = np.nan
    cross = 2.*np.einsum('j,ij->i', anchor, drift)
    error_squared = np.sum(error**2, axis=1)
    drift_squared = np.sum(drift**2, axis=1)
    identity = error-(anchor+drift)
    square_identity = error_squared-(np.sum(anchor**2)+drift_squared+cross)
    scored = published & (time >= .3)
    indices = np.flatnonzero(scored)
    top = None if not len(indices) else int(indices[np.argmax(error_squared[indices])])
    summary = dict(anchor_index=int(first), anchor_observation_time_s=float(time[first]),
                   anchor_error_m=float(np.linalg.norm(anchor)),
                   published_all=int(published.sum()), score_count=int(scored.sum()),
                   max_vector_identity_residual_m=float(np.max(
                       np.linalg.norm(identity[published], axis=1))),
                   max_square_identity_residual_m2=float(np.max(
                       abs(square_identity[published]))),
                   drift_max_m=None if not len(indices) else float(np.max(
                       np.sqrt(drift_squared[indices]))),
                   drift_p95_m=None if not len(indices) else float(np.percentile(
                       np.sqrt(drift_squared[indices]), 95)),
                   cross_positive_fraction=None if not len(indices) else float(
                       np.mean(cross[indices] > 0.)),
                   cross_negative_fraction=None if not len(indices) else float(
                       np.mean(cross[indices] < 0.)),
                   max_error_index=top)
    if top is not None:
        summary['max_error_squared_terms_m2'] = dict(
            anchor=float(np.sum(anchor**2)), drift=float(drift_squared[top]),
            cross=float(cross[top]), total=float(error_squared[top]))
    arrays = dict(anchor=anchor, error=error, drift=drift, cross=cross,
                  error_squared=error_squared, drift_squared=drift_squared,
                  scored=scored)
    return summary, arrays


def _pearson(x, y):
    if len(x) < 2 or np.std(x) == 0. or np.std(y) == 0.:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def gate_state_association(time, latest, selected, raw, retained, decomposition):
    """Join selected PRNs to the last block complete at availability time.

    This describes the eligibility/gating state, not the observation's RF
    evidence window or every block contributing to navigation history.
    """
    scored = decomposition['scored']
    valid = scored & (latest >= 0)
    active = valid & (raw[latest.clip(min=0)] >= 3.)
    inactive = valid & ~active
    state = {'inactive': inactive, 'active': active}
    entries = {name: int(np.sum(selected[epochs])) for name, epochs in state.items()}
    means = np.full(len(time), np.nan)
    by_state_prn = {}
    for name, epochs in state.items():
        by_prn = {}
        for prn in range(selected.shape[1]):
            rows = np.flatnonzero(epochs & selected[:, prn])
            if len(rows):
                values = retained[latest[rows], prn]
                by_prn[str(prn+1)] = dict(count=len(rows),
                                          median=float(np.median(values)),
                                          minimum=float(np.min(values)))
        by_state_prn[name] = by_prn
    for row in np.flatnonzero(active):
        stars = selected[row]
        if stars.any():
            means[row] = float(np.mean(retained[latest[row], stars]))
    associated = active & np.isfinite(means)
    x = means[associated]
    absolute = np.sqrt(decomposition['error_squared'][associated])
    drift = np.sqrt(decomposition['drift_squared'][associated])
    return dict(epochs_by_state={name: int(np.sum(epochs))
                                 for name, epochs in state.items()},
                selected_entries_by_state=entries,
                inactive_by_prn=by_state_prn['inactive'],
                active_by_prn=by_state_prn['active'],
                active_epoch_mean_retention=[None if np.isnan(value) else float(value)
                                             for value in means],
                associated_active_epochs=int(associated.sum()),
                correlation_retention_absolute_error=_pearson(x, absolute),
                correlation_retention_drift=_pearson(x, drift))


def evidence_window_association(time, selected, raw, retained, decomposition):
    """Overlap each [observation-20 ms, observation) RF window with fit blocks.

    A window at an exact block boundary belongs to the preceding block. A
    straddling window weights the two blocks by overlap duration. This is a
    current-observation association, not a reconstruction of navigation history.
    """
    scored = decomposition['scored']
    fractions = np.full(len(time), np.nan)
    weighted_retention = np.full((len(time), selected.shape[1]), np.nan)
    means = np.full(len(time), np.nan)
    for row in np.flatnonzero(scored):
        end_sample = int(np.rint(time[row]*study.FS))
        start_sample = end_sample-study.WIDTH
        first = start_sample//study.WIDTH
        stop = (end_sample-1)//study.WIDTH+1
        weights = []
        blocks = []
        for block in range(first, stop):
            overlap = (min(end_sample, (block+1)*study.WIDTH)
                       -max(start_sample, block*study.WIDTH))
            blocks.append(block)
            weights.append(overlap)
        assert sum(weights) == study.WIDTH
        weight = np.asarray(weights, dtype=float)/study.WIDTH
        fractions[row] = float(np.sum(weight*(raw[blocks] >= 3.)))
        weighted_retention[row] = np.sum(
            weight[:, None]*retained[blocks], axis=0)
        stars = selected[row]
        if stars.any():
            means[row] = float(np.mean(weighted_retention[row, stars]))
    active = scored & np.isclose(fractions, 1., atol=1e-8)
    inactive = scored & np.isclose(fractions, 0., atol=1e-8)
    mixed = scored & ~active & ~inactive
    state = {'inactive': inactive, 'mixed': mixed, 'active': active}
    by_state_prn = {}
    correlations = {}
    for name, epochs in state.items():
        by_prn = {}
        for prn in range(selected.shape[1]):
            rows = np.flatnonzero(epochs & selected[:, prn])
            if len(rows):
                values = weighted_retention[rows, prn]
                by_prn[str(prn+1)] = dict(count=len(rows),
                                          median=float(np.median(values)),
                                          minimum=float(np.min(values)))
        by_state_prn[name] = by_prn
        associated = epochs & np.isfinite(means)
        correlations[name] = dict(
            associated_epochs=int(associated.sum()),
            retention_absolute_error=_pearson(
                means[associated], np.sqrt(decomposition['error_squared'][associated])),
            retention_drift=_pearson(
                means[associated], np.sqrt(decomposition['drift_squared'][associated])))
    return dict(epochs_by_state={name: int(epochs.sum())
                                 for name, epochs in state.items()},
                selected_entries_by_state={name: int(selected[epochs].sum())
                                           for name, epochs in state.items()},
                inactive_by_prn=by_state_prn['inactive'],
                mixed_by_prn=by_state_prn['mixed'],
                active_by_prn=by_state_prn['active'],
                active_fraction=[None if np.isnan(value) else float(value)
                                 for value in fractions],
                epoch_mean_retention=[None if np.isnan(value) else float(value)
                                      for value in means],
                correlations_by_state=correlations)


def verified_path(scene, yaw, stage47, stage48):
    if yaw == 0:
        folder = STAGE47/str(study.SEED)/scene/'gate3'
        expected = stage47['cases'][str(study.SEED)][scene]['paths']['gate3']['output_sha256']
        names = ('tracking.npz', 'navigation.npz', 'fit.npz', 'trace.npz')
        navigation_name = 'navigation.npz'
    else:
        folder = study.OUTPUT/scene/('yaw_minus2' if yaw < 0 else 'yaw_plus2')
        path = stage48['cases'][scene]['paths']['yaw_minus2' if yaw < 0 else 'yaw_plus2']
        navigation_name = 'navigation_short.npz' if path['status'] == 'navigation_incomplete' else 'navigation.npz'
        if path['provenance'] == 'recovery_receipt':
            expected = {**path['frontend_sha256'],
                        navigation_name: path['navigation_sha256'],
                        'physical_diagnostics.npz': path['diagnostics_sha256']}
        else:
            expected = path['output_sha256']
        names = ('tracking.npz', navigation_name, 'fit.npz',
                 'physical_diagnostics.npz')
    for name in names:
        assert digest(folder/name) == expected[name], (scene, yaw, name)
    return folder, navigation_name


def assert_frozen_source_identity(plan, stage48):
    assert plan['source_sha256'] == stage48['frozen_source_sha256']


def analyze():
    assert digest(STAGE47/'summary.json') == study.STAGE47_SUMMARY_SHA256
    assert digest(study.OUTPUT/'recovery_summary.json') == STAGE48_SUMMARY_SHA256
    stage47 = json.loads((STAGE47/'summary.json').read_text(encoding='utf-8'))
    stage48 = json.loads((study.OUTPUT/'recovery_summary.json').read_text(encoding='utf-8'))
    plan = json.loads((study.OUTPUT/'plan.json').read_text(encoding='utf-8'))
    assert_frozen_source_identity(plan, stage48)
    verify_source_snapshot(study.OUTPUT/'source_snapshot.zip', plan['source_sha256'])
    assert digest(Path(study.__file__)) == plan['source_sha256'][
        'experiments/single_source_nominal_yaw.py']
    cases = {}
    for scene in study.SCENES:
        cases[scene] = {}
        for yaw in study.YAWS_DEG:
            folder, navigation_name = verified_path(scene, yaw, stage47, stage48)
            tracking = read_npz(folder/'tracking.npz')
            navigation = read_npz(folder/navigation_name)
            np.testing.assert_array_equal(tracking['time'], navigation['observation_time'])
            np.testing.assert_array_equal(tracking['available_time'], navigation['available_time'])
            position = navigation['short_position']
            result, decomposition = decompose_short(tracking['time'], position)
            if scene == 'weak':
                fit = read_npz(folder/'fit.npz')
                if yaw == 0:
                    source = STAGE47/str(study.SEED)/scene/'input'/'observations.npz'
                    expected = stage47['cases'][str(study.SEED)][scene]['input_sha256']['observations.npz']
                    assert digest(source) == expected
                    physical = study.physical_diagnostics(read_npz(source), study.SEED, folder)
                else:
                    physical = read_npz(folder/'physical_diagnostics.npz')
                result['gate_state_association'] = gate_state_association(
                    tracking['time'], navigation['latest_complete_block'],
                    navigation['short_selected'], fit['raw'],
                    physical['true_star_power_retained'], decomposition)
                result['evidence_window_association'] = evidence_window_association(
                    tracking['time'], navigation['short_selected'], fit['raw'],
                    physical['true_star_power_retained'], decomposition)
            cases[scene][f'{yaw:+g}'] = result
    return dict(scope=__doc__, analysis_source_sha256=digest(Path(__file__)),
                stage47_summary_sha256=study.STAGE47_SUMMARY_SHA256,
                stage48_recovery_summary_sha256=STAGE48_SUMMARY_SHA256,
                cases=cases)


def main():
    summary = analyze()
    destination = study.OUTPUT/'mechanism_summary.json'
    assert not destination.exists()
    destination.write_text(json.dumps(summary, indent=2, allow_nan=False),
                           encoding='utf-8')


if __name__ == '__main__':
    main()
