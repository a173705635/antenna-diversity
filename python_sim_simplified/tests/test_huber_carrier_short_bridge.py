import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize

from core.spoofing import mechanism_phase, mechanism_smoothing, single_source
from core.gps_simulation import receiver_reference
from experiments.spoofing import single_source_projection_long as study
from experiments.spoofing.single_source_phase_state_gate import projection_eligibility
from experiments.spoofing.single_source_suppression_study import digest
from experiments.spoofing.single_source_temporal_trigger import navigation_input
from tools.spoofing import huber_carrier_short_bridge as huber
from tools.spoofing import single_path_carrier_continuity as replay


def constellation(seed=3, count=9, spread=1.):
    rng = np.random.default_rng(seed)
    reference, rotation = receiver_reference()
    directions = np.column_stack((rng.normal(0, spread, (count, 2)),
                                   np.ones(count)))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    satellites = reference+directions@rotation*2e7
    h = np.column_stack((-directions, np.ones(count)))
    return satellites, h


def test_quadratic_region_matches_ols_and_geometry_consistent_shift():
    satellites, h = constellation()
    theta = np.array([.03, -.02, .01, .005])
    small_noise = np.array([.001, -.001, .002, .0, -.002, .001, 0., -.001, .002])
    measured = h@theta+small_noise
    fit, residual = huber.huber_fit(satellites, satellites, measured, True)
    ols = np.linalg.lstsq(h, measured, rcond=None)[0]
    np.testing.assert_allclose(fit, ols, atol=1e-9)
    assert np.max(np.abs(residual)) < huber.DELTA_M

    shift = np.array([.12, -.04, .02, .03])
    shifted, shifted_residual = huber.huber_fit(
        satellites, satellites, measured+h@shift, True)
    np.testing.assert_allclose(shifted, fit+shift, atol=1e-9)
    np.testing.assert_allclose(shifted_residual, residual, atol=1e-9)


def test_huber_tail_matches_independent_optimum_and_raw_gate():
    satellites, h = constellation()
    measured = h@np.array([.01, -.02, .0, .005])
    measured[-1] += .18
    record = {}
    fit, residual = huber.huber_fit(satellites, satellites, measured, True,
                                    record=record)
    ols = np.linalg.lstsq(h, measured, rcond=None)[0]
    independent = minimize(
        lambda theta: huber.huber_objective(h@theta-measured), ols,
        jac=lambda theta: h.T@np.clip(h@theta-measured,
                                      -huber.DELTA_M, huber.DELTA_M),
        method='BFGS', options={'gtol': 1e-11})
    np.testing.assert_allclose(record['ols_delta'], independent.x-ols, atol=1e-8)
    assert record['objective'] <= huber.huber_objective(h@ols-measured)
    assert record['gradient_inf'] <= 1e-8
    assert np.min(record['weights']) < 1
    assert not record['residual_pass']
    assert fit is None
    assert np.max(np.abs(residual)) > .1


def test_original_covariance_gate_uses_raw_huber_rss():
    satellites, h = constellation(spread=.04)
    measured = h@np.array([.02, -.01, 0., 0.])
    measured[-1] += .2
    record = {}
    huber.huber_fit(satellites, satellites, measured, True, record=record)
    assert record['geometry_pass']
    raw = record['raw_residual']
    variance = max(.02**2, raw@raw/(len(raw)-4))
    covariance = variance*np.linalg.inv(h.T@h)
    expected = np.sqrt(np.linalg.eigvalsh(covariance[:2, :2])[-1]) <= .5
    assert record['covariance_pass'] == expected
    assert not expected


def test_numerical_failure_is_explicit(monkeypatch):
    satellites, h = constellation()

    class Failed:
        success = False
        status = 0
        message = 'forced failure'
        x = np.zeros(4)

    monkeypatch.setattr(huber, 'least_squares', lambda *args, **kwargs: Failed())
    with pytest.raises(RuntimeError, match='forced failure'):
        huber.huber_fit(satellites, satellites, h@np.ones(4), True)


@pytest.mark.local_evidence
def test_private_globals_baseline_and_prefix_causality():
    frozen = json.loads((study.OUTPUT/'summary.json').read_text(encoding='utf-8'))
    assert digest(study.OUTPUT/'summary.json') == huber.STAGE47_SUMMARY_SHA256
    replay_path = study.OUTPUT/'single_path_carrier_continuity_summary.json'
    assert digest(replay_path) == huber.REPLAY_SUMMARY_SHA256
    assert digest(Path(replay.__file__)) == json.loads(
        replay_path.read_text(encoding='utf-8'))['source_sha256']
    root = study.OUTPUT/'2026100404'/'weak'/'gate3'
    expected = frozen['cases']['2026100404']['weak']['paths']['gate3']['output_sha256']
    result, arrays = huber.run_path(root, expected, 'weak', 2.)
    assert result['baseline_reproduced']
    assert result['fit_attempts'] > 0
    assert result['chosen_fits'] == int(arrays['candidate_phase_ok'].sum())
    np.testing.assert_array_equal(arrays['baseline_selected'],
                                  arrays['candidate_selected'])
    assert np.array_equal(arrays['fit_chosen'],
                          arrays['fit_chosen'] & arrays['fit_accepted'])
    assert mechanism_phase.phase_prediction_leave_one_out.__globals__[
        'fit_carrier_displacement'] is mechanism_phase.fit_carrier_displacement
    assert single_source.navigate.__globals__[
        'phase_prediction_leave_one_out'] is single_source.phase_prediction_leave_one_out
    assert mechanism_smoothing.fit_carrier_displacement is mechanism_phase.fit_carrier_displacement

    tracking = replay.read_npz(root/'tracking.npz')
    trace = replay.read_npz(root/'trace.npz')
    support = replay.read_npz(root/'support.npz')['coherent_support']
    z = navigation_input(tracking, support)
    eligible = projection_eligibility(tracking['available_time'], trace)[0]
    prefix = {key: value[:80].copy() for key, value in z.items()}
    candidate, _ = huber.candidate_navigate(prefix, eligible[:80])
    np.testing.assert_array_equal(candidate['position'], arrays['candidate_position'][:80])
    for key in z:
        np.testing.assert_array_equal(z[key][:80], prefix[key])


def test_fit_array_chosen_links_to_actual_fit():
    record = dict(epoch=4, lag=2, excluded_prn=2,
                  prn_mask=np.array([True, False, True]),
                  chosen=True, accepted=True, geometry_pass=True,
                  covariance_pass=True, residual_pass=True,
                  weights=np.array([1., .5]), raw_residual=np.array([0., .04]),
                  ols_delta=np.zeros(4), objective=.001, gradient_inf=0.)
    arrays = huber.fit_arrays([record], 3)
    assert arrays['chosen'][0] and arrays['accepted'][0]
    assert arrays['lag'][0] == 2 and arrays['excluded_prn'][0] == 2
    np.testing.assert_array_equal(arrays['weights'][0, [0, 2]], [1., .5])
    assert np.isnan(arrays['weights'][0, 1])
