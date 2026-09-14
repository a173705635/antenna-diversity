import numpy as np
import pytest

import spoofing

W = 299792458 / 1575.42e6


def api():
    # An explicit assertion makes the initial RED a missing-feature failure.
    import importlib.util
    assert importlib.util.find_spec('spatial_spoofing') is not None, 'multi-antenna detector missing'
    import spatial_spoofing
    return spatial_spoofing


@pytest.mark.parametrize('antennas', [2, 3, 4])
def test_common_source_and_arbitrary_channel_and_satellite_offsets(antennas):
    detector = api()
    rng = np.random.default_rng(18)
    x = rng.normal(size=(10, antennas, 1)) + rng.normal(size=(10, 1, 8))
    x += rng.integers(-8, 9, x.shape) * W / 2
    scores = detector.spatial_scores(x, W)
    assert scores['global'] < 1e-12
    assert scores['cluster3'] < 1e-12


def test_two_antenna_global_is_existing_statistic():
    detector = api()
    x = np.random.default_rng(1).normal(0, .06, (5, 10, 2, 8))
    expected = spoofing.spatial_dispersion(x[:, :, 1] - x[:, :, 0], W).mean(axis=1)
    np.testing.assert_allclose(detector.spatial_scores(x, W)['global'], expected)


def test_partial_three_source_detectable_and_permutation_invariant():
    detector = api()
    x = np.broadcast_to(np.random.default_rng(2).normal(0, .05, (1, 4, 8)), (10, 4, 8)).copy()
    x[:, :, :3] = np.array([.01, .02, -.03, .04])[None, :, None]
    scores = detector.spatial_scores(x, W)
    assert scores['global'] > .2
    assert scores['cluster3'] < 1e-12
    permuted = detector.spatial_scores(x[:, [2, 0, 3, 1]][:, :, [7, 3, 1, 5, 0, 2, 6, 4]], W)
    for key in scores:
        np.testing.assert_allclose(scores[key], permuted[key], atol=1e-12)


def test_bad_data_and_partial_windows_unknown_and_decision_causal():
    detector = api()
    t = np.arange(35) * .01
    x = np.zeros((35, 3, 8)); v = np.ones(x.shape, bool)
    v[12, 2, 1] = False
    x[22, 1, 4] = np.nan
    rows = detector.detect_windows(t, x, v, .1, W, method='cluster3')
    assert [r['state'] for r in rows] == [1, -1, -1, -1]
    assert rows[0]['decision_s'] >= t[9]
    changed = np.zeros_like(x); changed[10:] = 10
    assert detector.detect_windows(t, changed, np.ones(x.shape, bool), .1, W)[0] == rows[0]


def test_gap_does_not_become_evidence_and_bad_contracts_rejected():
    detector = api()
    x = np.zeros((20, 4, 8)); t = np.arange(20) * .01; t[5:] += .01
    rows = detector.detect_windows(t, x, np.ones(x.shape, bool), .1, W)
    assert [r['state'] for r in rows] == [-1, 1]
    for bad in [x[:, :1], x[:, :, :2], np.full_like(x, np.nan)]:
        with pytest.raises(ValueError): detector.spatial_scores(bad, W)
    with pytest.raises(ValueError): detector.detect_windows(t, x, np.ones(x.shape), .1, W)
    with pytest.raises(ValueError): detector.detect_windows(t[::-1], x, np.ones(x.shape, bool), .1, W)
    with pytest.raises(ValueError): detector.detect_windows(t, x, np.ones(x.shape, bool), .1, W, method='typo')


def test_calibration_uses_only_lower_tail_and_strict_comparison():
    detector = api()
    values = np.arange(1000) / 1000
    threshold = detector.calibrate_threshold(values, .01)
    assert threshold == .009
    assert np.mean(values < threshold) <= .01
    with pytest.raises(ValueError): detector.calibrate_threshold([1, np.nan], .01)


def test_tracking_adapter_refuses_prn_or_epoch_mismatch():
    detector = api()
    assert hasattr(detector, 'observations_from_records'), 'record adapter missing'
    from run_gps_spp import serializable
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    record = json.loads((root / 'docs/reports/data/2026-09-12-geometry-free-rf/L1CA_0.json').read_text())
    t, x, v = detector.observations_from_records([record, record], W)
    assert x.shape == v.shape == (len(t), 2, 8)
    np.testing.assert_allclose(x[:, 0], x[:, 1], equal_nan=True)
    other = json.loads(json.dumps(serializable(record)))
    other['satellite_tracking'] = other['satellite_tracking'][::-1]
    with pytest.raises(ValueError): detector.observations_from_records([record, other], W)
    other = dict(record, time_s=np.asarray(record['time_s']) + .001)
    with pytest.raises(ValueError): detector.observations_from_records([record, other], W)


def test_integrity_gate_does_not_retroactively_reject_earlier_positions():
    detector = api()
    assert hasattr(detector, 'output_gate'), 'causal gate missing'
    rows = [{'state': 0, 'decision_s': .3}, {'state': 1, 'decision_s': .4},
            {'state': -1, 'decision_s': .5}, {'state': 0, 'decision_s': .6}]
    t = np.array([.2, .3, .35, .4, .45, .5, .6])
    # Hold the last completed decision. Unknown before first complete window.
    assert detector.output_gate(t, rows).tolist() == [-1, 0, 0, 1, 1, -1, 0]
