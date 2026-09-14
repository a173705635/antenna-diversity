import numpy as np
import pytest

from spatial_direction import double_difference_scores, estimate_direction

W = 299792458 / 1575.42e6


def wave(positions, az=100., el=30., prns=5):
    az, el = np.deg2rad([az, el])
    u = [np.cos(el)*np.sin(az), np.cos(el)*np.cos(az), np.sin(el)]
    return np.broadcast_to((-np.asarray(positions) @ u)[None, :, None], (10, len(positions), prns)).copy()


def test_double_difference_erases_common_source_direction_and_channel_bias():
    p = np.array([[0, 0, 0], [.0759, 0, 0], [0, -.1609, 0], [.0759, -.1609, 0]])
    a, b = wave(p), wave(p, 220, 60)
    b += np.array([0, .03, -.07, .11])[None, :, None]
    for x in (a, b):
        assert max(double_difference_scores(x, W).values()) < 1e-12


def test_prn_dependent_bias_survives_double_difference():
    p = np.array([[0, 0, 0], [.0759, 0, 0], [0, -.1609, 0]])
    a = wave(p)
    a[:, 1, :] += np.array([0, .01, -.02, .04, -.01])
    assert double_difference_scores(a, W)['global'] > .1


def test_small_array_calibrated_direction_and_planar_mirror():
    p = np.array([[0, 0, 0], [.04, 0, 0], [0, -.04, 0], [.04, -.04, 0]])
    row = estimate_direction(wave(p), p, W, channel_bias_m=np.zeros(4), harmonic=1)
    assert row['status'] == 'single_upper_candidate'
    assert len(row['candidates']) == 2
    upper = max(row['candidates'], key=lambda c: c['elevation_deg'])
    assert abs(upper['azimuth_deg'] - 100) < .01
    assert abs(upper['elevation_deg'] - 30) < .01


def test_phone_array_keeps_half_cycle_aliases():
    p = np.array([[0, 0, 0], [.0759, 0, 0], [0, -.1609, 0], [.0759, -.1609, 0]])
    row = estimate_direction(wave(p), p, W, channel_bias_m=np.zeros(4), harmonic=2)
    assert row['status'] == 'ambiguous'
    assert len(row['candidates']) > 2
    assert any(abs(c['azimuth_deg']-100)<.01 and abs(c['elevation_deg']-30)<.01 for c in row['candidates'])


def test_unknown_calibration_and_two_antennas_cannot_claim_doa():
    p = np.array([[0, 0, 0], [.0759, 0, 0], [0, -.1609, 0]])
    assert estimate_direction(wave(p), p, W)['status'] == 'uncalibrated'
    assert estimate_direction(wave(p[:2]), p[:2], W, channel_bias_m=np.zeros(2))['status'] == 'rank_deficient'


def test_missing_phase_is_unknown():
    p = np.array([[0, 0, 0], [.0759, 0, 0], [0, -.1609, 0]])
    x = wave(p); x[0, 1, 0] = np.nan
    assert estimate_direction(x, p, W, channel_bias_m=np.zeros(3))['status'] == 'invalid_observation'
