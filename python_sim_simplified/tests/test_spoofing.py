import numpy as np
import pytest
from spoofing import spatial_dispersion, window_evidence

W = 299792458/1575.42e6


def test_common_bias_and_half_cycles_cancel_without_startup_calibration():
    d = np.arange(8)*W/2 + .043
    assert spatial_dispersion(d, W) < 1e-12
    spread = np.arange(8)*W/16
    assert spatial_dispersion(spread, W) > .99
    assert spatial_dispersion(spread+.37, W) == pytest.approx(spatial_dispersion(spread, W))


def test_missing_and_partial_blocks_are_unknown_not_clean():
    t = np.arange(25)*.01; a = np.zeros((25, 8)); v = np.ones_like(a, dtype=bool)
    v[12, 0] = False
    rows = window_evidence(t, a, a+.02, v, .1, W)
    assert [r['state'] for r in rows] == [1, -1, -1]


def test_gap_and_nan_are_unknown():
    t = np.arange(20)*.01; t[5:] += .01
    a = np.zeros((20, 8)); b = a.copy(); b[12, 0] = np.nan
    assert [r['state'] for r in window_evidence(t, a, b, np.ones_like(a, bool), .1, W)] == [-1, -1]


def test_rejects_nonboolean_validity_and_wrong_constellation():
    a = np.zeros((10, 8)); t = np.arange(10)*.01
    with pytest.raises(ValueError): window_evidence(t, a, a, np.ones_like(a), .1, W)
    with pytest.raises(ValueError): spatial_dispersion([np.nan, 0], W)
    with pytest.raises(ValueError): window_evidence(t, a[:, :7], a[:, :7], np.ones((10, 7), bool), .1, W)
