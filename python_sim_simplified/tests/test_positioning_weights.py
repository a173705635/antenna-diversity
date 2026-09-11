import numpy as np
import pytest
from test_carrier_positioning import observations
from positioning import solve_position, solve_code_carrier_float


def test_weighted_code_and_float_reject_low_quality_duplicate():
    states, positions, _, code, carrier, _ = observations()
    s = np.concatenate([states, states], axis=1)
    p = np.concatenate([code+np.arange(8)*4., code], axis=1)
    l = np.concatenate([carrier, carrier+17.3], axis=1)
    sigma = np.r_[np.full(8, 1000.), np.ones(8)]
    plain = solve_position(s[0], p[0])['position_m']
    weighted = solve_position(s[0], p[0], sigma_m=sigma)['position_m']
    assert np.linalg.norm(plain-positions[0]) > 1
    assert np.linalg.norm(weighted-positions[0]) < .001
    result = solve_code_carrier_float(s, p, l, code_sigma_m=sigma,
                                      carrier_sigma_m=np.full(16, .01))
    assert np.max(abs(result['position_m']-positions)) < .001
    # Zero-information L1 codes must not move the solution, while L1 phase stays.
    sigma[:8] = np.inf
    result = solve_code_carrier_float(s, p, l, code_sigma_m=sigma)
    assert np.max(abs(result['position_m']-positions)) < .0001
    changed = p.copy()
    changed[:, :8] += 10000
    invariant = solve_code_carrier_float(s, changed, l, code_sigma_m=sigma)
    np.testing.assert_allclose(invariant['position_m'], result['position_m'], atol=1e-5)
    # L1 phase is still active: changing its time variation changes the result.
    moved_phase = l.copy()
    moved_phase[:, :8] += np.arange(5)[:, None]*np.arange(8)[None, :]*.01
    active = solve_code_carrier_float(s, p, moved_phase, code_sigma_m=sigma)
    assert np.linalg.norm(active['position_m']-result['position_m']) > .01
    with pytest.raises(ValueError):
        solve_code_carrier_float(s, p, l, code_sigma_m=np.full(16, np.inf))
    with pytest.raises(ValueError):
        solve_position(s[0], p[0], sigma_m=np.zeros(16))
