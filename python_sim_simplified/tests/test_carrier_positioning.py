import numpy as np
import pytest


def observations():
    from core.gps_simulation import satellite_setup, receiver_reference
    times = np.arange(5)*.01
    satellites = satellite_setup()
    states = np.array([[s['position_m']+s['velocity_mps']*t for s in satellites] for t in times])
    origin, _ = receiver_reference()
    positions = origin+times[:, None]*np.array([3., -2., 1.])
    clocks = 75000.+times*10.
    code = np.linalg.norm(states-positions[:, None, :], axis=2)+clocks[:, None]
    biases = np.arange(8)*5.123+12345.678
    return states, positions, clocks, code, code+biases, biases


def test_float_joint_recovery_and_phase_origin_invariance():
    from core.positioning import solve_code_carrier_float
    states, positions, clocks, code, carrier, biases = observations()
    result = solve_code_carrier_float(states, code, carrier)
    np.testing.assert_allclose(result['position_m'], positions, atol=1e-4)
    np.testing.assert_allclose(result['clock_bias_m'], clocks, atol=1e-4)
    np.testing.assert_allclose(result['ambiguity_m'], biases, atol=1e-4)
    offset = np.array([10.32, -123., 45.1, 1e5, 2., -30., 66., .032])
    shifted = solve_code_carrier_float(states, code, carrier+offset)
    np.testing.assert_allclose(shifted['position_m'], result['position_m'], atol=1e-4)
    np.testing.assert_allclose(shifted['ambiguity_m']-result['ambiguity_m'], offset, atol=1e-4)


def test_joint_rejects_degenerate_geometry():
    from core.positioning import solve_code_carrier_float
    states, _, _, code, carrier, _ = observations()
    with pytest.raises(ValueError):
        solve_code_carrier_float(np.repeat(states[:, :1], 8, axis=1), code, code)


def test_common_time_carrier_sign_and_block_frequency():
    from core.positioning import carrier_range
    history = {'success': True, 'epoch_start_samples': np.array([10, 20]),
               'carrier_nco_phase_rad': np.array([3., 5.]),
               'carrier_nco_frequency_hz': np.array([100., 150.])}
    actual = carrier_range(history, np.array([.009, .013, .022, .031]), 1000., .01, .19)
    assert np.isnan(actual[[0, 3]]).all()
    np.testing.assert_allclose(actual[1:3], -np.array([3.+2*np.pi*100*.003, 5.+2*np.pi*150*.002])*.19/(2*np.pi))


def test_carrier_constrains_motion_without_static_receiver_prior():
    from core.positioning import solve_code_carrier_float, solve_position
    states, positions, _, code, carrier, _ = observations()
    noisy = code+np.random.default_rng(123).normal(0, 3., code.shape)
    baseline = np.array([solve_position(s, p)['position_m'] for s, p in zip(states, noisy)])
    result = solve_code_carrier_float(states, noisy, carrier)
    before = baseline-positions
    after = result['position_m']-positions
    before -= before.mean(axis=0)
    after -= after.mean(axis=0)
    assert np.linalg.norm(after) < .001
    assert np.linalg.norm(before) > 1.
