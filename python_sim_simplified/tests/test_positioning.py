import numpy as np
import pytest

from config import SPEED_OF_LIGHT_MPS, TrackingConfig
from tracking import track


def test_position_and_clock_recovery_and_degenerate_geometry():
    from positioning import solve_position
    receiver = np.array([4e6, 3e6, 4e6])
    satellites = np.array([[2e7, 0, 1e7], [0, 2e7, 1e7], [-2e7, 0, 1e7],
                           [0, -2e7, 1e7], [1e7, 1e7, 2e7], [1e7, 1e7, -2e7]])
    ranges = np.linalg.norm(satellites-receiver, axis=1) + 75000.
    solution = solve_position(satellites, ranges)
    np.testing.assert_allclose(solution['position_m'], receiver, atol=1e-5)
    assert abs(solution['clock_bias_m']-75000.) < 1e-5
    with pytest.raises(ValueError):
        solve_position(satellites[:3], ranges[:3])
    with pytest.raises(ValueError):
        solve_position(np.tile(satellites[0], (5, 1)), np.repeat(ranges[0], 5))
    with pytest.raises(ValueError):
        solve_position(satellites, ranges * np.nan)


def test_code_range_common_time_and_period_boundary():
    from positioning import code_pseudorange
    fs = 4.092e6
    starts = np.array([4092, 8184, 12276])
    # Range crosses a 1 ms ambiguity boundary between observation epochs.
    base_range = 70 * .001 * SPEED_OF_LIGHT_MPS - 2.
    velocity = 1500.
    t = starts/fs
    rate = 1.023e6*(1-velocity/SPEED_OF_LIGHT_MPS)
    history = {'success': True, 'epoch_start_samples': starts,
               'code_nco_phase_chips': (t*1.023e6-(base_range+velocity*t)*1.023e6/SPEED_OF_LIGHT_MPS) % 1023,
               'code_nco_rate_hz': np.full(3, rate)}
    query = t + .0003
    actual = code_pseudorange(history, query, fs, .001, base_range+10000.)
    np.testing.assert_allclose(actual, base_range+velocity*query, atol=1e-5)
    # Coarse aiding is used only to choose the period, not to correct fine error.
    np.testing.assert_allclose(code_pseudorange(history, query, fs, .001, base_range-10000.), actual)
    history['code_nco_phase_chips'] += .1
    shifted = code_pseudorange(history, query, fs, .001, base_range)
    np.testing.assert_allclose(shifted-actual, -.1*SPEED_OF_LIGHT_MPS/1.023e6, atol=1e-5)
    assert np.isnan(code_pseudorange(history, np.array([0., .005]), fs, .001, base_range)).all()


def test_invalid_tracking_returns_failure_without_nan_or_exception():
    for count in (16368, 20460, 24552):
        result = track(np.zeros(count, dtype=complex), 4.092e6, 1,
                       {'code_phase_samples': 0, 'doppler_hz': 0.},
                       TrackingConfig(), 1575.42e6, 1.023e6)
        assert not result['success']
        assert result['status'] in ('insufficient_samples', 'invalid_signal', 'invalid_correlation')


def test_multisatellite_source_matches_geometry_and_sums_before_rf():
    from run_gps_spp import build_scene, satellite_setup, receiver_reference
    reference, _ = receiver_reference()
    satellites = satellite_setup()
    signal, config = build_scene(satellites[:2], duration_ms=10)
    first, _ = build_scene(satellites[:1], duration_ms=10)
    second, _ = build_scene(satellites[1:2], duration_ms=10)
    np.testing.assert_allclose(signal, first+second, atol=1e-18)
    assert signal.shape == (1, 330000)
    assert not config.interferers and not config.signal.multipath_paths
    for satellite in satellites:
        distance = np.linalg.norm(satellite['position_m']-reference)
        assert abs(distance+75000.-satellite['range_at_zero_m']) < 1e-7
        assert abs(satellite['coarse_range_m']-satellite['range_at_zero_m']) <= 5000.
        assert np.isclose(np.linalg.norm(satellite['position_m']), 26560000.)
        step = .1
        later = np.linalg.norm(satellite['position_m']+satellite['velocity_mps']*step-reference)
        assert abs((later-distance)/step+satellite['doppler_hz']*SPEED_OF_LIGHT_MPS/1575.42e6) < 1e-6


def test_continuous_source_is_diagnostic_opt_in():
    from run_gps_spp import build_scene, satellite_setup
    satellites = satellite_setup()[:1]
    old, default = build_scene(satellites, 10)
    continuous, control = build_scene(satellites, 10, code_bandwidth_hz=1.8e6)
    assert default.signal.code_bandwidth_hz is None
    assert control.signal.code_bandwidth_hz == 1.8e6
    assert old.shape == continuous.shape and not np.allclose(old, continuous, atol=1e-14)
