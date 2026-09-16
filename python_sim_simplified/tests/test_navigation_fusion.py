import numpy as np
from navigation_fusion import fuse


def run(windows, admitted=True, use_spatial=True):
    time = np.arange(6.)
    inertial = dict(time_s=time, position_m=np.zeros((6, 3)), velocity_mps=np.zeros((6, 3)))
    return fuse(inertial, np.arange(1, 6), np.ones((5, 3))*100, np.zeros((5, 3)), windows,
                initial_covariance=np.eye(6), gnss_covariance=np.eye(6),
                acceleration_noise_intensity=np.eye(3)*.01,
                admitted=admitted, use_spatial=use_spatial)


def test_decision_latency_and_quiet_window_cannot_release_isolation():
    windows = [dict(decision_s=.9, state=0), dict(decision_s=2.1, state=1),
               dict(decision_s=3.1, state=0)]
    result = run(windows)
    np.testing.assert_array_equal(result['gnss_accepted'], [0, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(result['mode'], [-1, 0, 0, 1, 1, 1])


def test_unadmitted_start_never_uses_gnss_even_without_spatial_gate():
    result = run([dict(decision_s=.1, state=0)], admitted=False, use_spatial=False)
    assert not result['gnss_accepted'].any()
    assert (result['mode'] == -1).all()


def test_unknown_skips_update_and_covariance_is_positive():
    result = run([dict(decision_s=.1, state=0), dict(decision_s=1.5, state=-1)])
    np.testing.assert_array_equal(result['gnss_accepted'], [0, 1, 0, 0, 0, 0])
    assert np.linalg.eigvalsh(result['covariance']).min() > 0


def test_inertial_increments_preserve_constant_acceleration_without_updates():
    time = np.arange(11.)/10
    p = np.column_stack((time**2, time*0, time*0))
    v = np.column_stack((2*time, time*0, time*0))
    result = fuse(dict(time_s=time, position_m=p, velocity_mps=v), [], [], [], [],
                  initial_covariance=np.eye(6), gnss_covariance=np.eye(6),
                  acceleration_noise_intensity=np.zeros((3, 3)), admitted=False, use_spatial=True)
    np.testing.assert_allclose(result['position_m'], p, atol=1e-15)
    np.testing.assert_allclose(result['velocity_mps'], v, atol=1e-15)
