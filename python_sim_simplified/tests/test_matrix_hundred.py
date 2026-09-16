import numpy as np

from matrix_hundred_core import homogeneous_fit, pilot_fit, projective_error, scene


def test_full_matrix_requires_excitation_and_recovers_heldout_response():
    d = scene()
    estimated, _ = homogeneous_fit(d['a'], d['clean'])
    assert projective_error(estimated @ d['test_a'], d['test_y']) < 1e-10
    assert np.linalg.norm(pilot_fit(np.eye(4), d['h']) - d['h']) < 1e-10
    absent = np.outer(d['v'], d['amplitudes'])
    false_fit, singular = homogeneous_fit(d['a'], absent)
    assert np.linalg.matrix_rank(false_fit, tol=1e-8) == 1
    assert sum(singular < 1e-8) >= 4


def test_projected_and_reciprocal_calibration_remain_unobservable():
    from matrix_hundred_043 import experiment as full
    from matrix_hundred_044 import experiment as reciprocal
    assert full()['projected_nullity'] == 5
    assert reciprocal()['nullity'] == 2


def test_reference_monitor_removes_common_path_not_navigation_error():
    from matrix_hundred_054 import experiment as ambiguous
    from matrix_hundred_055 import experiment as monitored
    a=ambiguous();m=monitored()
    assert a['indistinguishable_measurement_norm'] < 1e-12
    assert not a['identified']
    assert m['monitored_error'] < m['unmonitored_error']/20


def test_two_antennas_have_no_remaining_noise_optimization_dimension():
    from matrix_hundred_059 import experiment
    rows=experiment()['arrays']
    assert abs(rows[0]['noise_ratio']-1) < 1e-8
    assert rows[2]['noise_ratio'] < rows[0]['noise_ratio']


def test_low_residual_does_not_prove_authentic_support():
    from matrix_hundred_060 import experiment
    rows=experiment()['cases']
    for absent,authentic in zip(rows[::2],rows[1::2]):
        assert abs(absent['mean_two_component_normalized_residual']-authentic['mean_two_component_normalized_residual']) < .01
        assert absent['total_nested_bic_support'] < 0 < authentic['total_nested_bic_support']
