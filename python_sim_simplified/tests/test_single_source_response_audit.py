"""Analytic checks for the offline complex-response audit."""

import numpy as np

from experiments.spoofing.single_source_complex_response_audit import effective_response, phase_changes


def test_effective_response_and_noise_gain_use_calibration_not_physical_noise_scaling():
    weights = np.array([[[1 + 1j], [2 - 1j]]])
    steering = np.array([[1 + 0j], [0 + 1j]])
    physical_gain = np.array([2j, .5])
    calibration = np.array([1j, -1j])
    rate = np.array([0., 0.])

    response, noise_gain = effective_response(
        weights, steering, physical_gain, calibration, rate, np.array([.01]))

    expected = np.vdot(weights[0, :, 0],
                        (physical_gain / calibration) * steering[:, 0])
    np.testing.assert_allclose(response[0, 0], expected)
    np.testing.assert_allclose(noise_gain[0, 0], 2. + 5.)


def test_boundary_jump_excludes_continuous_rate_drift():
    weights = np.ones((2, 1, 1), complex)
    steering = np.ones((1, 1), complex)
    physical_gain = calibration = np.ones(1, complex)
    rate = np.array([10.])

    boundary, midpoint = phase_changes(
        weights, steering, physical_gain, calibration, rate,
        np.array([.01, .03]), np.array([.02]))

    np.testing.assert_allclose(boundary, [[0.]], atol=1e-14)
    np.testing.assert_allclose(midpoint, [[-.2]], atol=1e-14)
