"""Measurement-domain fault transformations keep admission and causality explicit."""

import numpy as np

from experiments.single_source_float_validation import coherent_displacement, carrier_fault
from experiments.single_source_float_window_study import solve_history


def fixture():
    geometry = np.array([[1., 0., 0.], [-1., 0., 0.],
                         [0., 1., 0.], [0., -1., 0.],
                         [0., 0., 1.], [0., 0., -1.]])
    design = np.broadcast_to(geometry, (4, 6, 3)).copy()
    position = np.array([[0., 0., 0.], [1., 2., 0.], [2., 2., 0.], [3., 2., 0.]])
    clock = 75_000. + np.arange(4)
    code = np.einsum("tsd,td->ts", design, position) + clock[:, None]
    carrier = code + 1000 * np.arange(6)[None, :]
    valid = np.ones((4, 6), bool)
    arc = np.zeros((4, 6), int)
    return design, code, carrier, valid, arc


def test_coherent_false_motion_is_exactly_indistinguishable_in_linear_model():
    design, code, carrier, valid, arc = fixture()
    time = np.array([0., .1, .2, .3])
    original = solve_history(design, code, carrier, valid, arc)

    changed_code, changed_carrier, displacement = coherent_displacement(
        design, code, carrier, time, onset=.1, final_east_m=100.)
    changed = solve_history(design, changed_code, changed_carrier, valid, arc)

    np.testing.assert_array_equal(changed_code[0], code[0])
    np.testing.assert_array_equal(changed_carrier[0], carrier[0])
    np.testing.assert_allclose(changed["position_m"],
                               original["position_m"] + displacement, atol=1e-6)
    np.testing.assert_allclose(changed["formal_sigma_xyz_m"],
                               original["formal_sigma_xyz_m"], atol=1e-8)


def test_single_satellite_fault_only_changes_carrier_after_onset():
    _, code, carrier, valid, arc = fixture()
    time = np.array([0., .1, .2, .3])
    slipped = carrier_fault(carrier, time, satellite=2, onset=.2, kind="slip")
    drifted = carrier_fault(carrier, time, satellite=2, onset=.2, kind="drift")

    np.testing.assert_array_equal(slipped[:2], carrier[:2])
    np.testing.assert_array_equal(drifted[:2], carrier[:2])
    np.testing.assert_array_equal(slipped[:, :2], carrier[:, :2])
    np.testing.assert_allclose(slipped[2:, 2] - carrier[2:, 2],
                               [299792458. / 1575.42e6] * 2)
    np.testing.assert_allclose(drifted[2:, 2] - carrier[2:, 2], [0., .05])
    assert valid.all() and (arc == 0).all()
