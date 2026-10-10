"""Causal projector history, including its direction-change failure mode."""
import numpy as np

from experiments.single_source_subspace_history import smooth_nulls


def test_constant_direction_and_arbitrary_complex_phase():
    vector = np.array([1., 1j, 2.]) / np.sqrt(6.)
    directions = np.tile(vector, (8, 1))
    original = smooth_nulls(directions, np.ones(8, bool))
    rotated = smooth_nulls(directions * np.exp(1j * np.arange(8))[:, None], np.ones(8, bool))
    np.testing.assert_allclose(original[0], rotated[0], atol=1e-14)
    np.testing.assert_allclose(original[0] @ vector, 0, atol=1e-14)
    assert original[1].tolist() == [1, 2, 3, 4, 5, 5, 5, 5]


def test_reset_and_no_future_leakage():
    directions = np.array([[1., 0, 0]] * 5 + [[.5, np.sqrt(.75), 0]] * 5)
    trigger = np.ones(10, bool)
    trigger[4] = False
    full = smooth_nulls(directions, trigger)
    prefix = smooth_nulls(directions[:7], trigger[:7])
    np.testing.assert_array_equal(full[0][:7], prefix[0])
    np.testing.assert_array_equal(full[0][4], np.eye(3))
    assert full[1][4:7].tolist() == [0, 1, 2]
    np.testing.assert_allclose(full[0][5] @ directions[5], 0, atol=1e-14)


def test_direction_step_retains_interference_during_history_turnover():
    directions = np.array([[1., 0, 0]] * 5 + [[.5, np.sqrt(.75), 0]] * 5)
    projectors, _, _ = smooth_nulls(directions, np.ones(10, bool))
    power = np.linalg.norm(np.einsum('bij,bj->bi', projectors, directions), axis=1)**2
    np.testing.assert_allclose(power[5:], [.6386750490563073, .4055088817476933,
                                         .12203552699077258, .014637328302924635, 0], atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(projectors[4] @ directions[5])**2, .75)
