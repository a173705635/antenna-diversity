"""Causal code/carrier float-window observability checks."""

import numpy as np

from experiments.single_source_float_window_study import solve_window


GEOMETRY = np.array([[1., 0., 0.], [-1., 0., 0.],
                     [0., 1., 0.], [0., -1., 0.],
                     [0., 0., 1.], [0., 0., -1.]])


def observations(epochs=3):
    design = np.broadcast_to(GEOMETRY, (epochs, 6, 3)).copy()
    positions = np.column_stack((np.arange(epochs) * 2.,
                                 np.arange(epochs) * -.5,
                                 np.arange(epochs) * .25))
    clock = 75_000. + np.arange(epochs) * 3.
    code = np.einsum("tsd,td->ts", design, positions) + clock[:, None]
    carrier = code + np.arange(6)[None, :] * 1000.
    valid = np.ones((epochs, 6), bool)
    arc = np.zeros((epochs, 6), int)
    return design, code, carrier, valid, arc, positions


def test_noise_free_solution_and_arc_bias_are_separate():
    design, code, carrier, valid, arc, positions = observations()
    carrier[2, 0] += 90.
    arc[2, 0] = 1

    result = solve_window(design, code, carrier, valid, arc, target=2)

    assert result["observable"]
    np.testing.assert_allclose(result["position_m"], positions[2], atol=1e-6)
    assert result["formal_horizontal_std_m"] > 0
    assert result["arc_bias_count"] == 7


def test_current_position_can_be_observable_when_nuisance_state_is_rank_deficient():
    design, code, carrier, valid, arc, positions = observations(2)
    valid[0, 1:] = False

    result = solve_window(design, code, carrier, valid, arc, target=1)

    assert result["rank"] < result["state_count"]
    assert result["observable"]
    np.testing.assert_allclose(result["position_m"], positions[1], atol=1e-6)


def test_three_satellites_leave_current_position_unobservable():
    design, code, carrier, valid, arc, _ = observations(2)
    valid[:, 3:] = False

    result = solve_window(design, code, carrier, valid, arc, target=1)

    assert not result["observable"]
    assert np.isnan(result["position_m"]).all()
    assert result["target_null_norm"] >= 1e-7


def test_later_observations_do_not_change_past_target():
    design, code, carrier, valid, arc, _ = observations(3)
    earlier = solve_window(design[:2], code[:2], carrier[:2], valid[:2], arc[:2], target=1)
    code[2] += 100000.
    carrier[2] -= 100000.
    later = solve_window(design, code, carrier, valid, arc, target=1)

    np.testing.assert_array_equal(earlier["position_m"], later["position_m"])
    assert earlier["formal_horizontal_std_m"] == later["formal_horizontal_std_m"]
