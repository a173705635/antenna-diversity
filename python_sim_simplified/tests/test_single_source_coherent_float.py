"""Fixed-support float replay uses independent solvers and explicit clocks."""
import numpy as np

from experiments.spoofing.single_source_coherent_float import (
    float_metrics, observation_masks, score_pair, solve_pair,
)


def test_float_modes_only_exchange_support_and_start_separate_solvers():
    original = dict(support=np.array([[4.], [3.]]),
                    ranges=np.array([[1.], [2.]]),
                    carrier=np.array([[3.], [4.]]), arc=np.array([[0], [0]]))
    snapshot = {name: value.copy() for name, value in original.items()}
    candidate = np.array([[9.], [np.nan]])
    calls = []

    def geometry(observations):
        for name in ('ranges', 'carrier', 'arc'):
            np.testing.assert_array_equal(observations[name], snapshot[name])
        assert 'position_m' not in observations
        return (observations['support'], snapshot['ranges'], snapshot['carrier'],
                observations['support'] >= 3.)

    def solver(design, code, carrier, valid, arc):
        calls.append((design.copy(), valid.copy()))
        return dict(position_m=np.full((2, 3), len(calls), dtype=float))

    result = solve_pair(original, candidate, geometry, solver)
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0][0], snapshot['support'])
    np.testing.assert_array_equal(calls[1][0], candidate)
    np.testing.assert_array_equal(result['original'][0]['position_m'], np.ones((2, 3)))
    np.testing.assert_array_equal(result['coherent'][0]['position_m'], np.full((2, 3), 2.))
    for name, value in snapshot.items():
        np.testing.assert_array_equal(original[name], value)


def test_float_masks_keep_observation_and_availability_opportunities_distinct():
    time = np.array([.28, .30, .31, .32])
    available = np.array([.30, .32, .33, .34])
    observed, available_mask, tail = observation_masks(time, available, .30)
    np.testing.assert_array_equal(observed, [False, True, True, True])
    np.testing.assert_array_equal(available_mask, [True, True, True, True])
    assert tail.all()
    error = np.array([100., 1., 21., 31.])
    std = np.array([1., 2., 11., 3.])
    values = float_metrics(error, std, observed & (std <= 10.))
    assert values['count'] == 2
    assert values['error_ge_20'] == 1
    assert values['error_ge_30'] == 1
    assert values['max_error_m'] == 31.


def test_common_observable_uses_same_epochs_for_both_modes():
    observations = dict(time=np.array([.29, .30, .31, .32]),
                        available_time=np.array([.31, .32, .33, .34]))
    metadata = dict(case=dict(attack_start=.30, velocity_east=0.))
    def fit(east, observable):
        return dict(position_m=np.column_stack((east, np.zeros((4, 2)))),
                    observable=np.array(observable),
                    formal_horizontal_std_m=np.ones(4))
    pair = dict(original=(fit([0., 1., 2., 3.], [True, True, False, True]),
                          np.ones((4, 1), bool)),
                coherent=(fit([0., 10., 20., 30.], [True, False, True, True]),
                          np.ones((4, 1), bool)))
    scored, _, post, available, _ = score_pair(observations, metadata, pair)
    assert post.sum() == 3 and available.sum() == 4
    assert scored['original']['all_observable']['count'] == 2
    assert scored['coherent']['all_observable']['count'] == 2
    assert scored['original']['common_observable']['count'] == 1
    assert scored['coherent']['common_observable']['count'] == 1
    assert scored['original']['common_observable']['max_error_m'] == 3.
    assert scored['coherent']['common_observable']['max_error_m'] == 30.
