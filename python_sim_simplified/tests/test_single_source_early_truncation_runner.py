import numpy as np
import pytest

from experiments.single_source_early_truncation import (
    CASES, compare_arrays, near_tail_energy_check, saved_arrays, verify_inputs,
)


def test_scientific_array_comparison_preserves_dtype_and_nan_position():
    actual = {'x': np.array([1., np.nan]), 'y': np.array([1, 2])}
    compare_arrays(actual, {'x': actual['x'].copy(), 'y': actual['y'].copy()})
    with pytest.raises(AssertionError):
        compare_arrays(actual, {'x': np.array([1., 0.]), 'y': actual['y']})
    with pytest.raises(AssertionError):
        compare_arrays(actual, {'x': actual['x'], 'y': actual['y'].astype(float)})


@pytest.mark.local_evidence
def test_historical_snapshots_and_near_tail_energy_recompute():
    verify_inputs()
    for _, _, reference, _ in CASES:
        observations = dict(saved_arrays(reference/'tracking.npz'),
                            **saved_arrays(reference/'trace.npz'))
        history = saved_arrays(reference/'history.npz')
        assert near_tail_energy_check(observations, history) == len(
            observations['time'])-5
