"""Fixed PRN-14 carrier faults preserve observations before the fault."""
import numpy as np
import pytest

from core.recovery_geometry import WAVELENGTH
from experiments.single_source_carrier_fault import (
    assert_prefault_equal, compare_fault, inject_carrier_fault,
)


@pytest.mark.parametrize('kind,expected', [
    ('cycle', [0., 0., WAVELENGTH, WAVELENGTH, 0.]),
    ('drift', [0., 0., 0., .005, 0.]),
])
def test_fault_only_changes_prn14_arc0_carrier_at_or_after_two_seconds(kind, expected):
    record = dict(time=np.array([1.98, 1.99, 2., 2.01, 2.02]),
                  available_time=np.array([2., 2.01, 2.02, 2.03, 2.04]),
                  startup_prns=np.array([4, 13, 20]),
                  arc=np.zeros((5, 24), int),
                  carrier=np.ones((5, 24)), ranges=np.full((5, 24), 10.))
    record['arc'][4, 13] = 1
    fault, delta = inject_carrier_fault(record, kind)
    np.testing.assert_allclose(delta, expected)
    np.testing.assert_allclose(fault['carrier'][:, 13] - record['carrier'][:, 13], expected)
    other_columns = [index for index in range(24) if index != 13]
    np.testing.assert_array_equal(fault['carrier'][:, other_columns],
                                  record['carrier'][:, other_columns])
    for key in record:
        if key != 'carrier':
            np.testing.assert_array_equal(fault[key], record[key])


def test_prn14_must_be_in_acquired_startup_set():
    record = dict(time=np.array([2.]), startup_prns=np.array([4, 14, 20]),
                  arc=np.zeros((1, 24), int), carrier=np.ones((1, 24)))
    with pytest.raises(AssertionError):
        inject_carrier_fault(record, 'cycle')


def test_drift_requested_at_two_seconds_can_round_to_zero_in_carrier():
    record = dict(time=np.array([2. + 2e-15, 2.01]),
                  startup_prns=np.array([13]), arc=np.zeros((2, 24), int),
                  carrier=np.full((2, 24), 1e7))
    changed, requested = inject_carrier_fault(record, 'drift')
    actual = changed['carrier'][:, 13] - record['carrier'][:, 13]
    assert requested[0] > 0.
    assert actual[0] == 0.
    assert actual[1] > 0.


def test_prefault_comparison_covers_navigation_and_diagnostics():
    baseline = dict(position=np.array([[1.], [2.], [3.]]),
                    initialization_time=1.,
                    diagnostics=dict(reason=np.array(['a', 'b', 'c'])))
    fault = dict(position=np.array([[1.], [2.], [99.]]),
                 initialization_time=1.,
                 diagnostics=dict(reason=np.array(['a', 'b', 'changed'])))
    assert_prefault_equal(baseline, fault, np.array([True, True, False]))
    fault['diagnostics']['reason'][1] = 'changed'
    with pytest.raises(AssertionError):
        assert_prefault_equal(baseline, fault, np.array([True, True, False]))


def test_fault_comparison_uses_fault_window_and_same_blend_baseline():
    observations = dict(time=np.array([1.99, 2., 2.01]),
                        available_time=np.array([2.01, 2.02, 2.03]))
    baseline = dict(position=np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]))
    fault = dict(position=np.array([[100., 0., 0.], [3., 0., 0.], [np.nan] * 3]))
    metadata = dict(case=dict(attack_start=.3, velocity_east=0.))
    comparison, _ = compare_fault(baseline, fault, observations, metadata,
                                  np.array([False, True, True]))
    assert comparison['fault_window_opportunities'] == 2
    assert comparison['common_published'] == 1
    assert comparison['lost_publication'] == 1
    assert comparison['newly_published'] == 0
    assert comparison['position_displacement_m']['max_error_m'] == 2.
    assert comparison['baseline_common']['max_error_m'] == 1.
    assert comparison['fault_common']['max_error_m'] == 3.
