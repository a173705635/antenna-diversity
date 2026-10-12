import numpy as np
import pytest

from tools.spoofing import nominal_yaw_transition_shape as shape


def synthetic_paths():
    fs, width = 1000, 20
    samples = np.arange(300, 701, 10)
    time = samples/fs
    truth = np.column_stack((2*time, np.zeros(len(time))))
    baseline = truth + np.column_stack((time, np.zeros(len(time))))
    drift_minus = np.column_stack((time+.2*(time > .4)
                                   +np.maximum(time-.42, 0.), 2*time))
    minus = truth+drift_minus
    paths = {'+0': baseline, '-2': minus, '+2': baseline.copy()}
    active = {yaw: np.zeros(40, bool) for yaw in paths}
    active['-2'][20:] = True
    return fs, width, samples, paths, active


def test_isolated_switch_preserves_components_and_pairing():
    fs, width, samples, paths, active = synthetic_paths()
    result = shape.transition_shape(samples, paths, active, fs, width)
    minus = result['paths']['-2']
    assert minus['counts'] == {'on': 1, 'off': 0, 'boundary': 0,
                               'dense': 0, 'missing_common': 0}
    event = minus['events'][0]
    assert event['boundary_sample'] == 400
    np.testing.assert_allclose(event['before_velocity_en_mps'], [1., 2.], atol=1e-12)
    np.testing.assert_allclose(event['after20_excess_en_m'], [.2, 0.], atol=1e-12)
    np.testing.assert_allclose(event['after80_velocity_change_en_mps'], [1., 0.], atol=1e-12)
    np.testing.assert_allclose(event['paired_before_velocity_en_mps'], [0., 2.], atol=1e-12)
    np.testing.assert_allclose(event['paired_after20_excess_en_m'], [.2, 0.], atol=1e-12)
    assert minus['by_direction']['on']['count'] == 1


def test_dense_and_missing_common_are_not_shortened():
    fs, width, samples, paths, active = synthetic_paths()
    active['+0'][22:] = True
    dense = shape.transition_shape(samples, paths, active, fs, width)
    assert dense['paths']['-2']['counts']['dense'] == 1
    assert dense['paths']['-2']['events'] == []

    fs, width, samples, paths, active = synthetic_paths()
    paths['+2'][10] = np.nan
    missing = shape.transition_shape(samples, paths, active, fs, width)
    assert missing['paths']['-2']['counts']['missing_common'] == 1
    assert missing['paths']['-2']['events'] == []


def test_record_boundary_and_half_open_evidence_mapping():
    fs, width, samples, paths, active = synthetic_paths()
    active['-2'][:] = True
    active['-2'][15:] = False
    result = shape.transition_shape(samples, paths, active, fs, width)
    assert result['paths']['-2']['counts']['boundary'] == 1
    assert result['paths']['-2']['counts']['off'] == 0
    assert shape.evidence_blocks(400, width) == [(19, 20)]
    assert shape.evidence_blocks(410, width) == [(19, 10), (20, 10)]
    assert shape.evidence_blocks(420, width) == [(20, 20)]
