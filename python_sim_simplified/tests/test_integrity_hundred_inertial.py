import numpy as np

from integrity_hundred_inertial import make_case, receive, score


def test_unknown_initial_position_cannot_be_authenticated_by_matching_gnss():
    observed, truth = make_case('unknown', 991)
    result = receive(observed)
    assert not result['normal'].any()
    assert not score(result, truth)['all_epochs_le20']


def test_declared_domain_contains_truth_and_normal_output_is_useful():
    for name in ['normal', 'step', 'creep', 'outage', 'tilt', 'gyro',
                 'timing', 'lever', 'mounted', 'handheld', 'calibrated',
                 'drift', 'auth', 'zero_velocity', 'bias_calibration']:
        observed, truth = make_case(name, 992)
        result = receive(observed)
        metrics = score(result, truth)
        assert metrics['bound_violations'] == 0, (name, metrics)
        assert metrics['normal_error_ge30'] == 0, (name, metrics)
        assert metrics['normal_output_rate'] > 0, (name, metrics)


def test_ambiguous_hypotheses_need_independent_anchor():
    observed, truth = make_case('ambiguous', 993)
    assert not receive(observed)['normal'].any()
    observed, truth = make_case('resolved', 993)
    result = receive(observed)
    assert not result['normal'][:40].any()
    assert result['normal'][40:].any()


def test_translation_counterexample_has_identical_receiver_input():
    observed, truth = make_case('unknown', 994)
    result = receive(observed)
    translated_truth = truth + np.array([200., 0.])
    np.testing.assert_allclose(np.linalg.norm(truth - translated_truth, axis=1), 200.)
    assert not result['normal'].any()


def test_unknown_initialization_can_recover_only_with_independent_certificate():
    observed, truth = make_case('auth', 995)
    observed['initial_position_trusted'] = False
    result = receive(observed)
    assert not result['normal'][:40].any()
    assert result['normal'][40:].all()
    assert score(result, truth)['bound_violations'] == 0


def test_untrusted_gnss_cannot_move_final_integrity_center():
    observed, truth = make_case('normal', 996)
    original = receive(observed)
    observed['gnss'] += 200.
    spoofed = receive(observed)
    np.testing.assert_array_equal(original['position'], spoofed['position'])
    np.testing.assert_array_equal(original['bound'], spoofed['bound'])
    assert original['normal'].any()


def test_contradictory_independent_certificates_make_all_hypotheses_infeasible():
    observed, truth = make_case('ambiguous', 997)
    observed['absolute_certificates'][40] = (np.array([10000., 10000.]), np.ones(2))
    result = receive(observed)
    assert result['feasible'][:40].all()
    assert not result['feasible'][40:].any()
    assert not result['normal'][40:].any()
    assert np.isnan(result['position'][40:]).all()
    assert np.isinf(result['bound'][40:]).all()
