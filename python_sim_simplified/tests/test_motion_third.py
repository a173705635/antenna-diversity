import numpy as np
from motion_third import estimate
from motion_third_research import simulate
from motion_third_validate import counterexamples


def test_stationary_and_uniform_translation_are_identical_and_unavailable():
    a,_=simulate('static',97631);b,_=simulate('uniform',97631)
    assert np.array_equal(a['phase'],b['phase'])
    assert np.array_equal(a['gyro'],b['gyro'])
    assert estimate(a)['status']=='indeterminate'
    assert not estimate(a)['absolute_position_available']


def test_far_field_common_body_lever_cancels():
    a,_=simulate('common',97631);b,_=simulate('body_lever',97631)
    assert np.array_equal(a['phase'],b['phase'])
    assert estimate(a)==estimate(b)


def test_rotation_recovers_conditional_direction_but_not_origin():
    o,tr=simulate('common',97631)
    r=estimate(o)
    assert r['status']=='common' and r['direction_available']
    assert np.degrees(np.arccos(np.clip(np.array(r['direction'])@tr['direction'],-1,1)))<1
    assert not r['absolute_position_available'] and not r['support_is_certificate']


def test_common_drift_cancels_in_prn_contrast():
    a,_=simulate('common',97631);b,_=simulate('drift',97631)
    assert estimate(a)['status']==estimate(b)['status']=='common'


def test_independent_calibration_is_required_not_silent_truth():
    o,_=simulate('bias',97631)
    calibrated=estimate(o)
    o['bias_readings']=np.zeros_like(o['bias_readings'])
    wrong=estimate(o)
    assert np.linalg.norm(np.array(wrong['direction'])-calibrated['direction'])>.01


def test_single_axis_rotation_has_missing_direction_axis():
    o,_=simulate('yaw',97631)
    assert estimate(o)['status']=='indeterminate'


def test_missing_declared_gyro_domain_withdraws_evidence():
    o,_=simulate('common',97631);o['gyro_certified']=False
    assert estimate(o)['status']=='indeterminate'
    assert not estimate(o)['direction_available']


def test_unresolved_close_directions_and_shared_drift_gauge_are_retained():
    r=counterexamples(97631)
    assert r['clustered_false_alarm']
    assert r['shared_drift_gauge_phase_difference']<1e-12
    assert r['indistinguishable_direction_separation_deg']>60


def test_joint_nuisance_projection_handles_slip_plus_linear_drift():
    o,_=simulate('slip',97631)
    r=estimate(o)
    slope=np.arange(21).reshape(1,3,7)*.001
    o['phase']=np.angle(np.exp(1j*(np.unwrap(o['phase'],axis=0)+o['time'][:,None,None]*slope)))
    q=estimate(o)
    assert abs(r['spread']-q['spread'])<1e-10


def test_estimator_has_no_truth_or_scene_inputs():
    from pathlib import Path
    source=(Path(__file__).parents[1]/'motion_third.py').read_text()
    assert "['truth']" not in source and "['case']" not in source
