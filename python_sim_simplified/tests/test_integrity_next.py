"""Focused checks of inference boundaries and known failed claims."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from integrity_next import receive
from integrity_next_model import make_case, score
from integrity_next_validate import counterexamples

def test_gnss_cannot_create_independent_information():
    o,_=make_case('bounded',72200); first=receive(o)
    o['gnss']+=1e6; second=receive(o)
    np.testing.assert_array_equal(first['position'],second['position'])
    np.testing.assert_array_equal(first['bound'],second['bound'])

def test_no_hard_claim_from_missing_or_statistical_evidence():
    for name in ['unknown_origin','empirical_only','gaussian','real_phone_unavailable','holdout_fault']:
        o,_=make_case(name,72200)
        assert not receive(o)['normal'].any()

def test_fixture_and_accelerometer_stationarity_are_distinct():
    o,y=make_case('stationary',72200); assert score(receive(o),y)['full_success']
    o,y=make_case('false_stop',72200); good=receive(o)
    o['stationary_claim']=False
    np.testing.assert_array_equal(good['position'],receive(o)['position'])

def test_uncertified_anchor_recovery_only_at_fresh_independent_epoch():
    o,_=make_case('anchor_recovery',72200); r=receive(o)
    np.testing.assert_array_equal(np.flatnonzero(r['normal']),list(o['anchors']))
    for anchor in o['anchors'].values(): anchor['origin']='gnss'
    assert not receive(o)['normal'].any()

def test_clipping_and_expiry_remove_inertial_support():
    o,_=make_case('clipping',72200); assert np.isinf(receive(o)['bound'][100:]).all()
    o,_=make_case('expired',72200); assert np.isinf(receive(o)['bound'][41:]).all()

def test_delayed_transport_pulse_is_enclosed_after_history_correction():
    examples=counterexamples(72200)
    for name in ['delayed_anchor_acceleration_history','delayed_anchor_time_varying_history']:
        o,y=examples[name]; metric=score(receive(o),y)
        assert metric['normal_ge30']==0
        assert metric['bound_violations']==0
        assert metric['normal_output_rate']>0

def test_delayed_transport_unobserved_or_uncertified_history_unavailable():
    o,_=make_case('bounded',72200)
    o['initial_radius'][:]=1000
    o['anchors']={2:dict(z=np.zeros(3),radius=np.ones(3),lag=3,origin='independent_synthetic')}
    assert not receive(o)['normal'].any()
    o['envelope_origin']='empirical'
    o['anchors']={20:dict(z=np.zeros(3),radius=np.ones(3),lag=3,origin='independent_synthetic')}
    assert not receive(o)['normal'].any()

def test_synthetic_declared_bench_model_contains_truth_on_independent_seed():
    for name in ['bounded','drift','temperature','velocity','tilt3d','attitude','gyro','scale']:
        o,y=make_case(name,72200)
        assert score(receive(o),y)['bound_violations']==0
