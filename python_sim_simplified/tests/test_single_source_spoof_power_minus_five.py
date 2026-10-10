"""The fixed minus-five-dB pair changes spoof power only."""
from copy import deepcopy

import numpy as np
import pytest

from core import single_source_scene
from core.recovery_geometry import FS
from experiments.single_source_history_study import generate_new_record
from experiments.single_source_spoof_power_minus_five import (
    audit_new_fit,verify_power_pair,
)


@pytest.mark.parametrize('scenario',('weak','genuine_absent'))
def test_minus_five_db_pair_preserves_rng_and_clean_iq(monkeypatch,scenario):
    plan=dict(seed=2026100301,duration_s=.25,satellites=2,
              startup_cn0_db_hz=45.,post_0p3s_cn0_db_hz=30.,
              attack_start_s=.15,spoof_db=-5.,angle_deg=5.,fake_east_m=100.)
    original=np.random.default_rng
    generators=[]

    def tracked(seed):
        generator=original(seed)
        generators.append(generator)
        return generator

    monkeypatch.setattr(single_source_scene.np.random,'default_rng',tracked)
    candidate,candidate_identity=generate_new_record(plan,scenario)
    candidate_rng=[deepcopy(item.bit_generator.state) for item in generators]
    generators.clear()
    plan['spoof_db']=0.
    zero,zero_identity=generate_new_record(plan,scenario)
    zero_rng=[deepcopy(item.bit_generator.state) for item in generators]
    assert candidate_rng==zero_rng and len(candidate_rng)==3
    onset=round(plan['attack_start_s']*FS)
    verify_power_pair(candidate,candidate_identity,zero,zero_identity,
                      onset,-5.,0.)
    assert np.any(candidate['samples'][:,onset:]!=zero['samples'][:,onset:])


def test_power_pair_rejects_changed_geometry_or_clean_iq():
    record=dict(samples=np.ones((4,4)),nominal=np.ones((4,2)),
                codes=np.ones((2,3)),times=np.array([.1]),
                satellites=np.ones((2,3)),coarse=np.ones((2,3)))
    candidate={key:value.copy() for key,value in record.items()}
    old_identity=dict(case=dict(spoof_db=0.,angle_deg=5.),seed=1)
    identity=dict(case=dict(spoof_db=-5.,angle_deg=5.),seed=1)
    verify_power_pair(candidate,identity,record,old_identity,2,-5.,0.)
    candidate['samples'][0,0]=2.
    with pytest.raises(AssertionError):
        verify_power_pair(candidate,identity,record,old_identity,2,-5.,0.)
    candidate['samples'][0,0]=1.
    identity['case']['angle_deg']=6.
    with pytest.raises(AssertionError):
        verify_power_pair(candidate,identity,record,old_identity,2,-5.,0.)


def test_both_actual_projectors_and_gains_must_match_fresh_fit():
    identity=np.eye(3,dtype=complex)
    projectors=np.broadcast_to(identity,(6,3,3)).copy()
    default_p=identity.copy();default_p[2,2]=0.
    candidate_p=identity.copy();candidate_p[1,1]=0.
    default_trace=dict(trace_projectors=projectors.copy(),
                       trace_dimensions=np.array([3,3,3,3,3,2]))
    candidate_trace=dict(trace_projectors=projectors.copy(),
                         trace_dimensions=np.array([3,3,3,3,3,2]),
                         trace_raw_trigger=np.array([np.nan]*5+[4.2]))
    default_trace['trace_projectors'][5]=default_p
    candidate_trace['trace_projectors'][5]=candidate_p
    raw=np.full((4,6),np.nan);raw[0,5]=3.4;raw[3,5]=4.2
    fitted=np.broadcast_to(identity,(4,6,3,3)).copy()
    fitted[0,5]=default_p;fitted[3,5]=candidate_p
    gain=np.full((4,6,2),np.nan)
    gain[0,5]=[.6,.7];gain[3,5]=[.8,.9]
    fit=dict(raw=raw,would_projector=fitted,would_gain=gain)
    shared=dict(startup_prns=np.array([0,1]),
                calibration=np.ones(3,dtype=complex),
                calibration_rate=np.zeros(3),time=np.array([.1]))
    default=dict(**shared,gain=np.array([[.6,.7]]))
    candidate=dict(**shared,gain=np.array([[.8,.9]]))
    audit_new_fit(fit,default,default_trace,candidate,candidate_trace)
    default_trace['trace_projectors'][5]=identity
    with pytest.raises(AssertionError):
        audit_new_fit(fit,default,default_trace,candidate,candidate_trace)
    default_trace['trace_projectors'][5]=default_p
    default['gain'][0,0]=.5
    with pytest.raises(AssertionError):
        audit_new_fit(fit,default,default_trace,candidate,candidate_trace)
