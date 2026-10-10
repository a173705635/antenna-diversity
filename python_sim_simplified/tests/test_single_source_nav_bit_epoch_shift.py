"""Stage 32 paired IQ and independent beam-output checks."""
import numpy as np
import pytest

from experiments import single_source_nav_bit_epoch_shift as study


def test_new_epoch_pair_requires_same_clean_samples_and_observations(monkeypatch):
    monkeypatch.setattr(study,'PLAN',dict(attack_start_s=.3,
                                          nav_bit_epoch_s=.005))
    size=round(.3*study.FS)+2
    record=dict(samples=np.ones((3,size),dtype=complex),
                nominal=np.ones((3,2)),times=np.array([.1,.2]))
    identity=dict(case=dict(nav_bit_epoch_s=.005))
    prefix,other=study.verify_epoch_pair(record,identity,None,None)
    changed={key:value.copy() for key,value in record.items()}
    changed['samples'][0,-1]=2.
    study.verify_epoch_pair(changed,identity,prefix,other)
    changed['samples'][0,10]=2.
    with pytest.raises(AssertionError):
        study.verify_epoch_pair(changed,identity,prefix,other)
    changed['samples'][0,10]=1.
    changed['times'][0]=.11
    with pytest.raises(AssertionError):
        study.verify_epoch_pair(changed,identity,prefix,other)


def test_old_new_metadata_and_geometry_allow_only_bit_epoch_change():
    record=dict(nominal=np.eye(3),times=np.array([.1,.2]))
    old=dict(seed=7,scenario='weak',attacked=True,rf_sha256='old',
             case=dict(angle=5.,spoof_db=-5.,genuine=True))
    fresh=dict(seed=7,scenario='weak',attacked=True,rf_sha256='new',
               case=dict(**old['case'],nav_bit_epoch_s=.005))
    study.verify_historic_single_factor(record,fresh,old,record)
    wrong=dict(fresh,case=dict(fresh['case'],angle=6.))
    with pytest.raises(AssertionError):
        study.verify_historic_single_factor(record,wrong,old,record)
    wrong_record=dict(record,times=np.array([.1,.3]))
    with pytest.raises(AssertionError):
        study.verify_historic_single_factor(wrong_record,fresh,old,record)


def test_actual_five_and_ten_ms_projectors_use_own_fits():
    eye=np.eye(3,dtype=complex)
    five=eye.copy();five[1,1]=0.
    ten=eye.copy();ten[2,2]=0.
    projectors=np.broadcast_to(eye,(4,6,3,3)).copy()
    projectors[1,5]=five
    projectors[2,5]=ten
    raw=np.full((4,6),np.nan)
    raw[1,5]=3.4;raw[2,5]=4.2
    gain=np.full((4,6,2),np.nan)
    gain[1,5]=[.4,.5];gain[2,5]=[.6,.7]
    fit=dict(raw=raw,would_projector=projectors,would_gain=gain,
             startup_prns=np.array([0,1]),calibration=np.ones(3,dtype=complex),
             calibration_rate=np.zeros(3))
    for index,projector in ((1,five),(2,ten)):
        trace=dict(trace_raw_trigger=np.array([np.nan]*5+[raw[index,5]]),
                   trace_projectors=np.broadcast_to(eye,(6,3,3)).copy(),
                   trace_dimensions=np.array([3,3,3,3,3,2]))
        trace['trace_projectors'][5]=projector
        tracking=dict(startup_prns=fit['startup_prns'].copy(),
                      calibration=fit['calibration'].copy(),
                      calibration_rate=fit['calibration_rate'].copy(),
                      time=np.array([.1]),gain=gain[index,5:6].copy())
        study.audit_actual_beam(tracking,trace,fit,index)
        trace['trace_projectors'][5]=eye
        with pytest.raises(AssertionError):
            study.audit_actual_beam(tracking,trace,fit,index)
        trace['trace_projectors'][5]=projector
        tracking['gain'][0,0]=.2
        with pytest.raises(AssertionError):
            study.audit_actual_beam(tracking,trace,fit,index)


def test_arrival_bit_phase_uses_propagation_delay_plus_epoch(monkeypatch):
    def phases(_case,_seed,time):
        return ((time-np.array([.011,.0112]))*1.023e6,
                (time-np.array([.0115,.0117]))*1.023e6)

    monkeypatch.setattr(study,'truth_code_phases',phases)
    fit=dict(block_start_s=np.arange(6)*.02)
    record=dict(nominal=np.ones((3,2)))
    metadata=dict(case=dict(nav_bit_epoch_s=.005,attack_start=.3),
                  attacked=True,seed=1)
    metadata['case']['attack_start']=.1
    result=study.bit_boundaries(record,metadata,fit)
    np.testing.assert_allclose(result['true_arrival_bit_phase_mod_10_ms'][5],
                               [6.,6.2],atol=1e-9)
    np.testing.assert_allclose(result['fake_arrival_bit_phase_mod_5_ms'][5],
                               [1.5,1.7],atol=1e-9)
    np.testing.assert_allclose(result['fake_arrival_bit_phase_mod_20_ms'][5],
                               [16.5,16.7],atol=1e-9)


def test_transport_reports_no_fix_and_separates_anchor_bias():
    times=np.array([.1,.2,.3])
    missing=dict(position=np.full((3,3),np.nan))
    assert study.anchor_transport(missing,times,2.)==dict(
        first_observation_time_s=None,initial_horizontal_error_m=None,
        max_relative_motion_error_m=None)
    position=np.array([[np.nan,np.nan,np.nan],
                       [3.,4.,0.],[3.2,4.,0.]])
    value=study.anchor_transport(dict(position=position),times,2.)
    np.testing.assert_allclose(value['initial_horizontal_error_m'],
                               np.hypot(3.-.4,4.))
    np.testing.assert_allclose(value['max_relative_motion_error_m'],0.,atol=1e-12)
