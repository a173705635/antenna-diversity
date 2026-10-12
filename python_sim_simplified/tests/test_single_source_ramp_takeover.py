"""Fixed velocity-step takeover uses observation time and an actual IQ pair."""
import numpy as np

from core.spoofing.recovery_geometry import FS
from experiments.spoofing import single_source_spoof_power as power
from experiments.spoofing import single_source_ramp_takeover as ramp
from experiments.spoofing.single_source_history_study import generate_new_record
from experiments.spoofing.single_source_template_coherence import truth_code_phases


def small_plan():
    return dict(seed=2026100301,duration_s=.38,satellites=3,
                startup_cn0_db_hz=45.,post_0p3s_cn0_db_hz=30.,
                attack_start_s=.3,spoof_db=0.,angle_deg=5.,fake_east_m=0.)


def test_fake_position_uses_observation_time_and_rate_zero_is_exact():
    case=dict(attack_start=.3,fake_east=0.,fake_rate=3.)
    observation=np.array([.29,.30,.31,.38])
    np.testing.assert_allclose(power.fake_east_at_observation(case,observation),
                               [0.,0.,.03,.24],rtol=0,atol=1e-15)
    case=dict(case,fake_east=100.,fake_rate=0.)
    np.testing.assert_array_equal(
        power.fake_east_at_observation(case,observation),
        np.full(len(observation),100.))


def test_zero_offset_has_equal_code_and_carrier_phase_at_onset_but_rate_step():
    record,identity=generate_new_record(small_plan(),'weak',fake_rate=3.)
    case=identity['case']
    assert case['fake_east']==0. and case['fake_rate']==3.
    assert case['nav_bit_epoch_s']==0.
    onset=round(case['attack_start']*FS)
    assert (onset-1)/FS<case['attack_start']<=onset/FS
    true_at,fake_at=truth_code_phases(case,identity['seed'],case['attack_start'])
    np.testing.assert_array_equal(true_at,fake_at)
    true_later,fake_later=truth_code_phases(case,identity['seed'],
                                           case['attack_start']+.01)
    assert np.max(abs(true_later-fake_later))>0.
    baseline,baseline_identity=generate_new_record(small_plan(),'weak')
    np.testing.assert_array_equal(record['samples'][:,:onset],
                                  baseline['samples'][:,:onset])
    assert np.any(record['samples'][:,onset+1:]!=baseline['samples'][:,onset+1:])
    for key in record:
        if key!='samples':np.testing.assert_array_equal(record[key],baseline[key])
    assert baseline_identity['case']['fake_rate']==0.


def test_no_attack_iq_ignores_unused_fake_geometry_and_rate():
    plan=small_plan()
    newer,_=generate_new_record(plan,'no_attack_control',fake_rate=3.)
    older_plan={**plan,'angle_deg':60.,'spoof_db':10.,
                'fake_east_m':100.}
    older,_=generate_new_record(older_plan,'no_attack_control')
    for key in newer:
        np.testing.assert_array_equal(newer[key],older[key],err_msg=key)


def test_navigation_saved_fake_error_tracks_observation_not_availability(
        monkeypatch,tmp_path):
    observation=np.array([.29,.31,7.98])
    tracking=dict(time=observation,available_time=observation+.02,
                  support=np.ones((3,1)))
    position=np.column_stack((2*observation+np.array([0.,.03,23.04]),
                              np.zeros(3),np.zeros(3)))
    monkeypatch.setattr(power,'navigation_input',lambda *_:dict())
    monkeypatch.setattr(power,'navigate',lambda *_args,**_kwargs:dict(
        position=position,selected=np.ones((3,1),dtype=bool),
        diagnostics=dict(phase_ok=np.ones(3,dtype=bool))))
    monkeypatch.setattr(power,'assert_prefix',lambda *_args:None)
    monkeypatch.setattr(power,'quality_support_counts',lambda *_args:np.ones(3))
    monkeypatch.setattr(power,'score_mode',lambda *_args:dict())
    monkeypatch.setattr(power,'observation_quality',lambda *_args:dict())
    monkeypatch.setattr(power,'save_navigation',lambda *_args:None)
    _,_,arrays=power.run_navigation(tracking,tracking['support'],dict(
        velocity_east=2.,attack_start=.3,fake_east=0.,fake_rate=3.),tmp_path)
    np.testing.assert_allclose(arrays['short_east_minus_fake_m'],0.,atol=1e-12)
    with np.load(tmp_path / 'navigation.npz') as saved:
        np.testing.assert_allclose(saved['original_east_minus_fake_m'],0.,atol=1e-12)


def test_segment_masks_use_observation_time_and_report_phase_recovery():
    observation=np.array([.29,.30,.31,6.96,6.97,7.98])
    available=observation+.02
    fake=3*np.maximum(observation-.3,0.)
    position=np.column_stack((2*observation+fake,np.zeros(6),np.zeros(6)))
    position[3]=np.nan
    navigation=dict(position=position,diagnostics=dict(
        phase_ok=np.array([True,True,True,False,False,True]),
        recovery_confirmed=np.array([False,False,False,False,False,True])),
        selected=np.ones((6,6),dtype=bool))
    result=ramp.segment_scores(dict(time=observation,
        available_time=available,ranges=np.ones((6,6)),lock=np.ones((6,6)),
        cn0=np.full((6,6),30.),gain=np.ones((6,6))),navigation,
                               dict(velocity_east=2.,attack_start=.3,
                                    fake_east=0.,fake_rate=3.),
                               np.full(6,6))
    assert result['available_post']['opportunities']==6
    assert result['observed_attack']['opportunities']==5
    assert result['final_observed_second']['opportunities']==1
    tail=result['fake_offset_ge_20']
    assert tail['opportunities']==tail['published']==2
    assert tail['error_ge_20']==2 and tail['error_ge_30']==0
    np.testing.assert_allclose(tail['max_true_error_m'],23.04)
    np.testing.assert_allclose(tail['max_fake_error_m'],0.)
    np.testing.assert_allclose(tail['median_fake_error_m'],0.)
    np.testing.assert_allclose(tail['last_fake_error_m'],0.)
    assert tail['finite_ge6']==tail['quality_ge6']==2
    assert tail['support_ge6']==tail['selected_ge6']==2
    assert tail['published_phase_ok']==1
    assert tail['published_recovery_confirmed']==1
    np.testing.assert_allclose(tail['last_published_available_s'],8.)
