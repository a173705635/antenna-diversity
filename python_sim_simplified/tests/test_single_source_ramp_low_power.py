"""The lower-power takeover changes only fake amplitude and has its own outputs."""
import numpy as np
import pytest

from experiments.spoofing.single_source_ramp_low_power import (
    compare_iq_components, compare_power_pair, cross_power, verify_power_input,
)


def paired_records():
    case=dict(spoof_db=0.,genuine=True,fake_rate=3.,angle=5.)
    old=dict(samples=np.ones((2,20),complex),geometry=np.array([1.,2.]))
    new=dict(samples=old['samples'].copy(),geometry=old['geometry'].copy())
    new['samples'][:,10:]=2.
    identity=dict(scenario='weak',attacked=True,seed=7,
                  case={**case,'spoof_db':-5.},rf_sha256='new')
    reference=dict(scenario='weak',attacked=True,seed=7,
                   case=case,rf_sha256='old')
    return new,identity,old,reference


def test_only_power_may_change_with_clean_prefix_and_geometry_exact():
    new,identity,old,reference=paired_records()
    verify_power_input(new,identity,old,reference,10)
    new['geometry'][0]=3.
    with pytest.raises(AssertionError):
        verify_power_input(new,identity,old,reference,10)
    new,identity,old,reference=paired_records()
    new['samples'][0,9]=3.
    with pytest.raises(AssertionError):
        verify_power_input(new,identity,old,reference,10)
    new,identity,old,reference=paired_records()
    identity['case']['fake_rate']=4.
    with pytest.raises(AssertionError):
        verify_power_input(new,identity,old,reference,10)


def test_power_pair_checks_genuine_flag_and_shared_clean_position():
    weak,weak_id,_,_=paired_records()
    absent={**weak,'samples':weak['samples'].copy()}
    absent_id={**weak_id,'scenario':'genuine_absent',
               'case':{**weak_id['case'],'genuine':False}}
    compare_power_pair((weak,weak_id),(absent,absent_id),10)
    absent['samples'][0,9]=0.
    with pytest.raises(AssertionError):
        compare_power_pair((weak,weak_id),(absent,absent_id),10)


def test_postattack_genuine_component_differences_use_all_samples():
    base=np.ones((2,12),complex)
    genuine=np.array([[1.+2j]*12,[.5-1j]*12])
    low_weak=base+genuine
    low_absent=base.copy()
    high_weak=2*base+genuine
    high_absent=2*base
    assert compare_iq_components(low_weak,low_absent,high_weak,high_absent,
                                 block_samples=3)<1e-14
    low_weak[1,11]+=1e-7
    with pytest.raises(AssertionError):
        compare_iq_components(low_weak,low_absent,high_weak,high_absent,
                              block_samples=3)


def test_cross_power_keeps_common_added_lost_and_observed_opportunities():
    time=np.array([.29,.30,1.,7.,8.])
    tracking=dict(time=time,available_time=time+.02)
    truth=2*time
    old=np.column_stack((truth,np.zeros(5),np.zeros(5)))
    new=old.copy()
    old[[0,1,4]]=np.nan
    new[[0,1,3]]=np.nan
    reference=dict(observation_time=time,available_time=time+.02,
                   published_only_position=old,
                   published_only_initialization_time=np.array(.27))
    case=dict(attack_start=.3,velocity_east=2.,fake_east=0.,fake_rate=3.)
    candidate={'published_only':dict(position=new,initialization_time=np.array(.27))}
    result=cross_power(tracking,case,candidate,
                       reference,'genuine_absent')['published_only']['segments']
    assert result['observed_attack']['opportunities']==4
    assert result['observed_attack']['common_old']['count']==1
    assert result['observed_attack']['lost']['count']==1
    assert result['observed_attack']['added']['count']==1
    assert result['fake_offset_ge_20']['opportunities']==2
    reference['published_only_position'][2,0]+=1.
    with pytest.raises(AssertionError):
        cross_power(tracking,case,candidate,reference,'genuine_absent')
    reference['published_only_position'][2,0]-=1.
    reference['published_only_initialization_time']=np.array(.28)
    with pytest.raises(AssertionError):
        cross_power(tracking,case,candidate,reference,'genuine_absent')
    reference['published_only_initialization_time']=np.array(.27)
    reference['available_time']=time+.03
    with pytest.raises(AssertionError):
        cross_power(tracking,case,candidate,reference,'genuine_absent')
