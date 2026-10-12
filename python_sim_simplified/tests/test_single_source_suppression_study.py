import numpy as np
import pytest

from experiments.spoofing import single_source_suppression_study as study


def test_saved_energy_recalculates_split_support_and_detects_corruption():
    z=dict(support_split=np.array([[2.4,np.nan]]),
           split_on_energy=np.array([[6.,np.nan]]),
           split_off_energy=np.array([[2.5,np.nan]]))
    study.verify_split_energy(z)
    z['split_on_energy'][0,0]=7.
    with pytest.raises(AssertionError):study.verify_split_energy(z)


def test_alignment_uses_observation_end_sample_not_available_time():
    aligned=study.aligned_windows(np.array([.15,.16,.17,.18]))
    np.testing.assert_array_equal(aligned,[False,True,False,True])


def test_score_reports_unavailable_run_and_last_publication():
    times=np.array([.20,.21,.22,.23,.24])
    position=np.zeros((5,3));position[2:4]=np.nan
    navigation=dict(position=position,initialization_time=.20)
    result=study.score(navigation,times,dict(attack_start=.20,velocity_east=0.))
    assert result['longest_unavailable_s']==.02
    assert result['tail_published']==3
    assert result['last_published_available_time_s']==.24
