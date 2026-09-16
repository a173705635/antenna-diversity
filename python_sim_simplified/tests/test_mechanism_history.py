import numpy as np
from core.mechanism_smoothing import smooth_history


def record(arc):
    return dict(time=np.arange(4)*.01,ranges=np.array([[0.],[0.],[900.],[40.]]),
                carrier=np.arange(4)[:,None]*.01,arc=np.array(arc)[:,None])


def test_short_rejection_does_not_reset_a_continuous_tracking_arc():
    z=record([0,0,0,0]);valid=np.array([[True],[True],[False],[True]])
    ranges,ages=smooth_history(z,valid)
    assert np.isnan(ranges[2,0])
    assert ages[3,0]==3
    assert 13<ranges[3,0]<14


def test_reacquisition_does_not_reuse_old_smoothing_history():
    z=record([0,0,1,1]);valid=np.array([[True],[True],[False],[True]])
    ranges,ages=smooth_history(z,valid)
    assert ages[3,0]==1 and ranges[3,0]==40


def test_long_gap_cannot_reuse_smoothing_history():
    z=record([0,0,0,0]);z['time'][3]=.3
    ranges,ages=smooth_history(z,np.array([[True],[True],[False],[True]]))
    assert ages[3,0]==1 and ranges[3,0]==40
