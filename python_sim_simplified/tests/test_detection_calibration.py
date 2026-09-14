import numpy as np
import pytest

from detection_calibration import calibrate_threshold, residual_peak_score_db, binomial_interval


def test_threshold_uses_conservative_rank_and_strict_exceedance():
    scores=np.arange(500,dtype=float)
    threshold=calibrate_threshold(scores,.01)
    assert threshold==495
    assert np.sum(scores>threshold)==4
    with pytest.raises(ValueError): calibrate_threshold(scores[:10],.01)


def test_score_is_record_maximum_and_scale_invariant():
    power=np.array([1.,1.,1.,10.])
    assert residual_peak_score_db(power)==pytest.approx(10.)
    assert residual_peak_score_db(power*1e-20)==pytest.approx(10.)
    assert residual_peak_score_db(np.zeros(4)) is None


def test_zero_false_alarms_do_not_imply_zero_upper_bound():
    low,high=binomial_interval(0,20)
    assert low==0
    assert .16<high<.18  # two-sided 95% Clopper-Pearson interval
