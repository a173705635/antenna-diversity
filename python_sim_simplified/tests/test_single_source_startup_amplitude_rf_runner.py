"""Saved H0 RF comparisons keep distinct navigation epochs and fit sources."""
import numpy as np
import pytest

from experiments.single_source_startup_amplitude_rf import (
    audit_alignment,compare_paths,score_navigation,
)


def test_score_navigation_separates_publication_coverage_and_initial_motion():
    observation=np.array([.15,.16,.17,.18])
    available=observation+.02
    tracking=dict(time=observation,available_time=available)
    navigation=dict(position=np.array([[np.nan,np.nan,np.nan],
        [1.,0.,0.],[np.nan,np.nan,np.nan],[1.3,0.,0.]]),
        initialization_time=.18)
    result=score_navigation(tracking,navigation,velocity_east=2.)
    assert result['published_all']==2
    assert result['published_evaluation']==0
    assert result['first_published_available_time_s']==pytest.approx(.18)
    assert result['last_published_available_time_s']==pytest.approx(.20)
    assert result['initial_horizontal_error_m']==pytest.approx(.68)
    assert result['max_relative_motion_error_m']==pytest.approx(.26)


def test_alignment_rejects_wrong_model_gain_and_projector():
    nominal=np.ones((3,2),complex)
    amplitude=np.array([1.,1.,1.])
    trace=dict(trace_raw_trigger=np.r_[np.full(5,np.nan),1.],
               trace_projectors=np.tile(np.eye(3),(6,1,1)),
               trace_dimensions=np.full(6,3),trace_nominal=nominal)
    tracking=dict(time=np.array([.1]),gain=np.ones((1,2)),
                  startup_prns=np.array([0,1]),
                  calibration=np.ones(3,complex),calibration_rate=np.zeros(3),
                  startup_relative_amplitude=amplitude)
    reference=dict(raw=np.tile(trace['trace_raw_trigger'],(2,1)),
                   would_projector=np.tile(trace['trace_projectors'],(2,1,1,1)),
                   would_gain=np.ones((2,6,2)),relative_amplitude=amplitude)
    startup=dict(prns=np.array([0,1]),calibration=np.ones(3,complex),
                 calibration_rate=np.zeros(3),relative_amplitude=amplitude)
    audit_alignment(tracking,trace,reference,startup,nominal,mode=1)
    altered=dict(tracking,gain=np.full((1,2),.9))
    with pytest.raises(AssertionError):
        audit_alignment(altered,trace,reference,startup,nominal,mode=1)
    altered_trace=dict(trace,trace_projectors=trace['trace_projectors'].copy())
    altered_trace['trace_projectors'][5,0,0]=0.
    with pytest.raises(AssertionError):
        audit_alignment(tracking,altered_trace,reference,startup,nominal,mode=1)


def test_pair_comparison_uses_common_and_path_only_epochs():
    tracking=dict(time=np.array([.3,.4,.5]),
                  available_time=np.array([.32,.42,.52]))
    missing=np.array([np.nan,np.nan,np.nan])
    positions=dict(off=dict(original=np.stack(([1.,0.,0.],missing,[1.,0.,0.])),
                            short=np.stack(([1.,0.,0.],missing,[1.,0.,0.]))),
                   on=dict(original=np.stack(([2.,0.,0.],[2.,0.,0.],missing)),
                           short=np.stack(([2.,0.,0.],[2.,0.,0.],missing))))
    compared=compare_paths(tracking,positions,velocity_east=2.)
    assert compared['short']['common']==1
    assert compared['short']['new_only']==1
    assert compared['short']['lost_only']==1
    assert compared['short']['off_common']['max_error_m']==pytest.approx(.4)
    assert compared['short']['on_common']['max_error_m']==pytest.approx(1.4)
