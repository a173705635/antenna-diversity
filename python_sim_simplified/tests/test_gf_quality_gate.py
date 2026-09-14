import numpy as np
import pytest
from gf_quality_gate import classify


def test_interval_and_sign_symmetry():
    r=np.array([[0., .008, .012, -.012]])
    out=classify(r,.010,nuisance_bound_m=.001,valid=True,branch_verified=True)
    assert out['state'].tolist()==[[0,0,1,1]]
    assert np.allclose(out['lower_m'],[[0,.007,.011,.011]])


def test_unknown_bounds_and_unverified_branch_never_pass():
    r=np.zeros((2,2))
    assert (classify(r,.01,nuisance_bound_m=np.nan,valid=True,branch_verified=True)['state']==-1).all()
    assert (classify(r,.01,nuisance_bound_m=0,valid=True,branch_verified=False)['state']==-1).all()
    assert (classify(r,.01,nuisance_bound_m=0,valid=False,branch_verified=True)['state']==-1).all()


def test_more_uncertainty_cannot_produce_a_new_decision():
    r=np.array([[.009,.011]])
    assert classify(r,.01,nuisance_bound_m=0,valid=True,branch_verified=True)['state'].tolist()==[[0,1]]
    assert classify(r,.01,nuisance_bound_m=.003,valid=True,branch_verified=True)['state'].tolist()==[[-1,-1]]


def test_missing_values_and_threshold_boundary():
    r=np.array([[np.nan,.01,.012]])
    out=classify(r,.01,nuisance_bound_m=np.array([[0,0,.002]]),valid=True,branch_verified=True)
    assert out['state'].tolist()==[[-1,0,-1]]
