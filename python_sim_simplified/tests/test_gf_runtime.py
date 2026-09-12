import pytest
from gf_runtime import GFRuntime


def update(r,t,**kwargs):
    args=dict(valid=True,arc_id='a',calibration_id='c',calibration_valid=True,nuisance_bound_m=.001)
    args.update(kwargs)
    return r.step(t,.020,.010,**args)


def test_start_requires_explicit_approval():
    r=GFRuntime(.02,.25)
    assert update(r,0)['state']==-1
    assert update(r,.01,branch_confirmation=('a','c'))['base_weight_multiplier']==.25


@pytest.mark.parametrize('change',[{'valid':False},{'calibration_valid':False},{'nuisance_bound_m':float('nan')}])
def test_failure_clears_approval_and_recovery_requires_new_event(change):
    r=GFRuntime(.02,.25)
    update(r,0,branch_confirmation=('a','c'))
    assert update(r,.01,**change)['base_weight_multiplier']==1.
    assert update(r,.02)['state']==-1
    assert update(r,.03,branch_confirmation=('a','c'))['state']==1


@pytest.mark.parametrize('change',[{'arc_id':'b'},{'calibration_id':'d'}])
def test_old_context_confirmation_rejected(change):
    r=GFRuntime(.02,.25)
    update(r,0,branch_confirmation=('a','c'))
    assert update(r,.01,branch_confirmation=('a','c'),**change)['state']==-1


def test_gap_cannot_reuse_approval_on_gap_epoch():
    r=GFRuntime(.02,.25)
    update(r,0,branch_confirmation=('a','c'))
    assert update(r,.1,branch_confirmation=('a','c'))['state']==-1
    assert update(r,.11)['state']==-1
    assert update(r,.12,branch_confirmation=('a','c'))['state']==1


def test_reverse_time_revokes_approval():
    r=GFRuntime(.02,.25)
    update(r,1,branch_confirmation=('a','c'))
    with pytest.raises(ValueError): update(r,.5)
    assert update(r,1.01)['state']==-1
