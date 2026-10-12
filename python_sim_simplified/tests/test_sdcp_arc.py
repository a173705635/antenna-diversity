import pytest
from core.multipath.sdcp_arc import SDCPArc


def step(m,t,**kw):
    args=dict(quality_valid=True,phase_state_valid=True,tracking_ids=('a','b'),calibration_id='c')
    args.update(kw)
    return m.step(t,**args)


def test_quality_does_not_approve_reference_and_gap_opens_new_arc():
    m=SDCPArc('session',.015);r=step(m,0);old=r['arc_id']
    assert r['state']=='await_reference'
    m.accept_reference(old,'reference-1','evidence-1')
    assert step(m,.01)['state']=='reference_usable'
    assert step(m,.02,quality_valid=False)['state']=='unavailable'
    new=step(m,.03)
    assert new['arc_id']!=old and new['state']=='await_reference'
    with pytest.raises(ValueError):m.accept_reference(old,'reference-1','evidence-1')


@pytest.mark.parametrize('change',[{'tracking_ids':('a','new-b')},{'calibration_id':'new-c'},
                                 {'phase_state_valid':False}])
def test_context_or_half_state_revokes_reference(change):
    m=SDCPArc('s',.015);old=step(m,0)['arc_id'];m.accept_reference(old,'r','e')
    assert step(m,.01,**change)['state']!='reference_usable'


def test_time_gap_and_nonmonotonic_time_revoke_reference():
    m=SDCPArc('s',.015);old=step(m,0)['arc_id'];m.accept_reference(old,'r','e')
    assert step(m,.04)['state']=='await_reference'
    with pytest.raises(ValueError):step(m,.03)
    assert step(m,.05)['state']=='await_reference'


def test_fresh_approval_binds_current_arc_but_is_not_old_continuity():
    m=SDCPArc('s',.015);old=step(m,0)['arc_id'];m.accept_reference(old,'r','e')
    step(m,.01,quality_valid=False);new=step(m,.02)['arc_id']
    with pytest.raises(ValueError):m.accept_reference(new,'new-r','e')
    m.accept_reference(new,'new-r','fresh-e')
    r=step(m,.03);assert r['state']=='reference_usable' and r['reference_id']=='new-r'
    assert r['arc_id']!=old


def test_session_ids_separate_restarts():
    m=SDCPArc('one',.015);old=step(m,0)['arc_id'];m.accept_reference(old,'r','e')
    assert step(SDCPArc('two',.015),0)['arc_id']!=old
