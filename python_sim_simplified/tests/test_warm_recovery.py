import numpy as np
from warm_recovery import generate,subset,estimate

def test_same_record_and_truth_isolation():
    obs,truth=generate(111002,'nominal')
    assert set(obs)=={'init','z','noise','lag','dt','attack_start_s'}
    for m in (2,3,4):
        sub=subset(obs,m)
        assert np.array_equal(sub['z'],obs['z'][:,:,:m,:])
        result=estimate(sub,['init_average','peak_baseline'])
        assert result['init_completed_s']==.24
        assert abs(result['gap_s']-.08)<1e-12
        assert result['delay'].shape==(6,)

def test_absence_never_changes_receiver_interface():
    nominal,_=generate(111002,'nominal');absent,truth=generate(111002,'absent')
    assert not truth['exists']
    assert np.array_equal(nominal['init'],absent['init'])
    assert np.array_equal(nominal['noise'],absent['noise'])
    assert not np.array_equal(nominal['z'],absent['z'])

def test_initialization_failure_is_explicit():
    obs,_=generate(111002,'init_weak')
    result=estimate(obs,['init_average','peak_baseline','init_gate'])
    assert result['init_completed_s'] is None
    assert result['gap_s'] is None
    assert not result['init_ok'].all()
    assert np.all(~result['accepted'] | result['init_ok'])

def test_joint_delay_is_used_by_continuity_admission():
    obs,_=generate(111002,'nominal')
    from warm_recovery import GRID,fit_rows,CHIP_M
    methods=['init_average','peak_baseline','projected_fit','joint_fit','continuity_gate']
    result=estimate(obs,methods)
    initial=fit_rows(obs['init'].mean(axis=0)[:,0,:],GRID,obs['lag'])[0]
    assert np.all(abs(result['delay'][result['accepted']]-initial[result['accepted']])*CHIP_M<30)

def test_publication_requires_complete_initialization():
    from warm_recovery_release import estimate as publish,RESEARCH
    obs,_=generate(111002,'init_weak')
    result=publish(obs)
    research=RESEARCH.estimate(obs,RESEARCH.SELECTED_METHODS)
    assert result['init_completed_s'] is None
    assert np.array_equal(result['research_accepted'],research['accepted'])
    assert np.array_equal(result['delay'],research['delay'])
    assert not result['publish_accepted'].any()
    assert not result['accepted'].any()

def test_publication_preserves_initialized_research_results():
    from warm_recovery_release import estimate as publish
    obs,_=generate(111002,'nominal')
    result=publish(obs)
    assert result['init_completed_s'] is not None
    assert result['init_ok'].all()
    assert np.array_equal(result['publish_accepted'],result['research_accepted'])
    assert np.array_equal(result['accepted'],result['publish_accepted'])
