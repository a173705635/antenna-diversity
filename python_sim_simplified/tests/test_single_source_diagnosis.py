import numpy as np

from core import single_source
from experiments.single_source_diagnosis import outage_events


def test_optional_diagnostics_preserves_navigation_and_counts_gates(monkeypatch):
    monkeypatch.setattr(single_source,'smooth_history',lambda z,valid,**kwargs:(z['ranges'].copy(),np.ones_like(z['ranges'],int)))
    monkeypatch.setattr(single_source,'position_solution',lambda *args:(np.zeros(3),1.,0.))
    monkeypatch.setattr(single_source,'receiver_reference',lambda:(np.zeros(3),np.eye(3)))
    monkeypatch.setattr(single_source,'solve_position',lambda *args,**kwargs:dict(position_m=np.zeros(3)))
    monkeypatch.setattr(single_source,'phase_prediction',lambda *args:None)
    n=6;s=6
    z=dict(time=np.arange(n)*.01,available_time=np.arange(n)*.01+.02,
           ranges=np.ones((n,s)),lock=np.ones((n,s)),cn0=np.full((n,s),30.),
           gain=np.ones((n,s)),support=np.full((n,s),4.),carrier=np.ones((n,s)),
           arc=np.zeros((n,s),int),satellites=np.ones((n,s,3)))
    z['ranges'][3:,3:]=np.nan
    plain=single_source.navigate(z)
    audited=single_source.navigate(z,diagnostics=True)
    assert set(plain)==set(audited)-{'diagnostics'}
    for key in ('position','candidate_position','selected'):
        np.testing.assert_array_equal(plain[key],audited[key],strict=True)
    assert plain['initialization_time']==audited['initialization_time']==.04
    d=audited['diagnostics']
    np.testing.assert_array_equal(d['raw_count'],[6,6,6,3,3,3])
    np.testing.assert_array_equal(d['selected_count'],[6,6,6,3,3,3])
    assert d['unanchored_selected_common_arc_count'][5]==3
    assert d['anchored_selected_common_arc_count'][5]==0
    assert d['reason'].tolist()==['initializing','initializing','published',
                                  'fewer_than_four','fewer_than_four','fewer_than_four']
    assert outage_events(d['reason'],np.isfinite(plain['position']).all(axis=1),2)==dict(
        first_epoch=3,first_reason='fewer_than_four',
        sustained_epoch=3,sustained_reason='fewer_than_four')


def test_recovery_attempt_records_tenth_confirmation_before_reset(monkeypatch):
    monkeypatch.setattr(single_source,'smooth_history',lambda z,valid,**kwargs:(z['ranges'].copy(),np.ones_like(z['ranges'],int)))
    calls=iter(range(15))
    monkeypatch.setattr(single_source,'position_solution',
                        lambda *args:(np.zeros(3),2000. if next(calls)==3 else 1.,0.))
    monkeypatch.setattr(single_source,'receiver_reference',lambda:(np.zeros(3),np.eye(3)))
    monkeypatch.setattr(single_source,'solve_position',lambda *args,**kwargs:dict(position_m=np.zeros(3)))
    monkeypatch.setattr(single_source,'phase_prediction',lambda *args:None)
    n=15;s=6
    z=dict(time=np.arange(n)*.01,available_time=np.arange(n)*.01+.02,
           ranges=np.ones((n,s)),lock=np.ones((n,s)),cn0=np.full((n,s),30.),
           gain=np.ones((n,s)),support=np.full((n,s),4.),carrier=np.ones((n,s)),
           arc=np.zeros((n,s),int),satellites=np.ones((n,s,3)))
    result=single_source.navigate(z,diagnostics=True)
    d=result['diagnostics']
    assert d['recovery_attempt_count'][13]==10
    assert d['recovery_count'][13]==0
    assert not d['isolated'][13]
    assert np.isfinite(result['position'][13]).all()
