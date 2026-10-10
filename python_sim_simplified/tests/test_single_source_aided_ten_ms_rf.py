"""Fixed aided 10 ms RF orchestration without local saved IQ."""
import numpy as np
import pytest

import experiments.single_source_aided_ten_ms_rf as replay
import experiments.single_source_aided_coherent_rf as stage28


def test_saved_five_ms_paths_replay_from_separate_states(monkeypatch):
    tracking=dict(support=np.array([[1.,2.]]))
    coherent=np.array([[3.,4.]])
    inputs=[];checked=[]
    monkeypatch.setattr(replay,'navigation_input',
                        lambda _tracking,support:dict(support=support.copy()))

    def navigate(value,**options):
        inputs.append((value['support'].copy(),options.copy()))
        return dict(position=np.array([[float(len(inputs)),0.,0.]]))

    def check(value,_saved,prefix):
        checked.append((prefix,value['position'][0,0]))

    monkeypatch.setattr(replay,'navigate',navigate)
    monkeypatch.setattr(replay,'assert_saved_navigation',check)
    result=replay.replay_saved_five_ms(tracking,coherent,{})
    np.testing.assert_array_equal(inputs[0][0],tracking['support'])
    np.testing.assert_array_equal(inputs[1][0],coherent)
    assert inputs[0][1]==dict(diagnostics=True,code_blend=.1,
                              phase_leave_one_out=False)
    assert inputs[1][1]==dict(diagnostics=True,code_blend=0.,
                              phase_leave_one_out=True)
    assert checked==[('original_',1.),('short_',2.)]
    assert result['original'] is not result['short']


def test_actual_ten_ms_raw_projector_and_gain_must_match_offline_fit():
    identity=np.eye(3,dtype=complex)
    projector=identity.copy();projector[1,1]=0.
    matrices=np.broadcast_to(identity,(6,3,3)).copy()
    matrices[5]=projector
    trace=dict(trace_raw_trigger=np.array([np.nan]*5+[3.4]),
               trace_projectors=matrices,
               trace_dimensions=np.array([3,3,3,3,3,2]))
    fitted=np.broadcast_to(identity,(4,6,3,3)).copy()
    fitted[2,5]=projector
    raw=np.full((4,6),np.nan);raw[2,5]=3.4
    gain=np.full((4,6,2),np.nan);gain[2,5]=[.6,.8]
    fit=dict(raw=raw,would_projector=fitted,would_gain=gain)
    baseline=dict(startup_prns=np.array([0,1]),
                  calibration=np.ones(3,dtype=complex),
                  calibration_rate=np.zeros(3),
                  time=np.array([.1]),available_time=np.array([.12]))
    tracking={**baseline,'gain':np.array([[.6,.8]])}
    replay.audit_ten_ms_fit(tracking,trace,baseline,fit)
    trace['trace_projectors'][5]=identity
    with pytest.raises(AssertionError):
        replay.audit_ten_ms_fit(tracking,trace,baseline,fit)
    trace['trace_projectors'][5]=projector
    tracking['gain'][0,0]=.5
    with pytest.raises(AssertionError):
        replay.audit_ten_ms_fit(tracking,trace,baseline,fit)


def test_plot_output_does_not_replace_saved_stage28_figure(tmp_path,monkeypatch):
    old=tmp_path / 'stage28';old.mkdir()
    new=tmp_path / 'stage31';new.mkdir()
    (old / 'overview.png').write_bytes(b'frozen stage28 figure')
    monkeypatch.setattr(stage28,'OUTPUT',old)
    monkeypatch.setattr(stage28,'SCENES',('weak',))
    navigation=dict(selected=np.ones((2,1)),
                    position=np.array([[0.,0.,0.],[1.,0.,0.]]))
    view=dict(available=np.array([.1,.2]),observation=np.array([.1,.2]),
              velocity_east=0.,baseline=dict(original=navigation,short=navigation),
              candidate=dict(original=navigation,short=navigation))
    stage28.plot_cases(dict(weak=view),output=new)
    assert (new / 'overview.png').stat().st_size>0
    assert (old / 'overview.png').read_bytes()==b'frozen stage28 figure'
