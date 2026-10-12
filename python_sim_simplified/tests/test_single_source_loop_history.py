"""Optional accepted-loop logging does not enter the receiver calculation."""
import numpy as np

from core.spoofing import single_source
from core.spoofing.recovery_geometry import FS


def test_loop_history_copies_only_accepted_segments_without_changing_science(monkeypatch):
    monkeypatch.setattr(single_source,'acquire_array',
                        lambda *args,**kwargs:{'success':True})
    startup=dict(epoch_start_samples=np.array([round(.09*FS)]),
        code_nco_phase_chips=np.array([100.]),
        code_nco_rate_hz=np.array([1.023e6]),
        carrier_nco_frequency_hz=np.array([0.]))
    monkeypatch.setattr(single_source,'track',
                        lambda *args,**kwargs:{'success':True,**startup})
    monkeypatch.setattr(single_source,'calibrate',
                        lambda *args,**kwargs:(np.ones(3,complex),np.zeros(3)))
    monkeypatch.setattr(single_source,'local_signal',
                        lambda code,template,time:np.ones(len(time),complex))
    monkeypatch.setattr(single_source,'residual_trigger',
                        lambda prompts,desired,noise:(np.eye(3),1.))
    monkeypatch.setattr(single_source,'code_pseudorange',
                        lambda history,times,*args:np.zeros(len(times)))
    monkeypatch.setattr(single_source,'carrier_range',
                        lambda history,times,*args:np.zeros(len(times)))
    starts=np.arange(round(.11*FS),round(.18*FS),20460)
    n=len(starts)
    history=dict(epoch_start_samples=starts,
        prompt=np.arange(n)+1j*np.arange(n),pll_error_rad=np.arange(n)*.01,
        fll_error_hz=np.arange(n)*.1,
        carrier_nco_phase_rad=np.arange(n)*.2,
        carrier_nco_frequency_hz=np.arange(n)*.3,
        carrier_frequency_hz=np.arange(n)*.4,
        code_nco_phase_chips=np.arange(n)*.5,
        code_nco_rate_hz=np.full(n,1.023e6),dll_error=np.arange(n)*.6,
        cn0_db_hz=np.full(n,45.),phase_lock_metric=np.full(n,.95))
    monkeypatch.setattr(single_source,'receive_measured',
                        lambda signal,prn,*args:{'segments':[
                            dict(history=history)] if prn==1 else []})
    rng=np.random.default_rng(17)
    nominal=np.exp(1j*np.outer(np.arange(3),np.arange(4)*.2))
    record=dict(samples=rng.normal(size=(3,round(.2*FS)))+1j*rng.normal(
        size=(3,round(.2*FS))),nominal=nominal,
        codes=np.ones((4,2046)),times=np.array([.15,.16]),
        satellites=np.zeros((2,4,3)),coarse=np.zeros(4))
    baseline=single_source.frontend(record,[0,1,2])
    collected=[]
    logged=single_source.frontend(record,[0,1,2],loop_history=collected)
    assert set(baseline)==set(logged)
    for key,value in baseline.items():
        np.testing.assert_array_equal(value,logged[key],err_msg=key)
    assert len(collected)==1
    assert collected[0]['prn']==1 and collected[0]['arc']==0
    assert set(collected[0])==set(single_source.LOOP_HISTORY_FIELDS)|{'prn','arc'}
    for key in single_source.LOOP_HISTORY_FIELDS:
        np.testing.assert_array_equal(collected[0][key],history[key])
        assert not np.shares_memory(collected[0][key],history[key])
    history['prompt'][0]=1e9
    assert collected[0]['prompt'][0]!=history['prompt'][0]
