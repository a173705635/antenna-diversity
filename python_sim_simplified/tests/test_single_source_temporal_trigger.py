"""Causal raw-trigger history must select the current block projector."""
import numpy as np

from core import single_source
from core.recovery_geometry import FS


def test_history_uses_only_measured_blocks_and_drops_oldest_at_five():
    history=[]
    outputs=[single_source.temporal_beam_trigger(raw,history)
             for raw in (20.,1.,1.,1.,1.,1.)]
    np.testing.assert_allclose([item[0] for item in outputs],
                               [20.,10.5,22./3.,23./4.,24./5.,1.])
    np.testing.assert_allclose([item[1] for item in outputs],
                               [20.,10.5,22./3.,23./4.,24./5.,1.])
    assert [item[2] for item in outputs]==[False,True,True,True,True,False]
    assert sum(item[2] for item in outputs)==4
    assert history==[1.]*5
    low_history=[]
    assert all(single_source.temporal_beam_trigger(2.,low_history)[1]<3.
               for _ in range(8))


def test_history_prefix_is_unaffected_by_future_blocks():
    def run(raw):
        history=[]
        return [single_source.temporal_beam_trigger(value,history) for value in raw]
    full=run([4.,1.,9.,0.,0.,20.])
    assert full[:4]==run([4.,1.,9.,0.])
    assert full[0]==(4.,4.,False)


def test_history_only_uses_current_projector_for_beam_weight_and_gain(monkeypatch):
    p_first=np.diag([0.,1.,1.]).astype(complex)
    p_current=np.diag([1.,0.,1.]).astype(complex)
    sequence=iter(((p_first,20.),(p_current,1.)))
    monkeypatch.setattr(single_source,'residual_trigger',
                        lambda *args:next(sequence))
    steering=np.array([[1.,1.,1.,1.],
                       [1.,1j,-1.,-1j],
                       [1.,-1.,1.,-1.]],complex)
    block=np.ones((3,2046),complex)
    local=np.ones((1,2046),complex)
    history=[]
    single_source.beam(block,steering,local,np.eye(3),[0],
                       trigger_history=history)
    weight,gain,raw,mean,effective=single_source.beam(
        block,steering,local,np.eye(3),[0],trigger_history=history)
    projected=p_current@steering
    response=np.sum(steering.conj()*projected,axis=0).real
    np.testing.assert_array_equal(weight,projected/(response+.01)[None,:])
    np.testing.assert_array_equal(gain,response/3.)
    assert (raw,mean,effective)==(1.,10.5,10.5)


def test_frontend_history_only_records_current_basis_gain_and_trace(monkeypatch):
    monkeypatch.setattr(single_source,'acquire_array',
                        lambda *args,**kwargs:{'success':True})
    starts=np.arange(0.,.161,.001)
    history=dict(epoch_start_samples=np.rint(starts*FS).astype(int),
                 code_nco_phase_chips=np.zeros(len(starts)),
                 code_nco_rate_hz=np.zeros(len(starts)),
                 carrier_nco_phase_rad=np.zeros(len(starts)),
                 carrier_nco_frequency_hz=np.zeros(len(starts)),
                 phase_lock_metric=np.ones(len(starts)),
                 cn0_db_hz=np.full(len(starts),30.))
    monkeypatch.setattr(single_source,'track',
                        lambda *args,**kwargs:{'success':True,**history})
    monkeypatch.setattr(single_source,'local_signal',
                        lambda code,template,t:np.ones(len(t),complex))
    monkeypatch.setattr(single_source,'_sample_code',
                        lambda code,phase:np.ones(len(phase)))
    monkeypatch.setattr(single_source,'calibrate',
                        lambda block,local,desired:
                        (np.ones(len(block),complex),np.zeros(len(block))))
    received={}
    def receive_measured(signal,prn,*args):
        received[prn]=signal.copy()
        return {'segments':[{'history':history}]}
    monkeypatch.setattr(single_source,'receive_measured',receive_measured)
    monkeypatch.setattr(single_source,'code_pseudorange',
                        lambda h,t,*args:np.zeros(len(t)))
    monkeypatch.setattr(single_source,'carrier_range',
                        lambda h,t,*args:np.zeros(len(t)))
    projectors=[np.diag([0.,1.,1.]).astype(complex),
                np.diag([1.,0.,1.]).astype(complex),
                np.diag([1.,0.,1.]).astype(complex)]
    raw=[20.,1.,1.]
    sequence=iter(zip(projectors,raw))
    monkeypatch.setattr(single_source,'residual_trigger',
                        lambda *args:next(sequence))
    nominal=np.array([[1.,1.,1.,1.],[1.,1j,-1.,-1j],
                      [1.,-1.,1.,-1.]],complex)
    rng=np.random.default_rng(17)
    shape=(3,round(.16*FS))
    samples=np.ones(shape,complex)+.01*(rng.normal(size=shape)+1j*rng.normal(size=shape))
    record=dict(samples=samples,nominal=nominal,
                codes=np.zeros((4,1)),times=np.array([.10,.12]),
                satellites=np.zeros((2,4,3)),coarse=np.zeros(4))
    result=single_source.frontend(record,[0,1,2],split_support=True,
                    evidence_trace=True,temporal_beam_trigger_enabled=True)
    assert result['beam_trigger_raw'][-2]==1.
    assert result['beam_trigger_effective'][-2]==10.5
    assert result['beam_trigger_history_only'][-2]
    np.testing.assert_array_equal(result['trace_projectors'][6],projectors[1])
    response=np.sum(nominal.conj()*(projectors[1]@nominal),axis=0).real/3.
    np.testing.assert_array_equal(result['gain'][1],response)
    np.testing.assert_array_equal(result['ratio'],[20.,1.])
    block_start=round(.12*FS)
    block_end=round(.14*FS)
    projected=projectors[1]@nominal
    unscaled=np.sum(nominal.conj()*projected,axis=0).real
    current_weights=projected/(unscaled+.01)[None,:]
    np.testing.assert_allclose(received[1][block_start:block_end],
            current_weights[:,0].conj()@samples[:,block_start:block_end],atol=1e-14)
