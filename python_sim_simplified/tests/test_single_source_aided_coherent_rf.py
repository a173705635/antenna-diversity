"""Optional beam-template rate and complex five-millisecond prompts."""
import numpy as np
import pytest

from core import single_source
from core.recovery_geometry import FS


@pytest.mark.parametrize('coherent_ms',(5,10))
def test_beam_groups_same_prn_complex_sums_and_current_fit(monkeypatch,
                                                             coherent_ms):
    rng=np.random.default_rng(31)
    block=rng.normal(size=(3,round(.02*FS)))+1j*rng.normal(
        size=(3,round(.02*FS)))
    local=np.ones((2,block.shape[1]),complex)
    nominal=np.array([[1.,1.],[1.,-1.],[1.,1j]])
    current=np.diag([1.,0.,1.]).astype(complex)
    captured={}
    def fitted(prompts,desired,noise):
        captured.update(prompts=prompts.copy(),desired=desired.copy(),noise=noise)
        return current,4.
    monkeypatch.setattr(single_source,'residual_trigger',fitted)
    weight,gain,ratio=single_source.beam(block,nominal,local,
        np.eye(3),[0,1],coherent_ms=coherent_ms)
    one=block.reshape(3,20,2046).mean(axis=2)
    groups=20//coherent_ms
    expected=np.stack([one.reshape(3,groups,coherent_ms).sum(axis=2)/
                       np.sqrt(coherent_ms)]*2,
                      axis=1).reshape(3,-1)
    np.testing.assert_allclose(captured['prompts'],expected,rtol=0,atol=1e-14)
    np.testing.assert_array_equal(captured['desired'],
                                  np.repeat(nominal,groups,axis=1))
    assert captured['prompts'].shape==(3,2*groups)
    assert captured['noise']==float(np.mean(abs(block)**2)/2046)
    projected=current@nominal
    response=np.sum(nominal.conj()*projected,axis=0).real
    np.testing.assert_array_equal(weight,projected/(response+.01)[None,:])
    np.testing.assert_array_equal(gain,response/3.)
    assert ratio==4.


@pytest.mark.parametrize('coherent_ms',(5,10))
def test_frontend_aiding_keeps_startup_calibration_and_projects_actual_stream(
        monkeypatch,tmp_path,coherent_ms):
    monkeypatch.setattr(single_source,'acquire_array',
                        lambda *args,**kwargs:{'success':True})
    startup_rate=1.023e6+2.
    carrier_frequency=1000.
    history=dict(epoch_start_samples=np.array([round(.09*FS)]),
                 code_nco_phase_chips=np.array([123.]),
                 code_nco_rate_hz=np.array([startup_rate]),
                 carrier_nco_frequency_hz=np.array([carrier_frequency]))
    monkeypatch.setattr(single_source,'track',
                        lambda *args,**kwargs:{'success':True,**history})
    templates=[]
    def local(_code,template,time):
        templates.append(template)
        return np.exp(1j*(template[2]-1.023e6)*time)
    monkeypatch.setattr(single_source,'local_signal',local)
    calibration=[]
    def calibrate(_block,local0,desired):
        calibration.append(local0.copy())
        return np.ones(3,complex),np.zeros(3)
    monkeypatch.setattr(single_source,'calibrate',calibrate)
    current=np.diag([0.,1.,1.]).astype(complex)
    scores=[]
    def fitted(prompts,desired,noise):
        scores.append((prompts.shape,desired.shape,noise))
        return current,4.
    monkeypatch.setattr(single_source,'residual_trigger',fitted)
    received={}
    def receive(signal,prn,*_args):
        received[prn]=signal.copy()
        return {'segments':[]}
    monkeypatch.setattr(single_source,'receive_measured',receive)
    nominal=np.array([[1.,1.,1.,1.],[1.,1j,-1.,-1j],
                      [1.,-1.,1.,-1.]],complex)
    rng=np.random.default_rng(99)
    shape=(3,round(.14*FS))
    samples=rng.normal(size=shape)+1j*rng.normal(size=shape)
    record=dict(samples=samples,nominal=nominal,codes=np.zeros((4,1)),
                times=np.array([.10]),satellites=np.zeros((1,4,3)),
                coarse=np.zeros(4))
    result=single_source.frontend(record,[0,1,2],split_support=True,
            evidence_trace=True,beam_template_carrier_aiding=True,
            beam_coherent_ms=coherent_ms)
    aided_rate=1.023e6*(1+carrier_frequency/1575.42e6)
    assert len(calibration)==1
    assert all(template[2]==startup_rate for template in templates[:4])
    assert all(template[2]==aided_rate for template in templates[4:])
    assert len(scores)==2 and all(item[0]==(3,4*(20//coherent_ms))
                                  for item in scores)
    np.testing.assert_array_equal(result['trace_raw_trigger'][:5],
                                  np.full(5,np.nan))
    np.testing.assert_array_equal(result['trace_raw_trigger'][5:],[4.,4.])
    np.testing.assert_allclose(result['trace_projectors'][5],current,atol=1e-14)
    projected=current@nominal
    response=np.sum(nominal.conj()*projected,axis=0).real
    weight=projected/(response+.01)[None,:]
    begin=round(.1*FS);end=round(.12*FS)
    np.testing.assert_allclose(received[1][begin:end],
        weight[:,0].conj()@samples[:,begin:end],rtol=0,atol=1e-13)
    reference_stream={prn:signal.copy() for prn,signal in received.items()}
    stream_path=tmp_path/'beam_stream.dat'
    mapped=single_source.frontend(record,[0,1,2],split_support=True,
        evidence_trace=True,beam_template_carrier_aiding=True,
        beam_coherent_ms=coherent_ms,stream_memmap_path=stream_path)
    assert set(mapped)==set(result)
    for name,value in result.items():
        np.testing.assert_array_equal(mapped[name],value,err_msg=name)
    for prn,signal in reference_stream.items():
        np.testing.assert_array_equal(received[prn],signal)
    mapped_stream=np.memmap(stream_path,dtype=np.complex128,mode='r',
                            shape=(4,shape[1]),order='C')
    assert mapped_stream.flags.c_contiguous
    for prn,signal in reference_stream.items():
        np.testing.assert_array_equal(mapped_stream[prn-1],signal)
    del mapped_stream
    stream_path.unlink()
    assert not stream_path.exists()
    original=single_source.frontend(record,[0,1,2],split_support=True,
                                    evidence_trace=True)
    default_stream={prn:signal.copy() for prn,signal in received.items()}
    default_path=tmp_path/'default_stream.dat'
    mapped_default=single_source.frontend(record,[0,1,2],split_support=True,
        evidence_trace=True,stream_memmap_path=default_path)
    assert set(mapped_default)==set(original)
    for name,value in original.items():
        np.testing.assert_array_equal(mapped_default[name],value,err_msg=name)
    for prn,signal in default_stream.items():
        np.testing.assert_array_equal(received[prn],signal)
    default_path.unlink()
    explicit=single_source.frontend(record,[0,1,2],split_support=True,
        evidence_trace=True,beam_template_carrier_aiding=False,
        beam_coherent_ms=1)
    assert set(original)==set(explicit)
    assert 'trace_raw_trigger' not in original
    for name,value in original.items():
        np.testing.assert_array_equal(value,explicit[name],err_msg=name)
