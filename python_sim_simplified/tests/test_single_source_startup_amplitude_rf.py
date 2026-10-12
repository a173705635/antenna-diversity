"""The optional startup magnitude model changes the actual beam path."""
import numpy as np

from core.spoofing import single_source
from core.spoofing.array_processing import startup_relative_amplitude
from core.spoofing.recovery_geometry import FS


def test_three_antenna_covariance_recovers_rms_relative_magnitude():
    prns=np.array([3,0,2])
    nominal=np.exp(1j*np.array([[0.,.2,.5,-.1],
                                [.4,-.3,.7,.2],[-.7,.5,-.2,.9]]))
    magnitude=np.array([1.,1.15,.82])
    bits=np.array([[1.,-1.,1.,-1.],[-1.,-1.,1.,1.],[1.,1.,-1.,-1.]])
    prompts=magnitude[:,None,None]*nominal[:,prns,None]*bits[None,:,:]*(
        np.array([2.+1j,.7-2j,1.5+.4j])[None,:,None])
    estimate,_=startup_relative_amplitude(prompts,nominal,prns)
    np.testing.assert_allclose(estimate,magnitude/
        np.sqrt(np.mean(magnitude**2)),rtol=0,atol=1e-14)


def test_enabled_frontend_uses_model_for_fit_stream_gain_and_trace(monkeypatch):
    monkeypatch.setattr(single_source,'acquire_array',
                        lambda *args,**kwargs:{'success':True})
    history=dict(epoch_start_samples=np.array([round(.09*FS)]),
                 code_nco_phase_chips=np.array([123.]),
                 code_nco_rate_hz=np.array([1.023e6]),
                 carrier_nco_frequency_hz=np.array([0.]))
    monkeypatch.setattr(single_source,'track',
                        lambda *args,**kwargs:{'success':True,**history})
    monkeypatch.setattr(single_source,'calibrate',
                        lambda *args,**kwargs:(np.ones(3,complex),np.zeros(3)))
    def local(code,template,time):
        return np.exp(2j*np.pi*1000*int(code[0])*time)
    monkeypatch.setattr(single_source,'local_signal',local)
    projector=np.diag([1.,1.,0.]).astype(complex)
    fitted=[]
    def trigger(prompts,desired,noise):
        fitted.append(desired.copy())
        return projector,4.
    monkeypatch.setattr(single_source,'residual_trigger',trigger)
    streams={}
    def receive(signal,prn,*_):
        streams[prn]=signal.copy()
        return {'segments':[]}
    monkeypatch.setattr(single_source,'receive_measured',receive)
    nominal=np.exp(1j*np.array([[0.,.2,.5,-.1],
                                [.4,-.3,.7,.2],[-.7,.5,-.2,.9]]))
    magnitude=np.array([1.,1.15,.82])
    samples=np.arange(round(.14*FS))/FS
    carriers=np.stack([local([star],None,samples) for star in range(4)])
    iq=(magnitude[:,None,None]*nominal[:,:,None]*carriers[None,:,:]).sum(axis=1)
    record=dict(samples=iq,nominal=nominal,
                codes=np.arange(4)[:,None],times=np.array([.10]),
                satellites=np.zeros((1,4,3)),coarse=np.zeros(4))
    default=single_source.frontend(record,[0,1,2],split_support=True,
                                   evidence_trace=True,
                                   beam_template_carrier_aiding=True,
                                   beam_coherent_ms=5)
    default_stream=streams[1].copy()
    fitted.clear()
    enabled=single_source.frontend(record,[0,1,2],split_support=True,
                                   evidence_trace=True,
                                   beam_template_carrier_aiding=True,
                                   beam_coherent_ms=5,
                                   beam_startup_amplitude=True)
    model=magnitude/np.sqrt(np.mean(magnitude**2))
    expected=model[:,None]*nominal
    np.testing.assert_allclose(enabled['startup_relative_amplitude'],model,
                               rtol=0,atol=1e-12)
    np.testing.assert_allclose(enabled['trace_nominal'],expected,
                               rtol=0,atol=1e-12)
    for desired in fitted:
        np.testing.assert_allclose(desired,np.repeat(expected,4,axis=1),
                                   rtol=0,atol=1e-12)
    projected=projector@expected
    response=np.sum(expected.conj()*projected,axis=0).real
    weight=projected/(response+.01)[None,:]
    begin=round(.1*FS);end=round(.12*FS)
    np.testing.assert_allclose(streams[1][begin:end],
        weight[:,0].conj()@iq[:,begin:end],rtol=0,atol=1e-12)
    np.testing.assert_allclose(enabled['gain'][0],response/3,
                               rtol=0,atol=1e-12)
    assert not np.array_equal(default_stream[begin:end],streams[1][begin:end])
    assert 'startup_relative_amplitude' not in default
    explicit_off=single_source.frontend(record,[0,1,2],split_support=True,
        evidence_trace=True,beam_template_carrier_aiding=True,
        beam_coherent_ms=5,beam_startup_amplitude=False)
    assert set(default)==set(explicit_off)
    for name,value in default.items():
        np.testing.assert_array_equal(value,explicit_off[name],err_msg=name)
