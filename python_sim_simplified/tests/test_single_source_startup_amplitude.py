"""Startup amplitude uses calibrated, same-PRN 1 ms prompts only."""
import numpy as np

import experiments.spoofing.single_source_startup_amplitude as study
from core.spoofing.single_source import residual_trigger


def test_cross_prn_covariance_recovers_relative_amplitudes_without_bit_wiping():
    steering=np.exp(1j*np.array([[0.,.3],[.4,-.2],[-.6,.5]]))
    physical=np.array([1.,1.2,.8])
    bits=np.array([[1.,-1.,1.,-1.],[-1.,-1.,1.,1.]])
    prompts=(physical[:,None,None]*steering[:,:,None]*bits[None,:,:]
             *np.array([2.+1j,.7-2j])[None,:,None])
    ratio,covariance=study.estimate_amplitude(prompts,steering,np.array([0,1]))
    np.testing.assert_allclose(ratio,physical/np.sqrt(np.mean(physical**2)),
                               rtol=0,atol=1e-14)
    assert covariance.shape==(3,3)


def test_amplitude_covariance_uses_prn_ids_not_prompt_column_numbers():
    prns=np.array([3,0,2])
    nominal=np.exp(1j*np.array([[0.,.2,.5,-.1],
                                [.4,-.3,.7,.2],[-.7,.5,-.2,.9]]))
    physical=np.array([1.,1.15,.82])
    bits=np.array([[1.,-1.,1.,-1.],[-1.,-1.,1.,1.],[1.,1.,-1.,-1.]])
    strengths=np.array([2.+1j,.7-2j,1.5+.4j])
    prompts=physical[:,None,None]*nominal[:,prns,None]*bits[None,:,:]*(
        strengths[None,:,None])
    ratio,_=study.estimate_amplitude(prompts,nominal,prns)
    np.testing.assert_allclose(ratio,physical/np.sqrt(np.mean(physical**2)),
                               rtol=0,atol=1e-14)


def test_saved_startup_prompts_ignore_later_samples_and_are_direct_dot_products(monkeypatch):
    milliseconds=100
    samples=np.arange(milliseconds*2046,dtype=float)
    signal=(1.+samples/1e7)*np.exp(1j*samples/5000)
    record=dict(samples=np.vstack((signal,2*signal,3*signal)),
                codes=np.ones((1,2046)))
    template=np.zeros((1,4))
    monkeypatch.setattr(study,'local_signal',
                        lambda code,template,t:np.ones(len(t),complex))
    first=study.startup_prompts(record,np.array([0]),template,
                                np.ones(3,complex),np.zeros(3))
    expected=np.stack([signal.reshape(milliseconds,2046).mean(axis=1)*gain
                       for gain in (1,2,3)])[:,None,:]
    np.testing.assert_allclose(first,expected,rtol=0,atol=1e-12)
    record['samples']=np.pad(record['samples'],((0,0),(0,2046)),
                             constant_values=1e6)
    np.testing.assert_array_equal(first,study.startup_prompts(record,
        np.array([0]),template,np.ones(3,complex),np.zeros(3)))


def test_saved_post_groups_and_noise_match_independent_sample_sums(monkeypatch):
    rng=np.random.default_rng(31)
    samples=rng.normal(size=(3,study.WIDTH*6))+1j*rng.normal(
        size=(3,study.WIDTH*6))
    nominal=np.exp(1j*np.array([[0.,.2],[.4,-.3],[-.7,.5]]))
    record=dict(samples=samples,nominal=nominal,codes=np.ones((2,2046)))
    monkeypatch.setattr(study,'local_signal',
                        lambda code,template,t:np.ones(len(t),complex))
    fit=study.fit_blocks(record,np.array([0,1]),np.zeros((2,4)),
                         np.ones(3,complex),np.zeros(3),
                         np.array([1.1,1.,.9])/np.sqrt(np.mean(
                             np.array([1.1,1.,.9])**2)))
    assert np.isnan(fit['noise_power'][:5]).all()
    np.testing.assert_allclose(fit['noise_power'][5],
        np.mean(abs(samples[:,5*study.WIDTH:6*study.WIDTH])**2)/2046,
        rtol=0,atol=1e-15)
    block=samples[:,5*study.WIDTH:6*study.WIDTH]
    expected=block.reshape(3,4,5,2046).mean(axis=-1).sum(axis=-1)/np.sqrt(5)
    np.testing.assert_allclose(fit['grouped_prompts'][5],
        np.stack((expected,expected),axis=1).reshape(3,8),rtol=0,atol=1e-15)
    for mode,model in enumerate((nominal,fit['relative_amplitude'][:,None]*nominal)):
        p=fit['would_projector'][mode,5]
        np.testing.assert_allclose(fit['would_gain'][mode,5],
            np.einsum('mi,mn,ni->i',model.conj(),p,model).real/3,
            rtol=0,atol=1e-15)
    np.testing.assert_allclose(fit['nominal_gain'][1,5],
        np.einsum('mi,mn,ni->i',nominal.conj(),fit['would_projector'][1,5],
                  nominal).real/3,rtol=0,atol=1e-15)
    assert study.audit_saved_raw(fit,nominal,np.array([0,1]))<1e-12
    fit['noise_power'][5]*=2
    with np.testing.assert_raises(AssertionError):
        study.audit_saved_raw(fit,nominal,np.array([0,1]))


def test_raw_reduction_matches_augmented_complex_least_squares():
    rng=np.random.default_rng(42)
    prompts=rng.normal(size=(3,12))+1j*rng.normal(size=(3,12))
    model=np.exp(1j*rng.uniform(-1.,1.,size=(3,12)))*np.array(
        [1.1,1.,.9])[:,None]
    model/=np.sqrt(np.mean(abs(model)**2,axis=0))[None,:]
    noise=.37
    projector,raw=residual_trigger(prompts,model,noise)
    _,vectors=np.linalg.eigh(projector)
    nuisance=vectors[:,0]
    reduction=0.
    for column in range(prompts.shape[1]):
        desired=model[:,column]
        z=prompts[:,column]
        single=z-desired*(np.vdot(desired,z)/np.vdot(desired,desired))
        design=np.column_stack((desired,nuisance))
        penalized=np.vstack((design,[[np.sqrt(.01*3),0.]]))
        solution=np.linalg.lstsq(penalized,np.r_[z,0.],rcond=None)[0]
        residual=z-design@solution
        reduction+=np.vdot(single,single).real-np.vdot(residual,residual).real
    np.testing.assert_allclose(raw,reduction/(noise*(12+3-1)),rtol=0,atol=1e-13)


def test_model_weights_use_untriggered_and_triggered_branches():
    model=np.exp(1j*np.array([[0.,.2],[.4,-.3],[-.7,.5]]))*np.array(
        [1.1,1.,.9])[:,None]
    model/=np.sqrt(np.mean(abs(model[:,0])**2))
    projector=np.diag([1.,1.,0.]).astype(complex)
    weights=study.model_weights(np.stack((np.eye(3),projector)),model,
                                np.array([False,True]))
    np.testing.assert_allclose(weights[0],model/3,rtol=0,atol=1e-16)
    projected=projector@model
    expected=projected/(
        np.sum(model.conj()*projected,axis=0).real+.01)[None,:]
    np.testing.assert_allclose(weights[1],expected,rtol=0,atol=1e-16)
