import numpy as np
import pytest
from cold_next import align_prompts,complex_calibrate

def test_unknown_bpsk_and_residual_frequency_are_observation_invariant():
    rng=np.random.default_rng(4)
    b=np.array([1,1j,-1,.4+.2j]);bits=rng.choice([-1.,1.],100)
    z=(b[:,None]*bits*np.exp(2j*np.pi*7*np.arange(100)*.001))[...,None].transpose(0,2,1)
    aligned=align_prompts(z,'square')
    covariance=aligned[:,0]@aligned[:,0].conj().T
    np.testing.assert_allclose(covariance/np.trace(covariance),np.outer(b,b.conj())/np.vdot(b,b),atol=1e-10)
    assert np.linalg.norm(aligned.mean(axis=2))>1.5


def test_reference_alignment_survives_template_phase_restarts():
    rng=np.random.default_rng(19);vector=np.array([1.,1j,-.6,.3+.4j])
    phase=rng.uniform(-np.pi,np.pi,100)
    z=(vector[:,None]*np.exp(1j*phase))[...,None].transpose(0,2,1)
    aligned=align_prompts(z,'reference')
    np.testing.assert_allclose(aligned[:,0].mean(axis=1),vector,atol=1e-12)


def test_complex_hardware_is_recovered_without_gain_or_symbol_truth_input():
    from presets import mate70_phase_centers
    from signal_recovery import sky,steering
    rng=np.random.default_rng(77);a=steering(sky(77,8),np.array(list(mate70_phase_centers().values())),0.)
    g=np.array([1.,1.3*np.exp(.2j),.8*np.exp(-.3j),1.1*np.exp(.1j)])
    v=np.exp(1j*np.array([0.,.7,1.5,2.2]));v/=np.linalg.norm(v)
    alpha=np.exp(1j*rng.uniform(-np.pi,np.pi,8));beta=5*np.exp(1j*rng.uniform(-np.pi,np.pi,8))
    z=(g[:,None]*a*alpha+v[:,None]*beta)[:,:,None]*rng.choice([-1.,1.],(1,8,100))
    estimated,info=complex_calibrate(z,a)
    np.testing.assert_allclose(estimated,g,atol=.015)


def test_common_only_is_not_accepted_as_sky():
    from presets import mate70_phase_centers
    from signal_recovery import sky,steering
    rng=np.random.default_rng(79);a=steering(sky(79,8),np.array(list(mate70_phase_centers().values())),0.)
    v=np.exp(1j*np.array([0.,.7,1.5,2.2]))
    z=v[:,None,None]*rng.choice([-1.,1.],(1,8,100))*5
    z=z+.05*(rng.normal(size=z.shape)+1j*rng.normal(size=z.shape))
    _,info=complex_calibrate(z,a)
    assert not info['usable']


@pytest.mark.xfail(strict=True,reason='Frozen150 same-data phase alignment creates false sky support in pure noise; research candidate rejected.')
def test_frozen_complex_model_must_not_admit_pure_noise():
    from presets import mate70_phase_centers
    from signal_recovery import sky,steering
    a=steering(sky(48020,8),np.array(list(mate70_phase_centers().values())),0.)
    rng=np.random.default_rng(48020)
    z=(rng.normal(size=(4,8,100))+1j*rng.normal(size=(4,8,100)))/np.sqrt(2)
    _,info=complex_calibrate(z,a)
    assert not info['usable']
