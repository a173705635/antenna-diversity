"""Aligned residual signal should be distinguished from residual noise."""
import numpy as np
from core.spoofing.single_source import source_evidence


def test_residual_genuine_signal_has_directional_evidence():
    rng=np.random.default_rng(116004);local=np.ones(20*2046)
    desired=np.array([1.+0j,.5+.5j])
    prompts=desired[:,None]+.2*(rng.normal(size=(2,20))+1j*rng.normal(size=(2,20)))
    block=np.repeat(prompts,2046,axis=1)
    assert source_evidence(block,local,desired)>=3.


def test_residual_noise_is_not_directional_evidence():
    rng=np.random.default_rng(116005);local=np.ones(20*2046);desired=np.array([1.+0j,.5+.5j])
    prompts=rng.normal(size=(2,20))+1j*rng.normal(size=(2,20))
    assert source_evidence(np.repeat(prompts,2046,axis=1),local,desired)<3.


def test_evidence_does_not_depend_on_basis_or_steering_scale():
    rng=np.random.default_rng(116006);local=np.ones(20*2046);desired=np.array([1.+0j,.5+.5j])
    prompts=desired[:,None]+.2*(rng.normal(size=(2,20))+1j*rng.normal(size=(2,20)))
    block=np.repeat(prompts,2046,axis=1);unitary=np.array([[1,1j],[1j,1]])/np.sqrt(2)
    np.testing.assert_allclose(source_evidence(unitary@block,local,2*unitary@desired),source_evidence(block,local,desired),rtol=1e-12)
