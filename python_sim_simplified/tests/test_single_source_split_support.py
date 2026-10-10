"""Segmented spatial evidence must count every prompt in its real basis."""
import numpy as np

from core import single_source

source_evidence=single_source.source_evidence


def split_source_evidence(*args,**kwargs):
    return single_source.split_source_evidence(*args,**kwargs)


CHIP = 2046


def repeated_prompts(columns):
    return np.repeat(np.array(columns,dtype=complex).T,CHIP,axis=1)


def test_fixed_basis_matches_original_whole_window_evidence():
    block=repeated_prompts([[2,0,0],[0,1,0],[1,1,0],[1,0,1]])
    desired=np.array([1.,0.,0.])
    basis=np.eye(3)
    local=np.ones(block.shape[1])
    expected=source_evidence(block,local,desired)
    actual=split_source_evidence(block,local,desired,[basis,basis],0,4*CHIP,2*CHIP)
    np.testing.assert_allclose(actual,expected,rtol=1e-12)
    np.testing.assert_allclose(actual,4.,rtol=1e-12)


def test_crossing_block_sums_prompt_energy_with_each_dimension():
    block=repeated_prompts([[2,0,0],[0,1,0],[1,1,0],[1,1,0]])
    desired=np.array([1.,0.,0.]);local=np.ones(block.shape[1])
    basis3=np.eye(3);basis2=np.eye(3)[:,:2]
    actual=split_source_evidence(block,local,desired,[basis3,basis2],0,4*CHIP,2*CHIP)
    # On energy 6; off energy 1/(3-1) + 2/(2-1) = 2.5.
    np.testing.assert_allclose(actual,2.4,rtol=1e-12)


def test_each_segment_can_rotate_its_basis_independently():
    block=repeated_prompts([[2,0,0],[0,1,0],[1,1,0],[1,0,1]])
    desired=np.array([1.,0.,0.]);local=np.ones(block.shape[1])
    identity=np.eye(3,dtype=complex)
    rotated=np.array([[1,1j,0],[1j,1,0],[0,0,np.sqrt(2)]],complex)/np.sqrt(2)
    ordinary=split_source_evidence(block,local,desired,[identity,identity],0,4*CHIP,2*CHIP)
    changed=split_source_evidence(block,local,desired,[identity,rotated],0,4*CHIP,2*CHIP)
    np.testing.assert_allclose(changed,ordinary,rtol=1e-12)


def test_any_one_dimensional_or_low_gain_segment_invalidates_window():
    block=repeated_prompts([[2,0,0],[0,1,0],[1,1,0],[1,0,1]])
    desired=np.array([1.,0.,0.]);local=np.ones(block.shape[1])
    assert np.isnan(split_source_evidence(block,local,desired,
                  [np.eye(3),np.eye(3)[:,:1]],0,4*CHIP,2*CHIP))
    low_gain=np.eye(3)[:,:2].copy();low_gain[:,0]=[0,1,0];low_gain[:,1]=[0,0,1]
    assert np.isnan(split_source_evidence(block,local,desired,
                  [np.eye(3),low_gain],0,4*CHIP,2*CHIP))


def test_half_block_window_uses_start_and_next_basis_once_each():
    block=repeated_prompts([[2,0,0],[0,1,0]])
    desired=np.array([1.,0.,0.]);local=np.ones(block.shape[1])
    ratio=split_source_evidence(block,local,desired,
                                [np.eye(3),np.eye(3)[:,:2]],CHIP,3*CHIP,2*CHIP)
    np.testing.assert_allclose(ratio,4.,rtol=1e-12)


def test_energy_outputs_allow_independent_ratio_recalculation():
    block=repeated_prompts([[2,0,0],[0,1,0],[1,1,0],[1,1,0]])
    desired=np.array([1.,0.,0.]);local=np.ones(block.shape[1])
    ratio,on,off=split_source_evidence(block,local,desired,
                                        [np.eye(3),np.eye(3)[:,:2]],0,4*CHIP,2*CHIP,
                                        return_energy=True)
    np.testing.assert_allclose([on,off,ratio],[6.,2.5,2.4],rtol=1e-12)


def test_noise_and_mismatched_direction_stay_below_genuine_evidence():
    rng=np.random.default_rng(2701)
    desired=np.array([1.,.5,.25]);local=np.ones(20*CHIP)
    noise=.2*(rng.normal(size=(3,20))+1j*rng.normal(size=(3,20)))
    genuine=repeated_prompts((desired[:,None]+noise).T)
    other=repeated_prompts((np.array([0.,1.,0.])[:,None]+noise).T)
    white=repeated_prompts((noise/.2).T)
    bases=[np.eye(3),np.eye(3)]
    assert split_source_evidence(genuine,local,desired,bases,0,20*CHIP,10*CHIP)>3.
    assert split_source_evidence(other,local,desired,bases,0,20*CHIP,10*CHIP)<3.
    assert split_source_evidence(white,local,desired,bases,0,20*CHIP,10*CHIP)<3.
    covariance=white@white.conj().T
    _,vectors=np.linalg.eigh(covariance)
    estimated=[vectors[:,1:],vectors[:,1:]]
    assert np.isfinite(split_source_evidence(white,local,desired,estimated,0,20*CHIP,10*CHIP))
