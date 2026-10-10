"""Complex same-PRN prompt grouping for the fixed block-length diagnosis."""
from copy import deepcopy

import numpy as np
import pytest

from core.single_source import residual_trigger
from experiments.single_source_beam_coherence_lengths import (
    LENGTHS,audit_saved_lengths,fit_prompt_lengths,group_prompts,
)


def test_groups_never_mix_prns_and_constant_phase_gains_sqrt_length():
    prompts=np.empty((3,2,20),complex)
    first=np.array([1+1j,2-1j,-.5j])
    second=np.array([-2j,3+1j,1.5])
    prompts[:,0,:]=first[:,None]
    prompts[:,1,:]=second[:,None]
    for length in LENGTHS:
        grouped=group_prompts(prompts,length)
        assert grouped.shape==(3,2*(20//length))
        np.testing.assert_allclose(grouped[:,:20//length],
                                   np.repeat(np.sqrt(length)*first[:,None],
                                             20//length,axis=1),atol=1e-14)
        np.testing.assert_allclose(grouped[:,20//length:],
                                   np.repeat(np.sqrt(length)*second[:,None],
                                             20//length,axis=1),atol=1e-14)


def test_bit_flip_inside_twenty_ms_cancels_without_truth_wiping():
    prompts=np.ones((3,1,20),complex)
    prompts[:,:,10:]=-1.
    np.testing.assert_array_equal(group_prompts(prompts,20),np.zeros((3,1)))
    np.testing.assert_allclose(group_prompts(prompts,10),
                               np.array([[np.sqrt(10),-np.sqrt(10)]]*3),
                               atol=1e-14)
    np.testing.assert_allclose(group_prompts(prompts,5),
                               np.array([[np.sqrt(5),np.sqrt(5),
                                          -np.sqrt(5),-np.sqrt(5)]]*3),
                               atol=1e-14)


def test_fit_uses_grouped_column_count_and_correct_desired_repetition():
    rng=np.random.default_rng(120)
    prompts=rng.normal(size=(3,2,20))+1j*rng.normal(size=(3,2,20))
    nominal=np.array([[1.,1.],[1.,-1.],[1.,1j]])
    noise=.4
    fitted,raw,would,gain=fit_prompt_lengths(prompts,nominal,nominal,noise)
    for index,length in enumerate(LENGTHS):
        grouped=group_prompts(prompts,length)
        desired=np.repeat(nominal,20//length,axis=1)
        expected_p,expected_raw=residual_trigger(grouped,desired,noise)
        np.testing.assert_allclose(fitted[index],expected_p,atol=1e-14)
        assert raw[index]==expected_raw
        np.testing.assert_allclose(would[index],
            expected_p if expected_raw>=3 else np.eye(3),atol=1e-14)
        expected_gain=np.einsum('mi,mn,ni->i',nominal.conj(),
                                would[index],nominal).real/3
        np.testing.assert_allclose(gain[index],expected_gain,atol=1e-14)
    assert [group_prompts(prompts,length).shape[1] for length in LENGTHS]==[
        40,8,4,2]


def test_frozen_one_and_five_ms_oracle_must_match_raw_projector_and_gain():
    fitted=np.broadcast_to(np.eye(3),(4,6,3,3)).astype(complex).copy()
    gain=np.ones((4,6,2))
    raw=np.zeros((4,6))
    raw[:,5]=[2.,4.,2.5,4.5]
    fit=dict(raw=raw,fitted_projector=fitted,would_projector=fitted.copy(),
             would_gain=gain,noise_power_per_prompt=np.ones(6),
             startup_prns=np.array([0,1]),aided_code_rate_hz=np.ones(2),
             calibration=np.ones(3,dtype=complex),calibration_rate=np.zeros(3))
    reference=deepcopy(fit)
    for key in ('raw','fitted_projector','would_projector','would_gain'):
        reference[key][2,5]=fit[key][0,5]
        reference[key][3,5]=fit[key][1,5]
    audit_saved_lengths(fit,reference)
    for key in ('raw','fitted_projector','would_projector','would_gain'):
        corrupted=deepcopy(reference)
        corrupted[key][3,5]+=1.
        with pytest.raises(AssertionError):
            audit_saved_lengths(fit,corrupted)
