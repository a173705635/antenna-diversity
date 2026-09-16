"""Observable arc contracts; nominal covariance is not authentication."""
import numpy as np
import navigation_third as model
from navigation_third_cases import generate

def test_lock_loss_does_not_bridge_old_ambiguity():
    pair,_=generate(99101,'lock_ambiguity')
    steps=model.carrier_steps(pair['joint'])[0]
    assert np.linalg.norm(steps[31])<.1

def test_time_gap_has_no_phase_continuity_certificate():
    pair,_=generate(99103,'time_gap')
    steps=model.carrier_steps(pair['joint'])[0]
    assert np.isnan(steps[40]).all()

def test_constant_ambiguity_and_clock_are_nuisance():
    pair,_=generate(99107,'normal');z=pair['joint']
    before=model.carrier_steps(z)[0]
    z['calibrated_carrier']+=np.arange(8)[None,:]*123.41+np.arange(120)[:,None]*2.
    after=model.carrier_steps(z)[0]
    np.testing.assert_allclose(before,after,atol=1e-7,equal_nan=True)

def test_shared_endpoints_cancel_in_coefficient_history():
    pair,_=generate(99109,'normal')
    steps,slips,covs,history,noise=model.carrier_steps(pair['joint'])
    active=np.flatnonzero(np.linalg.norm(history[40],axis=(1,2))>1e-9)
    np.testing.assert_array_equal(active,[0,40])
    expected=np.einsum('tis,ts,tjs->ij',history[40],noise,history[40])
    np.testing.assert_allclose(covs[40],expected,atol=1e-12)

def test_code_covariance_excludes_rejected_geometry():
    pair,_=generate(99111,'weak_subset');z=pair['joint']
    before=model.covariance(z,20,'ranges')
    z['calibrated_ranges'][:,:2]=np.nan
    np.testing.assert_allclose(model.covariance(z,20,'ranges'),before)

def test_causal_prefix_with_shared_phase_covariance():
    pair,_=generate(99113,'normal')
    prefix={route:{k:(v[:45] if v.ndim and len(v)==120 else v) for k,v in z.items()} for route,z in pair.items()}
    np.testing.assert_allclose(model.candidate(prefix),model.candidate(pair)[:45],atol=1e-7,equal_nan=True)

def test_coherent_absolute_code_bias_remains_counterexample():
    pair,truth=generate(99117,'fixed_geometry_bias');out=model.candidate(pair)
    good=np.isfinite(out).all(axis=1)
    assert good.sum()>60
    assert np.min(np.linalg.norm((out-truth)[good,:2],axis=1))>150

def test_output_covariance_is_causal_and_positive_semidefinite():
    pair,_=generate(99119,'normal');z=pair['joint']
    prefix={k:(v[:45] if v.ndim and len(v)==120 else v) for k,v in z.items()}
    full,cov=model.calibrated(z,True);short,partial=model.calibrated(prefix,True)
    np.testing.assert_allclose(partial,cov[:45],atol=1e-8,equal_nan=True)
    valid=np.isfinite(full).all(axis=1)
    assert np.linalg.eigvalsh(cov[valid]).min()>-1e-9
