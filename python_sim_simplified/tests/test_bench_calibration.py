import numpy as np

from audit_w4_w6 import synthetic_bench,analyze_bench
from bench_calibration import fit_rf,evaluate_rf,fit_imu,correct_imu,time_offset_profile


def test_complete_synthetic_bench_holdout_and_outside_domain():
    data,_=synthetic_bench();result=analyze_bench(data)
    assert result['rf_model']['rank']==7 and result['imu_model']['rank']==8
    assert result['rf_holdout_valid'] and result['rf_outside_valid_count']==0
    assert np.max(result['rf_phase_holdout_rms_rad'])<.007
    assert np.max(result['imu_holdout_rms'][:3])<.01
    assert np.max(result['imu_holdout_rms'][3:])<.0001
    assert result['sync_profile']['offset_s']==.017


def test_unknown_fixture_can_bias_calibration_despite_good_holdout_fit():
    data,_=synthetic_bench();good=fit_rf(data['rf_train'])
    fixture_error=np.exp(1j*np.array([0.,.2,-.1,.15]))
    wrong=dict(data['rf_train'],fixture_transfer=data['rf_train']['fixture_transfer']*fixture_error)
    bad=fit_rf(wrong)
    assert np.linalg.norm(good['phase_coefficient'][0]-bad['phase_coefficient'][0])>.1
    test=dict(data['rf_holdout'],fixture_transfer=data['rf_holdout']['fixture_transfer']*fixture_error)
    assert np.sqrt(np.mean(evaluate_rf(bad,test)['phase_residual_rad']**2))<.007


def test_ignoring_large_fixture_phase_can_break_the_supplied_branch_anchor():
    data,_=synthetic_bench()
    train=dict(data['rf_train'],fixture_transfer=np.ones_like(data['rf_train']['fixture_transfer']))
    test=dict(data['rf_holdout'],fixture_transfer=np.ones_like(data['rf_holdout']['fixture_transfer']))
    assert np.sqrt(np.mean(evaluate_rf(fit_rf(train),test)['phase_residual_rad']**2))>.1


def test_imu_affine_calibration_recovers_independent_holdout_exactly():
    rng=np.random.default_rng(77);reference=rng.normal(size=(100,6));temp=rng.uniform(0,40,100)
    matrix=np.eye(6)+rng.normal(0,.01,(6,6));bias=rng.normal(0,.02,6);thermal=rng.normal(0,.001,6)
    measured=reference@matrix.T+bias+(temp-25)[:,None]*thermal
    model=fit_imu(reference[:70],measured[:70],temp[:70])
    np.testing.assert_allclose(correct_imu(model,measured[70:],temp[70:]),reference[70:],atol=1e-13)


def test_static_time_alignment_has_flat_profile_and_zero_excitation():
    t=np.linspace(0,1,101);reference=np.tile([0.,0.,.1],(101,1))
    result=time_offset_profile(t[10:-10],reference[10:-10],t,reference,np.array([-.01,0.,.01]))
    np.testing.assert_allclose(result['scores'],result['scores'][0],atol=1e-25)
    assert result['excitation']<1e-25
    assert result['offset_s'] is None and result['status']=='unobservable_constant_signal'
