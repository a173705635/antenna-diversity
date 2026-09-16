"""Mechanism tests, including explicit failures of authentication."""
import numpy as np
from navigation_candidate import candidate
from calibration_candidate import calibrate
from research_twenty import prompt_data
from run_gps_spp import receiver_reference
from signal_recovery import sky


def observations(east):
    ref,rot=receiver_reference();n=len(east);time=np.arange(n)*.01
    directions=sky(42001,8);sat=ref+directions*21e6
    positions=ref+np.column_stack((east,np.zeros(n),np.zeros(n)))@rot
    rho=np.linalg.norm(sat[None,:,:]-positions[:,None,:],axis=2)+75000
    z=dict(time_s=time,satellite_positions=np.tile(sat,(n,1,1)))
    for method in ('raw','calibrated'):
        z[method+'_ranges']=rho.copy();z[method+'_carrier']=rho+32100
        z[method+'_lock']=np.ones_like(rho);z[method+'_gain']=np.ones_like(rho)
        z[method+'_cn0']=np.full_like(rho,45.);z[method+'_spatial']=np.full_like(rho,20.)
    return z


def test_carrier_motion_and_prefix_causality():
    east=np.arange(100)*.02;z=observations(east);out=candidate(z)
    np.testing.assert_allclose(out[:,0],east,atol=1e-5)
    prefix={k:v[:61] for k,v in z.items()}
    np.testing.assert_allclose(candidate(prefix),out[:61],atol=1e-9)


def test_coherent_false_position_is_still_accepted_negative_control():
    # Receiver has no observation that distinguishes these from authentic signals.
    out=candidate(observations(np.full(80,200.)))
    np.testing.assert_allclose(out[:,0],200.,atol=1e-5)


def test_coherent_drag_is_still_accepted_negative_control():
    east=np.arange(100)*1.;out=candidate(observations(east))
    assert np.isfinite(out).all() and out[-1,0]>90.


def test_unique_single_carrier_slip_is_repaired_without_motion_loss():
    east=np.arange(100)*.02;z=observations(east)
    z['calibrated_carrier'][40:,0]+=100.
    out=candidate(z)
    np.testing.assert_allclose(out[:,0],east,atol=1e-5)


def test_two_simultaneous_slips_are_not_silently_integrated():
    z=observations(np.zeros(100));z['calibrated_carrier'][40:,0]+=100.
    z['calibrated_carrier'][40:,1]-=80.
    assert np.isnan(candidate(z)[40]).all()


def test_unobservable_calibration_is_not_declared_usable():
    for kind in ('two','absent'):
        for seed in (21001,21002,21003,21004):
            z,a,g=prompt_data(seed,kind)
            assert not calibrate(z,a)[1]['usable']


def test_joint_calibration_recovers_gain_with_unknown_per_prompt_phase():
    z,a,g=prompt_data(21001,'attack');estimate,info=calibrate(z,a)
    assert info['usable']
    assert np.max(abs(np.degrees(np.angle(estimate/g))))<3.
    phases=np.random.default_rng(98).uniform(-np.pi,np.pi,(8,80))
    other,other_info=calibrate(z*np.exp(1j*phases)[None,:,:],a)
    np.testing.assert_allclose(other,estimate,atol=1e-8)
    assert other_info['usable']==info['usable']
