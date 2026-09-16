"""Identifiability and causal-navigation regression checks for this campaign."""
import numpy as np
from research_hundred import load,ROOT
from test_research_twenty import observations


def test_duplicate_startup_evidence_does_not_anchor():
    candidate=load(ROOT/'iteration001/navigation_hundred.py').candidate
    z=observations(np.full(80,200.));z['startup_prns']=np.array([],int)
    assert np.isnan(candidate(z)).all()


def test_distinct_processing_still_cannot_authenticate_consistent_forgery():
    candidate=load(ROOT/'iteration001/navigation_hundred.py').candidate
    z=observations(np.full(80,200.));z['startup_prns']=np.arange(8)
    np.testing.assert_allclose(candidate(z)[:,0],200.,atol=1e-5)


def test_navigation_prefix_causality_with_provenance():
    candidate=load(ROOT/'iteration001/navigation_hundred.py').candidate
    z=observations(np.arange(100)*.02);z['startup_prns']=np.arange(8)
    prefix={k:(v if k=='startup_prns' else v[:61]) for k,v in z.items()}
    np.testing.assert_allclose(candidate(prefix),candidate(z)[:61],atol=1e-9)


def test_planar_mirror_directions_have_identical_array_phase():
    from presets import mate70_phase_centers
    from signal_recovery import steering
    geometry=np.array(list(mate70_phase_centers().values()))
    directions=np.array([[.3,.4,np.sqrt(.75)],[.3,.4,-np.sqrt(.75)]])
    a=steering(directions,geometry,0.)
    np.testing.assert_allclose(a[:,0]/a[0,0],a[:,1]/a[0,1],atol=1e-14)


def test_array_acquisition_unknown_bits_and_channel_phase():
    from code_replica import bandlimited_code_replica,_sample_code
    from gnss import ca_code
    from presets import default_config
    acquire=load(ROOT/'iteration032/array_acquisition.py').acquire_array
    fs=2046000.;t=np.arange(round(fs*.1))/fs
    local=_sample_code(bandlimited_code_replica(ca_code(1),1.023e6,.9e6),t*1.023e6-23.)
    bits=np.where((np.floor(t/.02).astype(int)%2)==0,1.,-1.)
    x=local*bits*np.exp(2j*np.pi*537.*t)
    cfg=default_config().acquisition;cfg.doppler_bins_hz=np.arange(0,1001,250.)
    gain=np.exp(1j*np.array([.1,1.,-2.,.7]))
    first=acquire(np.tile(x,(4,1)),fs,1,1.023e6,cfg)
    second=acquire(gain[:,None]*x,fs,1,1.023e6,cfg)
    assert first['success'] and abs(first['doppler_hz']-537.)<.1
    np.testing.assert_allclose(first['search_power'],second['search_power'],rtol=1e-10,atol=1e-4)


def test_gain_coherence_does_not_validate_wrong_delay_component():
    from audit_lag_support import generate
    from lag_source_support import estimate
    z,a,lag,g,true,fake=generate(31003,'two')
    _,info=estimate(z,a,lag)
    accepted=np.array(info['supported_prns'])
    assert not accepted[0] and accepted.sum()==7
    code_error=(np.array(info['selected_delay_chips'])-true)*299792458/1.023e6
    assert abs(code_error[0])>200. and np.max(abs(code_error[accepted]))<10.


def test_noise_only_cannot_pass_on_relative_sky_coherence_alone():
    from audit_lag_support import generate
    from lag_source_support import estimate
    z,a,lag,*_=generate(31002,'noise')
    _,info=estimate(z,a,lag)
    assert info['global_usable']
    assert not info['usable'] and not any(info['signal_supported_prns'])
