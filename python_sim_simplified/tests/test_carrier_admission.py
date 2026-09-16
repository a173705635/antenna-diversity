import numpy as np
from carrier_admission import Rule,smooth,spatial_evidence


def test_archived_disabled_thresholds_round_trip(tmp_path):
    import json
    from dataclasses import asdict
    from carrier_admission import rule_from_record
    from audit_recovery_fifty import save
    original=Rule(window=5,divergence=np.inf,spatial=-np.inf,consensus=np.inf)
    path=tmp_path/'rule.json';save(path,asdict(original))
    recovered=rule_from_record(json.loads(path.read_text()))
    assert recovered==original


def test_hatch_preserves_motion_and_constant_ambiguity_is_irrelevant():
    t=np.arange(100)*.01;truth=1000+7*t[:,None]+np.zeros((100,6))
    noise=np.random.default_rng(2).normal(0,5,truth.shape)
    carrier=truth+np.arange(6)*234.
    mask=np.ones_like(truth,bool);rule=Rule(window=20,mature=1)
    p,_=smooth(t,truth+noise,carrier,mask,rule)
    other,_=smooth(t,truth+noise,carrier+173254.,mask,rule)
    np.testing.assert_allclose(p,other,atol=1e-8)
    assert np.std(p[30:]-truth[30:])<np.std(noise[30:])/2
    prefix,_=smooth(t[:41],(truth+noise)[:41],carrier[:41],mask[:41],rule)
    np.testing.assert_array_equal(prefix,p[:41])


def test_loss_and_observation_gap_break_smoothing_arc():
    t=np.arange(20)*.01;r=np.zeros((20,6));c=r.copy();mask=np.ones_like(r,bool)
    mask[5]=False;r[6:]=80;c[6:]=80
    p,age=smooth(t,r,c,mask,Rule(mature=3))
    assert np.isnan(p[5:8]).all() and (age[6]==1).all()
    np.testing.assert_allclose(p[8:],80.,atol=1e-10)
    t[10:]+=.5
    p,age=smooth(t,r,c,np.ones_like(mask),Rule(mature=3))
    assert np.isnan(p[10:12]).all() and (age[10]==1).all()


def test_coherent_false_code_and_carrier_pass_smoothing_negative_control():
    t=np.arange(200)*.01;r=200+10*t[:,None]+np.zeros((200,6))
    p,_=smooth(t,r,r+54321,np.ones_like(r,bool),Rule(mature=1))
    np.testing.assert_allclose(p,r,atol=1e-8)


def test_smoothing_cannot_authenticate_a_consistent_200m_false_position():
    from run_gps_spp import receiver_reference
    from signal_recovery import sky
    from continuous_admission import Gate,admit
    ref,rot=receiver_reference();sat=ref+sky(123,8)*21e6
    false=ref+np.array([200.,0,0])@rot
    rho=np.linalg.norm(sat-false,axis=1)+75000
    ranges=np.tile(rho,(20,1));quality=np.ones_like(ranges)
    filtered,_=smooth(np.arange(20)*.01,ranges,ranges+321.,quality.astype(bool),Rule(mature=1))
    position=admit(np.tile(sat,(20,1,1)),filtered,quality,quality*45,quality,Gate(horizontal_sigma=100.))
    np.testing.assert_allclose(position['position'][:,0],200.,atol=1e-5)


def test_spatial_evidence_uses_past_samples_and_rejects_source_only():
    rng=np.random.default_rng(45);fs=2046000.;n=round(.2*fs)
    a=np.array([1,1j,-1,-1j]);v=np.ones(4,complex)/2
    noise=(rng.normal(size=(4,n))+1j*rng.normal(size=(4,n)))*8
    fake=20*v[:,None]*np.exp(2j*np.pi*4000*np.arange(n)/fs);clean=a[:,None]*np.ones((1,n))
    h=dict(epoch_start_samples=np.arange(200)*2046,code_nco_phase_chips=np.zeros(200),
           code_nco_rate_hz=np.ones(200)*1.023e6,carrier_nco_phase_rad=np.zeros(200),
           carrier_nco_frequency_hz=np.zeros(200))
    # Constant replica is sufficient to isolate the spatial statistic in this test.
    code=np.ones(1023);times=np.array([.15,.16])
    yes=spatial_evidence(fake+clean+noise,code,h,times,a,fs)
    no=spatial_evidence(fake+noise,code,h,times,a,fs)
    assert (yes>10).all() and (no<2).all()
    changed=fake+clean+noise;changed[:,round(.16*fs):]*=100
    np.testing.assert_array_equal(yes,spatial_evidence(changed,code,h,times,a,fs))
    coherent=20*v[:,None]*np.ones((1,n))+clean+noise
    # A perfectly coherent mixture is one covariance direction: projection
    # removes genuine and counterfeit together, despite genuine RF being present.
    assert (spatial_evidence(coherent,code,h,times,a,fs)<2).all()
