import numpy as np
from continuous_improvement import Case,generate,cancel_replicas,local_signal,short_projector,startup_phase
from continuous_admission import Gate,admit
from code_replica import bandlimited_code_replica
from gnss import ca_code
from signal_recovery import sky,steering
from presets import mate70_phase_centers
from run_gps_spp import receiver_reference


def test_spatial_replica_cancellation_preserves_authentic_and_handles_bit_signs():
    n=2046*4;t=np.arange(n)/2.046e6
    code=bandlimited_code_replica(ca_code(1),1.023e6,.9e6)
    temp=(0.,100.,1.023e6,500.)
    a=np.array([1,1j,-1,-1j],complex);v=np.ones(4,complex)/2
    genuine=local_signal(code,temp,t);fake=local_signal(code,temp,t,.25)
    fake*=np.repeat([3.,-3.,3.,-3.],2046)*np.exp(.7j)
    x=a[:,None]*genuine+v[:,None]*fake
    y,used=cancel_replicas(x,t,[code],[temp],[0],a[:,None],v,.5,.05)
    assert used[0]
    np.testing.assert_allclose(y,a[:,None]*genuine,atol=1e-10)
    y2,used2=cancel_replicas(x,t,[code],[temp],[0],(v*2)[:,None],v,.5,.05)
    assert not used2.any()
    np.testing.assert_array_equal(x,y2)


def test_short_prompt_direction_invariant_to_per_chunk_sign():
    rng=np.random.default_rng(12);a=np.exp(1j*rng.uniform(-3,3,(4,5)))
    local=np.exp(1j*rng.uniform(-3,3,(5,800)))
    x=(a+np.ones((4,1))*3)@local
    p,r=short_projector(x,local,a,100)
    signs=np.repeat([1,-1,-1,1,-1,1,1,-1],100)
    other,r2=short_projector(x*signs,local,a,100)
    np.testing.assert_allclose(p,other,atol=1e-12)
    np.testing.assert_allclose(r,r2,atol=1e-12)


def test_startup_relative_channel_phase_from_unknown_bits():
    n=2046*4;local=np.ones((1,n),complex)
    local*=np.repeat([1,-1,1,-1],2046)
    a=np.exp(1j*np.array([.2,.6,-.5,1.]))[:,None]
    gain=np.exp(1j*np.array([0.,.1,-.2,.3]))
    x=gain[:,None]*a@local
    np.testing.assert_allclose(startup_phase(x,local,a),gain,atol=1e-12)


def test_paired_iq_start_and_withdrawal_and_poisoned_start():
    from dataclasses import replace
    case=Case(duration=.4,attack_end=.3,genuine=False)
    clean=generate(case,77,False)[0];attacked=generate(case,77,True)[0]
    np.testing.assert_array_equal(clean[:,:round(.15*2.046e6)],attacked[:,:round(.15*2.046e6)])
    np.testing.assert_array_equal(clean[:,round(.3*2.046e6):],attacked[:,round(.3*2.046e6):])
    assert np.linalg.norm(clean-attacked)>0
    cold=generate(replace(case,attack_start=0.),77,True)[0]
    assert np.linalg.norm(clean[:,:2046]-cold[:,:2046])>0


def test_weighted_gate_cannot_authenticate_consistent_false_position_and_is_epoch_causal():
    ref,rot=receiver_reference();u=sky(123,8);sat=ref+u*21e6
    false=ref+np.array([200.,0,0])@rot
    rho=np.linalg.norm(sat-false,axis=1)+75000
    s=np.tile(sat,(3,1,1));r=np.tile(rho,(3,1));quality=np.ones_like(r)
    args=(s,r,quality,quality*45,quality,Gate(horizontal_sigma=100.))
    result=admit(*args)
    np.testing.assert_allclose(result['position'][:,0],200.,atol=1e-5)
    prefix=admit(*(x[:2] for x in args[:5]),args[5])
    np.testing.assert_array_equal(prefix['position'],result['position'][:2])
    poor=admit(s,r,quality*.1,quality*45,quality,args[5])
    assert np.isnan(poor['position']).all()
