import numpy as np
from dataclasses import replace
from recovery_admission import Policy,fit_ranges,admit
from signal_recovery import sky
from run_gps_spp import receiver_reference


def geometry():
    reference,rotation=receiver_reference()
    satellites=reference+sky(990001,12)*21e6
    ranges=np.linalg.norm(satellites-reference,axis=1)+75000
    return satellites,ranges


def test_isolated_large_code_error_excluded_not_averaged():
    satellites,ranges=geometry();ranges[5]+=22000
    p,mask,rms=fit_ranges(satellites,ranges,np.ones(12,bool),Policy())
    assert not mask[5]
    assert np.linalg.norm(p)<.001


def test_consistent_forged_position_has_small_residual():
    satellites,_=geometry();reference,rotation=receiver_reference()
    false=reference+np.array([200.,0,0])@rotation
    ranges=np.linalg.norm(satellites-false,axis=1)+75000
    p,_,_=fit_ranges(satellites,ranges,np.ones(12,bool),Policy())
    assert np.linalg.norm(p-[200,0,0])<.001
    from solution_separation import separation_scores
    score=separation_scores(satellites[None],ranges[None],np.ones((1,12),bool),p[None])
    assert score[0]<.001  # A consistent false solution also passes separation.


def test_prefix_and_no_implicit_coldstart_authentication():
    satellites,ranges=geometry();t=np.arange(8.)
    states=np.tile(satellites,(8,1,1));r=np.tile(ranges,(8,1));valid=np.ones(r.shape,bool)
    policy=Policy();a=admit(t,states,r,valid,policy)
    b=admit(t[:5],states[:5],r[:5],valid[:5],policy)
    np.testing.assert_array_equal(a['position'][:5],b['position'])
    assert np.isnan(a['position'][0]).all()
    assert np.isfinite(a['position'][1:]).all()
    assert np.isnan(admit(t,states,r,valid,replace(policy,admitted_start=False))['position']).all()


def test_four_exact_ranges_do_not_provide_separation_redundancy():
    from solution_separation import separation_scores
    satellites,ranges=geometry();valid=np.zeros(12,bool);valid[:4]=True
    p,_,_=fit_ranges(satellites,ranges,valid,Policy(minimum_stars=4,exclusions=0,geometry_limit=1000))
    assert np.isfinite(p).all()
    assert np.isinf(separation_scores(satellites[None],ranges[None],valid[None],p[None])[0])
