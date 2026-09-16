import numpy as np
from warm_navigation import estimate
from warm_navigation_cases import generate


def test_initialization_is_estimated_and_precedes_attack():
    z,truth,attack=generate(111001,"normal")
    r=estimate(z)
    assert r["initialized_epoch"]==29 and r["initialized_epoch"]<attack
    assert not np.isfinite(r["position"][:29]).any()
    shifted={key:value.copy() for key,value in z.items()}
    shifted["code"]+=z["design"]@np.array([7.,-4.,0.,0.])
    moved=estimate(shifted)
    assert np.allclose(moved["position"][29:]-r["position"][29:],[7.,-4.,0.],atol=1e-8)


def test_truth_is_scoring_only():
    z,truth,attack=generate(111001,"normal")
    first=estimate(z)["position"]
    truth[:]=1e9
    assert np.array_equal(first,estimate(z)["position"],equal_nan=True)
    assert "truth" not in z


def test_joint_clock_change_does_not_look_like_horizontal_attack():
    a=estimate(generate(111001,"normal")[0])["position"]
    b=estimate(generate(111001,"clock_drag")[0])["position"]
    assert np.allclose(a,b,equal_nan=True)


def test_dropout_has_no_synthetic_output_and_reacquired_attack_remains_untrusted():
    z,truth,attack=generate(111001,"dropout")
    r=estimate(z)["position"]
    assert np.isnan(r[80:100]).all()
    assert np.isfinite(r[100:]).all()
    z,_,_=generate(111001,"reacquire_attack")
    assert np.isnan(estimate(z)["position"][100:]).all()


def test_initialization_failure_is_explicit():
    z,_,_=generate(111001,"normal");z["lock"][:]=False
    r=estimate(z)
    assert r["initialized_epoch"] is None and np.isnan(r["position"]).all()


def test_common_step_protects_but_slow_pull_is_counterexample():
    z,_,attack=generate(111001,"joint_step")
    assert np.isnan(estimate(z)["position"][attack:]).all()
    z,truth,attack=generate(111001,"slow_drag")
    e=np.linalg.norm(estimate(z)["position"][attack:,:2]-truth[attack:,:2],axis=1)
    assert np.nanmax(e)>20
