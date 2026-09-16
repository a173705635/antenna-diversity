"""Causality, phase nuisance invariance and retained integrity counterexamples."""
from pathlib import Path
import numpy as np
import navigation_next as model
from navigation_next_cases import generate
from research_hundred import load

def test_prefix_causality():
    pair,_=generate(85201,'normal')
    prefix={name:{k:(v[:61] if v.ndim and len(v)==120 else v) for k,v in z.items()} for name,z in pair.items()}
    np.testing.assert_allclose(model.candidate(prefix),model.candidate(pair)[:61],atol=1e-8,equal_nan=True)

def test_shared_geometric_code_bias_is_still_unobservable():
    pair,truth=generate(85203,'fixed_geometry_bias')
    position=model.candidate(pair)
    valid=np.isfinite(position).all(axis=1)
    assert valid.sum()>100
    assert np.min(np.linalg.norm((position-truth)[valid,:2],axis=1))>150

def test_carrier_common_clock_increment_cancels():
    pair,_=generate(85207,'normal');z=pair['joint']
    before,_=model.carrier_steps(z)
    shifted={k:v.copy() for k,v in z.items()}
    shifted['calibrated_carrier']+=np.arange(120)[:,None]*3.1
    after,_=model.carrier_steps(shifted)
    np.testing.assert_allclose(before,after,atol=1e-7,equal_nan=True)

def test_constant_per_star_ambiguity_cancels():
    pair,_=generate(85211,'normal');z=pair['joint']
    before,_=model.carrier_steps(z)
    z['calibrated_carrier']+=np.arange(8)[None,:]*124.31
    after,_=model.carrier_steps(z)
    np.testing.assert_allclose(before,after,atol=1e-7,equal_nan=True)

def test_lock_loss_terminates_star_endpoint_arc():
    pair,_=generate(84501,'normal');z=pair['joint']
    z['calibrated_lock'][10,0]=0.
    z['calibrated_carrier'][10:,0]+=200.
    steps,_=model.carrier_steps(z)
    assert np.linalg.norm(steps[11])<.1

def test_huber_update_limits_single_code_impulse():
    pair,truth=generate(85213,'code_impulse')
    parent=load(Path('results/research-next-hundred/iteration194/navigation_next.py'))
    old=parent.calibrated(pair['joint']);new=model.calibrated(pair['joint'])
    assert np.linalg.norm(new[30]-truth[30])<np.linalg.norm(old[30]-truth[30])

def test_route_reagreement_does_not_clear_unresolved_conflict(monkeypatch):
    first=np.zeros((6,3));first[2,0]=60.;first[3:,0]=200.
    second=np.zeros((6,3));second[3:,0]=200.
    monkeypatch.setattr(model,'calibrated',lambda z:first.copy())
    monkeypatch.setattr(model,'navigate',lambda z:second.copy())
    out=model.candidate({'joint':{'joint_usable':True},'capture':{}})
    assert np.isfinite(out[:2]).all()
    assert np.isnan(out[2:]).all()
