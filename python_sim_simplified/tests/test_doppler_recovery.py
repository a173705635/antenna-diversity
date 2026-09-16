import numpy as np
from doppler_recovery import prompt_projector
from presets import mate70_phase_centers
from signal_recovery import sky,steering


def test_prompt_model_refinement_and_constant_bit_sign_invariance():
    rng=np.random.default_rng(93001)
    antennas=np.array(list(mate70_phase_centers().values()))
    a=steering(sky(93001,8),antennas,0.)
    v=steering(np.array([[1.,0.,0.]]),antennas,0.)[:,0]
    alpha=np.exp(1j*rng.uniform(-np.pi,np.pi,8))
    beta=3*np.exp(1j*rng.uniform(-np.pi,np.pi,8))
    z=a*alpha+v[:,None]*beta
    initial,_=prompt_projector(z,a,0)
    refined,_=prompt_projector(z,a,2)
    assert np.linalg.norm(refined@v)<np.linalg.norm(initial@v)
    changed,_=prompt_projector(z*np.array([1,-1,1,-1,-1,1,1,-1]),a,2)
    np.testing.assert_allclose(refined,changed,atol=1e-12)
