"""A genuine anisotropic sky must not trigger a spatial null."""
import numpy as np
from core.single_source import residual_trigger
from core.recovery_geometry import sky,steering
from core.presets import mate70_phase_centers


def test_genuine_anisotropic_sky_does_not_trigger():
    a=steering(sky(114001,8),np.array(list(mate70_phase_centers().values()))[:3],0.)
    p,score=residual_trigger(a,a)
    assert score<3., 'a genuine-only sky falsely activates suppression'


def test_noise_does_not_look_like_a_common_emitter():
    rng=np.random.default_rng(913)
    a=np.repeat(steering(sky(114001,8),np.array(list(mate70_phase_centers().values()))[:3],0.),20,axis=1)
    noise=(rng.normal(size=a.shape)+1j*rng.normal(size=a.shape))/np.sqrt(2)
    _,score=residual_trigger(.1*a+noise,a,1.)
    assert score<3., 'noise-only residual must not activate a null'


def test_single_emitter_is_suppressed():
    rng=np.random.default_rng(914)
    antennas=np.array(list(mate70_phase_centers().values()))[:3]
    a=np.repeat(steering(sky(114001,8),antennas,0.),20,axis=1)
    v=steering(np.array([[0.,1.,0.]]),antennas,0.)[:,0]
    fake=3*(rng.normal(size=a.shape[1])+1j*rng.normal(size=a.shape[1]))
    z=a+v[:,None]*fake+.03*(rng.normal(size=a.shape)+1j*rng.normal(size=a.shape))
    p,score=residual_trigger(z,a,.0018)
    assert score>=3.
    assert np.linalg.norm(p@v)/np.linalg.norm(v)<.1
