"""A continuously present, phase-locked 30 dB-Hz signal must not look lost."""
import numpy as np
from core.spoofing.single_source import coherent_cn0


def test_continuously_locked_signal_survives_quality_estimation():
    rng=np.random.default_rng(116006)
    z=(1+(rng.normal(size=(20,2000))+1j*rng.normal(size=(20,2000)))/np.sqrt(2))[0]
    cn=coherent_cn0(z,100);good=np.isfinite(cn)&(cn>=25)
    assert not np.any(np.convolve((~good[100:]).astype(int),np.ones(5,int),'valid')==5)
    assert 29<np.nanmedian(cn[100:])<31


def test_noise_alone_does_not_have_navigation_quality():
    rng=np.random.default_rng(116007);z=(rng.normal(size=2000)+1j*rng.normal(size=2000))/np.sqrt(2)
    assert np.mean(coherent_cn0(z,100)[100:]>=25)<.05


def test_fitted_initial_phase_has_no_single_sample_noise_estimate():
    with np.errstate(divide='raise',invalid='raise'):
        cn=coherent_cn0(np.array([2+0j,1+.5j,1-.5j]),100)
    assert np.isnan(cn[0])
    assert np.isfinite(cn[1:]).all()
