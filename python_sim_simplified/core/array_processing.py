"""Measured prompt projection for one common emitter."""
import numpy as np


def startup_relative_amplitude(prompts, nominal, prns):
    """Three-antenna RMS-one magnitude from phase-calibrated 1 ms prompts.

    Prompts have axes [antenna, acquired PRN, millisecond]. The equal-PRN
    covariance averages each PRN over the startup interval before averaging
    across PRNs with equal weights; PRN powers themselves are not normalized.
    """
    assert prompts.shape[0]==3
    aligned=prompts/nominal[:,prns,None]
    per_prn=np.einsum('mst,pst->mps',aligned,aligned.conj())/prompts.shape[2]
    covariance=np.mean(per_prn,axis=2)
    relative=np.array([1.,abs(covariance[1,2]/covariance[0,2]),
                       abs(covariance[2,1]/covariance[0,1])])
    return relative/np.sqrt(np.mean(relative**2)),covariance


def prompt_projector(prompts,desired,iterations):
    """Fit Z_s = a_s alpha_s + v beta_s; no emitter truth supplied."""
    m=len(prompts)
    eigen,vectors=np.linalg.eigh(prompts@prompts.conj().T)
    ratio=float(eigen[-1]/np.mean(eigen[:-1]));v=vectors[:,-1]
    for _ in range(iterations):
        projection=np.eye(m)-np.outer(v,v.conj())
        projected=projection@desired
        amplitude=np.sum(projected.conj()*prompts,axis=0)/(np.sum(desired.conj()*projected,axis=0).real+.01*m)
        remaining=prompts-desired*amplitude
        _,vectors=np.linalg.eigh(remaining@remaining.conj().T)
        v=vectors[:,-1]
    return np.eye(m)-np.outer(v,v.conj()),ratio
