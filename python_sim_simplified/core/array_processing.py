"""Measured prompt projection for one common emitter."""
import numpy as np


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
