"""Measured prompt evidence for a genuine sky plus one common emitter."""
import numpy as np
from doppler_recovery import prompt_projector


def residual_trigger(prompts,desired,noise=1.):
    """Score residual reduction per added complex parameter and noise power.

    Noise is measured raw sample power / coherent sample count, so unresolved
    signals inflate this estimate. Threshold 3 is a research setting, not a Pfa.
    """
    fitted=desired*np.sum(desired.conj()*prompts,axis=0)/len(desired)
    projector,_=prompt_projector(prompts,desired,2)
    projected=projector@desired
    amplitude=np.sum(projected.conj()*prompts,axis=0)/(np.sum(desired.conj()*projected,axis=0).real+.01*len(desired))
    residual=projector@(prompts-desired*amplitude)
    reduction=np.sum(abs(prompts-fitted)**2)-np.sum(abs(residual)**2)
    return projector,float(reduction/(noise*(prompts.shape[1]+len(desired)-1)))
