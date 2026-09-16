"""Directional energy ratio in an observable residual spatial subspace."""
import numpy as np


def source_evidence(block,local,desired):
    """At least two residual dimensions and a nonzero desired vector required.

    On-direction energy is compared with mean orthogonal energy. White noise
    gives a ratio near one; threshold three is an uncalibrated research choice.
    """
    prompts=(block*local.conj()).reshape(len(block),-1,2046).mean(axis=2)
    covariance=prompts@prompts.conj().T
    on=np.vdot(desired,covariance@desired).real/np.vdot(desired,desired).real
    off=(np.trace(covariance).real-on)/(len(block)-1)
    return float(on/off)
