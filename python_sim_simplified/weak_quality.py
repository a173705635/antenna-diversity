"""Bit-insensitive usable coherent power for an already locked Costas loop."""
import numpy as np


def coherent_cn0(prompt,width):
    """Estimate (E[I²]-E[Q²])/(2 E[Q²] T); phase alignment is required.

    Nonpositive coherent power is an unobservable signal estimate, represented
    as NaN. This is not an acquisition metric or an independent phase test.
    """
    count=np.minimum(np.arange(len(prompt))+1,width)
    i=np.convolve(prompt.real**2,np.ones(width),'full')[:len(prompt)]/count
    q=np.convolve(prompt.imag**2,np.ones(width),'full')[:len(prompt)]/count
    # The first prompt's phase was fitted; it has no noise degree of freedom.
    measured=(count>=2)&(i>q)
    result=np.full(len(prompt),np.nan)
    result[measured]=10*np.log10((i[measured]-q[measured])/(2*q[measured]*.001))
    return result
