"""Time-differenced spatial phase, with an estimated-rotation excitation gate.

Fixed channel x PRN phase offsets cancel. Time-varying offsets do not. Static
genuine and common-source scenes can both have zero phase change; no positive
evidence is asserted without five degrees of estimated rotation over the lag.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from spatial_spoofing import spatial_scores

LAGS = (.5, 1., 2., 5.)
NAMES = tuple(f'{method}_{lag:g}s' for lag in LAGS for method in ('global', 'cluster3'))


def temporal_scores(carrier, estimated_rotation, wavelength):
    n = len(carrier)//10; values = np.full((8, n), -1.)
    excitation = np.zeros((4, n), bool)
    for i, lag in enumerate(LAGS):
        offset = round(lag*100); first = offset//10
        differenced = carrier[offset:]-carrier[:-offset]
        scores = spatial_scores(differenced.reshape(n-first, 10, carrier.shape[1], carrier.shape[2]), wavelength)
        end = np.arange(offset+9, len(carrier), 10)
        relative = np.einsum('tij,tkj->tik', estimated_rotation[end], estimated_rotation[end-offset])
        active = Rotation.from_matrix(relative).magnitude() >= np.deg2rad(5.)
        excitation[i, first:] = active
        for j, method in enumerate(('global', 'cluster3')):
            # Larger is evidence, compatible with the causal confirmation helper.
            values[2*i+j, first:] = np.where(active, -scores[method], -1.)
    return values, excitation
