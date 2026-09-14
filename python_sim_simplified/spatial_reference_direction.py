"""Direction from double differences with an EXTERNALLY trusted reference PRN."""
import numpy as np
from spatial_direction import estimate_direction


def estimate_with_reference(carrier_m, positions_m, wavelength_m, *, reference_index,
                            target_indices, reference_direction, harmonic=2):
    x, p, u = map(lambda v: np.asarray(v, float), (carrier_m, positions_m, reference_direction))
    targets = list(target_indices)
    if x.ndim != 3 or p.shape != (x.shape[1], 3) or u.shape != (3,):
        raise ValueError('invalid reference geometry')
    if not np.isfinite(u).all() or not np.isclose(np.linalg.norm(u), 1.) or not targets:
        raise ValueError('unit reference direction and nonempty targets required')
    if not 0 <= reference_index < x.shape[2] or any(not 0 <= i < x.shape[2] for i in targets):
        raise ValueError('PRN index outside observations')
    if reference_index in targets:
        return {'status': 'reference_in_target', 'candidates': []}
    # Satellite difference first, followed by antenna differences inside the fit.
    # d_s - d_ref = -b.(u_s-u_ref); adding -b.u_ref restores -b.u_s.
    reconstructed = x[:, :, targets]-x[:, :, reference_index:reference_index+1]-(p@u)[None, :, None]
    result = estimate_direction(reconstructed, p, wavelength_m, channel_bias_m=np.zeros(len(p)), harmonic=harmonic)
    return {**result, 'reference_index': reference_index,
            'reference_assumed_trusted': True, 'reference_authentication_performed': False}
