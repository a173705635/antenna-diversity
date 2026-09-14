"""Direction from double differences with an EXTERNALLY trusted reference PRN."""
import numpy as np
from spatial_direction import estimate_direction


def estimate_with_reference(carrier_m, positions_m, wavelength_m, *, reference_index,
                            target_indices, reference_direction, harmonic=2):
    x, p, u = map(lambda v: np.asarray(v, float), (carrier_m, positions_m, reference_direction))
    targets = list(target_indices)
    if reference_index in targets:
        return {'status': 'reference_in_target', 'candidates': []}
    # Satellite difference first, followed by antenna differences inside the fit.
    # d_s - d_ref = -b.(u_s-u_ref); adding -b.u_ref restores -b.u_s.
    reconstructed = x[:, :, targets]-x[:, :, reference_index:reference_index+1]-(p@u)[None, :, None]
    result = estimate_direction(reconstructed, p, wavelength_m, channel_bias_m=np.zeros(len(p)), harmonic=harmonic)
    return {**result, 'reference_index': reference_index,
            'reference_assumed_trusted': True, 'reference_authentication_performed': False}
