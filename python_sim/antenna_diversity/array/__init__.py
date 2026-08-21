"""Array-geometry utilities."""

from .geometry import (
    active_corner_combinations,
    mate70_phase_centers,
    maximum_baseline_m,
    select_phase_centers,
)
from .manifold import (
    ElementPattern,
    IsotropicElementPattern,
    array_manifold,
    direction_unit_vector,
)
from .polarization import (
    LocalWave,
    WavePolarization,
    rotation_matrix_ypr,
    wave_in_element_frame,
)
from .patterns import (
    LocalDirection,
    ParametricElementPattern,
    ParametricPatternParameters,
    PolarizedElementPattern,
    ScalarPatternAdapter,
    TabulatedElementPattern,
)
from .phone_model import PhoneAntennaModel
from .steering import steering_vector

__all__ = [
    "active_corner_combinations",
    "array_manifold",
    "direction_unit_vector",
    "ElementPattern",
    "IsotropicElementPattern",
    "LocalDirection",
    "LocalWave",
    "mate70_phase_centers",
    "maximum_baseline_m",
    "ParametricElementPattern",
    "ParametricPatternParameters",
    "PolarizedElementPattern",
    "PhoneAntennaModel",
    "rotation_matrix_ypr",
    "select_phase_centers",
    "ScalarPatternAdapter",
    "steering_vector",
    "WavePolarization",
    "wave_in_element_frame",
    "TabulatedElementPattern",
]
