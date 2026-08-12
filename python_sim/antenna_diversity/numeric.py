"""Small numerical helpers used to preserve MATLAB conventions."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def matlab_round(values: ArrayLike) -> NDArray[np.float64]:
    """Round half ties away from zero, matching MATLAB ``round``."""

    array = np.asarray(values, dtype=np.float64)
    return np.sign(array) * np.floor(np.abs(array) + 0.5)
