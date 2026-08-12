"""GPS L1 C/A Gold-code generator."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


_G2_TAPS = np.array(
    [
        [2, 6], [3, 7], [4, 8], [5, 9], [1, 9], [2, 10], [1, 8], [2, 9],
        [3, 10], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10],
        [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9], [1, 3], [4, 6],
        [5, 7], [6, 8], [7, 9], [8, 10], [1, 6], [2, 7], [3, 8], [4, 9],
    ],
    dtype=np.int64,
)


def ca_code(prn: int) -> NDArray[np.float64]:
    """Generate one 1023-chip GPS L1 C/A sequence with values ±1."""

    if not isinstance(prn, (int, np.integer)) or not 1 <= int(prn) <= 32:
        raise ValueError("PRN must be an integer from 1 through 32")

    tap1, tap2 = _G2_TAPS[int(prn) - 1] - 1
    g1 = np.ones(10, dtype=np.uint8)
    g2 = np.ones(10, dtype=np.uint8)
    code = np.empty(1023, dtype=np.float64)

    for chip in range(1023):
        output_bit = g1[9] ^ g2[tap1] ^ g2[tap2]
        code[chip] = 1.0 - 2.0 * float(output_bit)

        g1_feedback = g1[2] ^ g1[9]
        g2_feedback = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        g1[1:] = g1[:-1]
        g2[1:] = g2[:-1]
        g1[0] = g1_feedback
        g2[0] = g2_feedback

    return code
