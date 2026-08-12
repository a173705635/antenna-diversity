"""Friis gain and noise-figure calculations."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from antenna_diversity.config import RFStage


def cascade_metrics(stages: Sequence[RFStage]) -> dict[str, Any]:
    """Compute total power gain and input-referred cascaded noise factor."""

    if not stages:
        raise ValueError("stages must be a nonempty sequence")
    gain_db = np.asarray([stage.gain_db for stage in stages], dtype=np.float64)
    noise_figure_db = np.asarray(
        [stage.noise_figure_db for stage in stages], dtype=np.float64
    )
    if (
        not np.all(np.isfinite(gain_db))
        or not np.all(np.isfinite(noise_figure_db))
        or np.any(noise_figure_db < 0)
    ):
        raise ValueError("stage gains must be finite and noise figures nonnegative")

    gain_linear = 10.0 ** (gain_db / 10.0)
    noise_factor = 10.0 ** (noise_figure_db / 10.0)
    cascade_noise_factor = float(noise_factor[0])
    preceding_gain = float(gain_linear[0])
    for index in range(1, len(stages)):
        cascade_noise_factor += float(noise_factor[index] - 1.0) / preceding_gain
        preceding_gain *= float(gain_linear[index])

    total_gain_linear = float(np.prod(gain_linear))
    return {
        "total_gain_linear": total_gain_linear,
        "total_gain_db": float(10.0 * np.log10(total_gain_linear)),
        "noise_factor": cascade_noise_factor,
        "noise_figure_db": float(10.0 * np.log10(cascade_noise_factor)),
        "stage_gain_db": gain_db,
        "stage_noise_figure_db": noise_figure_db,
    }
