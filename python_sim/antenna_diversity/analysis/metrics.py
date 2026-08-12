"""Compact branch-comparison metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

from antenna_diversity.models import SimulationResults


def compute_metrics(
    results: SimulationResults,
    truth: dict[str, Any],
) -> list[dict[str, Any]]:
    """Summarize acquisition, spatial rejection, and steady-state tracking."""

    records: list[dict[str, Any]] = []
    jammer_steering = truth["jammer_steering_nominal"]
    for mode, branch in results.branches.items():
        acquisition = branch.acquisition
        tracking = branch.tracking
        acquired = bool(acquisition["success"])
        tracking_locked = bool(tracking.get("success", False))
        jammer_direction_gain: float | None = None
        if np.linalg.norm(jammer_steering) > 0:
            jammer_direction_gain = float(
                abs(np.vdot(branch.combining["weights"], jammer_steering))
            )

        final_doppler_hz: float | None = None
        pll_rms_rad: float | None = None
        dll_rms: float | None = None
        mean_prompt_magnitude: float | None = None
        final_cn0_db_hz: float | None = None
        if tracking_locked:
            prompt = tracking["prompt"]
            window = slice(max(0, prompt.size - 10), prompt.size)
            final_doppler_hz = float(tracking["carrier_frequency_hz"][-1])
            pll_rms_rad = float(
                np.sqrt(np.mean(tracking["pll_error_rad"][window] ** 2))
            )
            dll_rms = float(
                np.sqrt(np.mean(tracking["dll_error"][window] ** 2))
            )
            mean_prompt_magnitude = float(np.mean(np.abs(prompt[window])))
            finite_cn0 = tracking["cn0_db_hz"][
                np.isfinite(tracking["cn0_db_hz"])
            ]
            if finite_cn0.size:
                final_cn0_db_hz = float(finite_cn0[-1])

        records.append(
            {
                "mode": mode,
                "acquired": acquired,
                "acquisition_metric": float(acquisition["metric"]),
                "acquired_doppler_hz": float(acquisition["doppler_hz"]),
                "jammer_direction_gain": jammer_direction_gain,
                "tracking": tracking_locked,
                "final_doppler_hz": final_doppler_hz,
                "pll_rms_rad": pll_rms_rad,
                "dll_rms": dll_rms,
                "mean_prompt_magnitude": mean_prompt_magnitude,
                "final_cn0_db_hz": final_cn0_db_hz,
            }
        )
    return records


def _format_value(value: Any, width: int) -> str:
    if value is None:
        text = "nan"
    elif isinstance(value, (bool, np.bool_)):
        text = "true" if value else "false"
    elif isinstance(value, (float, np.floating)):
        text = f"{float(value):.5g}"
    else:
        text = str(value)
    return text.rjust(width)


def format_metrics(metrics: list[dict[str, Any]]) -> str:
    """Format the small metric record set without a dataframe dependency."""

    columns = (
        ("Mode", "mode", 8),
        ("Acquired", "acquired", 9),
        ("AcqMetric", "acquisition_metric", 11),
        ("AcqDoppler", "acquired_doppler_hz", 12),
        ("JammerGain", "jammer_direction_gain", 12),
        ("Tracking", "tracking", 9),
        ("FinalDoppler", "final_doppler_hz", 13),
        ("PllRms", "pll_rms_rad", 9),
        ("DllRms", "dll_rms", 9),
        ("MeanPrompt", "mean_prompt_magnitude", 12),
        ("FinalCn0", "final_cn0_db_hz", 10),
    )
    header = " ".join(title.rjust(width) for title, _, width in columns)
    separator = " ".join("-" * width for _, _, width in columns)
    rows = [header, separator]
    for record in metrics:
        rows.append(
            " ".join(
                _format_value(record[key], width) for _, key, width in columns
            )
        )
    return "\n".join(rows)
