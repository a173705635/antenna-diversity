"""Small branch-comparison table for the teaching simulation."""

import numpy as np


def _final_cn0(tracking):
    if not tracking["success"]:
        return None
    finite = tracking["cn0_db_hz"][np.isfinite(tracking["cn0_db_hz"])]
    return float(finite[-1]) if finite.size else None


def compute_metrics(results):
    records = []
    for mode, branch in results["branches"].items():
        acquisition = branch["acquisition"]
        tracking = branch["tracking"]
        tracked = bool(tracking["success"])
        if tracked:
            window = slice(max(0, tracking["prompt"].size - 10), None)
            final_doppler = float(tracking["carrier_frequency_hz"][-1])
            pll_rms = float(np.sqrt(np.mean(tracking["pll_error_rad"][window] ** 2)))
            dll_rms = float(np.sqrt(np.mean(tracking["dll_error"][window] ** 2)))
            mean_prompt = float(np.mean(np.abs(tracking["prompt"][window])))
        else:
            final_doppler = pll_rms = dll_rms = mean_prompt = None
        records.append(
            {
                "mode": mode,
                "acquired": bool(acquisition["success"]),
                "acquisition_metric": float(acquisition["metric"]),
                "acquired_doppler_hz": float(acquisition["doppler_hz"]),
                "target_direction_gain": branch["combining"]["target_direction_gain"],
                "worst_interferer_gain": branch["combining"]["worst_interferer_gain"],
                "tracking": tracked,
                "final_doppler_hz": final_doppler,
                "pll_rms_rad": pll_rms,
                "dll_rms": dll_rms,
                "mean_prompt_magnitude": mean_prompt,
                "final_cn0_db_hz": _final_cn0(tracking),
            }
        )
    return records


def format_metrics(records):
    header = (
        f"{'Mode':>10} {'Acq':>6} {'Metric':>9} {'Tracking':>9} "
        f"{'Doppler':>10} {'C/N0':>9}"
    )
    lines = [header, "-" * len(header)]
    for record in records:
        doppler = record["final_doppler_hz"]
        cn0 = record["final_cn0_db_hz"]
        lines.append(
            f"{record['mode']:>10} {str(record['acquired']):>6} "
            f"{record['acquisition_metric']:>9.3f} "
            f"{str(record['tracking']):>9} "
            f"{doppler if doppler is not None else np.nan:>10.2f} "
            f"{cn0 if cn0 is not None else np.nan:>9.2f}"
        )
    return "\n".join(lines)
