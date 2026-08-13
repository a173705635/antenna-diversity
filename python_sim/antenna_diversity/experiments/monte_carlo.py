"""Sequential, repeatable Monte Carlo orchestration."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from antenna_diversity.config import SimulationConfig
from antenna_diversity.models import SimulationResults
from antenna_diversity.pipeline import run_end_to_end


_DISTRIBUTION_FIELDS = (
    "acquisition_metric",
    "final_doppler_error_hz",
    "output_sinr_db",
    "worst_interferer_gain",
)


@dataclass(frozen=True)
class MonteCarloRun:
    """Small per-seed records; sample arrays are intentionally discarded."""

    seed: int
    branches: dict[str, dict[str, bool | float | None]]

    @classmethod
    def from_results(cls, seed: int, results: SimulationResults) -> "MonteCarloRun":
        by_mode = {record["mode"]: record for record in results.metrics}
        branches: dict[str, dict[str, bool | float | None]] = {}
        for mode in results.branches:
            record = by_mode[mode]
            final_doppler = _optional_finite(record.get("final_doppler_hz"))
            branches[mode] = {
                "captured": bool(record["acquired"]),
                "tracking": bool(record["tracking"]),
                "acquisition_metric": _required_finite(
                    record["acquisition_metric"]
                ),
                "final_doppler_error_hz": (
                    None
                    if final_doppler is None
                    else abs(final_doppler - float(results.truth["doppler_hz"]))
                ),
                # The sampled end-to-end metric schema does not yet estimate
                # calibrated signal/interference/noise output powers.
                "output_sinr_db": _optional_finite(record.get("output_sinr_db")),
                "worst_interferer_gain": _optional_finite(
                    record.get("worst_interferer_gain")
                ),
            }
        return cls(int(seed), branches)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "branches": {
                mode: dict(metrics) for mode, metrics in self.branches.items()
            },
        }


@dataclass(frozen=True)
class MonteCarloResult:
    """Ordered runs, Boolean rates, and branch-wise scalar distributions."""

    runs: tuple[MonteCarloRun, ...]
    capture_rate: dict[str, float]
    tracking_rate: dict[str, float]
    distributions: dict[str, dict[str, list[float | None]]]

    @classmethod
    def from_runs(cls, runs: tuple[MonteCarloRun, ...]) -> "MonteCarloResult":
        if not runs:
            raise ValueError("runs must be nonempty")
        modes = tuple(runs[0].branches)
        if any(tuple(run.branches) != modes for run in runs):
            raise ValueError("all runs must contain the same ordered branches")
        count = float(len(runs))
        capture_rate = {
            mode: sum(run.branches[mode]["captured"] is True for run in runs) / count
            for mode in modes
        }
        tracking_rate = {
            mode: sum(run.branches[mode]["tracking"] is True for run in runs) / count
            for mode in modes
        }
        distributions = {
            field: {
                mode: [run.branches[mode][field] for run in runs]
                for mode in modes
            }
            for field in _DISTRIBUTION_FIELDS
        }
        return cls(runs, capture_rate, tracking_rate, distributions)

    def to_dict(self) -> dict[str, Any]:
        """Return a strict-JSON-compatible snapshot in caller seed order."""

        return {
            "runs": [run.to_dict() for run in self.runs],
            "capture_rate": dict(self.capture_rate),
            "tracking_rate": dict(self.tracking_rate),
            "distributions": {
                field: {mode: list(values) for mode, values in per_mode.items()}
                for field, per_mode in self.distributions.items()
            },
        }


def run_monte_carlo(
    cfg: SimulationConfig,
    seeds: Sequence[int],
) -> MonteCarloResult:
    """Run the end-to-end chain sequentially for each seed in supplied order."""

    normalized_seeds = tuple(int(seed) for seed in seeds)
    if not normalized_seeds:
        raise ValueError("seeds must be nonempty")
    runs = []
    for seed in normalized_seeds:
        run_cfg = copy.deepcopy(cfg)
        run_cfg.random_seed = seed
        run_cfg.plot.enable = False
        results = run_end_to_end(run_cfg)
        runs.append(MonteCarloRun.from_results(seed, results))
    return MonteCarloResult.from_runs(tuple(runs))


def _required_finite(value: object) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Monte Carlo metrics must be finite or None")
    return result


def _optional_finite(value: object) -> float | None:
    if value is None:
        return None
    return _required_finite(value)
