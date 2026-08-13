"""Run the legacy dual-antenna ULA GPS L1 C/A simulation."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt

from antenna_diversity.analysis.metrics import format_metrics
from antenna_diversity.analysis.plotting import plot_results, save_figures
from antenna_diversity.config import legacy_ula_config
from antenna_diversity.pipeline import run_end_to_end as run_pipeline


def _backend_supports_gui() -> bool:
    backend = str(matplotlib.get_backend()).lower()
    noninteractive_backends = {
        "agg",
        "cairo",
        "pdf",
        "pgf",
        "ps",
        "svg",
        "template",
    }
    return backend not in noninteractive_backends and "inline" not in backend


def main():
    cfg = legacy_ula_config()
    cfg.plot.output_dir = cfg.plot.output_dir / "legacy_ula"
    results = run_pipeline(cfg)
    print(format_metrics(results.metrics))
    if results.config.plot.enable:
        figures = plot_results(results)
        paths = save_figures(figures, results.config.plot.output_dir)
        print("\nSaved figures:")
        for name, path in paths.items():
            print(f"  {name}: {path}")
        if results.config.plot.show and _backend_supports_gui():
            plt.show()
    return results


if __name__ == "__main__":
    main()
