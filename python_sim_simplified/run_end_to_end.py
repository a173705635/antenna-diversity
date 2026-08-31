"""Run the complete simplified simulation and save its figures."""

from config import default_config
from metrics import format_metrics
from pipeline import run_end_to_end
from plotting import plot_results


def main():
    cfg = default_config()

    # Edit cfg here when studying frequency, antenna combinations, or phase.
    results = run_end_to_end(cfg)
    print(format_metrics(results["metrics"]))

    cfg.plot.output_dir.mkdir(parents=True, exist_ok=True)
    figures = plot_results(results)
    for name, figure in figures.items():
        figure.savefig(cfg.plot.output_dir / f"{name}.png", dpi=150)

    if cfg.plot.show:
        import matplotlib.pyplot as plt

        if plt.get_backend().lower() != "agg":
            plt.show()


if __name__ == "__main__":
    main()
