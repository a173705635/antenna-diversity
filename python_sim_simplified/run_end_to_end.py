"""Run the complete simplified simulation and save its figures."""

from pathlib import Path

from metrics import format_metrics
from pipeline import run_end_to_end
from plotting import plot_detailed_results, plot_results
from presets import default_config


def save_results_figures(results, output_dir):
    import matplotlib.pyplot as plt

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"figure.max_open_warning": 0}):
        figures = {
            **plot_results(results),
            **plot_detailed_results(results),
        }

    paths = {}
    for name, figure in figures.items():
        path = directory / f"{name}.png"
        figure.savefig(path, dpi=150)
        paths[name] = path.resolve()
        if not results["config"].plot.show:
            plt.close(figure)
    return paths


def main():
    cfg = default_config()

    # Edit cfg here when studying frequency, antenna combinations, or phase.
    results = run_end_to_end(cfg)
    print(format_metrics(results["metrics"]))

    save_results_figures(results, cfg.plot.output_dir)

    if cfg.plot.show:
        import matplotlib.pyplot as plt

        if plt.get_backend().lower() != "agg":
            plt.show()


if __name__ == "__main__":
    main()
