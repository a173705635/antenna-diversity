"""Run the complete simplified simulation and save its figures."""

from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from pathlib import Path

from core.metrics import format_metrics
from core.pipeline import run_end_to_end
from core.plotting import (
    plot_cn0,
    plot_detailed_results,
    plot_multipath_results,
    plot_tracking,
)
from core.presets import default_config
from core.provenance import source_hashes, freeze_source
from core.serialization import serializable


def save_results_figures(results, output_dir):
    import matplotlib.pyplot as plt

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"figure.max_open_warning": 0}):
        figures = {
            **plot_detailed_results(results),
            "tracking": plot_tracking(results),
            "cn0": plot_cn0(results),
            **plot_multipath_results(results),
        }

    for old_path in directory.glob("*.png"):
        old_path.unlink()

    paths = {}
    for name, figure in figures.items():
        path = directory / f"{name}.png"
        figure.savefig(path, dpi=150)
        paths[name] = path.resolve()
        if not results["config"].plot.show:
            plt.close(figure)
    return paths


def main():
    import argparse
    import json
    from dataclasses import asdict

    parser = argparse.ArgumentParser(description="RF/DFE, antenna combining, acquisition and tracking")
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1]/'results/full_chain')
    parser.add_argument('--duration-ms', type=int)
    parser.add_argument('--freeze-source', action='store_true')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    cfg = default_config()
    cfg.plot.output_dir = args.output
    args.output.mkdir(parents=True, exist_ok=False)
    if args.duration_ms is not None:
        cfg.signal.duration_ms = cfg.tracking.max_epochs = args.duration_ms
    if args.no_show:
        cfg.plot.show = False
    cfg.phone.channel_delay_ns = {
        "BL": 0,
        "BR": 0,
        "TL": 0,
        "TR": 0,
    }
    # cfg.phone.estimated_channel_delay_ns = {
    #     "BL": 1,
    #     "BR": 5,
    #     "TL": 3,
    #     "TR": 10,
    # }

    # Edit cfg here when studying frequency, antenna combinations, or phase.
    results = run_end_to_end(cfg)
    print(format_metrics(results["metrics"]))

    save_results_figures(results, cfg.plot.output_dir)
    output = Path(cfg.plot.output_dir)
    (output/'manifest.json').write_text(json.dumps(serializable({
        'config': asdict(cfg), 'source_sha256': source_hashes()}),
        default=str, indent=2, allow_nan=False), encoding='utf-8')
    (output/'summary.json').write_text(json.dumps(serializable(results['metrics']),
        indent=2, allow_nan=False), encoding='utf-8')
    if args.freeze_source:
        freeze_source(output)

    if cfg.plot.show:
        import matplotlib.pyplot as plt

        if plt.get_backend().lower() != "agg":
            plt.show()


if __name__ == "__main__":
    main()
