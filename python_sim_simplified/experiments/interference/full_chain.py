"""Run the complete simplified simulation, save figures, and display them."""

from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from pathlib import Path

from core.metrics import format_metrics
from core.pipeline import run_end_to_end
from core.plotting import (
    plot_cn0,
    plot_detailed_results,
    plot_tracking,
)
from core.presets import default_config, parametric_config
from core.provenance import source_hashes, freeze_source
from core.serialization import serializable
from experiments.paths import INTERFERENCE_RESULTS as RESULTS, reset_output

OUTPUT = RESULTS/'full_chain'
ANTENNA_MODEL = "parametric"  # "ideal" or "parametric"; synthetic, not measured phone data.
DURATION_MS = None
SHOW_PLOTS = True
FREEZE_SOURCE = False


def save_results_figures(results, output_dir):
    import matplotlib.pyplot as plt

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"figure.max_open_warning": 0}):
        figures = {
            **plot_detailed_results(results),
            "tracking": plot_tracking(results),
            "cn0": plot_cn0(results),
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
    import json
    from dataclasses import asdict
    cfg = {"ideal": default_config, "parametric": parametric_config}[ANTENNA_MODEL]()
    output = reset_output(OUTPUT)
    cfg.plot.output_dir = output
    if DURATION_MS is not None:
        cfg.signal.duration_ms = cfg.tracking.max_epochs = DURATION_MS
    cfg.plot.show = SHOW_PLOTS
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
    print(f"Antenna model: {cfg.phone.antenna_model.model_label}")
    print(format_metrics(results["metrics"]))

    save_results_figures(results, cfg.plot.output_dir)
    (output/'manifest.json').write_text(json.dumps(serializable({
        'config': asdict(cfg), 'source_sha256': source_hashes()}),
        default=str, indent=2, allow_nan=False), encoding='utf-8')
    (output/'summary.json').write_text(json.dumps(serializable(results['metrics']),
        indent=2, allow_nan=False), encoding='utf-8')
    if FREEZE_SOURCE:
        freeze_source(output)

    if cfg.plot.show:
        import matplotlib.pyplot as plt

        if plt.get_backend().lower() != "agg":
            plt.show()


if __name__ == "__main__":
    main()
