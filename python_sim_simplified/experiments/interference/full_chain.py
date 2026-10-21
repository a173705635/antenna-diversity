"""Run the complete simplified simulation, save figures, and display them."""

from pathlib import Path
import csv
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
from experiments.paths import INTERFERENCE_RESULTS_DIR, reset_output

OUTPUT_DIR = INTERFERENCE_RESULTS_DIR/'full_chain'
ANTENNA_MODEL = "parametric"  # "ideal" or "parametric"; synthetic, not measured phone data.
DURATION_MS = None
SHOW_PLOTS_FLAG = True
FREEZE_SOURCE_FLAG = False
# 馈线传播时延(ns)及载频总相位(deg)独立配置；仅时延不会自动改写载频相位。
FEEDLINE_DELAY_NS = {"BL": 0, "BR": 0, "TL": 0, "TR": 0}
FEEDLINE_PHASE_AT_CARRIER_DEG = {"BL": 0, "BR": 0, "TL": 0, "TR": 0}


def save_results_figures(results):
    import matplotlib.pyplot as plt

    output_dir = Path(results["config"].plot.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"figure.max_open_warning": 0}):
        figures = {
            **plot_detailed_results(results),
            "tracking": plot_tracking(results),
            "cn0": plot_cn0(results),
        }

    for old_path in output_dir.glob("*.png"):
        old_path.unlink()

    figure_paths = {}
    for name, figure in figures.items():
        # 返回键沿用绘图接口；落盘名称补上实验和指标，便于单独查看图片。
        metric_name = {
            "tracking": "tracking_loop_diagnostics",
            "cn0": "tracking_cn0_db_hz",
        }.get(name, name)
        figure_path = output_dir / f"full_chain__{metric_name}.png"
        figure.savefig(figure_path, dpi=150)
        figure_paths[name] = figure_path.resolve()
        if not results["config"].plot.show_flag:
            plt.close(figure)
    return figure_paths


def main():
    import json
    from dataclasses import asdict
    cfg = {"ideal": default_config, "parametric": parametric_config}[ANTENNA_MODEL]()
    output_dir = reset_output(OUTPUT_DIR)
    cfg.plot.output_dir = output_dir
    if DURATION_MS is not None:
        cfg.signal.duration_ms = cfg.tracking.max_epochs = DURATION_MS
    cfg.plot.show_flag = SHOW_PLOTS_FLAG
    cfg.feedline.delay_ns = FEEDLINE_DELAY_NS.copy()
    cfg.feedline.phase_at_carrier_deg = FEEDLINE_PHASE_AT_CARRIER_DEG.copy()

    # Edit cfg here when studying frequency, antenna combinations, or phase.
    results = run_end_to_end(cfg)
    print(f"Antenna model: {cfg.phone.antenna_model.model_label}")
    print(format_metrics(results["metrics"]))

    save_results_figures(results)
    (output_dir/'manifest.json').write_text(json.dumps(serializable({
        'config': asdict(cfg),
        'methods': {record['mode']: {
            'receiver_role': record['receiver_role'],
            'phase_estimate_source': record['phase_estimate_source'],
        } for record in results['metrics']},
        'source_sha256': source_hashes()}),
        default=str, indent=2, allow_nan=False), encoding='utf-8')
    (output_dir/'summary.json').write_text(json.dumps(serializable(results['metrics']),
        indent=2, allow_nan=False), encoding='utf-8')
    # 每种算法一行，复用同一份 metrics；完整追溯配置留在 manifest。
    with (output_dir / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        columns = ('mode', 'acquired_flag', 'tracking_flag',
                   'tracking_cn0_mean_db_hz', 'acquisition_metric',
                   'acquisition_code_bias_chips', 'tracking_code_bias_rms_m')
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: record[key] for key in columns} for record in results['metrics'])
    if FREEZE_SOURCE_FLAG:
        freeze_source(output_dir)

    if cfg.plot.show_flag:
        import matplotlib.pyplot as plt

        if plt.get_backend().lower() != "agg":
            plt.show()


if __name__ == "__main__":
    main()
