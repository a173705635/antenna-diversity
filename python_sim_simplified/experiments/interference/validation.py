"""Fixed anti-interference validation entry: conditions live in validation_config.py."""

from dataclasses import asdict
import csv
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.provenance import source_hashes
from core.serialization import serializable
from experiments.interference.validation_config import build_cases, build_settings
from experiments.interference.validation_support import plot_validation, run_validation
from experiments.paths import reset_output


def _write_json(path: Path, content: object) -> None:
    def paths_as_text(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {key: paths_as_text(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [paths_as_text(item) for item in value]
        return value

    path.write_text(json.dumps(serializable(paths_as_text(content)), indent=2,
                               allow_nan=False), encoding="utf-8")


def main() -> dict:
    """Generate, independently score, verify, save and plot selected conditions."""
    settings = build_settings()
    cases = build_cases()
    summary = run_validation(settings, cases)
    # 当前批次完成后刷新本入口的专用输出，避免换扫描因素后残留旧图。
    reset_output(settings.output_dir)
    selected_names = {item["name"] for item in summary["cases"]}
    actual_configs = {}
    for case in cases:
        if case.name not in selected_names:
            continue
        actual_configs[case.name] = {}
        for seed in settings.seeds:
            cfg = asdict(case.config)
            cfg["random_seed"] = seed
            cfg["signal"]["duration_ms"] = settings.duration_ms
            cfg["tracking"]["max_epochs"] = settings.duration_ms
            actual_configs[case.name][str(seed)] = cfg
    manifest = {
        "settings": asdict(settings),
        "cases": summary["cases"],
        "actual_receiver_configs": actual_configs,
        "dfe_phase_definition": "Whole DFE row including target, every enabled interferer, thermal noise and frontend residual; estimated phase is an externally supplied inverse rotation.",
        "source_sha256": source_hashes(),
    }
    _write_json(settings.output_dir / "manifest.json", manifest)
    _write_json(settings.output_dir / "summary.json", summary)
    _write_json(settings.output_dir / "verification.json", summary["verification"])
    rows = summary["rows"]
    with (settings.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        # 配对条件比普通条件多恢复误差列；普通条件对应单元格留空。
        columns = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                         for key, value in row.items()} for row in rows)
    plot_validation(rows, settings)
    if not summary["verification"]["passed"]:
        raise AssertionError("validation verification failed; inspect verification.json")
    return summary


if __name__ == "__main__":
    main()
