"""Command-line entry point for reproducible spatial research studies."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from antenna_diversity.studies import PRESET_NAMES, resolve_study_config, run_research


_MODES = {"single", "egc", "mvdr"}
_STAGES = {
    "baseline", "single_factor", "combined_error", "multi_jammer",
    "selection", "full_chain", "report",
}
_MANIFEST_FIELDS = {
    "schema_version", "name", "status", "created_at_utc", "updated_at_utc",
    "configuration_sha256", "git_sha", "git_status", "environment",
    "artifacts", "stages",
}
_STAGE_FIELDS = {
    "status", "record_count", "checkpoint_sha256", "checkpoint_path", "error",
}


def _modes(value: str) -> tuple[str, ...]:
    modes = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if not modes or len(set(modes)) != len(modes) or set(modes) - _MODES:
        raise argparse.ArgumentTypeError(
            "modes must be a unique comma-separated subset of single,egc,mvdr"
        )
    return modes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a checkpointed spatial antenna-diversity research study."
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--preset", choices=PRESET_NAMES)
    parser.add_argument("--name")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--combinations")
    parser.add_argument("--modes", type=_modes)
    parser.add_argument("--ranking-mode", choices=sorted(_MODES))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def _strict_json(path: Path) -> object:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        return json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"invalid strict JSON artifact: {path}") from error


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"completed study has invalid {label}")
    return value


def _confined_file(study_dir: Path, value: object, label: str) -> Path:
    if not isinstance(value, str):
        raise RuntimeError(f"completed study has invalid {label} path")
    relative = Path(value)
    if relative.is_absolute():
        raise RuntimeError(f"completed study has invalid {label} path")
    candidate = (study_dir / relative).resolve()
    try:
        candidate.relative_to(study_dir)
    except ValueError as error:
        raise RuntimeError(f"completed study has invalid {label} path") from error
    if not candidate.is_file():
        raise RuntimeError(f"completed study is missing {label}: {candidate}")
    return candidate


def _completed_artifacts(study_dir: Path) -> tuple[Path, Path, dict[str, Path], dict[str, Path]]:
    study_dir = Path(study_dir).resolve()
    manifest_path = study_dir / "manifest.json"
    manifest = _mapping(_strict_json(manifest_path), "manifest")
    if set(manifest) != _MANIFEST_FIELDS or manifest.get("schema_version") != 1:
        raise RuntimeError("completed study has invalid manifest schema")
    if manifest.get("status") != "completed":
        raise RuntimeError("study manifest is not completed")
    stages = _mapping(manifest.get("stages"), "manifest stages")
    if set(stages) != _STAGES:
        raise RuntimeError("completed study has invalid manifest stages")
    for name, raw_record in stages.items():
        record = _mapping(raw_record, f"{name} stage record")
        if set(record) != _STAGE_FIELDS or record.get("status") != "completed":
            raise RuntimeError(f"study stage is not completed: {name}")

    manifest_artifacts = _mapping(manifest.get("artifacts"), "manifest artifacts")
    report_checkpoint = _confined_file(
        study_dir, manifest_artifacts.get("report"), "report checkpoint"
    )
    report_document = _mapping(_strict_json(report_checkpoint), "report checkpoint")
    if set(report_document) != {"stage", "payload"} or report_document.get("stage") != "report":
        raise RuntimeError("completed study has invalid report checkpoint")
    payload = _mapping(report_document.get("payload"), "report payload")
    artifacts = _mapping(payload.get("artifacts"), "report artifacts")
    if set(artifacts) != {"summary", "report", "tables", "figures"}:
        raise RuntimeError("completed study has invalid report artifacts")
    report_path = _confined_file(study_dir, artifacts.get("report"), "report")
    tables = {
        str(name): _confined_file(study_dir, value, f"table {name}")
        for name, value in _mapping(artifacts.get("tables"), "report tables").items()
    }
    figures = {
        str(name): _confined_file(study_dir, value, f"figure {name}")
        for name, value in _mapping(artifacts.get("figures"), "report figures").items()
    }
    return manifest_path.resolve(), report_path, tables, figures


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    overrides = {
        name: value
        for name, value in (
            ("name", arguments.name),
            ("output_dir", arguments.output_dir),
            ("combinations", arguments.combinations),
            ("modes", arguments.modes),
            ("ranking_mode", arguments.ranking_mode),
            ("seed", arguments.seed),
        )
        if value is not None
    }
    try:
        resolved = resolve_study_config(
            config_path=arguments.config,
            preset=arguments.preset,
            cli_overrides=overrides,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    study_dir = run_research(resolved, resume=arguments.resume)
    manifest, report, tables, figures = _completed_artifacts(study_dir)
    print(f"manifest: {manifest}")
    print(f"report: {report}")
    for name, path in tables.items():
        print(f"table.{name}: {path}")
    for name, path in figures.items():
        print(f"figure.{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
