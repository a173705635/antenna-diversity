"""Atomic, reproducible storage for spatial-study stage artifacts."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
import numpy
import scipy

from .models import ResolvedStudyConfig


MANIFEST_SCHEMA_VERSION = 1
_MANIFEST_NAME = "manifest.json"
_CONFIG_NAME = "resolved-config.json"
_MANIFEST_STATUSES = {"running", "completed", "failed"}
_STAGE_STATUSES = {"pending", "completed", "failed"}
REQUIRED_STAGES = frozenset({
    "baseline", "single_factor", "combined_error", "multi_jammer",
    "selection", "full_chain", "report",
})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _strict_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _atomic_write(destination: Path, content: bytes) -> None:
    destination = Path(destination)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False
        ) as stream:
            temp_name = stream.name
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, destination)
        temp_name = None
    finally:
        if temp_name is not None:
            try:
                Path(temp_name).unlink()
            except FileNotFoundError:
                pass


def write_json_atomic(destination: Path, payload: object) -> None:
    """Serialize strict JSON fully before atomically replacing *destination*."""
    content = _strict_json_bytes(payload)
    _atomic_write(Path(destination), content)


def write_csv_atomic(destination: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    """Write CSV in the caller's exact column order, rejecting extra values."""
    ordered_fields = tuple(fieldnames)
    if len(set(ordered_fields)) != len(ordered_fields):
        raise ValueError("CSV field names must be unique")
    if any(not isinstance(field, str) for field in ordered_fields):
        raise ValueError("CSV field names must be strings")
    allowed = set(ordered_fields)
    for row in rows:
        unknown = set(row) - allowed
        if unknown:
            raise ValueError(f"unknown CSV field: {sorted(unknown)[0]}")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=ordered_fields, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write(Path(destination), stream.getvalue().encode("utf-8"))


def _git_metadata() -> tuple[str | None, str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None, "git_unavailable"
    return result.stdout.strip() or None, "available"


def _environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
        "executable": sys.executable,
    }


def _initial_stage() -> dict[str, object]:
    return {
        "status": "pending",
        "record_count": 0,
        "checkpoint_sha256": None,
        "checkpoint_path": None,
        "error": None,
    }


def _strict_json_load(content: bytes) -> object:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(content.decode("utf-8"), parse_constant=reject_constant)


def _record_count(payload: object) -> int:
    if not isinstance(payload, Mapping):
        return 1
    rows = payload.get("rows")
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)):
        return len(rows)
    tables = payload.get("tables")
    if isinstance(tables, Mapping):
        return sum(
            len(value) for value in tables.values()
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
        )
    return 1


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class StudyStore:
    """Own a study directory, its immutable configuration, and stage checkpoints."""

    REQUIRED_STAGES = REQUIRED_STAGES

    def __init__(self, resolved: ResolvedStudyConfig, study_dir: Path, manifest: dict[str, object]) -> None:
        self.resolved = resolved
        self.study_dir = study_dir.resolve()
        self.manifest_path = self.study_dir / _MANIFEST_NAME
        self.config_path = self.study_dir / _CONFIG_NAME
        self.manifest = manifest

    @property
    def completed_stages(self) -> tuple[str, ...]:
        return tuple(sorted(
            name for name in self.REQUIRED_STAGES
            if self._stage_record(name).get("status") == "completed"
        ))

    def is_stage_completed(self, stage: str) -> bool:
        """Return whether *stage* has a completed record; the runner verifies content."""
        return self._stage_record(stage).get("status") == "completed"

    @classmethod
    def create(cls, resolved: ResolvedStudyConfig) -> "StudyStore":
        study_dir = Path(resolved.study.output_dir) / resolved.study.name
        if study_dir.exists():
            raise FileExistsError(f"study directory already exists: {study_dir}")
        study_dir.mkdir(parents=True)
        git_sha, git_status = _git_metadata()
        now = _utc_now()
        manifest: dict[str, object] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "name": resolved.study.name,
            "status": "running",
            "created_at_utc": now,
            "updated_at_utc": now,
            "configuration_sha256": resolved.config_sha256,
            "git_sha": git_sha,
            "git_status": git_status,
            "environment": _environment(),
            "artifacts": {"configuration": _CONFIG_NAME, "manifest": _MANIFEST_NAME},
            "stages": {name: _initial_stage() for name in sorted(cls.REQUIRED_STAGES)},
        }
        store = cls(resolved, study_dir, manifest)
        write_json_atomic(store.config_path, resolved.to_dict())
        store._save_manifest()
        return store

    @classmethod
    def resume(cls, resolved: ResolvedStudyConfig) -> "StudyStore":
        study_dir = (Path(resolved.study.output_dir) / resolved.study.name).resolve()
        manifest_path = study_dir / _MANIFEST_NAME
        try:
            loaded = _strict_json_load(manifest_path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"invalid study manifest: {error}") from error
        if not isinstance(loaded, dict):
            raise ValueError("invalid study manifest schema")
        cls._validate_manifest(loaded, resolved)
        if loaded["configuration_sha256"] != resolved.config_sha256:
            raise ValueError("configuration hash does not match existing study")
        store = cls(resolved, study_dir, loaded)
        repaired = False
        for name in cls.REQUIRED_STAGES:
            record = store._stage_record(name)
            if record.get("status") == "completed" and not store._checkpoint_valid(name, record):
                record.update(_initial_stage())
                repaired = True
        all_valid = all(
            store._stage_record(name).get("status") == "completed"
            and store._checkpoint_valid(name, store._stage_record(name))
            for name in cls.REQUIRED_STAGES
        )
        if repaired or store.manifest.get("status") != "completed" or not all_valid:
            store.manifest["status"] = "running"
        store._save_manifest()
        return store

    @classmethod
    def _validate_manifest(cls, manifest: Mapping[str, object], resolved: ResolvedStudyConfig) -> None:
        required = {
            "schema_version", "name", "status", "created_at_utc", "updated_at_utc",
            "configuration_sha256", "git_sha", "git_status", "environment", "artifacts", "stages",
        }
        if set(manifest) != required:
            raise ValueError("invalid study manifest schema")
        if type(manifest["schema_version"]) is not int or manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported manifest schema")
        if manifest["name"] != resolved.study.name or not isinstance(manifest["name"], str):
            raise ValueError("invalid study manifest name")
        if not isinstance(manifest["status"], str) or manifest["status"] not in _MANIFEST_STATUSES:
            raise ValueError("invalid study manifest status")
        if not all(isinstance(manifest[key], str) for key in ("created_at_utc", "updated_at_utc", "configuration_sha256")):
            raise ValueError("invalid study manifest metadata")
        if manifest["git_sha"] is not None and not isinstance(manifest["git_sha"], str):
            raise ValueError("invalid study manifest Git metadata")
        if not isinstance(manifest["git_status"], str) or manifest["git_status"] not in {"available", "git_unavailable"}:
            raise ValueError("invalid study manifest Git metadata")
        if not isinstance(manifest["environment"], dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in manifest["environment"].items()
        ):
            raise ValueError("invalid study manifest environment")
        artifacts = manifest["artifacts"]
        if not isinstance(artifacts, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in artifacts.items()
        ):
            raise ValueError("invalid study manifest artifacts")
        stages = manifest["stages"]
        if not isinstance(stages, dict) or set(stages) != cls.REQUIRED_STAGES:
            raise ValueError("invalid study manifest stages")
        for name in cls.REQUIRED_STAGES:
            cls._validate_stage_record(name, stages[name])

    @staticmethod
    def _validate_stage_record(name: str, record: object) -> None:
        required = {"status", "record_count", "checkpoint_sha256", "checkpoint_path", "error"}
        if not isinstance(record, dict) or set(record) != required:
            raise ValueError(f"invalid study manifest stage: {name}")
        if (
            not isinstance(record["status"], str)
            or record["status"] not in _STAGE_STATUSES
            or not _is_int(record["record_count"])
            or record["record_count"] < 0
        ):
            raise ValueError(f"invalid study manifest stage: {name}")
        for key in ("checkpoint_sha256", "checkpoint_path"):
            if record[key] is not None and not isinstance(record[key], str):
                raise ValueError(f"invalid study manifest stage: {name}")
        error = record["error"]
        if error is not None and (
            not isinstance(error, dict)
            or set(error) != {"type", "message", "timestamp_utc"}
            or not all(isinstance(value, str) for value in error.values())
        ):
            raise ValueError(f"invalid study manifest stage: {name}")

    def _stage_record(self, stage: str) -> dict[str, object]:
        if stage not in self.REQUIRED_STAGES:
            raise ValueError(f"unknown stage: {stage}")
        stages = self.manifest.get("stages")
        if not isinstance(stages, dict) or not isinstance(stages.get(stage), dict):
            raise ValueError("invalid study manifest stages")
        return stages[stage]

    def _artifacts(self) -> dict[str, object]:
        artifacts = self.manifest.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError("invalid study manifest artifacts")
        return artifacts

    def _confined_path(self, relative_path: Path) -> Path:
        if relative_path.is_absolute():
            raise ValueError("artifact path must be relative")
        candidate = (self.study_dir / relative_path).resolve()
        try:
            candidate.relative_to(self.study_dir)
        except ValueError as error:
            raise ValueError("artifact path must be relative and confined to study directory") from error
        return candidate

    def _checkpoint_valid(self, stage: str, record: Mapping[str, object]) -> bool:
        relative_path = record.get("checkpoint_path")
        expected_hash = record.get("checkpoint_sha256")
        if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
            return False
        try:
            checkpoint = self._confined_path(Path(relative_path))
            content = checkpoint.read_bytes()
            document = _strict_json_load(content)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return False
        return (
            isinstance(document, dict)
            and document.get("stage") == stage
            and hashlib.sha256(content).hexdigest() == expected_hash
        )

    def _save_manifest(self) -> None:
        self.manifest["updated_at_utc"] = _utc_now()
        write_json_atomic(self.manifest_path, self.manifest)

    def write_stage(self, stage: str, payload: object, *, artifact_path: Path | None = None) -> Path:
        self._stage_record(stage)
        relative = artifact_path if artifact_path is not None else Path("checkpoints") / f"{stage}.json"
        checkpoint = self._confined_path(relative)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        document = {"stage": stage, "payload": payload}
        write_json_atomic(checkpoint, document)
        content = checkpoint.read_bytes()
        record = self._stage_record(stage)
        record.update({
            "status": "completed",
            "record_count": _record_count(payload),
            "checkpoint_sha256": hashlib.sha256(content).hexdigest(),
            "checkpoint_path": checkpoint.relative_to(self.study_dir).as_posix(),
            "error": None,
        })
        artifacts = self._artifacts()
        artifacts[stage] = checkpoint.relative_to(self.study_dir).as_posix()
        self._save_manifest()
        return checkpoint

    def read_stage(self, stage: str) -> object:
        record = self._stage_record(stage)
        if record.get("status") != "completed" or not self._checkpoint_valid(stage, record):
            raise ValueError(f"stage checkpoint is not valid: {stage}")
        path = self._confined_path(Path(str(record["checkpoint_path"])))
        document = _strict_json_load(path.read_bytes())
        if not isinstance(document, dict) or "payload" not in document:
            raise ValueError(f"stage checkpoint is not valid: {stage}")
        return document["payload"]

    def fail_stage(self, stage: str, error: BaseException) -> None:
        record = self._stage_record(stage)
        record.update({
            "status": "failed",
            "record_count": 0,
            "checkpoint_sha256": None,
            "checkpoint_path": None,
            "error": {"type": type(error).__name__, "message": str(error), "timestamp_utc": _utc_now()},
        })
        self.manifest["status"] = "failed"
        self._save_manifest()

    def complete(self) -> None:
        missing = sorted(name for name in self.REQUIRED_STAGES if self._stage_record(name).get("status") != "completed")
        if missing:
            raise ValueError("cannot complete until all stages are completed")
        self.manifest["status"] = "completed"
        self._save_manifest()
