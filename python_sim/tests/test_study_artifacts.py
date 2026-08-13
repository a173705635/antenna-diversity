import csv
import json
from pathlib import Path

import pytest

from antenna_diversity.studies.artifacts import StudyStore, write_csv_atomic, write_json_atomic
from tests.study_helpers import resolved_for_test


def test_strict_json_rejects_nan_without_publishing_destination(tmp_path: Path):
    """Rejecting non-finite payloads must not leave a published file behind."""
    destination = tmp_path / "bad.json"

    with pytest.raises(ValueError):
        write_json_atomic(destination, {"value": float("nan")})

    assert not destination.exists()


def test_resume_rejects_changed_scientific_configuration(tmp_path: Path):
    """A different scientific input must never resume an existing study."""
    first = resolved_for_test(
        output_dir=tmp_path, name="artifact-test", exclusion_angle_deg=15.0
    )
    StudyStore.create(first)
    changed = resolved_for_test(
        output_dir=tmp_path, name="artifact-test", exclusion_angle_deg=20.0
    )

    with pytest.raises(ValueError, match="configuration hash"):
        StudyStore.resume(changed)


def test_resume_skips_only_valid_completed_stage(tmp_path: Path):
    """Only a checkpoint whose bytes and stage metadata verify is reusable."""
    resolved = resolved_for_test(output_dir=tmp_path, name="artifact-test")
    store = StudyStore.create(resolved)
    store.write_stage("baseline", {"rows": [{"value": 1.0}]})

    resumed = StudyStore.resume(resolved)

    assert resumed.completed_stages == ("baseline",)


def test_create_refuses_existing_study_directory(tmp_path: Path):
    """Create must not overwrite an earlier study's artifacts."""
    resolved = resolved_for_test(output_dir=tmp_path, name="artifact-test")
    StudyStore.create(resolved)

    with pytest.raises(FileExistsError):
        StudyStore.create(resolved)


def test_resume_rejects_unknown_manifest_schema(tmp_path: Path):
    """A schema change must not silently reinterpret previous artifacts."""
    resolved = resolved_for_test(output_dir=tmp_path, name="artifact-test")
    store = StudyStore.create(resolved)
    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 999
    write_json_atomic(store.manifest_path, manifest)

    with pytest.raises(ValueError, match="schema"):
        StudyStore.resume(resolved)


def test_resume_marks_corrupt_completed_checkpoint_pending(tmp_path: Path):
    """Tampering with a completed checkpoint forces its stage to rerun."""
    resolved = resolved_for_test(output_dir=tmp_path, name="artifact-test")
    store = StudyStore.create(resolved)
    checkpoint = store.write_stage("baseline", {"rows": [{"value": 1.0}]})
    checkpoint.write_text("not json", encoding="utf-8")

    resumed = StudyStore.resume(resolved)

    assert resumed.completed_stages == ()
    assert resumed.manifest["stages"]["baseline"]["status"] == "pending"
    assert resumed.manifest["status"] == "running"


def test_resume_downgrades_completed_manifest_without_all_stages(tmp_path: Path):
    """A hand-edited completed flag cannot skip uncheckpointed canonical stages."""
    resolved = resolved_for_test(output_dir=tmp_path, name="artifact-test")
    store = StudyStore.create(resolved)
    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "completed"
    write_json_atomic(store.manifest_path, manifest)

    resumed = StudyStore.resume(resolved)

    assert resumed.manifest["status"] == "running"


def test_resume_downgrades_completed_manifest_with_corrupt_checkpoint(tmp_path: Path):
    """One bad checkpoint invalidates the completed study state and its stage."""
    resolved = resolved_for_test(output_dir=tmp_path, name="artifact-test")
    store = StudyStore.create(resolved)
    checkpoints = [store.write_stage(stage, {"rows": []}) for stage in StudyStore.STAGE_NAMES]
    store.complete()
    checkpoints[-1].write_text("not json", encoding="utf-8")

    resumed = StudyStore.resume(resolved)

    assert resumed.manifest["status"] == "running"
    assert resumed.manifest["stages"]["report"]["status"] == "pending"


def test_resume_rejects_malformed_artifacts_mapping(tmp_path: Path):
    """Malformed manifest structure fails at resume rather than a later public call."""
    resolved = resolved_for_test(output_dir=tmp_path, name="artifact-test")
    store = StudyStore.create(resolved)
    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"] = []
    write_json_atomic(store.manifest_path, manifest)

    with pytest.raises(ValueError, match="artifacts"):
        StudyStore.resume(resolved)


def test_resume_rejects_malformed_canonical_stage_record(tmp_path: Path):
    """Every canonical stage record must retain its controlled manifest shape."""
    resolved = resolved_for_test(output_dir=tmp_path, name="artifact-test")
    store = StudyStore.create(resolved)
    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    manifest["stages"]["baseline"] = {"status": "pending"}
    write_json_atomic(store.manifest_path, manifest)

    with pytest.raises(ValueError, match="baseline"):
        StudyStore.resume(resolved)


def test_csv_uses_caller_field_order_and_rejects_unknown_fields(tmp_path: Path):
    """CSV table shape comes from its public caller-supplied schema."""
    destination = tmp_path / "rows.csv"
    write_csv_atomic(destination, ("second", "first"), [{"first": 1, "second": 2}])

    with destination.open(newline="", encoding="utf-8") as stream:
        assert list(csv.reader(stream)) == [["second", "first"], ["2", "1"]]
    with pytest.raises(ValueError, match="unknown"):
        write_csv_atomic(destination, ("first",), [{"first": 1, "extra": 2}])


def test_stage_paths_are_confined_to_study_directory(tmp_path: Path):
    """Artifact path traversal or absolute paths must be refused."""
    store = StudyStore.create(resolved_for_test(output_dir=tmp_path, name="artifact-test"))

    with pytest.raises(ValueError, match="relative"):
        store.write_stage("baseline", {"rows": []}, artifact_path=Path("..") / "escape.json")
    with pytest.raises(ValueError, match="relative"):
        store.write_stage("baseline", {"rows": []}, artifact_path=tmp_path / "absolute.json")


def test_failed_stage_records_safe_error_summary(tmp_path: Path):
    """Failures keep diagnostic type/message/time but never traceback data."""
    store = StudyStore.create(resolved_for_test(output_dir=tmp_path, name="artifact-test"))
    store.fail_stage("baseline", ValueError("bad input"))

    record = store.manifest["stages"]["baseline"]
    assert record["status"] == "failed"
    assert record["error"]["type"] == "ValueError"
    assert record["error"]["message"] == "bad input"
    assert record["error"]["timestamp_utc"].endswith("Z")
    assert "traceback" not in json.dumps(record)


def test_complete_requires_every_stage_then_marks_manifest_completed(tmp_path: Path):
    """A study cannot report success until all canonical stages checkpoint."""
    store = StudyStore.create(resolved_for_test(output_dir=tmp_path, name="artifact-test"))

    with pytest.raises(ValueError, match="all stages"):
        store.complete()
    for stage in StudyStore.STAGE_NAMES:
        store.write_stage(stage, {"rows": []})

    store.complete()

    assert store.manifest["status"] == "completed"
