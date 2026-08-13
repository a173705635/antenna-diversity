# Reproducible Spatial Research Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python-only, reproducible research runner that compares arbitrary Mate 70 corner subarrays, quantifies MVDR robustness, validates representative cases through RF-to-tracking, and emits strict data files, figures, and a Markdown report.

**Architecture:** Add an `antenna_diversity.studies` orchestration layer above the existing analytical experiments and sample-domain pipeline. Resolve one immutable scientific configuration, generate stable scenario records, run analytical stages before representative full-chain cases, and publish each stage through hash-checked atomic checkpoints. Plotting and reporting consume saved normalized records only; they never rerun simulations.

**Tech Stack:** Python 3.11+, NumPy 2.x, SciPy 1.14+, Matplotlib 3.9+, pytest 9+, standard-library `argparse`, `csv`, `dataclasses`, `hashlib`, `importlib.metadata`, `json`, `pathlib`, `platform`, `tempfile`, and existing antenna-diversity modules.

## Global Constraints

- Implement only the Python research suite; do not modify MATLAB files.
- Keep the existing Mate 70 model explicitly idealized: outer-corner isotropic phase centers, no product-performance claim.
- Support one, several, or all 11 unique two-to-four-corner subarrays.
- Treat SINGLE as one physical corner and label it `SINGLE(TL)`, `SINGLE(TR)`, `SINGLE(BL)`, or `SINGLE(BR)`; do not report it as multi-antenna combining.
- Use analytical output SINR as the primary spatial metric and full-chain acquisition, tracking, final C/N0, and Doppler error as validation metrics.
- Fully scan one interferer; use deterministic templates for representative two- and three-interferer stress cases.
- Study MVDR with single-factor errors and light/medium/heavy combined-error realizations; keep EGC as the comparison baseline.
- Use the exact quick/standard/extended grids, sample counts, error ranges, and role definitions in `docs/superpowers/specs/2026-08-13-spatial-research-suite-design.md`.
- Rank geometries in `ranking_mode` only; default to MVDR and use resolvable-region output-SINR 10th percentile as the primary key.
- Keep analytical SINR named `analytic_output_sinr_db`; do not invent a sampled or measured SINR from the full chain.
- Save strict JSON with `allow_nan=False`, UTF-8 long-form CSV, PNG figures, and Markdown; unavailable values are `null` with an explanatory status.
- Do not save raw RF, ADC, DFE, acquisition-map, or combined-sample arrays in study artifacts.
- Run stages sequentially and derive all child seeds deterministically from the study seed.
- Resume only when schema version and scientific configuration SHA-256 match exactly.
- Do not add third-party dependencies or alter the existing RF, DFE, acquisition, tracking, C/N0, or combiner algorithms.
- Run every test command from `F:\antenna_diversity\python_sim` with warnings treated as errors.
- Preserve unrelated user changes and stage only the exact files listed for each task.

## File Structure

### New production files

- `python_sim/antenna_diversity/studies/__init__.py`: stable public study API.
- `python_sim/antenna_diversity/studies/models.py`: immutable resolved configuration, case, stage, selection, and full-chain result records.
- `python_sim/antenna_diversity/studies/config.py`: presets, JSON/CLI precedence, simulation overrides, normalization, and scientific hash input.
- `python_sim/antenna_diversity/studies/scenarios.py`: combination parsing, stable case IDs, deterministic seeds, error realizations, and multi-jammer templates.
- `python_sim/antenna_diversity/studies/analytical.py`: baseline, power, single-factor, combined-error, and multi-jammer analytical stages.
- `python_sim/antenna_diversity/studies/ranking.py`: direction masks, summaries, stable geometry ranking, and representative-case selection.
- `python_sim/antenna_diversity/studies/full_chain.py`: selected RF-to-tracking runs and compact C/N0 records.
- `python_sim/antenna_diversity/studies/artifacts.py`: strict JSON/CSV, atomic writes, manifest, checkpoints, and resume validation.
- `python_sim/antenna_diversity/studies/runner.py`: sequential stage orchestration.
- `python_sim/antenna_diversity/studies/plots.py`: deterministic study figures from normalized records.
- `python_sim/antenna_diversity/studies/report.py`: Markdown and summary generation from completed artifacts.
- `python_sim/run_spatial_research.py`: command-line entry point.
- `python_sim/studies/example_quick.json`: editable example configuration.

### New tests

- `python_sim/tests/__init__.py`: make shared study helpers importable without changing test discovery.
- `python_sim/tests/study_helpers.py`: shared resolved-config and normalized-row test builders; pytest does not collect it as a test module.
- `python_sim/tests/test_study_config.py`
- `python_sim/tests/test_study_scenarios.py`
- `python_sim/tests/test_study_analytical.py`
- `python_sim/tests/test_study_ranking.py`
- `python_sim/tests/test_study_full_chain.py`
- `python_sim/tests/test_study_artifacts.py`
- `python_sim/tests/test_study_reporting.py`
- `python_sim/tests/test_study_runner.py`
- `python_sim/tests/test_study_cli.py`

### Modified files

- `python_sim/README.md`: research commands, output contract, and interpretation limits.
- `python_sim/ARCHITECTURE.md`: study orchestration layer and data flow.

---

### Task 1: Resolved research configuration and precedence

**Files:**
- Create: `python_sim/antenna_diversity/studies/__init__.py`
- Create: `python_sim/antenna_diversity/studies/models.py`
- Create: `python_sim/antenna_diversity/studies/config.py`
- Create: `python_sim/tests/__init__.py`
- Create: `python_sim/tests/study_helpers.py`
- Create: `python_sim/tests/test_study_config.py`

**Interfaces:**
- Produces: `PRESET_NAMES = ("quick", "standard", "extended")`.
- Produces: `StudyConfig`, `ResolvedStudyConfig`, `ErrorLevel`, and `InterfererTemplate` frozen dataclasses.
- Produces: `resolve_study_config(config_path: Path | None = None, preset: str | None = None, cli_overrides: Mapping[str, object] | None = None) -> ResolvedStudyConfig`.
- Produces: `ResolvedStudyConfig.to_dict() -> dict[str, object]` and `ResolvedStudyConfig.scientific_dict() -> dict[str, object]` with copied containers.
- Produces: `ResolvedStudyConfig.from_parts(study: StudyConfig, simulation: SimulationConfig) -> ResolvedStudyConfig`, which validates, deep-copies, and recomputes the scientific hash.
- Consumes: `default_config()` as the final fallback and applies JSON `simulation` overrides before `SimulationConfig.validate()`.

- [ ] **Step 1: Write failing public-contract and preset tests**

Create `tests/test_study_config.py` with these first tests:

```python
import json
from pathlib import Path

import numpy as np
import pytest

from antenna_diversity.studies.config import resolve_study_config


def test_quick_preset_has_exact_axes_and_counts():
    resolved = resolve_study_config(preset="quick")

    np.testing.assert_array_equal(resolved.study.azimuths_deg, np.arange(0, 360, 30))
    np.testing.assert_array_equal(resolved.study.elevations_deg, np.arange(0, 91, 15))
    np.testing.assert_allclose(resolved.study.powers_dbm, [-110, -100, -90, -80])
    assert resolved.study.combined_error_samples_per_level == 50
    assert resolved.study.full_chain_run_count == 3
    assert resolved.study.ranking_mode == "mvdr"


def test_cli_overrides_json_which_overrides_preset(tmp_path: Path):
    path = tmp_path / "study.json"
    path.write_text(json.dumps({
        "study": {"name": "from-json", "exclusion_angle_deg": 20.0},
        "simulation": {"signal": {"carrier_hz": 1.2e9}},
    }), encoding="utf-8")

    resolved = resolve_study_config(
        path,
        preset="quick",
        cli_overrides={"name": "from-cli", "combinations": "TL+BR"},
    )

    assert resolved.study.name == "from-cli"
    assert resolved.study.exclusion_angle_deg == 20.0
    assert resolved.study.combinations == (("TL", "BR"),)
    assert resolved.simulation.signal.carrier_hz == 1.2e9


def test_unknown_json_key_is_rejected(tmp_path: Path):
    path = tmp_path / "study.json"
    path.write_text('{"study":{"exlusion_angle_deg":15}}', encoding="utf-8")

    with pytest.raises(ValueError, match="unknown.*exlusion_angle_deg"):
        resolve_study_config(path)


def test_resolved_snapshots_do_not_alias_caller_overrides():
    overrides = {"modes": ["egc", "mvdr"]}
    resolved = resolve_study_config(preset="quick", cli_overrides=overrides)
    before = resolved.to_dict()

    overrides["modes"].append("single")

    assert resolved.to_dict() == before
```

- [ ] **Step 2: Run the configuration tests and verify RED**

Run:

```powershell
python -W error -m pytest tests/test_study_config.py -v
```

Expected: collection fails with `ModuleNotFoundError: antenna_diversity.studies`.

- [ ] **Step 3: Define immutable result/configuration records**

In `studies/models.py`, define these exact public shapes; copy every NumPy axis in `__post_init__` and convert arrays to lists in `to_dict()`:

```python
@dataclass(frozen=True)
class ErrorLevel:
    name: str
    direction_sigma_deg: float
    position_sigma_m: float
    amplitude_sigma_db: float
    phase_sigma_deg: float


@dataclass(frozen=True)
class InterfererTemplate:
    name: str
    interferers: tuple[CWInterferer, ...]


@dataclass(frozen=True)
class StudyConfig:
    schema_version: int
    name: str
    preset: str
    seed: int
    output_dir: Path
    combinations: tuple[tuple[str, ...], ...]
    spatial_modes: tuple[str, ...]
    full_chain_modes: tuple[str, ...]
    single_antennas: tuple[str, ...]
    ranking_mode: str
    exclusion_angle_deg: float
    azimuths_deg: np.ndarray
    elevations_deg: np.ndarray
    powers_dbm: np.ndarray
    direction_errors_deg: np.ndarray
    position_errors_m: np.ndarray
    amplitude_errors_db: np.ndarray
    phase_errors_deg: np.ndarray
    loading_factors: np.ndarray
    error_levels: tuple[ErrorLevel, ...]
    combined_error_samples_per_level: int
    full_chain_run_count: int
    multi_jammer_templates: tuple[InterfererTemplate, ...]
    automatic_roles: tuple[str, ...]
    manual_cases: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class ResolvedStudyConfig:
    study: StudyConfig
    simulation: SimulationConfig
    config_sha256: str

    def to_dict(self) -> dict[str, object]:
        return strict_snapshot({
            "study": self.study,
            "simulation": self.simulation,
            "config_sha256": self.config_sha256,
        })

    def scientific_dict(self) -> dict[str, object]:
        snapshot = self.to_dict()
        study = dict(snapshot["study"])
        study.pop("name")
        study.pop("preset")
        study.pop("output_dir")
        snapshot["study"] = study
        simulation = dict(snapshot["simulation"])
        plot = dict(simulation["plot"])
        plot.pop("output_dir")
        plot.pop("enable")
        plot.pop("show")
        simulation["plot"] = plot
        snapshot["simulation"] = simulation
        snapshot.pop("config_sha256")
        return snapshot

    @classmethod
    def from_parts(
        cls,
        study: StudyConfig,
        simulation: SimulationConfig,
    ) -> "ResolvedStudyConfig":
        study_copy = copy.deepcopy(study)
        simulation_copy = copy.deepcopy(simulation)
        validate_study_config(study_copy)
        simulation_copy.validate()
        unhashed = cls(study_copy, simulation_copy, "")
        encoded = json.dumps(
            unhashed.scientific_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return cls(study_copy, simulation_copy, hashlib.sha256(encoded).hexdigest())
```

Use `schema_version=1`. Implement `strict_snapshot(value)` recursively for dataclasses, `Path`, dictionaries, tuples/lists, NumPy arrays/scalars, and primitive strict-JSON values; reject unsupported, complex, and nonfinite values. The scientific snapshot excludes study name, preset label, output directory, and plot enable/show controls; resume is a CLI control and is never stored in `StudyConfig`. It includes preset-expanded arrays, ranking rules, templates, the normalized simulation configuration, the master seed, and run counts. Set full-chain run counts to quick `3`, standard `40`, and extended `200`; concrete run seeds are derived in Task 6 from master seed, case ID, and run index.

- [ ] **Step 4: Implement presets, strict JSON, overrides, and hashing**

In `studies/config.py`:

```python
PRESET_NAMES = ("quick", "standard", "extended")


def resolve_study_config(
    config_path: Path | None = None,
    preset: str | None = None,
    cli_overrides: Mapping[str, object] | None = None,
) -> ResolvedStudyConfig:
    """Resolve defaults, preset, JSON, and CLI into one validated snapshot."""
```

Implement the exact precedence `CLI > JSON > preset > default_config()`. Accept the JSON top-level keys `study` and `simulation` only. `study` accepts the fields in `StudyConfig`; `simulation` accepts the existing groups `signal`, `phone`, `array`, `rf`, `dfe`, `diversity`, `acquisition`, `tracking`, and `interferers`. Recursively reject unknown dataclass fields. Construct `Direction` and `CWInterferer` from mappings instead of assigning raw dictionaries. Call `simulation.validate()` after all overrides.

Normalize combination strings through Task 2's `parse_combination_spec`; until Task 2 exists, keep a private equivalent and move it without behavior change in Task 2. Validate real finite axes, increasing loading factors, nonempty seeds, valid modes, `ranking_mode in spatial_modes`, exclusion angle in `[0, 180]`, unique study name safe for a directory, and all sample counts positive.

Generate the SHA-256 from `json.dumps(scientific_dict, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")`.

Create `tests/study_helpers.py` for later tasks:

```python
from dataclasses import replace
from pathlib import Path

from antenna_diversity.studies.config import resolve_study_config
from antenna_diversity.studies.models import ResolvedStudyConfig


def resolved_for_test(
    *,
    preset: str = "quick",
    output_dir: Path | None = None,
    **study_changes: object,
) -> ResolvedStudyConfig:
    base = resolve_study_config(preset=preset)
    changes = dict(study_changes)
    if output_dir is not None:
        changes["output_dir"] = output_dir
    study = replace(base.study, **changes)
    return ResolvedStudyConfig.from_parts(study, base.simulation)
```

- [ ] **Step 5: Verify strict serialization and all three presets**

Extend the test file with parametrized assertions for the standard and extended grids/counts, `json.dumps(resolved.to_dict(), allow_nan=False)`, complex/nonfinite rejection, unknown modes, duplicate combinations, and simulation frequency validation. Run:

```powershell
python -W error -m pytest tests/test_study_config.py tests/test_phone_config.py tests/test_config_numeric.py -v
```

Expected: all tests pass with no warnings.

- [ ] **Step 6: Commit the resolved configuration layer**

```powershell
git add -- python_sim/antenna_diversity/studies/__init__.py python_sim/antenna_diversity/studies/models.py python_sim/antenna_diversity/studies/config.py python_sim/tests/__init__.py python_sim/tests/study_helpers.py python_sim/tests/test_study_config.py
git commit -m "feat: add reproducible spatial study configuration"
```

---

### Task 2: Stable scenario catalog and deterministic perturbations

**Files:**
- Modify: `python_sim/antenna_diversity/studies/models.py`
- Modify: `python_sim/antenna_diversity/studies/config.py`
- Create: `python_sim/antenna_diversity/studies/scenarios.py`
- Create: `python_sim/tests/test_study_scenarios.py`

**Interfaces:**
- Produces: `parse_combination_spec(value: str | Sequence[Sequence[str]]) -> tuple[tuple[str, ...], ...]`.
- Produces: `StudyCase(case_id, stage, active_antennas, mode, seed, parameters, selection_reasons)`.
- Produces: `stable_case_id(...)`, `derive_child_seed(...)`, `sample_combined_errors(...)`, and `build_multi_jammer_cases(...)`.
- Preserves: caller configuration and input containers.

- [ ] **Step 1: Write failing combination, ID, and random-realization tests**

```python
import copy
import json

import numpy as np
import pytest

from antenna_diversity.array.geometry import active_corner_combinations
from antenna_diversity.config import default_config
from antenna_diversity.studies.config import resolve_study_config
from antenna_diversity.studies.scenarios import (
    build_multi_jammer_cases,
    parse_combination_spec,
    sample_combined_errors,
)


def test_combination_parser_supports_one_list_and_all():
    assert parse_combination_spec("TL+BR") == (("TL", "BR"),)
    assert parse_combination_spec("TL+BR,TL+TR+BR") == (
        ("TL", "BR"), ("TL", "TR", "BR"),
    )
    assert parse_combination_spec("all") == active_corner_combinations()


@pytest.mark.parametrize("value", ["TL", "TL+TL", "TL+XX", "TL+TR+BL+BR+TL"])
def test_combination_parser_rejects_invalid_subarrays(value):
    with pytest.raises(ValueError, match="combination"):
        parse_combination_spec(value)


def test_combined_error_samples_are_repeatable_and_zero_mean_across_channels():
    resolved = resolve_study_config(preset="quick")
    first = sample_combined_errors(resolved, "medium", ("TL", "TR", "BR"), 4)
    second = sample_combined_errors(resolved, "medium", ("TL", "TR", "BR"), 4)

    assert first == second
    for sample in first:
        assert sum(sample.amplitude_errors_db.values()) == pytest.approx(0.0)
        assert sum(sample.phase_errors_deg.values()) == pytest.approx(0.0)
    json.dumps([sample.to_dict() for sample in first], allow_nan=False)


def test_scenario_builders_do_not_mutate_base_configuration():
    cfg = default_config()
    before = copy.deepcopy(cfg)
    resolved = resolve_study_config(preset="quick")

    build_multi_jammer_cases(resolved, cfg, ("TL", "BR"), "mvdr")

    assert cfg == before
```

- [ ] **Step 2: Run and verify RED**

Run: `python -W error -m pytest tests/test_study_scenarios.py -v`

Expected: import fails because `studies.scenarios` does not exist.

- [ ] **Step 3: Implement canonical combinations, IDs, and seeds**

Add to `models.py`:

```python
@dataclass(frozen=True)
class StudyCase:
    case_id: str
    stage: str
    active_antennas: tuple[str, ...]
    mode: str
    seed: int | None
    parameters: dict[str, object]
    selection_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class CombinedErrorRealization:
    level: str
    sample_index: int
    child_seed: int
    azimuth_error_deg: float
    elevation_error_deg: float
    position_errors_m: dict[str, tuple[float, float, float]]
    amplitude_errors_db: dict[str, float]
    phase_errors_deg: dict[str, float]
```

In `scenarios.py`, canonicalize the order inside each combination according to `CORNER_ORDER`, reject duplicates, preserve the caller's combination-list order, and reject duplicate canonical combinations. Use BLAKE2b over `f"{master_seed}:{namespace}:{index}"` to derive an unsigned 64-bit child seed; do not use Python's randomized `hash()`.

Make `stable_case_id()` a readable prefix plus the first 12 lowercase hex characters of SHA-256 over canonical strict JSON. Include stage, combination, mode, parameters, and seed in the payload.

- [ ] **Step 4: Implement combined errors and the three exact jammer templates**

Use `np.random.default_rng(child_seed)` per realization. Draw azimuth/elevation, per-corner XYZ, amplitude, and phase errors in a fixed documented order. Reject elevation samples outside `[0, 90]` and redraw only that elevation until valid. Subtract the active-channel mean from amplitude and phase residual maps.

Build the three exact templates from the specification: separated two-jammer, close two-jammer, and three-jammer stress. Each returned `StudyCase.parameters` contains a JSON snapshot of all interferers; modifying a returned case must not mutate `ResolvedStudyConfig`.

- [ ] **Step 5: Verify boundaries and strict JSON**

Add tests for template values, elevation redraw with a controlled RNG stub, different master seeds, stable IDs independent of dictionary insertion order, duplicate input rejection, and `allow_nan=False`. Run:

```powershell
python -W error -m pytest tests/test_study_scenarios.py tests/test_study_config.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit the scenario catalog**

```powershell
git add -- python_sim/antenna_diversity/studies/models.py python_sim/antenna_diversity/studies/config.py python_sim/antenna_diversity/studies/scenarios.py python_sim/tests/test_study_scenarios.py
git commit -m "feat: add deterministic spatial study scenarios"
```

---

### Task 3: Analytical ideal baseline and power scans

**Files:**
- Modify: `python_sim/antenna_diversity/studies/models.py`
- Create: `python_sim/antenna_diversity/studies/analytical.py`
- Create: `python_sim/tests/test_study_analytical.py`

**Interfaces:**
- Produces: `StageRecords(name: str, tables: dict[str, tuple[dict[str, object], ...]])` in `studies.models`.
- Produces: `run_baseline(resolved: ResolvedStudyConfig) -> StageRecords`.
- Uses: existing `evaluate_configuration_spatially`, `scan_interferer_directions`, and `scan_interferer_powers` without modifying them.

- [ ] **Step 1: Write failing analytical baseline tests**

```python
import copy
import json

import numpy as np

from antenna_diversity.studies.analytical import run_baseline
from antenna_diversity.studies.config import resolve_study_config


def test_baseline_emits_single_corner_references_and_requested_subarrays():
    resolved = resolve_study_config(
        preset="quick",
        cli_overrides={
            "combinations": "TL+BR,TL+TR+BR",
            "modes": ["egc", "mvdr"],
            "single_antennas": ["TL", "BR"],
        },
    )
    before = copy.deepcopy(resolved.to_dict())

    result = run_baseline(resolved)

    labels = {row["display_label"] for row in result.tables["reference"]}
    assert {"SINGLE(TL)", "SINGLE(BR)", "EGC(TL+BR)", "MVDR(TL+BR)"} <= labels
    assert {tuple(row["active_antennas"]) for row in result.tables["direction"]} == {
        ("TL", "BR"), ("TL", "TR", "BR"),
    }
    assert resolved.to_dict() == before
    json.dumps(result.to_dict(), allow_nan=False)


def test_baseline_direction_rows_have_exact_quick_grid_size():
    resolved = resolve_study_config(
        preset="quick", cli_overrides={"combinations": "TL+BR", "modes": ["mvdr"]}
    )

    result = run_baseline(resolved)

    assert len(result.tables["direction"]) == 12 * 7
    assert all(np.isfinite(row["output_sinr_db"]) for row in result.tables["direction"])
```

- [ ] **Step 2: Run and verify RED**

Run: `python -W error -m pytest tests/test_study_analytical.py -k baseline -v`

Expected: import fails because `studies.analytical` does not exist.

- [ ] **Step 3: Implement normalized long-form analytical records**

Add to `models.py`:

```python
@dataclass(frozen=True)
class StageRecords:
    name: str
    tables: dict[str, tuple[dict[str, object], ...]]

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "tables": {
            key: [dict(row) for row in rows] for key, rows in self.tables.items()
        }}
```

`run_baseline()` must:

1. deep-copy the resolved simulation for every calculation;
2. produce one no-interferer noise reference record per SINGLE corner and requested EGC/MVDR combination;
3. produce one direction row per combination, spatial mode, elevation, and azimuth using the reference single interferer power;
4. produce one power row per combination, spatial mode, and requested power at the configured reference direction;
5. flatten all 14 existing analytical metric fields into JSON-safe scalars/lists;
6. add `stage`, `case_id`, `display_label`, `active_antennas`, `num_antennas`, `mode`, axes, reference power, and maximum baseline.

Do not include arrays in rows. Convert NumPy scalars and tuples recursively.

Because the existing phone configuration intentionally validates only 2–4 active antennas, evaluate `SINGLE(corner)` through a stable two-channel carrier pair: put the requested corner first and append the first different name in `CORNER_ORDER`. Existing SINGLE weights are nonzero only at index zero, so the companion channel cannot contribute. Store the internal carrier pair as `simulation_active_antennas`, but store `active_antennas=[corner]`, `num_antennas=1`, and `display_label="SINGLE(corner)"` in the public row. Add a numerical test showing the SINGLE result is unchanged when the zero-weight companion is changed to either remaining corner.

- [ ] **Step 4: Verify ordering, SINGLE semantics, and no receiver calls**

Monkeypatch `antenna_diversity.pipeline.run_end_to_end` to fail if called. Add assertions that elevation is the outer direction-loop axis, azimuth is inner, combination/mode order matches the resolved configuration, SINGLE has exactly one active antenna, and each `case_id` is stable and unique. Run:

```powershell
python -W error -m pytest tests/test_study_analytical.py tests/test_experiments.py -v
```

Expected: all tests pass and no end-to-end receiver is called.

- [ ] **Step 5: Commit the ideal analytical stage**

```powershell
git add -- python_sim/antenna_diversity/studies/models.py python_sim/antenna_diversity/studies/analytical.py python_sim/tests/test_study_analytical.py
git commit -m "feat: add ideal spatial baseline studies"
```

---

### Task 4: Single-factor, combined-error, and multi-jammer analytical stages

**Files:**
- Modify: `python_sim/antenna_diversity/studies/analytical.py`
- Modify: `python_sim/tests/test_study_analytical.py`

**Interfaces:**
- Produces: `run_single_factor_studies(resolved) -> StageRecords`.
- Produces: `run_combined_error_studies(resolved) -> StageRecords`.
- Produces: `run_multi_jammer_studies(resolved) -> StageRecords`.
- Uses: deterministic realizations and templates from Task 2 and `evaluate_configuration_spatially()`.

- [ ] **Step 1: Write failing single-factor behavior tests**

Add tests that use one two-corner combination and a small explicit axis override:

```python
def test_single_factor_changes_assumed_values_only_and_keeps_zero_baseline():
    resolved = resolve_study_config(
        preset="quick",
        cli_overrides={
            "combinations": "TL+BR",
            "direction_errors_deg": [-1.0, 0.0, 1.0],
            "position_errors_m": [-0.001, 0.0, 0.001],
            "amplitude_errors_db": [-0.5, 0.0, 0.5],
            "phase_errors_deg": [-5.0, 0.0, 5.0],
            "loading_factors": [1e-3, 1e-2, 1e-1],
        },
    )

    records = run_single_factor_studies(resolved).tables["single_factor"]

    assert {row["factor"] for row in records} == {
        "direction_azimuth", "direction_elevation", "position",
        "amplitude", "phase", "loading",
    }
    position = [row for row in records if row["factor"] == "position"]
    assert {(row["corner"], row["axis"]) for row in position} == {
        (corner, axis) for corner in ("TL", "BR") for axis in ("x", "y", "z")
    }
    assert all(row["true_geometry_unchanged"] for row in position)
```

Also add tests that each MVDR row has the same-case EGC comparison and `mvdr_minus_egc_sinr_db`, combined-error row counts equal `3 * samples_per_level * combinations`, and multi-jammer output contains exactly the configured enabled-interferer count.

- [ ] **Step 2: Run and verify RED**

Run: `python -W error -m pytest tests/test_study_analytical.py -k "single_factor or combined_error or multi_jammer" -v`

Expected: imports fail because the three functions do not exist.

- [ ] **Step 3: Implement single-factor scans**

For every requested combination:

- direction azimuth and elevation: change only `signal.assumed_direction`;
- position: change one active corner and one assumed XYZ coordinate at a time;
- amplitude/phase: keep true channel maps at zero and change exactly one estimated active-channel value;
- loading: change only `diversity.diagonal_loading_factor`.

Evaluate EGC and MVDR for the identical copied case even when the reporting mode list contains extra modes. Store factor, signed value, absolute normalized severity, corner/axis or `null`, both SINRs, the delta, and the full MVDR metric record. Require zero on every non-loading axis and require the default loading factor in the loading axis.

- [ ] **Step 4: Implement combined-error and multi-jammer analytical stages**

Apply each `CombinedErrorRealization` to a fresh deep copy: true geometry/channel stay fixed, assumed geometry/direction and estimated channel maps change. Evaluate EGC and MVDR, retain realization values, level, sample index, child seed, both outputs, delta, and MVDR diagnostics.

For each configured multi-jammer template and combination, evaluate requested spatial modes. Store every jammer's full parameters and direction gain, enabled count, output SINR, worst gain, rank, and condition number.

- [ ] **Step 5: Verify determinism, zero-error equivalence, and full focused regression**

Add tests that repeated calls are identical; zero-error MVDR equals baseline; sampled realized errors stay copied; EGC/MVDR comparisons use the same case; and a three-jammer case with two antennas is preserved rather than rejected as a program error. Run:

```powershell
python -W error -m pytest tests/test_study_analytical.py tests/test_study_scenarios.py tests/test_spatial_analysis.py tests/test_experiments.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit the robustness analytical stages**

```powershell
git add -- python_sim/antenna_diversity/studies/analytical.py python_sim/tests/test_study_analytical.py
git commit -m "feat: add mvdr robustness study stages"
```

---

### Task 5: Stable ranking and representative-case selection

**Files:**
- Modify: `python_sim/antenna_diversity/studies/models.py`
- Create: `python_sim/antenna_diversity/studies/ranking.py`
- Create: `python_sim/tests/test_study_ranking.py`

**Interfaces:**
- Produces: `direction_separation_deg(...) -> float`.
- Produces: `summarize_and_rank(direction_rows, resolved) -> RankingResult`.
- Produces: `select_representative_cases(ranking, direction_rows, robustness_rows, multi_rows, resolved) -> SelectionResult`.
- Consumes: Task 3/4 normalized rows only; does not rerun analytical or full-chain code.

- [ ] **Step 1: Write failing mask and ranking tests**

```python
import pytest

from antenna_diversity.config import Direction
from antenna_diversity.studies.ranking import direction_separation_deg


def test_direction_separation_uses_three_dimensional_unit_vectors():
    assert direction_separation_deg(Direction(0, 90), Direction(180, 90)) == pytest.approx(0.0)
    assert direction_separation_deg(Direction(0, 0), Direction(90, 0)) == pytest.approx(90.0)


def test_exclusion_boundary_is_in_resolvable_region():
    resolved = resolved_for_test(
        combinations=(("TL", "BR"),),
        spatial_modes=("mvdr",),
        ranking_mode="mvdr",
        exclusion_angle_deg=15.0,
    )
    rows = [
        {
            "case_id": f"case-{index}",
            "active_antennas": ["TL", "BR"],
            "mode": "mvdr",
            "azimuth_deg": 30.0,
            "elevation_deg": 60.0 - separation,
            "output_sinr_db": float(index),
            "white_noise_gain": 1.0,
            "maximum_baseline_m": 0.178,
            "elevation_index": 0,
            "azimuth_index": index,
        }
        for index, separation in enumerate([14.999, 15.0, 30.0])
    ]
    result = summarize_and_rank(rows, resolved)

    assert result.summaries[0]["resolvable_count"] == 2
```

Import `resolved_for_test` from `tests.study_helpers`. The production ranking function recomputes separation from the resolved true target and row direction. With the default target `(30°, 60°)`, keeping azimuth at 30° and setting elevation to `60° - separation` creates the requested great-circle separation along one vertical plane; the mask must use this computed value, not a caller-supplied separation field.

Add a synthetic tied-data test proving the exact key order: p10 descending, median descending, white-noise gain descending, baseline descending, stable combination name ascending.

- [ ] **Step 2: Run and verify RED**

Run: `python -W error -m pytest tests/test_study_ranking.py -v`

Expected: import fails because `studies.ranking` does not exist.

- [ ] **Step 3: Implement summaries and stable ranking**

Add these exact records to `studies/models.py`:

```python
@dataclass(frozen=True)
class RankingResult:
    summaries: tuple[dict[str, object], ...]
    ranked_combinations: tuple[tuple[str, ...], ...]

    def to_dict(self) -> dict[str, object]:
        return strict_snapshot(self)


@dataclass(frozen=True)
class SelectionResult:
    cases: tuple[StudyCase, ...]
    unselected_roles: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return strict_snapshot(self)
```

Use unit vectors from existing direction conventions and `np.clip(dot, -1, 1)` before `arccos`. Summarize all directions and `separation >= exclusion_angle_deg` separately. Reject empty ranking-mode rows or an empty resolvable region with a clear `ValueError`.

Use `np.percentile(values, 10, method="linear")`. Record p10, median, minimum, maximum, counts, representative p10 direction, white-noise gain, and maximum baseline. Select the p10 direction by absolute SINR distance, then input elevation index, then azimuth index.

- [ ] **Step 4: Implement best, typical, worst, knee, stress, and manual selection**

Choose typical with index `(len(ranked) - 1) // 2`. Implement the normalized-severity knee rule exactly from the design spec, including factor/corner/axis tie order and maximum-slope fallback. If EGC/MVDR or usable curve points are absent, emit a selection status `not_selected` and reason instead of failing the whole study.

Apply every multi-jammer automatic role to the ideal ranking's best combination. For each extended combined-error level, filter to that best combination and MVDR, compute the linear median output SINR, choose the realization with minimum absolute distance, and break ties with the lowest `sample_index`. Carry the complete realization/template parameters into `StudyCase.parameters` so Task 6 can reconstruct the exact case.

Merge identical `case_id` records by appending unique selection reasons. Preserve manual cases after automatic cases. Validate that each selected case identifies an existing combination and legal simulation parameters.

- [ ] **Step 5: Verify degenerate cases and no hidden reruns**

Test one-combination best/typical/worst de-duplication, even combination counts, equal p10 direction ties, no-crossing knee fallback, insufficient knee data, missing EGC, manual-case retention, and monkeypatch analytical functions to fail if called. Run:

```powershell
python -W error -m pytest tests/test_study_ranking.py tests/test_study_analytical.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit ranking and selection**

```powershell
git add -- python_sim/antenna_diversity/studies/models.py python_sim/antenna_diversity/studies/ranking.py python_sim/tests/test_study_ranking.py
git commit -m "feat: rank arrays and select validation cases"
```

---

### Task 6: Compact RF-to-tracking validation records

**Files:**
- Modify: `python_sim/antenna_diversity/studies/models.py`
- Create: `python_sim/antenna_diversity/studies/full_chain.py`
- Create: `python_sim/tests/test_study_full_chain.py`

**Interfaces:**
- Produces: `FullChainRecords(metrics: tuple[dict, ...], cn0: tuple[dict, ...])`.
- Produces: `run_selected_full_chain(resolved: ResolvedStudyConfig, selection: SelectionResult) -> FullChainRecords`.
- Uses: `run_end_to_end()` and `evaluate_configuration_spatially()` on independent deep copies.

- [ ] **Step 1: Write failing compact-record tests with a controlled pipeline result**

```python
import json
from types import SimpleNamespace

import numpy as np

from antenna_diversity.studies.models import SelectionResult, StudyCase
from antenna_diversity.studies.full_chain import run_selected_full_chain
from tests.study_helpers import resolved_for_test


def one_selected_case() -> SelectionResult:
    case = StudyCase(
        case_id="selected-001",
        stage="full_chain",
        active_antennas=("TL", "BR"),
        mode="mvdr",
        seed=None,
        parameters={
            "interferers": [{
                "direction": {"azimuth_deg": 120.0, "elevation_deg": 10.0},
                "enable": True, "power_dbm": -90.0, "offset_hz": 250e3,
                "initial_phase_rad": 0.8, "start_ms": 0.0, "stop_ms": None,
            }],
        },
        selection_reasons=("best",),
    )
    return SelectionResult((case,), {})


def controlled_results(cfg):
    modes = cfg.diversity.modes
    branches = {}
    metrics = []
    for index, mode in enumerate(modes):
        acquired = mode != "single"
        tracking = acquired
        track = {
            "success": tracking,
            "status": "tracking" if tracking else "acquisition_failed",
            "epoch_start_samples": np.array([0, 4092]) if tracking else np.array([]),
            "cn0_db_hz": np.array([36.0, 37.0]) if tracking else np.array([]),
        }
        branches[mode] = SimpleNamespace(tracking=track)
        metrics.append({
            "mode": mode, "acquired": acquired, "acquisition_metric": 3.0,
            "tracking": tracking, "final_doppler_hz": 1500.0 if tracking else None,
            "final_cn0_db_hz": 37.0 if tracking else None,
            "target_direction_gain": 1.0,
            "interferer_direction_gains": [0.01],
            "worst_interferer_gain": 0.01,
        })
    return SimpleNamespace(config=cfg, branches=branches, metrics=metrics)


def test_full_chain_records_keep_failures_and_never_store_sample_arrays(monkeypatch):
    monkeypatch.setattr(
        "antenna_diversity.studies.full_chain.run_end_to_end",
        controlled_results,
    )
    records = run_selected_full_chain(
        resolved_for_test(combinations=(("TL", "BR"),), full_chain_run_count=1),
        one_selected_case(),
    )

    failed = next(row for row in records.metrics if row["mode"] == "single")
    assert failed["acquired"] is False
    assert failed["tracking"] is False
    assert failed["final_cn0_db_hz"] is None
    assert failed["status"] == "acquisition_failed"
    assert "scene_input_v" not in json.dumps(records.to_dict())
    json.dumps(records.to_dict(), allow_nan=False)


def test_full_chain_uses_explicit_analytic_sinr_name(monkeypatch):
    monkeypatch.setattr(
        "antenna_diversity.studies.full_chain.run_end_to_end",
        controlled_results,
    )
    records = run_selected_full_chain(
        resolved_for_test(combinations=(("TL", "BR"),), full_chain_run_count=1),
        one_selected_case(),
    )

    assert all("analytic_output_sinr_db" in row for row in records.metrics)
    assert all("output_sinr_db" not in row for row in records.metrics)
```

- [ ] **Step 2: Run and verify RED**

Run: `python -W error -m pytest tests/test_study_full_chain.py -v`

Expected: import fails because `studies.full_chain` does not exist.

- [ ] **Step 3: Implement case reconstruction and compact metric extraction**

Add this exact frozen record to `studies/models.py`:

```python
@dataclass(frozen=True)
class FullChainRecords:
    metrics: tuple[dict[str, object], ...]
    cn0: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return strict_snapshot(self)
```

For every selected unique case and `run_index in range(full_chain_run_count)`, derive the concrete seed with `derive_child_seed(study.seed, f"full-chain:{case.case_id}", run_index)`, then:

1. deep-copy the resolved simulation;
2. apply combination, interferers, assumed direction/position/calibration errors, loading, modes, and derived run seed;
3. disable standard plotting;
4. compute analytical SINR for each compatible mode on a separate copy;
5. run the sample-domain pipeline once for all requested branch modes;
6. produce one metric row per branch with case/role/seed, acquired, acquisition metric, tracking, status, final C/N0, final Doppler error, direction gains, and `analytic_output_sinr_db`;
7. produce one C/N0 row per finite epoch with time in milliseconds; failed branches have no curve rows and retain a metric-row reason.

Use `epoch_start_samples / dfe.fs_out_hz * 1000`. Validate equal one-dimensional C/N0/time arrays before flattening. Never retain `SimulationResults` after extraction.

- [ ] **Step 4: Verify real pipeline behavior on one minimal representative case**

Add one integration-marked test using default duration, one `TL+BR` case, modes SINGLE/EGC/MVDR, and one seed. Assert that the real RF/DFE pipeline returns three branch records, acquisition failures remain valid, every available C/N0 row is finite, and no raw ndarray appears in the serialized result.

Run:

```powershell
python -W error -m pytest tests/test_study_full_chain.py tests/test_end_to_end.py tests/test_tracking.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit full-chain validation**

```powershell
git add -- python_sim/antenna_diversity/studies/models.py python_sim/antenna_diversity/studies/full_chain.py python_sim/tests/test_study_full_chain.py
git commit -m "feat: validate selected spatial cases end to end"
```

---

### Task 7: Atomic artifacts, manifest, and checkpoints

**Files:**
- Create: `python_sim/antenna_diversity/studies/artifacts.py`
- Create: `python_sim/tests/test_study_artifacts.py`

**Interfaces:**
- Produces: `StudyStore.create(resolved)`, `StudyStore.resume(resolved)`, `write_stage()`, `read_stage()`, `fail_stage()`, and `complete()`.
- Consumes: `ResolvedStudyConfig` and strict-JSON-compatible stage payloads; it does not import or execute scientific stages.

- [ ] **Step 1: Write failing strict-write and resume tests**

```python
import json
from pathlib import Path

import pytest

from antenna_diversity.studies.artifacts import StudyStore, write_json_atomic


def test_strict_json_rejects_nan_without_publishing_destination(tmp_path: Path):
    destination = tmp_path / "bad.json"

    with pytest.raises(ValueError):
        write_json_atomic(destination, {"value": float("nan")})

    assert not destination.exists()


def test_resume_rejects_changed_scientific_configuration(tmp_path: Path):
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
    resolved = resolved_for_test(output_dir=tmp_path, name="artifact-test")
    store = StudyStore.create(resolved)
    store.write_stage("baseline", {"rows": [{"value": 1.0}]})

    resumed = StudyStore.resume(resolved)

    assert resumed.completed_stages == ("baseline",)
```

Import `resolved_for_test` from `tests.study_helpers`. Task 1's helper calls `ResolvedStudyConfig.from_parts(study, simulation)`, which deep-copies both inputs, validates them, recomputes the scientific snapshot and SHA-256, and is the only supported way to create a modified resolved record in tests or callers.

- [ ] **Step 2: Run and verify RED**

Run: `python -W error -m pytest tests/test_study_artifacts.py -v`

Expected: import fails because `studies.artifacts` does not exist.

- [ ] **Step 3: Implement strict atomic writers and manifest state**

`write_json_atomic()` must serialize fully with `allow_nan=False` before opening a same-directory temporary file, flush and `os.fsync()`, then `os.replace()`. `write_csv_atomic()` fixes field order supplied by the caller, rejects unknown row fields, writes UTF-8 with `newline=""`, and publishes atomically.

Manifest schema includes schema version, name, `running/completed/failed`, Git SHA, Python version, NumPy/SciPy/Matplotlib versions, UTC timestamps, configuration SHA-256, per-stage status/record count/checkpoint SHA-256/error summary, and artifact paths. Use `git rev-parse HEAD` without network access; if unavailable, store `null` plus status `git_unavailable`.

`StudyStore.create()` rejects an existing study directory. `resume()` checks schema and scientific hash, verifies each completed checkpoint's SHA-256 and strict JSON parse, and marks corrupt stages pending. Output/control fields do not affect the scientific hash.

- [ ] **Step 4: Implement checkpoint verification and confined artifact paths**

Restrict every relative destination with `candidate.resolve().is_relative_to(study_dir.resolve())`; reject absolute paths and `..` escapes. A completed checkpoint is reusable only when the file exists, strict JSON parses, the stored stage name matches, and its SHA-256 matches the manifest. `StudyStore.write_stage()` returns the checkpoint path and updates the manifest to `completed`; `fail_stage()` writes only exception type/message and UTC timestamp, never a traceback or sample arrays; `complete()` requires all seven named stages and changes the study status atomically.

- [ ] **Step 5: Verify recovery, corruption, and store-state behavior**

Add tests for existing-directory refusal, config/schema mismatch, corrupt checkpoint becoming pending, strict CSV field order, path confinement, failed-stage metadata, refusal to complete with missing stages, and completed manifest state. Run:

```powershell
python -W error -m pytest tests/test_study_artifacts.py tests/test_study_config.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit atomic artifacts and checkpoints**

```powershell
git add -- python_sim/antenna_diversity/studies/artifacts.py python_sim/tests/test_study_artifacts.py
git commit -m "feat: checkpoint reproducible spatial studies"
```

---

### Task 8: Deterministic figures and Markdown report

**Files:**
- Create: `python_sim/antenna_diversity/studies/plots.py`
- Create: `python_sim/antenna_diversity/studies/report.py`
- Create: `python_sim/tests/test_study_reporting.py`

**Interfaces:**
- Produces: `plot_study_tables(tables: Mapping[str, Sequence[Mapping]]) -> dict[str, Figure]`.
- Produces: `save_study_figures(figures, output_dir) -> dict[str, Path]`.
- Produces: `build_summary(tables: Mapping[str, Sequence[Mapping[str, object]]], manifest: Mapping[str, object], selection: Mapping[str, object]) -> dict[str, object]`.
- Produces: `render_report(summary: Mapping[str, object], manifest: Mapping[str, object], figure_paths: Mapping[str, Path], table_paths: Mapping[str, Path]) -> str`.
- Consumes: normalized checkpoint/table records only; no simulation imports in report generation.

- [ ] **Step 1: Write failing plot/report contract tests**

```python
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from antenna_diversity.studies.plots import plot_study_tables
from antenna_diversity.studies.report import build_summary, render_report
from tests.study_helpers import completed_tables_for_reporting


def completed_manifest():
    return {
        "schema_version": 1,
        "name": "report-test",
        "status": "completed",
        "git_sha": "0123456789abcdef",
        "config_sha256": "a" * 64,
    }


def test_study_plots_have_stable_keys_and_shared_heatmap_limits():
    figures = plot_study_tables(completed_tables_for_reporting())

    assert tuple(figures) == (
        "baseline_rankings", "direction_heatmaps", "power_curves",
        "robustness_curves", "combined_error_distributions",
        "multi_jammer_comparison", "full_chain_validation", "cn0_validation",
    )
    heatmaps = figures["direction_heatmaps"].axes
    limits = {image.get_clim() for axis in heatmaps for image in axis.images}
    assert len(limits) == 1


def test_report_contains_evidence_boundaries_and_reproduction_links(tmp_path: Path):
    tables = completed_tables_for_reporting()
    manifest = completed_manifest()
    selection = {
        "cases": [{"case_id": "best-001", "selection_reasons": ["best"]}],
        "unselected_roles": {"knee": "no crossing"},
    }
    figures = {"baseline_rankings": Path("figures/baseline_rankings.png")}
    table_paths = {name: Path("tables") / f"{name}.csv" for name in tables}
    summary = build_summary(tables, manifest, selection)
    report = render_report(summary, manifest, figures, table_paths)

    assert "比较性研究" in report
    assert "不代表真实 Mate 70 产品性能" in report
    assert "analytic_output_sinr_db" in report
    assert "final_cn0_db_hz" in report
    assert "未建模" in report
    assert "配置哈希" in report
    json.dumps(summary, allow_nan=False)
```

Extend `tests/study_helpers.py` with this concrete, minimal dataset builder; expand rows only when a plot's documented schema requires an additional field:

```python
def completed_tables_for_reporting() -> dict[str, tuple[dict[str, object], ...]]:
    combination = ["TL", "BR"]
    direction = tuple(
        {
            "case_id": f"direction-{azimuth}-{elevation}",
            "active_antennas": combination,
            "display_label": "MVDR(TL+BR)",
            "mode": "mvdr",
            "azimuth_deg": float(azimuth),
            "elevation_deg": float(elevation),
            "azimuth_index": az_index,
            "elevation_index": el_index,
            "output_sinr_db": float(18 + az_index + el_index),
            "interferer_gain": float(0.1 + 0.01 * az_index),
        }
        for el_index, elevation in enumerate((10.0, 20.0))
        for az_index, azimuth in enumerate((90.0, 120.0))
    )
    return {
        "baseline_rankings": ({
            "active_antennas": combination,
            "display_label": "MVDR(TL+BR)",
            "mode": "mvdr",
            "p10_resolvable_db": 18.0,
            "median_resolvable_db": 19.5,
            "minimum_all_db": 2.0,
            "white_noise_gain": 1.0,
            "maximum_baseline_m": 0.178,
            "rank": 1,
        },),
        "direction_scan": direction,
        "power_scan": tuple({
            "active_antennas": combination,
            "display_label": f"{mode.upper()}(TL+BR)",
            "mode": mode,
            "power_dbm": power,
            "output_sinr_db": 20.0 - (power + 100.0) / 5.0 + (2.0 if mode == "mvdr" else 0.0),
        } for mode in ("egc", "mvdr") for power in (-100.0, -90.0)),
        "robustness_scans": tuple({
            "active_antennas": combination,
            "factor": "phase",
            "value": value,
            "normalized_severity": abs(value) / 5.0,
            "egc_output_sinr_db": 15.0,
            "mvdr_output_sinr_db": 18.0 - abs(value) / 5.0,
            "mvdr_minus_egc_sinr_db": 3.0 - abs(value) / 5.0,
        } for value in (-5.0, 0.0, 5.0)),
        "combined_errors": tuple({
            "active_antennas": combination,
            "level": level,
            "sample_index": 0,
            "output_sinr_db": value,
        } for level, value in (("light", 18.0), ("medium", 14.0), ("heavy", 8.0))),
        "multi_jammer": ({
            "active_antennas": combination,
            "template": "two_jammer_separated",
            "mode": "mvdr",
            "enabled_interferers": 2,
            "output_sinr_db": 12.0,
            "worst_interferer_gain": 0.04,
        },),
        "full_chain_validation": tuple({
            "case_id": "best-001",
            "selection_reasons": ["best"],
            "mode": mode,
            "run_index": 0,
            "seed": 7,
            "acquired": mode != "single",
            "tracking": mode != "single",
            "status": "tracking" if mode != "single" else "acquisition_failed",
            "acquisition_metric": 3.0 if mode != "single" else 1.0,
            "analytic_output_sinr_db": 18.0 if mode == "mvdr" else 10.0,
            "final_cn0_db_hz": 37.0 if mode != "single" else None,
            "final_doppler_error_hz": 5.0 if mode != "single" else None,
        } for mode in ("single", "egc", "mvdr")),
        "full_chain_cn0": tuple({
            "case_id": "best-001",
            "mode": mode,
            "run_index": 0,
            "seed": 7,
            "epoch_index": epoch,
            "time_ms": float(epoch),
            "cn0_db_hz": 36.0 + epoch,
        } for mode in ("egc", "mvdr") for epoch in (0, 1)),
    }
```

- [ ] **Step 2: Run with Agg and warnings-as-errors to verify RED**

Run:

```powershell
$env:MPLBACKEND='Agg'
python -W error -m pytest tests/test_study_reporting.py -v
```

Expected: imports fail because `studies.plots` and `studies.report` do not exist.

- [ ] **Step 3: Implement eight fixed figures from saved rows**

Use stable combination/mode ordering and units from the schema. Ranking bars show p10 and median; heatmaps use common finite color limits and overlay target plus 15-degree boundary; power and robustness plots show EGC/MVDR and delta where available; combined errors show level distributions; multi-jammer shows template × combination; full-chain shows capture/tracking rates and scalar summaries; C/N0 shows finite epoch rows and annotates unavailable branches.

Guard contour/boxplot/legend paths so constant, empty-valid, and failure-only datasets do not emit Matplotlib warnings. Close figures in tests. Save PNG at 150 dpi with deterministic filenames.

- [ ] **Step 4: Implement evidence-driven summary and report**

Build the summary exclusively from tables and selection records. Include winning/median/losing combinations, p10/median/worst values in both regions, knee status, combined-error quantiles, multi-jammer outcomes, full-chain rates, final C/N0 summaries, Git SHA, configuration hash, and unavailable reasons.

Render sections in the approved order: overview, baseline ranking, spatial panorama, power, robustness, combined errors, multi-jammer stress, RF-to-tracking validation, conclusions/limits, reproduction. Use relative Markdown links and verify every local link target before publishing.

- [ ] **Step 5: Verify warning-free edge cases and strict outputs**

Add tests for one combination, all-tied ranking, no knee, acquisition failures only, absent C/N0 curve, constant heatmap, missing required table, shared color limits, PNG existence, relative-link validation, and strict summary JSON. Run:

```powershell
$env:MPLBACKEND='Agg'
python -W error -m pytest tests/test_study_reporting.py tests/test_plotting.py -v
```

Expected: all tests pass without resource warnings.

- [ ] **Step 6: Commit plots and report generation**

```powershell
git add -- python_sim/antenna_diversity/studies/plots.py python_sim/antenna_diversity/studies/report.py python_sim/tests/test_study_reporting.py
git commit -m "feat: report spatial study evidence"
```

---

### Task 9: Sequential research runner

**Files:**
- Create: `python_sim/antenna_diversity/studies/runner.py`
- Create: `python_sim/tests/test_study_runner.py`

**Interfaces:**
- Produces: `STAGES = ("baseline", "single_factor", "combined_error", "multi_jammer", "selection", "full_chain", "report")`.
- Produces: `run_research(resolved: ResolvedStudyConfig, resume: bool = False) -> Path` returning the completed study directory.
- Consumes: Tasks 3–8 stage functions and Task 7 `StudyStore`; it is the only module that orders stages.

- [ ] **Step 1: Write failing ordered execution and resume tests**

Create `tests/test_study_runner.py`:

```python
from pathlib import Path

import pytest

from antenna_diversity.studies.runner import STAGES, run_research


def test_runner_executes_each_stage_once_in_fixed_order(monkeypatch, tmp_path: Path):
    calls = []
    resolved = resolved_for_test(output_dir=tmp_path, name="runner-test")
    monkeypatch.setattr(
        "antenna_diversity.studies.runner._STAGE_FUNCTIONS",
        {name: lambda context, name=name: calls.append(name) or {"stage": name}
         for name in STAGES},
    )

    output = run_research(resolved)

    assert calls == list(STAGES)
    assert output == tmp_path / resolved.study.name


def test_runner_resume_skips_valid_completed_stages(monkeypatch, tmp_path: Path):
    resolved = resolved_for_test(output_dir=tmp_path, name="runner-test")
    fail_once = stage_functions_failing_once_at("combined_error")
    monkeypatch.setattr("antenna_diversity.studies.runner._STAGE_FUNCTIONS", fail_once)
    with pytest.raises(RuntimeError, match="combined_error"):
        run_research(resolved)

    calls_before = tuple(fail_once.calls)
    run_research(resolved, resume=True)

    assert fail_once.calls.count("baseline") == 1
    assert fail_once.calls.count("single_factor") == 1
    assert tuple(fail_once.calls[:len(calls_before)]) == calls_before
```

Import `resolved_for_test` from `tests.study_helpers`. Implement `stage_functions_failing_once_at()` in this test as a `dict` subclass with a `.calls` list; its target callable raises only on its first invocation and every callable returns `{"stage": name}` otherwise.

- [ ] **Step 2: Run and verify RED**

Run: `python -W error -m pytest tests/test_study_runner.py -v`

Expected: import fails because `studies.runner` does not exist.

- [ ] **Step 3: Implement explicit stage adapters and dependency contexts**

Define one adapter per stage. Baseline/single-factor/combined-error/multi-jammer call Tasks 3–4. Selection receives the four loaded analytical payloads. Full-chain receives the selection payload. Report receives every prior payload, writes CSV/figures/summary/report through Tasks 7–8, and returns only artifact relative paths and record counts for its checkpoint.

The private `_STAGE_FUNCTIONS` maps each name to `Callable[[StageContext], dict[str, object]]`. `StageContext` contains the resolved configuration, `StudyStore`, and a read-only mapping of prerequisite payloads. Do not allow a stage to access a later checkpoint.

- [ ] **Step 4: Implement failure and resume semantics**

`run_research()` creates or resumes the store, verifies completed checkpoints, loads only prerequisites in `STAGES` order, and calls pending stages. After every successful call, publish the checkpoint before advancing. On an exception, call `fail_stage(name, error)`, retain earlier checkpoints, and re-raise. After report succeeds, require all stages then call `complete()`.

- [ ] **Step 5: Verify dependency isolation and real cheap analytical path**

Add tests that a stage cannot see later payloads, exceptions set manifest `failed`, retry starts at the failed stage, corrupt checkpoints rerun, and a minimal one-combination analytical configuration reaches selection using real stage adapters while monkeypatching only full-chain/report to cheap payloads. Run:

```powershell
python -W error -m pytest tests/test_study_runner.py tests/test_study_artifacts.py tests/test_study_ranking.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit the sequential runner**

```powershell
git add -- python_sim/antenna_diversity/studies/runner.py python_sim/tests/test_study_runner.py
git commit -m "feat: orchestrate spatial research stages"
```

---

### Task 10: CLI, editable example, documentation, and full acceptance

**Files:**
- Modify: `python_sim/antenna_diversity/studies/__init__.py`
- Create: `python_sim/run_spatial_research.py`
- Create: `python_sim/studies/example_quick.json`
- Create: `python_sim/tests/test_study_cli.py`
- Modify: `python_sim/README.md`
- Modify: `python_sim/ARCHITECTURE.md`

**Interfaces:**
- Produces CLI flags: `--config`, `--preset`, `--name`, `--output-dir`, `--combinations`, `--modes`, `--ranking-mode`, `--seed`, and `--resume`.
- Produces public exports: `resolve_study_config`, `run_research`, `StudyConfig`, and `ResolvedStudyConfig`.
- Returns process exit code 0 only after a completed manifest and report exist.

- [ ] **Step 1: Write failing CLI parsing and precedence tests**

```python
from pathlib import Path

import pytest

import run_spatial_research


def test_cli_forwards_only_explicit_overrides(monkeypatch, tmp_path: Path):
    captured = {}

    def capture_resolver(config_path=None, preset=None, cli_overrides=None):
        captured["config_path"] = config_path
        captured["preset"] = preset
        captured["cli_overrides"] = dict(cli_overrides or {})
        return object()

    def capture_runner(resolved, resume=False):
        captured["resume"] = resume
        completed = tmp_path / "completed"
        completed.mkdir()
        (completed / "manifest.json").write_text(
            '{"status":"completed","artifacts":{}}', encoding="utf-8"
        )
        return completed

    monkeypatch.setattr(run_spatial_research, "resolve_study_config", capture_resolver)
    monkeypatch.setattr(run_spatial_research, "run_research", capture_runner)

    result = run_spatial_research.main([
        "--config", "studies/example_quick.json",
        "--name", "chosen-name",
        "--combinations", "TL+BR",
        "--resume",
    ])

    assert result == 0
    assert captured["cli_overrides"] == {
        "name": "chosen-name", "combinations": "TL+BR",
    }
    assert captured["resume"] is True


def test_cli_rejects_unknown_mode_before_running(monkeypatch):
    with pytest.raises(SystemExit):
        run_spatial_research.main(["--modes", "mvdr,music"])
```

- [ ] **Step 2: Run and verify RED**

Run: `python -W error -m pytest tests/test_study_cli.py -v`

Expected: import fails because `run_spatial_research.py` does not exist.

- [ ] **Step 3: Implement CLI and public exports**

`main(argv: Sequence[str] | None = None) -> int` uses `argparse`. Leave unspecified options as `None` and omit them from the CLI override mapping so JSON and presets remain effective. Split `--modes` on commas, pass combination strings unchanged to the resolver, call `run_research`, then print the absolute manifest/report/table/figure paths from the completed manifest. Do not open GUI windows.

Catch only configuration errors at the CLI boundary, print a concise message through `parser.error`, and let unexpected runtime exceptions preserve their traceback after the manifest records failure.

- [ ] **Step 4: Add a small editable example and update documentation**

`studies/example_quick.json` selects quick, `TL+BR` and `TL+TR+BR`, EGC/MVDR analytical modes, SINGLE/EGC/MVDR full-chain modes, and writes beneath `results/studies/example-quick`. Keep every configured direction, power, error level, frequency, seed, and output path visible.

README must show all/single/list commands, JSON precedence, resume, result tree, metric distinctions, and the ideal-model warning. ARCHITECTURE must show the studies layer above existing analytical and full-chain paths and state that reporting never reruns simulations.

- [ ] **Step 5: Run a real minimal study through every stage**

Create a temporary test configuration inside pytest with one `TL+BR` combination, one direction point away from the exclusion region, two power points, three values per error axis, one combined-error sample per level, one full-chain seed, and automatic role `best`. Invoke `main()` with Agg and assert:

- exit code 0;
- manifest status `completed`;
- strict JSON for manifest/config/summary/checkpoints;
- all eight CSV tables (`baseline_rankings`, `direction_scan`, `power_scan`, `robustness_scans`, `combined_errors`, `multi_jammer`, `full_chain_validation`, and `full_chain_cn0`) exist and are nonempty where scientifically applicable;
- eight PNGs and report exist;
- report local links resolve;
- full-chain table includes SINGLE/EGC/MVDR and final C/N0 is either finite or `null` with a status.

Run:

```powershell
$env:MPLBACKEND='Agg'
python -W error -m pytest tests/test_study_cli.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Run focused study and complete regression gates**

Run:

```powershell
$env:MPLBACKEND='Agg'
python -W error -m pytest tests/test_study_config.py tests/test_study_scenarios.py tests/test_study_analytical.py tests/test_study_ranking.py tests/test_study_full_chain.py tests/test_study_artifacts.py tests/test_study_reporting.py tests/test_study_cli.py -v
python -W error -m pytest -q
python -m compileall -q antenna_diversity run_spatial_research.py
git diff --check
```

Expected: every command exits 0 with no warnings or whitespace errors.

- [ ] **Step 7: Commit the executable research suite**

```powershell
git add -- python_sim/antenna_diversity/studies/__init__.py python_sim/run_spatial_research.py python_sim/studies/example_quick.json python_sim/tests/test_study_cli.py python_sim/README.md python_sim/ARCHITECTURE.md
git commit -m "feat: run reproducible spatial research reports"
```

---

## Final Verification

- [ ] Run the complete suite once more from a clean process:

```powershell
cd F:\antenna_diversity\python_sim
$env:MPLBACKEND='Agg'
python -W error -m pytest -q
python -m compileall -q antenna_diversity run_spatial_research.py
```

- [ ] Run a CLI smoke study from the editable example after overriding it to a temporary output directory and a minimal one-seed/full-chain role set; verify completion without committing generated `results/`.

- [ ] Inspect the generated ranking, heatmap, robustness, multi-jammer, full-chain, and C/N0 PNGs at original resolution; verify readable labels, shared heatmap limits, no clipping, and no misleading missing-data lines.

- [ ] Validate every generated JSON with `json.load`, every CSV with `csv.DictReader`, and every local Markdown link with `Path.exists()`.

- [ ] Confirm `git status --short` contains no generated results, caches, or unrelated files; run `git diff --check` and review the complete branch diff against the design specification.
