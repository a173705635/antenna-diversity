# Pre-Acquisition Blind Nulling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a gated `blind_null` DFE branch that detects one dominant correlated interferer, forms target-steering-free projection weights, and validates recovery through GNSS acquisition and tracking.

**Architecture:** A new `diversity/blind_null.py` module owns covariance validation, normalized-correlation detection, principal-interference projection, and safe reference-channel bypass. The existing pipeline explicitly dispatches `blind_null` without passing desired steering, then reuses the common acquisition, tracking, metrics, and plotting chain. Analytical studies continue to exclude the sample-dependent mode while selected full-chain studies accept it and record analytic SINR as unavailable.

**Tech Stack:** Python 3.11+, NumPy 2.x, Matplotlib 3.9+, pytest 9.x

**Spec:** `docs/superpowers/specs/2026-08-27-pre-acquisition-blind-nulling-design.md`

## Global Constraints

- Support exactly 2–4 DFE channels in the phone-array use case and at least 2 channels in the core blind combiner.
- V1 estimates one covariance matrix and one fixed weight vector from the complete DFE sample block.
- Blind weight generation must not accept desired/assumed target steering, target direction, PRN, acquisition, or tracking inputs.
- Default detector threshold is exactly `6.0` dB; default reference channel is `0`; default minimum reference projection power is exactly `0.01`.
- Below-threshold, zero-power, degenerate-vector, and unsafe-projection cases use an exact reference-channel bypass and record a reason.
- Nonfinite samples, invalid shapes, invalid configuration, and out-of-range reference channels raise `ValueError`; no invalid call silently falls back.
- `blind_null` is valid in end-to-end/full-chain sample paths and invalid in analytical `spatial_modes`.
- No new runtime dependency is permitted.
- Existing `single`, `egc`, and `mvdr` numerical behavior must remain unchanged.
- Truth may be used after combining for evaluation only and must not influence blind weights.

---

## File Structure

- Create `python_sim/antenna_diversity/diversity/blind_null.py`: validate samples/configuration, estimate covariance diagnostics, detect a dominant correlated mode, form projection weights, and combine or bypass.
- Modify `python_sim/antenna_diversity/config.py`: define `BlindNullConfig`, nest it in `DiversityConfig`, validate fields and allowed diversity modes.
- Modify `python_sim/antenna_diversity/diversity/__init__.py`: export the blind combiner public entry point.
- Modify `python_sim/antenna_diversity/pipeline.py`: dispatch the blind branch without desired steering and attach truth-only evaluation metrics afterward.
- Modify `python_sim/antenna_diversity/analysis/plotting.py`: derive subplot count/order from available supported branches and include `blind_null`.
- Modify `python_sim/antenna_diversity/studies/models.py`: separate analytical-mode and full-chain-mode validation.
- Modify `python_sim/antenna_diversity/studies/config.py`: include `blind_null` in default full-chain modes.
- Modify `python_sim/antenna_diversity/studies/full_chain.py`: avoid analytical evaluation for `blind_null` and store `analytic_output_sinr_db=None`.
- Modify `python_sim/antenna_diversity/studies/plots.py`: include blind results in full-chain and C/N₀ figures while leaving unavailable analytical bars absent.
- Modify the two example study JSON files: show `blind_null` in `full_chain_modes`, not `spatial_modes`.
- Create `python_sim/tests/test_blind_null.py`: focused mathematical, safety, and phase-offset tests.
- Modify existing config, pipeline, plotting, Monte Carlo, and study tests only where the fourth branch changes an explicit contract.
- Modify `python_sim/README.md`, `python_sim/DESIGN.md`, `python_sim/ARCHITECTURE.md`, and `PROJECT_STATUS.md`: document the implemented algorithm, evidence boundary, phase requirement, and project progress.

---

### Task 1: Blind Detector and Projection Combiner

**Files:**
- Create: `python_sim/antenna_diversity/diversity/blind_null.py`
- Modify: `python_sim/antenna_diversity/config.py`
- Modify: `python_sim/antenna_diversity/diversity/__init__.py`
- Create: `python_sim/tests/test_blind_null.py`
- Modify: `python_sim/tests/test_config_numeric.py`

**Interfaces:**
- Produces: `BlindNullConfig(detection_threshold_db: float = 6.0, reference_channel: int = 0, min_reference_projection_power: float = 0.01)`.
- Produces: `combine_blind_null(x: ArrayLike, cfg: BlindNullConfig) -> tuple[NDArray[np.complex128], dict[str, Any]]`.
- Produces info keys specified in design sections 4–7, including `weights`, `sample_covariance`, `interference_detected`, `nulling_enabled`, and `bypass_reason`.
- Consumes only a channels-by-samples array and `BlindNullConfig`; no target-steering argument exists.

- [ ] **Step 1: Write failing default/configuration tests**

Add assertions to `test_config_numeric.py` and parameterized invalid cases:

```python
def test_blind_null_defaults_are_explicit():
    cfg = mate70_default_config()
    assert cfg.diversity.blind_null.detection_threshold_db == 6.0
    assert cfg.diversity.blind_null.reference_channel == 0
    assert cfg.diversity.blind_null.min_reference_projection_power == 0.01


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("detection_threshold_db", np.nan, "detection threshold"),
        ("reference_channel", True, "reference channel"),
        ("reference_channel", -1, "reference channel"),
        ("min_reference_projection_power", 0.0, "projection power"),
        ("min_reference_projection_power", 1.1, "projection power"),
    ],
)
def test_blind_null_config_rejects_invalid_values(field, value, message):
    cfg = mate70_default_config()
    setattr(cfg.diversity.blind_null, field, value)
    with pytest.raises(ValueError, match=message):
        cfg.validate()
```

- [ ] **Step 2: Run the configuration tests and verify failure**

Run:

```powershell
cd F:\antenna_diversity\python_sim
python -W error -m pytest tests/test_config_numeric.py -q
```

Expected: FAIL because `DiversityConfig` has no `blind_null` field.

- [ ] **Step 3: Add the nested configuration and validation**

Add before `DiversityConfig`:

```python
@dataclass
class BlindNullConfig:
    detection_threshold_db: float = 6.0
    reference_channel: int = 0
    min_reference_projection_power: float = 0.01


@dataclass
class DiversityConfig:
    modes: tuple[str, ...] = ("single", "egc", "mvdr")
    diagonal_loading_factor: float = 1e-2
    blind_null: BlindNullConfig = field(default_factory=BlindNullConfig)
```

In `SimulationConfig.validate()`, reject booleans as integers, nonfinite scalars, a negative reference index, and projection power outside `(0, 1]`. Keep the three-mode default until Task 2 so this task remains independently runnable.

- [ ] **Step 4: Run the configuration tests and verify pass**

Run the command from Step 2.

Expected: PASS.

- [ ] **Step 5: Write failing blind-combiner tests**

Create `test_blind_null.py` with deterministic helpers and these contracts:

```python
def complex_noise(rng, shape):
    return (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    ) / np.sqrt(2.0)


def test_independent_noise_bypasses_exact_reference_channel():
    rng = np.random.default_rng(120)
    x = complex_noise(rng, (3, 40_000))
    cfg = BlindNullConfig(detection_threshold_db=6.0, reference_channel=1)
    y, info = combine_blind_null(x, cfg)
    np.testing.assert_array_equal(y, x[1])
    np.testing.assert_array_equal(info["weights"], [0.0, 1.0, 0.0])
    assert not info["interference_detected"]
    assert not info["nulling_enabled"]
    assert info["bypass_reason"] == "no_interference_detected"


@pytest.mark.parametrize("channels", [2, 3, 4])
def test_dominant_correlated_interferer_is_detected_and_nulled(channels):
    rng = np.random.default_rng(121 + channels)
    snapshots = 20_000
    steering = np.exp(1j * np.arange(channels) * 0.47)
    jammer = complex_noise(rng, snapshots)
    noise = 0.05 * complex_noise(rng, (channels, snapshots))
    x = 20.0 * steering[:, None] * jammer[None, :] + noise
    y, info = combine_blind_null(x, BlindNullConfig())
    assert info["interference_detected"]
    assert info["nulling_enabled"]
    assert info["detection_score_db"] >= 6.0
    assert np.vdot(info["weights"], np.eye(channels)[:, 0]) == pytest.approx(1.0)
    assert abs(np.vdot(info["weights"], info["estimated_interference_vector"])) < 1e-10
    assert np.mean(abs(y) ** 2) < 0.01 * np.mean(abs(x[0]) ** 2)


def test_fixed_channel_phase_offsets_are_observed_from_samples():
    rng = np.random.default_rng(130)
    phase = np.deg2rad([0.0, 37.0, -81.0, 145.0])
    steering = np.exp(1j * phase)
    jammer = complex_noise(rng, 30_000)
    x = 50.0 * steering[:, None] * jammer[None, :] + 0.02 * complex_noise(rng, (4, 30_000))
    _, info = combine_blind_null(x, BlindNullConfig())
    assert info["nulling_enabled"]
    assert abs(np.vdot(info["weights"], steering)) < 1e-3


def test_unsafe_reference_projection_bypasses_without_large_weights():
    rng = np.random.default_rng(131)
    jammer = complex_noise(rng, 20_000)
    x = np.vstack([
        100.0 * jammer,
        0.01 * jammer + 1e-8 * complex_noise(rng, 20_000),
    ])
    y, info = combine_blind_null(
        x,
        BlindNullConfig(reference_channel=0, min_reference_projection_power=0.01),
    )
    np.testing.assert_array_equal(y, x[0])
    assert not info["nulling_enabled"]
    assert info["bypass_reason"] == "unsafe_reference_projection"
    assert np.linalg.norm(info["weights"]) == pytest.approx(1.0)
```

Add the exact invalid-input cases:

```python
@pytest.mark.parametrize(
    ("samples", "message"),
    [
        (np.ones(8, dtype=np.complex128), "two-dimensional"),
        (np.ones((1, 8), dtype=np.complex128), "at least two"),
        (np.empty((2, 0), dtype=np.complex128), "nonempty"),
        (
            np.array([[1.0 + 0j, np.nan + 0j], [1.0 + 0j, 1.0 + 0j]]),
            "finite",
        ),
    ],
)
def test_blind_null_rejects_invalid_sample_matrices(samples, message):
    with pytest.raises(ValueError, match=message):
        combine_blind_null(samples, BlindNullConfig())


def test_blind_null_rejects_out_of_range_reference_channel():
    with pytest.raises(ValueError, match="reference channel"):
        combine_blind_null(
            np.ones((2, 8), dtype=np.complex128),
            BlindNullConfig(reference_channel=2),
        )
```

- [ ] **Step 6: Run the blind-combiner tests and verify failure**

Run:

```powershell
python -W error -m pytest tests/test_blind_null.py -q
```

Expected: collection FAIL because `blind_null.py` and `combine_blind_null` do not exist.

- [ ] **Step 7: Implement covariance diagnostics, gating, projection, and bypass**

Implement the public function with focused private helpers. The central numerical path must be equivalent to:

```python
channels = _finite_channels(x)
_validate_blind_config(cfg, channels.shape[0])
covariance = channels @ channels.conj().T / channels.shape[1]
covariance = (covariance + covariance.conj().T) / 2.0
channel_powers = np.maximum(np.real(np.diag(covariance)), 0.0)

if np.all(channel_powers == 0.0):
    return _bypass(channels, cfg, covariance, "zero_input_power", diagnostics)

scales = np.sqrt(channel_powers)
correlation = covariance / np.outer(scales, scales)
correlation = (correlation + correlation.conj().T) / 2.0
eigenvalues, eigenvectors = np.linalg.eigh(correlation)
eigenvalues = np.maximum(eigenvalues, 0.0)
residual = float(np.mean(eigenvalues[:-1]))
score_db = (
    np.inf
    if residual == 0.0 and eigenvalues[-1] > 0.0
    else 10.0 * np.log10(eigenvalues[-1] / residual)
)

mapped = scales * eigenvectors[:, -1]
interference_vector = mapped / np.linalg.norm(mapped)
q = np.eye(channels.shape[0], dtype=np.complex128)[:, cfg.reference_channel]

if score_db < cfg.detection_threshold_db:
    return _bypass(channels, cfg, covariance, "no_interference_detected", diagnostics)

projector = np.eye(channels.shape[0]) - np.outer(
    interference_vector, interference_vector.conj()
)
projected_reference = projector @ q
gamma = float(np.vdot(q, projected_reference).real)
if gamma < cfg.min_reference_projection_power:
    return _bypass(channels, cfg, covariance, "unsafe_reference_projection", diagnostics)

weights = projected_reference / gamma
combined = weights.conj() @ channels
```

Guard the normalization and division paths so a zero scale or vector produces the specified legal bypass rather than a warning. Compute maximum pairwise coherence from off-diagonal absolute correlation values. Return fresh finite complex arrays and all fields from the spec.

- [ ] **Step 8: Run focused tests and the existing combiner regression**

Run:

```powershell
python -W error -m pytest tests/test_blind_null.py tests/test_diversity.py tests/test_config_numeric.py -q
```

Expected: PASS with no warnings.

- [ ] **Step 9: Commit Task 1**

```powershell
git add -- python_sim/antenna_diversity/config.py python_sim/antenna_diversity/diversity/__init__.py python_sim/antenna_diversity/diversity/blind_null.py python_sim/tests/test_blind_null.py python_sim/tests/test_config_numeric.py
git commit -m "feat: add blind interference nulling core"
```

---

### Task 2: End-to-End Fourth Branch

**Files:**
- Modify: `python_sim/antenna_diversity/config.py`
- Modify: `python_sim/antenna_diversity/pipeline.py`
- Modify: `python_sim/tests/test_end_to_end.py`
- Modify: `python_sim/tests/test_monte_carlo.py`
- Modify: `python_sim/tests/test_config_numeric.py`

**Interfaces:**
- Consumes: `combine_blind_null(x, cfg)` from Task 1.
- Produces: default `DiversityConfig.modes == ("single", "egc", "mvdr", "blind_null")`.
- Produces: `SimulationResults.branches["blind_null"]` with the same ACQ/TRK lifecycle as other branches.

- [ ] **Step 1: Write failing default-branch and validation tests**

Change the default assertion to:

```python
assert cfg.diversity.modes == ("single", "egc", "mvdr", "blind_null")
```

Add invalid-mode validation:

```python
def test_unknown_diversity_mode_is_rejected_before_pipeline_dispatch():
    cfg = mate70_default_config()
    cfg.diversity.modes = ("single", "music")
    with pytest.raises(ValueError, match="diversity.modes"):
        cfg.validate()
```

- [ ] **Step 2: Write failing end-to-end blind-branch tests**

Extend the legacy regression and add target-assumption independence:

```python
def test_legacy_pipeline_blind_null_recovers_before_acquisition():
    cfg = legacy_ula_config()
    cfg.plot.enable = False
    results = run_end_to_end(cfg)
    assert not results.branches["single"].acquisition["success"]
    blind = results.branches["blind_null"]
    assert blind.combining["interference_detected"]
    assert blind.combining["nulling_enabled"]
    assert blind.acquisition["success"]
    assert blind.tracking["success"]


def test_blind_branch_does_not_depend_on_assumed_target_steering():
    first_cfg = mate70_default_config()
    first_cfg.diversity.modes = ("blind_null",)
    first_cfg.plot.enable = False
    second_cfg = copy.deepcopy(first_cfg)
    second_cfg.signal.assumed_direction = Direction(300.0, 5.0)
    first = run_end_to_end(first_cfg).branches["blind_null"]
    second = run_end_to_end(second_cfg).branches["blind_null"]
    np.testing.assert_array_equal(first.combined, second.combined)
    np.testing.assert_array_equal(
        first.combining["weights"], second.combining["weights"]
    )
```

Change explicit branch-set assertions in end-to-end and Monte Carlo tests to:

```python
expected_modes = {"single", "egc", "mvdr", "blind_null"}
assert set(results.branches) == expected_modes
assert set(result.capture_rate) == expected_modes
assert set(result.tracking_rate) == expected_modes
```

- [ ] **Step 3: Run focused tests and verify failure**

Run:

```powershell
python -W error -m pytest tests/test_end_to_end.py tests/test_monte_carlo.py tests/test_config_numeric.py -q
```

Expected: FAIL because the default modes and pipeline dispatcher do not yet include `blind_null`.

- [ ] **Step 4: Add mode validation and explicit pipeline dispatch**

Change the default modes and validate a nonempty unique tuple drawn from the four allowed names. In `pipeline.py`, use:

```python
normalized_mode = mode.lower()
if normalized_mode == "blind_null":
    combined, combining_info = combine_blind_null(
        dfe_channels,
        cfg.diversity.blind_null,
    )
else:
    combined, combining_info = combine(
        dfe_channels,
        desired_steering,
        normalized_mode,
        cfg.diversity,
    )
```

Keep truth-based target/interferer gain calculation after this dispatch so all branches receive comparable evaluation fields. Store results under `normalized_mode`.

- [ ] **Step 5: Run focused integration tests**

Run the command from Step 3.

Expected: PASS. If the historical legacy fixture does not recover with the blind branch, add a deterministic dedicated single-jammer fixture by changing only jammer angle/power and acquisition threshold; do not pass truth into the algorithm or weaken the mathematical unit tests.

- [ ] **Step 6: Run core receiver regression**

Run:

```powershell
python -W error -m pytest tests/test_scenario.py tests/test_rf.py tests/test_dfe.py tests/test_diversity.py tests/test_acquisition.py tests/test_tracking.py tests/test_end_to_end.py -q
```

Expected: PASS and unchanged assertions for the original three branches.

- [ ] **Step 7: Commit Task 2**

```powershell
git add -- python_sim/antenna_diversity/config.py python_sim/antenna_diversity/pipeline.py python_sim/tests/test_config_numeric.py python_sim/tests/test_end_to_end.py python_sim/tests/test_monte_carlo.py
git commit -m "feat: integrate blind nulling before acquisition"
```

---

### Task 3: Four-Branch Diagnostic Plots

**Files:**
- Modify: `python_sim/antenna_diversity/analysis/plotting.py`
- Modify: `python_sim/tests/test_plotting.py`

**Interfaces:**
- Consumes: a `SimulationResults.branches` mapping containing any nonempty subset of supported modes.
- Produces: combined-spectrum and phone-response figures with one panel per present branch in stable order `single`, `egc`, `mvdr`, `blind_null`.

- [ ] **Step 1: Write failing dynamic-layout tests**

Update the default plot expectations and add a three-branch compatibility case:

```python
def test_combined_spectra_show_four_modes_in_stable_order():
    results = run_end_to_end(mate70_default_config())
    figure = plot_combined_spectra(results)
    titles = [axis.get_title() for axis in figure.axes if axis.get_ylabel() == "Magnitude (dB)"]
    assert titles == [
        "SINGLE combined output",
        "EGC combined output",
        "MVDR combined output",
        "BLIND_NULL combined output",
    ]


def test_combined_spectra_accept_original_three_mode_configuration():
    cfg = mate70_default_config()
    cfg.diversity.modes = ("single", "egc", "mvdr")
    results = run_end_to_end(cfg)
    figure = plot_combined_spectra(results)
    data_axes = [axis for axis in figure.axes if axis.get_ylabel() == "Magnitude (dB)"]
    assert len(data_axes) == 3
```

Change phone-array heatmap tests to expect four data axes by default and stable fourth title `BLIND_NULL array response`. Replace the old missing-required-branch rejection test with a test that a nonempty supported subset renders successfully.

- [ ] **Step 2: Run plotting tests and verify failure**

Run:

```powershell
$env:MPLBACKEND='Agg'
python -W error -m pytest tests/test_plotting.py -q
```

Expected: FAIL because combined-spectrum and phone-response functions still require exactly three modes.

- [ ] **Step 3: Implement stable dynamic branch ordering and subplot counts**

Add:

```python
_STANDARD_BRANCH_ORDER = ("single", "egc", "mvdr", "blind_null")


def _ordered_branch_names(results: SimulationResults) -> tuple[str, ...]:
    if not results.branches:
        raise ValueError("results must contain at least one combined branch")
    ordered = tuple(
        mode for mode in _STANDARD_BRANCH_ORDER if mode in results.branches
    )
    extras = tuple(mode for mode in results.branches if mode not in ordered)
    return ordered + extras
```

Use this helper in `plot_combined_spectra()` and `_plot_phone_array_response()`. Create spectrum axes with `squeeze=False`, height proportional to branch count, and iterate over the actual ordered modes. Keep acquisition, tracking, and C/N₀ code dynamic; only add regression assertions showing that the blind branch appears.

- [ ] **Step 4: Run plotting and end-to-end tests**

Run:

```powershell
python -W error -m pytest tests/test_plotting.py tests/test_end_to_end.py -q
```

Expected: PASS without Matplotlib warnings.

- [ ] **Step 5: Commit Task 3**

```powershell
git add -- python_sim/antenna_diversity/analysis/plotting.py python_sim/tests/test_plotting.py
git commit -m "feat: plot blind nulling receiver results"
```

---

### Task 4: Seven-Stage Full-Chain Study Support

**Files:**
- Modify: `python_sim/antenna_diversity/studies/models.py`
- Modify: `python_sim/antenna_diversity/studies/config.py`
- Modify: `python_sim/antenna_diversity/studies/full_chain.py`
- Modify: `python_sim/antenna_diversity/studies/plots.py`
- Modify: `python_sim/studies/example_quick.json`
- Modify: `python_sim/antenna_diversity/studies/example_antenna_realism.json`
- Modify: `python_sim/tests/test_study_config.py`
- Modify: `python_sim/tests/test_study_full_chain.py`
- Modify: `python_sim/tests/test_study_reporting.py`
- Modify: `python_sim/tests/test_study_cli.py`

**Interfaces:**
- Consumes: pipeline support for `simulation.diversity.modes=(..., "blind_null")`.
- Produces: `StudyConfig.full_chain_modes` accepting `blind_null` while `StudyConfig.spatial_modes` rejects it contextually.
- Produces: full-chain metric rows with `analytic_output_sinr_db: float | None`; blind rows always use `None`.

- [ ] **Step 1: Write failing study-mode validation tests**

Add:

```python
def test_blind_null_is_full_chain_only():
    resolved = resolve_study_config(
        preset="quick",
        cli_overrides={"full_chain_modes": ["single", "blind_null"]},
    )
    assert resolved.study.full_chain_modes == ("single", "blind_null")

    with pytest.raises(ValueError, match="blind_null.*full-chain"):
        resolve_study_config(
            preset="quick",
            cli_overrides={"spatial_modes": ["mvdr", "blind_null"]},
        )
```

Update preset expectations so `full_chain_modes` contains four modes while `spatial_modes` remains `("egc", "mvdr")`.

- [ ] **Step 2: Write failing full-chain unavailable-analytic test**

In `test_study_full_chain.py`, configure `full_chain_modes=("mvdr", "blind_null")`, monkeypatch `evaluate_configuration_spatially`, and assert it is called only for `mvdr`:

```python
assert analytic_calls == ["mvdr"]
rows = {row["mode"]: row for row in records.metrics}
assert isinstance(rows["mvdr"]["analytic_output_sinr_db"], float)
assert rows["blind_null"]["analytic_output_sinr_db"] is None
```

Add this reporting test so unavailable analytical science is not rendered as zero:

```python
def test_full_chain_plot_includes_blind_rates_without_fake_analytic_bar():
    tables = completed_tables_for_reporting()
    tables["full_chain_validation"] += ({
        "case_id": "best-001",
        "selection_reasons": ["best"],
        "mode": "blind_null",
        "run_index": 0,
        "seed": 7,
        "acquired": True,
        "tracking": True,
        "status": "tracking",
        "acquisition_metric": 3.2,
        "analytic_output_sinr_db": None,
        "final_cn0_db_hz": 36.5,
        "final_doppler_error_hz": 4.0,
    },)
    figure = plot_study_tables(tables)["full_chain_validation"]
    rate_labels = [tick.get_text() for tick in figure.axes[0].get_xticklabels()]
    analytic_labels = [tick.get_text() for tick in figure.axes[1].get_xticklabels()]
    assert "BLIND_NULL" in rate_labels
    assert "BLIND_NULL" not in analytic_labels
```

- [ ] **Step 3: Run focused study tests and verify failure**

Run:

```powershell
$env:MPLBACKEND='Agg'
python -W error -m pytest tests/test_study_config.py tests/test_study_cli.py tests/test_study_full_chain.py tests/test_study_reporting.py -q
```

Expected: FAIL because allowed modes and analytic evaluation still assume all full-chain modes have analytical weights.

- [ ] **Step 4: Separate analytical and full-chain mode validation**

In `validate_study_config()` use:

```python
analytical_modes = {"single", "egc", "mvdr"}
full_chain_modes = analytical_modes | {"blind_null"}
if "blind_null" in study.spatial_modes:
    raise ValueError(
        "blind_null requires DFE samples and is valid only in full-chain modes"
    )
if not study.spatial_modes or set(study.spatial_modes) - analytical_modes:
    raise ValueError("study.spatial_modes contains an unknown mode")
if not study.full_chain_modes or set(study.full_chain_modes) - full_chain_modes:
    raise ValueError("study.full_chain_modes contains an unknown mode")
```

Add `blind_null` to preset `full_chain_modes` and both example JSON files. Do not add it to `spatial_modes` or `ranking_mode`.

- [ ] **Step 5: Preserve unavailable analytical SINR in full-chain rows**

Replace the comprehension in `run_selected_full_chain()` with explicit dispatch:

```python
analytic_sinr: dict[str, float | None] = {}
for mode in simulation.diversity.modes:
    if mode == "blind_null":
        analytic_sinr[mode] = None
    else:
        analytic_sinr[mode] = float(
            evaluate_configuration_spatially(
                copy.deepcopy(simulation), mode
            ).metrics["output_sinr_db"]
        )
```

Change `_metric_row()` to retain `None` rather than call `float(None)`. Extend `_MODES` in study plots to `("single", "egc", "mvdr", "blind_null")`; the existing finite-value filter must continue omitting the blind analytic bar.

- [ ] **Step 6: Run focused study tests**

Run the command from Step 3.

Expected: PASS with strict JSON accepting `None` and without fabricated zero bars.

- [ ] **Step 7: Commit Task 4**

```powershell
git add -- python_sim/antenna_diversity/studies/models.py python_sim/antenna_diversity/studies/config.py python_sim/antenna_diversity/studies/full_chain.py python_sim/antenna_diversity/studies/plots.py python_sim/studies/example_quick.json python_sim/antenna_diversity/studies/example_antenna_realism.json python_sim/tests/test_study_config.py python_sim/tests/test_study_cli.py python_sim/tests/test_study_full_chain.py python_sim/tests/test_study_reporting.py
git commit -m "feat: add blind nulling to full-chain studies"
```

---

### Task 5: Documentation, Status, and Complete Verification

**Files:**
- Modify: `python_sim/README.md`
- Modify: `python_sim/DESIGN.md`
- Modify: `python_sim/ARCHITECTURE.md`
- Modify: `PROJECT_STATUS.md`

**Interfaces:**
- Consumes: the verified implementation and exact field names from Tasks 1–4.
- Produces: user-facing configuration instructions, algorithm/evidence boundaries, phase-stability statement, and an updated project-stage record.

- [ ] **Step 1: Update README usage and output descriptions**

Document the four default branches and add this runnable configuration example:

```python
cfg = default_config()
cfg.diversity.modes = ("single", "egc", "mvdr", "blind_null")
cfg.diversity.blind_null.detection_threshold_db = 6.0
cfg.diversity.blind_null.reference_channel = 0
cfg.diversity.blind_null.min_reference_projection_power = 0.01
result = run_end_to_end(cfg)
blind = result.branches["blind_null"]
print(blind.combining["detection_score_db"])
print(blind.combining["nulling_enabled"])
print(blind.combining["bypass_reason"])
```

State that fixed inter-channel phase offsets are observed by the covariance, while phase drift during the estimate/use window reduces null depth.

- [ ] **Step 2: Update DESIGN and ARCHITECTURE**

Add the exact detector and projection equations from the spec, the target-free function boundary, full-block fixed-weight behavior, bypass reasons, and the analytical/full-chain distinction. Update architecture paths from three to four branches without implying that blind nulling protects an unknown target.

- [ ] **Step 3: Update PROJECT_STATUS after evidence passes**

Change the blind-interference stage from `下一阶段` to `已完成 V1` only after Tasks 1–4 tests pass. Record:

- one-dominant-interferer scope;
- target-steering-free weight formation;
- detector-gated reference bypass;
- fixed-relative-phase assumption;
- next work: sliding covariance, phase drift/calibration tolerance, and multi-interferer subspace estimation.

- [ ] **Step 4: Run the complete warning-clean test suite**

Run:

```powershell
cd F:\antenna_diversity\python_sim
$env:MPLBACKEND='Agg'
python -W error -m pytest -q
```

Expected: all tests PASS, no warnings, exit code 0.

- [ ] **Step 5: Run the main end-to-end smoke test**

Run:

```powershell
python -W error run_end_to_end.py
```

Expected: exit code 0, four branch rows are printed, and standard figures are generated without opening an interactive GUI under `Agg`.

- [ ] **Step 6: Inspect scientific evidence from the smoke result**

Run a short Python command or focused test that prints these blind fields:

```text
detection_score_db
interference_detected
nulling_enabled
bypass_reason
target_direction_gain
worst_interferer_gain
acquisition success/metric
tracking success/final C/N₀
```

Confirm the representative acceptance case has SINGLE acquisition failure and blind acquisition/tracking success. If it does not, report the limitation and keep PROJECT_STATUS at `进行中`; do not label V1 complete merely because unit tests pass.

- [ ] **Step 7: Check documentation and repository diff**

Run:

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors; only intended implementation, tests, examples, and documentation are modified. Preserve pre-existing untracked `.idea`, requirements, memory, or user files unless explicitly part of this plan.

- [ ] **Step 8: Commit Task 5**

```powershell
git add -- python_sim/README.md python_sim/DESIGN.md python_sim/ARCHITECTURE.md PROJECT_STATUS.md
git commit -m "docs: document pre-acquisition blind nulling"
```

- [ ] **Step 9: Record final verification evidence**

Run:

```powershell
git log -6 --oneline --decorate
git status --short --branch
```

Report the exact test count, smoke-test result, representative SINGLE versus blind acquisition/tracking outcome, current branch, commits created, and any remaining untracked user files. Do not push or open a pull request without a separate user request.
