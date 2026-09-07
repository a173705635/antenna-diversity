# GNSS Multipath Detection and Suppression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add configurable static GNSS multipath, receive-data-only delay/DOA detection, and LOS-preserving LCMV suppression to the simplified Python simulation.

**Architecture:** Generate each reflected path as a delayed, attenuated copy of the desired RF waveform with its own antenna manifold, then pass the summed scene through the unchanged RF/DFE chain. After the four baseline branches run, use the best successful acquisition to build per-delay array correlation vectors, remove the assumed LOS spatial component, estimate reflected-path DOAs, and append a `multipath_lcmv` branch. Keep truth-only evaluation separate from detection.

**Tech Stack:** Python 3.10+, NumPy, Matplotlib, pytest, existing `python_sim_simplified` modules.

**Spec:** `docs/superpowers/specs/2026-09-07-multipath-detection-suppression-design.md`

## Global Constraints

- Modify only `python_sim_simplified/`, root `PROJECT_STATUS.md`, and the corresponding docs; do not modify `python_sim/`.
- Preserve the simplified project's flat, directly readable module structure.
- Do not restore full-version exception wrapping, schema/checkpoint checks, or generic defensive branches.
- Retain only physical validation, explicit algorithm states, and numerical stability required for trustworthy research results.
- Detection and LCMV weight construction must not receive `ScenarioTruth`; truth is allowed only in evaluation, metrics, and plotting.
- A path delay must move the complete sampled waveform and produce its RF-equivalent carrier phase.
- Existing no-multipath behavior must remain reproducible by setting `cfg.signal.multipath_paths = []`.
- Implement each task test-first and commit it independently.

---

## File Structure

- Modify `python_sim_simplified/config.py`: multipath path, detector, and suppressor configuration types.
- Modify `python_sim_simplified/presets.py`: one deterministic default reflected path.
- Modify `python_sim_simplified/gnss.py`: reflected-path waveform generation and scene summation.
- Create `python_sim_simplified/multipath.py`: receive-data-only correlation, LOS projection, peak detection, Bartlett scan, and truth-only evaluation.
- Modify `python_sim_simplified/diversity.py`: LCMV constraint selection and combining.
- Modify `python_sim_simplified/pipeline.py`: baseline → detection → LCMV execution order.
- Modify `python_sim_simplified/models.py`: typed dictionaries shared by those stages.
- Modify `python_sim_simplified/metrics.py`: code bias and multipath suppression metrics.
- Modify `python_sim_simplified/plotting.py`: independent multipath figures and fifth branch support.
- Modify `python_sim_simplified/run_end_to_end.py`: save new figures.
- Modify `python_sim_simplified/README.md`: configuration, outputs, assumptions, and limitations.
- Modify `PROJECT_STATUS.md`: mark static specular multipath V1 and its remaining limits.
- Create `python_sim_simplified/tests/test_multipath.py`: detector and evaluator unit tests.
- Modify existing simplified tests only where the public result structure or intentional default scenario changes.

---

### Task 1: Multipath Configuration and RF Scene Generation

**Files:**
- Modify: `python_sim_simplified/config.py`
- Modify: `python_sim_simplified/presets.py`
- Modify: `python_sim_simplified/gnss.py`
- Modify: `python_sim_simplified/models.py`
- Modify: `python_sim_simplified/tests/test_config_gnss.py`
- Modify: `python_sim_simplified/tests/test_scenario.py`

**Interfaces:**
- Produces: `MultipathPath`, `MultipathDetectionConfig`, `MultipathSuppressionConfig`.
- Produces: `SignalConfig.multipath_paths`, `SimulationConfig.multipath_detection`, and `SimulationConfig.multipath_suppression`.
- Produces: `ScenarioTruth["multipath_paths"]: list[MultipathTruth]`.
- Keeps: `generate_scenario(cfg) -> tuple[np.ndarray, ScenarioTruth]`.

- [ ] **Step 1: Add failing configuration tests**

Add `from math import pi` to `presets.py`. Add imports and assertions to `test_config_gnss.py`:

```python
from config import MultipathPath


def test_default_scenario_contains_one_static_specular_multipath():
    cfg = default_config()
    assert len(cfg.signal.multipath_paths) == 1
    path = cfg.signal.multipath_paths[0]
    assert isinstance(path, MultipathPath)
    assert path.direction.azimuth_deg == 110.0
    assert path.direction.elevation_deg == 15.0
    assert path.relative_power_db == -6.0
    assert path.excess_delay_chips == 0.5
    assert path.reflection_phase_rad == np.pi
    assert cfg.multipath_detection.enable
    assert cfg.multipath_suppression.enable
```

- [ ] **Step 2: Add failing scene-generation tests**

Add tests to `test_scenario.py` that disable interferers and RF noise is not involved because `generate_scenario` ends before RF:

```python
def test_empty_multipath_list_keeps_scene_equal_to_direct_component():
    cfg = default_config()
    cfg.interferers = []
    cfg.signal.multipath_paths = []
    scene, truth = generate_scenario(cfg)
    np.testing.assert_array_equal(scene, truth["desired_component"])
    assert truth["multipath_paths"] == []


def test_multipath_applies_power_delay_reflection_phase_and_direction():
    cfg = default_config()
    cfg.interferers = []
    cfg.signal.duration_ms = 4
    cfg.phone.channel_delay_ns = {name: 0.0 for name in CORNER_NAMES}
    path = cfg.signal.multipath_paths[0]
    scene, truth = generate_scenario(cfg)
    reflected = truth["multipath_paths"][0]
    expected_delay_s = path.excess_delay_chips / cfg.signal.code_rate_hz
    assert reflected["excess_delay_s"] == pytest.approx(expected_delay_s)
    assert reflected["relative_power_db"] == path.relative_power_db
    assert reflected["direction"] == path.direction
    np.testing.assert_allclose(
        scene,
        truth["desired_component"] + reflected["component"],
    )
    expected_ratio = 10.0 ** (path.relative_power_db / 20.0)
    valid = slice(round(expected_delay_s * cfg.signal.fs_rf_hz) + 32, -32)
    measured_ratio = np.linalg.norm(reflected["component"][:, valid]) / np.linalg.norm(
        truth["desired_component"][:, valid]
    )
    assert measured_ratio == pytest.approx(expected_ratio, rel=0.08)
```

Also add:

```python
def test_two_multipath_paths_are_summed_independently():
    cfg = default_config()
    cfg.interferers = []
    first = cfg.signal.multipath_paths[0]
    second = replace(
        first,
        direction=Direction(220.0, 25.0),
        relative_power_db=-10.0,
        excess_delay_chips=1.25,
    )
    cfg.signal.multipath_paths = [first, second]
    scene, truth = generate_scenario(cfg)
    assert len(truth["multipath_paths"]) == 2
    np.testing.assert_allclose(
        scene,
        truth["desired_component"]
        + truth["multipath_paths"][0]["component"]
        + truth["multipath_paths"][1]["component"],
    )
```

- [ ] **Step 3: Run the new tests and verify failure**

Run:

```powershell
python -m pytest -q python_sim_simplified/tests/test_config_gnss.py python_sim_simplified/tests/test_scenario.py
```

Expected: FAIL because multipath configuration and truth fields do not exist.

- [ ] **Step 4: Add configuration dataclasses and defaults**

Add to `config.py`:

```python
@dataclass
class MultipathPath:
    direction: Direction
    enable: bool = True
    relative_power_db: float = -6.0
    excess_delay_chips: float = 0.5
    reflection_phase_rad: float = 0.0
    doppler_offset_hz: float = 0.0
    polarization: WavePolarization = field(default_factory=WavePolarization)


@dataclass
class MultipathDetectionConfig:
    enable: bool = True
    max_excess_delay_chips: float = 3.0
    delay_step_chips: float = 0.25
    detection_threshold_db: float = 6.0
    azimuth_step_deg: float = 5.0
    elevation_step_deg: float = 5.0


@dataclass
class MultipathSuppressionConfig:
    enable: bool = True
    diagonal_loading_factor: float = 1e-2
    max_steering_correlation: float = 0.95
```

Add `multipath_paths: list[MultipathPath] = field(default_factory=list)` to `SignalConfig`, and add both configuration objects to `SimulationConfig`. In `default_config()`, set:

```python
signal=SignalConfig(
    multipath_paths=[
        MultipathPath(
            direction=Direction(110.0, 15.0),
            relative_power_db=-6.0,
            excess_delay_chips=0.5,
            reflection_phase_rad=pi,
        )
    ]
),
multipath_detection=MultipathDetectionConfig(),
multipath_suppression=MultipathSuppressionConfig(),
```

- [ ] **Step 5: Generate each multipath as a full delayed RF-equivalent sequence**

Add `_MultipathScene` and `_multipath_scene(path, cfg, antenna_model, desired_waveform, time_s, true_delays_s)` to `gnss.py`. Validate finite values and nonnegative excess delay at this single entry point. Implement total delay and amplitude as:

```python
excess_delay_s = path.excess_delay_chips / cfg.signal.code_rate_hz
total_delays_s = true_delays_s + excess_delay_s
relative_voltage = 10.0 ** (path.relative_power_db / 20.0)
path_waveform = (
    desired_waveform
    * relative_voltage
    * np.exp(1j * path.reflection_phase_rad)
    * np.exp(1j * 2.0 * np.pi * path.doppler_offset_hz * time_s)
)
response_base, diagnostics = antenna_response(
    antenna_model,
    path.direction,
    path.polarization,
)
component = apply_channel_delays(
    response_base[:, None] * path_waveform[None, :],
    total_delays_s,
    cfg.signal.fs_rf_hz,
    cfg.signal.carrier_hz,
    cfg.signal.if_hz,
)
```

Create all enabled paths, add their components to `scene`, and expose their exact generated components and parameters under `truth["multipath_paths"]`. Include their antenna diagnostics under `antenna_model_diagnostics["multipath_paths"]`.

- [ ] **Step 6: Run focused and full existing scenario tests**

Run:

```powershell
python -m pytest -q python_sim_simplified/tests/test_config_gnss.py python_sim_simplified/tests/test_channel_delay.py python_sim_simplified/tests/test_scenario.py
```

Expected: PASS.

- [ ] **Step 7: Commit Task 1**

```powershell
git add -- python_sim_simplified/config.py python_sim_simplified/presets.py python_sim_simplified/gnss.py python_sim_simplified/models.py python_sim_simplified/tests/test_config_gnss.py python_sim_simplified/tests/test_scenario.py
git commit -m "feat: model static GNSS multipath paths"
```

---

### Task 2: Receive-Data-Only Delay and DOA Detection

**Files:**
- Create: `python_sim_simplified/multipath.py`
- Create: `python_sim_simplified/tests/test_multipath.py`
- Modify: `python_sim_simplified/models.py`

**Interfaces:**
- Consumes: `AntennaModel`, `AcquisitionResult`, `MultipathDetectionConfig`, and DFE channel matrix `(M, N)`.
- Produces: `detect_multipath(channels, fs_hz, prn, code_rate_hz, acquisition, acquisition_cfg, antenna_model, los_direction, polarization, cfg, max_steering_correlation) -> MultipathDetectionResult`.
- Produces: `empty_detection(status, pilot_mode=None) -> MultipathDetectionResult`.
- Critical boundary: neither function accepts `ScenarioTruth`.

- [ ] **Step 1: Write failing LOS-projection and API-boundary tests**

Create `test_multipath.py`:

```python
import inspect
import numpy as np

from multipath import detect_multipath, project_out_los


def test_los_projection_removes_only_the_los_spatial_component():
    los = np.array([1.0, 1.0j, -1.0, -1.0j])
    echo = np.array([1.0, -1.0, 1.0j, -1.0j])
    vectors = los[:, None] * np.array([[4.0, 2.0]]) + echo[:, None]
    residual = project_out_los(vectors, los)
    np.testing.assert_allclose(los.conj() @ residual, 0.0, atol=1e-12)
    assert np.linalg.norm(residual) > 0.0


def test_detector_interface_cannot_receive_scenario_truth():
    assert "truth" not in inspect.signature(detect_multipath).parameters
```

- [ ] **Step 2: Write a failing deterministic separated-path detector test**

Build 10 ms of noiseless DFE-rate C/A samples using `ca_code`, an actual assumed ideal antenna model, a known acquisition result with zero Doppler/code shift, and one echo delayed by two DFE samples. Assert:

```python
result = detect_multipath(
    channels,
    fs_hz,
    cfg.signal.prn,
    cfg.signal.code_rate_hz,
    acquisition,
    cfg.acquisition,
    model,
    cfg.signal.assumed_direction,
    cfg.signal.assumed_polarization,
    cfg.multipath_detection,
    cfg.multipath_suppression.max_steering_correlation,
)
assert result["status"] == "detected"
candidate = result["candidates"][0]
assert candidate["excess_delay_chips"] == pytest.approx(0.5, abs=0.25)
assert candidate["spatial_status"] == "resolvable"
assert candidate["direction"].azimuth_deg == pytest.approx(110.0, abs=5.0)
assert candidate["direction"].elevation_deg == pytest.approx(15.0, abs=5.0)
```

- [ ] **Step 3: Run detector tests and verify failure**

Run:

```powershell
python -m pytest -q python_sim_simplified/tests/test_multipath.py
```

Expected: FAIL because `multipath.py` does not exist.

- [ ] **Step 4: Implement correlation vectors and LOS projection**

In `multipath.py`, implement:

```python
def project_out_los(vectors: np.ndarray, los: np.ndarray) -> np.ndarray:
    projector = np.eye(los.size) - np.outer(los, los.conj()) / np.vdot(los, los)
    return projector @ vectors
```

For each candidate excess delay, circularly shift the local C/A replica by:

```python
offset_samples = round(delay_chips * fs_hz / code_rate_hz)
code_shift = acquisition["code_phase_samples"] + offset_samples
aligned_code = np.roll(local_code, code_shift)
carrier = np.exp(-1j * 2.0 * np.pi * acquisition["doppler_hz"] * n / fs_hz)
z[:, delay_index] = np.mean(
    channels[:, :n_samples]
    * carrier[None, :]
    * aligned_code[None, :],
    axis=1,
)
```

Use the acquisition coherent/noncoherent span:

```python
n_samples = min(
    channels.shape[1],
    acquisition["coherent_samples"] * acquisition_cfg.noncoherent_ms,
)
```

The delay grid starts at one configured delay step and ends at `max_excess_delay_chips`, so the LOS sample itself is not reported as a reflected path.

- [ ] **Step 5: Implement thresholded local-peak detection and Bartlett scanning**

Compute:

```python
raw_power = np.sum(np.abs(z) ** 2, axis=0)
residual_vectors = project_out_los(z, los_steering)
residual_power = np.sum(np.abs(residual_vectors) ** 2, axis=0)
threshold = np.median(residual_power) * 10.0 ** (
    cfg.detection_threshold_db / 10.0
)
```

Keep local maxima above `threshold`, sorted by descending residual power. For each peak, scan:

```python
azimuths_deg = np.arange(0.0, 360.0, cfg.azimuth_step_deg)
elevations_deg = np.arange(0.0, 90.0 + cfg.elevation_step_deg, cfg.elevation_step_deg)
```

Build each assumed manifold with `antenna_response(antenna_model, direction, polarization, estimated=True)`, project that manifold out of LOS, and use the normalized Bartlett score:

```python
score = abs(np.vdot(projected_manifold, residual_vector)) ** 2 / np.vdot(
    projected_manifold, projected_manifold
).real
```

For a grid manifold whose projected norm is at or below machine precision, assign score zero because that direction is spatially indistinguishable from LOS. Return candidate direction, original assumed steering, delay, power, angle grids, and 2-D score map. Mark a candidate `spatially_unresolvable` when its normalized steering correlation with LOS is above the supplied `max_steering_correlation`; otherwise mark it `resolvable`.

- [ ] **Step 6: Add no-peak and near-LOS tests**

Test that LOS-only input returns `status == "no_multipath_detected"` and an empty candidate list. Test a delayed copy with the same spatial steering as LOS and assert it is not accepted as a resolvable path.

- [ ] **Step 7: Run focused tests**

Run:

```powershell
python -m pytest -q python_sim_simplified/tests/test_multipath.py
```

Expected: PASS.

- [ ] **Step 8: Commit Task 2**

```powershell
git add -- python_sim_simplified/multipath.py python_sim_simplified/models.py python_sim_simplified/tests/test_multipath.py
git commit -m "feat: detect GNSS multipath from array correlations"
```

---

### Task 3: LOS-Preserving LCMV Combining

**Files:**
- Modify: `python_sim_simplified/diversity.py`
- Modify: `python_sim_simplified/models.py`
- Modify: `python_sim_simplified/tests/test_diversity.py`

**Interfaces:**
- Produces: `combine_lcmv(channels, los_steering, multipath_steerings, diagonal_loading_factor, max_steering_correlation) -> tuple[np.ndarray, CombiningInfo]`.
- Consumes: only estimated steering vectors from Task 2, never true path steering.

- [ ] **Step 1: Write failing exact-constraint test**

Add to `test_diversity.py`:

```python
def test_lcmv_preserves_los_and_nulls_resolvable_multipath():
    rng = np.random.default_rng(9)
    channels = rng.normal(size=(4, 2000)) + 1j * rng.normal(size=(4, 2000))
    los = np.array([1.0, 1.0j, -1.0, -1.0j])
    echo = np.array([1.0, -1.0, 1.0j, -1.0j])
    combined, info = combine_lcmv(channels, los, [echo], 1e-2, 0.95)
    assert combined.shape == (channels.shape[1],)
    assert np.vdot(info["weights"], los) == pytest.approx(1.0 + 0.0j, abs=1e-10)
    assert np.vdot(info["weights"], echo) == pytest.approx(0.0 + 0.0j, abs=1e-10)
    assert info["suppression_status"] == "multipath_constraints_applied"
```

- [ ] **Step 2: Write failing fallback and collinearity tests**

Assert an empty multipath list yields the same samples and weights as `combine_known(channels, los, "mvdr", 1e-2)`, while mode/status become `multipath_lcmv` and `no_multipath_detected`. Assert a steering vector with normalized correlation greater than `0.95` is rejected and listed in `rejected_constraint_indices`.

- [ ] **Step 3: Run focused tests and verify failure**

Run:

```powershell
python -m pytest -q python_sim_simplified/tests/test_diversity.py
```

Expected: FAIL because `combine_lcmv` is absent.

- [ ] **Step 4: Implement constraint filtering and LCMV**

Reuse the same sample covariance and diagonal-loading convention as `combine_known`. Filter candidate columns by normalized correlation with LOS and previously accepted constraints, and keep at most `channels.shape[0] - 1` echoes. Compute:

```python
constraints = np.column_stack([los_steering, *accepted])
responses = np.zeros(constraints.shape[1], dtype=complex)
responses[0] = 1.0
solved = np.linalg.solve(loaded_covariance, constraints)
gram = constraints.conj().T @ solved
weights = solved @ np.linalg.solve(gram, responses)
combined = weights.conj() @ channels
```

Populate `CombiningInfo` with `constraint_matrix`, `constraint_response`, `multipath_constraint_steerings`, `multipath_constraint_responses`, `suppression_status`, and rejected indices. Do not use a blanket `try/except` or pseudoinverse; physical collinearity is removed before the solve and covariance stability comes from configured loading.

- [ ] **Step 5: Run diversity tests**

Run:

```powershell
python -m pytest -q python_sim_simplified/tests/test_diversity.py
```

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

```powershell
git add -- python_sim_simplified/diversity.py python_sim_simplified/models.py python_sim_simplified/tests/test_diversity.py
git commit -m "feat: add LOS-constrained multipath LCMV"
```

---

### Task 4: Pipeline Orchestration and Truth-Isolation Contract

**Files:**
- Modify: `python_sim_simplified/pipeline.py`
- Modify: `python_sim_simplified/models.py`
- Modify: `python_sim_simplified/tests/test_pipeline.py`
- Modify: `python_sim_simplified/tests/test_research_contract.py`

**Interfaces:**
- Consumes: Task 2 detector and Task 3 LCMV combiner.
- Produces: `SimulationResults["multipath_detection"]`.
- Produces: `branches["multipath_lcmv"]` whenever multipath suppression is enabled, including MVDR fallback cases.

- [ ] **Step 1: Write failing five-branch pipeline test**

Update the expected top-level keys and add:

```python
def test_pipeline_detects_before_appending_multipath_lcmv_branch():
    cfg = default_config()
    cfg.signal.duration_ms = 30
    cfg.tracking.max_epochs = 20
    results = run_end_to_end(cfg)
    assert set(results["branches"]) == {
        "single", "egc", "mvdr", "blind_null", "multipath_lcmv"
    }
    detection = results["multipath_detection"]
    assert detection["pilot_mode"] in {"single", "egc", "mvdr", "blind_null"}
    assert detection["status"] in {"detected", "no_multipath_detected"}
    assert results["branches"]["multipath_lcmv"]["combining"]["mode"] == "multipath_lcmv"
```

- [ ] **Step 2: Write failing no-pilot and truth-isolation tests**

Use an acquisition threshold too high to allow success and assert `status == "no_pilot_acquisition"`; still assert that the LCMV branch exists and records MVDR fallback. In `test_research_contract.py`, inspect the detector signature and source imports to assert that `ScenarioTruth` is not consumed by `detect_multipath`.

- [ ] **Step 3: Run pipeline tests and verify failure**

Run:

```powershell
python -m pytest -q python_sim_simplified/tests/test_pipeline.py python_sim_simplified/tests/test_research_contract.py
```

Expected: FAIL because detection and the fifth branch are not orchestrated.

- [ ] **Step 4: Refactor receiver finishing from combining**

Extract from `_run_branch`:

```python
def _finish_branch(
    combined: np.ndarray,
    combining: CombiningInfo,
    truth: ScenarioTruth,
    cfg: SimulationConfig,
    fs_out_hz: float,
    covariance_snapshots: int,
) -> BranchResult:
    weights = combining["weights"]
    interferer_gains = [
        float(abs(np.vdot(weights, steering)))
        for steering in truth["interferer_steering_effective"]
    ]
    multipath_gains = [
        float(abs(np.vdot(weights, path["steering_true"])))
        for path in truth["multipath_paths"]
    ]
    combining.update(
        {
            "target_direction_gain": float(
                abs(np.vdot(weights, truth["desired_steering_true"]))
            ),
            "interferer_direction_gains": interferer_gains,
            "worst_interferer_gain": max(interferer_gains, default=0.0),
            "multipath_direction_gains": multipath_gains,
            "worst_multipath_gain": max(multipath_gains, default=0.0),
            "covariance_snapshots": covariance_snapshots,
            "weights_are_fixed": True,
        }
    )
    acquisition = acquire(
        combined,
        fs_out_hz,
        cfg.signal.prn,
        cfg.signal.code_rate_hz,
        cfg.acquisition,
    )
    tracking = (
        track(
            combined,
            fs_out_hz,
            cfg.signal.prn,
            acquisition,
            cfg.tracking,
            cfg.signal.carrier_hz,
            cfg.signal.code_rate_hz,
        )
        if acquisition["success"]
        else {
            "success": False,
            "status": "acquisition_failed",
            "prompt": np.array([], dtype=complex),
        }
    )
    return {
        "combined": combined,
        "combining": combining,
        "acquisition": acquisition,
        "tracking": tracking,
    }
```

Pass `dfe_channels.shape[1]` as `covariance_snapshots`; do not add this implementation detail to scenario truth. Keep `_run_branch` responsible only for selecting the existing combiner, then call `_finish_branch`. This permits LCMV to use the identical ACQ/TRK path without copying it.

- [ ] **Step 5: Run baseline branches, choose a pilot, detect, then append LCMV**

Choose the pilot without truth:

```python
successful = [
    (mode, branch["acquisition"])
    for mode, branch in branches.items()
    if branch["acquisition"]["success"]
]
pilot = max(successful, key=lambda item: item[1]["metric"]) if successful else None
```

Build the assumed antenna model, call `detect_multipath` when a pilot exists, passing `cfg.multipath_suppression.max_steering_correlation`, and otherwise use `empty_detection("no_pilot_acquisition")`. Pass only candidates whose `spatial_status == "resolvable"` to `combine_lcmv`. Append its output through `_finish_branch`.

When `cfg.multipath_detection.enable` is false, record `detection_disabled`; when suppression is false, do not append the fifth branch. Store detection in `SimulationResults` before metrics run.

- [ ] **Step 6: Run pipeline and full focused tests**

Run:

```powershell
python -m pytest -q python_sim_simplified/tests/test_pipeline.py python_sim_simplified/tests/test_research_contract.py python_sim_simplified/tests/test_receiver.py
```

Expected: PASS.

- [ ] **Step 7: Commit Task 4**

```powershell
git add -- python_sim_simplified/pipeline.py python_sim_simplified/models.py python_sim_simplified/tests/test_pipeline.py python_sim_simplified/tests/test_research_contract.py
git commit -m "feat: integrate multipath detection and LCMV pipeline"
```

---

### Task 5: Detection Evaluation, Suppression Gain, and Code-Bias Metrics

**Files:**
- Modify: `python_sim_simplified/multipath.py`
- Modify: `python_sim_simplified/metrics.py`
- Modify: `python_sim_simplified/models.py`
- Modify: `python_sim_simplified/pipeline.py`
- Modify: `python_sim_simplified/tests/test_multipath.py`
- Modify: `python_sim_simplified/tests/test_pipeline.py`

**Interfaces:**
- Produces: `evaluate_multipath(detection, truth_paths) -> MultipathEvaluation`; this is the only multipath function in this task allowed to receive truth.
- Produces: `tracking_code_bias_chips(tracking, truth, fs_out_hz, coherent_ms) -> np.ndarray`.
- Extends each `MetricRecord` with direction gains and acquisition/tracking code-bias fields.

- [ ] **Step 1: Write failing truth-matching tests**

In `test_multipath.py`, create two truth paths and one matching/one false candidate. Assert one detection, one miss, one false alarm, exact delay error, and a DOA error computed from the 3-D unit-vector angle rather than separate azimuth/elevation subtraction.

```python
evaluation = evaluate_multipath(detection, truth_paths)
assert evaluation["detected_count"] == 1
assert evaluation["missed_count"] == 1
assert evaluation["false_alarm_count"] == 1
assert evaluation["matches"][0]["delay_error_chips"] == pytest.approx(0.05)
```

- [ ] **Step 2: Write failing acquisition and tracking code-bias tests**

Test the circular chip-error helper at the 0/1023 boundary. For tracking, use recorded epoch starts and code-state history and assert that a perfectly constructed history yields zero bias, while a `0.1` chip offset yields `0.1 × c/code_rate_hz` metres.

Add a deterministic pipeline experiment with interferers disabled, signal power raised to `-95 dBm`, one `-3 dB` echo at `0.5` chip and `(110°, 15°)`, and a `3 dB` detection threshold. Assert the path is resolvable, the LCMV true multipath direction gain is below EGC, and its RMS tracking code bias is below SINGLE/EGC when those branches track.

- [ ] **Step 3: Run metric tests and verify failure**

Run:

```powershell
python -m pytest -q python_sim_simplified/tests/test_multipath.py python_sim_simplified/tests/test_pipeline.py
```

Expected: FAIL because evaluation and code-bias fields are absent.

- [ ] **Step 4: Implement truth-only matching**

Match each detection to at most one truth path, sorted by absolute delay difference and then 3-D angular distance. Return counts plus per-match delay error, direction error, and estimated/true indexes. Call this function in `pipeline.py` only after `detect_multipath` returns, and store `results["multipath_evaluation"]`.

- [ ] **Step 5: Implement code-bias calculations**

Use circular chip error:

```python
def circular_chip_error(estimate, truth):
    return (np.asarray(estimate) - np.asarray(truth) + 511.5) % 1023.0 - 511.5
```

Acquisition bias is the acquired code-phase sample converted by `code_rate_hz / fs_out_hz`. Tracking history describes the next epoch, so compare it with truth at:

```python
comparison_samples = tracking["epoch_start_samples"] + round(
    fs_out_hz * coherent_ms * 1e-3
)
expected_phase = np.mod(
    comparison_samples * truth["code_rate_hz"] / fs_out_hz
    - truth["code_delay_chips"],
    1023.0,
)
```

Convert chips to metres with `SPEED_OF_LIGHT_MPS / nominal_code_rate_hz`.

- [ ] **Step 6: Extend branch metrics and console formatting**

Add to each record using the finalized branch weights and the code-bias helpers:

```python
multipath_gains = branch["combining"]["multipath_direction_gains"]
tracking_bias = tracking_code_bias_chips(
    tracking,
    results["truth"],
    results["dfe_info"]["fs_out_hz"],
    results["config"].tracking.coherent_ms,
)
record.update(
    {
        "multipath_direction_gains": multipath_gains,
        "worst_multipath_gain": max(multipath_gains, default=0.0),
        "acquisition_code_bias_chips": acquisition_code_bias_chips(
            acquisition,
            results["truth"],
            results["dfe_info"]["fs_out_hz"],
        ),
        "tracking_code_bias_rms_chips": (
            float(np.sqrt(np.mean(tracking_bias**2)))
            if tracking_bias.size
            else None
        ),
        "tracking_code_bias_rms_m": (
            float(np.sqrt(np.mean(tracking_bias**2)))
            * SPEED_OF_LIGHT_MPS
            / results["config"].signal.code_rate_hz
            if tracking_bias.size
            else None
        ),
    }
)
```

Use true steering only to evaluate `abs(w^H a_MP)` after weights are finalized. Add a compact `CodeBias(m)` column to `format_metrics`; use `NaN` when tracking fails.

- [ ] **Step 7: Run focused tests**

Run:

```powershell
python -m pytest -q python_sim_simplified/tests/test_multipath.py python_sim_simplified/tests/test_pipeline.py
```

Expected: PASS.

- [ ] **Step 8: Commit Task 5**

```powershell
git add -- python_sim_simplified/multipath.py python_sim_simplified/metrics.py python_sim_simplified/models.py python_sim_simplified/pipeline.py python_sim_simplified/tests/test_multipath.py python_sim_simplified/tests/test_pipeline.py
git commit -m "feat: report multipath detection and code bias"
```

---

### Task 6: Independent Multipath Research Figures

**Files:**
- Modify: `python_sim_simplified/plotting.py`
- Modify: `python_sim_simplified/run_end_to_end.py`
- Modify: `python_sim_simplified/tests/test_plotting_runner.py`

**Interfaces:**
- Produces: `plot_multipath_results(results) -> dict[str, Figure]`.
- Produces figure keys: `multipath_delay_profile`, `multipath_angle_path_<n>`, `multipath_correlation_comparison`, and `multipath_tracking_bias`.
- Keeps existing figure keys; total output count becomes scenario-dependent.

- [ ] **Step 1: Write failing figure-key tests**

Replace the exact “18 files” assertion with preservation plus new outputs:

```python
paths = save_results_figures(results, tmp_path)
assert {
    "detailed_rf_dfe_TL",
    "detailed_combined_mvdr",
    "detailed_combined_multipath_lcmv",
    "detailed_array_response_multipath_lcmv",
    "detailed_acquisition_multipath_lcmv",
    "tracking",
    "cn0",
    "multipath_delay_profile",
    "multipath_correlation_comparison",
    "multipath_tracking_bias",
} <= set(paths)
assert any(name.startswith("multipath_angle_path_") for name in paths)
```

Update detailed-mode expectations to derive modes from `results["branches"]`, not only `cfg.diversity.modes`.

- [ ] **Step 2: Run plotting tests and verify failure**

Run:

```powershell
$env:MPLBACKEND='Agg'
python -m pytest -q python_sim_simplified/tests/test_plotting_runner.py
```

Expected: FAIL because multipath figures and fifth-branch details are absent.

- [ ] **Step 3: Add fifth-branch ordering and dynamic overview layout**

Append `multipath_lcmv` to `MODE_ORDER`. Make the overview acquisition grid derive rows/columns from the number of branches and hide unused axes, so five modes are not truncated by the current fixed 2×2 layout. Mark configured true multipath directions with distinct blue dashed markers on spatial-response plots.

- [ ] **Step 4: Implement four independent multipath plot families**

Implement `plot_multipath_results` by composing focused helpers:

```python
def plot_multipath_results(results):
    figures = {
        "multipath_delay_profile": _plot_multipath_delay_profile(results),
        "multipath_correlation_comparison": _plot_multipath_correlation_comparison(results),
        "multipath_tracking_bias": _plot_multipath_tracking_bias(results),
    }
    for index, candidate in enumerate(
        results["multipath_detection"]["candidates"], start=1
    ):
        figures[f"multipath_angle_path_{index}"] = _plot_multipath_angle(
            candidate, index
        )
    return figures
```

The delay plot shows raw/residual normalized dB curves, threshold, truth delays, and detected delays. The angle plot uses one figure per candidate. The correlation comparison uses each branch's best-Doppler acquisition row around LOS. The tracking-bias plot uses `tracking_code_bias_chips` and supplies a metres secondary axis or a separate second subplot.

- [ ] **Step 5: Save multipath figures from the direct runner**

Merge into `save_results_figures`:

```python
figures = {
    **plot_detailed_results(results),
    "tracking": plot_tracking(results),
    "cn0": plot_cn0(results),
    **plot_multipath_results(results),
}
```

Keep the existing behavior that removes stale PNG files before saving.

- [ ] **Step 6: Run plotting tests**

Run:

```powershell
$env:MPLBACKEND='Agg'
python -m pytest -q python_sim_simplified/tests/test_plotting_runner.py
```

Expected: PASS with all figures nonempty and independent.

- [ ] **Step 7: Commit Task 6**

```powershell
git add -- python_sim_simplified/plotting.py python_sim_simplified/run_end_to_end.py python_sim_simplified/tests/test_plotting_runner.py
git commit -m "feat: visualize multipath detection and suppression"
```

---

### Task 7: Documentation, Regression Baseline, and End-to-End Verification

**Files:**
- Modify: `python_sim_simplified/README.md`
- Modify: `PROJECT_STATUS.md`
- Modify: `python_sim_simplified/tests/test_default_regression.py`

**Interfaces:**
- Documents all new configuration fields and states.
- Preserves the old deterministic numeric regression by explicitly disabling multipath in that test.
- Adds a deterministic default-multipath contract without hard-coding unstable full-pipeline floating-point values.

- [ ] **Step 1: Preserve the old no-multipath regression**

Refactor the existing test to create `cfg` before calling the pipeline, then add:

```python
cfg.signal.multipath_paths = []
cfg.multipath_detection.enable = False
cfg.multipath_suppression.enable = False
results = run_end_to_end(cfg)
```

Add a separate default contract asserting one configured path, five finite branch outputs, a populated detection status, finite multipath metrics, and generated truth components.

- [ ] **Step 2: Run regression tests before documentation edits**

Run:

```powershell
python -m pytest -q python_sim_simplified/tests/test_default_regression.py
```

Expected: PASS; the no-multipath exact baseline remains unchanged.

- [ ] **Step 3: Document configuration and interpretation**

Add to `python_sim_simplified/README.md`:

```python
from config import Direction, MultipathPath

cfg.signal.multipath_paths = [
    MultipathPath(
        direction=Direction(110.0, 15.0),
        relative_power_db=-6.0,
        excess_delay_chips=0.5,
        reflection_phase_rad=np.pi,
        doppler_offset_hz=0.0,
    )
]
```

Explain that delay shifts the complete sequence, `reflection_phase_rad` is additional surface phase, truth is evaluation-only, detection resolution is DFE/sample limited, spatially close paths cannot be nulled safely, and output figure count is now scenario-dependent rather than fixed at 18.

- [ ] **Step 4: Update project status**

In `PROJECT_STATUS.md`, add static specular multipath V1 to the current capability/stage table, list detection + LCMV as the first spatial mitigation baseline, and keep dynamic/diffuse/NLOS multipath and STAP/MEDLL as future work. State explicitly that only `python_sim_simplified` receives this feature.

- [ ] **Step 5: Run the entire simplified suite with warnings as errors**

Run:

```powershell
$env:MPLBACKEND='Agg'
python -W error -m pytest -q python_sim_simplified/tests
```

Expected: all tests PASS with no warnings.

- [ ] **Step 6: Run the direct end-to-end entry point**

Run:

```powershell
$env:MPLBACKEND='Agg'
$env:PYTHONWARNINGS='error'
python python_sim_simplified/run_end_to_end.py
```

Expected: exit code 0; console table includes `multipath_lcmv` and code bias; `python_sim_simplified/results/` contains existing figures plus the multipath delay, angle, correlation, and tracking-bias figures.

- [ ] **Step 7: Verify scope and repository cleanliness**

Run:

```powershell
git diff --check
git status --short
git diff --name-only origin/main...HEAD
```

Expected: no whitespace errors; no `python_sim/` paths in feature commits; only planned source, test, and documentation files changed; generated results remain ignored.

- [ ] **Step 8: Commit Task 7**

```powershell
git add -- python_sim_simplified/README.md PROJECT_STATUS.md python_sim_simplified/tests/test_default_regression.py
git commit -m "docs: explain multipath research workflow"
```

---

## Final Acceptance Checklist

- [ ] Empty multipath configuration reproduces the previous signal and exact default regression.
- [ ] One or multiple static reflected paths traverse antenna, channel-delay, RF, DFE, combining, ACQ, and TRK stages.
- [ ] Detector source and signature do not consume scenario truth.
- [ ] Separated paths produce delay and DOA estimates; close paths report the designed limitation.
- [ ] LCMV numerically preserves assumed LOS and nulls accepted estimated multipath directions.
- [ ] The default deterministic scenario shows reduced multipath direction gain and code bias relative to SINGLE/EGC, or the run is reported as spatially unresolvable rather than falsely successful.
- [ ] Existing independent figures remain, and new multipath figures are saved separately.
- [ ] Full simplified tests pass with warnings treated as errors.
- [ ] `python_sim/` remains unchanged.
