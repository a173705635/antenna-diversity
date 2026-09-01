# Simplified Research Simulation Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `python_sim_simplified` into a clear personal research simulation while preserving its default numerical behavior, result keys, console metrics, and seven figures.

**Architecture:** Keep the flat RF → DFE → combining → acquisition → tracking module flow. Separate configuration presets and measured-pattern I/O, restrict the antenna model to antenna-owned inputs, describe existing dictionaries with `TypedDict`, and split only the three oversized orchestration functions.

**Tech Stack:** Python 3.11, NumPy, SciPy, Matplotlib, pytest, standard-library dataclasses and typing.

**Spec:** `docs/superpowers/specs/2026-09-01-simplified-research-refactor-design.md`

## Global Constraints

- Do not change RF, DFE, antenna, delay, combining, acquisition, tracking, or C/N0 formulas.
- Preserve default values, random seed, MT19937 use, calculation order, result dictionary keys, console columns, and seven figure names.
- Keep `python_sim_simplified` independent of `python_sim`.
- Keep the flat research-code layout; do not introduce deep packages or a validation framework.
- Existing valid-input behavior is the compatibility target; do not restore the full project's defensive branches.
- Use `pytest.approx` or `numpy.testing` with explicit tolerances for floating-point regression checks.

---

### Task 1: Lock the Current Numerical Baseline

**Files:**
- Create: `python_sim_simplified/tests/test_default_regression.py`

**Interfaces:**
- Consumes: `presets.default_config()` after Task 2; until then import `config.default_config()`.
- Produces: a deterministic regression test that protects all later refactors.

- [ ] **Step 1: Add the characterization test using the current result structure**

```python
import pytest

from config import default_config
from pipeline import run_end_to_end


EXPECTED = {
    "single": (False, 1.0160803563227625, -1750.0, None),
    "egc": (True, 33.69005705636806, 1500.0, 50.051921894528355),
    "mvdr": (True, 34.961314521617254, 1500.0, 50.13246168366476),
    "blind_null": (True, 15.893485034538864, 1500.0, 45.123190884508226),
}


def test_default_simulation_matches_locked_baseline():
    results = run_end_to_end(default_config())
    assert results["scene_input_v"].shape == (4, 1_636_800)
    assert results["adc_codes"].shape == (4, 1_636_800)
    assert results["dfe_channels"].shape == (4, 409_184)
    assert results["rf_info"]["total_gain_db"] == pytest.approx(80.0)
    assert results["rf_info"]["noise_figure_db"] == pytest.approx(
        1.2350293663855842
    )
    assert results["dfe_info"]["fs_out_hz"] == pytest.approx(4_092_000.0)

    for record in results["metrics"]:
        tracked, metric, doppler_hz, cn0_db_hz = EXPECTED[record["mode"]]
        assert record["tracking"] is tracked
        assert record["acquisition_metric"] == pytest.approx(metric, rel=1e-10)
        assert record["acquired_doppler_hz"] == pytest.approx(doppler_hz)
        if cn0_db_hz is None:
            assert record["final_cn0_db_hz"] is None
        else:
            assert record["final_cn0_db_hz"] == pytest.approx(cn0_db_hz, rel=1e-10)
```

- [ ] **Step 2: Run the characterization test and the existing suite**

Run: `python -m pytest python_sim_simplified/tests/test_default_regression.py python_sim_simplified/tests -q`

Expected: 29 tests pass; running the same test twice through the command is harmless but avoid duplicate collection by using `python -m pytest python_sim_simplified/tests -q` after the focused run.

- [ ] **Step 3: Commit the baseline**

```bash
git add python_sim_simplified/tests/test_default_regression.py
git commit -m "test: lock simplified simulation baseline"
```

---

### Task 2: Establish Single-Source Configuration and Presets

**Files:**
- Create: `python_sim_simplified/presets.py`
- Modify: `python_sim_simplified/config.py`
- Modify: `python_sim_simplified/rf.py`
- Modify: `python_sim_simplified/dfe.py`
- Modify: `python_sim_simplified/acquisition.py`
- Modify: `python_sim_simplified/pipeline.py`
- Modify: `python_sim_simplified/run_end_to_end.py`
- Modify: tests importing `default_config`, `parametric_config`, or `measured_config`
- Test: `python_sim_simplified/tests/test_config_gnss.py`
- Test: `python_sim_simplified/tests/test_rf_dfe.py`
- Test: `python_sim_simplified/tests/test_receiver.py`

**Interfaces:**
- Produces: `presets.default_config()`, `presets.parametric_config()`, and `presets.measured_config()`.
- Produces: `SimulationConfig.reference_impedance_ohm: float` and `DFEConfig.nco_frequency_hz: float | None`.
- Produces: `process_rf(x_antenna_v, cfg, reference_impedance_ohm, seed)`, `process_dfe(adc_codes, cfg, fs_in_hz, signal_if_hz)`, and `acquire(x, fs_hz, prn, code_rate_hz, cfg)`.

- [ ] **Step 1: Write failing ownership tests**

```python
from presets import default_config


def test_shared_physical_parameters_have_one_default_source():
    cfg = default_config()
    assert cfg.reference_impedance_ohm == 50.0
    assert not hasattr(cfg.rf, "resistance_ohm")
    assert not hasattr(cfg.dfe, "fs_in_hz")
    assert not hasattr(cfg.dfe, "if_hz")
    assert cfg.dfe.nco_frequency_hz is None
    assert not hasattr(cfg.acquisition, "code_rate_hz")
```

Add a DFE test that sets `cfg.nco_frequency_hz = 3.0e6` and asserts `info["if_hz"] == 3.0e6`; add a pipeline test that changes `cfg.signal.fs_rf_hz`, `cfg.signal.if_hz`, and `cfg.signal.code_rate_hz` without updating another group and verifies the downstream `dfe_info` and acquisition code rate use those values.

- [ ] **Step 2: Run the new tests and verify the expected import/field failures**

Run: `python -m pytest python_sim_simplified/tests/test_config_gnss.py python_sim_simplified/tests/test_rf_dfe.py -q`

Expected: FAIL because `presets.py` and the new fields/signatures do not exist.

- [ ] **Step 3: Move preset construction and remove duplicate fields**

Implement these configuration shapes:

```python
@dataclass
class DFEConfig:
    decimation: int = 4
    cutoff_hz: float = 1.8e6
    num_taps: int = 129
    output_bits: int = 8
    output_binary_shift: int = 2
    nco_frequency_hz: float | None = None


@dataclass
class AcquisitionConfig:
    doppler_bins_hz: np.ndarray = field(
        default_factory=lambda: np.arange(-5_000.0, 5_000.1, 250.0)
    )
    coherent_ms: int = 1
    noncoherent_ms: int = 4
    exclusion_chips: float = 1.0
    threshold: float = 2.5


@dataclass
class SimulationConfig:
    random_seed: int
    reference_impedance_ohm: float
    signal: SignalConfig
    phone: PhoneConfig
    interferers: list[CWInterferer]
    rf: RFConfig
    dfe: DFEConfig
    diversity: DiversityConfig
    acquisition: AcquisitionConfig
    tracking: TrackingConfig
    plot: PlotConfig
```

Move `mate70_phase_centers`, `default_config`, `parametric_config`, and `measured_config` unchanged in value to `presets.py`. Remove `SignalConfig.resistance_ohm`, `RFConfig.resistance_ohm`, `DFEConfig.fs_in_hz`, `DFEConfig.if_hz`, `DFEConfig.fs_out_hz`, and `AcquisitionConfig.code_rate_hz`.

- [ ] **Step 4: Update stage functions without changing calculations**

```python
def process_rf(x_antenna_v, cfg, reference_impedance_ohm, seed):
    # Existing cascade and MT19937 calculations stay in the same order.
    noise_rms_v = np.sqrt(input_noise_power_w * reference_impedance_ohm)


def process_dfe(adc_codes, cfg, fs_in_hz, signal_if_hz):
    nco_frequency_hz = (
        signal_if_hz
        if cfg.nco_frequency_hz is None
        else cfg.nco_frequency_hz
    )
    h, info = design_decimator(
        fs_in_hz, cfg.decimation, cfg.cutoff_hz, cfg.num_taps
    )
    # Use nco_frequency_hz in the existing NCO expression.


def acquire(x, fs_hz, prn, code_rate_hz, cfg):
    code_phase = np.mod(n * code_rate_hz / fs_hz, 1023.0)
```

Keep `rf_info`, `dfe_info`, and acquisition result keys unchanged. Set `dfe_info["if_hz"]` to the actual NCO frequency and continue deriving `fs_out_hz` inside `design_decimator`.

- [ ] **Step 5: Update imports and pipeline arguments**

Use `from presets import default_config` in the runner, pipeline, plotting tests, and regression test. In `pipeline.run_end_to_end`, pass `cfg.reference_impedance_ohm`, `cfg.signal.fs_rf_hz`, `cfg.signal.if_hz`, and `cfg.signal.code_rate_hz` to their owning stage functions.

- [ ] **Step 6: Run focused and full tests**

Run: `python -m pytest python_sim_simplified/tests/test_config_gnss.py python_sim_simplified/tests/test_rf_dfe.py python_sim_simplified/tests/test_receiver.py python_sim_simplified/tests/test_default_regression.py -q`

Expected: all focused tests pass and the locked baseline remains unchanged.

Run: `python -m pytest python_sim_simplified/tests -q`

Expected: all tests pass.

- [ ] **Step 7: Commit configuration ownership**

```bash
git add python_sim_simplified/config.py python_sim_simplified/presets.py python_sim_simplified/rf.py python_sim_simplified/dfe.py python_sim_simplified/acquisition.py python_sim_simplified/pipeline.py python_sim_simplified/run_end_to_end.py python_sim_simplified/tests
git commit -m "refactor: clarify simplified simulation parameters"
```

---

### Task 3: Isolate the Antenna Model and Measured-Pattern I/O

**Files:**
- Create: `python_sim_simplified/measured_pattern.py`
- Modify: `python_sim_simplified/antenna.py`
- Modify: `python_sim_simplified/gnss.py`
- Modify: `python_sim_simplified/plotting.py`
- Modify: `python_sim_simplified/tests/test_antenna.py`
- Modify: `python_sim_simplified/tests/test_teaching_contract.py`

**Interfaces:**
- Produces: `build_antenna_model(phone: PhoneConfig, carrier_hz: float) -> AntennaModel`.
- Produces: `measured_pattern.MeasuredGrid`, `load_measured_pattern(path, frequency_hz)`, and `interpolate_measured(grid, azimuth_deg, elevation_deg)`.
- Preserves: `antenna_response(model, direction, polarization, estimated=False)` return values and diagnostic keys.

- [ ] **Step 1: Write the failing boundary test**

```python
from antenna import build_antenna_model
from presets import default_config


def test_antenna_model_owns_only_antenna_inputs():
    cfg = default_config()
    model = build_antenna_model(cfg.phone, cfg.signal.carrier_hz)
    assert not hasattr(model, "cfg")
    assert model.active_antennas == cfg.phone.active_antennas
    assert model.carrier_hz == cfg.signal.carrier_hz
```

- [ ] **Step 2: Run the boundary test and verify it fails**

Run: `python -m pytest python_sim_simplified/tests/test_antenna.py::test_antenna_model_owns_only_antenna_inputs -q`

Expected: FAIL because `build_antenna_model` still accepts `SimulationConfig` and `AntennaModel` stores `cfg`.

- [ ] **Step 3: Extract measured-pattern loading and interpolation**

Move `MeasuredGrid`, `load_measured_pattern`, `_bounded_neighbors`, `_periodic_neighbors`, and `interpolate_measured` to `measured_pattern.py` without changing their expressions or iteration order. Import `MeasuredGrid`, `load_measured_pattern`, and `interpolate_measured` in `antenna.py`.

- [ ] **Step 4: Replace the broad antenna model**

```python
@dataclass
class AntennaModel:
    active_antennas: tuple[str, ...]
    carrier_hz: float
    true_positions_m: dict[str, tuple[float, float, float]]
    assumed_positions_m: dict[str, tuple[float, float, float]]
    true_state: AntennaState
    assumed_state: AntennaState
    true_channel_response: np.ndarray
    assumed_channel_response: np.ndarray
    true_measured: dict[str, MeasuredGrid] | None
    assumed_measured: dict[str, MeasuredGrid] | None


def build_antenna_model(phone: PhoneConfig, carrier_hz: float) -> AntennaModel:
    true_state = phone.antenna_model.true_state
    assumed_state = phone.antenna_model.assumed_state
    true_measured = (
        load_measured_pattern(true_state.measured_pattern_csv, carrier_hz)
        if true_state.pattern_kind == "measured"
        else None
    )
    assumed_measured = (
        load_measured_pattern(assumed_state.measured_pattern_csv, carrier_hz)
        if assumed_state.pattern_kind == "measured"
        else None
    )
    return AntennaModel(
        active_antennas=phone.active_antennas,
        carrier_hz=carrier_hz,
        true_positions_m=phone.phase_centers_m,
        assumed_positions_m=phone.assumed_phase_centers_m,
        true_state=true_state,
        assumed_state=assumed_state,
        true_channel_response=channel_response(phone, estimated=False),
        assumed_channel_response=channel_response(phone, estimated=True),
        true_measured=true_measured,
        assumed_measured=assumed_measured,
    )
```

Move `channel_response(phone, estimated)` from `config.py` to `antenna.py`. Build and store true and assumed channel arrays once. Update `antenna_response` to use only `model` fields while preserving all diagnostic dictionary keys.

- [ ] **Step 5: Update call sites**

Use `build_antenna_model(cfg.phone, cfg.signal.carrier_hz)` in `gnss.py`, `plotting.py`, and tests. Rename `test_teaching_contract.py` to `test_research_contract.py` and keep its standalone-copy and antenna-combination coverage unchanged.

- [ ] **Step 6: Run antenna, scenario, plotting, and full tests**

Run: `python -m pytest python_sim_simplified/tests/test_antenna.py python_sim_simplified/tests/test_scenario.py python_sim_simplified/tests/test_plotting_runner.py python_sim_simplified/tests/test_research_contract.py python_sim_simplified/tests/test_default_regression.py -q`

Expected: all focused tests pass and the baseline metrics remain unchanged.

Run: `python -m pytest python_sim_simplified/tests -q`

Expected: all tests pass.

- [ ] **Step 7: Commit antenna isolation**

```bash
git add python_sim_simplified/antenna.py python_sim_simplified/measured_pattern.py python_sim_simplified/gnss.py python_sim_simplified/plotting.py python_sim_simplified/tests
git commit -m "refactor: isolate simplified antenna model"
```

---

### Task 4: Type Stage Contracts and Split Scenario Generation

**Files:**
- Create: `python_sim_simplified/models.py`
- Modify: `python_sim_simplified/gnss.py`
- Modify: `python_sim_simplified/rf.py`
- Modify: `python_sim_simplified/dfe.py`
- Modify: `python_sim_simplified/diversity.py`
- Modify: `python_sim_simplified/acquisition.py`
- Modify: `python_sim_simplified/tests/test_scenario.py`

**Interfaces:**
- Produces runtime-neutral `TypedDict` contracts: `AntennaDiagnostics`, `ScenarioTruth`, `RFInfo`, `DFEInfo`, `CombiningInfo`, `AcquisitionResult`, `TrackingResult`, `BranchResult`, `MetricRecord`, and `SimulationResults`.
- Preserves `generate_scenario(cfg) -> tuple[np.ndarray, ScenarioTruth]` and every current truth key.

- [ ] **Step 1: Add a scenario contract test**

```python
def test_scenario_public_keys_remain_stable():
    _, truth = generate_scenario(default_config())
    assert set(truth) == {
        "time_s", "code", "desired_waveform", "desired_component",
        "desired_steering_true", "desired_steering_assumed",
        "desired_open_circuit_response_true",
        "desired_open_circuit_response_assumed",
        "interferer_steering_effective", "interferer_components",
        "active_antennas", "positions_m", "channel_delay_true_ns",
        "channel_delay_estimated_ns", "channel_delay_true_samples",
        "channel_delay_estimated_samples", "desired_delay_phase_true",
        "desired_delay_phase_assumed", "interferer_delay_phases_true",
        "doppler_hz", "code_rate_hz", "code_delay_chips", "fs_rf_hz",
        "if_hz", "antenna_model_diagnostics",
    }
```

- [ ] **Step 2: Run the contract test**

Run: `python -m pytest python_sim_simplified/tests/test_scenario.py::test_scenario_public_keys_remain_stable -q`

Expected: PASS before refactoring; this is a characterization gate.

- [ ] **Step 3: Define `TypedDict` stage contracts**

In `models.py`, declare every existing public key with concrete scalar, array, tuple, list, dictionary, configuration, or nested contract types. Use `NotRequired` only for mode-specific combining keys and failed-tracking fields. Do not convert dictionaries to dataclass instances at runtime.

```python
class BranchResult(TypedDict):
    combined: np.ndarray
    combining: CombiningInfo
    acquisition: AcquisitionResult
    tracking: TrackingResult


class SimulationResults(TypedDict):
    config: SimulationConfig
    truth: ScenarioTruth
    scene_input_v: np.ndarray
    adc_codes: np.ndarray
    rf_info: RFInfo
    dfe_channels: np.ndarray
    dfe_info: DFEInfo
    branches: dict[str, BranchResult]
    metrics: list[MetricRecord]
```

- [ ] **Step 4: Split `generate_scenario` without changing arithmetic**

Introduce these private boundaries:

```python
def _channel_delays_s(phone: PhoneConfig, estimated: bool) -> np.ndarray:
    delay_map = (
        phone.estimated_channel_delay_ns
        if estimated
        else phone.channel_delay_ns
    )
    return 1e-9 * np.asarray(
        [delay_map[name] for name in phone.active_antennas]
    )

def _desired_waveform(
    cfg: SimulationConfig,
    sample_index: np.ndarray,
    time_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    received_code_rate_hz = cfg.signal.code_rate_hz * (
        1.0 + cfg.signal.doppler_hz / cfg.signal.carrier_hz
    )
    code = ca_code(cfg.signal.prn)
    code_phase = np.mod(
        sample_index * received_code_rate_hz / cfg.signal.fs_rf_hz
        - cfg.signal.code_delay_chips,
        1023.0,
    )
    sampled_code = code[np.floor(code_phase).astype(int)]
    signal_power_w = 10.0 ** ((cfg.signal.power_dbm - 30.0) / 10.0)
    signal_amplitude_v = np.sqrt(
        signal_power_w * cfg.reference_impedance_ohm
    )
    carrier = np.exp(
        1j
        * (
            2.0
            * np.pi
            * (cfg.signal.if_hz + cfg.signal.doppler_hz)
            * time_s
            + cfg.signal.initial_phase_rad
        )
    )
    return signal_amplitude_v * sampled_code * carrier, code, received_code_rate_hz

@dataclass
class _InterfererScene:
    response: np.ndarray
    component: np.ndarray
    delay_phase: np.ndarray
    diagnostics: AntennaDiagnostics

def _interferer_scene(
    interferer: CWInterferer,
    cfg: SimulationConfig,
    antenna_model: AntennaModel,
    time_s: np.ndarray,
    true_delays_s: np.ndarray,
) -> _InterfererScene:
    response_base, diagnostics = antenna_response(
        antenna_model, interferer.direction, interferer.polarization
    )
    amplitude_v = np.sqrt(
        10.0 ** ((interferer.power_dbm - 30.0) / 10.0)
        * cfg.reference_impedance_ohm
    )
    waveform = amplitude_v * np.exp(
        1j
        * (
            2.0 * np.pi * (cfg.signal.if_hz + interferer.offset_hz) * time_s
            + interferer.initial_phase_rad
        )
    )
    gate = np.zeros(time_s.size)
    first = round(interferer.start_ms * 1e-3 * cfg.signal.fs_rf_hz)
    stop = (
        time_s.size
        if interferer.stop_ms is None
        else round(interferer.stop_ms * 1e-3 * cfg.signal.fs_rf_hz)
    )
    gate[first:stop] = 1.0
    delay = delay_phase(
        true_delays_s, cfg.signal.carrier_hz + interferer.offset_hz
    )
    response = response_base if np.all(true_delays_s == 0.0) else response_base * delay
    component = apply_channel_delays(
        response_base[:, None] * (waveform * gate)[None, :],
        true_delays_s,
        cfg.signal.fs_rf_hz,
        cfg.signal.carrier_hz,
        cfg.signal.if_hz,
    )
    return _InterfererScene(response, component, delay, diagnostics)
```

Keep the target and interferer expression order byte-for-byte where practical. `generate_scenario` remains responsible only for orchestration, scene summation in existing interferer order, and building the same `ScenarioTruth` dictionary.

- [ ] **Step 5: Annotate the remaining stage return dictionaries**

Import the appropriate contracts in RF, DFE, diversity, and acquisition modules and add return annotations. Do not change dictionary contents or algorithm branches.

- [ ] **Step 6: Run scenario and regression tests**

Run: `python -m pytest python_sim_simplified/tests/test_scenario.py python_sim_simplified/tests/test_default_regression.py -q`

Expected: all tests pass with unchanged metrics.

Run: `python -m pytest python_sim_simplified/tests -q`

Expected: all tests pass.

- [ ] **Step 7: Commit scenario contracts**

```bash
git add python_sim_simplified/models.py python_sim_simplified/gnss.py python_sim_simplified/rf.py python_sim_simplified/dfe.py python_sim_simplified/diversity.py python_sim_simplified/acquisition.py python_sim_simplified/tests/test_scenario.py
git commit -m "refactor: clarify simplified stage contracts"
```

---

### Task 5: Split Tracking and Pipeline Orchestration

**Files:**
- Modify: `python_sim_simplified/tracking.py`
- Modify: `python_sim_simplified/pipeline.py`
- Modify: `python_sim_simplified/metrics.py`
- Modify: `python_sim_simplified/tests/test_receiver.py`
- Modify: `python_sim_simplified/tests/test_pipeline.py`

**Interfaces:**
- Preserves: `track(x, fs_hz, prn, acquisition, cfg, carrier_hz, nominal_code_rate_hz) -> TrackingResult` keys and loop update order.
- Produces: `_run_branch(mode, dfe_channels, truth, cfg, fs_out_hz) -> BranchResult`.
- Preserves: `run_end_to_end(cfg=None) -> SimulationResults` and its public result keys.

- [ ] **Step 1: Add pipeline and tracking key-contract tests**

Assert that the top-level result keys remain `config`, `truth`, `scene_input_v`, `adc_codes`, `rf_info`, `dfe_channels`, `dfe_info`, `branches`, and `metrics`. For a successful branch, assert the current tracking keys from `success` through `cn0_db_hz` are unchanged.

- [ ] **Step 2: Run the contract tests before refactoring**

Run: `python -m pytest python_sim_simplified/tests/test_receiver.py python_sim_simplified/tests/test_pipeline.py -q`

Expected: PASS as characterization tests.

- [ ] **Step 3: Introduce private tracking state and helpers**

```python
@dataclass
class _TrackingState:
    carrier_frequency_hz: float
    carrier_phase_rad: float
    code_phase_chips: float
    code_rate_hz: float
    dll_frequency_correction_hz: float = 0.0
    previous_prompt: complex | None = None


def _correlate_epoch(
    block: np.ndarray,
    sample_index: np.ndarray,
    code: np.ndarray,
    state: _TrackingState,
    fs_hz: float,
    early_late_spacing_chips: float,
) -> tuple[complex, complex, complex]:
    carrier_phase = state.carrier_phase_rad + (
        2.0 * np.pi * state.carrier_frequency_hz * sample_index / fs_hz
    )
    baseband = block * np.exp(-1j * carrier_phase)
    prompt_phase = state.code_phase_chips + (
        sample_index * state.code_rate_hz / fs_hz
    )
    half_spacing = early_late_spacing_chips / 2.0
    early = np.sum(baseband * _sample_code(code, prompt_phase + half_spacing))
    prompt = np.sum(baseband * _sample_code(code, prompt_phase))
    late = np.sum(baseband * _sample_code(code, prompt_phase - half_spacing))
    return early, prompt, late

def _update_tracking_loops(
    state: _TrackingState,
    prompt: complex,
    dll_error: float,
    pll_error_rad: float,
    fll_error_hz: float,
    epoch: int,
    epoch_duration_s: float,
    samples_per_epoch: int,
    fs_hz: float,
    cfg: TrackingConfig,
    carrier_hz: float,
    nominal_code_rate_hz: float,
) -> None:
    fll_correction = 0.0
    if 0 < epoch < cfg.fll_assist_epochs:
        fll_correction = cfg.fll_gain * fll_error_hz
    state.carrier_frequency_hz += fll_correction + cfg.pll_ki_hz * pll_error_rad
    state.carrier_phase_rad = _wrap_phase(
        state.carrier_phase_rad
        + 2.0 * np.pi * state.carrier_frequency_hz * epoch_duration_s
        + cfg.pll_kp * pll_error_rad
    )
    state.dll_frequency_correction_hz += cfg.dll_ki_hz * dll_error
    state.code_rate_hz = nominal_code_rate_hz * (
        1.0 + state.carrier_frequency_hz / carrier_hz
    ) + state.dll_frequency_correction_hz
    state.code_phase_chips = np.mod(
        state.code_phase_chips
        + samples_per_epoch * state.code_rate_hz / fs_hz
        + cfg.dll_kp_chips * dll_error,
        1023.0,
    )
    state.previous_prompt = prompt
```

Maintain the exact order: compute correlators, derive DLL/PLL/FLL errors, update carrier frequency, update carrier phase, update DLL frequency, update code rate, update code phase, then store histories and C/N0.

- [ ] **Step 4: Extract single-branch pipeline orchestration**

Move mode selection, direction-gain diagnostics, acquisition, and conditional tracking into `_run_branch`. Keep `run_end_to_end` as the visible signal flow and construct the same runtime dictionaries.

- [ ] **Step 5: Add result annotations to metrics and pipeline**

Use `SimulationResults`, `BranchResult`, `TrackingResult`, and `MetricRecord` without changing format strings or displayed values.

- [ ] **Step 6: Run receiver, pipeline, regression, and full tests**

Run: `python -m pytest python_sim_simplified/tests/test_receiver.py python_sim_simplified/tests/test_pipeline.py python_sim_simplified/tests/test_default_regression.py -q`

Expected: all focused tests pass and all four branch metrics match the baseline.

Run: `python -m pytest python_sim_simplified/tests -q`

Expected: all tests pass.

- [ ] **Step 7: Commit orchestration cleanup**

```bash
git add python_sim_simplified/tracking.py python_sim_simplified/pipeline.py python_sim_simplified/metrics.py python_sim_simplified/tests/test_receiver.py python_sim_simplified/tests/test_pipeline.py
git commit -m "refactor: simplify tracking and pipeline flow"
```

---

### Task 6: Update Research Documentation and Verify the Deliverable

**Files:**
- Modify: `python_sim_simplified/README.md`
- Modify: root progress/documentation files only where they explicitly call this directory a teaching copy

**Interfaces:**
- Documents the new import path `from presets import default_config, parametric_config, measured_config` and the single-source parameter model.
- Preserves the direct runner command and seven output filenames.

- [ ] **Step 1: Rewrite the README positioning and parameter examples**

Use the title `GNSS 天线分集精简研究仿真版`. Explain that this directory is the user's concise research implementation, not a teaching artifact. Update examples so changing carrier/IF/sample rate/code rate requires editing only `cfg.signal`; document `cfg.dfe.nco_frequency_hz` as the optional DDC mismatch override and `cfg.reference_impedance_ohm` as the shared impedance.

- [ ] **Step 2: Search and update stale terminology**

Run: `rg -n "教学版|教学副本|teaching" python_sim_simplified docs README.md`

Update only statements that identify `python_sim_simplified` as a teaching copy. Do not rewrite historical documents whose purpose is to record past decisions; add a short current-status note instead when necessary.

- [ ] **Step 3: Run formatting-neutral checks and the full test suite**

Run: `python -m compileall -q python_sim_simplified`

Expected: exit code 0.

Run: `python -m pytest python_sim_simplified/tests -q`

Expected: all tests pass, including the locked baseline.

Run: `python python_sim_simplified/run_end_to_end.py`

Expected: exit code 0, the same metric table columns, and seven PNG files in `python_sim_simplified/results/`.

- [ ] **Step 4: Review the final diff for scope and accidental artifacts**

Run: `git diff --check origin/main...HEAD`

Run: `git status --short`

Confirm no generated PNG, `__pycache__`, `.pytest_cache`, or unrelated file is staged.

- [ ] **Step 5: Commit documentation**

```bash
git add python_sim_simplified/README.md docs
git commit -m "docs: describe simplified research simulation"
```

- [ ] **Step 6: Push, open PR, merge, and synchronize local main**

```bash
git push -u origin codex/simplified-research-refactor
gh pr create --base main --head codex/simplified-research-refactor --title "Refactor simplified research simulation" --body-file <prepared-pr-body-file>
gh pr checks <pr-number> --watch
gh pr merge <pr-number> --merge --delete-branch
git switch main
git pull --ff-only origin main
```

The PR body must summarize configuration ownership, antenna isolation, typed stage contracts, function decomposition, documentation changes, and the exact verification commands run.
