# Python Simplified Wideband Interference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, power-normalized, band-limited complex Gaussian interferer to `python_sim_simplified`, enabled alongside the existing CW interferer and propagated through the full receiver chain.

**Architecture:** Keep CW and wideband configurations as separate dataclasses. A focused `interference.py` module generates and validates one wideband waveform; `gnss.py` applies the existing antenna response, time gate, and per-channel true delay before summing all enabled interferers. Existing narrowband combiners remain unchanged, while plots distinguish tones from occupied bands and documentation states that wideband gain is not represented by one center-frequency number.

**Tech Stack:** Python 3, NumPy, SciPy FIR filtering, Matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-04-wideband-interference-design.md`

## Global Constraints

- Limit production changes to `python_sim_simplified`; do not modify `python_sim`.
- Preserve the user's uncommitted channel-delay edits in `python_sim_simplified/run_end_to_end.py`.
- Preserve the untracked `python_sim_simplified/.idea/` directory.
- Default wideband total power is `-90.0 dBm`, center offset is `0.0 Hz`, occupied bandwidth is `2.0e6 Hz`, seed is `1001`, and FIR length is `129` taps.
- Default configuration enables the existing CW interferer and the new wideband interferer simultaneously at azimuth `80.0 deg`, elevation `10.0 deg`.
- Do not implement swept, pulsed, modulated, time-varying-DOA, subband-MVDR, or tapped-delay-line MVDR behavior.
- `worst_interferer_gain` remains a center-frequency spatial diagnostic; acquisition, tracking, C/N0, and spectra are the wideband performance evidence.

---

### Task 1: Wideband configuration and waveform generator

**Files:**
- Create: `python_sim_simplified/interference.py`
- Modify: `python_sim_simplified/config.py`
- Create: `python_sim_simplified/tests/test_wideband_interference.py`

**Interfaces:**
- Produces: `WidebandInterferer` with fields `direction`, `enable`, `power_dbm`, `center_offset_hz`, `bandwidth_hz`, `random_seed`, `filter_taps`, `start_ms`, `stop_ms`, and `polarization`.
- Produces: `generate_wideband_waveform(interferer: WidebandInterferer, num_samples: int, fs_hz: float, if_hz: float, impedance_ohm: float) -> np.ndarray`.
- Guarantees: returned complex array has `num_samples` entries and mean enabled-waveform power `mean(abs(x)**2) / impedance_ohm` equal to the configured total watts before time gating.

- [ ] **Step 1: Write failing waveform tests**

Add tests that instantiate `WidebandInterferer(Direction(80.0, 10.0), bandwidth_hz=2.0e6, random_seed=1001)`, call `generate_wideband_waveform(..., num_samples=131072, fs_hz=16.368e6, if_hz=4.092e6, impedance_ohm=50.0)`, and assert:

```python
assert first.shape == (131072,)
np.testing.assert_array_equal(first, second)
assert not np.array_equal(first, different_seed)
expected_watts = 10.0 ** ((interferer.power_dbm - 30.0) / 10.0)
measured_watts = np.mean(np.abs(first) ** 2) / 50.0
assert measured_watts == pytest.approx(expected_watts, rel=1e-12)
```

Estimate FFT power after removing the `if_hz` center and assert at least `95%` of power lies in `abs(frequency_hz) <= bandwidth_hz / 2 * 1.10`. Add parameterized invalid cases for zero bandwidth, bandwidth not below `fs_hz`, even or sub-three tap count, negative seed, and a shifted occupied band crossing complex Nyquist.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m pytest tests/test_wideband_interference.py -q
```

Expected: collection fails because `WidebandInterferer` or `interference` does not exist.

- [ ] **Step 3: Implement the minimal generator**

In `config.py`, add the independent dataclass and define:

```python
Interferer = CWInterferer | WidebandInterferer
```

Change `SimulationConfig.interferers` to `list[Interferer]`.

In `interference.py`, validate finite scalar parameters, the odd tap count, nonnegative integer seed, and the occupied band. Generate `(I + 1j * Q) / sqrt(2)` using `np.random.default_rng(random_seed)`, prepend `filter_taps - 1` warm-up samples, filter with `scipy.signal.firwin` and `lfilter`, discard warm-up output, normalize RMS, scale from dBm through the supplied impedance, and modulate by `exp(1j * 2*pi*(if_hz + center_offset_hz)*n/fs_hz)`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_wideband_interference.py -q
```

Expected: all waveform, spectrum, reproducibility, power, and validation tests pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add -- python_sim_simplified/config.py python_sim_simplified/interference.py python_sim_simplified/tests/test_wideband_interference.py
git commit -m "feat: generate band-limited wideband interference"
```

---

### Task 2: Scenario integration and simultaneous default interferers

**Files:**
- Modify: `python_sim_simplified/gnss.py`
- Modify: `python_sim_simplified/presets.py`
- Modify: `python_sim_simplified/tests/test_config_gnss.py`
- Modify: `python_sim_simplified/tests/test_scenario.py`

**Interfaces:**
- Consumes: `WidebandInterferer` and `generate_wideband_waveform(...)` from Task 1.
- Preserves: `truth["interferer_components"]`, `truth["interferer_steering_effective"]`, `truth["interferer_delay_phases_true"]`, and antenna diagnostic lists in configuration order.
- Produces: one default CW component followed by one default wideband component.

- [ ] **Step 1: Write failing integration tests**

Update the default test to assert:

```python
assert len(cfg.interferers) == 2
assert isinstance(cfg.interferers[0], CWInterferer)
assert isinstance(cfg.interferers[1], WidebandInterferer)
assert cfg.interferers[1].bandwidth_hz == 2.0e6
```

Add a short single-antenna scenario test with only the wideband interferer, `duration_ms = 8`, `start_ms = 2.0`, and `stop_ms = 5.0`. Assert that samples outside the gate are zero in `truth["interferer_components"][0]`, samples inside are nonzero, and the true delay phase uses `carrier_hz + center_offset_hz`. Add a coexistence assertion that the default truth contains two components and two steering vectors in CW-then-wideband order.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m pytest tests/test_config_gnss.py tests/test_scenario.py -q
```

Expected: failures show that the preset still has one interferer and `gnss.py` assumes every interferer has `offset_hz`.

- [ ] **Step 3: Dispatch waveform generation by concrete type**

In `gnss.py`, keep one shared antenna/delay path. For `CWInterferer`, retain the existing sinusoid and use `offset_hz`. For `WidebandInterferer`, call `generate_wideband_waveform` and use `center_offset_hz` as the center-frequency diagnostic. Apply the same sample gate before `apply_channel_delays` and calculate the center-frequency delay phase with:

```python
delay_phase(true_delays_s, cfg.signal.carrier_hz + center_offset_hz)
```

In `presets.py`, append the approved default `WidebandInterferer` after the existing CW object.

- [ ] **Step 4: Run scenario tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_config_gnss.py tests/test_scenario.py tests/test_pipeline.py -q
```

Expected: all tests pass and both interferers reach the pipeline.

- [ ] **Step 5: Commit Task 2**

```powershell
git add -- python_sim_simplified/gnss.py python_sim_simplified/presets.py python_sim_simplified/tests/test_config_gnss.py python_sim_simplified/tests/test_scenario.py
git commit -m "feat: propagate wideband jammer through GNSS scenario"
```

---

### Task 3: Wideband spectrum and direction annotations

**Files:**
- Modify: `python_sim_simplified/plotting.py`
- Modify: `python_sim_simplified/tests/test_plotting_runner.py`

**Interfaces:**
- Consumes: `CWInterferer | WidebandInterferer` objects from `results["config"].interferers`.
- Produces: CW center lines and wideband occupied-band spans in both RF-domain and DFE-domain spectrum figures.
- Preserves: exactly sixteen detailed diagnostic figures plus `tracking` and `cn0` when saved.

- [ ] **Step 1: Write failing annotation tests**

Extend `test_detailed_combined_spectra_share_absolute_limits_and_mark_frequencies` to assert that every combined-spectrum axis has one wideband span and lines for the data, target, CW center, and wideband center:

```python
assert all(len(axis.patches) == 1 for axis in axes)
assert all(len(axis.lines) == 4 for axis in axes)
```

In the detailed spatial test, assert that enabled CW and wideband directions both appear as separate labeled markers while the figure count remains sixteen.

- [ ] **Step 2: Run plotting tests and verify RED**

Run:

```powershell
python -m pytest tests/test_plotting_runner.py -q
```

Expected: `_mark_frequencies` raises for missing `offset_hz`, or the new patch/line assertions fail.

- [ ] **Step 3: Implement type-aware annotations**

Import both interferer types in `plotting.py`. In `_mark_frequencies`, keep `axvline` for CW. For wideband, call `axvspan` from `center_offset_hz - bandwidth_hz/2` to `center_offset_hz + bandwidth_hz/2`, add the RF-domain IF shift when requested, and draw a dotted center line. Label direction markers as `CW 1` and `Wideband 2` without changing response calculations.

- [ ] **Step 4: Run plotting tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_plotting_runner.py -q
```

Expected: all plotting tests pass, detailed count remains sixteen, and saved output count remains eighteen.

- [ ] **Step 5: Commit Task 3**

```powershell
git add -- python_sim_simplified/plotting.py python_sim_simplified/tests/test_plotting_runner.py
git commit -m "feat: annotate wideband interference in plots"
```

---

### Task 4: Documentation, locked baseline, and full verification

**Files:**
- Modify: `python_sim_simplified/README.md`
- Modify: `python_sim_simplified/tests/test_default_regression.py`

**Interfaces:**
- Documents: editable wideband parameters, total-power convention, deterministic seed, simultaneous default CW, and fixed-weight narrowband-MVDR limitation.
- Updates: locked default metrics only from a freshly verified run after the intentional default-scenario change.

- [ ] **Step 1: Run the locked regression and verify intentional RED**

Run:

```powershell
python -m pytest tests/test_default_regression.py -q
```

Expected: metric assertions fail because the default scenario now contains an additional `-90 dBm`, `2 MHz` wideband interferer; shape and receiver-configuration assertions remain valid.

- [ ] **Step 2: Record the new deterministic baseline**

Run a Python command that prints every record from `run_end_to_end(default_config())["metrics"]`. Replace only the four `EXPECTED` tuples with the observed deterministic values; do not loosen `rel=1e-10` and do not change expected array shapes.

- [ ] **Step 3: Document configuration and limitations**

Add a README example importing `WidebandInterferer` and show edits to `power_dbm`, `center_offset_hz`, `bandwidth_hz`, `random_seed`, `start_ms`, and `stop_ms`. State explicitly that total power covers the entire configured band, the default combines CW plus wideband noise, one fixed MVDR weight vector covers the whole DFE band, and center-frequency spatial gain is not a band-wide suppression metric.

- [ ] **Step 4: Run complete verification**

Run:

```powershell
python -m pytest -q
python -m compileall -q .
```

Then run a shortened noninteractive full-chain case with `duration_ms = 20`, `tracking.max_epochs = 10`, and `plot.show = False`. Print each branch's acquisition metric and final C/N0, and verify that the saved result set still contains exactly eighteen nonempty PNG files.

Expected: the full test suite and compile step exit zero; the smoke run returns all four branches without exceptions; eighteen PNGs are present.

- [ ] **Step 5: Commit Task 4 without user-owned files**

```powershell
git add -- python_sim_simplified/README.md python_sim_simplified/tests/test_default_regression.py
git commit -m "docs: explain wideband interference simulation"
git status --short
```

Expected final status still lists the user's pre-existing `python_sim_simplified/run_end_to_end.py` modification and untracked `.idea/`, unless the user separately asks to include them.
