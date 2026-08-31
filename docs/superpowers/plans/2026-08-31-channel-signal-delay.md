# Channel Signal Delay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add configurable true and estimated per-channel sequence delay to the phone-array GNSS simulation, first in `python_sim/` and then in `python_sim_simplified/`, without changing zero-delay behavior.

**Architecture:** A focused fractional-delay module applies a 33-tap Hamming-windowed sinc interpolation to each complex-IF channel and restores the carrier phase removed by downconversion. Scenario generation keeps antenna spatial response, static channel response, true physical delay, and assumed calibration delay as separate factors; known-steering combiners consume the assumed center-frequency steering vector, while Blind Null only observes the delayed samples.

**Tech Stack:** Python 3.11+, NumPy, dataclasses, pytest, existing `antenna_diversity` package.

**Spec:** `docs/superpowers/specs/2026-08-31-channel-signal-delay-design.md`

## Global Constraints

- Implement and verify `python_sim/` before changing `python_sim_simplified/`.
- Add `channel_delay_ns` and `estimated_channel_delay_ns` for all `TL/TR/BL/BR` corners; defaults are exactly `0.0` ns.
- Full configuration accepts only finite, nonnegative delays and requires the exact four-corner key set.
- Use a 33-tap (32nd-order) Hamming-windowed sinc fractional-delay FIR.
- Zero delay takes an exact pass-through branch so the baseline is numerically unchanged.
- Sequence delay acts on each desired/interferer component; estimated delay changes assumed target steering only.
- Keep existing geometric spatial phase and static channel amplitude/phase errors; do not count delay twice.
- Do not add DFE delay compensation, time-varying delay, clock drift, multipath, negative delay, or MATLAB code.
- Preserve channels-by-samples array layout and arbitrary active 2–4 corner ordering.

## File Map

- Create `python_sim/antenna_diversity/gnss/channel_delay.py`: complete delay mathematics and array application helpers.
- Create `python_sim/tests/test_channel_delay.py`: focused numerical tests for the delay helpers.
- Modify `python_sim/antenna_diversity/config.py`: full-project true/estimated delay fields, defaults, and validation.
- Modify `python_sim/antenna_diversity/gnss/scenario.py`: apply true delays to source sequences and estimated delays to assumed steering.
- Modify `python_sim/tests/test_phone_config.py`: delay configuration/default/order/validation coverage.
- Modify `python_sim/tests/test_scenario.py`: desired-signal delay, true/estimated separation, zero-delay compatibility, and truth diagnostics.
- Modify `python_sim/tests/test_multi_interferer_scenario.py`: per-interferer RF-frequency phase coverage.
- Modify `python_sim/tests/test_study_config.py`: JSON parsing, snapshots, and scientific hash coverage.
- Modify `python_sim/tests/test_study_full_chain.py`: reconstructed study cases preserve both delay maps.
- Modify `python_sim/README.md`: full-project configuration example and modeling boundary.
- Create `python_sim_simplified/channel_delay.py`: readable copy of the same numerical model.
- Modify `python_sim_simplified/config.py`: matching teaching-copy fields and zero defaults.
- Modify `python_sim_simplified/gnss.py`: matching source-sequence and steering integration.
- Create `python_sim_simplified/tests/test_channel_delay.py`: simplified helper coverage.
- Modify `python_sim_simplified/tests/test_config_gnss.py`: zero defaults and configured order coverage.
- Modify `python_sim_simplified/tests/test_scenario.py`: simplified true/estimated sequence-delay behavior.
- Modify `python_sim_simplified/README.md`: editable delay example and non-compensation warning.
- Modify `PROJECT_STATUS.md`: record completion and explicitly list remaining DFE calibration work.

---

### Task 1: Full-project fractional-delay primitives

**Files:**
- Create: `python_sim/antenna_diversity/gnss/channel_delay.py`
- Create: `python_sim/tests/test_channel_delay.py`

**Interfaces:**
- Consumes: NumPy one-dimensional complex sequences and channels-by-samples matrices.
- Produces: `fractional_delay(signal, delay_samples, num_taps=33)`, `apply_channel_delays(component, delays_s, fs_hz, carrier_hz, if_hz)`, and `delay_phase(delays_s, source_rf_frequency_hz)`.

- [ ] **Step 1: Write focused failing tests**

```python
import numpy as np

from antenna_diversity.gnss.channel_delay import (
    apply_channel_delays,
    delay_phase,
    fractional_delay,
)


def test_zero_delay_is_an_exact_copy():
    signal = np.array([1 + 2j, -3 + 4j, 5 - 6j], dtype=np.complex128)
    delayed = fractional_delay(signal, 0.0)
    assert delayed is not signal
    np.testing.assert_array_equal(delayed, signal)


def test_integer_delay_zero_fills_front_and_truncates_tail():
    signal = np.arange(6, dtype=np.float64).astype(np.complex128)
    np.testing.assert_array_equal(
        fractional_delay(signal, 2.0),
        np.array([0, 0, 0, 1, 2, 3], dtype=np.complex128),
    )


def test_complex_if_delay_has_rf_equivalent_phase():
    fs_hz = 8.0e6
    if_hz = 1.0e6
    offset_hz = 100.0e3
    carrier_hz = 1_575.42e6
    delay_s = 0.375 / fs_hz
    n = np.arange(512)
    tone = np.exp(1j * 2.0 * np.pi * (if_hz + offset_hz) * n / fs_hz)

    actual = apply_channel_delays(
        tone[None, :], np.array([delay_s]), fs_hz, carrier_hz, if_hz
    )[0]
    expected = tone * np.exp(
        -1j * 2.0 * np.pi * (carrier_hz + offset_hz) * delay_s
    )

    np.testing.assert_allclose(actual[32:-32], expected[32:-32], atol=2e-3)
    np.testing.assert_allclose(
        delay_phase(np.array([delay_s]), carrier_hz + offset_hz),
        [np.exp(-1j * 2.0 * np.pi * (carrier_hz + offset_hz) * delay_s)],
    )
```

- [ ] **Step 2: Run the focused tests and confirm the missing-module failure**

Run:

```powershell
python_sim\.venv\Scripts\python.exe -m pytest python_sim/tests/test_channel_delay.py -v
```

Expected: collection fails because `antenna_diversity.gnss.channel_delay` does not exist.

- [ ] **Step 3: Implement the minimal delay module**

```python
"""Per-channel true-time delay for complex-IF source components."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


DEFAULT_FRACTIONAL_DELAY_TAPS = 33


def fractional_delay(
    signal: NDArray[np.complex128],
    delay_samples: float,
    num_taps: int = DEFAULT_FRACTIONAL_DELAY_TAPS,
) -> NDArray[np.complex128]:
    samples = np.asarray(signal, dtype=np.complex128)
    if delay_samples == 0.0:
        return samples.copy()
    if not np.isfinite(delay_samples) or delay_samples < 0.0:
        raise ValueError("delay_samples must be finite and nonnegative")
    if num_taps < 3 or num_taps % 2 == 0:
        raise ValueError("num_taps must be an odd integer of at least three")

    nearest_integer = round(delay_samples)
    if abs(delay_samples - nearest_integer) <= 1e-12:
        delay_samples = float(nearest_integer)

    integer_delay = int(np.floor(delay_samples))
    fraction = delay_samples - integer_delay
    output = np.zeros(samples.shape, dtype=np.complex128)
    if integer_delay >= samples.size:
        return output
    if fraction == 0.0:
        output[integer_delay:] = samples[: samples.size - integer_delay]
        return output

    half = num_taps // 2
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.sinc(offsets - fraction) * np.hamming(num_taps)
    kernel /= np.sum(kernel)
    full = np.convolve(samples, kernel, mode="full")
    interpolated = full[half : half + samples.size]
    output[integer_delay:] = interpolated[: samples.size - integer_delay]
    return output


def delay_phase(
    delays_s: NDArray[np.float64], source_rf_frequency_hz: float
) -> NDArray[np.complex128]:
    return np.exp(
        -1j
        * 2.0
        * np.pi
        * source_rf_frequency_hz
        * np.asarray(delays_s, dtype=np.float64)
    )


def apply_channel_delays(
    component: NDArray[np.complex128],
    delays_s: NDArray[np.float64],
    fs_hz: float,
    carrier_hz: float,
    if_hz: float,
) -> NDArray[np.complex128]:
    values = np.asarray(component, dtype=np.complex128)
    delays = np.asarray(delays_s, dtype=np.float64)
    if values.ndim != 2 or delays.shape != (values.shape[0],):
        raise ValueError("component and delays_s must describe the same channels")
    if np.all(delays == 0.0):
        return values.copy()

    result = np.empty(values.shape, dtype=np.complex128)
    residual_frequency_hz = carrier_hz - if_hz
    for channel, delay_s in enumerate(delays):
        result[channel] = fractional_delay(
            values[channel], delay_s * fs_hz
        ) * np.exp(-1j * 2.0 * np.pi * residual_frequency_hz * delay_s)
    return result
```

- [ ] **Step 4: Run the helper tests**

Run:

```powershell
python_sim\.venv\Scripts\python.exe -m pytest python_sim/tests/test_channel_delay.py -v
```

Expected: all three tests pass.

- [ ] **Step 5: Commit the numerical primitive**

```powershell
git add -- python_sim/antenna_diversity/gnss/channel_delay.py python_sim/tests/test_channel_delay.py
git commit -m "feat: add channel fractional delay primitive"
```

---

### Task 2: Full-project delay configuration and study persistence

**Files:**
- Modify: `python_sim/antenna_diversity/config.py`
- Modify: `python_sim/tests/test_phone_config.py`
- Modify: `python_sim/tests/test_study_config.py`
- Modify: `python_sim/tests/test_study_full_chain.py`

**Interfaces:**
- Consumes: the four phone-corner names and `active_antennas` ordering.
- Produces: `PhoneArrayConfig.channel_delay_ns` and `PhoneArrayConfig.estimated_channel_delay_ns`, both `dict[str, float]`.

- [ ] **Step 1: Add failing configuration tests**

Add assertions equivalent to:

```python
def test_phone_channel_delays_default_to_zero_for_all_corners():
    cfg = mate70_default_config()
    assert cfg.array.channel_delay_ns == {name: 0.0 for name in CORNERS}
    assert cfg.array.estimated_channel_delay_ns == {name: 0.0 for name in CORNERS}


def test_phone_delay_maps_require_exact_finite_nonnegative_four_corner_values():
    cfg = mate70_default_config()
    del cfg.array.channel_delay_ns["BR"]
    with pytest.raises(ValueError, match="corner-name"):
        cfg.validate()

    cfg = mate70_default_config()
    cfg.array.channel_delay_ns["TR"] = -0.1
    with pytest.raises(ValueError, match="delay.*nonnegative"):
        cfg.validate()

    cfg = mate70_default_config()
    cfg.array.estimated_channel_delay_ns["BL"] = np.inf
    with pytest.raises(ValueError, match="delay.*finite"):
        cfg.validate()

    cfg = mate70_default_config()
    cfg.array.channel_delay_ns = {name: 5.0 for name in CORNERS}
    cfg.array.estimated_channel_delay_ns = {name: 5.0 for name in CORNERS}
    cfg.validate()
```

In `test_study_config.py`, write a JSON file whose `simulation.array` object contains complete `channel_delay_ns` and `estimated_channel_delay_ns` maps, resolve it, and assert both maps survive and change `config_sha256` relative to the zero-delay preset. In `test_study_full_chain.py`, set distinct true and estimated delays before `ResolvedStudyConfig.from_parts(...)` and assert `_reconstruct_case(...)` preserves them.

- [ ] **Step 2: Run the focused configuration tests and confirm failure**

Run:

```powershell
python_sim\.venv\Scripts\python.exe -m pytest python_sim/tests/test_phone_config.py python_sim/tests/test_study_config.py python_sim/tests/test_study_full_chain.py -q
```

Expected: tests fail because the two delay fields are absent.

- [ ] **Step 3: Add fields, defaults, and root validation**

Extend `PhoneArrayConfig` before its defaulted fields:

```python
channel_delay_ns: dict[str, float]
estimated_channel_delay_ns: dict[str, float]
```

Extend `mate70_default_config()` construction:

```python
channel_delay_ns=zero_errors.copy(),
estimated_channel_delay_ns=zero_errors.copy(),
```

Keep error and delay validation separate while using all maps for the exact-key check:

```python
channel_error_maps = (
    self.array.channel_amplitude_error_db,
    self.array.channel_phase_error_deg,
    self.array.estimated_channel_amplitude_error_db,
    self.array.estimated_channel_phase_error_deg,
)
delay_maps = (
    self.array.channel_delay_ns,
    self.array.estimated_channel_delay_ns,
)
corner_maps = (
    self.phone.assumed_phase_centers_m,
    *channel_error_maps,
    *delay_maps,
)
if corner_names != PHONE_CORNER_NAMES or any(
    set(corner_map) != PHONE_CORNER_NAMES for corner_map in corner_maps
):
    raise ValueError(
        "phone coordinate, channel-error, and delay maps must use the same corner-name set"
    )
if any(
    not np.isfinite(error)
    for error_map in channel_error_maps
    for error in error_map.values()
):
    raise ValueError("phone channel errors must be finite")
if any(
    not _is_finite_real_scalar(delay)
    for delay_map in delay_maps
    for delay in delay_map.values()
):
    raise ValueError("phone channel delays must be finite real values")
if any(delay < 0.0 for delay_map in delay_maps for delay in delay_map.values()):
    raise ValueError("phone channel delays must be nonnegative")
```

No special study serializer is required because `strict_snapshot()` walks dataclass fields and `_canonical_simulation()` deep-copies the array. The new tests must prove that existing generic JSON update, snapshot, hash, and reconstruction paths retain the fields.

- [ ] **Step 4: Run focused configuration and study tests**

Run:

```powershell
python_sim\.venv\Scripts\python.exe -m pytest python_sim/tests/test_phone_config.py python_sim/tests/test_study_config.py python_sim/tests/test_study_full_chain.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit configuration support**

```powershell
git add -- python_sim/antenna_diversity/config.py python_sim/tests/test_phone_config.py python_sim/tests/test_study_config.py python_sim/tests/test_study_full_chain.py
git commit -m "feat: configure phone channel delays"
```

---

### Task 3: Full-project scenario integration

**Files:**
- Modify: `python_sim/antenna_diversity/gnss/scenario.py`
- Modify: `python_sim/tests/test_scenario.py`
- Modify: `python_sim/tests/test_multi_interferer_scenario.py`

**Interfaces:**
- Consumes: Task 1 helpers and Task 2 delay maps.
- Produces: delayed `desired_component`, delayed `interferer_components`, center-frequency delay-aware steering vectors, and ordered delay diagnostics.

- [ ] **Step 1: Add failing scenario tests**

Cover these exact outcomes with 1 ms scenes and plotting disabled:

```python
def test_true_delay_shifts_samples_but_estimated_delay_only_changes_assumed_steering():
    cfg = mate70_default_config()
    cfg.signal.duration_ms = 1
    cfg.array.active_antennas = ("TL", "TR")
    cfg.interferers = ()
    one_sample_ns = 1e9 / cfg.signal.fs_rf_hz
    cfg.array.channel_delay_ns["TR"] = one_sample_ns

    actual, truth = generate_scenario(cfg)
    np.testing.assert_allclose(truth["channel_delay_true_samples"], [0.0, 1.0])
    np.testing.assert_allclose(
        truth["channel_delay_estimated_samples"], [0.0, 0.0]
    )
    assert truth["desired_component"][1, 0] == 0.0j

    cfg.array.estimated_channel_delay_ns["TR"] = one_sample_ns / 2.0
    changed, changed_truth = generate_scenario(cfg)
    np.testing.assert_array_equal(changed, actual)
    np.testing.assert_array_equal(
        changed_truth["desired_component"], truth["desired_component"]
    )
    assert not np.array_equal(
        changed_truth["desired_steering_assumed"],
        truth["desired_steering_assumed"],
    )
```

Also add tests that:

- default zero-delay `desired_component` remains exactly `desired_steering_true[:, None] * desired_waveform[None, :]`;
- parameterized active sets `("TL", "BR")`, `("BR", "TL", "TR")`, and `("TL", "TR", "BL", "BR")` produce true/estimated delay arrays in the exact configured order;
- changing a true relative delay changes the actual array samples and `x @ x.conj().T / n_samples`, while changing only the estimated delay leaves both unchanged;
- a CW interferer with `offset_hz=250e3` has `interferer_steering_effective[0]` multiplied by `exp(-j2π(carrier_hz + offset_hz)tau)` relative to its zero-delay case;
- a one-sample delayed gated interferer moves its leading gate edge by one sample rather than merely rotating it.

- [ ] **Step 2: Run the scenario tests and confirm failure**

Run:

```powershell
python_sim\.venv\Scripts\python.exe -m pytest python_sim/tests/test_scenario.py python_sim/tests/test_multi_interferer_scenario.py -q
```

Expected: new assertions fail because scenario generation does not yet read delay maps or emit delay diagnostics.

- [ ] **Step 3: Integrate ordered delay vectors and desired-signal processing**

Import Task 1 helpers and derive ordered seconds once per scene:

```python
if isinstance(cfg.array, PhoneArrayConfig):
    true_delays_s = 1e-9 * np.asarray(
        [cfg.array.channel_delay_ns[name] for name in cfg.array.active_antennas],
        dtype=np.float64,
    )
    estimated_delays_s = 1e-9 * np.asarray(
        [
            cfg.array.estimated_channel_delay_ns[name]
            for name in cfg.array.active_antennas
        ],
        dtype=np.float64,
    )
else:
    true_delays_s = np.zeros(n_antennas, dtype=np.float64)
    estimated_delays_s = np.zeros(n_antennas, dtype=np.float64)
```

Rename the values returned from `_desired_responses()` to `desired_assumed_base` and `desired_true_base`, then apply:

```python
desired_component = apply_channel_delays(
    desired_true_base[:, None] * desired_waveform[None, :],
    true_delays_s,
    cfg.signal.fs_rf_hz,
    cfg.signal.carrier_hz,
    cfg.signal.if_hz,
)
desired_delay_phase_true = delay_phase(
    true_delays_s, cfg.signal.carrier_hz + cfg.signal.doppler_hz
)
desired_delay_phase_assumed = delay_phase(
    estimated_delays_s, cfg.signal.carrier_hz + cfg.signal.doppler_hz
)
desired_true = desired_true_base * desired_delay_phase_true
desired_assumed = desired_assumed_base * desired_delay_phase_assumed
```

- [ ] **Step 4: Integrate each interferer independently and expose diagnostics**

For each interferer, preserve `effective_base`, then use:

```python
source_rf_frequency_hz = cfg.signal.carrier_hz + interferer.offset_hz
interferer_delay_phase = delay_phase(true_delays_s, source_rf_frequency_hz)
effective = effective_base * interferer_delay_phase
component = apply_channel_delays(
    effective_base[:, None] * waveform[None, :],
    true_delays_s,
    cfg.signal.fs_rf_hz,
    cfg.signal.carrier_hz,
    cfg.signal.if_hz,
)
```

Append `effective`, `component`, and `interferer_delay_phase` to their respective lists. Add these exact truth keys:

```python
"channel_delay_true_ns": true_delays_s * 1e9,
"channel_delay_estimated_ns": estimated_delays_s * 1e9,
"channel_delay_true_samples": true_delays_s * cfg.signal.fs_rf_hz,
"channel_delay_estimated_samples": estimated_delays_s * cfg.signal.fs_rf_hz,
"desired_delay_phase_true": desired_delay_phase_true,
"desired_delay_phase_assumed": desired_delay_phase_assumed,
"interferer_delay_phases_true": interferer_delay_phases_true,
```

Keep `desired_steering_effective` and `jammer_steering_effective` as aliases of the delay-aware true steering values. Keep the legacy ULA path all-zero and unchanged.

- [ ] **Step 5: Run scenario and pipeline regressions**

Run:

```powershell
python_sim\.venv\Scripts\python.exe -m pytest python_sim/tests/test_channel_delay.py python_sim/tests/test_scenario.py python_sim/tests/test_multi_interferer_scenario.py python_sim/tests/test_end_to_end.py python_sim/tests/test_blind_null.py -q
```

Expected: all selected tests pass; default end-to-end and Blind Null behavior remains green.

- [ ] **Step 6: Commit full scenario integration**

```powershell
git add -- python_sim/antenna_diversity/gnss/scenario.py python_sim/tests/test_scenario.py python_sim/tests/test_multi_interferer_scenario.py
git commit -m "feat: delay full-project channel sequences"
```

---

### Task 4: Full-project documentation and complete regression

**Files:**
- Modify: `python_sim/README.md`

**Interfaces:**
- Consumes: completed full-project delay behavior.
- Produces: a copyable configuration example and explicit statement that DFE does not compensate delay.

- [ ] **Step 1: Add the documented configuration example**

Add a short section near the active-antenna example:

```python
cfg = default_config()
cfg.array.active_antennas = ("TL", "TR")
cfg.array.channel_delay_ns["TR"] = 10.0
cfg.array.estimated_channel_delay_ns["TR"] = 8.0
```

State that the first map delays actual target/interferer sequences, the second map changes known target steering, Blind Null only observes the resulting samples, and the current DFE performs no inter-channel time alignment.

- [ ] **Step 2: Run the full project with warnings promoted to errors**

Run:

```powershell
python_sim\.venv\Scripts\python.exe -W error -m pytest python_sim/tests -q
```

Expected: the previous 680 tests plus newly added tests all pass with no warnings.

- [ ] **Step 3: Commit the full-project documentation**

```powershell
git add -- python_sim/README.md
git commit -m "docs: explain channel sequence delays"
```

---

### Task 5: Simplified-project parity

**Files:**
- Create: `python_sim_simplified/channel_delay.py`
- Create: `python_sim_simplified/tests/test_channel_delay.py`
- Modify: `python_sim_simplified/config.py`
- Modify: `python_sim_simplified/gnss.py`
- Modify: `python_sim_simplified/tests/test_config_gnss.py`
- Modify: `python_sim_simplified/tests/test_scenario.py`
- Modify: `python_sim_simplified/README.md`

**Interfaces:**
- Consumes: the verified full-project formulas and truth-key names.
- Produces: the same two delay maps, 33-tap interpolation, RF-equivalent phase, and diagnostics in the teaching copy.

- [ ] **Step 1: Write simplified-project parity tests**

Copy the three numerical expectations from `python_sim/tests/test_channel_delay.py` using:

```python
from channel_delay import apply_channel_delays, delay_phase, fractional_delay
```

Extend config/scenario tests with:

```python
def test_default_channel_delays_are_zero():
    cfg = default_config()
    assert cfg.phone.channel_delay_ns == {name: 0.0 for name in CORNER_NAMES}
    assert cfg.phone.estimated_channel_delay_ns == {
        name: 0.0 for name in CORNER_NAMES
    }


def test_estimated_delay_changes_assumed_steering_not_actual_samples():
    cfg = default_config()
    cfg.signal.duration_ms = 1
    cfg.phone.active_antennas = ("TL", "TR")
    cfg.phone.channel_delay_ns["TR"] = 1e9 / cfg.signal.fs_rf_hz
    first, first_truth = generate_scenario(cfg)

    cfg.phone.estimated_channel_delay_ns["TR"] = 5.0
    second, second_truth = generate_scenario(cfg)

    np.testing.assert_array_equal(second, first)
    np.testing.assert_array_equal(
        second_truth["desired_component"], first_truth["desired_component"]
    )
    assert not np.array_equal(
        second_truth["desired_steering_assumed"],
        first_truth["desired_steering_assumed"],
    )
```

- [ ] **Step 2: Run simplified tests and confirm failure**

Run:

```powershell
python_sim\.venv\Scripts\python.exe -m pytest python_sim_simplified/tests/test_channel_delay.py python_sim_simplified/tests/test_config_gnss.py python_sim_simplified/tests/test_scenario.py -q
```

Expected: tests fail because the simplified helper and delay fields do not exist.

- [ ] **Step 3: Add the readable helper and configuration fields**

Copy the public constants and three functions from the verified full helper without importing the full package. Add to `PhoneConfig`:

```python
channel_delay_ns: dict[str, float]
estimated_channel_delay_ns: dict[str, float]
```

Pass two additional `zero.copy()` maps in `default_config()`. Do not add a second full validation framework to the teaching copy.

- [ ] **Step 4: Mirror full scenario integration**

In `python_sim_simplified/gnss.py`, derive delay vectors in `cfg.phone.active_antennas` order, apply actual delay to target and each interferer, apply estimated delay only to `desired_steering_assumed`, and emit the same seven truth keys defined in Task 3. Use `carrier_hz + doppler_hz` for desired steering and `carrier_hz + offset_hz` for each interferer.

- [ ] **Step 5: Document the same editable example**

Extend the existing “常用修改” block:

```python
cfg.phone.channel_delay_ns["BR"] = 10.0
cfg.phone.estimated_channel_delay_ns["BR"] = 8.0
```

Add one sentence that the simplified DFE does not align channels automatically.

- [ ] **Step 6: Run the complete simplified suite and standalone contract**

Run:

```powershell
python_sim\.venv\Scripts\python.exe -W error -m pytest python_sim_simplified/tests -q
```

Expected: the previous 23 tests plus the new delay tests pass, including standalone-copy execution.

- [ ] **Step 7: Commit simplified parity**

```powershell
git add -- python_sim_simplified/channel_delay.py python_sim_simplified/config.py python_sim_simplified/gnss.py python_sim_simplified/README.md python_sim_simplified/tests/test_channel_delay.py python_sim_simplified/tests/test_config_gnss.py python_sim_simplified/tests/test_scenario.py
git commit -m "feat: mirror channel delays in simplified simulation"
```

---

### Task 6: Project status and final verification

**Files:**
- Modify: `PROJECT_STATUS.md`

**Interfaces:**
- Consumes: verified full and simplified implementations.
- Produces: an accurate project-stage record and a clean local branch ready for user-directed push/PR.

- [ ] **Step 1: Update the project status record**

Record that true per-channel sequence delay and estimated steering delay now exist in both Python projects. Under remaining work, retain DFE channel-delay estimation/alignment, time-varying skew, and multipath as future stages; do not claim those are modeled.

- [ ] **Step 2: Run both complete suites once more**

Run:

```powershell
python_sim\.venv\Scripts\python.exe -W error -m pytest python_sim/tests -q
python_sim\.venv\Scripts\python.exe -W error -m pytest python_sim_simplified/tests -q
```

Expected: both suites pass with no warnings.

- [ ] **Step 3: Check patch quality and repository state**

Run:

```powershell
git diff --check
git status --short
git log --oneline -7
```

Expected: `git diff --check` is silent; only `PROJECT_STATUS.md` is uncommitted before the final commit; log shows the isolated feature commits.

- [ ] **Step 4: Commit status and verification record**

```powershell
git add -- PROJECT_STATUS.md
git commit -m "docs: update channel delay project status"
```

- [ ] **Step 5: Confirm clean local completion**

Run:

```powershell
git status --short
git log -1 --oneline
```

Expected: working tree is clean. Do not push or open a pull request until the user explicitly requests remote integration.
