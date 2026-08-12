# Python GPS L1 C/A RF→DFE→TRK Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不修改 `matlab_sim` 的前提下，建立可独立运行和测试的 Python GPS L1 C/A 双天线 RF→DFE→分集合并→捕获→跟踪仿真链路。

**Architecture:** 新建 `python_sim`，按 MATLAB 当前算法边界拆分为小型 Python 包；阶段函数用 NumPy 数组传递信号，用嵌套 dataclass 传递配置，用少量顶层结果 dataclass 汇总端到端输出。单元测试先固定每层的数学行为，再由端到端测试验证默认强干扰场景中 single/EGC 失败而 MVDR 成功的结论。

**Tech Stack:** Python 3.11+、NumPy 2.x、SciPy 1.x、Matplotlib 3.x、pytest 9.x。

## Global Constraints

- `matlab_sim` 只读，不修改、移动或删除其中任何文件。
- 第一阶段只迁移 GPS L1 C/A；不导入 `matlab_sim/+galileo` 和 Galileo E5/AltBOC。
- 多通道数组形状统一为 `(n_antennas, n_samples)`，单路数组统一为 `(n_samples,)`。
- 默认使用 `float64`/`complex128`；量化复数的 I/Q 分量必须保持整数值。
- 公开的码相位和历元起点使用零基采样索引。
- ADC 与 DFE 量化使用 MATLAB 风格“中点远离零”的舍入。
- 固定 NumPy 随机种子必须可复现，不要求与 MATLAB 噪声逐样点一致。
- 不引入 pandas、Numba、JAX、GPU 或实时流式依赖。
- 当前工作目录不是 Git 仓库，不自行初始化仓库；每项任务以独立 pytest 验证作为检查点。

---

### Task 1: Python package foundation, configuration, and numeric conventions

**Files:**
- Create: `python_sim/pyproject.toml`
- Create: `python_sim/antenna_diversity/__init__.py`
- Create: `python_sim/antenna_diversity/config.py`
- Create: `python_sim/antenna_diversity/models.py`
- Create: `python_sim/antenna_diversity/numeric.py`
- Create: `python_sim/tests/test_config_numeric.py`

**Interfaces:**
- Produces: `default_config() -> SimulationConfig`
- Produces: `matlab_round(values: ArrayLike) -> numpy.ndarray`
- Produces: `BranchResult` and `SimulationResults` dataclasses used by the pipeline.

- [ ] **Step 1: Create package metadata with exact runtime dependencies**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "antenna-diversity-sim"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["numpy>=2.0", "scipy>=1.14", "matplotlib>=3.9"]

[project.optional-dependencies]
test = ["pytest>=9.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Write failing configuration and rounding tests**

```python
import numpy as np
import pytest

from antenna_diversity.config import default_config
from antenna_diversity.numeric import matlab_round


def test_default_config_matches_matlab_sampling_constraints():
    cfg = default_config()
    assert cfg.signal.fs_rf_hz == 16.368e6
    assert cfg.dfe.fs_out_hz == 4.092e6
    assert cfg.dfe.fs_out_hz / cfg.signal.code_rate_hz == 4
    assert cfg.dfe.num_taps == 129
    assert cfg.diversity.modes == ("single", "egc", "mvdr")


def test_matlab_round_moves_half_ties_away_from_zero():
    actual = matlab_round(np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]))
    np.testing.assert_array_equal(actual, [-3, -2, -1, 1, 2, 3])


def test_invalid_even_fir_length_is_rejected():
    cfg = default_config()
    cfg.dfe.num_taps = 128
    with pytest.raises(ValueError, match="odd"):
        cfg.validate()
```

- [ ] **Step 3: Run tests and verify missing modules fail**

Run: `cd python_sim; python -m pytest tests/test_config_numeric.py -v`

Expected: collection fails because `antenna_diversity.config` and `numeric` do not exist.

- [ ] **Step 4: Implement dataclass configuration and MATLAB rounding**

Implement all values from `matlab_sim/default_config.m`. Derived values are properties rather than separately mutable copies:

```python
@dataclass
class DFEConfig:
    fs_in_hz: float
    if_hz: float
    decimation: int = 4
    cutoff_hz: float = 1.8e6
    num_taps: int = 129
    input_bits: int = 10
    output_bits: int = 8
    output_binary_shift: int = 2

    @property
    def fs_out_hz(self) -> float:
        return self.fs_in_hz / self.decimation
```

```python
def matlab_round(values):
    values = np.asarray(values)
    return np.sign(values) * np.floor(np.abs(values) + 0.5)
```

`SimulationConfig.validate()` checks the integer samples per code, integer samples per chip, at least two samples per chip, positive physical values, and odd FIR length.

- [ ] **Step 5: Implement top-level result dataclasses**

```python
@dataclass
class BranchResult:
    combined: np.ndarray
    combining: dict[str, Any]
    acquisition: dict[str, Any]
    tracking: dict[str, Any]


@dataclass
class SimulationResults:
    config: SimulationConfig
    truth: dict[str, Any]
    scene_input_v: np.ndarray
    adc_codes: np.ndarray
    rf_info: dict[str, Any]
    dfe_channels: np.ndarray
    dfe_info: dict[str, Any]
    branches: dict[str, BranchResult]
    metrics: list[dict[str, Any]]
```

- [ ] **Step 6: Run the foundation tests**

Run: `cd python_sim; python -m pytest tests/test_config_numeric.py -v`

Expected: all tests pass.

### Task 2: GPS C/A code, steering vector, and dual-antenna scenario

**Files:**
- Create: `python_sim/antenna_diversity/gnss/__init__.py`
- Create: `python_sim/antenna_diversity/gnss/ca_code.py`
- Create: `python_sim/antenna_diversity/gnss/scenario.py`
- Create: `python_sim/antenna_diversity/array/__init__.py`
- Create: `python_sim/antenna_diversity/array/steering.py`
- Create: `python_sim/tests/test_core.py`
- Create: `python_sim/tests/test_scenario.py`

**Interfaces:**
- Consumes: `SimulationConfig` from Task 1.
- Produces: `ca_code(prn: int) -> np.ndarray`
- Produces: `steering_vector(positions_m, angle_deg, wavelength_m) -> np.ndarray`
- Produces: `generate_scenario(cfg: SimulationConfig) -> tuple[np.ndarray, dict[str, Any]]`

- [ ] **Step 1: Write C/A code and steering tests**

```python
def test_prn1_has_expected_gold_code_autocorrelation():
    code = ca_code(1)
    assert code.shape == (1023,)
    assert set(np.unique(code)) == {-1.0, 1.0}
    correlation = np.fft.ifft(np.abs(np.fft.fft(code)) ** 2).real
    assert correlation[0] == pytest.approx(1023.0)
    assert set(np.rint(correlation[1:]).astype(int)) == {-65, -1, 63}


def test_broadside_referenced_steering_phase():
    cfg = default_config()
    a = steering_vector([0.0, cfg.array.spacing_m], 30.0, cfg.signal.wavelength_m)
    expected = np.exp(-1j * 2 * np.pi * cfg.array.spacing_m * np.sin(np.deg2rad(30)) / cfg.signal.wavelength_m)
    np.testing.assert_allclose(a, [1.0, expected], atol=1e-12)
```

- [ ] **Step 2: Run core tests and observe missing implementations**

Run: `cd python_sim; python -m pytest tests/test_core.py -v`

Expected: import failure for `ca_code` or `steering_vector`.

- [ ] **Step 3: Translate the two 10-stage LFSRs and array phase convention**

Use the exact PRN tap table from `matlab_sim/+gnss/ca_code.m`. Validate integer PRN 1 through 32. Convert MATLAB tap indices to Python indices exactly once when accessing G2.

```python
def steering_vector(positions_m, angle_deg, wavelength_m):
    positions = np.asarray(positions_m, dtype=float).reshape(-1)
    if positions.size == 0 or not np.isfinite(angle_deg) or wavelength_m <= 0:
        raise ValueError("positions, angle, and wavelength must be valid")
    return np.exp(-1j * 2 * np.pi * positions * np.sin(np.deg2rad(angle_deg)) / wavelength_m)
```

- [ ] **Step 4: Write scenario tests**

```python
def test_scenario_projects_signal_through_both_antennas():
    cfg = default_config()
    cfg.jammer.enable = False
    x, truth = generate_scenario(cfg)
    expected_samples = round(cfg.signal.fs_rf_hz * cfg.signal.duration_ms * 1e-3)
    assert x.shape == (cfg.array.num_elements, expected_samples)
    expected_rate = cfg.signal.code_rate_hz * (1 + cfg.signal.doppler_hz / cfg.signal.carrier_hz)
    assert truth["code_rate_hz"] == pytest.approx(expected_rate)
    ratio = np.mean(x[1] / x[0])
    expected_ratio = truth["desired_steering_effective"][1] / truth["desired_steering_effective"][0]
    assert ratio == pytest.approx(expected_ratio, abs=1e-10)
```

- [ ] **Step 5: Implement signal, channel-error, and jammer generation**

Preserve these equations:

```python
code_phase = np.mod(
    n * received_code_rate_hz / cfg.signal.fs_rf_hz - cfg.signal.code_delay_chips,
    1023.0,
)
sampled_code = code[np.floor(code_phase).astype(int)]
channel_response = 10 ** (amplitude_error_db / 20) * np.exp(1j * np.deg2rad(phase_error_deg))
desired_component = desired_steering_effective[:, None] * desired_waveform[None, :]
```

Return the same truth concepts as MATLAB, using snake_case keys.

- [ ] **Step 6: Run core and scenario tests**

Run: `cd python_sim; python -m pytest tests/test_core.py tests/test_scenario.py -v`

Expected: all tests pass.

### Task 3: RF cascade, thermal noise, limiting, and ADC

**Files:**
- Create: `python_sim/antenna_diversity/rf/__init__.py`
- Create: `python_sim/antenna_diversity/rf/cascade.py`
- Create: `python_sim/antenna_diversity/rf/frontend.py`
- Create: `python_sim/tests/test_rf.py`

**Interfaces:**
- Consumes: `RFConfig`, `RFStage`, and `matlab_round`.
- Produces: `cascade_metrics(stages) -> dict[str, Any]`
- Produces: `process_frontend(x_antenna_v, cfg, random_seed) -> tuple[np.ndarray, dict[str, Any]]`

- [ ] **Step 1: Write RF regression tests**

```python
def test_single_stage_cascade_metrics():
    result = cascade_metrics([RFStage("OnlyStage", 20.0, 2.0)])
    assert result["total_gain_db"] == pytest.approx(20.0)
    assert result["noise_figure_db"] == pytest.approx(2.0)


def test_frontend_is_repeatable_and_quantized():
    cfg = default_config()
    cfg.jammer.enable = False
    x, _ = generate_scenario(cfg)
    adc1, info1 = process_frontend(x, cfg.rf, cfg.random_seed)
    adc2, info2 = process_frontend(x, cfg.rf, cfg.random_seed)
    np.testing.assert_array_equal(adc1, adc2)
    np.testing.assert_array_equal(info1["saturation_fraction"], info2["saturation_fraction"])
    assert adc1.shape == x.shape
    assert np.all((-512 <= adc1.real) & (adc1.real <= 511))
    assert np.all((-512 <= adc1.imag) & (adc1.imag <= 511))
    np.testing.assert_array_equal(adc1.real, np.rint(adc1.real))
```

- [ ] **Step 2: Run the RF tests and verify imports fail**

Run: `cd python_sim; python -m pytest tests/test_rf.py -v`

Expected: missing `rf` functions.

- [ ] **Step 3: Implement Friis cascade and local seeded noise source**

```python
noise_power_input_w = (
    1.380649e-23 * cfg.temperature_k * cfg.noise_bandwidth_hz * cascade["noise_factor"]
)
rng = np.random.Generator(np.random.MT19937(random_seed))
noise = noise_rms_v / np.sqrt(2) * (
    rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)
)
```

Implement soft limiting with component-wise `tanh`, hard limiting with `np.clip`, and signed ADC ranges using `matlab_round`.

- [ ] **Step 4: Run the RF tests**

Run: `cd python_sim; python -m pytest tests/test_rf.py -v`

Expected: all tests pass with no runtime warnings.

### Task 4: DDC, windowed-sinc FIR, delay alignment, decimation, and requantization

**Files:**
- Create: `python_sim/antenna_diversity/dfe/__init__.py`
- Create: `python_sim/antenna_diversity/dfe/decimator.py`
- Create: `python_sim/antenna_diversity/dfe/processing.py`
- Create: `python_sim/tests/test_dfe.py`

**Interfaces:**
- Consumes: complex ADC array, `DFEConfig`, and `matlab_round`.
- Produces: `design_decimator(fs_in_hz, decimation, cutoff_hz, num_taps) -> tuple[np.ndarray, dict]`
- Produces: `process_channels(adc_codes, cfg) -> tuple[np.ndarray, dict]`

- [ ] **Step 1: Write FIR and DFE tests**

```python
def test_decimator_has_unity_dc_gain_and_expected_delay():
    h, info = design_decimator(16.368e6, 4, 1.8e6, 129)
    assert h.shape == (129,)
    assert np.sum(h) == pytest.approx(1.0)
    assert info["group_delay_input_samples"] == 64
    assert info["fs_out_hz"] == pytest.approx(4.092e6)


def test_dfe_moves_if_to_dc_and_preserves_channel_phase():
    cfg = default_config()
    cfg.dfe.output_bits = 0
    n_samples = round(cfg.dfe.fs_in_hz * 2e-3)
    n = np.arange(n_samples)
    tone = 100 * np.exp(1j * 2 * np.pi * cfg.dfe.if_hz * n / cfg.dfe.fs_in_hz)
    y, info = process_channels(np.vstack([tone, tone * np.exp(1j * 0.37)]), cfg.dfe)
    expected_length = (n_samples - info["group_delay_input_samples"] - 1) // cfg.dfe.decimation + 1
    assert y.shape == (2, expected_length)
    assert np.mean(y[1] / y[0]) == pytest.approx(np.exp(1j * 0.37), abs=1e-10)
```

- [ ] **Step 2: Run the DFE tests and verify missing code fails**

Run: `cd python_sim; python -m pytest tests/test_dfe.py -v`

Expected: missing DFE functions.

- [ ] **Step 3: Implement the exact Hamming-windowed normalized sinc**

```python
group_delay = (num_taps - 1) // 2
m = np.arange(-group_delay, group_delay + 1)
u = 2 * cutoff_hz / fs_in_hz * m
h = (2 * cutoff_hz / fs_in_hz) * np.sinc(u)
window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(num_taps) / (num_taps - 1))
h = h * window
h = h / np.sum(h)
```

- [ ] **Step 4: Implement common-NCO mixing and causal filtering**

Use `scipy.signal.lfilter(h, [1.0], mixed, axis=1)`. The first valid Python index is `group_delay`; decimate with:

```python
aligned = filtered[:, group_delay:]
x_out = aligned[:, ::cfg.decimation]
```

When `output_bits > 0`, divide by `2**output_binary_shift`, use `matlab_round`, and clip I/Q independently.

- [ ] **Step 5: Run DFE tests**

Run: `cd python_sim; python -m pytest tests/test_dfe.py -v`

Expected: all tests pass.

### Task 5: Single, EGC, and loaded-MVDR combining

**Files:**
- Create: `python_sim/antenna_diversity/diversity/__init__.py`
- Create: `python_sim/antenna_diversity/diversity/combiner.py`
- Create: `python_sim/tests/test_diversity.py`

**Interfaces:**
- Produces: `combine(x, desired_steering, mode, cfg) -> tuple[np.ndarray, dict[str, Any]]`

- [ ] **Step 1: Write target-normalization and rejection tests**

```python
def deterministic_array_case(cfg):
    a = steering_vector(
        cfg.array.positions_m,
        cfg.array.desired_angle_deg,
        cfg.signal.wavelength_m,
    )
    b = steering_vector(
        cfg.array.positions_m,
        cfg.jammer.angle_deg,
        cfg.signal.wavelength_m,
    )
    rng = np.random.Generator(np.random.MT19937(11))
    n_samples = 20_000
    desired = (
        rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
    ) / np.sqrt(2)
    jammer = (
        rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
    ) / np.sqrt(2)
    x = a[:, None] * desired[None, :] + 20 * b[:, None] * jammer[None, :]
    return a, b, x


@pytest.mark.parametrize("mode", ["egc", "mvdr"])
def test_combiner_has_unit_desired_response(mode):
    cfg = default_config()
    a, b, x = deterministic_array_case(cfg)
    _, info = combine(x, a, mode, cfg.diversity)
    assert info["desired_response"] == pytest.approx(1.0 + 0j, abs=1e-10)


def test_mvdr_reduces_jammer_direction_gain_by_tenfold():
    cfg = default_config()
    cfg.diversity.diagonal_loading_factor = 1e-4
    a, b, x = deterministic_array_case(cfg)
    _, single = combine(x, a, "single", cfg.diversity)
    _, mvdr = combine(x, a, "mvdr", cfg.diversity)
    assert abs(np.vdot(mvdr["weights"], b)) < 0.1 * abs(np.vdot(single["weights"], b))
```

The test helper uses `np.random.Generator(np.random.MT19937(11))`, 20,000 samples, and `X = a*s + 20*b*j`.

- [ ] **Step 2: Run diversity tests and observe missing implementation**

Run: `cd python_sim; python -m pytest tests/test_diversity.py -v`

Expected: missing `combine`.

- [ ] **Step 3: Implement covariance, loading, and weights**

```python
r = x.astype(np.complex128) @ x.astype(np.complex128).conj().T / x.shape[1]
loading = cfg.diagonal_loading_factor * np.trace(r).real / n_antennas
r_loaded = r + loading * np.eye(n_antennas)

if mode == "single":
    w = np.zeros(n_antennas, dtype=complex)
    w[0] = 1 / np.conj(a[0])
elif mode == "egc":
    w = a / np.vdot(a, a)
elif mode == "mvdr":
    z = np.linalg.solve(r_loaded, a)
    w = z / np.vdot(a, z)

y = w.conj() @ x
```

For the matrix output compute `y = w.conj() @ x`; `np.vdot` is reserved for vector inner products because it flattens matrix inputs.

- [ ] **Step 4: Run diversity tests**

Run: `cd python_sim; python -m pytest tests/test_diversity.py -v`

Expected: all tests pass.

### Task 6: FFT parallel-code-phase acquisition

**Files:**
- Create: `python_sim/antenna_diversity/receiver/__init__.py`
- Create: `python_sim/antenna_diversity/receiver/acquisition.py`
- Create: `python_sim/tests/test_acquisition.py`

**Interfaces:**
- Consumes: one-dimensional complex branch samples, sample rate, PRN, and `AcquisitionConfig`.
- Produces: `acquire(x, fs_hz, prn, cfg) -> dict[str, Any]`

- [ ] **Step 1: Write high-SNR acquisition test**

```python
def test_fft_acquisition_estimates_doppler_and_code_phase():
    cfg = default_config()
    fs_hz = cfg.dfe.fs_out_hz
    samples_per_ms = round(fs_hz * 1e-3)
    n = np.arange(cfg.acquisition.noncoherent_ms * samples_per_ms)
    true_delay_samples = 600
    true_delay_chips = true_delay_samples * cfg.signal.code_rate_hz / fs_hz
    code = ca_code(cfg.signal.prn)
    sampled = code[np.floor(np.mod(n * cfg.signal.code_rate_hz / fs_hz - true_delay_chips, 1023)).astype(int)]
    rng = np.random.Generator(np.random.MT19937(7))
    noise = 0.05 / np.sqrt(2) * (rng.standard_normal(n.size) + 1j * rng.standard_normal(n.size))
    x = sampled * np.exp(1j * 2 * np.pi * cfg.signal.doppler_hz * n / fs_hz) + noise
    acq = acquire(x, fs_hz, cfg.signal.prn, cfg.acquisition)
    assert acq["success"]
    assert abs(acq["doppler_hz"] - cfg.signal.doppler_hz) <= 250
    circular_error = min(abs(acq["code_phase_samples"] - 600), samples_per_ms - abs(acq["code_phase_samples"] - 600))
    assert circular_error <= 1
    assert acq["second_peak_distance_samples"] > acq["exclusion_samples"]
```

- [ ] **Step 2: Run acquisition test and verify missing implementation**

Run: `cd python_sim; python -m pytest tests/test_acquisition.py -v`

Expected: missing `acquire`.

- [ ] **Step 3: Implement noncoherent FFT search**

For each Doppler bin and noncoherent block:

```python
correlation = np.fft.ifft(
    np.fft.fft(block * carrier_wipeoff) * np.conj(np.fft.fft(local_code))
)
accumulated_power += np.abs(correlation) ** 2
```

Find the global peak, exclude a circular interval of one configured chip around it on the best Doppler row, find the second peak, and set `metric = peak / second_peak`. Return the full `search_power` for plotting.

- [ ] **Step 4: Run acquisition tests**

Run: `cd python_sim; python -m pytest tests/test_acquisition.py -v`

Expected: all tests pass.

### Task 7: DLL/FLL/PLL tracking

**Files:**
- Create: `python_sim/antenna_diversity/receiver/tracking.py`
- Create: `python_sim/tests/test_tracking.py`

**Interfaces:**
- Consumes: branch samples, acquisition result, tracking configuration, carrier frequency, and nominal code rate.
- Produces: `track(x, fs_hz, prn, acq, cfg, carrier_hz, nominal_code_rate_hz) -> dict[str, Any]`

- [ ] **Step 1: Write closed-loop tracking test**

```python
def make_tracking_test_signal(cfg, duration_ms, delay_samples, noise_sigma, seed):
    fs_hz = cfg.dfe.fs_out_hz
    samples_per_ms = round(fs_hz * 1e-3)
    n = np.arange(duration_ms * samples_per_ms)
    delay_chips = delay_samples * cfg.signal.code_rate_hz / fs_hz
    code = ca_code(cfg.signal.prn)
    code_phase = np.mod(
        n * cfg.signal.code_rate_hz / fs_hz - delay_chips,
        1023.0,
    )
    sampled_code = code[np.floor(code_phase).astype(int)]
    carrier = np.exp(
        1j * (2 * np.pi * cfg.signal.doppler_hz * n / fs_hz + 0.4)
    )
    rng = np.random.Generator(np.random.MT19937(seed))
    noise = noise_sigma / np.sqrt(2) * (
        rng.standard_normal(n.size) + 1j * rng.standard_normal(n.size)
    )
    return sampled_code * carrier + noise


def test_tracking_converges_after_successful_acquisition():
    cfg = default_config()
    x = make_tracking_test_signal(cfg, duration_ms=40, delay_samples=600, noise_sigma=0.15, seed=9)
    acq = acquire(x, cfg.dfe.fs_out_hz, cfg.signal.prn, cfg.acquisition)
    trk = track(
        x, cfg.dfe.fs_out_hz, cfg.signal.prn, acq, cfg.tracking,
        cfg.signal.carrier_hz, cfg.signal.code_rate_hz,
    )
    assert trk["success"]
    assert len(trk["prompt"]) >= 20
    assert np.all(np.isfinite(trk["carrier_frequency_hz"]))
    assert np.mean(np.abs(trk["prompt"][-10:])) > 100
    assert abs(trk["carrier_frequency_hz"][-1] - cfg.signal.doppler_hz) <= 250
    assert np.sqrt(np.mean(trk["dll_error"][-10:] ** 2)) < 0.5
    assert np.sqrt(np.mean(trk["pll_error_rad"][-10:] ** 2)) < 0.5
```

- [ ] **Step 2: Run tracking test and verify missing implementation**

Run: `cd python_sim; python -m pytest tests/test_tracking.py -v`

Expected: missing `track`.

- [ ] **Step 3: Translate epoch start and correlator generation carefully**

Keep acquisition and epoch positions zero-based:

```python
first_epoch = int(acq["code_phase_samples"])
while first_epoch < acquisition_samples:
    first_epoch += round(fs_hz * 1e-3)
```

Generate Early/Prompt/Late replicas by flooring modulo-1023 chip phases. Preserve the MATLAB signs: Early uses `prompt_phase + spacing/2`, Late uses `prompt_phase - spacing/2`.

- [ ] **Step 4: Implement discriminators and loop updates**

```python
dll_error = (abs(early) - abs(late)) / (abs(early) + abs(late) + eps)
pll_error = np.arctan2(prompt.imag, prompt.real)
fll_error = np.angle(prompt * np.conj(previous_prompt)) / (2 * np.pi * epoch_duration_s)
carrier_frequency_hz += fll_correction + cfg.pll_ki_hz * pll_error
carrier_phase_rad = wrap_phase(
    carrier_phase_rad + 2 * np.pi * carrier_frequency_hz * epoch_duration_s + cfg.pll_kp * pll_error
)
```

Return all histories required by the MATLAB result and metrics code.

- [ ] **Step 5: Run tracking tests**

Run: `cd python_sim; python -m pytest tests/test_tracking.py -v`

Expected: all tests pass.

### Task 8: End-to-end pipeline and metrics

**Files:**
- Create: `python_sim/antenna_diversity/analysis/__init__.py`
- Create: `python_sim/antenna_diversity/analysis/metrics.py`
- Create: `python_sim/antenna_diversity/pipeline.py`
- Create: `python_sim/tests/test_end_to_end.py`

**Interfaces:**
- Produces: `compute_metrics(results, truth) -> list[dict[str, Any]]`
- Produces: `format_metrics(metrics) -> str`
- Produces: `run_end_to_end(cfg: SimulationConfig | None = None) -> SimulationResults`

- [ ] **Step 1: Write end-to-end behavior test**

```python
def test_default_pipeline_reproduces_diversity_conclusion():
    cfg = default_config()
    cfg.plot.enable = False
    results = run_end_to_end(cfg)
    assert set(results.branches) == {"single", "egc", "mvdr"}
    assert not results.branches["single"].acquisition["success"]
    assert not results.branches["egc"].acquisition["success"]
    assert results.branches["mvdr"].acquisition["success"]
    assert results.branches["mvdr"].tracking["success"]
    assert abs(results.branches["mvdr"].acquisition["doppler_hz"] - 1500) <= 250
    assert abs(results.branches["mvdr"].tracking["carrier_frequency_hz"][-1] - 1500) <= 250
    b = results.truth["jammer_steering_nominal"]
    single_gain = abs(np.vdot(results.branches["single"].combining["weights"], b))
    mvdr_gain = abs(np.vdot(results.branches["mvdr"].combining["weights"], b))
    assert mvdr_gain < single_gain
    assert len(results.metrics) == 3
```

- [ ] **Step 2: Run end-to-end test and verify pipeline is absent**

Run: `cd python_sim; python -m pytest tests/test_end_to_end.py -v`

Expected: missing `pipeline.run_end_to_end`.

- [ ] **Step 3: Implement orchestration without duplicating algorithms**

```python
for mode in cfg.diversity.modes:
    combined, combining = combine(dfe_channels, truth["desired_steering_nominal"], mode, cfg.diversity)
    acquisition = acquire(combined, dfe_info["fs_out_hz"], cfg.signal.prn, cfg.acquisition)
    tracking = (
        track(combined, dfe_info["fs_out_hz"], cfg.signal.prn, acquisition,
              cfg.tracking, cfg.signal.carrier_hz, cfg.signal.code_rate_hz)
        if acquisition["success"]
        else {"success": False, "status": "acquisition_failed", "prompt": np.array([], dtype=complex)}
    )
    branches[mode] = BranchResult(combined, combining, acquisition, tracking)
```

- [ ] **Step 4: Implement metrics and compact text formatting**

Each record contains mode, acquired, acquisition metric, acquired Doppler, jammer direction gain, tracking state, final Doppler, PLL/DLL RMS, mean Prompt magnitude, and final C/N0. Use `None` for unavailable scalar metrics and format them as `nan` only at display time.

- [ ] **Step 5: Run the complete non-plot test set**

Run: `cd python_sim; python -m pytest tests/test_config_numeric.py tests/test_core.py tests/test_scenario.py tests/test_rf.py tests/test_dfe.py tests/test_diversity.py tests/test_acquisition.py tests/test_tracking.py tests/test_end_to_end.py -v`

Expected: all tests pass; default end-to-end result has single/EGC acquisition failure and MVDR tracking success.

### Task 9: Matplotlib visualization, runnable entry point, and README

**Files:**
- Create: `python_sim/antenna_diversity/analysis/plotting.py`
- Create: `python_sim/run_end_to_end.py`
- Create: `python_sim/README.md`
- Create: `python_sim/tests/test_plotting.py`
- Create directory at runtime: `python_sim/results/`

**Interfaces:**
- Produces: `plot_acquisition(results) -> matplotlib.figure.Figure`
- Produces: `plot_spectra(results) -> Figure`
- Produces: `plot_array_response(results) -> Figure`
- Produces: `plot_tracking(results) -> Figure`
- Produces: `plot_results(results) -> dict[str, Figure]`
- Produces: `save_figures(figures, output_dir) -> dict[str, pathlib.Path]`

- [ ] **Step 1: Write noninteractive plotting test**

```python
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


def test_acquisition_plot_contains_heatmap_slice_and_peak_markers(tmp_path):
    cfg = default_config()
    cfg.plot.enable = False
    results = run_end_to_end(cfg)
    figure = plot_acquisition(results)
    assert len(figure.axes) >= 6
    assert any(axis.images for axis in figure.axes)
    paths = save_figures({"acquisition_search": figure}, tmp_path)
    assert paths["acquisition_search"].is_file()
    assert paths["acquisition_search"].stat().st_size > 0
    plt.close(figure)
```

- [ ] **Step 2: Run plotting test and verify missing functions fail**

Run: `cd python_sim; python -m pytest tests/test_plotting.py -v`

Expected: missing plotting functions.

- [ ] **Step 3: Implement four focused plot functions**

Normalize acquisition power to its primary peak, clip the map and slice at `-40 dB`, mark the main and second peaks, and include PASS/FAIL, ratio, Doppler, and code phase in the title. Use explicit axis objects rather than global pyplot state inside plotting helpers.

- [ ] **Step 4: Implement the runnable script**

```python
import matplotlib.pyplot as plt

from antenna_diversity.analysis.metrics import format_metrics
from antenna_diversity.analysis.plotting import plot_results, save_figures
from antenna_diversity.pipeline import run_end_to_end


def main():
    results = run_end_to_end()
    print(format_metrics(results.metrics))
    if results.config.plot.enable:
        figures = plot_results(results)
        save_figures(figures, results.config.plot.output_dir)
        if results.config.plot.show:
            plt.show()
    return results
```

- [ ] **Step 5: Write README usage and model boundaries**

Document these commands exactly:

```powershell
cd F:\antenna_diversity\python_sim
python -m pip install -e ".[test]"
python -m pytest -v
python run_end_to_end.py
```

Explain the RF→DFE→diversity→acquisition→tracking flow, major config fields, array shapes, MATLAB/Python indexing/rounding/random differences, default result interpretation, and that Galileo E5 is outside this phase.

- [ ] **Step 6: Run plotting test and command-line smoke test**

Run: `cd python_sim; python -m pytest tests/test_plotting.py -v`

Run: `cd python_sim; $env:MPLBACKEND='Agg'; python run_end_to_end.py`

Expected: test passes; script prints three metric rows and writes four nonempty PNG files under `python_sim/results/` without opening a GUI window.

### Task 10: Full regression and MATLAB/Python behavioral comparison

**Files:**
- Modify only if verification exposes a defect: files under `python_sim/`
- Verify unchanged: all files under `matlab_sim/`

**Interfaces:**
- Consumes: the complete Python implementation.
- Produces: verified test and runtime evidence; no new public API.

- [ ] **Step 1: Run all Python tests in a clean process**

Run: `cd python_sim; $env:MPLBACKEND='Agg'; python -m pytest -v`

Expected: all tests pass with no unhandled warnings or skipped tests.

- [ ] **Step 2: Run the Python end-to-end script and record key values**

Run: `cd python_sim; $env:MPLBACKEND='Agg'; python run_end_to_end.py`

Expected:

```text
single: acquired false
egc: acquired false
mvdr: acquired true, tracking true
mvdr acquired Doppler: within 250 Hz of 1500 Hz
mvdr final Doppler: within 250 Hz of 1500 Hz
mvdr jammer-direction gain: lower than single
```

- [ ] **Step 3: Re-run the MATLAB baseline to prove the reference was preserved**

Run:

```powershell
& 'D:\MATLAB\R2023a\bin\matlab.exe' -batch "cd('F:/antenna_diversity/matlab_sim'); addpath(genpath(pwd)); run('tests/run_all_tests.m');"
```

Expected: `PASS all MATLAB simulation tests`.

- [ ] **Step 4: Inspect generated images**

Open or render each PNG and verify readable titles, axes, color scale, markers, legends, and no clipping:

```text
python_sim/results/rf_dfe_spectra.png
python_sim/results/array_response.png
python_sim/results/acquisition_search.png
python_sim/results/tracking_comparison.png
```

- [ ] **Step 5: Confirm scope and workspace integrity**

Run: `Get-ChildItem matlab_sim -Recurse -File | Measure-Object`

Compare against the pre-implementation MATLAB file inventory and ensure all implementation changes are confined to `python_sim` plus this plan/spec documentation.

- [ ] **Step 6: Deliver the verified entry points**

Report the full Python test result, MATLAB regression result, default branch outcomes, key MVDR estimates, generated figure paths, and links to `python_sim/README.md` and `python_sim/run_end_to_end.py`.
