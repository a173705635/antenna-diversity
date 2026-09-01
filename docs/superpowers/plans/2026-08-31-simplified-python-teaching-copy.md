# Simplified Python Teaching Copy Implementation Plan

> Current status (2026-09-01): this plan records the original copy creation. The directory is now maintained as the personal simplified research simulation described by `2026-09-01-simplified-research-refactor.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an independent, readable `python_sim_simplified/` copy of the RF → DFE → antenna combining → acquisition → tracking simulation while leaving `python_sim/` unchanged.

**Architecture:** The teaching copy is a flat set of focused Python modules that use sibling imports and exchange NumPy arrays plus ordinary dictionaries. The implementation preserves the physical formulas and algorithm decisions of the research package, but omits validation layers, exception normalization, recovery, study orchestration, fixed carrier-NCO comparison, and degenerate-input fallbacks.

**Tech Stack:** Python 3.11+, NumPy 2.0+, SciPy 1.14+, Matplotlib 3.9+, pytest 9.0+

**Spec:** `docs/superpowers/specs/2026-08-31-simplified-python-teaching-copy-design.md`

## Global Constraints

- Do not modify, import, or copy at runtime from any file below `python_sim/`.
- Create the teaching implementation only below `python_sim_simplified/`, apart from this plan and final verification records.
- Preserve GPS L1 C/A, complex equivalent IF, carrier and code Doppler, code delay, 2–4 Mate 70 corner antennas, one target, and 0–3 gated CW interferers.
- Preserve ideal, parametric, and measured polarized antenna responses, true/assumed states, orientation, blockage, mutual coupling, and per-channel amplitude/phase response.
- Preserve RF gain/noise/limiting/ADC, common-NCO DDC/FIR/decimation/DFE quantization, SINGLE/EGC/MVDR/Blind Null, FFT acquisition, float-NCO DLL/FLL/PLL tracking, C/N0, and plots.
- Delete explicit bad-input protection branches, exception wrappers, hashes, schema audits, zero-denominator fallbacks, unsafe-projection bypasses, checkpoints, studies, Monte Carlo, reports, and fixed carrier-NCO comparisons.
- Keep the Blind Null detection decision and the acquisition-success gate because they are algorithm decisions rather than engineering protection.
- The default valid scenario uses the same principal physical values and random seed as `python_sim/antenna_diversity/config.py`.
- Every module uses direct sibling imports so `python python_sim_simplified/run_end_to_end.py` works without package installation.

## File and Interface Map

| File | Responsibility | Public interface |
|---|---|---|
| `config.py` | Editable dataclasses and default/parametric/measured presets | `default_config()`, `parametric_config()`, `measured_config()` |
| `antenna.py` | Geometry, Jones polarization, pattern interpolation, orientation, blockage, coupling, true/assumed response | `build_antenna_model(cfg)`, `antenna_response(model, direction, polarization, estimated=False)` |
| `gnss.py` | C/A code and antenna-port signal scene | `ca_code(prn)`, `generate_scenario(cfg)` |
| `rf.py` | Friis cascade, thermal noise, limiting, complex ADC | `process_rf(x_antenna_v, cfg, seed)` |
| `dfe.py` | Common DDC, windowed-sinc FIR, alignment, decimation, output quantization | `design_decimator(fs_in_hz, decimation, cutoff_hz, num_taps)`, `process_dfe(adc_codes, cfg)` |
| `diversity.py` | SINGLE, EGC, loaded MVDR, sample-only Blind Null | `combine_known(channels, desired, mode, diagonal_loading_factor)`, `combine_blind_null(channels, cfg)` |
| `acquisition.py` | Doppler/code-phase FFT search | `acquire(x, fs_hz, prn, cfg)` |
| `tracking.py` | Float carrier/code NCO, EPL correlators, DLL/FLL/PLL, C/N0 | `estimate_cn0_db_hz(prompt_window, coherent_interval_s)`, `track(x, fs_hz, prn, acq, cfg, carrier_hz, nominal_code_rate_hz)` |
| `metrics.py` | Compact branch comparison | `compute_metrics(results)`, `format_metrics(records)` |
| `plotting.py` | Seven independent diagnostic figures | `plot_results(results)` |
| `pipeline.py` | End-to-end orchestration and branch result dictionaries | `run_end_to_end(cfg=None)` |
| `run_end_to_end.py` | Human-editable command entry point | `main()` |

The stable data contracts are:

```python
# Multi-antenna sample matrices are always channels x samples.
scene_input_v: np.ndarray       # complex128, shape (M, Nrf)
adc_codes: np.ndarray           # complex128 integer-valued I/Q, shape (M, Nrf)
dfe_channels: np.ndarray        # complex128 integer-valued I/Q, shape (M, Ndfe)
combined: np.ndarray            # complex128, shape (Ndfe,)

# run_end_to_end() returns this ordinary dictionary.
results = {
    "config": cfg,
    "truth": truth,
    "scene_input_v": scene_input_v,
    "adc_codes": adc_codes,
    "rf_info": rf_info,
    "dfe_channels": dfe_channels,
    "dfe_info": dfe_info,
    "branches": {
        mode: {
            "combined": combined,
            "combining": combining_info,
            "acquisition": acquisition,
            "tracking": tracking,
        }
    },
    "metrics": records,
}
```

---

### Task 1: Editable Configuration and GPS C/A Primitive

**Files:**
- Create: `python_sim_simplified/config.py`
- Create: `python_sim_simplified/gnss.py`
- Create: `python_sim_simplified/tests/conftest.py`
- Create: `python_sim_simplified/tests/test_config_gnss.py`

**Interfaces:**
- Consumes: No earlier teaching-copy modules.
- Produces: all configuration record classes, `default_config() -> SimulationConfig`, `parametric_config(active_antennas) -> SimulationConfig`, `measured_config(csv_path, active_antennas) -> SimulationConfig`, and `ca_code(prn) -> np.ndarray`.

- [ ] **Step 1: Create test import setup and failing configuration/C/A tests**

```python
# python_sim_simplified/tests/conftest.py
from pathlib import Path
import sys

SIMPLIFIED_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIMPLIFIED_ROOT))

# python_sim_simplified/tests/test_config_gnss.py
import numpy as np

from config import default_config
from gnss import ca_code


def test_default_config_uses_mate70_four_corners_and_original_signal_values():
    cfg = default_config()
    assert cfg.random_seed == 42
    assert cfg.signal.carrier_hz == 1_575.42e6
    assert cfg.signal.fs_rf_hz == 16.368e6
    assert cfg.signal.if_hz == 4.092e6
    assert cfg.phone.active_antennas == ("TL", "TR", "BL", "BR")
    assert cfg.phone.phase_centers_m["TL"] == (-0.0759 / 2, 0.1609 / 2, 0.0)
    assert len(cfg.interferers) == 1


def test_ca_code_is_repeatable_bipolar_and_has_1023_chips():
    first = ca_code(1)
    second = ca_code(1)
    assert first.shape == (1023,)
    assert set(np.unique(first)) == {-1.0, 1.0}
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[:10], [-1, -1, 1, 1, -1, 1, 1, 1, 1, 1])
```

- [ ] **Step 2: Run the tests and verify the modules are absent**

Run: `python -m pytest python_sim_simplified/tests/test_config_gnss.py -v`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'config'`.

- [ ] **Step 3: Implement the compact configuration records and presets**

Create dataclasses without `validate()` methods. Use these exact names so later tasks share one vocabulary:

```python
@dataclass
class Direction:
    azimuth_deg: float
    elevation_deg: float

@dataclass
class WavePolarization:
    kind: str = "rhcp"
    tilt_deg: float = 0.0
    az_real: float = 1.0
    az_imag: float = 0.0
    el_real: float = 0.0
    el_imag: float = 0.0

@dataclass
class ElementOrientation:
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0

@dataclass
class ParametricPattern:
    peak_gain_db: float = 0.0
    cosine_exponent: float = 1.0
    back_lobe_gain_db: float = -20.0
    nominal_polarization: WavePolarization = field(default_factory=WavePolarization)
    cross_polar_discrimination_db: float = 30.0
    cross_phase_deg: float = 0.0

@dataclass
class AttenuationRegion:
    elements: tuple[str, ...]
    center_azimuth_deg: float
    center_elevation_deg: float
    radius_deg: float
    attenuation_db: float
    transition_width_deg: float = 0.0

@dataclass
class CouplingPath:
    source: str
    receiver: str
    magnitude_db: float
    phase_deg: float

@dataclass
class AntennaState:
    pattern_kind: str = "ideal"
    parametric_patterns: dict[str, ParametricPattern] = field(default_factory=dict)
    measured_pattern_csv: Path | None = None
    orientations: dict[str, ElementOrientation] = field(default_factory=dict)
    attenuation_regions: tuple[AttenuationRegion, ...] = ()
    coupling_paths: tuple[CouplingPath, ...] = ()

@dataclass
class AntennaModelConfig:
    model_label: str
    true_state: AntennaState
    assumed_state: AntennaState
```

Also define `SignalConfig`, `PhoneConfig`, `CWInterferer`, `RFStage`, `RFConfig`, `DFEConfig`, `BlindNullConfig`, `DiversityConfig`, `AcquisitionConfig`, `TrackingConfig`, `PlotConfig`, and `SimulationConfig`. Copy the numeric defaults from the approved spec/source: 100 ms, PRN 1, -128.5 dBm target, 1500 Hz Doppler, 137.25 chips delay, 16.368 MHz RF rate, 4.092 MHz IF, four Mate 70 corners, -90 dBm/250 kHz interferer at `(120°, 10°)`, 10-bit ADC, decimation 4, 8-bit DFE output, and all four combining modes.

Implement the channel response directly:

```python
def channel_response(phone: PhoneConfig, estimated: bool = False) -> np.ndarray:
    amplitude = phone.estimated_channel_amplitude_error_db if estimated else phone.channel_amplitude_error_db
    phase = phone.estimated_channel_phase_error_deg if estimated else phone.channel_phase_error_deg
    names = phone.active_antennas
    amplitude_db = np.array([amplitude[name] for name in names])
    phase_deg = np.array([phase[name] for name in names])
    return 10.0 ** (amplitude_db / 20.0) * np.exp(1j * np.deg2rad(phase_deg))
```

`default_config()` creates ideal true/assumed states. `parametric_config()` replaces them with the four synthetic per-corner pattern/orientation records and coupling/blockage values already used by `python_sim/antenna_diversity/config.py`. `measured_config()` replaces both pattern kinds and CSV paths with the supplied `Path` while leaving true and assumed geometry independently editable.

- [ ] **Step 4: Implement GPS L1 C/A generation**

```python
G2_TAPS = np.array([
    [2, 6], [3, 7], [4, 8], [5, 9], [1, 9], [2, 10], [1, 8], [2, 9],
    [3, 10], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10],
    [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9], [1, 3], [4, 6],
    [5, 7], [6, 8], [7, 9], [8, 10], [1, 6], [2, 7], [3, 8], [4, 9],
])

def ca_code(prn: int) -> np.ndarray:
    tap1, tap2 = G2_TAPS[prn - 1] - 1
    g1 = np.ones(10, dtype=np.uint8)
    g2 = np.ones(10, dtype=np.uint8)
    code = np.empty(1023)
    for chip in range(1023):
        code[chip] = 1.0 - 2.0 * float(g1[9] ^ g2[tap1] ^ g2[tap2])
        g1_feedback = g1[2] ^ g1[9]
        g2_feedback = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        g1[1:], g2[1:] = g1[:-1], g2[:-1]
        g1[0], g2[0] = g1_feedback, g2_feedback
    return code
```

- [ ] **Step 5: Run the focused tests**

Run: `python -m pytest python_sim_simplified/tests/test_config_gnss.py -v`

Expected: 2 passed.

- [ ] **Step 6: Commit the configuration and code primitive**

```bash
git add python_sim_simplified/config.py python_sim_simplified/gnss.py python_sim_simplified/tests/conftest.py python_sim_simplified/tests/test_config_gnss.py
git commit -m "feat: add simplified simulation configuration"
```

---

### Task 2: Full Phone Antenna Model

**Files:**
- Create: `python_sim_simplified/antenna.py`
- Create: `python_sim_simplified/tests/test_antenna.py`

**Interfaces:**
- Consumes: `SimulationConfig`, `Direction`, `WavePolarization`, `AntennaState`, and the antenna effect records from `config.py`.
- Produces: `AntennaModel`, `load_measured_pattern(path, frequency_hz) -> dict`, `interpolate_measured(grid, azimuth_deg, elevation_deg) -> np.ndarray`, `build_antenna_model(cfg) -> AntennaModel`, and `antenna_response(model, direction, polarization, estimated=False) -> tuple[np.ndarray, dict]`.

- [ ] **Step 1: Write failing tests for composition, fixed phase, and measured interpolation**

```python
import csv
import numpy as np

from antenna import (
    antenna_response,
    build_antenna_model,
    interpolate_measured,
    load_measured_pattern,
)
from config import Direction, WavePolarization, default_config, measured_config


def test_effective_response_composes_open_circuit_coupling_and_channel():
    cfg = default_config()
    cfg.phone.channel_phase_error_deg["TR"] = 90.0
    model = build_antenna_model(cfg)
    response, stages = antenna_response(
        model, cfg.signal.true_direction, cfg.signal.true_polarization
    )
    np.testing.assert_allclose(
        response,
        stages["channel_response"]
        * (stages["coupling_matrix"] @ stages["open_circuit_response"]),
    )
    assert np.angle(stages["channel_response"][1], deg=True) == 90.0


def test_measured_csv_interpolates_complex_jones_components(tmp_path):
    path = tmp_path / "pattern.csv"
    columns = ["element_name", "frequency_hz", "azimuth_deg", "elevation_deg",
               "h_az_real", "h_az_imag", "h_el_real", "h_el_imag"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for name in ("TL", "TR", "BL", "BR"):
            for elevation in (0.0, 90.0):
                for azimuth in (0.0, 180.0):
                    writer.writerow(dict(element_name=name, frequency_hz=1_575.42e6,
                        azimuth_deg=azimuth, elevation_deg=elevation,
                        h_az_real=1.0 + azimuth / 180.0, h_az_imag=elevation / 90.0,
                        h_el_real=0.25, h_el_imag=-0.5))
    cfg = measured_config(path)
    model = build_antenna_model(cfg)
    grids = load_measured_pattern(path, cfg.signal.carrier_hz)
    height = interpolate_measured(grids["TL"], 90.0, 45.0)
    np.testing.assert_allclose(height, [1.5 + 0.5j, 0.25 - 0.5j])
    response, stages = antenna_response(
        model, Direction(90.0, 45.0), WavePolarization("rhcp")
    )
    assert response.shape == (4,)
    assert np.all(np.isfinite(response))
    assert stages["pattern_polarization_voltage"].shape == (4,)
```

- [ ] **Step 2: Run the tests and verify `antenna.py` is absent**

Run: `python -m pytest python_sim_simplified/tests/test_antenna.py -v`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'antenna'`.

- [ ] **Step 3: Implement coordinate and polarization helpers without validation wrappers**

```python
def direction_vector(direction):
    az, el = np.deg2rad([direction.azimuth_deg, direction.elevation_deg])
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])

def jones_vector(polarization):
    if polarization.kind == "rhcp":
        vector = np.array([1.0, -1.0j])
    elif polarization.kind == "lhcp":
        vector = np.array([1.0, 1.0j])
    elif polarization.kind == "linear":
        tilt = np.deg2rad(polarization.tilt_deg)
        vector = np.array([np.cos(tilt), np.sin(tilt)], dtype=complex)
    else:
        vector = np.array([
            complex(polarization.az_real, polarization.az_imag),
            complex(polarization.el_real, polarization.el_imag),
        ])
    return vector / np.linalg.norm(vector)

def rotation_matrix_ypr(orientation):
    yaw, pitch, roll = np.deg2rad(
        [orientation.yaw_deg, orientation.pitch_deg, orientation.roll_deg]
    )
    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cx, sx = np.cos(roll), np.sin(roll)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return rz @ ry @ rx
```

Complete the last function with the same explicit `Rz @ Ry @ Rx` matrices as the research implementation. Transform both the direction vector and electric-field vector into each element frame, then project onto local azimuth/elevation basis vectors.

- [ ] **Step 4: Implement ideal, parametric, and measured pattern evaluation**

```python
def parametric_height(parameters, local_elevation_deg):
    front_power = max(np.sin(np.deg2rad(local_elevation_deg)), 0.0) ** parameters.cosine_exponent
    relative_db = -np.inf if front_power == 0.0 else 10.0 * np.log10(front_power)
    gain_db = parameters.peak_gain_db + max(
        relative_db, parameters.back_lobe_gain_db - parameters.peak_gain_db
    )
    nominal = jones_vector(parameters.nominal_polarization)
    orthogonal = np.array([-np.conj(nominal[1]), np.conj(nominal[0])])
    cross = 10.0 ** (-parameters.cross_polar_discrimination_db / 20.0)
    mixture = nominal + cross * np.exp(1j * np.deg2rad(parameters.cross_phase_deg)) * orthogonal
    return 10.0 ** (gain_db / 20.0) * mixture / np.linalg.norm(mixture)
```

Read measured rows with `csv.DictReader`, group by element, sort unique azimuth/elevation axes, build a complex `heights[elevation, azimuth, 2]` array, and interpolate real/imaginary components together. Azimuth interpolation wraps through 360°, while elevation uses bounded neighbors. Do not calculate a digest, check file existence first, or wrap CSV/NumPy exceptions.

- [ ] **Step 5: Implement the ordered antenna composition**

```python
@dataclass
class AntennaModel:
    cfg: object
    true_measured: dict | None
    assumed_measured: dict | None

def build_antenna_model(cfg):
    true = load_measured_pattern(cfg.phone.antenna_model.true_state.measured_pattern_csv,
                                 cfg.signal.carrier_hz) \
        if cfg.phone.antenna_model.true_state.pattern_kind == "measured" else None
    assumed = load_measured_pattern(cfg.phone.antenna_model.assumed_state.measured_pattern_csv,
                                    cfg.signal.carrier_hz) \
        if cfg.phone.antenna_model.assumed_state.pattern_kind == "measured" else None
    return AntennaModel(cfg, true, assumed)

def antenna_response(model, direction, polarization, estimated=False):
    cfg = model.cfg
    state = cfg.phone.antenna_model.assumed_state if estimated else cfg.phone.antenna_model.true_state
    positions_map = cfg.phone.assumed_phase_centers_m if estimated else cfg.phone.phase_centers_m
    positions = np.array([positions_map[name] for name in cfg.phone.active_antennas])
    spatial = np.exp(-1j * 2.0 * np.pi * cfg.signal.carrier_hz / 299_792_458.0
                     * (positions @ direction_vector(direction)))
    measured = model.assumed_measured if estimated else model.true_measured
    pattern_voltage = []
    for name in cfg.phone.active_antennas:
        if state.pattern_kind == "ideal":
            voltage = 1.0 + 0.0j
        else:
            local_direction, local_jones = wave_in_element_frame(
                direction, polarization, rotation_matrix_ypr(state.orientations[name])
            )
            if state.pattern_kind == "parametric":
                height = parametric_height(
                    state.parametric_patterns[name], local_direction.elevation_deg
                )
            else:
                height = interpolate_measured(
                    measured[name], local_direction.azimuth_deg,
                    local_direction.elevation_deg,
                )
            voltage = np.vdot(height, local_jones)
        pattern_voltage.append(voltage)
    pattern_voltage = np.asarray(pattern_voltage)
    blockage = attenuation_voltage(state.attenuation_regions,
                                    cfg.phone.active_antennas, direction)
    open_circuit = spatial * pattern_voltage * blockage
    coupling = coupling_matrix(state.coupling_paths, cfg.phone.active_antennas)
    channel = channel_response(cfg.phone, estimated)
    effective = channel * (coupling @ open_circuit)
    return effective, {
        "positions_m": positions,
        "spatial_phase": spatial,
        "pattern_polarization_voltage": pattern_voltage,
        "blockage_voltage": blockage,
        "open_circuit_response": open_circuit,
        "coupling_matrix": coupling,
        "coupled_response": coupling @ open_circuit,
        "channel_response": channel,
        "effective_manifold": effective,
    }
```

Define a small `LocalDirection` dataclass and have `wave_in_element_frame()` return `(LocalDirection, normalized_local_jones)`. `interpolate_measured()` returns the two-component complex effective height for one element grid.

- [ ] **Step 6: Run antenna tests**

Run: `python -m pytest python_sim_simplified/tests/test_antenna.py -v`

Expected: 2 passed.

- [ ] **Step 7: Commit the antenna model**

```bash
git add python_sim_simplified/antenna.py python_sim_simplified/tests/test_antenna.py
git commit -m "feat: add simplified full antenna model"
```

---

### Task 3: GNSS Scene Generation

**Files:**
- Modify: `python_sim_simplified/gnss.py`
- Create: `python_sim_simplified/tests/test_scenario.py`

**Interfaces:**
- Consumes: `ca_code()`, `build_antenna_model()`, `antenna_response()`, and `SimulationConfig`.
- Produces: `generate_scenario(cfg) -> tuple[scene_input_v, truth]` where truth contains true/assumed target responses and every effective interferer response.

- [ ] **Step 1: Write failing scenario tests**

```python
import numpy as np

from config import default_config
from gnss import generate_scenario


def test_scenario_has_expected_shape_code_doppler_and_interferer_gate():
    cfg = default_config()
    cfg.signal.duration_ms = 8
    cfg.phone.active_antennas = ("TL", "BR")
    cfg.interferers[0].start_ms = 2.0
    cfg.interferers[0].stop_ms = 5.0
    samples, truth = generate_scenario(cfg)
    assert samples.shape == (2, round(cfg.signal.fs_rf_hz * 0.008))
    expected_code_rate = cfg.signal.code_rate_hz * (
        1.0 + cfg.signal.doppler_hz / cfg.signal.carrier_hz
    )
    assert truth["code_rate_hz"] == expected_code_rate
    jammer = truth["interferer_components"][0]
    assert np.all(jammer[:, :round(0.002 * cfg.signal.fs_rf_hz)] == 0.0)
    assert np.any(jammer[:, round(0.002 * cfg.signal.fs_rf_hz):round(0.005 * cfg.signal.fs_rf_hz)] != 0.0)


def test_relative_channel_phase_is_present_before_rf():
    cfg = default_config()
    cfg.signal.duration_ms = 2
    cfg.interferers = []
    cfg.phone.active_antennas = ("TL", "TR")
    cfg.phone.channel_phase_error_deg["TR"] = 60.0
    _, truth = generate_scenario(cfg)
    phase = np.angle(truth["desired_steering_true"][1] /
                     truth["desired_open_circuit_response_true"][1], deg=True)
    assert np.isclose(phase, 60.0)
```

- [ ] **Step 2: Run the tests and verify `generate_scenario` is absent**

Run: `python -m pytest python_sim_simplified/tests/test_scenario.py -v`

Expected: FAIL importing `generate_scenario`.

- [ ] **Step 3: Implement target signal, code Doppler, and static CW interferers**

```python
def generate_scenario(cfg):
    n_samples = round(cfg.signal.fs_rf_hz * cfg.signal.duration_ms * 1e-3)
    n = np.arange(n_samples)
    time_s = n / cfg.signal.fs_rf_hz
    received_code_rate = cfg.signal.code_rate_hz * (
        1.0 + cfg.signal.doppler_hz / cfg.signal.carrier_hz
    )
    code_phase = np.mod(
        n * received_code_rate / cfg.signal.fs_rf_hz - cfg.signal.code_delay_chips,
        1023.0,
    )
    sampled_code = ca_code(cfg.signal.prn)[np.floor(code_phase).astype(int)]
    signal_amplitude_v = np.sqrt(
        10.0 ** ((cfg.signal.power_dbm - 30.0) / 10.0) * cfg.signal.resistance_ohm
    )
    carrier = np.exp(1j * (2 * np.pi * (cfg.signal.if_hz + cfg.signal.doppler_hz)
                           * time_s + cfg.signal.initial_phase_rad))
    desired_waveform = signal_amplitude_v * sampled_code * carrier
```

Build the antenna model once, evaluate the target in true and assumed states, and evaluate each enabled interferer in the true state. Each interferer waveform uses `if_hz + offset_hz`, its own phase and power, and a sample-index time gate. Return `desired_true[:, None] * desired_waveform[None, :] + sum(interferer_components)` plus a truth dictionary containing directions, effective steering vectors, component arrays, timing, rates, and antenna-stage diagnostics.

- [ ] **Step 4: Run scenario tests**

Run: `python -m pytest python_sim_simplified/tests/test_scenario.py -v`

Expected: 2 passed.

- [ ] **Step 5: Commit scene generation**

```bash
git add python_sim_simplified/gnss.py python_sim_simplified/tests/test_scenario.py
git commit -m "feat: add simplified GNSS signal scene"
```

---

### Task 4: RF and DFE Chain

**Files:**
- Create: `python_sim_simplified/rf.py`
- Create: `python_sim_simplified/dfe.py`
- Create: `python_sim_simplified/tests/test_rf_dfe.py`

**Interfaces:**
- Consumes: `(M, Nrf)` antenna voltage and RF/DFE configs.
- Produces: `process_rf(x_antenna_v, cfg, seed) -> (adc_codes, rf_info)`, `design_decimator(fs_in_hz, decimation, cutoff_hz, num_taps) -> (h, filter_info)`, and `process_dfe(adc_codes, cfg) -> (dfe_channels, dfe_info)`.

- [ ] **Step 1: Write failing deterministic RF/DFE tests**

```python
import numpy as np

from config import DFEConfig, RFConfig
from dfe import design_decimator, process_dfe
from rf import process_rf


def test_rf_is_repeatable_and_adc_codes_are_signed_integers():
    x = np.zeros((2, 4096), dtype=complex)
    first, info = process_rf(x, RFConfig(), 42)
    second, _ = process_rf(x, RFConfig(), 42)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first.real, np.round(first.real))
    np.testing.assert_array_equal(first.imag, np.round(first.imag))
    assert info["noise_figure_db"] > 0.0


def test_dfe_moves_if_to_dc_and_preserves_interchannel_phase():
    cfg = DFEConfig(output_bits=0)
    n = np.arange(round(cfg.fs_in_hz * 0.004))
    tone = np.exp(1j * 2 * np.pi * cfg.if_hz * n / cfg.fs_in_hz)
    channels = np.vstack([tone, tone * np.exp(1j * np.deg2rad(70.0))])
    output, info = process_dfe(channels, cfg)
    phase = np.angle(np.vdot(output[0], output[1]), deg=True)
    assert np.isclose(phase, 70.0, atol=0.1)
    assert info["fs_out_hz"] == cfg.fs_in_hz / cfg.decimation
    h, filter_info = design_decimator(cfg.fs_in_hz, cfg.decimation,
                                      cfg.cutoff_hz, cfg.num_taps)
    assert np.isclose(h.sum(), 1.0)
    assert filter_info["group_delay_input_samples"] == (cfg.num_taps - 1) // 2
```

- [ ] **Step 2: Run the tests and verify both modules are absent**

Run: `python -m pytest python_sim_simplified/tests/test_rf_dfe.py -v`

Expected: FAIL during import of `dfe` or `rf`.

- [ ] **Step 3: Implement Friis cascade, thermal noise, limiting, and ADC**

```python
def process_rf(x_antenna_v, cfg, seed):
    gain = 10.0 ** (np.array([s.gain_db for s in cfg.stages]) / 10.0)
    noise_factor = 10.0 ** (np.array([s.noise_figure_db for s in cfg.stages]) / 10.0)
    cascade_nf = noise_factor[0]
    preceding_gain = gain[0]
    for index in range(1, len(gain)):
        cascade_nf += (noise_factor[index] - 1.0) / preceding_gain
        preceding_gain *= gain[index]
    input_noise_w = 1.380649e-23 * cfg.temperature_k * cfg.noise_bandwidth_hz * cascade_nf
    noise_rms_v = np.sqrt(input_noise_w * cfg.resistance_ohm)
    rng = np.random.Generator(np.random.MT19937(seed))
    noise = noise_rms_v / np.sqrt(2.0) * (
        rng.standard_normal(x_antenna_v.shape) + 1j * rng.standard_normal(x_antenna_v.shape)
    )
    before_limit = (x_antenna_v + noise) * np.sqrt(np.prod(gain))
```

Apply component-wise `tanh` for soft limiting or component-wise `np.clip` for hard limiting. Quantize I and Q with MATLAB-compatible half-away-from-zero rounding, `lsb_v = adc_vpp / 2**adc_bits`, and signed limits. Return gain, noise, saturation, RMS, ADC step, and bit depth in `rf_info`.

- [ ] **Step 4: Implement windowed-sinc DDC and decimation**

```python
def design_decimator(fs_in_hz, decimation, cutoff_hz, num_taps):
    delay = (num_taps - 1) // 2
    m = np.arange(-delay, delay + 1)
    h = (2.0 * cutoff_hz / fs_in_hz) * np.sinc(2.0 * cutoff_hz / fs_in_hz * m)
    h *= np.hamming(num_taps)
    h /= h.sum()
    return h, {"fs_in_hz": fs_in_hz, "fs_out_hz": fs_in_hz / decimation,
               "decimation": decimation, "cutoff_hz": cutoff_hz,
               "num_taps": num_taps, "group_delay_input_samples": delay}

def process_dfe(adc_codes, cfg):
    h, info = design_decimator(cfg.fs_in_hz, cfg.decimation, cfg.cutoff_hz, cfg.num_taps)
    n = np.arange(adc_codes.shape[1])
    nco = np.exp(-1j * 2.0 * np.pi * cfg.if_hz * n / cfg.fs_in_hz)
    mixed = adc_codes * nco[None, :]
    filtered = scipy.signal.lfilter(h, [1.0], mixed, axis=1)
    output = filtered[:, info["group_delay_input_samples"]::cfg.decimation]
```

When `output_bits > 0`, divide I/Q by `2**output_binary_shift`, apply half-away-from-zero rounding, and clip to signed output limits. Do not add input-length, output-scale, or nonfinite fallbacks.

- [ ] **Step 5: Run RF/DFE tests**

Run: `python -m pytest python_sim_simplified/tests/test_rf_dfe.py -v`

Expected: 2 passed.

- [ ] **Step 6: Commit the front-end chain**

```bash
git add python_sim_simplified/rf.py python_sim_simplified/dfe.py python_sim_simplified/tests/test_rf_dfe.py
git commit -m "feat: add simplified RF and DFE chain"
```

---

### Task 5: Known-Steering and Blind Spatial Combining

**Files:**
- Create: `python_sim_simplified/diversity.py`
- Create: `python_sim_simplified/tests/test_diversity.py`

**Interfaces:**
- Consumes: DFE channels, assumed target response for known combiners, and diversity configs.
- Produces: `combine_known(channels, desired, mode, diagonal_loading_factor)` and `combine_blind_null(channels, cfg)`, both returning `(combined, info)`.

- [ ] **Step 1: Write failing tests for unit target response, nulling, and phase covariance**

```python
import numpy as np

from config import BlindNullConfig
from diversity import combine_blind_null, combine_known


def test_known_combiners_have_unit_target_response():
    covariance_source = np.array([[1, 2, 3, 4], [1j, 2j, 1j, 3j]], dtype=complex)
    desired = np.array([1.0, np.exp(1j * 0.4)])
    for mode in ("single", "egc", "mvdr"):
        _, info = combine_known(covariance_source, desired, mode, 1e-2)
        assert np.isclose(np.vdot(info["weights"], desired), 1.0)


def test_blind_null_observes_fixed_interchannel_phase_and_suppresses_mode():
    rng = np.random.default_rng(7)
    tone = np.exp(1j * 2 * np.pi * 0.03 * np.arange(4096))
    phase_rad = np.deg2rad(55.0)
    channels = np.vstack([tone, 0.8 * tone * np.exp(1j * phase_rad)])
    channels += 0.02 * (rng.standard_normal(channels.shape) + 1j * rng.standard_normal(channels.shape))
    combined, info = combine_blind_null(channels, BlindNullConfig(detection_threshold_db=3.0))
    assert np.isclose(np.angle(info["sample_covariance"][0, 1]), -phase_rad, atol=0.03)
    assert info["interference_detected"]
    assert info["nulling_enabled"]
    interference_steering = np.array([1.0, 0.8 * np.exp(1j * phase_rad)])
    assert abs(np.vdot(info["weights"], interference_steering)) < 0.05
    assert np.mean(np.abs(combined) ** 2) < 0.1 * np.mean(np.abs(channels[0]) ** 2)
```

- [ ] **Step 2: Run the tests and verify `diversity.py` is absent**

Run: `python -m pytest python_sim_simplified/tests/test_diversity.py -v`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'diversity'`.

- [ ] **Step 3: Implement SINGLE, EGC, and loaded MVDR**

```python
def combine_known(channels, desired, mode, diagonal_loading_factor):
    covariance = channels @ channels.conj().T / channels.shape[1]
    loading = diagonal_loading_factor * np.trace(covariance).real / channels.shape[0]
    loaded = covariance + loading * np.eye(channels.shape[0])
    if mode == "single":
        weights = np.zeros(channels.shape[0], dtype=complex)
        weights[0] = 1.0 / np.conj(desired[0])
    elif mode == "egc":
        weights = desired / np.vdot(desired, desired)
    elif mode == "mvdr":
        solved = np.linalg.solve(loaded, desired)
        weights = solved / np.vdot(desired, solved)
    combined = weights.conj() @ channels
    return combined, {"mode": mode, "weights": weights,
                      "sample_covariance": covariance,
                      "loaded_covariance": loaded,
                      "diagonal_loading": loading}
```

Do not add unknown-mode handling because the editable config supplies the supported names.

- [ ] **Step 4: Implement the approved Blind Null main line**

```python
def combine_blind_null(channels, cfg):
    covariance = channels @ channels.conj().T / channels.shape[1]
    covariance = (covariance + covariance.conj().T) / 2.0
    powers = np.real(np.diag(covariance))
    scales = np.sqrt(powers)
    correlation = covariance / np.outer(scales, scales)
    correlation = (correlation + correlation.conj().T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    score_db = 10.0 * np.log10(eigenvalues[-1] / np.mean(eigenvalues[:-1]))
    interference_vector = scales * eigenvectors[:, -1]
    interference_vector /= np.linalg.norm(interference_vector)
    q = np.eye(channels.shape[0], dtype=complex)[:, cfg.reference_channel]
    if score_db < cfg.detection_threshold_db:
        weights = q
        return channels[cfg.reference_channel].copy(), {
            "mode": "blind_null", "weights": weights,
            "sample_covariance": covariance, "normalized_correlation": correlation,
            "correlation_eigenvalues": eigenvalues, "detection_score_db": score_db,
            "estimated_interference_vector": interference_vector,
            "interference_detected": False, "nulling_enabled": False,
        }
    projector = np.eye(channels.shape[0]) - np.outer(interference_vector,
                                                     interference_vector.conj())
    projected_reference = projector @ q
    gamma = np.vdot(q, projected_reference).real
    weights = projected_reference / gamma
    return weights.conj() @ channels, {
        "mode": "blind_null", "weights": weights,
        "sample_covariance": covariance, "normalized_correlation": correlation,
        "correlation_eigenvalues": eigenvalues, "detection_score_db": score_db,
        "estimated_interference_vector": interference_vector,
        "reference_projection_power": gamma,
        "interference_detected": True, "nulling_enabled": True,
    }
```

The implementation deliberately has no zero-power, degenerate-eigenvector, minimum-projection, nonfinite-weight, or nonfinite-output branches.

- [ ] **Step 5: Run combiner tests**

Run: `python -m pytest python_sim_simplified/tests/test_diversity.py -v`

Expected: 2 passed.

- [ ] **Step 6: Commit spatial combining**

```bash
git add python_sim_simplified/diversity.py python_sim_simplified/tests/test_diversity.py
git commit -m "feat: add simplified spatial combiners"
```

---

### Task 6: Acquisition, Tracking, and C/N0

**Files:**
- Create: `python_sim_simplified/acquisition.py`
- Create: `python_sim_simplified/tracking.py`
- Create: `python_sim_simplified/tests/test_receiver.py`

**Interfaces:**
- Consumes: one combined DFE sample vector, sample rate, PRN, configs, carrier frequency, and nominal code rate.
- Produces: acquisition search dictionary, tracking history dictionary, and `estimate_cn0_db_hz(prompt_window, coherent_interval_s)`.

- [ ] **Step 1: Write failing acquisition and tracking tests**

```python
import numpy as np

from acquisition import acquire
from config import AcquisitionConfig, TrackingConfig
from gnss import ca_code
from tracking import estimate_cn0_db_hz, track


def receiver_signal(fs_hz=4.092e6, duration_ms=30, doppler_hz=1250.0, delay_chips=175.0):
    n = np.arange(round(fs_hz * duration_ms * 1e-3))
    phase = np.mod(n * 1.023e6 / fs_hz - delay_chips, 1023.0)
    code = ca_code(1)[np.floor(phase).astype(int)]
    clean = code * np.exp(1j * 2 * np.pi * doppler_hz * n / fs_hz)
    rng = np.random.default_rng(11)
    noise = 0.2 * (rng.standard_normal(n.size) + 1j * rng.standard_normal(n.size))
    return clean + noise


def test_acquisition_and_tracking_follow_valid_signal():
    fs_hz = 4.092e6
    samples = receiver_signal(fs_hz)
    acq = acquire(samples, fs_hz, 1, AcquisitionConfig())
    assert acq["success"]
    assert abs(acq["doppler_hz"] - 1250.0) <= 250.0
    trk = track(samples, fs_hz, 1, acq, TrackingConfig(max_epochs=20),
                1_575.42e6, 1.023e6)
    assert trk["success"]
    assert trk["prompt"].size > 5
    assert np.isfinite(trk["cn0_db_hz"][-1])


def test_cn0_matches_coherent_to_noise_power_ratio():
    prompt = np.array([10 + 1j, 10 - 1j, 10 + 1j, 10 - 1j])
    expected = 10.0 * np.log10(100.0 / (1.0 * 0.001))
    assert np.isclose(estimate_cn0_db_hz(prompt, 0.001), expected)
```

- [ ] **Step 2: Run the tests and verify receiver modules are absent**

Run: `python -m pytest python_sim_simplified/tests/test_receiver.py -v`

Expected: FAIL importing `acquisition` or `tracking`.

- [ ] **Step 3: Implement Doppler × code-phase FFT acquisition**

```python
def acquire(x, fs_hz, prn, cfg):
    samples_per_ms = round(fs_hz * 1e-3)
    coherent_samples = samples_per_ms * cfg.coherent_ms
    n = np.arange(coherent_samples)
    local_phase = np.mod(n * cfg.code_rate_hz / fs_hz, 1023.0)
    local_code = ca_code(prn)[np.floor(local_phase).astype(int)]
    code_spectrum = np.conj(np.fft.fft(local_code))
    search_power = np.zeros((len(cfg.doppler_bins_hz), coherent_samples))
    for row, doppler_hz in enumerate(cfg.doppler_bins_hz):
        wipeoff = np.exp(-1j * 2.0 * np.pi * doppler_hz * n / fs_hz)
        for block_index in range(cfg.noncoherent_ms):
            first = block_index * coherent_samples
            block = x[first:first + coherent_samples]
            correlation = np.fft.ifft(np.fft.fft(block * wipeoff) * code_spectrum)
            search_power[row] += np.abs(correlation) ** 2
```

Find the global maximum, suppress a circular ±`exclusion_chips` area in its Doppler row, use the remaining maximum as the second peak, and set `success = peak / second_peak >= threshold`. Return both the 2-D search matrix and axes for plotting. Do not substitute epsilon when the second peak is zero.

- [ ] **Step 4: Implement EPL tracking and float NCO updates**

For each 1 ms epoch, wipe off the carrier, correlate early/prompt/late codes, compute the normalized early-minus-late DLL discriminator, prompt phase PLL discriminator, and prompt-to-prompt FLL discriminator. Update with:

```python
carrier_frequency_hz += fll_correction + cfg.pll_ki_hz * pll_error_rad
carrier_phase_rad = wrap_phase(
    carrier_phase_rad
    + 2.0 * np.pi * carrier_frequency_hz * epoch_duration_s
    + cfg.pll_kp * pll_error_rad
)
dll_frequency_correction_hz += cfg.dll_ki_hz * dll_error
code_rate_hz = nominal_code_rate_hz * (
    1.0 + carrier_frequency_hz / carrier_hz
) + dll_frequency_correction_hz
code_phase_chips = np.mod(
    code_phase_chips
    + samples_per_epoch * code_rate_hz / fs_hz
    + cfg.dll_kp_chips * dll_error,
    1023.0,
)
```

Start tracking at the first acquisition code phase at or beyond the acquisition interval. Record prompt values, errors, carrier/code histories, phase-lock metric, and C/N0. `estimate_cn0_db_hz()` returns `nan` until two prompt samples exist; after that it uses coherent power divided by `(total_power - coherent_power) * coherent_interval_s` without a zero-noise fallback.

- [ ] **Step 5: Run receiver tests**

Run: `python -m pytest python_sim_simplified/tests/test_receiver.py -v`

Expected: 2 passed.

- [ ] **Step 6: Commit receiver processing**

```bash
git add python_sim_simplified/acquisition.py python_sim_simplified/tracking.py python_sim_simplified/tests/test_receiver.py
git commit -m "feat: add simplified acquisition and tracking"
```

---

### Task 7: End-to-End Pipeline and Metrics

**Files:**
- Create: `python_sim_simplified/metrics.py`
- Create: `python_sim_simplified/pipeline.py`
- Create: `python_sim_simplified/tests/test_pipeline.py`

**Interfaces:**
- Consumes: every processing function from Tasks 1–6.
- Produces: `run_end_to_end(cfg=None) -> dict`, `compute_metrics(results) -> list[dict]`, and `format_metrics(records) -> str`.

- [ ] **Step 1: Write failing integration tests**

```python
import numpy as np

from config import default_config
from pipeline import run_end_to_end


def test_pipeline_runs_all_four_branches_and_tracks_an_antijam_branch():
    cfg = default_config()
    cfg.signal.duration_ms = 30
    cfg.tracking.max_epochs = 20
    results = run_end_to_end(cfg)
    assert set(results["branches"]) == {"single", "egc", "mvdr", "blind_null"}
    assert results["dfe_channels"].shape[0] == 4
    assert all(np.all(np.isfinite(branch["combined"]))
               for branch in results["branches"].values())
    assert any(results["branches"][mode]["tracking"]["success"]
               for mode in ("mvdr", "blind_null"))


def test_pipeline_uses_assumed_target_only_for_known_combiner_constraints():
    cfg = default_config()
    cfg.signal.duration_ms = 12
    cfg.signal.assumed_direction.azimuth_deg += 10.0
    results = run_end_to_end(cfg)
    expected = results["truth"]["desired_steering_assumed"]
    np.testing.assert_allclose(
        results["branches"]["mvdr"]["combining"]["constraint_steering"], expected
    )
    assert "constraint_steering" not in results["branches"]["blind_null"]["combining"]
```

- [ ] **Step 2: Run the tests and verify pipeline modules are absent**

Run: `python -m pytest python_sim_simplified/tests/test_pipeline.py -v`

Expected: FAIL importing `pipeline`.

- [ ] **Step 3: Implement pipeline orchestration with only algorithmic gates**

```python
def run_end_to_end(cfg=None):
    cfg = default_config() if cfg is None else cfg
    scene_input_v, truth = generate_scenario(cfg)
    adc_codes, rf_info = process_rf(scene_input_v, cfg.rf, cfg.random_seed)
    dfe_channels, dfe_info = process_dfe(adc_codes, cfg.dfe)
    branches = {}
    for mode in cfg.diversity.modes:
        if mode == "blind_null":
            combined, combining = combine_blind_null(dfe_channels, cfg.diversity.blind_null)
        else:
            combined, combining = combine_known(
                dfe_channels, truth["desired_steering_assumed"], mode,
                cfg.diversity.diagonal_loading_factor,
            )
            combining["constraint_steering"] = truth["desired_steering_assumed"]
        acquisition = acquire(combined, dfe_info["fs_out_hz"],
                              cfg.signal.prn, cfg.acquisition)
        tracking = track(combined, dfe_info["fs_out_hz"], cfg.signal.prn,
                         acquisition, cfg.tracking, cfg.signal.carrier_hz,
                         cfg.signal.code_rate_hz) if acquisition["success"] else {
                             "success": False, "status": "acquisition_failed",
                             "prompt": np.array([], dtype=complex),
                         }
        branches[mode] = {"combined": combined, "combining": combining,
                          "acquisition": acquisition, "tracking": tracking}
```

After weights are known, add true target gain, all true interferer gains, worst interferer gain, snapshot count, and fixed-weight status. Assemble the stable result dictionary from the interface map and call `compute_metrics(results)` once.

- [ ] **Step 4: Implement concise metrics**

Each record contains `mode`, acquisition success/metric/Doppler, target gain, worst interferer gain, tracking success, final Doppler, PLL RMS, DLL RMS, mean prompt magnitude, and final finite C/N0. `format_metrics()` uses fixed-width plain text and no pandas dependency.

```python
def compute_metrics(results):
    records = []
    for mode, branch in results["branches"].items():
        acq = branch["acquisition"]
        trk = branch["tracking"]
        tracked = bool(trk["success"])
        window = slice(max(0, trk["prompt"].size - 10), trk["prompt"].size)
        records.append({
            "mode": mode,
            "acquired": bool(acq["success"]),
            "acquisition_metric": float(acq["metric"]),
            "tracking": tracked,
            "final_doppler_hz": float(trk["carrier_frequency_hz"][-1]) if tracked else None,
            "mean_prompt_magnitude": float(np.mean(np.abs(trk["prompt"][window]))) if tracked else None,
            "final_cn0_db_hz": final_finite_cn0_or_none(trk),
        })
    return records
```

Implement `final_finite_cn0_or_none()` as a short availability helper, not an exception or numerical fallback.

- [ ] **Step 5: Run pipeline tests**

Run: `python -m pytest python_sim_simplified/tests/test_pipeline.py -v`

Expected: 2 passed.

- [ ] **Step 6: Commit the integrated chain**

```bash
git add python_sim_simplified/metrics.py python_sim_simplified/pipeline.py python_sim_simplified/tests/test_pipeline.py
git commit -m "feat: integrate simplified receiver pipeline"
```

---

### Task 8: Independent Plots, Runner, and Teaching Documentation

**Files:**
- Create: `python_sim_simplified/plotting.py`
- Create: `python_sim_simplified/run_end_to_end.py`
- Create: `python_sim_simplified/requirements.txt`
- Create: `python_sim_simplified/README.md`
- Create: `python_sim_simplified/tests/test_plotting_runner.py`

**Interfaces:**
- Consumes: the result dictionary from `run_end_to_end()`.
- Produces: `plot_results(results) -> dict[str, matplotlib.figure.Figure]` and a direct script entry point.

- [ ] **Step 1: Write failing plot contract test**

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import default_config
from pipeline import run_end_to_end
from plotting import plot_results


def test_plot_results_returns_seven_independent_figures():
    cfg = default_config()
    cfg.signal.duration_ms = 20
    cfg.tracking.max_epochs = 10
    results = run_end_to_end(cfg)
    figures = plot_results(results)
    assert set(figures) == {
        "rf_spectrum", "dfe_spectrum", "combined_spectra",
        "array_response", "acquisition", "tracking", "cn0",
    }
    assert len({id(figure) for figure in figures.values()}) == 7
    for figure in figures.values():
        assert figure.axes
        plt.close(figure)
```

- [ ] **Step 2: Run the test and verify `plotting.py` is absent**

Run: `python -m pytest python_sim_simplified/tests/test_plotting_runner.py -v`

Expected: FAIL importing `plotting`.

- [ ] **Step 3: Implement spectrum, array, acquisition, tracking, and C/N0 figures**

```python
def spectrum(x, fs_hz):
    window = np.hanning(x.size)
    values = np.fft.fftshift(np.fft.fft(x * window))
    frequency_hz = np.fft.fftshift(np.fft.fftfreq(x.size, 1.0 / fs_hz))
    power_db = 20.0 * np.log10(np.abs(values) / np.max(np.abs(values)))
    return frequency_hz, power_db

def plot_results(results):
    figures = {}
    figures["rf_spectrum"] = plot_rf_spectrum(results)
    figures["dfe_spectrum"] = plot_dfe_spectrum(results)
    figures["combined_spectra"] = plot_combined_spectra(results)
    figures["array_response"] = plot_array_response(results)
    figures["acquisition"] = plot_acquisition(results)
    figures["tracking"] = plot_tracking(results)
    figures["cn0"] = plot_cn0(results)
    return figures
```

RF and DFE figures display every antenna channel. The combined-spectrum figure displays SINGLE, EGC, MVDR, and Blind Null together but remains separate from RF/DFE. The array-response figure evaluates true effective manifolds across azimuth at the target elevation and plots `|wᴴa(az)|`. Acquisition uses Doppler/code-phase heatmaps plus peak markers. Tracking plots carrier frequency, prompt magnitude, PLL error, and DLL error. C/N0 is its own figure.

- [ ] **Step 4: Implement the direct runner and requirements**

```python
# run_end_to_end.py
from config import default_config
from metrics import format_metrics
from pipeline import run_end_to_end
from plotting import plot_results

def main():
    cfg = default_config()
    results = run_end_to_end(cfg)
    print(format_metrics(results["metrics"]))
    figures = plot_results(results)
    for name, figure in figures.items():
        figure.savefig(cfg.plot.output_dir / f"{name}.png", dpi=150)
    if cfg.plot.show:
        import matplotlib.pyplot as plt
        plt.show()

if __name__ == "__main__":
    main()
```

The actual runner creates `cfg.plot.output_dir` before saving; this is normal requested output creation, not a failure-recovery branch. Set `requirements.txt` to:

```text
numpy>=2.0
scipy>=1.14
matplotlib>=3.9
pytest>=9.0
```

- [ ] **Step 5: Write README with the explicit teaching boundary**

Document these exact points:

- Run `python python_sim_simplified/run_end_to_end.py` from the repository root.
- Edit frequencies, duration, active corners, channel phase, interferers, and modes in `config.py` or immediately after `cfg = default_config()`.
- Use `parametric_config()` to exercise orientation/polarization/blockage/coupling and `measured_config(Path("pattern.csv"))` for CSV patterns.
- Read modules in the order `config → antenna/gnss → rf → dfe → diversity → acquisition → tracking → pipeline → plotting`.
- State clearly that the copy assumes valid controlled inputs and deliberately omits file checks, schema checks, denominator guards, matrix-degeneracy fallbacks, recovery, studies, and fixed-NCO comparison.
- Point to `../python_sim/` as the protected research implementation without importing it.

- [ ] **Step 6: Run plotting test and direct-script smoke test**

Run: `python -m pytest python_sim_simplified/tests/test_plotting_runner.py -v`

Expected: 1 passed.

Run with a noninteractive backend:

```powershell
$env:MPLBACKEND = 'Agg'
python python_sim_simplified/run_end_to_end.py
```

Expected: exit code 0, a four-row metric table, and seven nonempty PNG files below `python_sim_simplified/results/`.

- [ ] **Step 7: Commit plots and documentation**

```bash
git add python_sim_simplified/plotting.py python_sim_simplified/run_end_to_end.py python_sim_simplified/requirements.txt python_sim_simplified/README.md python_sim_simplified/tests/test_plotting_runner.py
git commit -m "docs: add simplified simulation runner and plots"
```

---

### Task 9: Final Contract and Regression Verification

**Files:**
- Create: `python_sim_simplified/tests/test_teaching_contract.py`
- Modify: `python_sim_simplified/README.md` only if a verified command or output name differs from the documentation.

**Interfaces:**
- Consumes: the completed teaching copy.
- Produces: evidence that the copy is independent, supports 2–4 corners and measured patterns, preserves phase in Blind Null, and does not regress the protected research package.

- [ ] **Step 1: Add the final independence and configuration tests**

```python
from itertools import combinations
from pathlib import Path

from antenna import antenna_response, build_antenna_model
from config import default_config
from pipeline import run_end_to_end


def test_teaching_modules_do_not_import_research_package():
    root = Path(__file__).resolve().parents[1]
    source = "\n".join(path.read_text(encoding="utf-8")
                       for path in root.glob("*.py"))
    assert "antenna_diversity" not in source
    assert "python_sim" not in source


def test_all_eleven_two_to_four_corner_combinations_form_a_response():
    names = ("TL", "TR", "BL", "BR")
    tested = []
    for size in (2, 3, 4):
        for active in combinations(names, size):
            cfg = default_config()
            cfg.phone.active_antennas = active
            response, _ = antenna_response(
                build_antenna_model(cfg),
                cfg.signal.true_direction,
                cfg.signal.true_polarization,
            )
            assert response.shape == (size,)
            tested.append(active)
    assert len(tested) == 11


def test_each_supported_corner_count_reaches_dfe_and_combining():
    for names in (("TL", "BR"), ("TL", "TR", "BR"), ("TL", "TR", "BL", "BR")):
        cfg = default_config()
        cfg.signal.duration_ms = 8
        cfg.tracking.max_epochs = 1
        cfg.phone.active_antennas = names
        results = run_end_to_end(cfg)
        assert results["dfe_channels"].shape[0] == len(names)
        assert set(results["branches"]) == set(cfg.diversity.modes)
```

- [ ] **Step 2: Run all teaching-copy tests**

Run: `python -m pytest python_sim_simplified/tests -v`

Expected: all teaching-copy tests pass with no warning promoted to an error.

- [ ] **Step 3: Run the protected research package test suite**

Run:

```powershell
Push-Location python_sim
python -m pytest tests -q
Pop-Location
```

Expected: the full existing suite passes; the last known baseline is 680 passed.

- [ ] **Step 4: Verify original source files are untouched**

Run: `git diff --name-only 4f1cb6c -- python_sim`

Expected: no output.

Run: `git diff --check 4f1cb6c -- python_sim_simplified docs/superpowers/plans`

Expected: no output.

- [ ] **Step 5: Inspect the generated plots**

Open the seven PNG files and confirm axes, labels, legends, acquisition peak markers, all configured combining modes, and a separate C/N0 figure are visible. If a label is clipped, adjust only `plotting.py`, rerun `test_plotting_runner.py`, and regenerate the PNG files.

- [ ] **Step 6: Commit the final contract**

```bash
git add python_sim_simplified/tests/test_teaching_contract.py python_sim_simplified/README.md
git commit -m "test: verify simplified simulation contract"
```

- [ ] **Step 7: Record final status**

Run: `git status --short`

Expected: no uncommitted files except intentionally ignored/generated `python_sim_simplified/results/*.png` if the repository ignore rules do not already cover them. Do not commit generated result images unless the user asks.
