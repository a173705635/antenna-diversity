# TRK Carrier NCO Fixed-Point Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, hardware-style fixed-point Carrier NCO to the Python TRK loop, preserve the current floating-point path as the default reference, and provide deterministic float/30-bit/43-bit comparison evidence.

**Architecture:** Keep the RF, ADC, DFE, diversity combiner, acquisition, correlators, discriminators, loop filters, Code NCO, lock metrics, and C/N0 estimator unchanged. Add one small stateful `FixedCarrierNCO` at the TRK carrier-replica boundary; `tracking.py` selects float or fixed state handling while sharing the remainder of the epoch loop. Add a focused analysis module and a thin CLI for reproducible comparisons.

**Tech Stack:** Python 3.11+, NumPy, Matplotlib, dataclasses, argparse, csv, pytest, Git.

**Spec:** [`docs/superpowers/specs/2026-08-25-tracking-carrier-nco-fixed-point-design.md`](../specs/2026-08-25-tracking-carrier-nco-fixed-point-design.md)

## Global Constraints

- Work only on branch `codex/tracking-fixed-nco`.
- Do not stage or modify `CODEX_MEMORY_GNSS_ANTENNA_DIVERSITY.md`, `python_sim/.idea/`, or `python_sim/requirements.txt`.
- Do not change DFE arithmetic. It remains floating-point DDC/FIR followed by configured I/Q output quantization.
- Do not fixed-point the correlators, discriminators, loop filters, Code NCO, lock metric, or C/N0 estimator.
- Preserve `TrackingConfig.carrier_nco_mode == "float"` as the default and keep existing float results within current test tolerances.
- Use `antenna_diversity.numeric.matlab_round` for all NCO word conversions; do not use Python `round` or `numpy.rint` for quantization.
- Reject FCW overflow and invalid state with `ValueError`; do not saturate, wrap, or hide it.
- Do not add warning filters. Run relevant tests and CLIs with `-W error` and `MPLBACKEND=Agg`.
- Do not add generated CSV, PNG, cache, or `results/` artifacts to Git.
- After every task, inspect `git status --short` and stage only the files listed in that task.

---

## Task 1: Implement and unit-test the fixed Carrier NCO core

**Files:**

- Create: `python_sim/antenna_diversity/receiver/carrier_nco.py`
- Create: `python_sim/tests/test_carrier_nco.py`

- [ ] **Step 1: Write the frequency-resolution and quantization tests first**

Create `python_sim/tests/test_carrier_nco.py` with these imports and initial tests:

```python
import numpy as np
import pytest

from antenna_diversity.receiver.carrier_nco import FixedCarrierNCO


def make_nco(*, bits: int = 30, initial_frequency_hz: float = 0.0):
    return FixedCarrierNCO(
        clock_hz=6.6e6,
        word_bits=bits,
        coherent_interval_s=1e-3,
        initial_frequency_hz=initial_frequency_hz,
    )


def test_frequency_lsb_matches_word_length():
    nco_30 = make_nco(bits=30)
    nco_43 = make_nco(bits=43)

    assert nco_30.frequency_lsb_hz == pytest.approx(6.6e6 / 2**30)
    assert nco_43.frequency_lsb_hz == pytest.approx(6.6e6 / 2**43)
    assert nco_30.frequency_lsb_hz / nco_43.frequency_lsb_hz == pytest.approx(8192.0)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_half_lsb_rounds_away_from_zero(sign):
    nco = make_nco()

    update = nco.update(
        command_frequency_hz=sign * 0.5 * nco.frequency_lsb_hz,
        phase_correction_rad=0.0,
    )

    assert update.fcw == int(sign)
    assert update.frequency_hz == pytest.approx(sign * nco.frequency_lsb_hz)
    assert update.quantization_error_hz == pytest.approx(
        update.frequency_hz - update.command_frequency_hz
    )


def test_initial_frequency_is_quantized_and_exposed():
    nco = make_nco(initial_frequency_hz=1500.123)

    assert nco.fcw == int(np.sign(1500.123) * np.floor(abs(1500.123) / nco.frequency_lsb_hz + 0.5))
    assert nco.frequency_hz == pytest.approx(nco.fcw * nco.frequency_lsb_hz)
    assert nco.command_frequency_hz == 1500.123
    assert nco.quantization_error_hz == pytest.approx(nco.frequency_hz - 1500.123)
```

- [ ] **Step 2: Run the new test and verify it fails because the module is absent**

Run from `F:\antenna_diversity\python_sim`:

```powershell
$env:MPLBACKEND='Agg'
python -W error -m pytest tests/test_carrier_nco.py -q
```

Expected: collection fails with `ModuleNotFoundError: antenna_diversity.receiver.carrier_nco`.

- [ ] **Step 3: Add the public NCO data model and frequency quantizer**

Create `python_sim/antenna_diversity/receiver/carrier_nco.py` with this public interface and validation shape:

```python
"""Hardware-style fixed-point carrier NCO used at TRK epoch boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from antenna_diversity.numeric import matlab_round


@dataclass(frozen=True)
class CarrierNCOUpdate:
    command_frequency_hz: float
    fcw: int
    frequency_hz: float
    quantization_error_hz: float
    phase_word: int


def _finite_real_scalar(value: object, name: str) -> float:
    candidate = np.asarray(value)
    if (
        candidate.ndim != 0
        or not np.issubdtype(candidate.dtype, np.number)
        or np.issubdtype(candidate.dtype, np.complexfloating)
        or isinstance(value, (bool, np.bool_))
        or not bool(np.isfinite(candidate))
    ):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(candidate)


class FixedCarrierNCO:
    """N-bit signed FCW plus N-bit unsigned phase accumulator."""

    def __init__(
        self,
        *,
        clock_hz: float,
        word_bits: int,
        coherent_interval_s: float,
        initial_frequency_hz: float,
    ) -> None:
        self.clock_hz = _finite_real_scalar(clock_hz, "clock_hz")
        if self.clock_hz <= 0.0:
            raise ValueError("clock_hz must be positive")
        if isinstance(word_bits, (bool, np.bool_)) or not isinstance(
            word_bits, (int, np.integer)
        ):
            raise ValueError("word_bits must be an integer")
        self.word_bits = int(word_bits)
        if not 2 <= self.word_bits <= 62:
            raise ValueError("word_bits must lie in [2, 62]")

        interval_s = _finite_real_scalar(
            coherent_interval_s, "coherent_interval_s"
        )
        if interval_s <= 0.0:
            raise ValueError("coherent_interval_s must be positive")
        ticks_exact = self.clock_hz * interval_s
        ticks = int(np.floor(ticks_exact + 0.5))
        if not np.isclose(ticks_exact, ticks, rtol=0.0, atol=1e-9):
            raise ValueError("coherent interval must contain an integer number of NCO ticks")

        self.modulus = 1 << self.word_bits
        self.minimum_fcw = -(1 << (self.word_bits - 1))
        self.maximum_fcw = (1 << (self.word_bits - 1)) - 1
        self.n_ticks = ticks
        self.phase_word = 0
        initial = self._quantize_frequency(initial_frequency_hz)
        self.command_frequency_hz = initial.command_frequency_hz
        self.fcw = initial.fcw
        self.frequency_hz = initial.frequency_hz
        self.quantization_error_hz = initial.quantization_error_hz

    @property
    def frequency_lsb_hz(self) -> float:
        return self.clock_hz / self.modulus

    @property
    def phase_rad(self) -> float:
        return float(2.0 * np.pi * self.phase_word / self.modulus)

    def _quantize_frequency(self, command_frequency_hz: float) -> CarrierNCOUpdate:
        command = _finite_real_scalar(command_frequency_hz, "command_frequency_hz")
        fcw = int(matlab_round(command / self.frequency_lsb_hz).item())
        if not self.minimum_fcw <= fcw <= self.maximum_fcw:
            raise ValueError("carrier FCW exceeds the signed N-bit range")
        actual = fcw * self.frequency_lsb_hz
        if not np.isfinite(actual):
            raise ValueError("quantized carrier frequency must be finite")
        return CarrierNCOUpdate(
            command_frequency_hz=command,
            fcw=fcw,
            frequency_hz=actual,
            quantization_error_hz=actual - command,
            phase_word=self.phase_word,
        )
```

Do not export a saturation option. Keep `_quantize_frequency` internal so all state transitions go through `update()`.

- [ ] **Step 4: Add phase-accumulator, wrap, and invalid-input tests**

Append these tests:

```python
def test_update_advances_unsigned_phase_word_with_new_fcw_and_phase_correction():
    nco = make_nco()
    command_hz = 25.0 * nco.frequency_lsb_hz
    correction_words = 7
    correction_rad = 2.0 * np.pi * correction_words / nco.modulus

    update = nco.update(command_hz, correction_rad)

    assert update.fcw == 25
    assert update.phase_word == (25 * nco.n_ticks + correction_words) % nco.modulus
    assert nco.phase_word == update.phase_word
    assert nco.phase_rad == pytest.approx(2.0 * np.pi * update.phase_word / nco.modulus)


def test_negative_fcw_wraps_phase_word_into_unsigned_range():
    nco = make_nco()

    update = nco.update(-nco.frequency_lsb_hz, 0.0)

    assert update.fcw == -1
    assert update.phase_word == (-nco.n_ticks) % nco.modulus
    assert 0 <= update.phase_word < nco.modulus


def test_signed_frequency_range_includes_negative_nyquist_only():
    nco = make_nco()

    update = nco.update(-0.5 * nco.clock_hz, 0.0)
    assert update.fcw == nco.minimum_fcw
    assert update.frequency_hz == -0.5 * nco.clock_hz

    with pytest.raises(ValueError, match="FCW"):
        nco.update(0.5 * nco.clock_hz, 0.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"clock_hz": 0.0}, "clock_hz"),
        ({"clock_hz": np.nan}, "clock_hz"),
        ({"word_bits": True}, "word_bits"),
        ({"word_bits": 1}, "word_bits"),
        ({"word_bits": 63}, "word_bits"),
        ({"coherent_interval_s": 1.5 / 6.6e6}, "integer number"),
    ],
)
def test_constructor_rejects_invalid_configuration(kwargs, message):
    arguments = dict(
        clock_hz=6.6e6,
        word_bits=30,
        coherent_interval_s=1e-3,
        initial_frequency_hz=0.0,
    )
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        FixedCarrierNCO(**arguments)


@pytest.mark.parametrize("command_hz", [np.nan, np.inf, -np.inf, 3.3e6])
def test_update_rejects_nonfinite_or_overflowing_frequency(command_hz):
    nco = make_nco()
    with pytest.raises(ValueError, match="frequency|FCW"):
        nco.update(command_hz, 0.0)


@pytest.mark.parametrize("correction_rad", [np.nan, np.inf, -np.inf])
def test_update_rejects_nonfinite_phase_correction(correction_rad):
    nco = make_nco()
    with pytest.raises(ValueError, match="phase_correction_rad"):
        nco.update(0.0, correction_rad)
```

The boundary test fixes the supported interval as `[-f_clk/2, f_clk/2)`: `-3.3 MHz` is representable, while `+3.3 MHz` maps to `+2^(N-1)` and is rejected.

- [ ] **Step 5: Implement `update()` using the new FCW for the epoch-boundary phase advance**

Add this method to `FixedCarrierNCO`:

```python
    def update(
        self,
        command_frequency_hz: float,
        phase_correction_rad: float,
    ) -> CarrierNCOUpdate:
        quantized = self._quantize_frequency(command_frequency_hz)
        correction = _finite_real_scalar(
            phase_correction_rad, "phase_correction_rad"
        )
        correction_word = int(
            matlab_round(correction / (2.0 * np.pi) * self.modulus).item()
        )
        self.phase_word = (
            self.phase_word + quantized.fcw * self.n_ticks + correction_word
        ) % self.modulus
        self.command_frequency_hz = quantized.command_frequency_hz
        self.fcw = quantized.fcw
        self.frequency_hz = quantized.frequency_hz
        self.quantization_error_hz = quantized.quantization_error_hz
        return CarrierNCOUpdate(
            command_frequency_hz=self.command_frequency_hz,
            fcw=self.fcw,
            frequency_hz=self.frequency_hz,
            quantization_error_hz=self.quantization_error_hz,
            phase_word=self.phase_word,
        )
```

- [ ] **Step 6: Run focused tests and commit the NCO core**

```powershell
cd F:\antenna_diversity\python_sim
$env:MPLBACKEND='Agg'
python -W error -m pytest tests/test_carrier_nco.py -q
cd F:\antenna_diversity
git diff --check
git status --short
git add -- python_sim/antenna_diversity/receiver/carrier_nco.py python_sim/tests/test_carrier_nco.py
git commit -m "feat: add fixed-point tracking carrier nco"
```

Expected: all `test_carrier_nco.py` tests pass; only the two task files are staged.

---

## Task 2: Add validated configuration and preserve study JSON overrides

**Files:**

- Modify: `python_sim/antenna_diversity/config.py`
- Modify: `python_sim/tests/test_config_numeric.py`
- Modify: `python_sim/tests/test_study_config.py`

- [ ] **Step 1: Write failing validation tests for the three new fields**

Append to `python_sim/tests/test_config_numeric.py`:

```python
@pytest.mark.parametrize("mode", ["integer", "FIXED", "", None, ["fixed"]])
def test_tracking_rejects_unknown_carrier_nco_mode(mode):
    cfg = default_config()
    cfg.tracking.carrier_nco_mode = mode
    with pytest.raises(ValueError, match="carrier_nco_mode"):
        cfg.validate()


@pytest.mark.parametrize("clock_hz", [0.0, -1.0, np.nan, np.inf, True, 1.0 + 0.0j])
def test_tracking_rejects_invalid_carrier_nco_clock(clock_hz):
    cfg = default_config()
    cfg.tracking.carrier_nco_clock_hz = clock_hz
    with pytest.raises(ValueError, match="carrier_nco_clock_hz"):
        cfg.validate()


@pytest.mark.parametrize("word_bits", [True, 1, 63, 30.5])
def test_tracking_rejects_invalid_carrier_nco_word_bits(word_bits):
    cfg = default_config()
    cfg.tracking.carrier_nco_word_bits = word_bits
    with pytest.raises(ValueError, match="carrier_nco_word_bits"):
        cfg.validate()


def test_fixed_tracking_rejects_fractional_nco_ticks_per_epoch():
    cfg = default_config()
    cfg.tracking.carrier_nco_mode = "fixed"
    cfg.tracking.carrier_nco_clock_hz = 6_600_000.5
    with pytest.raises(ValueError, match="integer number of NCO ticks"):
        cfg.validate()
```

If the test module uses different fixture/import conventions, reuse its existing `default_config`, `np`, and `pytest` imports rather than duplicating them.

- [ ] **Step 2: Run the config tests and verify they fail on missing validation**

```powershell
cd F:\antenna_diversity\python_sim
python -W error -m pytest tests/test_config_numeric.py -q
```

Expected: the new invalid cases do not raise yet.

- [ ] **Step 3: Add fields and chain-level validation to `TrackingConfig`/`SimulationConfig.validate()`**

Extend `TrackingConfig` directly after the existing tracking limits:

```python
    carrier_nco_mode: str = "float"
    carrier_nco_clock_hz: float = 6.6e6
    carrier_nco_word_bits: int = 30
```

After the existing tracking integration checks in `SimulationConfig.validate()`, add:

```python
        if (
            not isinstance(self.tracking.carrier_nco_mode, str)
            or self.tracking.carrier_nco_mode not in {"float", "fixed"}
        ):
            raise ValueError('tracking.carrier_nco_mode must be "float" or "fixed"')
        if (
            isinstance(self.tracking.carrier_nco_clock_hz, (bool, np.bool_))
            or not _is_finite_real_scalar(self.tracking.carrier_nco_clock_hz)
            or self.tracking.carrier_nco_clock_hz <= 0.0
        ):
            raise ValueError(
                "tracking.carrier_nco_clock_hz must be a finite positive real scalar"
            )
        if (
            isinstance(self.tracking.carrier_nco_word_bits, (bool, np.bool_))
            or not isinstance(
                self.tracking.carrier_nco_word_bits, (int, np.integer)
            )
            or not 2 <= int(self.tracking.carrier_nco_word_bits) <= 62
        ):
            raise ValueError(
                "tracking.carrier_nco_word_bits must be an integer in [2, 62]"
            )
        if self.tracking.carrier_nco_mode == "fixed":
            nco_ticks = (
                self.tracking.carrier_nco_clock_hz
                * self.tracking.coherent_ms
                * 1e-3
            )
            if not np.isclose(nco_ticks, round(nco_ticks), rtol=0.0, atol=1e-9):
                raise ValueError(
                    "tracking coherent interval must contain an integer number of NCO ticks"
                )
```

Keep exact validation duplicated at the configuration boundary and NCO boundary: configuration errors should fail before RF sample generation, while direct `FixedCarrierNCO` callers remain protected.

- [ ] **Step 4: Prove study JSON overrides resolve the flat tracking fields**

In `python_sim/tests/test_study_config.py`, add this test using its existing `json`, `Path`, and `resolve_study_config` imports:

```python
def test_json_overrides_tracking_carrier_nco_fields(tmp_path: Path):
    path = tmp_path / "tracking-nco.json"
    path.write_text(
        json.dumps({
            "simulation": {
                "tracking": {
                    "carrier_nco_mode": "fixed",
                    "carrier_nco_clock_hz": 6_600_000.0,
                    "carrier_nco_word_bits": 43,
                }
            }
        }),
        encoding="utf-8",
    )

    resolved = resolve_study_config(path, preset="quick")

    assert resolved.simulation.tracking.carrier_nco_mode == "fixed"
    assert resolved.simulation.tracking.carrier_nco_clock_hz == 6.6e6
    assert resolved.simulation.tracking.carrier_nco_word_bits == 43
    resolved.simulation.validate()
```

Do not add NCO-specific JSON parsing because `_update_dataclass()` already handles flat dataclass fields.

- [ ] **Step 5: Run focused configuration/study tests and commit**

```powershell
cd F:\antenna_diversity\python_sim
$env:MPLBACKEND='Agg'
python -W error -m pytest tests/test_config_numeric.py tests/test_study_config.py -q
cd F:\antenna_diversity
git diff --check
git status --short
git add -- python_sim/antenna_diversity/config.py python_sim/tests/test_config_numeric.py python_sim/tests/test_study_config.py
git commit -m "feat: configure fixed tracking carrier nco"
```

---

## Task 3: Integrate fixed NCO state into the shared TRK loop

**Files:**

- Modify: `python_sim/antenna_diversity/receiver/tracking.py`
- Modify: `python_sim/tests/test_tracking.py`

- [ ] **Step 1: Lock down default float output semantics before refactoring**

Extend `test_tracking_converges_after_successful_acquisition()` with:

```python
    assert trk["carrier_nco_mode"] == "float"
    np.testing.assert_array_equal(
        trk["carrier_frequency_command_hz"],
        trk["carrier_frequency_hz"],
    )
    np.testing.assert_array_equal(
        trk["carrier_frequency_quantization_error_hz"],
        np.zeros_like(trk["carrier_frequency_hz"]),
    )
    assert trk["carrier_nco_frequency_lsb_hz"] == 0.0
    assert "carrier_fcw" not in trk
    assert "carrier_phase_word" not in trk
```

This test defines `carrier_frequency_hz` as the actual replica frequency and the new command history as the pre-quantization loop command.

- [ ] **Step 2: Add fixed 30-bit/43-bit integration tests**

Add a helper that reuses the same deterministic `x` and `acq`, then add:

```python
@pytest.mark.parametrize("word_bits", [30, 43])
def test_fixed_carrier_nco_tracks_and_exposes_hardware_state(word_bits):
    cfg = default_config()
    cfg.tracking.carrier_nco_mode = "fixed"
    cfg.tracking.carrier_nco_word_bits = word_bits
    x = make_tracking_signal(cfg)
    acq = acquire(x, cfg.dfe.fs_out_hz, cfg.signal.prn, cfg.acquisition)

    trk = track(
        x,
        cfg.dfe.fs_out_hz,
        cfg.signal.prn,
        acq,
        cfg.tracking,
        cfg.signal.carrier_hz,
        cfg.signal.code_rate_hz,
    )

    assert trk["success"]
    assert trk["carrier_nco_mode"] == "fixed"
    assert trk["carrier_nco_frequency_lsb_hz"] == pytest.approx(6.6e6 / 2**word_bits)
    assert trk["carrier_fcw"].dtype == np.int64
    assert trk["carrier_phase_word"].dtype == np.uint64
    assert len(trk["carrier_fcw"]) == len(trk["prompt"])
    assert np.all(trk["carrier_phase_word"] < 2**word_bits)
    np.testing.assert_allclose(
        trk["carrier_frequency_hz"],
        trk["carrier_fcw"].astype(np.float64) * 6.6e6 / 2**word_bits,
        rtol=0.0,
        atol=np.finfo(float).eps * 6.6e6,
    )
    np.testing.assert_allclose(
        trk["carrier_frequency_quantization_error_hz"],
        trk["carrier_frequency_hz"] - trk["carrier_frequency_command_hz"],
        rtol=0.0,
        atol=np.finfo(float).eps * 6.6e6,
    )
    assert abs(trk["carrier_frequency_hz"][-1] - cfg.signal.doppler_hz) <= (
        cfg.tracking.max_final_doppler_error_hz
    )


def test_43_bit_fixed_nco_has_no_larger_quantization_bound_than_30_bit():
    histories = {}
    for bits in (30, 43):
        cfg = default_config()
        cfg.tracking.carrier_nco_mode = "fixed"
        cfg.tracking.carrier_nco_word_bits = bits
        x = make_tracking_signal(cfg)
        acq = acquire(x, cfg.dfe.fs_out_hz, cfg.signal.prn, cfg.acquisition)
        histories[bits] = track(
            x, cfg.dfe.fs_out_hz, cfg.signal.prn, acq, cfg.tracking,
            cfg.signal.carrier_hz, cfg.signal.code_rate_hz,
        )

    for bits, trk in histories.items():
        assert np.max(np.abs(trk["carrier_frequency_quantization_error_hz"])) <= (
            0.5 * trk["carrier_nco_frequency_lsb_hz"]
            + np.finfo(float).eps * 6.6e6
        )
    assert histories[43]["carrier_nco_frequency_lsb_hz"] < histories[30][
        "carrier_nco_frequency_lsb_hz"
    ]


def test_direct_tracking_call_rejects_unknown_carrier_nco_mode():
    cfg = default_config()
    cfg.tracking.carrier_nco_mode = "integer"
    x = make_tracking_signal(cfg)
    acq = acquire(x, cfg.dfe.fs_out_hz, cfg.signal.prn, cfg.acquisition)

    with pytest.raises(ValueError, match="carrier_nco_mode"):
        track(
            x, cfg.dfe.fs_out_hz, cfg.signal.prn, acq, cfg.tracking,
            cfg.signal.carrier_hz, cfg.signal.code_rate_hz,
        )
```

Do not assert that the 43-bit trajectory has smaller final Doppler error; loop dynamics and noise do not guarantee that scientific conclusion.

- [ ] **Step 3: Run the new tracking tests and verify the missing keys/fixed path fail**

```powershell
cd F:\antenna_diversity\python_sim
$env:MPLBACKEND='Agg'
python -W error -m pytest tests/test_tracking.py -q
```

- [ ] **Step 4: Add NCO selection and histories without duplicating the correlator loop**

In `tracking.py`, import:

```python
from antenna_diversity.receiver.carrier_nco import FixedCarrierNCO
```

At the start of the successful-acquisition path, reject an unknown mode even when `track()` is called directly instead of through `run_end_to_end()`:

```python
    if cfg.carrier_nco_mode not in {"float", "fixed"}:
        raise ValueError('carrier_nco_mode must be "float" or "fixed"')
```

Immediately after computing `epoch_duration_s`, initialize common command state and optional fixed state:

```python
    carrier_frequency_command_hz = float(acq["doppler_hz"])
    fixed_nco = None
    if cfg.carrier_nco_mode == "fixed":
        fixed_nco = FixedCarrierNCO(
            clock_hz=cfg.carrier_nco_clock_hz,
            word_bits=cfg.carrier_nco_word_bits,
            coherent_interval_s=epoch_duration_s,
            initial_frequency_hz=carrier_frequency_command_hz,
        )
        carrier_frequency_hz = fixed_nco.frequency_hz
        carrier_phase_rad = fixed_nco.phase_rad
        frequency_lsb_hz = fixed_nco.frequency_lsb_hz
        carrier_fcw_history = np.zeros(n_epochs, dtype=np.int64)
        carrier_phase_word_history = np.zeros(n_epochs, dtype=np.uint64)
    else:
        carrier_frequency_hz = carrier_frequency_command_hz
        carrier_phase_rad = 0.0
        frequency_lsb_hz = 0.0
```

Allocate in both modes:

```python
    carrier_frequency_command_history = np.zeros(n_epochs, dtype=np.float64)
    carrier_frequency_quantization_error_history = np.zeros(
        n_epochs, dtype=np.float64
    )
```

Keep the existing carrier-replica expression unchanged: in fixed mode its two inputs are simply the NCO's current `phase_rad` and `frequency_hz`.

- [ ] **Step 5: Replace only the carrier update block with float/fixed state transitions**

After the discriminator and FLL assist calculation, compute the loop command once:

```python
        carrier_frequency_command_hz += (
            fll_correction + cfg.pll_ki_hz * pll_error_rad[epoch]
        )
        phase_correction_rad = cfg.pll_kp * pll_error_rad[epoch]

        if fixed_nco is None:
            carrier_frequency_hz = carrier_frequency_command_hz
            carrier_phase_rad = _wrap_phase(
                carrier_phase_rad
                + 2.0 * np.pi * carrier_frequency_hz * epoch_duration_s
                + phase_correction_rad
            )
            quantization_error_hz = 0.0
        else:
            nco_update = fixed_nco.update(
                carrier_frequency_command_hz,
                phase_correction_rad,
            )
            carrier_frequency_hz = nco_update.frequency_hz
            carrier_phase_rad = fixed_nco.phase_rad
            quantization_error_hz = nco_update.quantization_error_hz
            carrier_fcw_history[epoch] = nco_update.fcw
            carrier_phase_word_history[epoch] = nco_update.phase_word
```

Continue using `carrier_frequency_hz`—the actual replica frequency—in the existing code-rate Doppler coupling. This preserves the physical meaning of the current Code NCO input without fixed-pointing it.

Record post-update state:

```python
        carrier_frequency_command_history[epoch] = carrier_frequency_command_hz
        carrier_frequency_history[epoch] = carrier_frequency_hz
        carrier_frequency_quantization_error_history[epoch] = quantization_error_hz
```

- [ ] **Step 6: Extend only successful result dictionaries**

Add common success keys:

```python
        "carrier_nco_mode": cfg.carrier_nco_mode,
        "carrier_frequency_command_hz": carrier_frequency_command_history,
        "carrier_frequency_hz": carrier_frequency_history,
        "carrier_frequency_quantization_error_hz": (
            carrier_frequency_quantization_error_history
        ),
        "carrier_nco_frequency_lsb_hz": frequency_lsb_hz,
```

Build the result dict first, then conditionally add fixed state so acquisition-failure and float outputs do not pretend to contain hardware words:

```python
    if fixed_nco is not None:
        result["carrier_fcw"] = carrier_fcw_history
        result["carrier_phase_word"] = carrier_phase_word_history
    return result
```

Leave the existing early `acquisition_failed` dictionary byte-for-byte equivalent in shape and meaning.

- [ ] **Step 7: Run focused and end-to-end regressions, then commit**

```powershell
cd F:\antenna_diversity\python_sim
$env:MPLBACKEND='Agg'
python -W error -m pytest tests/test_carrier_nco.py tests/test_tracking.py tests/test_end_to_end.py tests/test_plotting.py -q
python -W error run_end_to_end.py
cd F:\antenna_diversity
git diff --check
git status --short
git add -- python_sim/antenna_diversity/receiver/tracking.py python_sim/tests/test_tracking.py
git commit -m "feat: integrate fixed carrier nco into tracking"
```

Expected: default CLI still uses float mode; no DFE file appears in the diff.

---

## Task 4: Add deterministic float/30-bit/43-bit comparison analysis

**Files:**

- Create: `python_sim/antenna_diversity/analysis/tracking_nco.py`
- Create: `python_sim/tests/test_tracking_nco_comparison.py`

- [ ] **Step 1: Define the comparison result shape and write failing metric tests**

Create `python_sim/tests/test_tracking_nco_comparison.py`. Use a short deterministic configuration (`plot.enable = False`, enough duration for acquisition plus at least two tracking epochs) and test:

```python
import numpy as np
import pytest

from antenna_diversity.analysis.tracking_nco import (
    L1_CARRIER_HZ,
    compare_tracking_ncos,
)
from antenna_diversity.config import SPEED_OF_LIGHT_MPS, default_config


def test_comparison_runs_float_30_and_43_without_mutating_source_config():
    cfg = default_config()
    cfg.plot.enable = False
    original_mode = cfg.tracking.carrier_nco_mode
    original_bits = cfg.tracking.carrier_nco_word_bits

    comparison = compare_tracking_ncos(
        cfg, clock_hz=6.6e6, fixed_word_bits=(30, 43)
    )

    assert tuple(comparison.runs) == ("float", "fixed_30", "fixed_43")
    assert cfg.tracking.carrier_nco_mode == original_mode
    assert cfg.tracking.carrier_nco_word_bits == original_bits
    assert comparison.runs["float"].config.random_seed == cfg.random_seed
    assert comparison.runs["fixed_30"].config.random_seed == cfg.random_seed
    assert comparison.runs["fixed_43"].config.random_seed == cfg.random_seed
    reference = comparison.runs["float"]
    for case_name in ("fixed_30", "fixed_43"):
        candidate = comparison.runs[case_name]
        np.testing.assert_array_equal(candidate.scene_input_v, reference.scene_input_v)
        np.testing.assert_array_equal(candidate.adc_codes, reference.adc_codes)
        np.testing.assert_array_equal(candidate.dfe_channels, reference.dfe_channels)
        for branch_name in cfg.diversity.modes:
            branch_key = branch_name.lower()
            candidate_acq = candidate.branches[branch_key].acquisition
            reference_acq = reference.branches[branch_key].acquisition
            assert candidate_acq["success"] == reference_acq["success"]
            if reference_acq["success"]:
                assert candidate_acq["doppler_hz"] == reference_acq["doppler_hz"]
                assert candidate_acq["code_phase_samples"] == reference_acq[
                    "code_phase_samples"
                ]


def test_comparison_rows_cover_every_run_and_diversity_branch():
    cfg = default_config()
    cfg.plot.enable = False
    comparison = compare_tracking_ncos(cfg, clock_hz=6.6e6, fixed_word_bits=(30, 43))

    assert len(comparison.rows) == 3 * len(cfg.diversity.modes)
    assert {(row["nco_case"], row["branch"]) for row in comparison.rows} == {
        (case, branch)
        for case in ("float", "fixed_30", "fixed_43")
        for branch in (name.lower() for name in cfg.diversity.modes)
    }
    row_30 = next(
        row for row in comparison.rows
        if row["nco_case"] == "fixed_30" and row["branch"] == "mvdr"
    )
    expected_lsb = 6.6e6 / 2**30
    assert row_30["frequency_lsb_hz"] == pytest.approx(expected_lsb)
    assert row_30["l1_velocity_lsb_mm_s"] == pytest.approx(
        1000.0 * SPEED_OF_LIGHT_MPS / L1_CARRIER_HZ * expected_lsb
    )
    assert row_30["fcw_unique_count"] >= 1
    assert row_30["fcw_jump_count"] >= 0
```

The comparison result is:

```python
@dataclass(frozen=True)
class TrackingNCOComparison:
    rows: tuple[dict[str, object], ...]
    runs: dict[str, SimulationResults]
```

- [ ] **Step 2: Run the new test and confirm the analysis module is absent**

```powershell
cd F:\antenna_diversity\python_sim
$env:MPLBACKEND='Agg'
python -W error -m pytest tests/test_tracking_nco_comparison.py -q
```

- [ ] **Step 3: Implement deep-copied deterministic runs and fixed CSV columns**

Create `python_sim/antenna_diversity/analysis/tracking_nco.py` with:

```python
"""Reproducible float/fixed Carrier NCO comparison helpers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from antenna_diversity.config import SPEED_OF_LIGHT_MPS, SimulationConfig
from antenna_diversity.models import SimulationResults
from antenna_diversity.pipeline import run_end_to_end


L1_CARRIER_HZ = 1_575.42e6
CSV_COLUMNS = (
    "nco_case", "branch", "mode", "word_bits", "clock_hz",
    "frequency_lsb_hz", "l1_velocity_lsb_mm_s", "final_fcw",
    "fcw_unique_count", "fcw_jump_count", "final_command_hz",
    "final_actual_hz", "quantization_error_mean_hz",
    "quantization_error_rms_hz", "quantization_error_max_abs_hz",
    "final_doppler_error_hz", "pll_error_rms_rad", "dll_error_rms",
    "prompt_magnitude_mean", "final_cn0_db_hz",
)


@dataclass(frozen=True)
class TrackingNCOComparison:
    rows: tuple[dict[str, object], ...]
    runs: dict[str, SimulationResults]
```

Implement `compare_tracking_ncos(cfg, *, clock_hz, fixed_word_bits)` by:

1. Rejecting non-positive/non-finite clock and any empty/duplicate/out-of-range word-bit tuple, including bool values.
2. Creating ordered cases `("float", None)`, then `("fixed_30", 30)`, `("fixed_43", 43)` in the caller's bit order.
3. Deep-copying `cfg` for every case, disabling plots, setting only the three NCO fields, calling `validate()`, then `run_end_to_end()`.
4. Never mutating `cfg` or reusing a result object between cases.
5. Building one row for every configured diversity branch.

Use this exact row policy:

```python
def _tracking_row(case_name: str, branch_name: str, result: SimulationResults) -> dict[str, object]:
    cfg = result.config
    branch_key = branch_name.lower()
    tracking = result.branches[branch_key].tracking
    is_fixed = cfg.tracking.carrier_nco_mode == "fixed"
    lsb = (
        cfg.tracking.carrier_nco_clock_hz / 2**cfg.tracking.carrier_nco_word_bits
        if is_fixed
        else 0.0
    )
    identity = {
        "nco_case": case_name,
        "branch": branch_key,
        "mode": cfg.tracking.carrier_nco_mode,
        "word_bits": cfg.tracking.carrier_nco_word_bits if is_fixed else "",
        "clock_hz": cfg.tracking.carrier_nco_clock_hz if is_fixed else "",
        "frequency_lsb_hz": lsb,
        "l1_velocity_lsb_mm_s": (
            1000.0 * SPEED_OF_LIGHT_MPS / L1_CARRIER_HZ * lsb
        ),
    }
    if not tracking.get("success", False):
        return {
            **identity,
            **{name: "" for name in CSV_COLUMNS[7:]},
        }

    command = np.asarray(tracking["carrier_frequency_command_hz"], dtype=float)
    actual = np.asarray(tracking["carrier_frequency_hz"], dtype=float)
    error = np.asarray(tracking["carrier_frequency_quantization_error_hz"], dtype=float)
    pll = np.asarray(tracking["pll_error_rad"], dtype=float)
    dll = np.asarray(tracking["dll_error"], dtype=float)
    prompt = np.asarray(tracking["prompt"], dtype=complex)
    cn0 = np.asarray(tracking["cn0_db_hz"], dtype=float)
    fcw = np.asarray(tracking["carrier_fcw"], dtype=np.int64) if is_fixed else np.array([], dtype=np.int64)
    return {
        **identity,
        "final_fcw": int(fcw[-1]) if is_fixed else "",
        "fcw_unique_count": int(np.unique(fcw).size) if is_fixed else "",
        "fcw_jump_count": int(np.count_nonzero(np.diff(fcw))) if is_fixed else "",
        "final_command_hz": float(command[-1]),
        "final_actual_hz": float(actual[-1]),
        "quantization_error_mean_hz": float(np.mean(error)),
        "quantization_error_rms_hz": float(np.sqrt(np.mean(error**2))),
        "quantization_error_max_abs_hz": float(np.max(np.abs(error))),
        "final_doppler_error_hz": float(actual[-1] - cfg.signal.doppler_hz),
        "pll_error_rms_rad": float(np.sqrt(np.mean(pll**2))),
        "dll_error_rms": float(np.sqrt(np.mean(dll**2))),
        "prompt_magnitude_mean": float(np.mean(np.abs(prompt))),
        "final_cn0_db_hz": float(cn0[-1]) if np.isfinite(cn0[-1]) else "",
    }
```

Before returning, assert/validate that every numeric scalar written to a row is finite; use an empty string only for semantically unavailable fields.

- [ ] **Step 4: Add independent frequency and quantization-error figures**

Add:

```python
def plot_tracking_nco_comparison(
    comparison: TrackingNCOComparison,
) -> dict[str, plt.Figure]:
```

Figure rules:

- Return exactly `{"tracking_nco_frequency": fig1, "tracking_nco_quantization_error": fig2}`.
- Use one subplot per branch, shared x-axis in tracking epoch number.
- Frequency figure overlays the actual `carrier_frequency_hz` for float, fixed 30, and fixed 43; also show the true Doppler as a thin dashed reference.
- Quantization figure overlays `carrier_frequency_quantization_error_hz`; float must appear as a zero reference.
- When a branch fails acquisition, annotate that subplot as unavailable for that case and do not access nonexistent tracking histories.
- Each axis has units, grid, case labels, and branch title; use `tight_layout()`.
- Return figures without `show()` and without saving.

Add tests that each returned object has `len(cfg.diversity.modes)` axes and that plotted y-data equals the corresponding stored history. Close those test-created figures with `plt.close(figure)` so the warning-clean full suite cannot accumulate open figures.

- [ ] **Step 5: Add CSV/PNG persistence with explicit outputs**

Add:

```python
def save_tracking_nco_comparison(
    comparison: TrackingNCOComparison,
    output_dir: Path,
) -> dict[str, Path]:
```

Implementation requirements:

- Create only the requested `output_dir` with `parents=True, exist_ok=True`.
- Write `tracking_nco_comparison.csv` using `csv.DictWriter(..., fieldnames=CSV_COLUMNS, extrasaction="raise")` and `newline=""`, UTF-8 encoding.
- Generate/save figures at 160 dpi as `tracking_nco_frequency.png` and `tracking_nco_quantization_error.png`.
- Close only the figures created by this function.
- Return absolute resolved paths under keys `csv`, `frequency_figure`, and `quantization_error_figure`.

Test with `tmp_path`: exact CSV header, expected row count, non-empty PNGs, and all three returned paths contained by `tmp_path.resolve()`.

- [ ] **Step 6: Run focused tests and commit the reusable analysis layer**

```powershell
cd F:\antenna_diversity\python_sim
$env:MPLBACKEND='Agg'
python -W error -m pytest tests/test_tracking_nco_comparison.py -q
cd F:\antenna_diversity
git diff --check
git status --short
git add -- python_sim/antenna_diversity/analysis/tracking_nco.py python_sim/tests/test_tracking_nco_comparison.py
git commit -m "feat: compare tracking carrier nco resolutions"
```

---

## Task 5: Add the standalone comparison CLI

**Files:**

- Create: `python_sim/run_tracking_nco_comparison.py`
- Modify: `python_sim/tests/test_tracking_nco_comparison.py`

- [ ] **Step 1: Write CLI parser and smoke tests first**

Add `from pathlib import Path` and tests that import `build_parser` and `main` from `run_tracking_nco_comparison`:

```python
def test_cli_defaults_match_l1_validation_case():
    args = build_parser().parse_args([])
    assert args.clock_hz == 6.6e6
    assert args.word_bits == (30, 43)
    assert args.output_dir == Path("results/tracking_nco")


def test_cli_writes_comparison_artifacts(tmp_path, capsys):
    exit_code = main(["--output-dir", str(tmp_path)])

    assert exit_code == 0
    assert (tmp_path / "tracking_nco_comparison.csv").stat().st_size > 0
    assert (tmp_path / "tracking_nco_frequency.png").stat().st_size > 0
    assert (tmp_path / "tracking_nco_quantization_error.png").stat().st_size > 0
    output = capsys.readouterr().out
    assert str((tmp_path / "tracking_nco_comparison.csv").resolve()) in output
```

The CLI artifact test must exercise the real `run_end_to_end` chain; do not mock the pipeline.

- [ ] **Step 2: Implement a thin CLI with comma-separated bit parsing**

Create `python_sim/run_tracking_nco_comparison.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from antenna_diversity.analysis.tracking_nco import (
    compare_tracking_ncos,
    save_tracking_nco_comparison,
)
from antenna_diversity.config import default_config


def _word_bits(value: str) -> tuple[int, ...]:
    try:
        bits = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("word bits must be comma-separated integers") from error
    if not bits:
        raise argparse.ArgumentTypeError("at least one word length is required")
    if len(set(bits)) != len(bits) or any(not 2 <= item <= 62 for item in bits):
        raise argparse.ArgumentTypeError("word bits must be unique integers in [2, 62]")
    return bits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare float, 30-bit, and 43-bit tracking Carrier NCOs."
    )
    parser.add_argument("--clock-hz", type=float, default=6.6e6)
    parser.add_argument("--word-bits", type=_word_bits, default=(30, 43))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/tracking_nco")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = default_config()
    cfg.plot.enable = False
    comparison = compare_tracking_ncos(
        cfg,
        clock_hz=args.clock_hz,
        fixed_word_bits=args.word_bits,
    )
    paths = save_tracking_nco_comparison(comparison, args.output_dir)
    for label, path in paths.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run parser, focused test, and real CLI smoke test**

```powershell
cd F:\antenna_diversity\python_sim
$env:MPLBACKEND='Agg'
python -W error -m pytest tests/test_tracking_nco_comparison.py -q
python -W error run_tracking_nco_comparison.py --output-dir results/tracking_nco
Get-Item results/tracking_nco/tracking_nco_comparison.csv
Get-Item results/tracking_nco/tracking_nco_frequency.png
Get-Item results/tracking_nco/tracking_nco_quantization_error.png
```

Inspect the CSV to confirm nine rows for the default three NCO cases × SINGLE/EGC/MVDR and no `nan`/`inf` text. The `results/` directory must remain ignored/untracked.

- [ ] **Step 4: Commit only CLI and its test changes**

```powershell
cd F:\antenna_diversity
git diff --check
git status --short
git add -- python_sim/run_tracking_nco_comparison.py python_sim/tests/test_tracking_nco_comparison.py
git commit -m "feat: add tracking nco comparison cli"
```

---

## Task 6: Document numerical boundaries and user workflow

**Files:**

- Modify: `python_sim/README.md`
- Modify: `python_sim/DESIGN.md`

- [ ] **Step 1: Add a concise README usage section**

Add a `Carrier NCO 定点验证` section after the environment/entry-point section. It must show:

```python
cfg = default_config()
cfg.tracking.carrier_nco_mode = "fixed"
cfg.tracking.carrier_nco_clock_hz = 6.6e6
cfg.tracking.carrier_nco_word_bits = 30
result = run_end_to_end(cfg)
```

Also show:

```powershell
$env:MPLBACKEND='Agg'
python -W error run_tracking_nco_comparison.py `
  --clock-hz 6600000 --word-bits 30,43 `
  --output-dir results/tracking_nco
```

State explicitly:

- Default mode is float.
- 30-bit/6.6 MHz L1 LSB is `6.6e6 / 2^30 ≈ 0.006147 Hz`.
- 43-bit is 8192 times finer than 30-bit at the same clock.
- `carrier_frequency_hz` is actual NCO frequency and `carrier_frequency_command_hz` is the float loop command.
- This comparison does not prove the observed measured bias is caused by NCO quantization.

- [ ] **Step 2: Update DESIGN section 7 and model boundaries**

Extend `DESIGN.md` section 7 with the fixed Carrier NCO equations:

```text
FCW = round_away_from_zero(f_cmd / f_clk * 2^N)
f_actual = FCW * f_clk / 2^N
phase_word[k+1] = (phase_word[k] + FCW[k+1] * n_ticks + phase_correction_word) mod 2^N
```

Document the exact update ordering and output histories. In the unmodeled list, add:

- full TRK fixed-point arithmetic;
- fixed correlator/loop-filter/Code-NCO word lengths;
- asynchronous DDS/sample-clock behavior and DDS spurs;
- L5 signal/code end-to-end support.

Clarify the DFE boundary: DDC/FIR are float and only the DFE output samples are quantized to configured I/Q codes.

- [ ] **Step 3: Review docs against implemented names and commit**

```powershell
cd F:\antenna_diversity
rg -n "carrier_nco|carrier_frequency_command|30-bit|43-bit|DFE" python_sim/README.md python_sim/DESIGN.md
git diff --check
git status --short
git add -- python_sim/README.md python_sim/DESIGN.md
git commit -m "docs: explain tracking carrier nco validation"
```

---

## Task 7: Full verification and final branch audit

**Files:**

- No planned source changes; fix only failures directly caused by Tasks 1–6.

- [ ] **Step 1: Run the complete warning-clean test suite**

```powershell
cd F:\antenna_diversity\python_sim
$env:MPLBACKEND='Agg'
python -W error -m pytest -q
```

Expected: all existing and new tests pass with no warning filters.

- [ ] **Step 2: Run all maintained CLI regressions**

```powershell
python -W error run_end_to_end.py
python -W error run_legacy_ula.py
python -W error run_tracking_nco_comparison.py --output-dir results/tracking_nco
python -W error run_spatial_research.py --config antenna_diversity/studies/example_antenna_realism.json --preset quick --name tracking-nco-regression
```

If the study output name already exists, choose a fresh ignored name; do not delete user results. Confirm float mode remains the default in the pre-existing end-to-end, legacy ULA, and spatial-research workflows.

- [ ] **Step 3: Compile, inspect artifacts, and verify scientific invariants**

```powershell
python -W error -m compileall -q antenna_diversity run_tracking_nco_comparison.py
Select-String -Path results/tracking_nco/tracking_nco_comparison.csv -Pattern 'nan|inf'
cd F:\antenna_diversity
git diff --check
git status --short --branch
git log --oneline --decorate -8
```

Expected:

- `Select-String` returns no matches.
- Generated results do not appear in `git status`.
- User-owned untracked files remain untracked and unchanged.
- No file under `python_sim/antenna_diversity/dfe/` is modified.
- The branch contains the design, plan, implementation, tests, CLI, and docs commits.

- [ ] **Step 4: Verify the branch diff is scoped and contains no placeholders**

```powershell
git diff --stat main...HEAD
git diff --name-only main...HEAD
rg -n "TODO|TBD|FIXME|NotImplemented" python_sim/antenna_diversity/receiver/carrier_nco.py python_sim/antenna_diversity/receiver/tracking.py python_sim/antenna_diversity/analysis/tracking_nco.py python_sim/run_tracking_nco_comparison.py python_sim/tests/test_carrier_nco.py python_sim/tests/test_tracking_nco_comparison.py
```

Any hit must be removed or explained before completion. Do not claim success until all commands above have current passing output.

- [ ] **Step 5: Commit any verification-only correction separately**

Only if Task 7 exposes a real defect, make the smallest fix, rerun the failing command and full suite, then stage the exact affected files and commit:

```powershell
git commit -m "fix: correct tracking nco validation regression"
```

If no correction is needed, do not create an empty commit.
