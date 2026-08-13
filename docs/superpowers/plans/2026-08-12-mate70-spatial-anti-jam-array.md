# Mate 70 Spatial Anti-Jam Array Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the Python GPS L1 C/A simulation from a fixed dual-element ULA into a Mate 70 four-corner spatial anti-jam platform with arbitrary 2–4 antenna subsets, 3-D directions, up to three CW interferers, fast spatial experiments, and full RF-to-tracking verification.

**Architecture:** Add phone geometry, direction, pattern, and interferer configuration as focused types; compute a common 3-D array manifold before the existing RF/DFE/receiver chain; and preserve the old ULA through an explicit legacy constructor. Reuse the existing channels-by-samples interfaces and generalized combiner, then add analytical spatial evaluation, combination sweeps, Monte Carlo orchestration, and a 2-D azimuth/elevation response plot around the same pipeline.

**Tech Stack:** Python 3.11+, NumPy 2.x, SciPy 1.14+, Matplotlib 3.9+, pytest 9+, standard-library `dataclasses`, `itertools`, `copy`, and `json`.

## Global Constraints

- Use HUAWEI Mate 70 outer dimensions `length=0.1609 m`, `width=0.0759 m`, `thickness=0.0078 m`.
- Use phone coordinates `+x=right`, `+y=top`, `+z=screen normal`; azimuth is counterclockwise from `+x`, elevation is `[0°, 90°]` above the phone plane.
- Candidate phase centers are ideal outer corners `TL/TR/BL/BR`; they are not claims about Huawei's real antenna locations.
- Support every unique subset of 2, 3, or 4 corners: 6 + 4 + 1 = 11 combinations.
- The new default is all four corners; preserve the current dual-antenna ULA as `legacy_ula_config()` and `run_legacy_ula.py`.
- First-stage element patterns are isotropic complex gain `1+0j`, but all 3-D manifold calls must pass through an element-pattern interface.
- Support zero to three independently configured static CW interferers, including power, direction, offset, phase, and start/stop time.
- Keep RF, DFE, acquisition, and tracking array interfaces as `(num_antennas, num_samples)` and combined signals as `(num_samples,)`.
- Keep `single`, `egc`, and diagonally loaded `mvdr`; do not add LCMV, MUSIC, ESPRIT, GSC, or robust Capon in this plan.
- Use test-driven development: write each behavioral test, run it and observe the expected failure, implement the minimum behavior, then run focused and regression tests.
- Do not add new third-party dependencies.
- Preserve unrelated user changes; stage only the exact files listed by each task.

## File Structure

### New production files

- `python_sim/antenna_diversity/array/geometry.py`: Mate 70 phase centers, active-corner selection, baselines, and 11-combination enumeration.
- `python_sim/antenna_diversity/array/manifold.py`: 3-D unit vectors, element-pattern protocol, isotropic pattern, and array-manifold calculation.
- `python_sim/antenna_diversity/analysis/spatial.py`: analytical covariance, combiner spatial metrics, and azimuth/elevation response grids.
- `python_sim/antenna_diversity/experiments/__init__.py`: public experiment exports.
- `python_sim/antenna_diversity/experiments/spatial_scan.py`: all-combination and direction-grid fast experiments.
- `python_sim/antenna_diversity/experiments/monte_carlo.py`: repeatable full-chain seed orchestration and rate summaries.
- `python_sim/run_legacy_ula.py`: explicit CLI for the original dual-antenna regression case.

### Modified production files

- `python_sim/antenna_diversity/config.py`: new direction, phone-array, and CW-interferer configuration types; new and legacy constructors.
- `python_sim/antenna_diversity/gnss/scenario.py`: common desired-signal generation plus legacy or phone manifold and multiple time-gated CW components.
- `python_sim/antenna_diversity/diversity/combiner.py`: extract reusable covariance-to-weights calculation while preserving sample-data combining.
- `python_sim/antenna_diversity/pipeline.py`: use assumed steering for combining and carry expanded truth through all branches.
- `python_sim/antenna_diversity/analysis/metrics.py`: multiple-interferer gains and spatial-quality fields.
- `python_sim/antenna_diversity/analysis/plotting.py`: multiple jammer markers and 2-D phone-array response.
- `python_sim/antenna_diversity/models.py`: typed experiment result containers only where dictionaries become ambiguous.
- `python_sim/run_end_to_end.py`: run the four-corner default.
- `python_sim/pyproject.toml`: update the description from dual antenna to 2–4 antenna phone-array research.
- `python_sim/README.md`, `python_sim/ARCHITECTURE.md`, `python_sim/DESIGN.md`: document the new default and modeling limits.
- `python_sim/docs/simulation_architecture.svg`: replace the dual-antenna-only architecture labels and limitations.

### New test files

- `python_sim/tests/test_phone_config.py`
- `python_sim/tests/test_geometry.py`
- `python_sim/tests/test_manifold.py`
- `python_sim/tests/test_multi_interferer_scenario.py`
- `python_sim/tests/test_spatial_analysis.py`
- `python_sim/tests/test_experiments.py`
- `python_sim/tests/test_monte_carlo.py`

### Existing tests to modify

- `python_sim/tests/test_config_numeric.py`
- `python_sim/tests/test_core.py`
- `python_sim/tests/test_scenario.py`
- `python_sim/tests/test_diversity.py`
- `python_sim/tests/test_end_to_end.py`
- `python_sim/tests/test_plotting.py`

---

### Task 1: Configuration types and explicit legacy/new constructors

**Files:**
- Modify: `python_sim/antenna_diversity/config.py`
- Create: `python_sim/tests/test_phone_config.py`
- Modify: `python_sim/tests/test_config_numeric.py`

**Interfaces:**
- Produces: `Direction`, `PhoneGeometryConfig`, `PhoneArrayConfig`, `CWInterferer`, `mate70_default_config()`, and `legacy_ula_config()`.
- Transitional behavior: keep `default_config()` returning the legacy configuration until Task 9, so all existing runtime tests remain green while upstream support is added.
- `PhoneGeometryConfig` is the single owner of real and assumed phase-center coordinates; do not duplicate coordinates in `PhoneArrayConfig`.
- `PhoneArrayConfig.channel_response(estimated=False)` returns the real or estimated complex channel-calibration vector in `active_antennas` order. Zero estimated errors represent an uncalibrated nominal constraint; copying real errors into estimated errors represents ideal calibration; intermediate values represent partial calibration.
- `SignalConfig.true_direction` and `SignalConfig.assumed_direction` are `Direction | None`; both are `None` in the legacy ULA and non-`None` in the phone model.
- `SimulationConfig.array` is `ArrayConfig | PhoneArrayConfig`, `SimulationConfig.phone` is `PhoneGeometryConfig | None`, and `SimulationConfig.interferers` is `tuple[CWInterferer, ...]`. The existing singular `jammer` remains only for legacy compatibility.

- [ ] **Step 1: Write failing tests for the new configuration API**

```python
import numpy as np
import pytest

from antenna_diversity.config import (
    CWInterferer,
    Direction,
    legacy_ula_config,
    mate70_default_config,
)


def test_mate70_config_uses_all_four_ideal_corners():
    cfg = mate70_default_config()

    assert cfg.phone.model_name == "HUAWEI Mate 70"
    assert cfg.phone.length_m == pytest.approx(0.1609)
    assert cfg.phone.width_m == pytest.approx(0.0759)
    assert cfg.array.active_antennas == ("TL", "TR", "BL", "BR")
    np.testing.assert_allclose(
        [cfg.phone.phase_centers_m[name] for name in cfg.array.active_antennas],
        [
            [-0.03795, 0.08045, 0.0],
            [0.03795, 0.08045, 0.0],
            [-0.03795, -0.08045, 0.0],
            [0.03795, -0.08045, 0.0],
        ],
    )


def test_phone_config_accepts_arbitrary_unique_two_to_four_corner_subset():
    cfg = mate70_default_config()
    cfg.array.active_antennas = ("TR", "BL")
    cfg.validate()

    assert cfg.array.num_elements == 2
    np.testing.assert_allclose(
        [cfg.phone.phase_centers_m[name] for name in cfg.array.active_antennas],
        [[0.03795, 0.08045, 0.0], [-0.03795, -0.08045, 0.0]],
    )


@pytest.mark.parametrize(
    "active",
    [("TL",), ("TL", "TL"), ("TL", "XX"), ("TL", "TR", "BL", "BR", "TL")],
)
def test_phone_config_rejects_invalid_corner_selections(active):
    cfg = mate70_default_config()
    cfg.array.active_antennas = active

    with pytest.raises(ValueError, match="active_antennas"):
        cfg.validate()


@pytest.mark.parametrize("elevation", [-0.1, 90.1])
def test_direction_rejects_elevation_outside_visible_hemisphere(elevation):
    with pytest.raises(ValueError, match="elevation"):
        Direction(30.0, elevation).validate()


def test_phone_config_rejects_more_than_three_interferers():
    cfg = mate70_default_config()
    cfg.interferers = tuple(
        CWInterferer(direction=Direction(index * 30.0, 10.0))
        for index in range(4)
    )

    with pytest.raises(ValueError, match="three"):
        cfg.validate()


def test_legacy_constructor_preserves_original_sampling_and_array():
    cfg = legacy_ula_config()

    assert cfg.array.num_elements == 2
    assert cfg.signal.fs_rf_hz == 16.368e6
    assert cfg.dfe.fs_out_hz == 4.092e6
    assert cfg.array.desired_angle_deg == -10.0
```

- [ ] **Step 2: Run the focused tests and verify the API is missing**

Run: `cd F:\antenna_diversity\python_sim; python -m pytest tests/test_phone_config.py -v`

Expected: collection fails because `Direction`, `PhoneGeometryConfig`, `PhoneArrayConfig`, `CWInterferer`, `mate70_default_config`, and `legacy_ula_config` do not exist.

- [ ] **Step 3: Implement the configuration types and validation**

Add these exact public shapes to `config.py`:

```python
@dataclass(frozen=True)
class Direction:
    azimuth_deg: float
    elevation_deg: float

    def validate(self) -> None:
        if not np.isfinite(self.azimuth_deg):
            raise ValueError("direction.azimuth_deg must be finite")
        if not np.isfinite(self.elevation_deg) or not 0.0 <= self.elevation_deg <= 90.0:
            raise ValueError("direction.elevation_deg must lie in [0, 90]")


@dataclass
class PhoneGeometryConfig:
    model_name: str
    length_m: float
    width_m: float
    thickness_m: float
    phase_centers_m: dict[str, tuple[float, float, float]]
    assumed_phase_centers_m: dict[str, tuple[float, float, float]]


@dataclass
class PhoneArrayConfig:
    active_antennas: tuple[str, ...]
    channel_amplitude_error_db: dict[str, float]
    channel_phase_error_deg: dict[str, float]
    estimated_channel_amplitude_error_db: dict[str, float]
    estimated_channel_phase_error_deg: dict[str, float]
    pattern_model: str = "isotropic"

    @property
    def num_elements(self) -> int:
        return len(self.active_antennas)

    def channel_response(self, estimated: bool = False) -> NDArray[np.complex128]:
        amplitude_source = (
            self.estimated_channel_amplitude_error_db
            if estimated
            else self.channel_amplitude_error_db
        )
        phase_source = (
            self.estimated_channel_phase_error_deg
            if estimated
            else self.channel_phase_error_deg
        )
        amplitude_db = np.asarray(
            [amplitude_source[name] for name in self.active_antennas]
        )
        phase_deg = np.asarray(
            [phase_source[name] for name in self.active_antennas]
        )
        return 10.0 ** (amplitude_db / 20.0) * np.exp(1j * np.deg2rad(phase_deg))


@dataclass(frozen=True)
class CWInterferer:
    direction: Direction
    enable: bool = True
    power_dbm: float = -90.0
    offset_hz: float = 250e3
    initial_phase_rad: float = 0.8
    start_ms: float = 0.0
    stop_ms: float | None = None
```

Add `true_direction: Direction | None = None` and `assumed_direction: Direction | None = None` to `SignalConfig`. Declare `SimulationConfig.array: ArrayConfig | PhoneArrayConfig`, `phone: PhoneGeometryConfig | None`, and `interferers: tuple[CWInterferer, ...]`; retain `jammer: JammerConfig` for the legacy constructor. Validation must require both directions and a non-`None` phone config for `PhoneArrayConfig`, and require both directions to be `None` for legacy `ArrayConfig`. Validate real and assumed coordinate maps plus all four real/estimated channel-error maps against the same corner-name set. `mate70_default_config()` initializes assumed coordinates equal to real coordinates, all real/estimated channel errors to zero, target `(30°, 60°)`, one default CW at `(120°, 10°)`, and all corners. `legacy_ula_config()` must contain the current `default_config()` body unchanged. Keep `default_config()` as `return legacy_ula_config()` until Task 9.

- [ ] **Step 4: Run focused and existing configuration tests**

Run: `cd F:\antenna_diversity\python_sim; python -m pytest tests/test_phone_config.py tests/test_config_numeric.py -v`

Expected: all new tests pass; the legacy sampling test still passes.

- [ ] **Step 5: Commit only configuration work**

```powershell
git add -- python_sim/antenna_diversity/config.py python_sim/tests/test_phone_config.py python_sim/tests/test_config_numeric.py
git commit -m "feat: add Mate 70 phone-array configuration"
```

---

### Task 2: Mate 70 geometry and all active-corner combinations

**Files:**
- Create: `python_sim/antenna_diversity/array/geometry.py`
- Create: `python_sim/tests/test_geometry.py`
- Modify: `python_sim/antenna_diversity/array/__init__.py`

**Interfaces:**
- Produces: `mate70_phase_centers()`, `select_phase_centers()`, `active_corner_combinations()`, and `maximum_baseline_m()`.
- Consumers: configuration construction in Task 1 is refactored to call `mate70_phase_centers()` rather than duplicate constants.

- [ ] **Step 1: Write geometry tests**

```python
import itertools
import numpy as np
import pytest

from antenna_diversity.array.geometry import (
    active_corner_combinations,
    mate70_phase_centers,
    maximum_baseline_m,
    select_phase_centers,
)


def test_active_corner_combinations_returns_all_eleven_subsets():
    combinations = active_corner_combinations()

    assert len(combinations) == 11
    assert len(set(combinations)) == 11
    assert sum(len(item) == 2 for item in combinations) == 6
    assert sum(len(item) == 3 for item in combinations) == 4
    assert sum(len(item) == 4 for item in combinations) == 1


def test_select_phase_centers_preserves_requested_order():
    centers = mate70_phase_centers()
    selected = select_phase_centers(centers, ("BR", "TL", "TR"))

    np.testing.assert_allclose(
        selected,
        [centers["BR"], centers["TL"], centers["TR"]],
    )


def test_diagonal_maximum_baseline_matches_phone_outline():
    positions = select_phase_centers(mate70_phase_centers(), ("TL", "BR"))

    assert maximum_baseline_m(positions) == pytest.approx(
        np.hypot(0.1609, 0.0759)
    )
```

- [ ] **Step 2: Verify tests fail because the geometry module is absent**

Run: `cd F:\antenna_diversity\python_sim; python -m pytest tests/test_geometry.py -v`

Expected: import failure for `antenna_diversity.array.geometry`.

- [ ] **Step 3: Implement focused geometry helpers**

```python
CORNER_ORDER = ("TL", "TR", "BL", "BR")


def mate70_phase_centers(
    inset_x_m: float = 0.0,
    inset_y_m: float = 0.0,
    z_offsets_m: Mapping[str, float] | None = None,
) -> dict[str, tuple[float, float, float]]:
    half_width = 0.0759 / 2.0 - inset_x_m
    half_length = 0.1609 / 2.0 - inset_y_m
    z = {name: 0.0 for name in CORNER_ORDER}
    if z_offsets_m is not None:
        z.update(z_offsets_m)
    return {
        "TL": (-half_width, half_length, z["TL"]),
        "TR": (half_width, half_length, z["TR"]),
        "BL": (-half_width, -half_length, z["BL"]),
        "BR": (half_width, -half_length, z["BR"]),
    }


def active_corner_combinations() -> tuple[tuple[str, ...], ...]:
    return tuple(
        combination
        for size in (2, 3, 4)
        for combination in itertools.combinations(CORNER_ORDER, size)
    )
```

Validate nonnegative insets that do not invert the outline, finite 3-D centers, known corner names, and at least two positions for baseline calculation. Refactor `mate70_default_config()` to consume `mate70_phase_centers()`.

- [ ] **Step 4: Run geometry, phone-config, and full regression tests**

Run: `cd F:\antenna_diversity\python_sim; python -m pytest tests/test_geometry.py tests/test_phone_config.py -v`

Then run: `python -m pytest -q`

Expected: geometry tests pass and all existing tests remain green.

- [ ] **Step 5: Commit geometry work**

```powershell
git add -- python_sim/antenna_diversity/array/geometry.py python_sim/antenna_diversity/array/__init__.py python_sim/antenna_diversity/config.py python_sim/tests/test_geometry.py
git commit -m "feat: add Mate 70 corner geometry"
```

---

### Task 3: 3-D direction, isotropic element pattern, and array manifold

**Files:**
- Create: `python_sim/antenna_diversity/array/manifold.py`
- Create: `python_sim/tests/test_manifold.py`
- Modify: `python_sim/antenna_diversity/array/__init__.py`

**Interfaces:**
- Consumes: `Direction` and `(M, 3)` phase-center arrays.
- Produces: `ElementPattern` protocol, `IsotropicElementPattern`, `direction_unit_vector()`, and `array_manifold()`.
- Exact signature: `array_manifold(positions_m, direction, frequency_hz, element_names=None, pattern=None) -> NDArray[np.complex128]`.

- [ ] **Step 1: Write direction and manifold tests**

```python
import numpy as np

from antenna_diversity.array.manifold import (
    IsotropicElementPattern,
    array_manifold,
    direction_unit_vector,
)
from antenna_diversity.array.geometry import select_phase_centers
from antenna_diversity.config import Direction, mate70_default_config


def test_direction_unit_vectors_follow_phone_axes():
    np.testing.assert_allclose(direction_unit_vector(Direction(0.0, 0.0)), [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(direction_unit_vector(Direction(90.0, 0.0)), [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(direction_unit_vector(Direction(0.0, 90.0)), [0, 0, 1], atol=1e-12)


def test_isotropic_manifold_has_unit_magnitude():
    cfg = mate70_default_config()
    positions = select_phase_centers(
        cfg.phone.phase_centers_m,
        cfg.array.active_antennas,
    )
    response = array_manifold(
        positions,
        Direction(30.0, 60.0),
        cfg.signal.carrier_hz,
        cfg.array.active_antennas,
        IsotropicElementPattern(),
    )

    np.testing.assert_allclose(np.abs(response), np.ones(4), atol=1e-12)


def test_coplanar_array_has_identical_response_at_zenith():
    cfg = mate70_default_config()
    positions = select_phase_centers(
        cfg.phone.phase_centers_m,
        cfg.array.active_antennas,
    )
    response = array_manifold(
        positions,
        Direction(217.0, 90.0),
        cfg.signal.carrier_hz,
    )

    np.testing.assert_allclose(response, np.ones(4), atol=1e-12)


def test_selected_subarray_equals_full_array_slice():
    cfg = mate70_default_config()
    full_positions = select_phase_centers(cfg.phone.phase_centers_m, cfg.array.active_antennas)
    full = array_manifold(full_positions, Direction(120.0, 10.0), cfg.signal.carrier_hz)
    cfg.array.active_antennas = ("TR", "BR")
    selected_positions = select_phase_centers(cfg.phone.phase_centers_m, cfg.array.active_antennas)
    selected = array_manifold(selected_positions, Direction(120.0, 10.0), cfg.signal.carrier_hz)

    np.testing.assert_allclose(selected, full[[1, 3]], atol=1e-12)
```

- [ ] **Step 2: Verify the new manifold tests fail**

Run: `cd F:\antenna_diversity\python_sim; python -m pytest tests/test_manifold.py -v`

Expected: import failure for `array.manifold`.

- [ ] **Step 3: Implement the pattern protocol and 3-D phase model**

```python
class ElementPattern(Protocol):
    def complex_gain(
        self,
        element_name: str,
        direction: Direction,
        frequency_hz: float,
    ) -> complex: ...


@dataclass(frozen=True)
class IsotropicElementPattern:
    def complex_gain(
        self,
        element_name: str,
        direction: Direction,
        frequency_hz: float,
    ) -> complex:
        return 1.0 + 0.0j


def direction_unit_vector(direction: Direction) -> NDArray[np.float64]:
    direction.validate()
    azimuth = np.deg2rad(direction.azimuth_deg)
    elevation = np.deg2rad(direction.elevation_deg)
    return np.asarray(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ],
        dtype=np.float64,
    )


def array_manifold(
    positions_m: ArrayLike,
    direction: Direction,
    frequency_hz: float,
    element_names: Sequence[str] | None = None,
    pattern: ElementPattern | None = None,
) -> NDArray[np.complex128]:
    positions = np.asarray(positions_m, dtype=np.float64)
    unit = direction_unit_vector(direction)
    spatial = np.exp(-1j * 2.0 * np.pi * frequency_hz / SPEED_OF_LIGHT_MPS * (positions @ unit))
    names = tuple(str(index) for index in range(positions.shape[0])) if element_names is None else tuple(element_names)
    selected_pattern = IsotropicElementPattern() if pattern is None else pattern
    gains = np.asarray(
        [selected_pattern.complex_gain(name, direction, frequency_hz) for name in names],
        dtype=np.complex128,
    )
    return gains * spatial
```

Replace the ellipsis in the protocol body with Python's literal `...`; this is the only deliberate protocol stub and not an unimplemented runtime path. Validate shape `(M,3)`, finite values, positive frequency, and matching element-name length.

- [ ] **Step 4: Run manifold and legacy steering tests**

Run: `cd F:\antenna_diversity\python_sim; python -m pytest tests/test_manifold.py tests/test_core.py -v`

Expected: all pass; the existing 1-D `steering_vector()` behavior remains unchanged.

- [ ] **Step 5: Commit manifold work**

```powershell
git add -- python_sim/antenna_diversity/array/manifold.py python_sim/antenna_diversity/array/__init__.py python_sim/tests/test_manifold.py
git commit -m "feat: add three-dimensional array manifold"
```

---

### Task 4: Multiple independently gated CW interferers in the scenario

**Files:**
- Modify: `python_sim/antenna_diversity/gnss/scenario.py`
- Create: `python_sim/tests/test_multi_interferer_scenario.py`
- Modify: `python_sim/tests/test_scenario.py`

**Interfaces:**
- Consumes: legacy ULA configuration or phone-array configuration.
- Produces truth keys `desired_steering_true`, `desired_steering_assumed`, `interferer_steering_nominal`, `interferer_steering_effective`, `interferer_components`, `active_antennas`, and `positions_m`.
- Preserves legacy aliases `desired_steering_nominal`, `desired_steering_effective`, `jammer_steering_nominal`, `jammer_steering_effective`, and `jammer_component` for the first legacy jammer.

- [ ] **Step 1: Write failing multi-interferer scenario tests**

```python
import numpy as np

from antenna_diversity.config import CWInterferer, Direction, mate70_default_config
from antenna_diversity.gnss.scenario import generate_scenario


def test_phone_scenario_generates_four_channels_and_two_interferer_components():
    cfg = mate70_default_config()
    cfg.interferers = (
        CWInterferer(Direction(120.0, 10.0), power_dbm=-90.0, offset_hz=250e3),
        CWInterferer(Direction(220.0, 20.0), power_dbm=-93.0, offset_hz=-400e3),
    )

    x, truth = generate_scenario(cfg)

    assert x.shape[0] == 4
    assert len(truth["interferer_components"]) == 2
    assert len(truth["interferer_steering_nominal"]) == 2
    np.testing.assert_allclose(
        x,
        truth["desired_component"] + sum(truth["interferer_components"]),
    )


def test_cw_start_and_stop_times_gate_only_that_interferer():
    cfg = mate70_default_config()
    cfg.signal.duration_ms = 10
    cfg.interferers = (
        CWInterferer(
            Direction(120.0, 10.0),
            power_dbm=-90.0,
            start_ms=2.0,
            stop_ms=6.0,
        ),
    )

    _, truth = generate_scenario(cfg)
    component = truth["interferer_components"][0]
    first = round(2e-3 * cfg.signal.fs_rf_hz)
    stop = round(6e-3 * cfg.signal.fs_rf_hz)

    assert np.all(component[:, :first] == 0)
    assert np.any(component[:, first:stop] != 0)
    assert np.all(component[:, stop:] == 0)


def test_true_and_assumed_target_manifolds_are_distinct_under_doa_error():
    cfg = mate70_default_config()
    cfg.signal.assumed_direction = Direction(35.0, 60.0)

    _, truth = generate_scenario(cfg)

    assert not np.allclose(
        truth["desired_steering_true"],
        truth["desired_steering_assumed"],
    )
```

- [ ] **Step 2: Verify tests fail at the missing phone scenario behavior**

Run: `cd F:\antenna_diversity\python_sim; python -m pytest tests/test_multi_interferer_scenario.py -v`

Expected: failures because `generate_scenario()` still expects legacy array fields and one jammer.

- [ ] **Step 3: Refactor scenario generation around resolved array responses**

Add private helpers with these contracts:

```python
def _desired_responses(cfg: SimulationConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return true nominal, assumed nominal, and true effective steering."""


def _active_interferers(cfg: SimulationConfig) -> tuple[CWInterferer, ...]:
    """Return phone interferers or a CWInterferer view of the legacy jammer."""


def _time_gate(
    n_samples: int,
    fs_hz: float,
    start_ms: float,
    stop_ms: float | None,
) -> NDArray[np.float64]:
    gate = np.zeros(n_samples, dtype=np.float64)
    first = round(start_ms * 1e-3 * fs_hz)
    stop = n_samples if stop_ms is None else round(stop_ms * 1e-3 * fs_hz)
    gate[first:stop] = 1.0
    return gate
```

For phone arrays, resolve true positions from `phone.phase_centers_m` and assumed positions from `phone.assumed_phase_centers_m`. Generate the desired and interferer scene responses with true positions, the isotropic pattern, and `PhoneArrayConfig.channel_response(estimated=False)`. Generate the EGC/MVDR constraint with assumed positions, `assumed_direction`, and `channel_response(estimated=True)`. For legacy arrays retain `steering_vector()` and the existing array-valued channel errors. Generate a list of individually gated jammer components, sum them after desired generation, and populate both expanded keys and legacy aliases.

- [ ] **Step 4: Run scenario and full regression tests**

Run: `cd F:\antenna_diversity\python_sim; python -m pytest tests/test_multi_interferer_scenario.py tests/test_scenario.py -v`

Then run: `python -m pytest -q`

Expected: all scenario tests pass and the legacy suite remains green.

- [ ] **Step 5: Commit scenario work**

```powershell
git add -- python_sim/antenna_diversity/gnss/scenario.py python_sim/tests/test_multi_interferer_scenario.py python_sim/tests/test_scenario.py
git commit -m "feat: support multiple spatial CW interferers"
```

---

### Task 5: Reusable combiner weights, assumed steering, and 2–4 channel pipeline

**Files:**
- Modify: `python_sim/antenna_diversity/diversity/combiner.py`
- Modify: `python_sim/antenna_diversity/pipeline.py`
- Modify: `python_sim/antenna_diversity/analysis/metrics.py`
- Modify: `python_sim/tests/test_diversity.py`
- Modify: `python_sim/tests/test_end_to_end.py`

**Interfaces:**
- Produces: `compute_weights(covariance, desired_steering, mode, cfg)` and preserves `combine(x, desired_steering, mode, cfg)`.
- Pipeline must pass `truth["desired_steering_assumed"]` to EGC/MVDR while metrics evaluate target gain against `truth["desired_steering_true"]`.
- The pipeline augments every branch's `combining` dictionary with `target_direction_gain`, `interferer_direction_gains`, `worst_interferer_gain`, `covariance_snapshots`, and `weights_are_fixed=True`; `compute_metrics()` copies these fields into its branch summary records.

- [ ] **Step 1: Write failing generalized-combiner and phone-pipeline tests**

```python
import numpy as np
import pytest

from antenna_diversity.config import mate70_default_config
from antenna_diversity.diversity.combiner import combine, compute_weights
from antenna_diversity.pipeline import run_end_to_end


@pytest.mark.parametrize("num_antennas", [2, 3, 4])
@pytest.mark.parametrize("mode", ["single", "egc", "mvdr"])
def test_combiner_supports_two_to_four_channels(num_antennas, mode):
    rng = np.random.default_rng(7)
    desired = np.exp(1j * np.arange(num_antennas) * 0.31)
    samples = rng.standard_normal((num_antennas, 4000)) + 1j * rng.standard_normal((num_antennas, 4000))

    output, info = combine(samples, desired, mode, mate70_default_config().diversity)

    assert output.shape == (4000,)
    assert np.vdot(info["weights"], desired) == pytest.approx(1.0 + 0j, abs=1e-10)


def test_phone_pipeline_uses_assumed_target_steering_for_weights():
    cfg = mate70_default_config()
    cfg.plot.enable = False
    cfg.signal.duration_ms = 8
    cfg.tracking.max_epochs = 2

    results = run_end_to_end(cfg)

    assert results.dfe_channels.shape[0] == 4
    assert set(results.branches) == {"single", "egc", "mvdr"}
    np.testing.assert_allclose(
        results.branches["mvdr"].combining["constraint_steering"],
        results.truth["desired_steering_assumed"],
    )
```

- [ ] **Step 2: Verify focused tests fail for missing `compute_weights` and truth use**

Run: `cd F:\antenna_diversity\python_sim; python -m pytest tests/test_diversity.py tests/test_end_to_end.py -v`

Expected: import or assertion failure for `compute_weights`/`constraint_steering`; existing legacy tests continue to identify any numerical regression.

- [ ] **Step 3: Extract weight calculation and update the pipeline**

Implement:

```python
def compute_weights(
    covariance: ArrayLike,
    desired_steering: ArrayLike,
    mode: str,
    cfg: DiversityConfig,
) -> tuple[NDArray[np.complex128], dict[str, Any]]:
    covariance_matrix = np.asarray(covariance, dtype=np.complex128)
    a = np.asarray(desired_steering, dtype=np.complex128).reshape(-1)
    n_antennas = a.size
    loading = cfg.diagonal_loading_factor * float(np.trace(covariance_matrix).real) / n_antennas
    loaded = covariance_matrix + loading * np.eye(n_antennas)
    if mode.lower() == "single":
        weights = np.zeros(n_antennas, dtype=np.complex128)
        weights[0] = 1.0 / np.conj(a[0])
    elif mode.lower() == "egc":
        weights = a / np.vdot(a, a)
    elif mode.lower() == "mvdr":
        solved = np.linalg.solve(loaded, a)
        weights = solved / np.vdot(a, solved)
    else:
        raise ValueError(f"unknown diversity mode: {mode}")
    return weights, {
        "constraint_steering": a,
        "desired_response": np.vdot(weights, a),
        "diagonal_loading": loading,
        "loaded_covariance": loaded,
    }
```

Retain the existing nonpositive-loading fallback and all input validation when extracting. `combine()` estimates sample covariance, calls `compute_weights()`, adds sample covariance, condition number, weight norm, and returns `weights.conj() @ channels`. After `combine()`, the pipeline evaluates `abs(wᴴa_true)` and every `abs(wᴴa_jammer)` and writes the named spatial fields into `combining_info`; set `covariance_snapshots=dfe_channels.shape[1]` and `weights_are_fixed=True`. Update `compute_metrics()` to copy these values without changing acquisition/tracking APIs.

- [ ] **Step 4: Run diversity, end-to-end, and full tests**

Run: `cd F:\antenna_diversity\python_sim; python -m pytest tests/test_diversity.py tests/test_end_to_end.py -v`

Then run: `python -m pytest -q`

Expected: 2–4 channel tests and legacy end-to-end tests pass.

- [ ] **Step 5: Commit combiner and pipeline work**

```powershell
git add -- python_sim/antenna_diversity/diversity/combiner.py python_sim/antenna_diversity/pipeline.py python_sim/antenna_diversity/analysis/metrics.py python_sim/tests/test_diversity.py python_sim/tests/test_end_to_end.py
git commit -m "feat: run diversity pipeline on phone subarrays"
```

---

### Task 6: Analytical spatial covariance, SINR, and response grids

**Files:**
- Create: `python_sim/antenna_diversity/analysis/spatial.py`
- Create: `python_sim/tests/test_spatial_analysis.py`
- Modify: `python_sim/antenna_diversity/analysis/__init__.py`
- Modify: `python_sim/antenna_diversity/analysis/metrics.py`

**Interfaces:**
- Produces: `analytic_covariance()`, `evaluate_spatial_case()`, and `response_grid_db()`.
- `evaluate_spatial_case()` returns JSON-compatible scalars/lists for target gain, per-interferer gains, output signal/interference/noise powers, output SINR, weight norm, white-noise gain, covariance rank/condition, loading, and snapshot metadata.
- Exact signature: `evaluate_spatial_case(weights, desired_steering, signal_power_w, interferer_steering, interferer_powers_w, noise_power_per_element_w, covariance=None, diagonal_loading=None, covariance_snapshots=None, weights_are_fixed=True) -> dict[str, Any]`.

- [ ] **Step 1: Write deterministic analytical tests**

```python
import numpy as np
import pytest

from antenna_diversity.analysis.spatial import (
    analytic_covariance,
    evaluate_spatial_case,
    response_grid_db,
)
from antenna_diversity.config import Direction, mate70_default_config


def test_analytic_covariance_sums_signal_interferers_and_white_noise():
    desired = np.array([1.0, 1.0j])
    jammer = np.array([1.0, -1.0])
    covariance = analytic_covariance(
        desired,
        signal_power_w=2.0,
        interferer_steering=(jammer,),
        interferer_powers_w=(5.0,),
        noise_power_per_element_w=0.25,
    )

    expected = 2.0 * np.outer(desired, desired.conj())
    expected += 5.0 * np.outer(jammer, jammer.conj())
    expected += 0.25 * np.eye(2)
    np.testing.assert_allclose(covariance, expected)


def test_spatial_metrics_report_exact_output_power_decomposition():
    weights = np.array([0.5, 0.5])
    desired = np.array([1.0, 1.0])
    jammer = np.array([1.0, -1.0])
    metrics = evaluate_spatial_case(
        weights,
        desired,
        signal_power_w=2.0,
        interferer_steering=(jammer,),
        interferer_powers_w=(5.0,),
        noise_power_per_element_w=0.25,
    )

    assert metrics["target_direction_gain"] == pytest.approx(1.0)
    assert metrics["interferer_direction_gains"][0] == pytest.approx(0.0)
    assert metrics["output_signal_power_w"] == pytest.approx(2.0)
    assert metrics["output_noise_power_w"] == pytest.approx(0.125)
    assert metrics["output_sinr_db"] == pytest.approx(10.0 * np.log10(16.0))


def test_response_grid_has_elevation_by_azimuth_shape_and_marks_unit_constraint():
    cfg = mate70_default_config()
    weights = np.ones(4, dtype=complex) / 4.0
    azimuths = np.array([0.0, 90.0, 180.0])
    elevations = np.array([0.0, 45.0, 90.0])

    response = response_grid_db(cfg, weights, azimuths, elevations)

    assert response.shape == (3, 3)
    assert np.all(np.isfinite(response))
```

- [ ] **Step 2: Verify tests fail because spatial analysis is absent**

Run: `cd F:\antenna_diversity\python_sim; python -m pytest tests/test_spatial_analysis.py -v`

Expected: import failure for `analysis.spatial`.

- [ ] **Step 3: Implement analytical covariance and metrics**

Use the exact power equations:

```python
target_response = np.vdot(weights, desired_steering)
jammer_responses = tuple(np.vdot(weights, steering) for steering in interferer_steering)
output_signal_power_w = signal_power_w * abs(target_response) ** 2
output_interference_power_w = sum(
    power * abs(response) ** 2
    for power, response in zip(interferer_powers_w, jammer_responses, strict=True)
)
white_noise_gain = float(np.vdot(weights, weights).real)
output_noise_power_w = noise_power_per_element_w * white_noise_gain
output_sinr_db = 10.0 * np.log10(
    output_signal_power_w / max(output_interference_power_w + output_noise_power_w, np.finfo(float).eps)
)
```

`response_grid_db()` must build each 3-D manifold with the configured active phone positions and isotropic pattern, calculate `abs(wᴴa)`, and normalize to the largest response in the returned grid with a floor of `-80 dB`.

When `covariance` is supplied, add `covariance_rank=int(np.linalg.matrix_rank(covariance))` and `covariance_condition_number=float(np.linalg.cond(covariance))`; otherwise return both as `None`. Convert NumPy scalar values and arrays to Python scalars/lists before returning so `json.dumps()` succeeds without a custom encoder.

- [ ] **Step 4: Run focused and metrics regression tests**

Run: `cd F:\antenna_diversity\python_sim; python -m pytest tests/test_spatial_analysis.py tests/test_end_to_end.py -v`

Expected: analytical identities pass and legacy receiver metrics still format.

- [ ] **Step 5: Commit spatial analysis**

```powershell
git add -- python_sim/antenna_diversity/analysis/spatial.py python_sim/antenna_diversity/analysis/__init__.py python_sim/antenna_diversity/analysis/metrics.py python_sim/tests/test_spatial_analysis.py
git commit -m "feat: add analytical spatial anti-jam metrics"
```

---

### Task 7: Fast combination and direction-scan experiments

**Files:**
- Create: `python_sim/antenna_diversity/experiments/__init__.py`
- Create: `python_sim/antenna_diversity/experiments/spatial_scan.py`
- Create: `python_sim/tests/test_experiments.py`
- Modify: `python_sim/antenna_diversity/models.py`

**Interfaces:**
- Produces: `evaluate_configuration_spatially(cfg, mode)`, `compare_corner_combinations(cfg, mode)`, `scan_interferer_directions(cfg, interferer_index, azimuths_deg, elevations_deg, mode)`, `scan_assumed_target_directions(cfg, azimuths_deg, elevations_deg, mode)`, `scan_interferer_powers(cfg, interferer_index, powers_dbm, mode)`, `scan_diagonal_loading(cfg, factors, mode)`, and `scan_position_errors(cfg, corner_name, xyz_errors_m, mode)`.
- These functions use analytical covariance and never call RF, DFE, acquisition, or tracking.
- Return values are dataclasses with `to_dict()` methods whose output is JSON serializable.

- [ ] **Step 1: Write failing experiment tests**

```python
import json
import numpy as np

from antenna_diversity.config import mate70_default_config
from antenna_diversity.experiments.spatial_scan import (
    compare_corner_combinations,
    scan_diagonal_loading,
    scan_interferer_directions,
    scan_interferer_powers,
    scan_position_errors,
)


def test_combination_comparison_returns_all_eleven_phone_subarrays():
    records = compare_corner_combinations(mate70_default_config(), mode="mvdr")

    assert len(records) == 11
    assert {len(record.active_antennas) for record in records} == {2, 3, 4}
    assert len({record.active_antennas for record in records}) == 11
    json.dumps([record.to_dict() for record in records])


def test_direction_scan_returns_requested_grid_without_running_receiver():
    result = scan_interferer_directions(
        mate70_default_config(),
        interferer_index=0,
        azimuths_deg=np.array([0.0, 120.0, 240.0]),
        elevations_deg=np.array([0.0, 30.0]),
        mode="mvdr",
    )

    assert result.output_sinr_db.shape == (2, 3)
    assert result.interferer_gain.shape == (2, 3)


def test_loading_and_power_scans_preserve_requested_parameter_order():
    cfg = mate70_default_config()

    loading = scan_diagonal_loading(cfg, factors=np.array([1e-4, 1e-2, 1e-1]), mode="mvdr")
    powers = scan_interferer_powers(cfg, interferer_index=0, powers_dbm=np.array([-100.0, -90.0]), mode="mvdr")

    np.testing.assert_array_equal(loading.parameter_values, [1e-4, 1e-2, 1e-1])
    np.testing.assert_array_equal(powers.parameter_values, [-100.0, -90.0])
    assert loading.output_sinr_db.shape == (3,)
    assert powers.output_sinr_db.shape == (2,)


def test_assumed_position_error_scan_does_not_move_the_true_phase_center():
    cfg = mate70_default_config()
    original_true = cfg.phone.phase_centers_m["TR"]
    errors = np.array([[0.0, 0.0, 0.0], [0.002, 0.0, 0.0]])

    result = scan_position_errors(cfg, corner_name="TR", xyz_errors_m=errors, mode="mvdr")

    assert result.output_sinr_db.shape == (2,)
    assert cfg.phone.phase_centers_m["TR"] == original_true
```

- [ ] **Step 2: Verify experiment tests fail at missing package/functions**

Run: `cd F:\antenna_diversity\python_sim; python -m pytest tests/test_experiments.py -v`

Expected: import failure for `antenna_diversity.experiments`.

- [ ] **Step 3: Implement analytical experiment orchestration**

For each configuration:

1. Deep-copy the config.
2. Select positions and compute true/assumed desired manifolds and each jammer manifold.
3. Convert dBm powers to watts.
4. Compute the exact input-referred per-element noise power used by the RF model: `k * cfg.rf.temperature_k * cfg.rf.noise_bandwidth_hz * cascade_metrics(cfg.rf.stages)["noise_factor"]`.
5. Build analytical covariance.
6. Call `compute_weights()`.
7. Call `evaluate_spatial_case()`.

Define result types:

```python
@dataclass(frozen=True)
class SpatialCombinationResult:
    active_antennas: tuple[str, ...]
    positions_m: tuple[tuple[float, float, float], ...]
    maximum_baseline_m: float
    mode: str
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_antennas": list(self.active_antennas),
            "positions_m": [list(position) for position in self.positions_m],
            "maximum_baseline_m": self.maximum_baseline_m,
            "mode": self.mode,
            "metrics": self.metrics,
        }
```

`evaluate_spatial_case()` from Task 6 guarantees that `metrics` is JSON compatible; do not store NumPy arrays in `SpatialCombinationResult.metrics`. Do not mutate the caller's config. Interferer-direction scanning replaces only the selected interferer's direction in each copied case. Assumed-target scanning leaves `true_direction` fixed and replaces only `assumed_direction`, so it measures DOA mismatch. Position-error scanning adds each `(dx,dy,dz)` only to `phone.assumed_phase_centers_m[corner_name]` while leaving the real coordinate fixed. Power and diagonal-loading scans preserve caller order and return one-dimensional arrays for every scalar metric.

- [ ] **Step 4: Run experiment and full tests**

Run: `cd F:\antenna_diversity\python_sim; python -m pytest tests/test_experiments.py tests/test_spatial_analysis.py -v`

Then run: `python -m pytest -q`

Expected: 11 combinations, requested scan shapes, JSON serialization, and all regressions pass.

- [ ] **Step 5: Commit experiment orchestration**

```powershell
git add -- python_sim/antenna_diversity/experiments python_sim/antenna_diversity/models.py python_sim/tests/test_experiments.py
git commit -m "feat: add fast phone-array spatial experiments"
```

---

### Task 8: Two-dimensional array-response and multi-interferer plotting

**Files:**
- Modify: `python_sim/antenna_diversity/analysis/plotting.py`
- Modify: `python_sim/tests/test_plotting.py`

**Interfaces:**
- Preserve the five output keys from `plot_results()`.
- For legacy ULA, `array_response` remains the current 1-D line plot.
- For phone arrays, `array_response` becomes three azimuth/elevation heat maps, one for SINGLE/EGC/MVDR, sharing color limits and marking the true target plus all enabled interferers.
- RF/DFE and combined-spectrum plots mark every enabled interferer frequency.

- [ ] **Step 1: Write failing phone-array plotting tests**

```python
import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

from antenna_diversity.analysis.plotting import plot_array_response
from antenna_diversity.config import mate70_default_config
from antenna_diversity.pipeline import run_end_to_end


def test_phone_array_response_contains_three_heatmaps_and_direction_markers(tmp_path):
    cfg = mate70_default_config()
    cfg.plot.enable = False
    cfg.signal.duration_ms = 8
    cfg.tracking.max_epochs = 2
    results = run_end_to_end(cfg)

    figure = plot_array_response(results)
    image_axes = [axis for axis in figure.axes if axis.images]

    assert len(image_axes) == 3
    assert all(axis.get_xlabel() == "Azimuth (deg)" for axis in image_axes)
    assert all(axis.get_ylabel() == "Elevation (deg)" for axis in image_axes)
    assert all(len(axis.lines) >= 2 for axis in image_axes)
    plt.close(figure)
```

Extend the combined-spectrum test with two interferers and assert each main axis contains one data line, one target marker, and two jammer markers.

- [ ] **Step 2: Verify plotting tests fail on the legacy-only plotting paths**

Run: `cd F:\antenna_diversity\python_sim; python -m pytest tests/test_plotting.py -v`

Expected: phone response fails because the current function accesses 1-D `desired_angle_deg`; multiple-jammer marker counts fail.

- [ ] **Step 3: Implement phone heat maps and marker helpers**

Add:

```python
def _enabled_interferers(results: SimulationResults):
    return tuple(interferer for interferer in results.config.interferers if interferer.enable)


def _plot_phone_array_response(results: SimulationResults) -> Figure:
    azimuths = np.arange(0.0, 360.0 + 2.0, 2.0)
    elevations = np.arange(0.0, 90.0 + 1.0, 1.0)
    grids = {
        mode: response_grid_db(results.config, branch.combining["weights"], azimuths, elevations)
        for mode, branch in results.branches.items()
    }
```

Create one subplot per branch with `imshow(origin="lower", aspect="auto", extent=(0,360,0,90), vmin=-60, vmax=0)`. Mark the target with a green star and each jammer with a red `x`; add a `-3 dB` contour to expose high-response ambiguity regions; label axes, add one shared colorbar, and state `Normalized response (dB)`. Dispatch between phone and legacy plotting based on the array configuration type. Replace every singular `cfg.jammer` marker path with iteration over resolved active interferers.

- [ ] **Step 4: Run plotting tests with warnings promoted to errors**

Run: `cd F:\antenna_diversity\python_sim; python -W error -m pytest tests/test_plotting.py -v`

Expected: all plotting tests pass with no Matplotlib warnings.

- [ ] **Step 5: Commit plotting work**

```powershell
git add -- python_sim/antenna_diversity/analysis/plotting.py python_sim/tests/test_plotting.py
git commit -m "feat: plot phone-array spatial response"
```

---

### Task 9: New default, explicit legacy entry point, and reproducible end-to-end baselines

**Files:**
- Modify: `python_sim/antenna_diversity/config.py`
- Modify: `python_sim/run_end_to_end.py`
- Create: `python_sim/run_legacy_ula.py`
- Modify: `python_sim/tests/test_config_numeric.py`
- Modify: `python_sim/tests/test_core.py`
- Modify: `python_sim/tests/test_scenario.py`
- Modify: `python_sim/tests/test_diversity.py`
- Modify: `python_sim/tests/test_end_to_end.py`
- Modify: `python_sim/tests/test_plotting.py`
- Modify: `python_sim/pyproject.toml`

**Interfaces:**
- `default_config()` becomes an alias for `mate70_default_config()`.
- Tests whose expected values describe the original ULA must explicitly call `legacy_ula_config()`.
- `run_legacy_ula.py` runs `run_pipeline(legacy_ula_config())`, prints metrics, and saves plots to `results/legacy_ula`.

- [ ] **Step 1: Write failing default/legacy acceptance tests**

```python
from antenna_diversity.config import default_config, legacy_ula_config
from antenna_diversity.pipeline import run_end_to_end


def test_default_config_is_mate70_four_corner_array():
    cfg = default_config()

    assert cfg.array.active_antennas == ("TL", "TR", "BL", "BR")
    assert cfg.array.num_elements == 4


def test_legacy_pipeline_preserves_original_diversity_conclusion():
    cfg = legacy_ula_config()
    cfg.plot.enable = False
    results = run_end_to_end(cfg)

    assert not results.branches["single"].acquisition["success"]
    assert not results.branches["egc"].acquisition["success"]
    assert results.branches["mvdr"].acquisition["success"]
    assert results.branches["mvdr"].tracking["success"]


def test_mate70_baseline_mvdr_improves_spatial_and_receiver_metrics():
    cfg = default_config()
    cfg.plot.enable = False
    results = run_end_to_end(cfg)

    single = results.branches["single"]
    mvdr = results.branches["mvdr"]
    assert mvdr.combining["interferer_direction_gains"][0] < 0.1 * single.combining["interferer_direction_gains"][0]
    assert mvdr.acquisition["metric"] > single.acquisition["metric"]
    assert mvdr.acquisition["success"]
    assert mvdr.tracking["success"]
```

- [ ] **Step 2: Verify the default-config assertion fails before the switch**

Run: `cd F:\antenna_diversity\python_sim; python -m pytest tests/test_config_numeric.py::test_default_config_is_mate70_four_corner_array -v`

Expected: FAIL because `default_config()` still returns the legacy ULA.

- [ ] **Step 3: Switch the default and migrate old-assumption tests**

Change:

```python
def default_config() -> SimulationConfig:
    """Return the baseline Mate 70 four-corner GPS L1 C/A configuration."""
    return mate70_default_config()
```

In every existing test that asserts old ULA angles, spacing, single-jammer aliases, or the original numerical conclusion, import and call `legacy_ula_config()`. Leave generic RF/DFE/acquisition/tracking tests on `default_config()` only when their setup does not access phone/legacy-specific fields. Implement `run_legacy_ula.py` by copying the CLI structure from `run_end_to_end.py`, passing the explicit config, and changing only the output directory.

If the Mate 70 acceptance case does not satisfy the specified physical result with the already approved default geometry and directions, do not weaken the test or change algorithms. Adjust only the default CW power within the documented strong-interference scenario, record the chosen value in `mate70_default_config()`, and retain `-90 dBm` as the initial candidate. The acceptance condition remains MVDR acquisition and tracking success plus at least tenfold interferer-direction gain reduction relative to SINGLE.

- [ ] **Step 4: Run both CLIs and the complete test suite**

Run:

```powershell
cd F:\antenna_diversity\python_sim
$env:MPLBACKEND='Agg'
python -W error run_end_to_end.py
python -W error run_legacy_ula.py
python -W error -m pytest -q
```

Expected: both CLIs exit 0 and save their plots; every test passes without warnings.

- [ ] **Step 5: Commit default and compatibility work**

```powershell
git add -- python_sim/antenna_diversity/config.py python_sim/run_end_to_end.py python_sim/run_legacy_ula.py python_sim/pyproject.toml python_sim/tests/test_config_numeric.py python_sim/tests/test_core.py python_sim/tests/test_scenario.py python_sim/tests/test_diversity.py python_sim/tests/test_end_to_end.py python_sim/tests/test_plotting.py
git commit -m "feat: make Mate 70 four-corner array the default"
```

---

### Task 10: Monte Carlo orchestration, documentation, architecture artifact, and final verification

**Files:**
- Create: `python_sim/antenna_diversity/experiments/monte_carlo.py`
- Create: `python_sim/tests/test_monte_carlo.py`
- Modify: `python_sim/antenna_diversity/experiments/__init__.py`
- Modify: `python_sim/README.md`
- Modify: `python_sim/ARCHITECTURE.md`
- Modify: `python_sim/DESIGN.md`
- Modify: `python_sim/docs/simulation_architecture.svg`
- Regenerate: `python_sim/docs/simulation_architecture.png`

**Interfaces:**
- Produces: `run_monte_carlo(cfg, seeds) -> MonteCarloResult`.
- `MonteCarloResult.to_dict()` is JSON serializable and contains per-seed branch records plus capture/tracking rates and spatial metric distributions.
- Runs are sequential in this phase to keep random-state handling and result ordering deterministic; no parallel-execution parameter is exposed.

- [ ] **Step 1: Write failing Monte Carlo tests**

```python
import json

from antenna_diversity.config import default_config
from antenna_diversity.experiments.monte_carlo import run_monte_carlo


def test_monte_carlo_runs_each_seed_and_summarizes_branch_rates():
    cfg = default_config()
    cfg.plot.enable = False
    cfg.signal.duration_ms = 8
    cfg.tracking.max_epochs = 2

    result = run_monte_carlo(cfg, seeds=(101, 102))

    assert [run.seed for run in result.runs] == [101, 102]
    assert set(result.capture_rate) == {"single", "egc", "mvdr"}
    assert set(result.tracking_rate) == {"single", "egc", "mvdr"}
    assert all(0.0 <= value <= 1.0 for value in result.capture_rate.values())
    json.dumps(result.to_dict())


def test_monte_carlo_does_not_mutate_the_input_config():
    cfg = default_config()
    cfg.plot.enable = False
    original_seed = cfg.random_seed

    run_monte_carlo(cfg, seeds=(11,))

    assert cfg.random_seed == original_seed
```

- [ ] **Step 2: Verify Monte Carlo tests fail for the missing module**

Run: `cd F:\antenna_diversity\python_sim; python -m pytest tests/test_monte_carlo.py -v`

Expected: import failure for `experiments.monte_carlo`.

- [ ] **Step 3: Implement sequential repeatable orchestration**

```python
def run_monte_carlo(
    cfg: SimulationConfig,
    seeds: Sequence[int],
) -> MonteCarloResult:
    normalized_seeds = tuple(int(seed) for seed in seeds)
    if not normalized_seeds:
        raise ValueError("seeds must be nonempty")
    runs = []
    for seed in normalized_seeds:
        run_cfg = copy.deepcopy(cfg)
        run_cfg.random_seed = seed
        run_cfg.plot.enable = False
        results = run_end_to_end(run_cfg)
        runs.append(MonteCarloRun.from_results(seed, results))
    return MonteCarloResult.from_runs(tuple(runs))
```

Store only JSON-compatible metric records, not raw sample arrays. Compute capture/tracking rates as Boolean means per branch, and include lists for acquisition metric, final Doppler error, output SINR, and worst interferer gain when available.

- [ ] **Step 4: Update documentation and architecture graphic**

Update the README with:

- Mate 70 dimensions and ideal-corner disclaimer;
- coordinate/angle convention;
- arbitrary active-corner examples;
- zero-to-three `CWInterferer` examples;
- `default_config()` versus `legacy_ula_config()`;
- fast combination/direction scans and Monte Carlo examples;
- explicit current omissions: real pattern, real antenna locations, mutual coupling, polarization, multipath, dynamic DOA, non-CW interference, multiple satellites, and PVT.

Update `ARCHITECTURE.md` and `DESIGN.md` to match the approved spec. Change `simulation_architecture.svg` so its central path shows `2–4 selected corners`, `3-D manifold`, and `0–3 CW interferers`; its modeled/unmodeled panels must distinguish isotropic pattern interface from real measured antenna patterns. Regenerate the PNG from the SVG and visually inspect it at full resolution.

- [ ] **Step 5: Run final evidence checks**

Run:

```powershell
cd F:\antenna_diversity\python_sim
$env:MPLBACKEND='Agg'
python -W error -m pytest -q
python -W error run_end_to_end.py
python -W error run_legacy_ula.py
cd F:\antenna_diversity
git diff --check
```

Also parse both SVG files as UTF-8 XML, check every local Markdown link in `python_sim/README.md`, `python_sim/ARCHITECTURE.md`, and `python_sim/DESIGN.md`, and verify `simulation_architecture.png` is at least 1920×1200. Inspect `git status --short` to ensure generated `results/` files remain ignored and no temporary browser profile is present.

Expected: complete pytest suite passes with no warnings; both CLIs exit 0; XML/link/image checks pass; `git diff --check` emits no errors.

- [ ] **Step 6: Commit Monte Carlo and documentation**

```powershell
git add -- python_sim/antenna_diversity/experiments/monte_carlo.py python_sim/antenna_diversity/experiments/__init__.py python_sim/tests/test_monte_carlo.py python_sim/README.md python_sim/ARCHITECTURE.md python_sim/DESIGN.md python_sim/docs/simulation_architecture.svg python_sim/docs/simulation_architecture.png
git commit -m "feat: add Monte Carlo and phone-array documentation"
```

---

## Completion Criteria

The implementation is complete only when all of the following are demonstrated by fresh command output:

- `default_config()` runs a four-corner Mate 70 ideal phase-center model.
- Any unique 2–4 corner subset validates and runs through the common channels-by-samples chain.
- All 11 subsets are enumerated and evaluated without mutating the source config.
- Directions use the approved phone azimuth/elevation convention and a 3-D manifold.
- Zenith produces identical phase for coplanar ideal phase centers.
- The isotropic element-pattern interface is exercised by tests.
- Zero to three CW interferers generate separate truth components and independent time gates.
- SINGLE, EGC, and MVDR accept 2, 3, and 4 channels and maintain the target constraint.
- Fast spatial metrics report power decomposition, output SINR, covariance diagnostics, and per-interferer gains.
- Phone-array response is plotted over azimuth/elevation with target and all jammer markers.
- The new default demonstrates repeatable MVDR spatial suppression plus acquisition/tracking success.
- `legacy_ula_config()` and `run_legacy_ula.py` reproduce the original dual-antenna conclusion.
- Monte Carlo runs are repeatable, JSON serializable, and do not retain raw sample arrays.
- Documentation clearly states that real Mate 70 antenna locations/patterns are unknown and unmodeled.
- The full warning-as-error test suite and both CLI runs pass.
