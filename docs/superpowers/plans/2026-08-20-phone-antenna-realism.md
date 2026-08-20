# Phone Antenna Realism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a backward-compatible, polarized, oriented, blocked, mutually coupled phone-antenna model that is shared by the analytical and sampled Python simulation paths and can later ingest measured pattern data.

**Architecture:** Keep the legacy scalar `array_manifold()` API unchanged and add a layered `PhoneAntennaModel` for phone configurations. The model transforms the incident Jones field into each element frame, evaluates a polarized pattern, applies directional attenuation, mutual coupling, and channel calibration, and exposes separate true and assumed responses. All studies and reports consume saved diagnostics from this same model.

**Tech Stack:** Python 3.12, NumPy, standard-library `csv`/`hashlib`/`pathlib`, Matplotlib, pytest. No new runtime dependency.

**Spec:** `docs/superpowers/specs/2026-08-20-phone-antenna-realism-design.md`

## Global Constraints

- Development is Python-only; do not modify the MATLAB implementation.
- Preserve `default_config()` and `legacy_ula_config()` numerical behavior.
- Preserve the existing public scalar `ElementPattern.complex_gain()` and `array_manifold()` APIs.
- Keep external source elevation in `[0°, 90°]`; local element elevation may be in `[-90°, 90°]`.
- Use the existing spatial phase convention `exp(-j k rᵀu)`.
- `signal.power_dbm` and jammer power remain referenced to a unit-gain, unit-polarization-match antenna.
- Apply effects in the fixed order: pattern/position/polarization, blockage, coupling, channel response.
- Real samples and covariance use the true state; the MVDR desired constraint uses the assumed state.
- The measured-pattern path must record and verify SHA-256 and must not frequency-interpolate.
- All configuration failures exposed to users must be stable `ValueError` instances.
- Existing JSON without new antenna fields resolves to the ideal phone model.
- Reports must identify results as `ideal`, `parametric`, or `measured`; parameterized data must never be described as measured Mate 70 performance.
- Every task follows RED, GREEN, focused regression, self-review, and a task-local commit.

## File Map

- `python_sim/antenna_diversity/array/polarization.py`: Jones presets, spherical bases, yaw/pitch/roll matrices, and phone-to-element wave transformation.
- `python_sim/antenna_diversity/array/patterns.py`: polarized pattern protocol, scalar adapter, parametric pattern, tabulated CSV loading/interpolation and digest verification.
- `python_sim/antenna_diversity/array/effects.py`: element orientation records, angular attenuation regions, coupling paths, attenuation vectors and ordered coupling matrices.
- `python_sim/antenna_diversity/array/antenna_config.py`: serializable pattern/state/phone-antenna configuration records and validation.
- `python_sim/antenna_diversity/array/phone_model.py`: the only phone effective-manifold composition service.
- `python_sim/antenna_diversity/config.py`: attach polarizations and structured antenna configuration to existing simulation records and presets.
- `python_sim/antenna_diversity/gnss/scenario.py`: generate desired/jammer samples and truth from `PhoneAntennaModel`.
- `python_sim/antenna_diversity/analysis/spatial.py`: true/assumed response grids through the shared model.
- `python_sim/antenna_diversity/experiments/spatial_scan.py`: analytical covariance and weights through the shared model.
- `python_sim/antenna_diversity/studies/config.py`: strict nested JSON parsing and measured-resource resolution.
- `python_sim/antenna_diversity/studies/models.py`: immutable snapshots and scientific hash coverage for new records.
- `python_sim/antenna_diversity/studies/analytical.py`: save antenna diagnostics during baseline, not during reporting.
- `python_sim/antenna_diversity/studies/runner.py`: publish one additional diagnostics table and four additional fixed figures.
- `python_sim/antenna_diversity/studies/plots.py`: render antenna evidence only from saved rows.
- `python_sim/antenna_diversity/studies/report.py`: summarize and label the antenna model and evidence.
- `python_sim/README.md`, `python_sim/ARCHITECTURE.md`, `python_sim/DESIGN.md`: scientific scope and usage.
- `python_sim/docs/simulation_architecture.svg`, `python_sim/docs/simulation_architecture.png`: editable and rendered architecture diagram.

---

### Task 1: Polarization and coordinate foundations

**Files:**
- Create: `python_sim/antenna_diversity/array/polarization.py`
- Modify: `python_sim/antenna_diversity/array/__init__.py`
- Test: `python_sim/tests/test_polarization.py`

**Interfaces:**
- Produces: `WavePolarization.jones_vector() -> NDArray[np.complex128]`
- Produces: `rotation_matrix_ypr(yaw_deg, pitch_deg, roll_deg) -> NDArray[np.float64]`
- Produces: `wave_in_element_frame(azimuth_deg, elevation_deg, polarization, local_to_phone) -> LocalWave`
- Produces: immutable `LocalWave(azimuth_deg, elevation_deg, jones)` with local elevation in `[-90, 90]`

- [ ] **Step 1: Write failing convention and validation tests**

```python
def test_rhcp_and_lhcp_are_unit_and_orthogonal():
    rhcp = WavePolarization("rhcp").jones_vector()
    lhcp = WavePolarization("lhcp").jones_vector()
    np.testing.assert_allclose(np.vdot(rhcp, rhcp), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.vdot(lhcp, lhcp), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.vdot(rhcp, lhcp), 0.0, atol=1e-12)


def test_linear_tilt_and_explicit_jones_are_normalized():
    linear = WavePolarization("linear", tilt_deg=90.0).jones_vector()
    explicit = WavePolarization(
        "jones", az_real=2.0, az_imag=0.0, el_real=0.0, el_imag=0.0
    ).jones_vector()
    np.testing.assert_allclose(abs(linear[1]), 1.0, atol=1e-12)
    np.testing.assert_allclose(explicit, [1.0, 0.0], atol=1e-12)


@pytest.mark.parametrize("value", [np.nan, np.inf, 1 + 1j, [1.0]])
def test_polarization_rejects_nonfinite_or_nonscalar_fields(value):
    with pytest.raises(ValueError):
        WavePolarization("linear", tilt_deg=value).jones_vector()
```

- [ ] **Step 2: Run the tests and record RED**

Run: `cd python_sim && python -W error -m pytest tests/test_polarization.py -v`

Expected: collection fails with `ModuleNotFoundError: antenna_diversity.array.polarization`.

- [ ] **Step 3: Implement the polarization records and basis transform**

```python
@dataclass(frozen=True)
class WavePolarization:
    kind: str = "rhcp"
    tilt_deg: float = 0.0
    az_real: float = 1.0
    az_imag: float = 0.0
    el_real: float = 0.0
    el_imag: float = 0.0

    def jones_vector(self) -> NDArray[np.complex128]:
        values = (self.tilt_deg, self.az_real, self.az_imag, self.el_real, self.el_imag)
        if any(not _finite_real_scalar(value) for value in values):
            raise ValueError("polarization fields must be finite real scalars")
        if self.kind == "rhcp":
            vector = np.array([1.0, -1.0j], dtype=np.complex128)
        elif self.kind == "lhcp":
            vector = np.array([1.0, 1.0j], dtype=np.complex128)
        elif self.kind == "linear":
            tilt = np.deg2rad(float(self.tilt_deg))
            vector = np.array([np.cos(tilt), np.sin(tilt)], dtype=np.complex128)
        elif self.kind == "jones":
            vector = np.array(
                [complex(self.az_real, self.az_imag), complex(self.el_real, self.el_imag)],
                dtype=np.complex128,
            )
        else:
            raise ValueError("polarization kind must be rhcp, lhcp, linear, or jones")
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise ValueError("polarization Jones vector must be nonzero")
        result = vector / norm
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class LocalWave:
    azimuth_deg: float
    elevation_deg: float
    jones: NDArray[np.complex128]


def rotation_matrix_ypr(
    yaw_deg: float, pitch_deg: float, roll_deg: float
) -> NDArray[np.float64]:
    angles = np.asarray([yaw_deg, pitch_deg, roll_deg])
    if angles.shape != (3,) or np.iscomplexobj(angles) or not np.all(np.isfinite(angles)):
        raise ValueError("yaw, pitch, and roll must be finite real scalars")
    yaw, pitch, roll = np.deg2rad(angles.astype(float))
    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cx, sx = np.cos(roll), np.sin(roll)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return rz @ ry @ rx


def wave_in_element_frame(
    azimuth_deg: float,
    elevation_deg: float,
    polarization: WavePolarization,
    local_to_phone: ArrayLike,
) -> LocalWave:
    rotation = _validated_rotation(local_to_phone)
    unit_phone, az_basis_phone, el_basis_phone = _spherical_frame(
        azimuth_deg, elevation_deg
    )
    field_phone = (
        polarization.jones_vector()[0] * az_basis_phone
        + polarization.jones_vector()[1] * el_basis_phone
    )
    unit_local = rotation.T @ unit_phone
    field_local = rotation.T @ field_phone
    local_azimuth = float(np.rad2deg(np.arctan2(unit_local[1], unit_local[0])) % 360.0)
    local_elevation = float(np.rad2deg(np.arcsin(np.clip(unit_local[2], -1.0, 1.0))))
    _, az_basis_local, el_basis_local = _spherical_frame(local_azimuth, local_elevation)
    local_jones = np.array(
        [np.vdot(az_basis_local, field_local), np.vdot(el_basis_local, field_local)],
        dtype=np.complex128,
    )
    local_jones /= np.linalg.norm(local_jones)
    return LocalWave(local_azimuth, local_elevation, local_jones)
```

Implement `_finite_real_scalar`, `_spherical_frame` and `_validated_rotation` in the same file. `_spherical_frame` returns the unit, increasing-azimuth and increasing-elevation unit vectors. `_validated_rotation` requires shape `(3,3)`, finite real values, `R.T @ R == I` within `1e-12`, and determinant `+1` within `1e-12`. `LocalWave.__post_init__` copies its two-component Jones vector, verifies unit norm, and marks it read-only. Canonicalize azimuth to `[0, 360)` and clip only floating-point roundoff at the elevation poles.

- [ ] **Step 4: Add identity, yaw, pitch and field-preservation tests**

```python
def test_identity_frame_preserves_direction_and_jones():
    wave = wave_in_element_frame(120.0, 30.0, WavePolarization("rhcp"), np.eye(3))
    assert wave.azimuth_deg == pytest.approx(120.0)
    assert wave.elevation_deg == pytest.approx(30.0)
    np.testing.assert_allclose(wave.jones, WavePolarization("rhcp").jones_vector(), atol=1e-12)


def test_positive_yaw_moves_source_to_negative_local_azimuth():
    rotation = rotation_matrix_ypr(90.0, 0.0, 0.0)
    wave = wave_in_element_frame(120.0, 20.0, WavePolarization("linear"), rotation)
    assert wave.azimuth_deg == pytest.approx(30.0)
    assert wave.elevation_deg == pytest.approx(20.0)
```

- [ ] **Step 5: Run focused and existing manifold tests**

Run: `cd python_sim && python -W error -m pytest tests/test_polarization.py tests/test_manifold.py -v`

Expected: all tests pass without warnings.

- [ ] **Step 6: Commit Task 1**

```bash
git add -- python_sim/antenna_diversity/array/polarization.py python_sim/antenna_diversity/array/__init__.py python_sim/tests/test_polarization.py
git commit -m "feat: add polarized antenna coordinate foundations"
```

---

### Task 2: Polarized parametric and measured element patterns

**Files:**
- Create: `python_sim/antenna_diversity/array/patterns.py`
- Modify: `python_sim/antenna_diversity/array/__init__.py`
- Test: `python_sim/tests/test_antenna_patterns.py`
- Test data: `python_sim/tests/data/pattern_grid.csv`

**Interfaces:**
- Consumes: `LocalWave` from Task 1
- Produces: protocol `PolarizedElementPattern.effective_height(element_name, direction, frequency_hz) -> complex128[2]`
- Produces: `ParametricPatternParameters`
- Produces: `ParametricElementPattern`
- Produces: `TabulatedElementPattern.from_csv(path, frequency_hz, expected_sha256=None)`
- Produces: `ScalarPatternAdapter`

- [ ] **Step 1: Write failing parametric-pattern tests**

```python
def test_parametric_pattern_has_configured_peak_and_back_floor():
    parameters = {
        "TL": ParametricPatternParameters(
            peak_gain_db=3.0,
            cosine_exponent=2.0,
            back_lobe_gain_db=-20.0,
            nominal_polarization=WavePolarization("rhcp"),
            cross_polar_discrimination_db=30.0,
            cross_phase_deg=0.0,
        )
    }
    pattern = ParametricElementPattern(parameters)
    front = pattern.effective_height("TL", LocalDirection(0.0, 90.0), 1_575.42e6)
    back = pattern.effective_height("TL", LocalDirection(0.0, -90.0), 1_575.42e6)
    assert np.linalg.norm(front) == pytest.approx(10 ** (3.0 / 20.0))
    assert np.linalg.norm(back) == pytest.approx(10 ** (-20.0 / 20.0))
```

- [ ] **Step 2: Run the parametric tests and record RED**

Run: `cd python_sim && python -W error -m pytest tests/test_antenna_patterns.py -k parametric -v`

Expected: import failure for `antenna_diversity.array.patterns`.

- [ ] **Step 3: Implement the pattern protocol, local direction and parametric model**

```python
class PolarizedElementPattern(Protocol):
    def effective_height(
        self, element_name: str, direction: LocalDirection, frequency_hz: float
    ) -> NDArray[np.complex128]: ...


@dataclass(frozen=True)
class LocalDirection:
    azimuth_deg: float
    elevation_deg: float


@dataclass(frozen=True)
class ParametricPatternParameters:
    peak_gain_db: float = 0.0
    cosine_exponent: float = 1.0
    back_lobe_gain_db: float = -20.0
    nominal_polarization: WavePolarization = WavePolarization("rhcp")
    cross_polar_discrimination_db: float = 30.0
    cross_phase_deg: float = 0.0
```

Use voltage gain `10 ** (gain_db / 20)`. Define front power shape as `max(cos(90°-elevation), 0) ** cosine_exponent`, convert it to dB relative to the peak, and floor it at `back_lobe_gain_db`. Build the effective-height vector from the nominal Jones vector plus its orthogonal vector scaled by XPD and phase, then normalize that polarization mixture before applying voltage gain.

- [ ] **Step 4: Write failing measured-grid tests**

```python
def test_tabulated_pattern_interpolates_complex_components_and_wraps_azimuth(tmp_path):
    path = write_rectangular_pattern_csv(tmp_path, azimuths=(0.0, 180.0), elevations=(-90.0, 90.0))
    pattern = TabulatedElementPattern.from_csv(path, 1_575.42e6)
    at_90 = pattern.effective_height("TL", LocalDirection(90.0, 0.0), 1_575.42e6)
    at_270 = pattern.effective_height("TL", LocalDirection(270.0, 0.0), 1_575.42e6)
    np.testing.assert_allclose(at_90, expected_interpolated_90, atol=1e-12)
    np.testing.assert_allclose(at_270, expected_interpolated_270, atol=1e-12)


def test_tabulated_pattern_verifies_sha256(tmp_path):
    path = write_rectangular_pattern_csv(tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    TabulatedElementPattern.from_csv(path, 1_575.42e6, digest)
    with pytest.raises(ValueError, match="SHA-256"):
        TabulatedElementPattern.from_csv(path, 1_575.42e6, "0" * 64)
```

- [ ] **Step 5: Implement strict CSV loading and bilinear complex interpolation**

```python
@classmethod
def from_csv(
    cls,
    path: Path,
    frequency_hz: float,
    expected_sha256: str | None = None,
) -> "TabulatedElementPattern":
    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read measured pattern CSV: {source}") from error
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest.lower() != expected_sha256.lower():
        raise ValueError("measured pattern CSV SHA-256 does not match configuration")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("measured pattern CSV must be UTF-8") from error
    reader = csv.DictReader(io.StringIO(text))
    required = {
        "element_name", "frequency_hz", "azimuth_deg", "elevation_deg",
        "h_az_real", "h_az_imag", "h_el_real", "h_el_imag",
    }
    if reader.fieldnames is None or set(reader.fieldnames) != required:
        raise ValueError("measured pattern CSV columns do not match the required schema")
    rows = list(reader)
    if not rows:
        raise ValueError("measured pattern CSV must contain data rows")
    parsed = _parse_pattern_rows(rows, float(frequency_hz))
    return cls(parsed, float(frequency_hz), digest)
```

Implement `_parse_pattern_rows` to convert every numeric column with `float`, reject bool/complex/nonfinite values, normalize azimuth modulo 360 before duplicate detection, group by element, verify one matching frequency and require the Cartesian product of unique azimuth/elevation coordinates. Store sorted axes and `(elevation, azimuth, component)` arrays. Interpolate real and imaginary components independently. Use periodic azimuth neighbors across the 0°/360° seam. Reject elevation outside the file coverage and reject table frequency when `abs(file_f - requested_f) > max(1.0, 1e-9 * requested_f)`.

- [ ] **Step 6: Add malformed table tests**

Cover duplicate `0°/360°`, missing grid cell, unknown/missing column, nonfinite component, missing element, out-of-range elevation, frequency mismatch, bad digest and empty file. Each case must raise `ValueError`, not `KeyError`, `TypeError` or a NumPy exception.

- [ ] **Step 7: Run focused tests and commit Task 2**

Run: `cd python_sim && python -W error -m pytest tests/test_antenna_patterns.py tests/test_polarization.py tests/test_manifold.py -v`

```bash
git add -- python_sim/antenna_diversity/array/patterns.py python_sim/antenna_diversity/array/__init__.py python_sim/tests/test_antenna_patterns.py python_sim/tests/data/pattern_grid.csv
git commit -m "feat: add polarized antenna pattern models"
```

---

### Task 3: Structured antenna configuration, blockage and coupling

**Files:**
- Create: `python_sim/antenna_diversity/array/effects.py`
- Create: `python_sim/antenna_diversity/array/antenna_config.py`
- Modify: `python_sim/antenna_diversity/config.py`
- Test: `python_sim/tests/test_antenna_effects.py`
- Test: `python_sim/tests/test_phone_config.py`

**Interfaces:**
- Consumes: `WavePolarization`, `ParametricPatternParameters`
- Produces: `ElementOrientation`, `AttenuationRegion`, `CouplingPath`
- Produces: `AntennaStateConfig`, `PhoneAntennaConfig`
- Produces: `resolved_phone_antenna_config(array_cfg) -> PhoneAntennaConfig`, mapping the legacy isotropic selector to two ideal states
- Produces: `attenuation_voltage(regions, active_antennas, direction) -> float64[M]`
- Produces: `coupling_matrix(paths, active_antennas) -> complex128[M,M]`
- Extends: `PhoneArrayConfig.antenna_model`, `SignalConfig.true_polarization`, `SignalConfig.assumed_polarization`, `CWInterferer.polarization`

- [ ] **Step 1: Write failing ordered-coupling and blockage tests**

```python
def test_coupling_matrix_preserves_requested_corner_order():
    paths = (CouplingPath("TL", "BR", -20.0, 90.0),)
    matrix = coupling_matrix(paths, ("BR", "TL"))
    np.testing.assert_allclose(np.diag(matrix), 1.0)
    assert matrix[0, 1] == pytest.approx(0.1j)
    assert matrix[1, 0] == 0.0


def test_overlapping_attenuation_regions_add_in_db():
    regions = (
        AttenuationRegion(("TL",), 120.0, 30.0, 20.0, 6.0, 0.0),
        AttenuationRegion(("TL",), 120.0, 30.0, 10.0, 4.0, 0.0),
    )
    voltage = attenuation_voltage(regions, ("TL", "BR"), Direction(120.0, 30.0))
    np.testing.assert_allclose(voltage, [10 ** (-10.0 / 20.0), 1.0])
```

- [ ] **Step 2: Run effects tests and record RED**

Run: `cd python_sim && python -W error -m pytest tests/test_antenna_effects.py -v`

Expected: import failure for `antenna_diversity.array.effects`.

- [ ] **Step 3: Implement immutable effect records and pure builders**

```python
@dataclass(frozen=True)
class ElementOrientation:
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0


@dataclass(frozen=True)
class AttenuationRegion:
    elements: tuple[str, ...]
    center_azimuth_deg: float
    center_elevation_deg: float
    radius_deg: float
    attenuation_db: float
    transition_width_deg: float = 0.0


@dataclass(frozen=True)
class CouplingPath:
    source: str
    receiver: str
    magnitude_db: float
    phase_deg: float
```

Compute region membership from the clipped dot product of phone-frame direction vectors. An empty `elements` tuple means all four corners; otherwise validate a unique corner subset. A positive transition width applies a raised-cosine attenuation from full attenuation inside `radius - width/2` to zero outside `radius + width/2`. Build coupling from identity and write each path whose source and receiver are both active to `[receiver_index, source_index]`; paths involving inactive corners do not alter the selected submatrix.

- [ ] **Step 4: Write failing structured-config and compatibility tests**

```python
def test_default_phone_resolves_to_ideal_antenna_without_changing_old_fields():
    cfg = default_config()
    cfg.validate()
    assert cfg.array.pattern_model == "isotropic"
    assert cfg.array.antenna_model is None


def test_structured_state_requires_exact_four_corner_maps():
    cfg = default_config()
    cfg.array.antenna_model = phone_antenna_config_for_test()
    del cfg.array.antenna_model.true_state.orientations["BR"]
    with pytest.raises(ValueError, match="TL/TR/BL/BR"):
        cfg.validate()
```

- [ ] **Step 5: Implement serializable model configuration and root validation**

```python
@dataclass
class AntennaStateConfig:
    pattern_kind: str = "isotropic"
    parametric_patterns: dict[str, ParametricPatternParameters] = field(default_factory=dict)
    measured_pattern_csv: Path | None = None
    measured_pattern_sha256: str | None = None
    orientations: dict[str, ElementOrientation] = field(default_factory=zero_orientations)
    attenuation_regions: tuple[AttenuationRegion, ...] = ()
    coupling_paths: tuple[CouplingPath, ...] = ()


@dataclass
class PhoneAntennaConfig:
    model_label: str
    model_source: str  # ideal | parametric | measured
    true_state: AntennaStateConfig
    assumed_state: AntennaStateConfig
```

Add `antenna_model: PhoneAntennaConfig | None = None` after the existing `pattern_model` field. Keep `default_config()` with `pattern_model="isotropic"` and `antenna_model=None` so serialized legacy snapshots do not change until Task 8 adds an explicit optional preset. Validate exact corners, real finite fields, path existence only for measured mode, SHA syntax, nonnegative attenuation/radius/XPD, no duplicate/self coupling paths, and no semantic conflict between old and new pattern selectors.

- [ ] **Step 6: Add strict invalid-input coverage**

Parameterize complex/vector/bool/nonfinite values, bad polarization kind, zero Jones vector, unknown/duplicate corners, negative transition or attenuation, duplicate coupling path, self path, invalid model source, measured mode without path/digest and parametric mode without four parameter records. Assert stable `ValueError` for every case.

- [ ] **Step 7: Run configuration regressions and commit Task 3**

Run: `cd python_sim && python -W error -m pytest tests/test_antenna_effects.py tests/test_phone_config.py tests/test_config_numeric.py tests/test_study_config.py -v`

```bash
git add -- python_sim/antenna_diversity/array/effects.py python_sim/antenna_diversity/array/antenna_config.py python_sim/antenna_diversity/config.py python_sim/tests/test_antenna_effects.py python_sim/tests/test_phone_config.py
git commit -m "feat: configure phone antenna effects"
```

---

### Task 4: Unified PhoneAntennaModel composition service

**Files:**
- Create: `python_sim/antenna_diversity/array/phone_model.py`
- Modify: `python_sim/antenna_diversity/array/__init__.py`
- Test: `python_sim/tests/test_phone_antenna_model.py`

**Interfaces:**
- Consumes: Tasks 1–3 records and functions plus existing phase centers/channel response
- Produces: `PhoneAntennaModel.from_config(cfg)`
- Produces: `open_circuit_response(direction, polarization, frequency_hz, estimated=False)`
- Produces: `coupling_matrix(estimated=False)`
- Produces: `effective_manifold(direction, polarization, frequency_hz, estimated=False)`
- Produces: `diagnostics(direction, polarization, frequency_hz, estimated=False)`

- [ ] **Step 1: Write the failing ideal-equivalence test**

```python
def test_ideal_phone_model_exactly_matches_existing_manifold_and_channel_response():
    cfg = default_config()
    names = cfg.array.active_antennas
    expected_nominal = array_manifold(
        select_phase_centers(cfg.phone.phase_centers_m, names),
        cfg.signal.true_direction,
        cfg.signal.carrier_hz,
        names,
        IsotropicElementPattern(),
    )
    model = PhoneAntennaModel.from_config(cfg)
    np.testing.assert_allclose(
        model.open_circuit_response(
            cfg.signal.true_direction, cfg.signal.true_polarization,
            cfg.signal.carrier_hz, estimated=False,
        ),
        expected_nominal,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        model.effective_manifold(
            cfg.signal.true_direction, cfg.signal.true_polarization,
            cfg.signal.carrier_hz, estimated=False,
        ),
        cfg.array.channel_response(False) * expected_nominal,
        rtol=0.0,
        atol=1e-12,
    )
```

- [ ] **Step 2: Run the model test and record RED**

Run: `cd python_sim && python -W error -m pytest tests/test_phone_antenna_model.py -v`

Expected: import failure for `antenna_diversity.array.phone_model`.

- [ ] **Step 3: Implement the model factory and fixed composition order**

```python
class PhoneAntennaModel:
    @classmethod
    def from_config(cls, cfg: SimulationConfig) -> "PhoneAntennaModel":
        candidate = copy.deepcopy(cfg)
        candidate.validate()
        if not isinstance(candidate.array, PhoneArrayConfig) or candidate.phone is None:
            raise ValueError("PhoneAntennaModel requires a phone-array configuration")
        antenna = resolved_phone_antenna_config(candidate.array)
        names = tuple(candidate.array.active_antennas)
        return cls(
            active_antennas=names,
            true_positions=select_phase_centers(candidate.phone.phase_centers_m, names),
            assumed_positions=select_phase_centers(
                candidate.phone.assumed_phase_centers_m, names
            ),
            true_state=_build_runtime_state(antenna.true_state, names, candidate.signal.carrier_hz),
            assumed_state=_build_runtime_state(
                antenna.assumed_state, names, candidate.signal.carrier_hz
            ),
            true_channel=candidate.array.channel_response(False),
            assumed_channel=candidate.array.channel_response(True),
            model_label=antenna.model_label,
            model_source=antenna.model_source,
        )

    def effective_manifold(
        self,
        direction: Direction,
        polarization: WavePolarization,
        frequency_hz: float,
        *,
        estimated: bool = False,
    ) -> NDArray[np.complex128]:
        open_circuit = self.open_circuit_response(
            direction, polarization, frequency_hz, estimated=estimated
        )
        coupled = self.coupling_matrix(estimated=estimated) @ open_circuit
        return self.channel_response(estimated=estimated) * coupled
```

Define the class as a frozen dataclass holding detached positions, runtime states and channel vectors. `_build_runtime_state` selects `ScalarPatternAdapter(IsotropicElementPattern())`, `ParametricElementPattern`, or `TabulatedElementPattern.from_csv()` from `pattern_kind`, prepares the four rotation matrices and ordered coupling matrix, and never reloads a file during grid evaluation. For each element, compute the existing position phase in phone coordinates, transform direction and field into its local frame, evaluate the selected polarized effective height, apply `np.vdot(height, local_wave.jones)`, then apply the phone-frame attenuation voltage. Do not normalize the response.

- [ ] **Step 4: Add composition, order and mismatch tests**

```python
def test_effective_response_is_channel_times_coupling_times_open_circuit():
    cfg = configured_two_corner_case()
    model = PhoneAntennaModel.from_config(cfg)
    open_circuit = model.open_circuit_response(DIRECTION, POLARIZATION, FREQ)
    expected = cfg.array.channel_response(False) * (model.coupling_matrix(False) @ open_circuit)
    np.testing.assert_allclose(model.effective_manifold(DIRECTION, POLARIZATION, FREQ), expected)


def test_estimated_response_uses_only_assumed_state_and_estimated_channel():
    cfg = deliberately_mismatched_true_and_assumed_case()
    model = PhoneAntennaModel.from_config(cfg)
    assert not np.allclose(
        model.effective_manifold(D_TRUE, P_TRUE, FREQ, estimated=False),
        model.effective_manifold(D_ASSUMED, P_ASSUMED, FREQ, estimated=True),
    )
```

Also cover arbitrary 2/3/4 active order, orientation, orthogonal polarization, overlapping attenuation, asymmetric coupling, and detached finite diagnostics.

- [ ] **Step 5: Run focused regressions and commit Task 4**

Run: `cd python_sim && python -W error -m pytest tests/test_phone_antenna_model.py tests/test_antenna_effects.py tests/test_antenna_patterns.py tests/test_manifold.py tests/test_phone_config.py -v`

```bash
git add -- python_sim/antenna_diversity/array/phone_model.py python_sim/antenna_diversity/array/__init__.py python_sim/tests/test_phone_antenna_model.py
git commit -m "feat: compose realistic phone antenna responses"
```

---

### Task 5: Scenario and analytical-path integration

**Files:**
- Modify: `python_sim/antenna_diversity/gnss/scenario.py`
- Modify: `python_sim/antenna_diversity/analysis/spatial.py`
- Modify: `python_sim/antenna_diversity/experiments/spatial_scan.py`
- Test: `python_sim/tests/test_scenario.py`
- Test: `python_sim/tests/test_multi_interferer_scenario.py`
- Test: `python_sim/tests/test_spatial_analysis.py`
- Test: `python_sim/tests/test_experiments.py`
- Test: `python_sim/tests/test_end_to_end.py`

**Interfaces:**
- Consumes: `PhoneAntennaModel` from Task 4
- Preserves: existing truth/metrics keys and legacy ULA path
- Adds truth: `antenna_model_source`, open-circuit responses, blockage voltages, coupling matrices, polarization-match losses
- Extends: `response_grid_db(cfg, weights, azimuths_deg, elevations_deg, model_view="true")`

- [ ] **Step 1: Write failing scenario single-source-of-truth tests**

```python
def test_scenario_truth_and_samples_use_shared_phone_antenna_model():
    cfg = parametric_two_corner_case()
    model = PhoneAntennaModel.from_config(cfg)
    samples, truth = generate_scenario(cfg)
    expected = model.effective_manifold(
        cfg.signal.true_direction, cfg.signal.true_polarization,
        cfg.signal.carrier_hz, estimated=False,
    )
    np.testing.assert_allclose(truth["desired_steering_true"], expected)
    np.testing.assert_allclose(
        truth["desired_component"],
        expected[:, None] * truth["desired_waveform"][None, :],
    )
```

- [ ] **Step 2: Run scenario/experiment tests and record RED**

Run: `cd python_sim && python -W error -m pytest tests/test_scenario.py tests/test_multi_interferer_scenario.py tests/test_experiments.py -v`

Expected: new parameterized assertions fail because these modules still construct `IsotropicElementPattern` directly.

- [ ] **Step 3: Replace phone-only manifold construction in scenario**

Construct one `PhoneAntennaModel` per `generate_scenario()` call. For the desired source return:

```python
desired_nominal = model.open_circuit_response(true_direction, true_polarization, frequency, estimated=False)
desired_true = model.effective_manifold(true_direction, true_polarization, frequency, estimated=False)
desired_assumed = model.effective_manifold(assumed_direction, assumed_polarization, frequency, estimated=True)
```

For every enabled jammer use its actual polarization and the true state. Populate old aliases exactly as before and add detached diagnostics without storing the runtime model object.

- [ ] **Step 4: Replace phone-only manifold construction in analytical experiments**

In `evaluate_configuration_spatially()`, build true desired, assumed desired and every jammer steering through one model instance. Remove the rejection that restricts `pattern_model` to isotropic. Keep signal/interferer power and RF noise formulas unchanged.

- [ ] **Step 5: Extend response-grid behavior**

```python
def response_grid_db(
    cfg: SimulationConfig,
    weights: ArrayLike,
    azimuths_deg: ArrayLike,
    elevations_deg: ArrayLike,
    model_view: str = "true",
) -> NDArray[np.float64]:
    view = str(model_view).lower()
    if view not in {"true", "assumed"}:
        raise ValueError("model_view must be true or assumed")
    model = PhoneAntennaModel.from_config(cfg)
    polarization = (
        cfg.signal.true_polarization if view == "true"
        else cfg.signal.assumed_polarization
    )
    estimated = view == "assumed"
    response = np.empty((elevations.size, azimuths.size), dtype=float)
    for row, elevation in enumerate(elevations):
        for column, azimuth in enumerate(azimuths):
            steering = model.effective_manifold(
                Direction(float(azimuth), float(elevation)),
                polarization,
                cfg.signal.carrier_hz,
                estimated=estimated,
            )
            response[row, column] = abs(np.vdot(w, steering))
    return _normalized_response_db(response, floor_db=-80.0)
```

Retain the existing weight/grid validation before this block. `_normalized_response_db` preserves the current exact-null behavior, `-80 dB` floor and max normalization of the final plotted array response; it does not normalize the underlying manifold.

- [ ] **Step 6: Add analytical/sample consistency and legacy regressions**

Test that manually computed covariance uses the same true effective vectors as scenario truth, while MVDR uses the assumed desired vector. Test a polarization mismatch, a blocked element and asymmetric coupling. Assert the legacy ULA and ideal phone expected arrays are unchanged.

- [ ] **Step 7: Run the integration suite and commit Task 5**

Run: `cd python_sim && python -W error -m pytest tests/test_scenario.py tests/test_multi_interferer_scenario.py tests/test_spatial_analysis.py tests/test_experiments.py tests/test_end_to_end.py -v`

```bash
git add -- python_sim/antenna_diversity/gnss/scenario.py python_sim/antenna_diversity/analysis/spatial.py python_sim/antenna_diversity/experiments/spatial_scan.py python_sim/tests/test_scenario.py python_sim/tests/test_multi_interferer_scenario.py python_sim/tests/test_spatial_analysis.py python_sim/tests/test_experiments.py python_sim/tests/test_end_to_end.py
git commit -m "feat: use phone antenna model across simulation paths"
```

---

### Task 6: Reproducible study configuration and measured resources

**Files:**
- Modify: `python_sim/antenna_diversity/studies/config.py`
- Modify: `python_sim/antenna_diversity/studies/models.py`
- Modify: `python_sim/antenna_diversity/studies/scenarios.py`
- Modify: `python_sim/antenna_diversity/studies/full_chain.py`
- Test: `python_sim/tests/test_study_config.py`
- Test: `python_sim/tests/test_study_scenarios.py`
- Test: `python_sim/tests/test_study_full_chain.py`

**Interfaces:**
- Consumes: structured antenna records and `PhoneAntennaModel`
- Produces: strict nested JSON parsing for true/assumed antenna states and source polarizations
- Produces: `resolve_pattern_resources(simulation, config_dir) -> SimulationConfig`
- Guarantees: `ResolvedStudyConfig.config_sha256` includes antenna fields and measured file SHA-256

- [ ] **Step 1: Write failing nested JSON and digest tests**

```python
def test_resolver_parses_parametric_true_and_assumed_antenna_states(tmp_path):
    path = write_study_json(tmp_path, PARAMETRIC_ANTENNA_OVERRIDE)
    resolved = resolve_study_config(path, preset="quick")
    assert resolved.simulation.array.antenna_model.model_source == "parametric"
    assert resolved.simulation.signal.true_polarization.kind == "rhcp"
    assert resolved.simulation.interferers[0].polarization.kind == "linear"


def test_measured_pattern_digest_changes_scientific_hash(tmp_path):
    csv_path = write_pattern_csv(tmp_path)
    first = resolve_measured_study(tmp_path, csv_path)
    mutate_one_finite_component(csv_path)
    second = resolve_measured_study(tmp_path, csv_path)
    assert first.config_sha256 != second.config_sha256
```

- [ ] **Step 2: Run study tests and record RED**

Run: `cd python_sim && python -W error -m pytest tests/test_study_config.py tests/test_study_scenarios.py tests/test_study_full_chain.py -v`

Expected: nested dataclass parsing rejects or leaves raw dictionaries, and the digest is absent from the resolved hash.

- [ ] **Step 3: Implement exact-key nested parsers**

Add explicit functions rather than extending the generic dataclass updater blindly:

```python
def _polarization(raw: object, context: str) -> WavePolarization:
    data = _exact_record(raw, WavePolarization, context)
    value = WavePolarization(**data)
    value.jones_vector()
    return value


def _orientation_map(raw: object, context: str) -> dict[str, ElementOrientation]:
    mapping = _mapping(raw, context)
    if set(mapping) != PHONE_CORNER_NAMES:
        raise ValueError(f"{context} must contain exactly TL/TR/BL/BR")
    return {
        name: ElementOrientation(**_exact_record(value, ElementOrientation, f"{context}.{name}"))
        for name, value in mapping.items()
    }


def _attenuation_regions(raw: object, context: str) -> tuple[AttenuationRegion, ...]:
    return tuple(
        AttenuationRegion(**_exact_record(item, AttenuationRegion, f"{context}[{index}]"))
        for index, item in enumerate(_sequence(raw, context))
    )


def _coupling_paths(raw: object, context: str) -> tuple[CouplingPath, ...]:
    return tuple(
        CouplingPath(**_exact_record(item, CouplingPath, f"{context}[{index}]"))
        for index, item in enumerate(_sequence(raw, context))
    )
```

Implement `_antenna_state` by accepting exactly the `AntennaStateConfig` fields, routing its nested pattern/orientation/region/path fields through the parsers above, converting its CSV string to `Path`, and calling `validate()`. Implement `_antenna_model` the same way for `model_label`, `model_source`, `true_state`, and `assumed_state`; when no base exists, start from `ideal_phone_antenna_config()`. `_mapping`, `_sequence`, and `_exact_record` must type-check before membership or constructor calls and translate constructor `TypeError` into contextual `ValueError`. Each parser rejects missing, unknown, wrong-shape, complex, bool-as-number and nonfinite values. Resolve measured CSV paths relative to the JSON file directory, compute SHA-256 from bytes, and replace the resolved state digest before `ResolvedStudyConfig.from_parts()`.

- [ ] **Step 4: Make immutable snapshots cover every new container**

Extend canonical copying so pattern maps, orientation maps, attenuation tuples, coupling tuples and polarization records are detached. Add a test that mutating any original nested dict/list after `from_parts()` cannot change `to_dict()` while `config_sha256` stays fixed.

- [ ] **Step 5: Preserve selected-case reconstruction**

Ensure direction, position, amplitude and phase error application modifies only the already established assumed fields. Full-chain selected cases must deep-copy the complete antenna model and must not drop polarizations, blockage, coupling or measured resource digest.

- [ ] **Step 6: Add stable failure-path tests**

Cover incomplete pattern parameters, malformed region/path entries, changed expected digest, missing CSV, relative path resolution, raw nested lists and invalid source polarization. Check that all failures are `ValueError` and that caller configs/files remain unchanged.

- [ ] **Step 7: Run study regressions and commit Task 6**

Run: `cd python_sim && python -W error -m pytest tests/test_study_config.py tests/test_study_scenarios.py tests/test_study_full_chain.py tests/test_phone_config.py -v`

```bash
git add -- python_sim/antenna_diversity/studies/config.py python_sim/antenna_diversity/studies/models.py python_sim/antenna_diversity/studies/scenarios.py python_sim/antenna_diversity/studies/full_chain.py python_sim/tests/test_study_config.py python_sim/tests/test_study_scenarios.py python_sim/tests/test_study_full_chain.py
git commit -m "feat: resolve reproducible antenna study inputs"
```

---

### Task 7: Saved antenna diagnostics, figures and report evidence

**Files:**
- Modify: `python_sim/antenna_diversity/studies/analytical.py`
- Modify: `python_sim/antenna_diversity/studies/runner.py`
- Modify: `python_sim/antenna_diversity/studies/plots.py`
- Modify: `python_sim/antenna_diversity/studies/report.py`
- Modify: `python_sim/run_spatial_research.py`
- Test: `python_sim/tests/test_study_analytical.py`
- Test: `python_sim/tests/test_study_reporting.py`
- Test: `python_sim/tests/test_study_runner.py`
- Test: `python_sim/tests/test_study_cli.py`

**Interfaces:**
- Adds saved table: `antenna_diagnostics`
- Adds fixed figures: `element_patterns`, `polarization_losses`, `coupling_matrices`, `blockage_comparison`
- Changes publication totals from 8 tables/8 figures to 9 tables/12 figures
- Preserves report-stage rule: no simulation, model evaluation or random draws during reporting

- [ ] **Step 1: Write failing diagnostics-table tests**

```python
def test_baseline_saves_complete_antenna_diagnostics():
    resolved = resolved_parametric_quick_config()
    records = run_baseline(resolved)
    rows = records.tables["antenna_diagnostics"]
    assert {row["model_source"] for row in rows} == {"parametric"}
    assert {row["model_view"] for row in rows} == {"true", "assumed"}
    assert {row["element_name"] for row in rows} == {"TL", "TR", "BL", "BR"}
    json.dumps(records.to_dict(), allow_nan=False)
```

Rows must identify record kind (`pattern_grid`, `source_match`, `coupling_entry`, `blockage_grid`), model source/view, element or matrix indices, azimuth/elevation and finite scalar real/imaginary or dB values. The fixed CSV schema is `record_kind, model_source, model_label, model_view, element_name, receiver, source, source_name, azimuth_deg, elevation_deg, response_real, response_imag, response_db, attenuation_db, polarization_match_db, measured_pattern_sha256`. Unused fields are `None`; do not place raw NumPy arrays or runtime objects in checkpoints.

- [ ] **Step 2: Run analytical tests and record RED**

Run: `cd python_sim && python -W error -m pytest tests/test_study_analytical.py -k antenna -v`

Expected: missing `antenna_diagnostics` table.

- [ ] **Step 3: Generate diagnostics once in the baseline stage**

Add a pure `antenna_diagnostic_rows(resolved) -> tuple[dict[str, object], ...]` that evaluates the configured study azimuth/elevation axes through `PhoneAntennaModel`. Include source-direction records for the GPS signal and every enabled jammer, and matrix records in fixed corner order. Attach the rows to `run_baseline()` output. Reporting must only receive these saved rows.

- [ ] **Step 4: Write failing fixed-publication and no-recompute tests**

```python
def test_report_publication_has_nine_tables_and_twelve_figures(tmp_path):
    output = run_minimal_parametric_study(tmp_path)
    assert len(list((output / "tables").glob("*.csv"))) == 9
    assert len(list((output / "figures").glob("*.png"))) == 12


def test_plotting_uses_saved_diagnostics_without_model_calls(monkeypatch):
    monkeypatch.setattr(phone_model, "PhoneAntennaModel", forbidden_call)
    figures = plot_study_tables(completed_tables_with_antenna_diagnostics())
    assert set(NEW_FIGURE_KEYS) <= set(figures)
```

- [ ] **Step 5: Extend fixed table and figure schemas**

Add `antenna_diagnostics` to `_TABLE_ORDER`, `_CSV_FIELDS` and required-table validation. Add the four figure keys to `FIGURE_KEYS`. Builders must filter missing/nonfinite optional rows before sorting, use shared finite color limits where panels compare true/assumed states, preserve nonuniform axis geometry, close partially constructed figures on any error and continue using atomic PNG publication.

- [ ] **Step 6: Extend summary and Markdown wording**

Add:

```python
"antenna_model": {
    "source": "ideal|parametric|measured",
    "label": "synthetic_phone_corner_v1",
    "measured_pattern_sha256": None,
    "active_antennas": ["TL", "TR", "BL", "BR"],
}
```

Render links and concise findings for element patterns, polarization match, coupling and blockage. Replace the old claim that these effects are unmodeled with precise current-state wording. Keep multipath, dynamic DOA, non-CW interference, multi-satellite/PVT, correlated external antenna noise and full-wave platform scattering in the omissions.

- [ ] **Step 7: Update CLI artifact verification tests**

The real minimal CLI acceptance must verify the existing 10 strict JSON artifacts, 9 CSV files, 12 PNG files, valid report links, and model-source text. Do not add warning filters.

- [ ] **Step 8: Run reporting/runner/CLI regressions and commit Task 7**

Run: `cd python_sim && $env:MPLBACKEND='Agg'; python -W error -m pytest tests/test_study_analytical.py tests/test_study_reporting.py tests/test_study_runner.py tests/test_study_cli.py tests/test_plotting.py -v`

```bash
git add -- python_sim/antenna_diversity/studies/analytical.py python_sim/antenna_diversity/studies/runner.py python_sim/antenna_diversity/studies/plots.py python_sim/antenna_diversity/studies/report.py python_sim/run_spatial_research.py python_sim/tests/test_study_analytical.py python_sim/tests/test_study_reporting.py python_sim/tests/test_study_runner.py python_sim/tests/test_study_cli.py
git commit -m "feat: report phone antenna model evidence"
```

---

### Task 8: Parametric preset, examples, diagrams and final verification

**Files:**
- Modify: `python_sim/antenna_diversity/config.py`
- Modify: `python_sim/antenna_diversity/__init__.py`
- Create: `python_sim/antenna_diversity/studies/example_antenna_realism.json`
- Modify: `python_sim/README.md`
- Modify: `python_sim/ARCHITECTURE.md`
- Modify: `python_sim/DESIGN.md`
- Modify: `python_sim/docs/simulation_architecture.svg`
- Modify: `python_sim/docs/simulation_architecture.png`
- Test: `python_sim/tests/test_phone_config.py`
- Test: `python_sim/tests/test_study_cli.py`

**Interfaces:**
- Produces: `parametric_phone_config(active_antennas=("TL", "TR", "BL", "BR")) -> SimulationConfig`
- Preserves: `default_config()` as the ideal baseline
- Publishes: a runnable parameterized study example and updated editable architecture figure

- [ ] **Step 1: Write the failing preset test**

```python
def test_parametric_phone_preset_is_explicit_nonmeasured_and_runnable():
    cfg = parametric_phone_config(("TL", "BR"))
    cfg.validate()
    assert cfg.array.antenna_model.model_source == "parametric"
    assert cfg.array.active_antennas == ("TL", "BR")
    assert cfg.signal.true_polarization.kind == "rhcp"
    assert cfg.array.antenna_model.true_state.coupling_paths
    result = evaluate_configuration_spatially(cfg, "mvdr")
    assert np.isfinite(result.metrics["output_sinr_db"])
```

- [ ] **Step 2: Run preset tests and record RED**

Run: `cd python_sim && python -W error -m pytest tests/test_phone_config.py -k parametric_phone_preset -v`

Expected: `parametric_phone_config` is not defined.

- [ ] **Step 3: Implement the explicit parameterized preset**

Use deterministic, documented synthetic values: four distinct yaw orientations, finite cosine-rolloff patterns, GPS RHCP, linearly polarized jammers, one reproducible attenuation region and weak asymmetric coupling paths. Label the model `synthetic_phone_corner_v1` and `model_source="parametric"`. Do not change `default_config()`.

- [ ] **Step 4: Add and run the example study**

Create `example_antenna_realism.json` with two antenna combinations, EGC/MVDR analytical modes, SINGLE/EGC/MVDR full-chain modes, a small direction grid, two jammer powers, one combined-error sample, one full-chain run and explicit antenna overrides. Run:

```powershell
$env:MPLBACKEND='Agg'
python -W error run_spatial_research.py --config antenna_diversity/studies/example_antenna_realism.json --preset quick --name antenna-realism-smoke
```

Expected: exit 0, completed seven-stage manifest, 9 CSVs, 12 PNGs, strict JSON, and a report marked `parametric`.

- [ ] **Step 5: Update technical documentation and architecture assets**

Document the composition equation, true/assumed split, Jones convention, power reference, all three model sources, CSV schema/digest, preset commands and current omissions. Update the editable SVG central path from “isotropic element interface” to the layered antenna model, regenerate the 1920×1200 PNG from the SVG using the existing bundled Node/Sharp runtime, parse the SVG as UTF-8 XML and inspect the PNG at original resolution for clipping or false claims.

- [ ] **Step 6: Run complete verification**

Run:

```powershell
cd python_sim
$env:MPLBACKEND='Agg'
python -W error -m pytest -q
python -m compileall antenna_diversity run_simulation.py run_legacy_ula.py run_spatial_research.py
git diff --check
```

Also run the ideal default CLI, legacy ULA CLI and parameterized study CLI once each with `-W error`. Confirm existing ideal outputs still contain the original branches and figures, the parameterized run publishes all new evidence, no warning filter was added, no result directory is staged, and `git status --short` contains only intended source/test/doc changes before commit.

- [ ] **Step 7: Commit Task 8**

```bash
git add -- python_sim/antenna_diversity/config.py python_sim/antenna_diversity/__init__.py python_sim/antenna_diversity/studies/example_antenna_realism.json python_sim/README.md python_sim/ARCHITECTURE.md python_sim/DESIGN.md python_sim/docs/simulation_architecture.svg python_sim/docs/simulation_architecture.png python_sim/tests/test_phone_config.py python_sim/tests/test_study_cli.py
git commit -m "feat: complete parameterized phone antenna research model"
```

- [ ] **Step 8: Final branch audit**

Verify `git log --oneline origin/main..HEAD`, `git diff --check origin/main..HEAD`, full file scope, documentation links, measured/parametric wording, and a clean worktree. Record exact test counts and CLI artifact counts for the PR description.
