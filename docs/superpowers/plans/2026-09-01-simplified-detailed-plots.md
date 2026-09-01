# Simplified Detailed Plots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the seven existing simplified overview figures and add clear per-antenna and per-combining-mode diagnostic figures derived from the full simulator.

**Architecture:** Preserve `plot_results()` as the stable overview API. Add `plot_detailed_results()` in the same plotting module, backed by focused private helpers for absolute spectra, frequency markers, two-dimensional array response, and acquisition detail; the runner merges both dictionaries only when saving. No receiver-stage data or algorithms change.

**Tech Stack:** Python 3.11+, NumPy, Matplotlib, pytest

**Spec:** `docs/superpowers/specs/2026-09-01-simplified-detailed-plots-design.md`

## Global Constraints

- Keep all seven existing `plot_results()` names and figures unchanged.
- Do not import runtime code from `python_sim`.
- Use absolute FFT magnitude dB for detailed spectra and one shared y-axis range across combining modes.
- Generate one detailed RF→ADC→DFE figure per active antenna and three detailed figures per active combining branch.
- Keep Tracking and C/N₀ as overview-only figures.
- Do not change simulation, combining, acquisition, or tracking calculations.

---

### Task 1: Lock the detailed plotting contract

**Files:**
- Modify: `python_sim_simplified/tests/test_plotting_runner.py`
- Test: `python_sim_simplified/tests/test_plotting_runner.py`

**Interfaces:**
- Consumes: `default_config()`, `run_end_to_end(cfg)`, and existing result-dictionary fields.
- Produces: required public API `plot_detailed_results(results) -> dict[str, Figure]` and its stable key contract.

- [ ] **Step 1: Write the failing tests**

Add imports for `numpy` and `plot_detailed_results`, then add tests equivalent to:

```python
def test_detailed_results_create_one_clear_figure_per_channel_and_mode():
    cfg = default_config()
    cfg.signal.duration_ms = 20
    cfg.tracking.max_epochs = 10
    results = run_end_to_end(cfg)

    figures = plot_detailed_results(results)
    expected = {
        *(f"detailed_rf_dfe_{name}" for name in cfg.phone.active_antennas),
        *(f"detailed_combined_{mode}" for mode in cfg.diversity.modes),
        *(f"detailed_array_response_{mode}" for mode in cfg.diversity.modes),
        *(f"detailed_acquisition_{mode}" for mode in cfg.diversity.modes),
    }

    assert set(figures) == expected
    assert len(figures) == 16
    for name in cfg.phone.active_antennas:
        assert len(figures[f"detailed_rf_dfe_{name}"].axes) == 3
    for figure in figures.values():
        plt.close(figure)


def test_detailed_combined_spectra_share_absolute_limits_and_mark_frequencies():
    cfg = default_config()
    cfg.signal.duration_ms = 20
    cfg.tracking.max_epochs = 10
    results = run_end_to_end(cfg)
    figures = plot_detailed_results(results)
    axes = [
        figures[f"detailed_combined_{mode}"].axes[0]
        for mode in cfg.diversity.modes
    ]

    assert all(axis.get_ylim() == axes[0].get_ylim() for axis in axes)
    assert all(axis.get_ylabel() == "Magnitude (dB)" for axis in axes)
    assert all(len(axis.lines) == 3 for axis in axes)
    for figure in figures.values():
        plt.close(figure)


def test_detailed_spatial_and_acquisition_figures_keep_diagnostics_separate():
    cfg = default_config()
    cfg.signal.duration_ms = 20
    cfg.tracking.max_epochs = 10
    results = run_end_to_end(cfg)
    figures = plot_detailed_results(results)

    for mode in cfg.diversity.modes:
        spatial = figures[f"detailed_array_response_{mode}"]
        spatial_axis = next(axis for axis in spatial.axes if axis.images)
        assert spatial_axis.images[0].get_clim() == (-60.0, 0.0)
        assert len(spatial_axis.lines) == 2

        acquisition = figures[f"detailed_acquisition_{mode}"]
        assert any(axis.images for axis in acquisition.axes)
        assert any(axis.get_xlabel() == "Code phase (chips)" for axis in acquisition.axes)
    for figure in figures.values():
        plt.close(figure)
```

These tests catch missing channel/mode outputs, accidental normalized spectra, branch-dependent visual scaling, omitted direction markers, and a regression back to acquisition maps without code slices.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
python -m pytest python_sim_simplified/tests/test_plotting_runner.py -q
```

Expected: collection fails because `plot_detailed_results` does not exist.

- [ ] **Step 3: Commit the failing contract tests**

```powershell
git add -- python_sim_simplified/tests/test_plotting_runner.py
git commit -m "test: define detailed plot outputs"
```

---

### Task 2: Implement independent detailed diagnostics

**Files:**
- Modify: `python_sim_simplified/plotting.py`
- Test: `python_sim_simplified/tests/test_plotting_runner.py`

**Interfaces:**
- Consumes: the existing `SimulationResults` dictionary, `MODE_ORDER`, `build_antenna_model(phone, carrier_hz)`, and `antenna_response(model, direction, polarization)`.
- Produces: `plot_detailed_results(results) -> dict[str, matplotlib.figure.Figure]` with the exact keys established by Task 1.

- [ ] **Step 1: Add absolute-spectrum and frequency-marker helpers**

Add `_absolute_spectrum(x, fs_hz, n_fft=None)` returning MHz and unnormalized magnitude dB. Add `_mark_frequencies(axis, results, rf_domain)` that draws one green target line and one red line for each enabled interferer, adding `cfg.signal.if_hz` only for RF/ADC plots.

```python
def _absolute_spectrum(x, fs_hz, n_fft=None):
    samples = np.asarray(x).reshape(-1)
    size = min(samples.size, 65_536) if n_fft is None else n_fft
    samples = samples[:size]
    values = np.fft.fftshift(np.fft.fft(samples * np.hanning(size)))
    frequency_mhz = np.fft.fftshift(np.fft.fftfreq(size, 1.0 / fs_hz)) / 1e6
    return frequency_mhz, 20.0 * np.log10(np.abs(values))
```

- [ ] **Step 2: Add per-antenna RF→ADC→DFE figures**

Implement `_plot_detailed_rf_dfe(results)` returning one named figure per active antenna. Each figure has three vertical axes using `scene_input_v[index]`, `adc_codes[index]`, and `dfe_channels[index]`; RF/ADC use `cfg.signal.fs_rf_hz`, DFE uses `results["dfe_info"]["fs_out_hz"]`.

- [ ] **Step 3: Add per-mode absolute combined spectra**

Implement `_plot_detailed_combined(results)` by calculating all branch spectra with one common FFT length, calculating one global `(y_min, y_max)` pair, and creating one figure per mode. Each figure contains one spectrum plus target/interferer markers and the same y limits.

- [ ] **Step 4: Add per-mode two-dimensional spatial response figures**

Build the true manifold once over `azimuths = np.arange(0.0, 361.0, 2.0)` and `elevations = np.arange(0.0, 91.0, 2.0)`. For each mode, apply its fixed combining weights, normalize the grid to its peak, display `-60～0 dB`, and plot the target as a green star and each enabled interferer as a red cross.

- [ ] **Step 5: Add per-mode acquisition map and slice figures**

For each branch, create a `1×2` figure: normalized Doppler/code search power on the left and the best-Doppler row on the right. Convert sample position to chips using `samples_per_ms` and `1023.0`; mark the primary and second peaks and include acquisition status, metric, Doppler, and code phase in the titles.

- [ ] **Step 6: Add the public aggregator**

```python
def plot_detailed_results(results):
    return {
        **_plot_detailed_rf_dfe(results),
        **_plot_detailed_combined(results),
        **_plot_detailed_array_response(results),
        **_plot_detailed_acquisition(results),
    }
```

- [ ] **Step 7: Run the focused tests and verify GREEN**

Run:

```powershell
python -m pytest python_sim_simplified/tests/test_plotting_runner.py -q
```

Expected: all plotting-runner tests pass without warnings or open-figure leaks.

- [ ] **Step 8: Commit the detailed plotting implementation**

```powershell
git add -- python_sim_simplified/plotting.py
git commit -m "feat: add independent detailed diagnostics"
```

---

### Task 3: Save overview and detailed figures from the runner

**Files:**
- Modify: `python_sim_simplified/run_end_to_end.py`
- Modify: `python_sim_simplified/tests/test_plotting_runner.py`
- Modify: `python_sim_simplified/README.md`

**Interfaces:**
- Consumes: `plot_results(results)` and `plot_detailed_results(results)`.
- Produces: one runner execution that saves 23 PNG files for the default four-antenna/four-mode configuration.

- [ ] **Step 1: Add a failing runner-save test**

Expose `save_results_figures(results, output_dir)` from `run_end_to_end.py`, then test the wished-for behavior with `tmp_path`:

```python
def test_save_results_figures_writes_overview_and_detailed_outputs(tmp_path):
    cfg = default_config()
    cfg.signal.duration_ms = 20
    cfg.tracking.max_epochs = 10
    results = run_end_to_end(cfg)

    paths = save_results_figures(results, tmp_path)

    assert len(paths) == 23
    assert set(paths) >= {
        "rf_spectrum",
        "detailed_rf_dfe_TL",
        "detailed_combined_mvdr",
        "detailed_array_response_blind_null",
        "detailed_acquisition_egc",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```powershell
python -m pytest python_sim_simplified/tests/test_plotting_runner.py::test_save_results_figures_writes_overview_and_detailed_outputs -q
```

Expected: FAIL because `save_results_figures` does not exist.

- [ ] **Step 3: Implement runner saving**

Import both plotting aggregators, merge their results inside `save_results_figures`, save each PNG at 150 dpi, close every figure after saving when plots are not shown, and have `main()` call the helper with `cfg.plot.output_dir`.

- [ ] **Step 4: Update the README output description**

Document the seven overview figures, the per-antenna/per-mode detailed naming scheme, and the default total of 23 PNG files.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```powershell
python -m pytest python_sim_simplified/tests/test_plotting_runner.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Run full verification**

Run:

```powershell
python -m pytest python_sim_simplified/tests -q
python -m compileall -q python_sim_simplified
$env:MPLBACKEND='Agg'; python python_sim_simplified/run_end_to_end.py
```

Expected: all simplified tests pass, compilation exits zero, runner exits zero, and the default output directory contains seven overview PNGs plus sixteen detailed PNGs.

- [ ] **Step 7: Commit runner and documentation changes**

```powershell
git add -- python_sim_simplified/run_end_to_end.py python_sim_simplified/tests/test_plotting_runner.py python_sim_simplified/README.md
git commit -m "feat: save detailed simulation figures"
```
