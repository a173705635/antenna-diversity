# Standalone C/N₀ Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test and expose the receiver's measured Prompt-window C/N₀ estimator, then save a standalone SINGLE/EGC/MVDR C/N₀ comparison figure.

**Architecture:** Keep one C/N₀ definition in `receiver/tracking.py` by extracting the existing formula into a small public estimator that the tracking loop calls. Add `plot_cn0()` as a read-only consumer of per-branch tracking results, integrate it into the standard six-figure report, and make failed acquisition visible as status text instead of invented C/N₀ data.

**Tech Stack:** Python 3.11, NumPy, Matplotlib, pytest, existing antenna-diversity result dataclasses and CLI entry points.

## Global Constraints

- The only measured C/N₀ source is the tracking Prompt-window estimator; do not add theoretical or pre-acquisition C/N₀.
- Use `TrackingConfig.cn0_window_epochs` and the actual coherent integration duration already used by the tracking loop.
- A failed acquisition or tracking branch has no C/N₀ curve and must be annotated explicitly.
- Plot time comes from `epoch_start_samples / results.config.dfe.fs_out_hz`, expressed in milliseconds.
- Keep the five existing plot keys unchanged and add exactly `cn0_comparison`.
- Do not change acquisition thresholds, tracking loops, combining weights, RF/DFE models, or smooth/clip plotted C/N₀ values.
- All test commands run from `F:\antenna_diversity\python_sim` with warnings treated as errors.

---

### Task 1: Extract and verify the Prompt-window C/N₀ estimator

**Files:**
- Modify: `python_sim/antenna_diversity/receiver/tracking.py`
- Modify: `python_sim/tests/test_tracking.py`

**Interfaces:**
- Consumes: one-dimensional Prompt correlator samples and one finite positive coherent integration duration.
- Produces: `estimate_cn0_db_hz(prompt_window: ArrayLike, coherent_interval_s: float) -> float`.
- Produces: `track(...)["cn0_db_hz"]`, computed by calling the same estimator for each sliding window.

- [ ] **Step 1: Write failing estimator contract tests**

Add the import and tests below to `tests/test_tracking.py`:

```python
from antenna_diversity.receiver.tracking import estimate_cn0_db_hz, track


def test_prompt_window_cn0_matches_coherent_to_noise_power_ratio():
    cn0_db_hz = estimate_cn0_db_hz(
        np.array([1.0 + 0.0j, 3.0 + 0.0j]),
        coherent_interval_s=1e-3,
    )

    assert cn0_db_hz == pytest.approx(10.0 * np.log10(4000.0))


@pytest.mark.parametrize(
    "prompt_window",
    [np.array([], dtype=complex), np.array([1.0 + 0.0j])],
)
def test_prompt_window_cn0_is_unavailable_before_two_samples(prompt_window):
    assert np.isnan(estimate_cn0_db_hz(prompt_window, 1e-3))


@pytest.mark.parametrize("coherent_interval_s", [0.0, -1e-3, np.nan, np.inf])
def test_prompt_window_cn0_rejects_invalid_coherent_interval(coherent_interval_s):
    with pytest.raises(ValueError, match="coherent_interval_s"):
        estimate_cn0_db_hz(np.ones(2, dtype=complex), coherent_interval_s)


def test_prompt_window_cn0_rejects_nonvector_prompt_samples():
    with pytest.raises(ValueError, match="one-dimensional"):
        estimate_cn0_db_hz(np.ones((2, 1), dtype=complex), 1e-3)
```

- [ ] **Step 2: Run the estimator tests and verify RED**

Run:

```powershell
D:\Python\Python311\python.exe -W error -m pytest tests/test_tracking.py -k "prompt_window_cn0" -v
```

Expected: collection fails because `estimate_cn0_db_hz` does not exist.

- [ ] **Step 3: Implement the estimator and route tracking through it**

Add this function near `_wrap_phase()` in `receiver/tracking.py`:

```python
def estimate_cn0_db_hz(
    prompt_window: ArrayLike,
    coherent_interval_s: float,
) -> float:
    """Estimate C/N0 from a window of coherent Prompt correlators."""

    prompt_samples = np.asarray(prompt_window, dtype=np.complex128)
    if prompt_samples.ndim != 1:
        raise ValueError("prompt_window must be one-dimensional")
    if not np.isfinite(coherent_interval_s) or coherent_interval_s <= 0.0:
        raise ValueError("coherent_interval_s must be finite and positive")
    if prompt_samples.size < 2:
        return float("nan")

    coherent_power = abs(np.mean(prompt_samples)) ** 2
    total_power = np.mean(np.abs(prompt_samples) ** 2)
    eps = np.finfo(float).eps
    noise_power = max(float(total_power - coherent_power), eps)
    return float(
        10.0
        * np.log10(
            max(
                coherent_power / (noise_power * coherent_interval_s),
                eps,
            )
        )
    )
```

Replace the in-loop duplicated formula with:

```python
cn0_db_hz[epoch] = estimate_cn0_db_hz(
    prompt_window,
    epoch_duration_s,
)
```

Keep the existing `prompt_window.size >= 2` behavior either by retaining the guard or by relying on the estimator's `NaN` return; do not calculate a value from a single Prompt.

- [ ] **Step 4: Run tracking tests and verify GREEN**

Run:

```powershell
D:\Python\Python311\python.exe -W error -m pytest tests/test_tracking.py -v
```

Expected: all tracking tests pass with no warnings.

- [ ] **Step 5: Commit the estimator**

```powershell
git add -- python_sim/antenna_diversity/receiver/tracking.py python_sim/tests/test_tracking.py
git commit -m "feat: expose prompt-window cn0 estimator"
```

---

### Task 2: Add the standalone C/N₀ figure

**Files:**
- Modify: `python_sim/antenna_diversity/analysis/plotting.py`
- Modify: `python_sim/tests/test_plotting.py`

**Interfaces:**
- Consumes: `SimulationResults` branches containing acquisition and tracking records.
- Produces: `plot_cn0(results: SimulationResults) -> Figure`.
- Requires for a successful tracking branch: equal-length one-dimensional `tracking["epoch_start_samples"]` and `tracking["cn0_db_hz"]` arrays.

- [ ] **Step 1: Write failing plot behavior tests**

Add `plot_cn0` to the imports in `tests/test_plotting.py`, then add:

```python
def test_cn0_plot_shows_tracked_modes_and_failed_acquisition_status():
    cfg = default_config()
    cfg.plot.enable = False
    results = run_end_to_end(cfg)

    figure = plot_cn0(results)
    axis = figure.axes[0]

    assert [line.get_label() for line in axis.lines] == ["EGC", "MVDR"]
    assert axis.get_xlabel() == "Tracking time (ms)"
    assert axis.get_ylabel() == "Estimated C/N₀ (dB-Hz)"
    assert axis.get_title() == "Tracking C/N₀ comparison"
    assert any(
        "SINGLE: acquisition failed" in text.get_text()
        for text in axis.texts
    )
    for line in axis.lines:
        assert np.all(np.isfinite(line.get_ydata()))


def test_cn0_plot_returns_annotated_figure_when_no_branch_tracks():
    cfg = default_config()
    cfg.plot.enable = False
    results = run_end_to_end(cfg)
    for branch in results.branches.values():
        branch.acquisition["success"] = False
        branch.tracking = {
            "success": False,
            "status": "acquisition_failed",
            "prompt": np.array([], dtype=complex),
        }

    figure = plot_cn0(results)
    axis = figure.axes[0]

    assert not axis.lines
    assert len(axis.texts) == 3
    assert all("C/N₀ unavailable" in text.get_text() for text in axis.texts)


def test_cn0_plot_rejects_mismatched_tracking_arrays():
    cfg = default_config()
    cfg.plot.enable = False
    results = run_end_to_end(cfg)
    results.branches["mvdr"].tracking["epoch_start_samples"] = np.array([0])

    with pytest.raises(ValueError, match="mvdr.*length"):
        plot_cn0(results)
```

Also add `import numpy as np` if it is not already present.

- [ ] **Step 2: Run the focused plotting tests and verify RED**

Run:

```powershell
D:\Python\Python311\python.exe -W error -m pytest tests/test_plotting.py -k "cn0_plot" -v
```

Expected: collection fails because `plot_cn0` does not exist.

- [ ] **Step 3: Implement `plot_cn0`**

Add the following behavior before `plot_results()` in `analysis/plotting.py`:

```python
def plot_cn0(results: SimulationResults) -> Figure:
    """Plot measured tracking C/N0 for every available branch."""

    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    unavailable_messages: list[str] = []
    for mode, branch in results.branches.items():
        tracking = branch.tracking
        if not branch.acquisition.get("success", False):
            unavailable_messages.append(
                f"{mode.upper()}: acquisition failed — C/N₀ unavailable"
            )
            continue
        if not tracking.get("success", False):
            unavailable_messages.append(
                f"{mode.upper()}: tracking unavailable — C/N₀ unavailable"
            )
            continue

        epoch_starts = np.asarray(tracking["epoch_start_samples"])
        cn0_db_hz = np.asarray(tracking["cn0_db_hz"])
        if (
            epoch_starts.ndim != 1
            or cn0_db_hz.ndim != 1
            or epoch_starts.size != cn0_db_hz.size
        ):
            raise ValueError(f"{mode} C/N0 tracking arrays have inconsistent length")
        finite = np.isfinite(cn0_db_hz)
        if not np.any(finite):
            unavailable_messages.append(
                f"{mode.upper()}: tracking unavailable — C/N₀ unavailable"
            )
            continue
        time_ms = epoch_starts / results.config.dfe.fs_out_hz * 1e3
        axis.plot(time_ms[finite], cn0_db_hz[finite], label=mode.upper())

    for index, message in enumerate(unavailable_messages):
        axis.text(
            0.02,
            0.98 - 0.07 * index,
            message,
            transform=axis.transAxes,
            va="top",
        )
    axis.grid(True)
    axis.set_xlabel("Tracking time (ms)")
    axis.set_ylabel("Estimated C/N₀ (dB-Hz)")
    axis.set_title("Tracking C/N₀ comparison")
    if axis.lines:
        axis.legend(loc="best")
    return figure
```

Use the exact Unicode subscript-zero labels above; source files are UTF-8. Do not plot nonfinite warm-up values.

- [ ] **Step 4: Run plotting tests and verify GREEN**

Run:

```powershell
$env:MPLBACKEND='Agg'
D:\Python\Python311\python.exe -W error -m pytest tests/test_plotting.py -k "cn0_plot" -v
```

Expected: all three C/N₀ plot tests pass with no Matplotlib warnings.

- [ ] **Step 5: Commit the plot**

```powershell
git add -- python_sim/antenna_diversity/analysis/plotting.py python_sim/tests/test_plotting.py
git commit -m "feat: plot standalone cn0 comparison"
```

---

### Task 3: Integrate the sixth figure and explicit table units

**Files:**
- Modify: `python_sim/antenna_diversity/analysis/plotting.py`
- Modify: `python_sim/antenna_diversity/analysis/metrics.py`
- Modify: `python_sim/tests/test_plotting.py`
- Modify: `python_sim/tests/test_end_to_end.py`

**Interfaces:**
- Consumes: `plot_cn0(results)` from Task 2 and existing `final_cn0_db_hz` metric records.
- Produces: `plot_results()` mapping with six keys, including `"cn0_comparison"`.
- Produces: metric table header `FinalCn0DbHz`.

- [ ] **Step 1: Update tests first for six figures and explicit units**

Rename `test_plot_results_produces_five_named_figures` to `test_plot_results_produces_six_named_figures` and require:

```python
assert set(figures) == {
    "rf_dfe_spectra",
    "combined_spectra",
    "array_response",
    "acquisition_search",
    "tracking_comparison",
    "cn0_comparison",
}
```

Extend `test_metric_formatter_includes_every_branch_and_tracking_state` in `tests/test_end_to_end.py`:

```python
assert "FinalCn0DbHz" in table
```

Extend `test_cli_avoids_show_warning_with_noninteractive_backend` in `tests/test_plotting.py`:

```python
assert "cn0_comparison:" in completed.stdout
```

- [ ] **Step 2: Run integration tests and verify RED**

Run:

```powershell
$env:MPLBACKEND='Agg'
D:\Python\Python311\python.exe -W error -m pytest tests/test_plotting.py::test_plot_results_produces_six_named_figures tests/test_plotting.py::test_cli_avoids_show_warning_with_noninteractive_backend tests/test_end_to_end.py::test_metric_formatter_includes_every_branch_and_tracking_state -v
```

Expected: failures show the missing `cn0_comparison` key/output and old `FinalCn0` header.

- [ ] **Step 3: Add the figure and rename the metric column**

Append the sixth mapping entry in `plot_results()`:

```python
"cn0_comparison": plot_cn0(results),
```

Change the metrics column definition in `analysis/metrics.py` to:

```python
("FinalCn0DbHz", "final_cn0_db_hz", 13),
```

No CLI source change is required because both entry points already iterate over the `plot_results()` mapping returned by the plotting module.

- [ ] **Step 4: Run focused integration and complete regression suites**

Run:

```powershell
$env:MPLBACKEND='Agg'
D:\Python\Python311\python.exe -W error -m pytest tests/test_tracking.py tests/test_plotting.py tests/test_end_to_end.py -v
D:\Python\Python311\python.exe -W error -m pytest -q
```

Expected: all focused and full-suite tests pass with no warnings.

- [ ] **Step 5: Run both real CLI entry points**

Run:

```powershell
$env:MPLBACKEND='Agg'
D:\Python\Python311\python.exe -W error run_end_to_end.py
D:\Python\Python311\python.exe -W error run_legacy_ula.py
```

Expected: both commands exit zero, the table includes `FinalCn0DbHz`, and each `Saved figures` section includes `cn0_comparison` with a real PNG path.

- [ ] **Step 6: Commit the integration**

```powershell
git add -- python_sim/antenna_diversity/analysis/plotting.py python_sim/antenna_diversity/analysis/metrics.py python_sim/tests/test_plotting.py python_sim/tests/test_end_to_end.py
git commit -m "feat: include cn0 in standard reports"
```

---

## Final Review and Delivery

- [ ] Run `git diff --check` and verify the worktree contains no uncommitted implementation changes.
- [ ] Review the complete diff from `17380dc` through the final implementation commit against `docs/superpowers/specs/2026-08-13-cn0-comparison-design.md`.
- [ ] Push `feature/antenna-diversity`; verify the existing GitHub PR #1 updates to the final commit.
- [ ] Do not merge PR #1 without separate user authorization.
