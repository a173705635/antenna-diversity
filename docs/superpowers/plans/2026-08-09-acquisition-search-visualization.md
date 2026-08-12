# Acquisition Search Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an intuitive 2-D acquisition search map and best-Doppler 1-D correlation slice for every diversity branch.

**Architecture:** Keep acquisition numerical processing in `receiver.acquire` and place all figure creation in a new `analysis.plot_acquisition` function. Extend the acquisition result with the second-peak location so both peaks can be marked without recomputing detection logic in the plotting layer.

**Tech Stack:** MATLAB R2023a, base MATLAB graphics, existing script-based regression tests.

## Global Constraints

- Do not change the existing FFT acquisition search calculation or detection threshold semantics.
- Do not create figures when `cfg.plot.enable` is false.
- Do not require Signal Processing Toolbox.
- Use normalized dB values with a -40～0 dB display range.
- The workspace is not a Git repository, so verification replaces commit steps.

---

### Task 1: Acquisition Peak Metadata

**Files:**
- Modify: `matlab_sim/+receiver/acquire.m`
- Modify: `matlab_sim/tests/test_acquisition.m`

**Interfaces:**
- Produces: `acq.secondPeakCodePhaseSamples` and `acq.exclusionSamples`.

- [ ] Add assertions that the new fields exist and that the second peak lies outside the circular exclusion interval.
- [ ] Run `test_acquisition` and observe the expected missing-field failure.
- [ ] Capture the second-peak index when searching the masked best-Doppler row and return both fields.
- [ ] Run `test_acquisition` and require exit code 0.

### Task 2: Search Visualization

**Files:**
- Create: `matlab_sim/+analysis/plot_acquisition.m`
- Create: `matlab_sim/tests/test_acquisition_plot.m`

**Interfaces:**
- Produces: `fig = analysis.plot_acquisition(results)`.

- [ ] Create a hidden-figure test using a deterministic high-SNR acquisition result; assert that the returned handle, heatmap image and line objects exist.
- [ ] Run the test and observe the expected undefined-function failure.
- [ ] Implement an `Nmode × 2` tiled layout with normalized dB heatmaps, best-bin slices, peak markers and result annotations.
- [ ] Run the test and require exit code 0.

### Task 3: Integration and Verification

**Files:**
- Modify: `matlab_sim/+analysis/plot_results.m`
- Modify: `matlab_sim/tests/run_all_tests.m`
- Modify: `matlab_sim/README.md`

- [ ] Replace the old acquisition one-dimensional comparison block with `analysis.plot_acquisition(results)`.
- [ ] Add the plotting test to the aggregate runner and document standalone usage.
- [ ] Run all tests, run `checkcode` on every `.m` file, and run `run_end_to_end` with hidden figures.
- [ ] Require zero test failures, zero `checkcode` messages, and a valid acquisition figure.
