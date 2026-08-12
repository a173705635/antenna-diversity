# Python Combined-Branch Spectra Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增一张独立的 `combined_spectra.png`，用共享频率和幅度尺度显示 SINGLE、EGC、MVDR 三路天线合并后的绝对幅度频谱。

**Architecture:** 在现有 `analysis.plotting` 中增加一个专用绘图函数，直接读取三个 `BranchResult.combined` 数组，统一计算 Hamming 窗双边 FFT，并让三个纵向子图共享坐标范围。现有 `plot_results` 和 `save_figures` 继续负责汇总与保存，不修改信号处理算法。

**Tech Stack:** Python 3.11+、NumPy、Matplotlib、pytest。

## Global Constraints

- 输出文件固定为 `python_sim/results/combined_spectra.png`。
- 三个子图顺序固定为 SINGLE、EGC、MVDR，并共享 x/y 范围。
- 使用相同的起始采样段、FFT 长度和 Hamming 窗。
- 显示绝对幅度 `20*log10(abs(FFT))`，不得逐分支归一化。
- 标记真实目标 Doppler 和启用状态下的干扰频偏。
- 不修改 RF、DFE、合并、捕获、跟踪或 MATLAB 代码。
- 当前目录不是 Git 仓库，不自行初始化仓库；以 pytest 和图片目视检查作为完成证据。

---

### Task 1: Add and verify the combined-branch spectrum figure

**Files:**
- Modify: `python_sim/tests/test_plotting.py`
- Modify: `python_sim/antenna_diversity/analysis/plotting.py`
- Modify: `python_sim/antenna_diversity/analysis/__init__.py`
- Modify: `python_sim/README.md`
- Generate: `python_sim/results/combined_spectra.png`

**Interfaces:**
- Consumes: `SimulationResults.branches[mode].combined` and `SimulationResults.dfe_info["fs_out_hz"]`.
- Produces: `plot_combined_spectra(results: SimulationResults) -> Figure`.
- Extends: `plot_results(results)` with key `combined_spectra`.

- [ ] **Step 1: Write a failing plotting test**

Add this behavior test to `tests/test_plotting.py`:

```python
def test_combined_spectra_share_axes_and_show_all_modes(tmp_path):
    cfg = default_config()
    cfg.plot.enable = False
    results = run_end_to_end(cfg)

    figure = plot_combined_spectra(results)
    paths = save_figures({"combined_spectra": figure}, tmp_path)
    main_axes = figure.axes

    assert len(main_axes) == 3
    assert [axis.get_title() for axis in main_axes] == [
        "SINGLE combined output",
        "EGC combined output",
        "MVDR combined output",
    ]
    assert all(len(axis.lines) >= 3 for axis in main_axes)
    assert all(axis.get_xlim() == main_axes[0].get_xlim() for axis in main_axes)
    assert all(axis.get_ylim() == main_axes[0].get_ylim() for axis in main_axes)
    assert paths["combined_spectra"].is_file()
    assert paths["combined_spectra"].stat().st_size > 0
    plt.close(figure)
```

Update the existing figure-set assertion to require these five keys:

```python
assert set(figures) == {
    "rf_dfe_spectra",
    "combined_spectra",
    "array_response",
    "acquisition_search",
    "tracking_comparison",
}
```

- [ ] **Step 2: Run the focused tests and verify the red state**

Run:

```powershell
cd F:\antenna_diversity\python_sim
$env:MPLBACKEND='Agg'
python -m pytest tests/test_plotting.py -v
```

Expected: collection fails because `plot_combined_spectra` is not defined, or the five-key assertion fails because `plot_results` does not include it.

- [ ] **Step 3: Implement one shared spectrum calculator and the three-row figure**

Refactor the existing private plotting helper only enough to share the exact spectrum calculation:

```python
def _calculate_spectrum(x: np.ndarray, fs_hz: float, n_fft: int):
    samples = np.asarray(x).reshape(-1)[:n_fft]
    window = 0.54 - 0.46 * np.cos(
        2.0 * np.pi * np.arange(n_fft) / max(1, n_fft - 1)
    )
    spectrum = np.fft.fftshift(np.fft.fft(samples * window))
    frequency_mhz = np.fft.fftshift(
        np.fft.fftfreq(n_fft, 1.0 / fs_hz)
    ) / 1e6
    magnitude_db = 20.0 * np.log10(
        np.maximum(np.abs(spectrum), np.finfo(float).eps)
    )
    return frequency_mhz, magnitude_db
```

The public function must validate nonempty branches and one-dimensional nonempty combined arrays, choose one common `n_fft`, calculate all spectra before plotting, derive one global y range, and draw reference lines:

```python
def plot_combined_spectra(results: SimulationResults) -> Figure:
    mode_names = list(results.branches)
    if not mode_names:
        raise ValueError("results must contain at least one combined branch")
    if mode_names != ["single", "egc", "mvdr"]:
        raise ValueError("combined spectrum requires single, egc, and mvdr branches")

    combined = [np.asarray(results.branches[mode].combined) for mode in mode_names]
    if any(values.ndim != 1 or values.size == 0 for values in combined):
        raise ValueError("each combined branch must be a nonempty one-dimensional array")
    n_fft = min(65_536, *(values.size for values in combined))
    spectra = [
        _calculate_spectrum(values, results.dfe_info["fs_out_hz"], n_fft)
        for values in combined
    ]
```

Use `plt.subplots(3, 1, sharex=True, sharey=True, figsize=(10, 9), constrained_layout=True)`. Plot the spectrum plus the green target-Doppler and red jammer-frequency lines on every axis. Add the reference-line legend only to the first axis.

- [ ] **Step 4: Export and register the new figure**

Add `plot_combined_spectra` to `analysis/__init__.py` and insert this entry in `plot_results`:

```python
"combined_spectra": plot_combined_spectra(results),
```

- [ ] **Step 5: Run focused and full regression tests**

Run:

```powershell
cd F:\antenna_diversity\python_sim
$env:MPLBACKEND='Agg'
python -m pytest tests/test_plotting.py -v -W error
python -m pytest -q -W error
```

Expected: all plotting tests and the full suite pass without warnings.

- [ ] **Step 6: Update usage documentation**

Add `combined_spectra.png` to the README result list and explain that its three shared-scale panels compare the absolute post-combining spectra and reveal the MVDR jammer null.

- [ ] **Step 7: Generate and inspect the figure**

Run:

```powershell
cd F:\antenna_diversity\python_sim
$env:MPLBACKEND='Agg'
python -W error run_end_to_end.py
```

Expected: the terminal lists five saved figures. Inspect `results/combined_spectra.png` and confirm three readable panels, common axes, target and jammer markers, and visible spectral differences without clipping.

- [ ] **Step 8: Recheck MATLAB immutability**

Recompute the existing 38-file aggregate hash and require:

```text
0C5CB2DAEB5739BD951AED810B94E561FDC1A87898AAFA4D74CDC46E5F18DDF0
```
