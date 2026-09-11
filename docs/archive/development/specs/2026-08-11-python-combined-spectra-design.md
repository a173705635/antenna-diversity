# Python 天线合并后频谱图设计

## 目标

在现有 Python GPS L1 C/A 仿真中新增一张独立的天线合并后频谱图，直观比较 SINGLE、EGC、MVDR 三个分支对目标信号、噪声和窄带干扰的影响。

输出文件为：

```text
python_sim/results/combined_spectra.png
```

## 布局

图片使用三个上下排列的子图：

1. `SINGLE combined output`
2. `EGC combined output`
3. `MVDR combined output`

三个子图共享频率轴和幅度轴范围。这样既避免三条噪声状频谱互相遮挡，又能直接比较不同分支的绝对频谱幅度。

## 频谱计算约定

- 输入数据依次取自 `results.branches[mode].combined`。
- 采样率取自 `results.dfe_info["fs_out_hz"]`。
- 每个分支使用相同的 FFT 长度：`min(65536, 最短分支采样数)`。
- 每个分支截取相同数量的起始采样点。
- 使用与现有 RF/DFE 频谱图相同的 Hamming 窗。
- 使用 `fftshift(fft(x * window))` 计算双边频谱。
- 幅度采用 `20*log10(max(abs(spectrum), eps))`，不进行逐分支峰值归一化。
- 横轴单位为 MHz，范围由 DFE 输出采样率决定。
- 三个子图使用全部分支频谱的全局最小值和最大值设置相同的 y 轴范围，并留少量显示余量。

不对频谱进行单边化、功率谱密度标定或 RBW 归一化，因为本图的用途是比较当前离散仿真中三个合并分支的相对输出，而不是充当校准频谱分析仪。

## 参考频率标记

每个子图增加两条参考线：

- 绿色虚线：目标残余 Doppler，`truth["doppler_hz"] / 1e6` MHz。
- 红色虚线：DDC 后干扰频偏，`cfg.jammer.offset_hz / 1e6` MHz；仅在干扰启用时绘制。

只在第一个子图显示参考线图例，避免重复图例占据绘图区。

## 接口调整

在 `antenna_diversity.analysis.plotting` 中新增：

```python
plot_combined_spectra(results: SimulationResults) -> Figure
```

并将它加入：

- `analysis.__init__` 的公开导出。
- `plot_results(results)` 返回字典，键名为 `combined_spectra`。
- `save_figures` 的现有保存流程，无需修改该函数。
- README 的结果图片列表和绘图说明。

现有 `rf_dfe_spectra.png` 保持原样，不把三路合并频谱塞入已有三行图。

## 输入校验

如果 `results.branches` 为空，抛出带明确信息的 `ValueError`。如果任一分支的 `combined` 不是非空一维数组，同样抛出 `ValueError`，避免空图或维度广播掩盖上游错误。

## 测试与验收

先新增失败测试，再实现绘图函数。测试要求：

- `plot_combined_spectra` 返回 Matplotlib `Figure`。
- 图片包含三个主坐标轴。
- 三个主坐标轴标题分别包含 SINGLE、EGC、MVDR。
- 每个主坐标轴至少包含一条频谱曲线、目标 Doppler 参考线和启用干扰时的干扰参考线。
- 三个主坐标轴的 x/y 范围一致。
- `plot_results` 的返回键由四个增加为五个，包含 `combined_spectra`。
- `save_figures` 能生成非空的 `combined_spectra.png`。
- 全部现有 Python 测试继续通过，并把警告视为错误。
- 重新运行端到端脚本后，对 `combined_spectra.png` 做目视检查，确认标题、频率标记、共享尺度和三路差异清楚可读。

## 不变范围

- 不修改 RF、DFE、合并权值、捕获或跟踪算法。
- 不修改 MATLAB 参考实现。
- 不更改已有四张图片的文件名和含义。
