# Python 双天线 GPS L1 C/A RF→DFE→TRK 仿真

本目录是 `matlab_sim` 中 GPS L1 C/A 链路的 Python 原生实现。MATLAB 参考代码保持不变；Galileo E5/AltBOC 不属于本阶段迁移范围。

## 链路结构

```text
GPS L1 C/A 与窄带 CW 干扰
→ 双天线空间响应和通道幅相误差
→ RF 级联增益、Friis 噪声、限幅、复数 ADC
→ DDC、低通 FIR、群延迟对齐、4 倍抽取、输出量化
→ single / EGC / MVDR
→ FFT 多普勒×码相位捕获
→ Early/Prompt/Late DLL、FLL 辅助 PLL 跟踪
→ 指标表、频谱、方向图、捕获热力图和跟踪曲线
```

模型使用复数等效中频，不直接以 1.57542 GHz 对载波采样。多通道数组始终采用 `(天线数, 采样点数)`，合并后的单路数据采用 `(采样点数,)`。

## 环境与运行

当前代码要求 Python 3.11 或更高版本，并使用 NumPy、SciPy、Matplotlib 和 pytest：

```powershell
cd F:\antenna_diversity\python_sim
python -m pip install -e ".[test]"
python -m pytest -v
python run_end_to_end.py
```

运行脚本会在终端显示 single、EGC 和 MVDR 的指标，并把图保存到 `python_sim/results/`：

- `rf_dfe_spectra.png`
- `combined_spectra.png`
- `array_response.png`
- `acquisition_search.png`
- `tracking_comparison.png`

`combined_spectra.png` 使用三个共享坐标范围的子图，依次显示 SINGLE、EGC 和 MVDR 合并后的绝对幅度频谱。绿色虚线标记目标 Doppler，红色虚线标记 DDC 后的干扰频偏；共享尺度可以直接观察 MVDR 对窄带干扰谱线的抑制程度。

无图形界面运行时可使用：

```powershell
$env:MPLBACKEND='Agg'
python run_end_to_end.py
```

## 修改参数

```python
from antenna_diversity.config import default_config
from antenna_diversity.pipeline import run_end_to_end

cfg = default_config()
cfg.jammer.power_dbm = -95.0
cfg.jammer.angle_deg = 45.0
cfg.array.channel_amplitude_error_db[1] = 0.5
cfg.array.channel_phase_error_deg[1] = 5.0
cfg.diversity.diagonal_loading_factor = 1e-3
cfg.plot.enable = False

results = run_end_to_end(cfg)
```

常用扫描入口包括：

- `cfg.jammer.power_dbm`、`offset_hz`、`angle_deg`
- `cfg.array.spacing_m`、`positions_m` 和通道幅相误差
- `cfg.rf.stages`、`adc_bits`、`adc_vpp`
- `cfg.dfe.cutoff_hz`、`num_taps`、`output_bits`
- `cfg.diversity.diagonal_loading_factor`
- `cfg.acquisition.doppler_bins_hz`、积分时长和阈值
- `cfg.tracking` 中的 DLL、FLL、PLL 参数

修改阵元间距时，应同时更新 `cfg.array.positions_m`；`spacing_m` 是便于阅读和记录的参数，导向矢量实际使用 `positions_m`。

## 默认结果的含义

默认场景故意设置了很强的窄带干扰。single 和 EGC 通常无法通过捕获阈值；MVDR 在已知目标导向矢量约束下形成空间零陷，从而能够捕获并跟踪目标。这个用例说明 MVDR 在该场景中的干扰抑制作用，不表示 MVDR 在导向矢量失配、协方差估计不足或所有传播环境下都必然更优。

## MATLAB 与 Python 的数值约定

- Python 内部和输出的码相位采样索引均为零基。
- MATLAB `firstValid = groupDelay + 1` 对应 Python 的 `filtered[:, group_delay:]`。
- ADC 与 DFE 量化使用“半整数远离零”的 MATLAB 舍入规则，而不是 NumPy 默认的银行家舍入。
- 固定随机种子在 Python 内部可复现，但 NumPy 与 MATLAB 的噪声样本不要求逐点相同。
- 跨语言一致性通过确定性数学测试、估计误差容限和端到端物理结论验证。
