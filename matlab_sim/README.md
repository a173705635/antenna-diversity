# MATLAB 双天线 GNSS RF→DFE→TRK 仿真

本目录提供一套可运行的 GPS L1 C/A 双天线端到端基线：

```text
目标卫星与窄带干扰
→ 双天线空间相位
→ RF 增益、级联 NF、热噪声、压缩与 ADC
→ DDC、低通 FIR、4 倍抽取与低位宽量化
→ single / EGC / MVDR
→ FFT 捕获
→ 闭环 DLL、FLL、PLL 跟踪
```

## 建模边界

代码使用复数等效中频，不以 1.57542 GHz 载波频率直接采样。阵列相位仍按 GPS L1 波长计算，因此能够研究天线间距、来波方向、通道幅相误差、干扰和分集合并的影响。

第一版包含一颗 GPS L1 C/A 卫星、两个阵元和一个连续波窄带干扰。暂不包含多星、导航电文解调、观测量、定位、多径和宽带 STAP。

## 运行

在 MATLAB R2023a 中：

```matlab
cd('F:/antenna_diversity/matlab_sim');
addpath(genpath(pwd));
results = run_end_to_end();
```

默认绘图包含捕获搜索结果：每种合并方式左侧为“码相位 × Doppler”二维归一化功率热力图，右侧为最佳 Doppler 下的一维码相位切片，并标出主峰、第二峰和峰值比。

已有结果也可以单独重新绘制捕获图：

```matlab
fig = analysis.plot_acquisition(results);
```

无图批处理：

```matlab
cfg = default_config();
cfg.plot.enable = false;
results = run_end_to_end(cfg);
```

全部测试：

```matlab
cd('F:/antenna_diversity/matlab_sim');
addpath(genpath(pwd));
run('tests/run_all_tests.m');
```

命令行运行：

```powershell
& 'D:\MATLAB\R2023a\bin\matlab.exe' -batch "cd('F:/antenna_diversity/matlab_sim'); addpath(genpath(pwd)); run('tests/run_all_tests.m');"
```

## 主要输出

`results` 包含：

- `truth`：目标和干扰导向矢量、真实 Doppler、码延迟；
- `rf.info`：总增益、级联 NF、ADC LSB、饱和比例；
- `dfe.info`：采样率、FIR 系数、群时延和量化尺度；
- `branches.single/egc/mvdr`：合并权值、捕获结果和跟踪时序；
- `metrics.summaryTable`：三种模式的捕获峰值比、干扰方向增益、跟踪状态和稳态误差。

## 推荐扫参入口

- `cfg.jammer.powerDbm`、`angleDeg`、`offsetHz`；
- `cfg.array.spacingM` 和通道幅相误差；
- `cfg.rf.stages`、`adcBits`、`adcVpp`；
- `cfg.dfe.outputBits`、`cutoffHz`、`numTaps`；
- `cfg.diversity.diagonalLoadingFactor`；
- 捕获 Doppler 步进与非相干累加时间；
- DLL/PLL/FLL 增益。

## 分集与波束形成的区别

EGC 是典型的相位对齐分集合并；MVDR 则利用阵列空间自由度形成干扰零陷。两者都在同一 RF/DFE 数据上比较，但研究问题不同：前者关注多支路增益，后者关注带约束的空间抑制。

设计说明见 [链路设计](../docs/superpowers/specs/2026-08-04-matlab-rf-dfe-trk-chain-design.md)，实施步骤见 [实施计划](../docs/superpowers/plans/2026-08-04-matlab-rf-dfe-trk-chain.md)。

## Galileo E5 全带宽联合跟踪（独立验证）

新增 E5 验证链与上述 GPS L1/天线分集链完全独立。它不调用阵列模型、`diversity.combine`、原 RF/DFE、GPS 捕获或 GPS 跟踪函数。

默认配置为：

- Galileo SVID 1；
- 正交下变频后的复数 I/Q，采样率 132 MHz；
- LO 为 1188 MHz，E5 中心低中频为 `1191.795-1188=3.795 MHz`；
- E5a 和 E5b 相对数字零频的中心分别为 -11.550 MHz 和 19.140 MHz；
- 完整 E5a-I、E5a-Q、E5b-I、E5b-Q AltBOC(15,10)；
- 接收端已知 E5a-I/E5b-I 数据符号，四分量共同参与相干跟踪；
- 直接对完整 AltBOC 波形运行公共 DLL/FLL/PLL，不先分离 E5a/E5b；
- 不包含捕获，入口使用带可配置误差的 Doppler、码相位和载波相位初值。

运行：

```matlab
cd('F:/antenna_diversity/matlab_sim');
addpath(genpath(pwd));
cfg = e5_default_config();
cfg.plot.enable = false;
results = run_e5_fullband_tracking(cfg);
```

独立 E5 测试：

```matlab
run('tests/run_e5_tests.m');
```

`results` 包含完整复数低中频输入、真实参数、已知数据符号、初始化状态、E/P/L 相关值、DLL/FLL/PLL 历史、公共 Doppler/码 NCO、旁峰监视与最终误差摘要。

66 Msps 验证以及把完整 AltBOC 分离成 E5a/E5b 两路复基带后的对比属于后续独立扩展，不包含在当前入口中。E5 设计见 [设计说明](../docs/superpowers/specs/2026-08-08-galileo-e5-fullband-joint-tracking-design.md)，实施步骤见 [实施计划](../docs/superpowers/plans/2026-08-08-galileo-e5-fullband-joint-tracking.md)。
