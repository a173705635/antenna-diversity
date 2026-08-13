# Mate 70 理想角点空域抗干扰仿真架构

![Mate 70 理想角点空域抗干扰仿真架构](docs/simulation_architecture.svg)

图片文件：[`docs/simulation_architecture.svg`](docs/simulation_architecture.svg) 是可编辑矢量主文件；[`docs/simulation_architecture.png`](docs/simulation_architecture.png) 用于报告和演示；实现细节见 [`DESIGN.md`](DESIGN.md)。

## 一句话定位

这是一个以 HUAWEI Mate 70 外形尺寸为几何边界、以 2–4 个理想角点为可选相位中心的 GPS L1 C/A 端到端研究链，用于比较 SINGLE、EGC 和 MVDR 在 0–3 个静态 CW 干扰下的空间响应、捕获与跟踪结果。

## 中央数据路径

```text
GPS L1 C/A + 0–3 CW interferers
→ HUAWEI Mate 70 ideal 2–4 selected corners
→ isotropic element-pattern interface × 3-D manifold
→ RF / complex ADC
→ coherent DFE
→ SINGLE / EGC / loaded MVDR
→ Doppler × code-phase ACQ
→ gated DLL / FLL-assisted PLL TRK
→ branch metrics, spatial scans, sequential Monte Carlo
```

RF 到 DFE 始终保留 `(channels, samples)` 形状，因此同一处理链可接受 2、3 或 4 个激活角点。三种合并器输出单通道样本后共享同一捕获与跟踪配置；只有捕获成功的分支才进入跟踪。

## 当前已建模

- Mate 70 外形尺寸 160.9 × 75.9 × 7.8 mm，以及中心原点的理想角点坐标；
- 任意不重复的 2–4 角点子集，共 11 种稳定顺序组合；
- 手机坐标中的方位角/仰角和位置点积构成的 3-D 阵列流形；
- `ElementPattern` 接口及其首版 `IsotropicElementPattern`；
- 单星 GPS L1 C/A、码延迟、载波/码 Doppler、等效复数 IF；
- 0–3 个静态 CW 干扰，每个具有独立方向、功率、频偏、相位与起止时间门；
- RF 级联增益/NF、热噪声、限幅、复数 ADC，以及相干 DDC/FIR/抽取/重量化；
- SINGLE、EGC、loaded MVDR，FFT 捕获和 E/P/L DLL + FLL-assisted PLL；
- 功率分解、输出 SINR、协方差诊断、逐干扰增益的解析空间指标和扫描；
- 顺序、种子有序、严格 JSON 的 Monte Carlo 汇总。

## 当前未建模

- **真实天线位置**：角点是理想相位中心接口，不声称对应 Mate 70 内部天线；
- **真实测量方向图**：当前各向同性增益 `1+0j` 只是可替换接口的首版；
- 互耦、极化、人体/机身遮挡与平台散射；
- 多径、NLOS、运动和动态 DOA；
- 宽带、扫频、脉冲或其他非 CW 干扰；
- 多颗卫星、多 PRN 并发、导航数据、位/帧同步和 PVT；
- 在线 DOA、鲁棒导向矢量估计和 MVDR 滑窗实时更新；
- 完整硬件非理想，例如相位噪声、I/Q 失衡、时钟抖动和逐级压缩。

## 对外表述边界

推荐：

> 已构建一套基于 Mate 70 外形理想角点的 GPS L1 C/A 空域抗干扰仿真，覆盖 2–4 通道三维阵列流形、0–3 个静态 CW 干扰、RF/DFE、SINGLE/EGC/MVDR、捕获跟踪、解析扫描和顺序 Monte Carlo。

避免称为“Mate 70 真实天线仿真”或宣称产品抗干扰性能，因为真实位置、方向图、互耦、极化和传播测量均未输入模型。

## 两条基线

`default_config()` 是手机理想四角默认；`legacy_ula_config()` 与 `run_legacy_ula.py` 保留原始双阵元半波长 ULA 结论。两者并存，使新手机接口的扩展不改变旧基线的含义。
