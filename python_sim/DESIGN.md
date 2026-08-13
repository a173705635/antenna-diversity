# GPS L1 C/A 手机角点空域抗干扰仿真：设计与实现

> 当前默认模型：HUAWEI Mate 70 外形上的四个**理想角点相位中心**，不是已知真实天线。阵元方向图首版为各向同性接口。

![端到端信号流与模型边界](docs/antenna_diversity_chain.svg)

图源包括 [`docs/antenna_diversity_chain.svg`](docs/antenna_diversity_chain.svg)、[`docs/antenna_diversity_chain.mmd`](docs/antenna_diversity_chain.mmd) 和 [`docs/antenna_diversity_chain.png`](docs/antenna_diversity_chain.png)；对外范围图见 [`docs/simulation_architecture.svg`](docs/simulation_architecture.svg)，视觉规范见 [`docs/diagram-style.md`](docs/diagram-style.md)。

## 1. 研究问题和边界

仿真比较 `single`、`egc`、`mvdr` 三个分支。比较不仅限于阵列响应：每个合并输出经过相同的 Doppler–码相位捕获，捕获成功后再进入闭环跟踪。因此问题是“空间处理后的信号能否被 GNSS 基带捕获并跟踪”。

L1 载频 1575.42 MHz 用于波长、空间相位和码 Doppler 比例；实际生成并采样的是 4.092 MHz 等效复数 IF。默认强 CW 案例中可重复看到 MVDR 空域抑制和后续捕获/跟踪成功，但这只适用于当前理想几何、静态方向和已知目标导向矢量，不是实际手机性能结论。

## 2. 配置族

- `default_config()` / `mate70_default_config()`：Mate 70 外形 160.9 × 75.9 × 7.8 mm，默认 `("TL", "TR", "BL", "BR")` 四个理想共面角点。
- `PhoneArrayConfig.active_antennas`：任意不重复的 2–4 角点子集；[`antenna_diversity/array/geometry.py`](antenna_diversity/array/geometry.py) 枚举全部 11 种组合。
- `legacy_ula_config()`：原始双阵元、L1 半波长 ULA，供旧结论回归。
- `interferers`：0–3 个 `CWInterferer`；每个有独立方向、功率、频偏、初相、`start_ms` 和 `stop_ms`。

手机阵列和 legacy ULA 都进入相同的 `(channels, samples)` RF/DFE/合并链。`jammer` 字段只服务 legacy 配置；手机配置读取 `interferers`。

## 3. 手机坐标和 3-D 阵列流形

手机中心是原点；`+x` 向右、`+y` 向顶端、`+z` 垂直屏幕向外。方位角 `az` 从 `+x` 朝 `+y` 逆时针增加，仰角 `el` 从手机平面向 `+z` 增加：

\[
\mathbf u(az,el)=
[\cos(el)\cos(az),\ \cos(el)\sin(az),\ \sin(el)]^T.
\]

位置 \(\mathbf p_m\) 的空间响应是

\[
a_m(az,el,f)=g_m(az,el,f)
\exp\left(-j\frac{2\pi f}{c}\mathbf p_m^T\mathbf u\right).
\]

`ElementPattern.complex_gain()` 是可替换的复数方向图协议；当前 `IsotropicElementPattern` 对每个阵元返回 `1+0j`。因为默认理想角点共面且 `z=0`，天顶方向的点积全部为零，空间相位相同。

角点坐标只是按手机外形边界生成的接口初值。代码没有输入或推断真实天线位置、朝向、增益或相位方向图。

## 4. 场景与干扰

GPS L1 C/A 复数 IF 目标为

\[
s[n]=\sqrt{P_sR}\,c[n]
\exp\{j[2\pi(f_{IF}+f_D)n/f_s+\phi_0]\}.
\]

第 \(k\) 个开启的 CW 干扰为

\[
j_k[n]=w_k[n]\sqrt{P_kR}
\exp\{j[2\pi(f_{IF}+\Delta f_k)n/f_s+\phi_k]\},
\]

其中 `w_k[n]` 由各自的起止时间门产生。场景输出分别保留每个干扰的 truth 分量；零干扰时保持合法的空集合。阵元通道还可配置真实/估计幅相误差，用于导向矢量失配研究。

## 5. RF、ADC 和 DFE

[`antenna_diversity/rf/frontend.py`](antenna_diversity/rf/frontend.py) 计算 Friis 级联增益与噪声系数，在输入参考温度/带宽下加入热噪声，再执行 I/Q 限幅和复数 ADC 量化。RF 采样率为 16.368 Msps。

[`antenna_diversity/dfe/processing.py`](antenna_diversity/dfe/processing.py) 对所有通道使用公共 NCO 做 −4.092 MHz DDC，随后经过 129-tap Hamming-windowed sinc FIR、64 样点群延迟对齐、4 倍抽取和 8-bit 重量化。输出率为 4.092 Msps，即 4 samples/chip。公共处理保持通道间相对相位。

## 6. SINGLE、EGC 和 MVDR

三个合并器接受 2、3 或 4 通道并产生一维样本：

- SINGLE 选第一激活通道；
- EGC 按假设目标导向矢量做相位对齐和等增益合并；
- MVDR 从 DFE 样本协方差计算 loaded 权值。

MVDR 使用

\[
\mathbf w=\frac{(\widehat{\mathbf R}+\delta\mathbf I)^{-1}\mathbf a_0}
{\mathbf a_0^H(\widehat{\mathbf R}+\delta\mathbf I)^{-1}\mathbf a_0},
\qquad
\delta=\alpha\,\mathrm{tr}(\widehat{\mathbf R})/M,
\]

并保持 \(\mathbf w^H\mathbf a_0=1\) 的目标约束。当前端到端权值对整段 DFE 数据一次计算，不是在线滑窗 MVDR。

## 7. 捕获与跟踪

捕获对每个 Doppler bin 做载波擦除，以 FFT 相关搜索全部 C/A 码相位，并用排除主峰邻域后的 peak/second-peak 比值判决。默认是 1 ms coherent × 4 noncoherent，门限 2.5。

捕获失败的分支明确记录 `acquisition_failed`。成功分支进入 1 ms epoch 的 Early/Prompt/Late DLL 与 FLL-assisted PLL，输出载波频率、码相位、鉴别器、prompt 和 C/N₀ 记录。

## 8. 分析与扫描

端到端结果由 [`antenna_diversity/analysis/metrics.py`](antenna_diversity/analysis/metrics.py) 压缩为分支指标。绘图覆盖 RF/DFE 与合并频谱、手机阵列方位/仰角响应（标出目标及所有干扰）、捕获热力图和跟踪曲线。

[`antenna_diversity/experiments/spatial_scan.py`](antenna_diversity/experiments/spatial_scan.py) 是不生成时域样本的快速解析路径。它报告目标/逐干扰方向增益、最差干扰增益、输出信号/干扰/噪声功率、输出 SINR、权值范数、白噪声增益、协方差秩/条件数、加载量和快拍策略。入口包括：

- `compare_corner_combinations()`：全部 11 个 2–4 角点组合；
- `scan_interferer_directions()` 和 `scan_assumed_target_directions()`；
- `scan_interferer_powers()`、`scan_diagonal_loading()`、`scan_position_errors()`。

所有扫描深拷贝配置，并保持调用者提供的参数顺序。

## 9. 顺序 Monte Carlo

[`antenna_diversity/experiments/monte_carlo.py`](antenna_diversity/experiments/monte_carlo.py) 的 `run_monte_carlo(cfg, seeds)` 按传入顺序逐个运行完整链路。每次都深拷贝配置、设置 seed 并关闭绘图；API 不暴露并行参数，以保证随机状态和输出排序确定。

`MonteCarloRun` 只保存 seed 和每分支的标量记录。`MonteCarloResult` 汇总捕获率、跟踪率，以及 acquisition metric、最终 Doppler 误差、输出 SINR、最差干扰增益的 branch-wise 列表。若端到端指标不存在，列表位置明确为 `None`。`to_dict()` 不含 NumPy 数组，可用 `json.dumps(..., allow_nan=False)`。

## 10. 可重复实验入口

```python
import numpy as np
from antenna_diversity.config import default_config, legacy_ula_config
from antenna_diversity.experiments import (
    compare_corner_combinations, run_monte_carlo,
    scan_interferer_directions,
)
from antenna_diversity.pipeline import run_end_to_end

phone = default_config()
phone.array.active_antennas = ("TL", "TR", "BR")
phone.plot.enable = False
result = run_end_to_end(phone)

all_eleven = compare_corner_combinations(phone, "mvdr")
grid = scan_interferer_directions(
    phone, 0, np.arange(0, 360, 30), np.arange(0, 91, 15), "mvdr"
)
mc = run_monte_carlo(phone, seeds=(101, 102, 103))

legacy = legacy_ula_config()
legacy.plot.enable = False
legacy_result = run_end_to_end(legacy)
```

命令行入口是 [`run_end_to_end.py`](run_end_to_end.py) 和 [`run_legacy_ula.py`](run_legacy_ula.py)。

## 11. 明确未建模项

以下均不在当前结论中：真实 Mate 70 天线位置；真实测量方向图；互耦；极化；人体、机身遮挡与散射；多径/NLOS；动态 DOA；非 CW 干扰；多颗卫星与多 PRN 并发；导航数据、位/帧同步和 PVT；在线 DOA 与鲁棒导向矢量估计；实时 MVDR 更新；完整 RF 非理想。

因此“理想角点”和“各向同性”只能描述首版接口，不能扩展为实际硬件声明。下一步应以测量得到的相位中心、复数方向图、耦合矩阵和动态传播数据替换这些接口，再重新验证空间自由度、捕获和跟踪结论。
