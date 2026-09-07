# GNSS 天线分集精简研究仿真版

本目录是用于个人研究的精简主代码，包含 RF → DFE → 天线合并 → 捕获 → 跟踪完整链路。它独立于 `python_sim/`，保留当前研究需要的物理模型和算法公式，同时省去批处理、报告和复杂异常诊断框架，便于修改参数和追踪数值结果。

## 运行

在仓库根目录执行：

```powershell
python python_sim_simplified/run_end_to_end.py
```

程序打印 SINGLE、EGC、MVDR、Blind Null 和 Multipath LCMV 的捕获、跟踪、C/N₀、码偏差与多径方向增益，并把图片保存到 `python_sim_simplified/results/`。

程序只保存不重复的研究图，避免同一数据同时出现在总览图和详细图中：

- `detailed_rf_dfe_<天线名>.png`：每个活动天线独立的 RF、ADC 和 DFE 频谱。
- `detailed_combined_<分支名>.png`：每种合并方式独立的绝对幅度频谱，各分支使用相同纵轴范围。
- `detailed_array_response_<分支名>.png`：每种合并方式独立的方位角–俯仰角二维响应。
- `detailed_acquisition_<分支名>.png`：每种合并方式独立的捕获搜索面和码相位切片。
- `tracking.png`：载波频率、Prompt、PLL误差和DLL误差对比。
- `cn0.png`：各跟踪成功分支的 C/N₀ 对比。
- `multipath_delay_profile.png`：直达径投影前后的延迟相关功率及检测门限。
- `multipath_angle_path_<序号>.png`：每条已检测候选径的方位角–俯仰角扫描面。
- `multipath_correlation_comparison.png`：各合并分支的码相关曲线对比。
- `multipath_tracking_bias.png`：各合并分支的跟踪码偏差对比。

图片数量随活动天线数、合并分支数和检测到的多径数变化。默认四天线、五分支时至少保存 24 张 PNG，每检测到一条候选多径再增加一张角度扫描图。每次运行前会清理输出目录中的旧 PNG，避免历史图片与本次结果混合。

## 配置入口

预设配置位于 `presets.py`，配置数据类型位于 `config.py`：

```python
from presets import default_config

cfg = default_config()
```

同一物理量只有一个默认来源。修改信号参数时不需要同步修改 DFE 或捕获配置：

```python
cfg.signal.carrier_hz = 1_575.42e6
cfg.signal.fs_rf_hz = 16.368e6
cfg.signal.if_hz = 4.092e6
cfg.signal.code_rate_hz = 1.023e6
cfg.reference_impedance_ohm = 50.0
```

- DFE 输入采样率自动使用 `cfg.signal.fs_rf_hz`。
- DDC NCO 默认跟随 `cfg.signal.if_hz`。
- 捕获本地码率自动使用 `cfg.signal.code_rate_hz`。
- 信号功率到电压以及 RF 热噪声共同使用 `cfg.reference_impedance_ohm`。

研究 DDC 频率失配时，可以单独覆盖 NCO：

```python
cfg.dfe.nco_frequency_hz = cfg.signal.if_hz + 500.0
```

设置为 `None` 时恢复为自动跟随信号中频。

## 静态多径检测与抑制

默认加入一条相对功率 `-6 dB`、超额时延 `0.5 chip`、反射相位 `π`、方向为方位角 `110°`/仰角 `15°` 的镜面反射径。可直接替换为任意静态路径列表：

```python
from math import pi

from config import Direction, MultipathPath

cfg.signal.multipath_paths = [
    MultipathPath(
        direction=Direction(110.0, 15.0),
        relative_power_db=-6.0,
        excess_delay_chips=0.5,
        reflection_phase_rad=pi,
        doppler_offset_hz=0.0,
    )
]
cfg.multipath_detection.max_excess_delay_chips = 3.0
cfg.multipath_detection.delay_step_chips = 0.25
cfg.multipath_detection.detection_threshold_db = 6.0
cfg.multipath_suppression.max_steering_correlation = 0.95
```

每条反射径先生成独立的延迟、相位、多普勒与天线方向响应，再与直达径相加并进入同一 RF → DFE 链路，因此延迟作用于完整接收序列，不是只移动本地码。检测器使用捕获结果形成逐天线相关向量，投影去除 LOS 后搜索超额时延和到达方向；检测结果再作为 LOS 保持约束 LCMV 的零响应约束。真实多径参数只用于最后计算漏检、虚警及误差，不会送给检测器或权值计算器。

当前是静态镜面多径 V1。DFE 约为 4 samples/chip，延迟搜索步长虽可设为 `0.25 chip`，实际可分辨能力仍受采样率、相关主瓣、信噪比和阵列孔径限制；与 LOS 空间导向高度相关的路径会标记为不可空间分辨，并不会强行加入 LCMV 约束。可完全关闭多径研究路径：

```python
cfg.signal.multipath_paths = []
cfg.multipath_detection.enable = False
cfg.multipath_suppression.enable = False
```

## CW与宽带干扰

默认场景同时启用一个 `-90 dBm` CW干扰和一个 `-90 dBm`、`2 MHz` 带限复高斯噪声干扰，两者都位于方位角 `80°`、仰角 `10°`。宽带干扰的功率表示整个配置带宽内的总功率，不是功率谱密度。

可以直接修改默认宽带干扰；它在 `cfg.interferers` 中位于CW之后：

```python
wideband = cfg.interferers[1]
wideband.power_dbm = -95.0
wideband.center_offset_hz = 200e3
wideband.bandwidth_hz = 2.0e6
wideband.random_seed = 1001
wideband.start_ms = 10.0
wideband.stop_ms = 80.0
```

也可以独立创建宽带干扰：

```python
from config import Direction, WidebandInterferer

cfg.interferers = [
    WidebandInterferer(
        direction=Direction(80.0, 10.0),
        power_dbm=-90.0,
        center_offset_hz=0.0,
        bandwidth_hz=2.0e6,
        random_seed=1001,
    )
]
```

同一随机种子会生成相同的宽带波形，不影响RF热噪声使用的 `cfg.random_seed`。宽带干扰会接受完整的天线响应、时间门控和逐通道真实时延，并与其他干扰一起进入RF、DFE和后续接收机。

当前MVDR仍在整个DFE带宽上使用一组固定复权值，并不是分频段宽带MVDR。`worst_interferer_gain` 对宽带干扰只表示中心频率处的空间增益；整个带宽的抑制效果应结合合并频谱、捕获指标、跟踪状态和C/N₀判断。

## 天线组合与通道误差

```python
cfg.phone.active_antennas = ("TL", "BR")
cfg.phone.channel_amplitude_error_db["BR"] = -1.0
cfg.phone.channel_phase_error_deg["BR"] = 45.0
cfg.phone.channel_delay_ns["BR"] = 10.0
cfg.phone.estimated_channel_delay_ns["BR"] = 8.0
cfg.interferers[0].power_dbm = -95.0
```

`channel_delay_ns` 延迟实际目标和干扰序列；`estimated_channel_delay_ns` 只改变已知目标导向矢量。当前 DFE 不自动对齐通道，所以真实延迟与估计延迟不一致时会保留合并损失。

参数化和实测方向图入口：

```python
from pathlib import Path

from presets import measured_config, parametric_config

cfg = parametric_config(("TL", "TR", "BR"))
cfg = measured_config(Path("pattern.csv"))
```

测量 CSV 列为：

```text
element_name,frequency_hz,azimuth_deg,elevation_deg,h_az_real,h_az_imag,h_el_real,h_el_imag
```

## 代码结构

- `presets.py`：默认、参数化和实测方向图场景。
- `config.py`：配置数据类型和常量。
- `antenna.py`：坐标变换、方向图、极化、遮挡、耦合和有效阵列流形。
- `measured_pattern.py`：实测 CSV 加载与二维角度插值。
- `interference.py`：带限复高斯宽带干扰波形生成和参数校验。
- `gnss.py`：C/A 码、目标信号、干扰源和天线端场景。
- `rf.py` / `dfe.py`：RF 噪声与量化、DDC 和抽取。
- `diversity.py`：SINGLE、EGC、MVDR、Blind Null 和 LOS 保持 LCMV。
- `multipath.py`：数据驱动的延迟/到达角检测及 truth-only 检测评估。
- `acquisition.py` / `tracking.py`：捕获、DLL、PLL、FLL 和 C/N₀。
- `pipeline.py`：端到端信号流与各合并分支编排。
- `models.py`：阶段间结果字典的类型说明。
- `plotting.py` / `metrics.py`：图形和指标输出。

推荐按 `presets.py` → `config.py` → `gnss.py` / `antenna.py` → `rf.py` → `dfe.py` → `diversity.py` → `acquisition.py` → `tracking.py` → `pipeline.py` 的顺序阅读。

## 研究范围边界

本目录假定研究配置有效，没有包含完整版的配置全量校验、异常包装、文件哈希和 schema 审计、退化矩阵保护、Blind Null 安全旁路、断点恢复、Monte Carlo、批处理、报告系统和定点载波 NCO 对比。

保留下来的内容包括 2～4 天线组合、手机四角几何、参数化/实测方向图、极化、遮挡、互耦、通道幅相误差、真实/估计通道延迟、CW与带限噪声干扰、静态镜面多径、RF/ADC、定点化 DFE 输出、五种合并方式、捕获、跟踪、码偏差和 C/N₀。尚未包含连续延迟估计、弥散多径、动态路径、NLOS 判决、路径跟踪或多星联合处理。需要大规模鲁棒性研究或异常诊断时，使用 `../python_sim/`。
