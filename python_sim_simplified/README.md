# GNSS 接收链路与定位研究代码

真实录波完整入口现为 `recorded_receiver.py`，复用 `gps_lnav.py`、`gps_broadcast.py`、`gps_observables.py` 和既有 DFE/捕获/跟踪/求解器。已从 40 秒公开 HackRF IQ 解出 10 星星历并输出 380 个离线 GPS L1 码位置；见[方法、结果与复现](../docs/reports/2026-09-13-recorded-receiver.md)。`audit_real_gnss.py` 保留为配置与比较实验入口，内部调用通用通道处理。参考位置只用于报告评估。

新增真实录波入口 `audit_real_gnss.py`：读取已采集的 signed int8 IQ，按其采样率/中频经过现有 DFE、捕获和跟踪，默认使用 Costas 处理未知 GPS L1 导航比特。已完成 ION/HackRF 10 秒片段验证，见[报告及复现命令](../docs/reports/2026-09-13-real-single-antenna.md)。该入口不叠加仿真 RF，不输出实测定位或 SDCP。历史 `TrackingConfig` 默认仍为 atan2；真实 L1 电文请显式选择 `carrier_discriminator='costas'`。

当前唯一开发版本。代码和输出路径保持原位；下表用于选择实验入口。

| 任务 | 脚本 | 输入与输出 |
|---|---|---|
| 真实 IQ 接收与 GPS 码定位 | `recorded_receiver.py` | `--iq` 指定已采集文件，`--output` 指定新输出目录；输入格式/参数见报告 |
| 双通道相位基准 | `audit_dual_tracking_phase.py` | 仿真双通道信号，默认输出 `results/dual_tracking_phase/` |
| GPS 定位 | `run_gps_spp.py` | 仿真 GPS 观测，默认输出 `results/gps_spp/` |
| L1/L5 定位 | `run_gps_l1_l5.py` | 双频仿真与权重对照，参数见下文 |
| 校准取余 SDCP | `audit_calibrated_sdcp.py` | 复用已有双通道存档，见下文报告 |
| 窗口检测 | `audit_sdcp_long.py` | 长记录窗口实验，见下文报告 |
| SDCP 定位效果 | `audit_sdcp_positioning.py` | 多星双天线定位对照，见下文报告 |
| 空域抗干扰 | `run_end_to_end.py` | 按配置生成信号并评估合并链路 |

多径与外部干扰按不同研究任务评估。`results/` 是运行输出，正式实验快照在 `../docs/reports/data/`；完整报告导航见 [项目入口](../README.md#研究入口)。

## SDCP 自检研究入口

- `sdcp_self_check.py`：因果三态检测，输入必须附带观测有效性和几何不确定性；不自动消除公共偏差。
- `generate_sdcp_self_check_rf.py`：独立RF记录生成；`audit_sdcp_self_check.py`：固定参数压力测试与离线定位对照。
- `audit_sdcp_geometry_stress.py`：假设基线方向误差回放；`audit_sdcp_low_signal.py`：输入降低6 dB的RF对照。
- [详细报告及复现命令](../docs/reports/2026-09-12-sdcp-self-check.md)。这组研究不替换默认定位入口。

## 实验说明与历史边界

> 2026-09-10 多径专题阶段归档：当前是实验检测基线，未完成通用可靠性验证。详见[阶段总结](../docs/reports/2026-09-10-multipath-stage-summary.md)。下列研究命令生成的 `results/` 均相对本目录；入库快照见总结链接。

双 RF / 双 DFE / 双独立捕获和跟踪环路相位差评估：

3秒记录、500 ms/1 s窗口：`python python_sim_simplified/audit_sdcp_long.py`。结果与边界见[长记录报告](../docs/reports/2026-09-11-sdcp-long.md)。

校准取余SDCP：`audit_calibrated_sdcp.py` 复用双通道存档，验证启动静态多径和通道偏差灵敏度；函数尚未替换原定位权重流程。见[验证报告](../docs/reports/2026-09-11-calibrated-sdcp.md)。

Trimble 单频 SDCP 观测加权验证：`python python_sim_simplified/audit_trimble_sdcp.py`。使用已有 H0 归档冻结门限后执行25条新记录；有干净参考段、共同理想时钟和几何已知等前提，不包含多星定位。方法与失败边界见[专利验证报告](../docs/reports/2026-09-11-trimble-single-frequency-validation.md)。

1 秒 × 10 组长记录验证：`python python_sim_simplified/audit_dual_tracking_stability.py`。默认目录已存在时需用 `--output` 指定新目录；参数、固定统计窗口和结果见[长记录验证报告](../docs/reports/2026-09-11-dual-tracking-stability.md)。

```powershell
python python_sim_simplified/audit_dual_tracking_phase.py
```

固定 TL/TR、无人为通道延迟、无干扰与多径，保留热噪声；比较同一物理时刻的载波 NCO 相位。输出在本目录 `results/dual_tracking_phase/`。时钟假设、启动段与后段统计以及无噪声量化失效见[报告](../docs/reports/2026-09-10-dual-tracking-phase.md)。

## 天线与 RF 幅相偏差

外部源交叉验证入口：`external_gnss_generate.jl` 使用 GNSSSimulator.jl 原生函数生成两路无噪声 IQ；`audit_external_gnss.py --iq <IQ目录> --output <新目录>` 接入当前独立接收链。需先准备归档版本的 Julia 环境，详见[实际结果与复现步骤](../docs/reports/2026-09-13-external-gnss-source.md)。

`AntennaState.amplitude_error_db` / `phase_error_deg` 按天线名字设置，在方向图之后、单元耦合之前生效；true_state 和 assumed_state 分别表示实际响应和估计响应。`RFConfig.amplitude_error_db` / `phase_error_deg` 支持统一标量或按输入行排序的元组，在等效信号与噪声增益之后、限幅和 ADC 之前生效。新增字段默认零。

以下为人为敏感性参数，不是硬件实测值；两个通道使用独立 RF/DFE/环路：

```python
from presets import default_config
from audit_dual_tracking_phase import run_case

cfg = default_config()
cfg.phone.active_antennas = ('TL', 'TR')
cfg.phone.antenna_model.true_state.amplitude_error_db = {'TR': -1.0}
cfg.phone.antenna_model.true_state.phase_error_deg = {'TR': 5.0}
cfg.rf.amplitude_error_db = (0.0, 1.0)
cfg.rf.phase_error_deg = (0.0, 3.0)
case = run_case(duration_ms=1000, seed_base=9900, receiver_config=cfg)
# raw_phase_difference_rad 保留整周；phase_difference_rad 是历史展示兼容字段。
```

完整六场景×两噪声对验证：`python python_sim_simplified/audit_sdcp_hardware_bias.py --output <新目录>`。该审计直接从累计 NCO 重建距离，未取余、未减参考均值；历史检测入口未整体替换。参见[幅相偏差验证](../docs/reports/2026-09-13-sdcp-hardware-bias.md)。

# GNSS 天线分集精简研究仿真版

载波相位基准审计：`python python_sim_simplified/audit_phase_origin.py --output python_sim_simplified/results/phase_origin_repeat`。不拟合参考常数，分别改变启动时刻、NCO初相、共同信号初相和已知通道偏置，检查几何补偿后的小数周残差。[验证与适用边界](../docs/reports/2026-09-11-phase-origin.md)。

单频SDCP定位评估：`python python_sim_simplified/audit_sdcp_positioning.py --output python_sim_simplified/results/sdcp_positioning_repeat`。当前L1链路重新独立标定门限，再比较双天线多星原始、软降权、硬剔除码/载波定位；含共同反射及静态参考污染边界。[评估设计与结果](../docs/reports/2026-09-11-sdcp-positioning.md)。

双频入口现默认输出冻结权重对照，推荐 `L5_code_dual_phase`：L5码确定位置基准，两频载波加权约束变化；L1码显式不参与绝对约束。保留单频、原等权和全观测加权结果。权重配置见 `dual_frequency_weights.json`，适用于当前仿真参数。[优化方法与独立验证](../docs/reports/2026-09-11-dual-frequency-optimization.md)。

单天线GPS L1/L5对比：`python python_sim_simplified/run_gps_l1_l5.py --output python_sim_simplified/results/gps_l1_l5_repeat`。L1继续33 MHz/5，L5 Q5使用33 MHz/1；两个频段各8条跟踪环路，共享物理时间轴，输出单频及双频码/载波浮点定位。包括Q5主码和NH同步，未含I5/CNAV、整周固定或电离层模型。[实现与验证](../docs/reports/2026-09-11-gps-l5.md)。

当前默认采样链：**33 MHz RF → DFE抽取5倍 → 6.6 MHz捕获/跟踪**。继续使用矩形本地码副本；中频4.092 MHz。[新采样率验证及运行命令](../docs/reports/2026-09-11-sampling-33mhz.md)。下方较早研究报告的数值属于各自归档参数。

逐星伪距偏差诊断：`python python_sim_simplified/audit_pseudorange_bias.py`。比较相同RF/DFE样本的矩形/连续本地副本，并运行线性、隔离PRN和连续源对照。只换本地副本的实验中，联合浮点三维RMS降至1.6～2.8 m；默认配置未切换。详见[伪距诊断报告](../docs/reports/2026-09-11-pseudorange-bias-diagnosis.md)。

单天线码/载波联合浮点定位：`python python_sim_simplified/audit_code_carrier_float.py`。读取已有SPP归档做整段离线解算，估计每颗卫星的有效浮点相位偏差，不固定整周数。绝对三维RMS约9～12 m，去均值散布约3.2～3.4 mm；后者不是绝对定位精度。已发现一周突变可漏过粗残差门限，详见[浮点定位验证](../docs/reports/2026-09-11-code-carrier-float.md)。

单天线 GPS L1 时间辅助仿真定位：`python python_sim_simplified/run_gps_spp.py`（从仓库根目录运行）。8个PRN在RF前叠加，共用一套RF/DFE，分别捕获跟踪后构造伪距并解算位置/钟差。无干扰、无多径，暂不包含导航电文或广播星历解码。默认输出已存在时用 `--output` 指定新目录。结果及边界见[定位验证报告](../docs/reports/2026-09-11-gps-spp-validation.md)。

本目录是用于个人研究的精简主代码，包含 RF → DFE → 天线合并 → 捕获 → 跟踪完整链路。它独立于 `python_sim/`，保留当前研究需要的物理模型和算法公式，同时省去批处理、报告和复杂异常诊断框架，便于修改参数和追踪数值结果。

## 运行

当前多径检测研究入口（无外加干扰，含 RF 热噪声，只评估检测）：

```powershell
python python_sim_simplified/audit_detection_only.py --trials 20
```

输出 `results/detection_only/baseline.json`，不运行抑制与跟踪。研究前提、候选检出与正确识别的区别、
同方向边界及门限验证计划见 [多径检测方案](../docs/validation/2026-09-08-multipath-detection-plan.md)。

记录级存在性门限校准与独立评估：

```powershell
python python_sim_simplified/audit_detection_calibration.py
```

默认 500 条 H0 校准、1000 条独立 H0 评估、三组各 200 条 H1 评估；目标记录级虚警 1%。
门限先冻结再评估，不修改默认检测器。输出在 `results/detection_calibration/`，
统计口径与适用范围见 [门限校准记录](../docs/reports/2026-09-08-detection-threshold-calibration.md)。

冻结门限的功率/时延/角间隔单因素扫描：

```powershell
python python_sim_simplified/audit_detection_boundaries.py
```

默认 22 点×100 次，复用逐点结果，输出 `results/detection_boundaries/summary.json`
和 `boundaries.png`。不能将单因素结果当成完整联合性能地图，详见
[检测边界](../docs/reports/2026-09-10-detection-boundaries.md)。

纯多径研究的直达径基线自查（关闭外加干扰与多径，比较噪声、量化和采样率）：

```powershell
python python_sim_simplified/audit_multipath_baseline.py
```

诊断结果保存到 `results/multipath_baseline_audit/baseline.json`。详见
[`基线自查记录`](../docs/reports/2026-09-08-multipath-baseline-audit.md)。
该入口共用接收机代码，不更改以下默认演示场景。

连续本地码研究参考与无噪声 DLL 鉴别曲线：

```powershell
python python_sim_simplified/audit_multipath_baseline.py --bandlimited
python python_sim_simplified/audit_code_discriminator.py
```

`cfg.tracking.code_replica_bandwidth_hz = cfg.dfe.cutoff_hz` 可启用理想低通本地码，
数值是单边基带截止频率，须低于接收采样率的一半。默认 `None` 保留矩形本地码。
该参考使用矩形码片的解析傅里叶系数、64 samples/chip 周期表与线性插值，
不等同于实际 DFE FIR，也未消除生成端有限采样误差。对比结果保存为
`baseline_bandlimited.json`，不会覆盖原 `baseline.json`。

连续带限信号源与单反射径理想参考：

```powershell
python python_sim_simplified/audit_multipath_oracle.py
```

该入口同时设置 `cfg.signal.code_bandwidth_hz` 与跟踪本地码截止频率，关闭外加干扰，
使用线性无噪声链路。`oracle_lcmv` 使用真实方向，仅代表理想参考，不是检测结果。
详见 [连续信号源与理想参考记录](../docs/reports/2026-09-08-continuous-source-oracle.md)。

在连续本地码参考下，可设置 `cfg.tracking.refine_initial_phase = True` 启用数据驱动
小数码相位/载波初相细化，或运行 `python python_sim_simplified/audit_multipath_oracle.py --refine`。
该步骤拟合接收相关峰，不保证多径下的直达径识别；记录位于
[初始化验证](../docs/reports/2026-09-08-tracking-initialization.md)。

运行 `python python_sim_simplified/audit_multipath_oracle.py --refine --compare-detector`
可比较原检测驱动 LCMV 与理想 LCMV，并检查研究用单反射径连续时延细化。
包含非网格方向的失败场景；详见 [检测与理想参考差距](../docs/reports/2026-09-08-detector-versus-oracle.md)。

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
cfg.signal.fs_rf_hz = 33e6
cfg.dfe.decimation = 5  # DFE / acquisition / tracking: 6.6 MHz
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

保留下来的内容包括 2～4 天线组合、手机四角几何、参数化/实测方向图、极化、遮挡、互耦、通道幅相误差、真实/估计通道延迟、CW与带限噪声干扰、静态镜面多径、RF/ADC、定点化 DFE 输出、五种合并方式、捕获、跟踪、码偏差和 C/N₀。尚未包含连续延迟估计、弥散多径、动态路径、NLOS 判决、路径跟踪或多星联合处理。旧版批量研究框架已[归档](../docs/archive/python_sim_2026-09-12/README.md)，需要追溯时先恢复环境。


## 天线分集抗欺骗首轮

`spoofing.py` 提供同频双天线固定8PRN圆统计空间一致性检测。`audit_spoofing_model.py --output <新目录>` 冻结模型门限；`audit_spoofing_rf.py --output <新目录> --model <模型summary.json>` 复用双RF/DFE/独立跟踪验证。检测共同来源，不认证单星，不恢复定位；默认定位入口未变。见[完整报告](../docs/reports/2026-09-12-antenna-spoofing-validation.md)。
