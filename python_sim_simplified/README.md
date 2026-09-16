# GNSS 接收链路与定位研究代码

独立专题：[不接定位的双通道 SDCP 验证报告](SDCP_NO_POSITIONING_REPORT.md)（2026-09-16）。八场景 × 三组噪声，原始 NCO 不取余、不去均值；正常基线标准差 1.543～1.697 mm，单路中断后保留约 +3/−2/+3 周偏移。见[实验入口](audit_sdcp_no_positioning.py)、[原始证据](validation/sdcp_no_positioning_2026-09-16/)及[独立核验](verify_sdcp_no_positioning.py)。不涉及定位或检测率验证。

四轮机制研究已完成，最新研究入口为 `single_source_receiver_history.py`：区分短暂观测筛选间隙与真实失锁，保留同跟踪弧平滑历史，并用最近已发布位置及实际载波位移接续。旧困难四天线样本165→171/171、354→371/371，最大误差25.76→12.59m、33.87→16.68m，输出未减少；冻结新种子4秒四天线保持371/371，最大误差7.91→5.30m。三天线新种子仅4→15/371，仍不能保底。真实消失负例停发且无错误恢复；当前统一版本强信号近源及撤除仍有约0.1秒中断，不能沿用更早版本的全部通过结论。44组配对、64份导航重演、397项测试通过，未证明商用或长期导航。见[四轮结果](MECHANISM_FOUR_REPORT.md)、[数学机制与边界](MECHANISM_FOUR_METHODS.md)、[研究计划](MECHANISM_FOUR_PLAN.md)。

弱信号10轮研究已完成，`single_source_receiver_weak.py` 只保留为研究候选，未通过可靠导航要求。改进已锁定跟踪的质量估计，并将投影后方向证据用于观测准入：24星30dB-Hz开发样本三/四天线均171/171，无攻击控制由8/10改善到171/171；但冻结新种子三天线仅4/171、四天线165/171。另一种子4秒三天线4/371、四天线354/371，四天线在攻击切入转换时出现2次≥30m错误，最大33.87m。三天线尚不能保底，绝对码位置与载波接续约束仍未解决。43组同RF对照、58份导航重演和输入/源码哈希核对，389项测试通过；失败版本与负例完整保留。见[10轮结果](WEAK_TEN_REPORT.md)、[数学方法和失败原因](WEAK_TEN_METHODS.md)、[计划及调整](WEAK_TEN_PLAN.md)。

空间触发修复10轮研究已完成，研究候选入口为 `single_source_receiver_spatial.py`：以噪声归一化双模型残差改进触发，并将公共源拟合增至6次。8星三天线正常案例142/171→171/171；漂移案例三/四天线合格138/101→171/170。24星新种子正常和近源簇保持全部合格；6秒近源簇三/四天线各571/571，最大误差1.21/1.07m。弱信号和8星近源仍大量中断，未证明商用或分钟级导航。27组同RF对照、52份导航重演核对，381项测试通过。见[10轮结果](SPATIAL_TEN_REPORT.md)、[数学方法](SPATIAL_TEN_METHODS.md)、[研究计划](SPATIAL_TEN_PLAN.md)。

24星受控扩展已完成：保持原8星和录波条件、追加16星，单源同步伪造全部24个PRN。三组天空的无攻击、一星重合、三星近源共18次三/四天线运行均连续输出且≤20m，最大误差约2m；每份录波仅2秒。旧8星无攻击中断已定位为正常天空也触发空间投影、削弱真实信号，关闭投影消融恢复171/171；该批尚未修复触发机制，后续修复见上方10轮报告。378项测试通过。见[24星结果及无攻击根因](SATELLITE_COUNT_24_REPORT.md)、[实验方案](SATELLITE_COUNT_24_PLAN.md)。

单源、正常初始化后20轮连续接收机研究已完成，包含逐轮源码/同录波对比、36份四路RF验证及状态交接审计修正。该批审计修正入口为 `single_source_receiver_fixed.py`；冻结第20轮保留作证据，修正版回放单列。常规持续攻击补充组中三/四天线各513/513机会≤20m；初始化后转弱的已就绪记录仅18/342、167/513（另1份三天线未按时就绪），近方向仍大量中断，未证实三天线保底或商用长期导航。376项测试通过。见[本批结果与审计修正](SINGLE_SOURCE_TWENTY_REPORT.md)、[数学方法与验证边界](SINGLE_SOURCE_TWENTY_METHODS.md)、[20轮台账](SINGLE_SOURCE_TWENTY_LEDGER.md)。

更新需求后的100个机制候选实验已完成：同四路记录比较二/三/四天线，另研究来源、恢复与导航。冻结延长采样码验证中，三天线901/1098历元≤20m、3次≥30m；四天线850/1098、6次≥50m，漂移仍导致失败，未达到三天线保底。真实信号消失时停发位置，连续RF闭环未贯通。370项测试通过。见[本批结果](RESEARCH_WARM_HUNDRED_REPORT.md)、[方法与边界](RESEARCH_WARM_HUNDRED_METHODS.md)、[100轮台账](RESEARCH_WARM_HUNDRED_LEDGER.md)。

当前研究范围已按用户要求排除冷启动攻击及多欺骗源攻击：先在无欺骗条件下完成正常初始化，再引入单个物理发射源的欺骗；同源伪造多个PRN仍属单源。低C/N₀、攻击后真实信号消失、失锁及重捕获仍在范围内。三天线为最低目标配置，四天线为主研究配置，是否达标仍需验证；二天线仅作性能对照，性能差不要求必须解决，但须用可复现实验解释原因。后续主验收按此口径独立统计，历史数据与报告保持原样。详见[当前研究范围](RESEARCH_SCOPE.md)。

连续恢复第 201–400 轮已完成：新增因果码载波平滑、原始多天线空间证据、运动及渐变拖曳，含240条新合成RF观测归档、60个同输入规则比较与20个观测域反例。相对上一轮备选，独立留出攻击全历元≤20m由74.8%提高到77.3%，≥30m错误88降至84；边界组≥30m反而196升至211。两组总≥50m由56降至24，但≥30m总数284升至295，仍不满足用户失败标准。正常输出率为95.0%/98.4%，最长RF仅8秒；启动校准污染和准入/恢复偏差仍待解决。266项测试通过。见 [200轮结果、失败与复现](RECOVERY_FOUR_HUNDRED_RESEARCH.md)、[数学模型与假设](RECOVERY_FOUR_HUNDRED_METHODS.md)、[预注册计划](RECOVERY_FOUR_HUNDRED_PLAN.md)。

连续恢复第 101–200 轮已完成，新增短时解扩、启动相位校准、空间辅助副本抵消及质量加权准入。主方案独立留出仍有56个≥30m错误，边界最大已输出误差108.03m；另用两套新天空几何做40条配对，校准解扩备选将攻击下全时钟≤20m比例从主方案71.6%提高到84.8%、≥30m错误4降到1。260项测试通过。未达到长期可靠导航或商用放行要求。见 [100轮结果、负例与复现](RECOVERY_TWO_HUNDRED_RESEARCH.md)、[数学方法与所给DOI论文对应](RECOVERY_TWO_HUNDRED_METHODS.md)。

空间恢复研究第 51–100 轮已完成：独立快照候选将原始解的 73 个 ≥30 m 错误全部拒绝，但攻击/正常输出率仅 66.87%/86.57%。逐 PRN 解扩改善部分连续多普勒工况；快照选出的 5 m 子集解门限用于连续跟踪时大量误拒，未知比特攻击组全时钟合格输出仅 13.49%，仍未满足持续导航目标。255 项测试通过。见 [本轮数学方法、50 轮结果、失败分析与复现](RECOVERY_HUNDRED_RESEARCH.md)。

新增空间抑制与信号恢复研究：从真假混合 IQ 提取码延迟并解算位置，比较阵列投影、MVDR 与迭代源估计，含 50 轮开发/留出配置、10/30 分钟分时快照和独立连续 DLL/PLL 检查。原始位置解仍可能出现公里级错误，未接入商用可信准入。见 [50 轮方法、结果、重大失败与复现](RECOVERY_FIFTY_RESEARCH.md)。

新增第 11–50 轮研究：6464 组配对评估。空间相位时间差分在最终名义转弯组把固定偏差＋平滑伪加速度隔离从 0/8 提升到 8/8，静止仍 0/8；实际新增告警延迟约 9.9 s，八类正常误差仍有基线导致的 13/128 误隔离。默认接收机未切换策略。见 [40 轮方法、逐轮证据、独立留出与边界](FORTY_ROUND_RESEARCH.md)。

十轮导航研究已完成：832 组主实验 + 384 组终轮基线对照。候选暂停、速度创新和有效加速度估计使最终留出 3 PRN 快速接管的前 10 s 最坏误差由 40.64 m 降到 1.78 m；差异偏差+平滑伪加速度仍 0/48 隔离、48/48 在允许更新时超过 30 m。候选仅保留为研究选项。见 [十轮方法、逐轮结果、失败分析与复现](TEN_ROUND_RESEARCH.md)。

连续误报实验：`audit_spatial_confirmation.py --output results/spatial-confirmation --navigation`。独立正常留出误报由 14/48 降至 0/48，但 3 PRN 最慢确认增至 5.291 s，选取该失败轨迹的前 10 s 导航误差增至 45.04 m；新策略未通过要求，默认单窗行为保留。见 [连续确认方法与代价](SPATIAL_CONFIRMATION.md)。

导航闭环首轮：`audit_navigation_fusion.py --output results/navigation-closed-loop`，48 个连续合成场景、96 次配对 GNSS/INS 运行。理想接管短时误差降低，但正常组 3/8 误隔离、20 mm 通道×PRN 差异偏差导致攻击后 8/8 无证据；PVT 拉偏为显式假设，尚未经过 RF 定位求解。详见 [方法、结果与商用缺口](NAVIGATION_FUSION.md)。

W4–W6 软件流程已新增：`audit_w4_w6.py --output <新目录>`，包含 30 个惯性误差场景、RF/IMU 台架标定演示、条件协方差预算及空间检测敏感性。`--bench-dir <目录>` 可读取约定格式的输入。当前用户无台架数据，演示全部标记 synthetic；硬件测量与放行保持待测。详见 [方法、输入格式、结果与验收边界](W4_W6.md)。

四路三 PRN 自主重捕与观测冗余：`audit_array_reacquisition.py --output <新目录>`。TL/TR 单路中断后，2/3/4 天线分别保留 0/1/3 对故障前已有参考的无故障基线，固定尾段观测可用比例为 0/33.3%/50%；这不是导航可用率或欺骗检出率。方法与限制见 [同步回放说明](SYNC_REPLAY.md)末节。

新增质量驱动的自主重捕实验：`audit_autonomous_tracking.py --output <新目录>`。新生成 RF 中，干净组无额外重捕，中断两组自行确认质量丢失、重试捕获并进入新环路；旧参考保持撤销。当前为单 PRN 双通道离线验证，参数未标定，结果和边界见 [同步回放说明](SYNC_REPLAY.md)末节。

RF 跟踪历史消费已补充：`audit_sdcp_arc.py --output <新目录>` 现在重建载波单差并控制实际残差输出，增加原环路与重捕历史的连续拼接。干净尾段 140/140 输出，中断/重捕各尾段质量均恢复但旧参考输出为 0；这是已有数字 RF 回放，尚非实时接收机集成。方法和验证见 [同步回放说明](SYNC_REPLAY.md)。

首阶段 W2 已新增窄带同步误差回放：`audit_sync_replay.py --output <新文件.json>`，覆盖 RF 延迟、ADC 偏斜、采样频差、公共中频 NCO 时差及标签错位，共 90 场景。公式、9 项相关测试和数值分析见 [同步回放说明](SYNC_REPLAY.md)。这是误差模型子项，尚非自动故障检测或导航闭环。

W2 后续流式事件回放：同一入口加 `--stream`，覆盖 60 个丢重样/NCO 重置场景，并复用既有相位弧控制残差输出。如实上报的 30 场撤销旧参考，隐藏事件的 30 场仍不可识别；相关测试共 24 项通过。详见同一说明的“丢重样、重置事件与相位弧消费”，目前未接入真实跟踪器。

2026-09-14 精简清理：删除通用字段/类型/形状验证包装和仅为其存在的测试；`sensor_contract.py` 只保留直接的旧 RF 记录适配。定位秩亏/不收敛使用 `PositioningFailure` 表示既有数学失败，实验输出保留具体原因；其他异常直接暴露，不再统一吞成 NaN。捕获失败、相位弧状态、检测窗口、量化/饱和及归档不可覆盖语义保留。输入单位、同步和数组顺序由实验配置明确提供，不再依赖通用拒收层。

RF 几何现在必须显式选择：新阈值文件默认读取 `positions_m`；复现不带坐标的旧手机阈值时，使用 `audit_spatial_diversity_rf.py --geometry phone --model <旧阈值文件> --output <新目录>`。不再因字段缺失自动回退到手机阵列；当前工作尺寸仍为 0.0759 × 0.1609 m。

本次清理验证：`python -m pytest -q` 为 191 passed（96.05 s），保留已有零噪声 C/N0 除零警告，未添加兜底。与提交 `68e3815806db49c58173ddc0cf70156cd851e7d8` 的独立进程对照中，48 份原尺寸 RF 记录重建的 36 个 2/3/4 天线布局批次，以及检测窗口、SD/DD 分数、固定种子宽带波形、时延、SDCP/权重和正常 SPP 数值完全一致；这属于既有数据回放与确定性计算对照，不是新增攻击性能实测。原尺寸、1 米及测向双差三个冻结归档共 605 个文件 SHA-256 均未变化。仅验证防御性拒收的测试已删除，保留模型失败测试并增加异常原因记录与未捕获卫星适配测试。

新增研究入口：`audit_spatial_direction.py --output <新目录> --archive <原手机尺寸归档>`，验证星间双差统计与方向候选；`audit_reference_direction.py` 对照可信/被污染 PRN 8 参考。`spatial_direction.py` 输出校准后的方向候选和拒绝状态，`spatial_reference_direction.py` 使用显式参考方向。无 IMU；多解、镜像及混合信号测向失败均保留。详见[方法与结果](../docs/reports/2026-09-14-doa-double-difference.md)。尚非唯一测向，未自动接入原检测器。

当前工作基线已回到 **0.0759 × 0.1609 m** 的手机尺寸，保持 2–4 天线配置。运行 `audit_spatial_diversity.py --output <新目录>` 即使用该默认尺寸；RF 可使用新生成的原尺寸阈值，或通过 `--geometry phone` 使用 `../docs/reports/data/2026-09-13-gpspatron-spatial/model/thresholds.json`（相对本代码目录）。默认读取阈值文件中的坐标，勿将历史 1 米实验阈值作为当前基线。尺寸参数仍可用于复现实验：[1 米历史对照](../docs/reports/2026-09-14-spacing-1m.md)。

新增2–4天线空间抗欺骗入口：`audit_spatial_diversity.py --output <新目录>`生成5000条H0标定与22场景各5000条留出事件；`audit_spatial_diversity_rf.py --output <新目录> --model <模型目录>/thresholds.json`运行四RF链并取2/3/4天线子集比较。检测模块`spatial_spoofing.py`接受同步载波距离和有效性，提供全星/三星簇判据及三态输出。不是GPSPATRON专有算法复刻。[完整方法、收益、失效与复现](../docs/reports/2026-09-14-gpspatron-spatial-diversity.md)，[配置调用示例与归档](../docs/reports/data/2026-09-13-gpspatron-spatial/README.md)。默认旧仿真入口保留。

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

第三批 [100轮研究（201–300）](RESEARCH_THIRD_HUNDRED_REPORT.md) 覆盖源存在性、相关误差、真信号恢复和IMU转动；[逐轮账本](RESEARCH_THIRD_HUNDRED_LEDGER.md)保留全部改动与失败。新RF导航172/204→154/204，未采用为默认接收机；提示层误放行改善尚未贯通RF/PVT，弱星恢复和窄角多源误判仍未解决。数学和验证边界见[方法](RESEARCH_THIRD_HUNDRED_METHODS.md)。

第二批 [100 次实际研究闭环（101–200）](RESEARCH_NEXT_HUNDRED_REPORT.md) 已整理为[逐轮账本](RESEARCH_NEXT_HUNDRED_LEDGER.md)和[数学与验证方法](RESEARCH_NEXT_HUNDRED_METHODS.md)。时延标定有条件收益，冷启动纯噪声门控仍失败；导航保留共同偏差反例，惯性600秒验证主要受可信输入和可用性限制，尚不能商用。原始失败、弧段与延迟运输纠错、新种子冻结结果均保留。

本批 [100 次实际研究闭环](RESEARCH_HUNDRED_REPORT.md) 覆盖导航、空间—延迟分离、RF捕获、矩阵标定、测向及惯性误差边界，提供[逐轮索引](RESEARCH_HUNDRED_LEDGER.md)与[方法和限制](RESEARCH_HUNDRED_METHODS.md)。低信噪比捕获有所改善，但冷攻击、群延迟偏差及持续定位仍未解决；包含失败和可用性退步，不具备商用放行依据。

前一批 [20 次研究闭环](RESEARCH_TWENTY_REPORT.md) 保留逐次假设、源码、对照和失败结论。新增双波束锚点/载波增量导航与启动联合校准；旧攻击记录回归的 ≥30m 错误从 295 降至 6，但仍出现约 71m 错误，正常可用性也有下降，尚不具备商用放行依据。数学前提和复现见 [方法说明](RESEARCH_TWENTY_METHODS.md)。

`spoofing.py` 提供同频双天线固定8PRN圆统计空间一致性检测。`audit_spoofing_model.py --output <新目录>` 冻结模型门限；`audit_spoofing_rf.py --output <新目录> --model <模型summary.json>` 复用双RF/DFE/独立跟踪验证。检测共同来源，不认证单星，不恢复定位；默认定位入口未变。见[完整报告](../docs/reports/2026-09-12-antenna-spoofing-validation.md)。
