# GNSS 抗干扰仿真与实验

日常只需要看 `core/` 和 `experiments/`：公共接收算法集中维护，每个实验有独立入口。2026-10-14 已移除抗欺骗、多径专用算法、实验、诊断工具、测试，以及公共配置和接收链中的多径接口。保留通用 GPS L1/L5 定位、真实录波接收和现有抗干扰研究。

所有实验已暂时注释事后分量分析及其处理、输出和绘图，不再生成分量功率、SNR/SINR、抑制比、分量增益、交叉项或联合量化残差。实际接收、相位处理、捕获跟踪及接收机 C/N₀ 保留；旧结果不改写。

分集实验优先看 `case_summary.csv` 用例汇总，`summary.csv` 保存明细，FULL CHAIN 的 `summary.csv` 每算法一行。`manifest.json` 保存条件、实际配置和源码哈希，通用验证的 `summary.json` 保存统计与核验。两入口的跟踪 C/N₀ 统一为原始历元后半段有效 dB-Hz 值的均值；细节见[抗干扰说明](docs/interference/README.md)。

表头缩写：`ACQ Pass` 为过捕获门限，`ACQ OK` 为正确捕获（主表为次数、明细为布尔值），`TRK` 为跟踪成功；`N` 是总 seed 数，`CN0 Seeds` 是有效 seed 数，`CN0 Epochs` 是有效历元数，`SD` 是标准差，`Nom` 是名义值。

CSV 与终端中的 CN0、标准差、捕获比值、频差和米制误差显示两位小数，码片误差三位，计数为整数；计算和 JSON 保留原始精度。

每次修改及实际验证统一记录在 [CHANGELOG](CHANGELOG.md)。抗干扰的建模前提、实验条件和已有结果见 [详细说明](docs/interference/README.md)。

```text
core/          天线、GNSS、RF/DFE、捕获跟踪与定位；interference/ 放干扰场景与盲零陷
experiments/   interference/ 的通用验证与完整链路，以及 GPS/录波入口
configs/       通用定位实验的冻结权重等配置
tests/         当前功能测试与测试数据，仅本地保留；冻结对照以 local_evidence 单独标记
tools/         frozen_version.py 校验与恢复原始冻结版
docs/          interference/ 为当前说明；spoofing/、multipath/ 为历史报告
results/       本地输出；已有历史目录与来源清单保留原样
archive/       本地冻结源码，不参与维护、导入或默认测试
```

## 运行入口

在本目录运行下列命令，无需命令行参数。输入、输出、种子、时长和绘图设置固定在各脚本顶部，修改条件直接编辑这些常量。生成型实验重跑只重建自己的固定输出目录。

目录常量以 `_DIR` 结尾（如 `INTERFERENCE_RESULTS_DIR`、`OUTPUT_DIR`），输入文件路径以 `_PATH` 结尾（如 `IQ_PATH`）；`EVALUATION_EDGE_TRIM_MS` 表示评估时首尾各舍弃的毫秒数，不裁掉捕获跟踪的数据。

| 实验 | 命令 | 固定输出 |
|---|---|---|
| 通用条件验证：实验类型与 seed 在 `validation_config.py` 选择 | `python -m experiments.interference.validation` | `results/interference/<EXPERIMENT_TYPE>/` |
| 完整接收链及绘图 | `python -m experiments.interference.full_chain` | `results/interference/full_chain/` |
| GPS L1 仿真码/载波浮点定位 | `python -m experiments.gps_spp` | `results/gps_spp/` |
| GPS L1/L5 仿真定位与冻结权重对照 | `python -m experiments.gps_l1_l5` | `results/gps_l1_l5/` |
| 真实录波 GPS L1 码定位 | `python -m experiments.recorded_iq` | `results/recorded_iq/` |

也可以在 PyCharm 直接运行 `experiments/interference/full_chain.py`，或执行 `python experiments/interference/full_chain.py`；入口会定位当前精简版根目录。

`full_chain` 默认启用合成参数化天线，包含 CW 和 2 MHz 带限复高斯噪声干扰，只运行 SINGLE 和 Blind Null；两者求权都不读取目标幅相。公共 `default_config()` 仍为理想各向同性模型。回波生成、SDCP 检测、多径 LCMV 和抗欺骗接收机已删除，配置中也没有重新开启这些方向的开关。

L1 默认捕获为4 ms相干×5次非相干累积，共20 ms；跟踪自动从捕获使用数据后的码边界开始，跟踪相干时长仍为1 ms。L5采用1 ms相干×5次累积，另需1 ms输入以处理NH码边界。

### 通用条件验证

日常修改 [validation_config.py](experiments/interference/validation_config.py)，再运行固定入口 `python -m experiments.interference.validation`；也可直接在 IDE 运行 `experiments/interference/validation.py`。各类实验共用 `MODES`、`ACTIVE_ANTENNAS`、`SEEDS`、`DURATION_MS`、`WORKERS` 与统计去边；验证默认运行 SINGLE、Blind Null、预设相位 EGC 和已知响应 MVDR，后两者标为对照。FULL CHAIN 仍只运行前两种方法。

`EXPERIMENT_TYPE` 可选 `phase_offset`（默认，12条件）、`rf_group_delay`（18条件）、`antenna_temperature`、`phase_center_offset`（17条件，PC-001～017）、`mutual_coupling`（14条件，MC-001～014）、`antenna_gain`（17条件，TC-001～017）；输出统一为 `results/interference/<EXPERIMENT_TYPE>/`。温偏实验分别扫描真实天线增益和相位，名义响应固定在参考温度；系数仅为示例敏感性假设。条件与限制见[抗干扰说明](docs/interference/README.md)。

相位实验通过 `PHASE_ANTENNA`、`PHASE_SCAN_DEG` 和 `SCENES` 设置受影响通道、相位及场景，每个场景每个相位独立运行一次，0°作为基准；不再包含方向、增益扫描和准确补偿条件。`_base_config()` 集中设置共享配置，`build_cases()` 统一展开种子，`build_settings()` 设置统计、输出和并行参数。`TARGET_TRUE_DIRECTION`、`TARGET_ASSUMED_DIRECTION` 分别设置真实与名义目标方向；`ACTIVE_ANTENNAS` 的第一根是 SINGLE 参考，增益、互耦和位置偏移条件作用于前两根，额外激活阵元保留基础配置。

RF物理延时实验设 `EXPERIMENT_TYPE="rf_group_delay"`，由 `RF_GROUP_DELAY_ANTENNA`、`RF_GROUP_DELAY_SCAN_NS`、`RF_GROUP_DELAY_SCENES` 控制；默认只给BR增加0/10/20/50/100/200 ns延时，覆盖无干扰、单CW和纯2 MHz宽带干扰。`cfg.rf.group_delay_offset_ns` 按物理通道指定非负时延，空映射表示零；包络延迟与RF载波相移同时施加，馈线和名义模型保持不变，不作新增延时的逆补偿。输出带标记点连线的CN0、必要的ACQ图及理想中心/带内相位图；这是纯延时敏感性模型，非器件实测群延时起伏或硬件规格，见[模型与结果](docs/interference/README.md#rf物理延时2026-10-22)。

增益、互耦与相位中心汇总按“条件＋算法”统计捕获跟踪；日常表保留总 seed 数、成功次数、C/N₀ 均值、样本标准差和有效 seed 数，完整缺测统计在 JSON。`scan_label`、`scan_unit` 用于真实扫描的图轴与文件名，离散条件不再填写 TC/MC/PC 伪单位。

验证图按实验目标组织：相位按场景分面比较四方法的 C/N₀，仅有捕获失败时增加 ACQ 图（单 seed 为成功/失败标记，多 seed 为正确捕获比例）；单路增益比较两根天线的衰减；互耦分别扫描电压比与相位，PCO分别移动两根天线，温偏分别扫描增益与相位。所有实验共用 `GENERATE_PLOTS_FLAG`、`SHOW_PLOTS_FLAG`；C/N₀ 缺测留空，重合曲线不人为错开。

专项统计由 `EXPERIMENT_TYPE` 选择；日常核验保留实际合并输出与功率的独立复算，依赖分量的理想噪声收益检查暂时停用。模型与权值公式检查放在本地测试；核验通过不代表捕获跟踪成功。

运行顺序为：用例配置 → 各方向源施加自身来波响应并汇总为干净阵元输入 → 130 K独立输入噪声与所有源共经固定天线矩阵 → 阵元损耗噪声与馈线 → RF 附加噪声及滤波 → ADC → DFE → 合并/捕获/跟踪 → 评估与绘图。外部 `scene=` 也应提供已含逐源来波响应的干净阵元输入，公共矩阵只施加无源损耗和互耦。`core/pipeline.py` 的 `run_frontend()` 直接列出这些阶段；每个用例按完整配置独立执行，同一用例启用的方法共用采样。实验层 `run_case()` 执行接收，`evaluate_case()` 统计指标，[validation_plotting.py](experiments/interference/validation_plotting.py) 绘图。

默认独立阵元输入噪温为固定130 K（`cfg.phone.antenna_noise_temperature_k`），不随离散来波的方向图、遮挡或阵元幅相偏置改变；这些响应逐源施加。阵元无源损耗默认0 dB、物理温度290 K（`AntennaState.passive_loss_db`、`temperature_k`），其损耗与互耦仍作用于输入噪声。馈线默认损耗1 dB、物理温度290 K，可按天线名覆盖损耗、载频总相位和传播时延；RF 附加噪声由级联噪声系数确定。独立噪声经天线矩阵后可跨通道相关；`noise_info` 给出 RF 输入载频中心的温度矩阵，不能当作整带采样协方差。[模型与参考面](docs/interference/README.md#接收噪声参考面与馈线2026-10-21)。

完整 `MODES` 直接写入用例配置，`run_receiver()` 集中分派算法，结果统一放在 `branches`；各分支返回 `combined_samples`（合并采样）、`combining_info`（权值等信息）及算法角色、相位来源。通用验证的 `receiver_samples` 是相位处理后实际送入接收机的多通道采样。

SINGLE 直接取第一根激活天线，不按目标响应归一化。MRC 已移除；EGC 使用显式输入的相位，权值为 `exp(jφ)/N`，各路等幅且不依赖目标强度。EGC 和 MVDR 只供显式对照，前者目前使用预设相位、后者使用名义目标响应，均不代表已实现自动估相。`AntennaState` 的幅相偏置只改变来波响应，器件通道误差由馈线和 RF 配置。接收机跟踪 C/N₀ 缺测不填零，实验条件见 [抗干扰说明](docs/interference/README.md)。

`phase_offset` 在 DFE 输出旋转指定完整通道（含目标、干扰、噪声），不代表天线端相位误差；名义模型固定，注入相位不自动提供给 EGC/MVDR，也不执行逆补偿。

GPS 仿真定位的 `DURATION_MS` 至少为 210 ms 才包含当前首个定位时刻，不保证定位成功。真实录波输入固定为 `results/real_gnss_hackrf/first40s.int8`，需在本地具备录波且实际解出健康星历；记录需长于 1.01 s 才存在当前定位时刻。

## 修改核心

| 职责 | 主要文件 |
|---|---|
| 配置与合成预设 | [config.py](core/config.py)、[presets.py](core/presets.py) |
| 天线响应、几何、极化、阵元损耗与互耦 | [antenna.py](core/antenna.py) |
| GNSS 信号生成、噪声、馈线、RF 与 DFE | [gnss.py](core/gnss.py)、[noise.py](core/noise.py)、[feedline.py](core/feedline.py)、[rf.py](core/rf.py)、[dfe.py](core/dfe.py) |
| 捕获、跟踪与重捕 | [acquisition.py](core/acquisition.py)、[tracking.py](core/tracking.py)、[autonomous_tracking.py](core/autonomous_tracking.py) |
| 公共分集合并与完整链路 | [diversity.py](core/diversity.py)、[pipeline.py](core/pipeline.py) |
| 干扰生成、场景响应与盲零陷 | [core/interference/](core/interference/) |
| 结果结构、分量分析、指标与绘图 | [models.py](core/models.py)、[evaluation.py](core/evaluation.py)、[metrics.py](core/metrics.py)、[plotting.py](core/plotting.py) |
| 仿真定位与真实接收 | [gps_simulation.py](core/gps_simulation.py)、[positioning.py](core/positioning.py)、[recorded_receiver.py](core/recorded_receiver.py)、`gps_*.py` |
| 来源哈希和显式源码冻结 | [provenance.py](core/provenance.py) |

当前实验不保留事后分析专用的噪声副本，也不调用 `core.evaluation.linear_components()`；显式启用分析时才保留实际各噪声源传播到 RF 输入端的合成采样，供本地独立测试。`scene_input_v` 和外部 `scene=` 为干净阵元等效输入源电压。

`core` 不导入实验入口或历史脚本。实验独有的参数、评分口径与统计留在实验脚本；公共接收链只包含当前维护方向所需接口。天线参数仍是合成研究假设，不能作为真实手机规格或标定结果。

## 本地备份与历史报告

本次清理前的版本在 `archive/pre_interference_only_2026-10-14_156b994/`，包含 271 个文件，保留根目录相对路径；`FREEZE_MANIFEST.json` 记录每个文件 SHA256 和起始 Git 提交。恢复时按清单将文件复制回对应位置，再配合原数据运行；不要在冻结目录直接运行会写结果的实验。

历史报告保留：[单源抗欺骗报告](docs/spoofing/SINGLE_SOURCE_SUPPRESSION_REPORT.md)、[SDCP 多径报告](docs/multipath/SDCP_MULTIPATH_REPORT.md)、[无定位 SDCP 基线](docs/multipath/SDCP_NO_POSITIONING_REPORT.md)。其中入口、代码路径、默认配置和实验结论对应当时版本；当前代码不能直接执行这些实验。旧来源哈希不改写，严格复现需使用当时 `source_snapshot.zip`、本地冻结版或 Git 历史及原始数据。

`archive/` 和 `results/` 被 Git 忽略，仅本地保存，不随克隆或拉取分发，不继续上库维护。旧的 `archive/pre_refactor_2026-09-16/` 仍保留；以下工具只校验/恢复这一原始冻结版，不操作本次备份：

```powershell
python tools/frozen_version.py
python tools/frozen_version.py --restore results/restored_original
```

日常实验记录参数、种子和 `source_sha256`。仅当需要冻结当前源码时，将脚本顶部 `FREEZE_SOURCE_FLAG` 改为 `True`；运行结果不会默认生成源码副本。

## 验证

当前 `tests/`、`test/` 目录和测试脚本已加入忽略规则，只在本地维护，不提交或推送。新克隆的仓库不包含这些测试文件；以下命令需要本地具备 `tests/`。`pytest.ini` 与测试依赖继续保留，供本地验证使用；历史报告中的冻结测试副本保持原样。

本机现有测试已保留。其他已克隆副本若需要保留其测试，应在首次拉取本次取消跟踪提交前另存 `tests/`，因为 Git 会在那些副本中移除原已跟踪且未修改的测试文件。

```powershell
python -m pytest -q
```

默认只收集当前 `tests/`，排除 `local_evidence`，不依赖本地归档或研究数据，也不收集归档和结果中的测试副本。本地具备本次清理前冻结源码时，额外执行 `python -m pytest -m local_evidence -q`，在独立解释器中对照 GPS L1/L5 数值。本次删除了已退出维护方向的测试，测试数量减少不是原有研究重新通过的证据；实际清理验证见 CHANGELOG。旧重构过程记录在 `REFACTOR_PLAN.md` 与 `REFACTOR_VALIDATION.md`，按日期区分历史范围。
