# GNSS 抗干扰仿真与实验

日常只需要看 `core/` 和 `experiments/`：公共接收算法集中维护，每个实验有独立入口。2026-10-14 已移除抗欺骗、多径专用算法、实验、诊断工具、测试，以及公共配置和接收链中的多径接口。保留通用 GPS L1/L5 定位、真实录波接收和现有抗干扰研究。

每次修改及实际验证统一记录在 [CHANGELOG](CHANGELOG.md)。抗干扰的建模前提、实验条件和已有结果见 [详细说明](docs/interference/README.md)。

```text
core/          天线、GNSS、RF/DFE、捕获跟踪与定位；interference/ 放干扰场景与盲零陷
experiments/   interference/ 的通用验证入口与四个既有实验，以及 GPS/录波入口和 paths.py
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
| 通用条件验证：当前默认 TC-001～044 增益基准批次 | `python -m experiments.interference.validation` | `results/interference/pattern_gain_first44/` |
| 完整接收链及绘图 | `python -m experiments.interference.full_chain` | `results/interference/full_chain/` |
| 噪声标定、天线配对消融与独立功率预算 | `python -m experiments.interference.antenna_ablation` | `results/interference/antenna_ablation/` |
| 目标/CW/宽带响应直接赋值，已知/未知目标相位扫描 | `python -m experiments.interference.direct_response_boundary` | `results/interference/direct_response_boundary/` |
| 理想双天线整通道相位误差、准确补偿与单 CW 对照 | `python -m experiments.interference.channel_phase_validation` | `results/interference/channel_phase_validation/` |
| GPS L1 仿真码/载波浮点定位 | `python -m experiments.gps_spp` | `results/gps_spp/` |
| GPS L1/L5 仿真定位与冻结权重对照 | `python -m experiments.gps_l1_l5` | `results/gps_l1_l5/` |
| 真实录波 GPS L1 码定位 | `python -m experiments.recorded_iq` | `results/recorded_iq/` |

也可以在 PyCharm 直接运行 `experiments/interference/full_chain.py`，或执行 `python experiments/interference/full_chain.py`；入口会定位当前精简版根目录。

`full_chain` 默认启用合成参数化天线，包含 CW 和 2 MHz 带限复高斯噪声干扰，比较 SINGLE、EGC、MVDR、Blind Null。公共 `default_config()` 仍为理想各向同性模型。回波生成、SDCP 检测、多径 LCMV 和抗欺骗接收机已删除，配置中也没有重新开启这些方向的开关。

### 通用条件验证

日常修改 [validation_config.py](experiments/interference/validation_config.py)，再运行固定入口 `python -m experiments.interference.validation`；也可直接在 IDE 运行 `experiments/interference/validation.py`。已有参数的单点验证或单因素扫描共用这一入口，不需要再复制接收流程。默认比较 SINGLE、EGC、MVDR 和 Blind Null（`blind_null`）。

配置中的 `build_cases()` 一次完成条件选择、相位基准配对和种子展开，返回已写入种子、时长的独立用例；`build_settings()` 只设置统计窗口、输出位置及是否弹图。运行器直接执行用例，保存端直接记录实际配置。目标真实方向使用 `cfg.signal.true_direction`，合并器假设方向使用 `cfg.signal.assumed_direction`，两者分别修改。天线组合使用 `cfg.phone.active_antennas`，第一根是 SINGLE 的参考通道；逐通道相位按天线名字典赋值。

运行顺序为：用例配置 → 信号源与天线响应 → RF → ADC → DFE → 合并/捕获/跟踪 → 评估与绘图。公共链路在 `core/pipeline.py`，共用分量分析在 `core/evaluation.py`，实验条件、统计与图留在实验层。每个用例按自己的完整配置独立生成前端并执行接收；同一用例的四种算法比较同一份采样，基准与补偿用例使用相同种子分别执行。`validation_support.py` 中的 `run_case()` 执行接收，`evaluate_case()` 负责分量分析、指标计算和基准比较。

相位补偿量由用户配置，作用于整个 DFE 复数序列。它不代表自动估相或宽带时延对齐。天线增益作用在接收机热噪声产生前，不能用 DFE 整通道缩放代替。热噪声收益、含干扰的线性 SINR、接收机跟踪 C/N₀ 分别输出；失捕与缺测保留原状态。具体参数示例和本轮验证记录见 [抗干扰说明](docs/interference/README.md)。

GPS 仿真定位的 `DURATION_MS` 至少为 210 ms 才包含当前首个定位时刻，不保证定位成功。真实录波输入固定为 `results/real_gnss_hackrf/first40s.int8`，需在本地具备录波且实际解出健康星历；记录需长于 1.01 s 才存在当前定位时刻。

## 修改核心

| 职责 | 主要文件 |
|---|---|
| 配置与合成预设 | [config.py](core/config.py)、[presets.py](core/presets.py) |
| 天线响应、几何、极化、互耦与通道时延 | [antenna.py](core/antenna.py)、[channel_delay.py](core/channel_delay.py) |
| GNSS 场景、RF 与 DFE | [gnss.py](core/gnss.py)、[rf.py](core/rf.py)、[dfe.py](core/dfe.py) |
| 捕获、跟踪与重捕 | [acquisition.py](core/acquisition.py)、[tracking.py](core/tracking.py)、[autonomous_tracking.py](core/autonomous_tracking.py) |
| 公共分集合并与完整链路 | [diversity.py](core/diversity.py)、[pipeline.py](core/pipeline.py) |
| 干扰生成、场景响应与盲零陷 | [core/interference/](core/interference/) |
| 结果结构、分量分析、指标与绘图 | [models.py](core/models.py)、[evaluation.py](core/evaluation.py)、[metrics.py](core/metrics.py)、[plotting.py](core/plotting.py) |
| 仿真定位与真实接收 | [gps_simulation.py](core/gps_simulation.py)、[positioning.py](core/positioning.py)、[recorded_receiver.py](core/recorded_receiver.py)、`gps_*.py` |
| 来源哈希和显式源码冻结 | [provenance.py](core/provenance.py) |

`run_end_to_end(cfg)` 默认只运行实际接收链；需要精确分量指标时传入 `retain_analysis_flag=True`，再调用 `core.evaluation.linear_components(results)`。评估复用源真值、本次 RF 噪声和实际滤波参数，不重抽噪声，也不将分量送入合并器求权。线性分量不经过限幅/量化，其总和不等于实际输出，交叉项和联合残差分别统计。

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
