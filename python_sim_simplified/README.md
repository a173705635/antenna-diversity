# GNSS 仿真与实验

日常只需要看 `core/` 和 `experiments/`。公共算法只有一份维护位置，每个实验有独立入口。历史研究代码冻结在 `archive/pre_refactor_2026-09-16/`，不参与当前导入与测试收集。

```text
core/          公共物理模型、RF/DFE、捕获跟踪、观测、检测与定位
experiments/   当前实验：参数、场景、运行、统计和绘图
configs/       冻结权重等实验配置
tests/        当前功能与新旧版本数值对照测试
tools/        冻结版校验与恢复
results/       实验输出与既有大体积研究数据
archive/       历史代码、报告与验证证据，仅在本地保留
```

## 运行入口

2026-09-17 更新：后续多径研究排除直达波与回波同方向、以及仅一路存在多径的场景。`sdcp_multipath` 当前为 13 类场景 × 3 组噪声（39 条测试）：10 类双路不同方向回波场景及 3 类无多径控制。强弱、时延、反射相位、方向、启动/中途进入的对比均保留；重复参数场景已合并。两路相位影响由天线空间响应产生，并非强制相同。历史 16 类/48 条、14 类/42 条结果保留原样；旧同名场景可能只对 B 加多径，应以各次 `plan.json` 的 `channels` 为准，不可混作新场景结果。

在 `python_sim_simplified` 目录执行下列命令，不需要命令行参数。输入、输出和实验参数都固定在对应脚本顶部；要换条件时直接修改这些常量。生成型实验重复运行会先删除并重建自己的固定输出目录，不影响其他实验目录。

| 实验 | 命令 |
|---|---|
| 完整 RF→DFE→合并→捕获→跟踪及绘图 | `python -m experiments.full_chain` |
| 无定位 SDCP 基线：仿真、绘图与独立核验一次完成 | `python -m experiments.sdcp_baseline` |
| SDCP 多径：标定、测试、独立复算及绘图一次完成 | `python -m experiments.sdcp_multipath` |
| 最新单源接收机，比较三/四天线 | `python -m experiments.single_source` |
| GPS L1 仿真定位 | `python -m experiments.gps_spp` |
| GPS L1/L5 仿真定位与冻结权重对照 | `python -m experiments.gps_l1_l5` |
| 真实录波 GPS L1 码定位 | `python -m experiments.recorded_iq` |

单源实验的顶部常量默认用同一份四路录波分别选择三/四天线、24 星、45 dB-Hz、2 秒、0.3 秒引入攻击。设置 `ATTACKED = False` 运行正常对照，修改 `ANTENNAS` 增减天线配置，设置 `CASE` 使用 JSON 配置，设置 `INPUT` 回放已有 `samples.npy`、`observations.npz`、`metadata.json`。`SAVE_IQ = True` 才额外保存新生成的 RF。

SDCP 基线只运行 `python -m experiments.sdcp_baseline`，自动生成数据、绘图并独立核验；全部检查通过后写入 `verification.json`。

SDCP 多径只运行 `python -m experiments.sdcp_multipath`。同一脚本中，`run_simulation()` 负责生成与首次判决，`verify_and_plot()` 从保存的 NCO 独立复算并绘图，`main()` 按顺序调用；无需再运行单独的 verify 脚本。已移除相邻两个 200 ms 窗口均值做差的检测方案及其输出；保留绝对 SDCP 残差的 200 ms 均值判据。核验阶段保留独立 NCO 重建和原有判据，不通过删除失败场景提高结果。

当前双路多径实验的方法、逐场景结果与限制见 [SDCP 多径验证报告](SDCP_MULTIPATH_REPORT.md)。

## 修改核心

| 内容 | 主要文件 |
|---|---|
| 配置、手机几何、天线 | `core/config.py`、`presets.py`、`antenna.py` |
| 信号、RF、DFE | `core/gnss.py`、`rf.py`、`dfe.py` |
| 捕获、跟踪、重捕 | `core/acquisition.py`、`tracking.py`、`autonomous_tracking.py` |
| 分集合并、多径检测 | `core/diversity.py`、`multipath.py`、`pipeline.py` |
| SDCP 与相位弧 | `core/sdcp_observation.py`、`sdcp.py`、`sdcp_arc.py`、`sdcp_self_check.py` |
| 最新单源接收机 | `core/single_source.py`；平滑与载波预测分别在 `mechanism_smoothing.py`、`mechanism_phase.py` |
| 单源场景与空间投影 | `core/single_source_scene.py`、`recovery_geometry.py`、`array_processing.py` |
| 仿真定位与真实录波 | `core/gps_simulation.py`、`positioning.py`、`recorded_receiver.py`、`gps_*.py` |

`core` 不导入实验入口或历史脚本。只服务一个实验的参数和统计留在实验内；有意保留的不同研究方法不强行合并。此次调整不修正既有算法局限，不改变门限或数值模型。

## 版本与归档

日常运行记录 `source_sha256`、实验参数和种子，不再复制源码，不使用旧的逐轮源码缓存。需要把当前源码一并保存时，将生成脚本顶部的 `FREEZE_SOURCE` 改为 `True`。

未冻结源码的 SDCP 实验只能在源码哈希仍匹配时通过源码核验；代码已改就不能把当前实现冒充当时版本。正式要长期复现的结果请显式冻结。

原版共 584 个文件按字节冻结，包含当时未提交的文档生成脚本，详见 `archive/pre_refactor_2026-09-16/FREEZE_MANIFEST.json`。原始 `results/`、`validation/` 和仓库 `docs/reports/data/` 留在原位，没有复制大体积证据。旧报告 Markdown 保留原路径便于查阅；其中旧 `.py` 入口的源码现位于冻结目录。

`archive/` 已被 Git 忽略，只保留在本地，不随克隆或拉取分发，也不继续上库维护。下面的校验/恢复工具仅用于拥有本地冻结版的机器；已提交的历史源码仍可从 Git 历史取得，未提交文件则仅存在于本地冻结版。

```powershell
python tools/frozen_version.py
python tools/frozen_version.py --restore results/restored_original
```

恢复命令只恢复当时源码、配置、测试、报告和日志。历史实验引用的既有 RF 数据、results、validation 及外部资料仍需按原报告放到相应路径；不要直接在冻结目录运行会写结果的旧脚本。历史报告原文索引见冻结版 `README.md`。

## 验证

```powershell
python -m pytest -q
```

默认只收集当前 `tests/`，并排除 `local_evidence` 测试，不依赖本地归档和研究结果。不会收集 results/validation/archive 中的测试副本。完整原版测试保留在本地冻结目录；当前测试针对本次迁移范围，测试数量不能和原版全部研究专题直接比较。

有本地冻结源码和原始研究观测时，额外执行 `python -m pytest -m local_evidence -q`。显式运行会要求这些输入实际存在，不会静默跳过缺失证据。

验证结果和已知限制见 `REFACTOR_VALIDATION.md`。这次重构改善组织和复现方式，不代表三天线可靠导航问题已经解决。

当前双路多径代码的执行顺序、变量和输出说明见 [代码讲解](SDCP_MULTIPATH_CODE_GUIDE.md)。

## 修改记录时长

SDCP 基线和多径均在各自脚本顶部修改 `DURATION_MS`；观测终点取记录结束前 10 ms，尾段取记录最后 400 ms 与有效观测的交集。事件时刻（回波 600 ms、中断 600～750 ms 等）不自动提前。

- baseline：至少 30 ms；缺少有效样本的区域统计以 `status=not_evaluated` 和 `null` 表示，未观察到中断恢复时写明原因。短记录没有足够拟合样本时不输出伪造的幅度、斜率。
- multipath：默认窗口设置至少需要 429 ms；更短时在清理旧输出前报出时长要求。事件未进入观测范围时 `detected=null`；`full_post_event_window=false` 表示尚无完整事件后窗口。700 ms 可运行，但不覆盖 600 ms 回波进入后的完整 200 ms 窗口。
- GPS L1/L5 仿真定位：至少 210 ms 才包含 200 ms 的首个定位时刻；此下限只保证存在观测时刻，不保证定位成功。
- 实测录波定位：记录需长于 1.01 s 才存在当前配置下的定位时刻；仍必须实际解出健康星历，代码不会用默认星历代替。
- 单源抗欺骗：生成记录需长于 170 ms；无攻击后观测时评估状态为未评估、最长失效时长为 null。未初始化与无攻击后数据分别记录，不解释为零误差。

时长检查只针对已经发现的观测窗口约束；不改变物理事件或接收算法来制造有效结果。新统计字段由当前运行产生，历史结果和源码哈希不追改。

多径场景采用完整参数名，例如 `mp_t600ms_p-6dB_d0p3chip_phi90_az110_el15`。`t` 为进入毫秒数，`p` 为相对功率 dB，`d` 为时延 chip（小数点写作 p），`phi` 为反射系数相位度，`az/el` 为回波方向角度。新运行的输出文件使用新名；既有运行和冻结证据保留原名，当前报告使用带来源映射的 renamed view，不将改名当成新实验。
