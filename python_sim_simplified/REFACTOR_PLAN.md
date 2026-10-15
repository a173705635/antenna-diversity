# Core and experiments implementation plan

## 2026-10-15 布尔命名统一

范围：维护版自有布尔名称统一为 `_flag` / `_FLAG`，同步调用方和本地测试；保留第三方 API 名称、既有算法和实验配置，不迁移历史结果或归档。

- [x] 保存改前源码并运行完整测试，区分已有问题。
- [x] 修改布尔声明、参数、局部变量和结果键，核对同名非布尔字段。
- [x] 对照改前源码与测试结果，审查调用关系并更新命名及格式变化说明。

完成：改前 191 项通过；改后完整运行 190 项通过，1 项测试旧键引用修正后相关专项 19 项通过。29 个源码文件均核对命名等价，审查发现的摘要白名单与限幅掩码遗漏已修复。

## 2026-10-15 分离单用例接收与评估

范围：只调整现有验证模块，不新增文件或框架，不改接收算法、指标公式与输出格式。

- [x] `run_case()` 仅运行实际链，`evaluate_case()` 集中分量分析、指标和配对；合并原 `_score()` 避免额外跳转。
- [x] 批次循环显式执行接收、评估并释放本次结果，保留最小参考缓存及按 seed 隔离。
- [x] 验证两阶段独立、数组生命周期与四方法 224 行完整结果一致，更新说明。

完成：32 项专项通过，224 行完整摘要、顺序与核验报告严格一致。

## 2026-10-15 用例独立执行

目标：每个用例使用自身完整配置独立运行前端和接收，删除配置分组判断；同一用例的算法共用采样，参考数据仅用于接收后的配对评估。

- [x] 删除物理键与组执行，逐用例调用完整链，缩短非参考大数组生命周期。
- [x] 参考按名称和种子隔离，只保留比较所需数据并在最后消费后释放。
- [x] 验证独立执行、配置生效、配对与内存生命周期；逐身份对照改前 224 行结果并同步文档。

完成：30 项专项通过，224 行按身份对照及完整配置/核验报告一致，审查无阻塞问题。

## 2026-10-15 验证运行器按物理组简化（已由独立执行方案替代）

范围：仅验证运行器与本地测试，保留物理分组、两步相位旋转、接收/评分及输出口径。

- [x] 组内执行收至一个函数，仅返回结果行；分组只保存 case，缩短大数组生命周期。
- [x] 共用注入/逆补偿系数，DFE 恢复误差每条件算一次，参考计数只覆盖实际配对消费者。
- [x] 验证跨组数组释放与原行为，对照改前四方法 224 行结果，更新记录。

完成：29 项专项通过，四方法 224 行完整摘要与核验严格一致，静态复核无阻塞问题；未测量内存降幅。

## 2026-10-15 用例一次构造

目标：配置阶段确定选择、参考顺序、seed 和时长，运行阶段仅执行完整用例并复用物理前端，保存阶段直接记录实际配置；不改接收算法、指标或绘图。

- [x] 核查三个 validation 文件与调用方，保存重构前两场景、相位/方向/增益、两个 seed 的 168 行结果。
- [x] 集中构造最终用例，移除运行器和保存入口的二次补全；迁移本地测试。
- [x] 对照数值、验证用例独立性与实际清单，审查并同步使用说明。

完成：168 行结果与完整核验报告严格一致，完整回归 167 passed、2 deselected，审查无阻塞问题。

## 2026-10-15 统一前端与按需分量评估

**目标：** 参考 FULLCHAIN 的公共阶段组织，统一实际接收入口，合并三份线性分量分析；保留全部用例、数值口径与物理顺序。
**实施方式：** Main 取基准和维护文档，一个 Coder 修改源码与本地测试，Reviewer 只读复核，Main 最终验证。使用 subagent-driven-development 和 test-driven-development；用户已要求完成实现，无需再次确认。
**范围：** 当前 `core/` 和抗干扰入口；不改归档、历史结果、GPS 多星求解算法，不提交本地测试，不增加依赖或通用插件层。

### 已确认的数据流与决定

源波形经方向图/极化/幅相/互耦/通道时延成为天线端电压，目标和干扰在天线端相加；RF 滤波后加入本次 RF 热噪声，再施加 RF 增益与幅相、限幅和 ADC，随后 DDC/FIR/抽取/DFE 量化、合并、捕获跟踪。实际链不读取分量评分结果。

现有真实前端本就每物理条件运行一次，冗余主要是三份线性评估实现、重复噪声生成和 DFE 系数设计。精确分量功率仍需传播各分量的线性响应，不能直接从非线性输出分离；共同残差和交叉项继续单列。

### 实施与验证

- [x] Main 在修改前保存短记录的主链数组指纹和各实验数值：理想/参数化、无干扰/CW/宽带/同时干扰、RF 幅相、非零时延、软硬限幅、DFE 量化开关；另涵盖通用相位/方向/增益、旧相位和直接响应入口。证据只存本地 results。
- [x] Coder 在 `core/rf.py`、`core/pipeline.py`、`core/models.py` 增加显式可选分析真值保留：默认完整链不生成分量分析，只在评估调用时保留实际噪声；不改变物理运算顺序和默认输出。
- [x] Coder 在 `core/dfe.py` 提取共用线性 DDC/FIR/抽取函数，实际量化仍由原入口完成；新增一个职责明确的 `core/evaluation.py`，复用源分量真值、实际 RF 噪声、滤波器与输出尺度，集中线性分量传播。按分量顺序处理，避免把所有 RF 大数组拼接导致内存峰值增长。
- [x] 迁移 validation、channel_phase_validation、antenna_ablation 的重复分量处理；旧相位入口改用公共前端/接收函数，保留自身字段与判据。direct_response 继续复用 ablation。保持独立 ADC 重建核验，不能直接把主链 ADC 值当重建结果；不以每个分量各自量化代替合成量化。
- [x] 先增加有意义的失败测试：默认无分析开销、分析保留不改变实际样本、噪声只生成一次、前端组复用、互耦与RF幅相顺序、饱和/量化残差、不同干扰集合和通道顺序。改后验证这些测试以及现有专项与完整回归。
- [x] Reviewer 审查上述物理节点、真值隔离、缓存身份与指标口径；Main 对照改前基准（真实链要求逐位一致，分析允许仅浮点舍入级误差），更新既有 README/CHANGELOG，报告真实验证范围。

完成：完整回归 164 passed、2 deselected；十组实际链指纹及 138 行评估结果严格一致，旧代码独立核对 48 个分量一致，审查无剩余实质问题。实际条件与命令见抗干扰 README，本地对照数据位于 `results/interference/frontend_refactor/`。

## 2026-10-14：通用抗干扰验证框架设计（已实现）

### 用户目标与本轮范围

用户已选择“短 Python 条件文件＋固定运行器”：后续更换方向、增益、相位、天线组合或扫描点，主要修改条件，不再为每轮验证创建新的实验脚本，也不反复阅读复制的 RF/DFE/捕获/跟踪代码。本轮在当前抗干扰维护版上构建这一能力，保留现有中文注释、独立复算、结果输出与绘图；不恢复已删除的多径或抗欺骗接口。

首次迁移以现有整通道相位验证为数值基准，并增加目标/干扰方向、阵元增益条件。通用运行不等于所有条件都必须恢复同一个基准；条件的物理层级与对应验证判据分别明确。

### 固定文件职责

- `experiments/interference/validation_config.py`：唯一日常编辑位置。使用现有 `SimulationConfig` 及具名条件对象，直接配置目标真实/名义方向、干扰列表、天线模型真实/假设状态、逐通道参数；集中设置种子、时长、方法、条件选择和单因素扫描。不使用 JSON/YAML、点分字段路径或反射式覆盖器。
- `experiments/interference/validation.py`：固定运行入口，支持模块运行和直接运行；只连接条件、运行、保存、绘图及核验结果，不包含另一份接收算法。
- `experiments/interference/validation_support.py`：固定的条件类型、共享运行/评分/核验/绘图函数。放在实验层而非公共物理模型层，避免 `core` 依赖实验或将研究评分塞进公共接收机。配置和入口都依赖此模块，避免循环导入。

上述文件一次建立，后续增加已有参数的条件只修改配置。已有四个实验入口与历史结果先保留，不一次性重写 RF 噪声标定、全量天线消融或源级响应实验；新框架不是简单调度这些旧脚本，而是提取并复用相位验证中的实际处理阶段。必要时只做小范围公共接收阶段提取，保持原 `run_end_to_end` 默认行为。

### 条件与扫描的使用约定

- 一份完整基础 `SimulationConfig` 加具名条件表，条件从基础配置独立复制；前一个条件的改动不能污染下一个条件。
- 普通常量和现有 dataclass 字段承担物理配置。真实方向与名义方向分别设置，不能因为修改真实角度就默默假定接收机也知道新方向。
- DFE 后的整通道相位作为独立实验参数：`dfe_phase_deg`、`dfe_estimated_phase_deg` 使用物理天线名字典；逆旋转作用于整个样本序列，估计值是用户提供的校准量，不是自动估相。配置明确是否补偿，不修改名义导向进行第二次补偿。
- 一次可以选择一个条件或一组条件；扫描默认只改变一个具名因素，例如目标方位、指定阵元增益或指定通道相位。多个条件分别列出，不隐式做全部参数笛卡尔积。
- 任意两至四根天线组合均沿用现有模型；SINGLE 参考按组合第一根明确标记。噪声收益理论使用实际天线数，且只在理想等增益、正确对齐、独立等功率噪声前提下启用。
- 条件可指向具名 `reference_case`，供配对核验使用；执行所选条件时，必要的参考条件也显式记录在输出中。改变真实方向/增益等物理输入的普通条件不自动套用相位闭环的逐样本相等要求。

### 必须区分的物理层级

1. 天线增益/相位：作用在入射信号的阵列响应中，使用天线真实/假设状态；可随方向变化，不能解释成接收机热噪声一同被旋转。
2. RF 增益、NF 与逐通道幅相：沿用现有 RF 模型，作用位置在限幅/ADC 前；不将其与 DFE 后相位注入混为一项。
3. DFE 后通道实验：首次迁移只处理已有的静态相位模型，对完整通道样本共同旋转，包括目标、干扰、热噪声和已有前端残差；不在本轮新增 DFE 后增益校准算法。
4. 通道时延：沿用真实序列时延与估计窄带导向相位的既有区别；新框架不声称实现码包络自动对齐或宽带波束形成。

### 固定处理流程与复用边界

构造条件 → 生成场景/RF/DFE → 应用显式 DFE 通道误差与补偿 → 合并/捕获/跟踪 → 固定权值独立重建与投影分量 → 汇总/核验 → 保存与绘图。

- 纯 DFE 相位或接收假设变化的配对条件复用相同前端输入；真实方向、阵元增益等改变物理场景时重新生成相应前端。相同 seed、通道名和噪声配置保持噪声配对身份，不为取得收益调整功率、NF 或捕获门限。
- 接收复用公共算法，不读取事后分离的真目标/干扰/热噪声分量。研究专用控制与指标留在实验支持模块。
- 独立线性预算按启用干扰列表逐项核算，不固定假定同时存在 CW 和宽带，也不把非线性/量化残差或交叉项称为热噪声。
- 所有条件共有的功率、目标增益、各干扰剩余功率、带内线性 SINR、捕获及跟踪指标自动输出。热噪声收益和实际跟踪 C/N₀ 分开；无跟踪估计保留缺测身份。
- 核验分为公共数学一致性、适用条件下的理想噪声收益、明确选用的相位配对闭环。接收失败是研究结果；核验不适用或失败必须有状态与理由，不能静默记为通过。

### 保存、图和历史身份

统一运行将实际条件、种子、扫描因素、参考关系、来源哈希、逐条件指标和核验结果写入自身输出目录；不默认复制源码或 IQ。方向/增益/相位扫描图使用实际扫描字段与单位，方法及不同指标分开显示，保存并按开关弹出。仅刷新本次明确选择的输出，不删除历史实验目录，不追改旧结果或来源哈希。

### 验收与限制

- 在同条件下对照旧相位入口的前端、权值、分量功率、捕获/跟踪和闭环状态；0°、90°、180°使用独立手推与样本核对。
- 增加真实/名义方向分离、阵元增益电压/功率比例、任意天线顺序、单点/单因素扫描、条件隔离、可用 C/N₀ 缺测拒绝、输出隔离的测试。
- 完整默认回归、独立只读审查及保存图检查完成后，更新既有抗干扰说明和 CHANGELOG；不能使用历史“606 项”等清理前计数冒充当前测试结果。
- 不加防御兜底、插件注册、GUI、新依赖或未出现需求的包装层；不承诺新算法/新物理模型仅改参数即可完成。

用户已确认设计并要求继续实现，按下面的范围直接实施、验证，不再重复确认计划。本轮不提交、推送或合并 Git。

完成情况：三个固定模块已实现，公共阶段提取保持基准数值；全套回归 159 passed、2 deselected，最终绘图调整后专项 17 passed。实际命令、失败修正、实验条件与证据见 `CHANGELOG.md` 的 2026-10-14～15 记录。独立子 Agent 审查被额度限制中断，后续由 Main 接手检查与实现。以下是本轮的具体实施记录，未将原先计划的逐项 RED 流程当作已执行证据。

### 通用验证框架 Implementation Plan

> 一个 Coder 负责实现，Reviewer 只读审查，Main 维护说明并验证最终结果；以下勾选项记录实际完成情况。

**Goal:** 日常只修改一个短 Python 条件文件，即可运行单条件、方向/增益扫描或已有相位闭环，不为每个条件新增脚本。

**Architecture:** 沿用现有物理模型与接收算法，拆出可复用的前端和接收阶段；实验支持层承接条件、独立预算、核验与图。用户条件文件依赖支持层，固定入口依赖两者，`core` 不依赖实验。旧入口保留其既有条件与数据身份。

**Tech Stack:** 当前 Python、NumPy、SciPy、matplotlib、pytest；不增加依赖。

**Spec:** 本文件上方“通用抗干扰验证框架设计”，用户已确认。

**Global Constraints:** 只修改 `python_sim_simplified/`；保留所有已有收口改动和历史记录，不恢复抗欺骗/多径；不加防御兜底、插件或 GUI；不重写旧结果/来源清单；不提交、推送或合并 Git。

**Review Focus:**

- 真实场景与名义导向混用、错误缓存前端：改变真实方向/增益必须影响数据，纯名义方向变化不能影响同种子 IQ。
- 多条件共享可变配置：运行一组条件后不得污染基础配置或其他条件。
- 通道名字/顺序及热噪声基准：二至四天线重排仍对应正确物理通道，理论收益仅在明确前提下评价。
- 缺测/失捕与核验通过混淆：基准可用、比较组缺失必须拒绝，双方失捕不能伪造 C/N₀。
- 结果目录和来源身份：单次运行不能刷新其他实验或历史数据，配置实际值和参考关系必须可追溯。

#### Task 1：提取前端与分支接收，保持默认链路数值

**Files:** 修改 `core/pipeline.py`、`tests/test_pipeline.py`；不修改物理算法文件。

**Interfaces:** `run_frontend(cfg, *, scene=None) -> SimulationResults` 返回场景、ADC、DFE、元数据以及尚为空的 branches/metrics；`run_receiver(dfe_channels, truth, cfg, fs_out_hz) -> dict[str, BranchResult]` 复用现有分支函数；`run_end_to_end` 连接两阶段并生成原有指标。

- [x] 增加可复用阶段测试，覆盖 SINGLE/EGC/MVDR/Blind Null；Main 另外保存改动前数据哈希，再与提取后的四分支数据逐项比较。
- [x] 将已有代码按阶段提取，数值处理不变，复用同一份前端记录。
- [x] 公共阶段相关测试及最终默认回归通过，基准哈希一致。

#### Task 2：统一条件、单因素扫描和独立预算

**Files:** 新增 `experiments/interference/validation_support.py`、`validation_config.py`、`tests/test_validation.py`；如共享现有相位助手，只改 `channel_phase_validation.py` 的相应依赖，不复制另一份接收算法。

**Interfaces:** 支持层定义 `ValidationCase` 与 `ValidationSettings`；配置层提供 `build_settings() -> ValidationSettings`、`build_cases() -> list[ValidationCase]`；支持层提供 `run_validation(settings, cases) -> dict`，结果含条件清单、rows 与 verification。

`ValidationCase` 的显式接口包括 `name`、`config: SimulationConfig`、`scene_label`、`series_label`、`scan_label`、`scan_value`、`scan_unit`、`dfe_phase_deg`、`dfe_estimated_phase_deg`、`compensate_dfe_phase`、`reference_case`、`checks`。图的标签仅用于显示，不使用字符串路径修改配置。`ValidationSettings` 包括 `seeds`、`duration_ms`、`power_trim_ms`、`selected_cases`、`output_dir`、`show_plots`。

- [x] 测试整行 90° 的 `1j*X` 旋转、逆旋转、真实/名义参数分离、6.020599913 dB 的电压/功率比例、条件隔离和参考条件选择。
- [x] 实现上述默认相位网格，以及可选方向/阵元增益示例；配置字段直接赋值，不做反射式修改或隐式笛卡尔积。
- [x] 区分真实物理输入和接收假设，名义导向改变复用 IQ；真实增益改变重新生成前端，已用实际输入数组检查。
- [x] 逐个重放启用的干扰及同 seed 热噪声，分列残差、交叉项、线性 SINR 与跟踪 C/N₀。
- [x] 公共功率恒等式、适用前提下的理想收益及相位配对闭环通过；不适用/缺测保留状态，负测试能够拒绝篡改和单侧缺测。
- [x] 默认回归覆盖旧相位/消融/响应实验；新测试覆盖四类干扰组合、四天线反序及仅运行 MVDR。

#### Task 3：固定入口、输出、图和数值验收

**Files:** 新增 `experiments/interference/validation.py`；完善 `validation_support.py`、`tests/test_validation.py`；Main 更新 `README.md`、`docs/interference/README.md`、`CHANGELOG.md` 及本节状态。

**Interfaces:** 固定入口 `main()` 读取当前 `build_settings()/build_cases()`，运行 Task 2 的 `run_validation`；支持层 `plot_validation(rows, settings) -> list[Path]` 使用实际扫描标签/单位；配置与最终判据保存到 manifest/summary/verification。

- [x] 固定模块入口实际运行，直接路径的导入启动检查通过；短记录测试覆盖 JSON/CSV/图、输出目录隔离、混合配对字段和失败核验保存。
- [x] 按真实因素/单位分图。热噪声/目标/SINR 用线性比保留抵消细节，各干扰抑制单独显示；缺测坐标轴不伪装成零值。默认启用弹图，自动验证仅检查 Agg 保存图。
- [x] 同条件旧/新相位入口在 0/90/180°、seed 42、两个场景共 54 行数值对照通过；不修改旧清单。
- [x] 默认相位批次与短方向/增益示例实际运行；Main 用独立几何公式核对目标功率比，最大绝对差 2.22e-16。
- [x] 实际默认回归及专项通过，Main 检查静态图和差异；来源核对随最终输出更新记录。
- [x] 更新已有 README、抗干扰说明与 CHANGELOG，没有另建实施报告或源码副本。

**Ownership and handoff:** 一个 Coder 顺序完成 Task 1–3 的源码与测试；Main 维护文档并处理已有工作区重叠；Reviewer 只读审查数值保持、物理分层、验证与输出边界。用户已授权实施；本轮使用当前工作区，不提交或推送。

## 2026-10-14: interference-only maintained code

**User scope:** back up the current maintained code, then remove anti-spoofing and multipath implementation so subsequent interference research does not modify those directions. This supersedes the active three-direction layout below; earlier plans and research evidence remain historical.

**Backup:** `archive/pre_interference_only_2026-10-14_156b994/` contains 271 tracked files from the simplified project plus repository instructions/navigation at Git HEAD `156b994bacc9fed03b38ae8b5b4b927e3e5bae11`. `FREEZE_MANIFEST.json` records their SHA256 hashes. Archive and existing result data stay local. The snapshot is the rollback reference; restoring source means copying its root-relative files back, not editing archived implementations.

**Boundaries:** retain common antenna, channel-delay, GNSS L1/L5, RF/DFE, acquisition/tracking, navigation/positioning and recorded-IQ functionality; retain all current interference algorithms and experiment parameters. Remove dedicated spoofing/multipath modules, entries, tools and tests. Remove multipath configuration, scene generation, detection/suppression orchestration, metrics, plot overlays and optional GPS scene inputs from shared code. Do not add compatibility wrappers, fallback branches or new dependencies. Historical reports remain available as historical material; their experiments require frozen source. Existing result bytes and prior archives are untouched. Work in the authorized existing checkout; no commit or push is requested.

### Task 1: implementation and tests (one Coder)

- [x] First update existing layout/contract tests to require interference-only imports and outputs; run them before runtime edits to establish the expected failure.
- [x] Remove `core/spoofing/`, `core/multipath/`, `experiments/spoofing/`, `experiments/multipath/`, `tools/spoofing/` and exclusively dedicated tests. Classify tests by actual imports and behavior, not filenames alone.
- [x] Update `config.py`, `presets.py`, `gnss.py`, `pipeline.py`, `models.py`, `metrics.py`, `plotting.py`, `gps_simulation.py`, `provenance.py`, `experiments/paths.py` and the four interference entries. Retain generic callers and applicable tests. Delete unused dedicated historical source mappings only after checking remaining consumers.
- [x] Run affected tests, then the retained default suite. Source import and reference scans must contain no dependency on removed directions.

### Task 2: current documentation and validation (Main)

- [x] Verify the byte-exact local backup before deletion; capture default suite baseline and ideal/parametric interference numerical fingerprints before runtime edits.
- [x] Update current project instructions, READMEs and status/navigation for interference-only development; keep historical reports and prior CHANGELOG entries unchanged.
- [x] Compare retained scene, ADC, DFE, combining, acquisition, tracking, truth and metrics against both pre-cleanup fingerprints, excluding intentionally removed multipath fields and configuration schema.
- [x] Independently review the final diff, fix substantive findings and reverify; verify all snapshot hashes and archive/results ignore rules. Record actual outcomes and limitations in CHANGELOG.

**Interface audit:** Task 1 owns runtime source/tests; Task 2 owns documentation. Both share the public entry names and removed-directory list above; documentation must follow the final retained contract. Numerical preservation applies to interference-only scenes, not historical echo scenes. Old multi-direction instructions below are historical and cannot require deleted APIs. The cleanup does not establish new performance claims.

**Review ruling:** preserve common PLL convergence/slope/step tests from a deleted spoofing-named test file by moving them into the existing receiver tests. The optional `track(loss_quality=...)` callback had exactly one production consumer, the deleted spoofing receiver; remove that unused hook and its specialized tests. Keep standard causal loss detection used by autonomous tracking. GPS numerical equivalence must compare this cleanup against its verified 2026-10-14 snapshot, not the older 2026-09-16 physical-model version; never change old evidence or algorithms to force agreement.

## 2026-10-12: approved research-direction layout

**Goal:** share the common receiver code, separate interference/spoofing/multipath algorithms and experiments, and update their consumers and documentation together.

**Approved design:** keep common modules directly in `core/`; place dedicated modules in `core/interference/`, `core/spoofing/`, and `core/multipath/`. Mirror those directions under `experiments/`, `docs/`, and `results/`. General GPS/recorded-IQ experiments and `experiments/paths.py` remain shared. Keep filenames and numerical behavior; no forwarding wrappers or duplicate algorithms. `full_chain` belongs to interference as a documented mixed interference/multipath demonstration, with its existing configuration unchanged.

**Global constraints:** work in this existing authorized workspace; modify only the maintained simplified project. Preserve local archive and frozen result bytes, research parameters, source identity checks, and independent verification. Do not commit or push as part of this task. No new dependencies or defensive fallback paths. Preserve general navigation/recorded data locations when not owned by a direction.

### Follow-up ownership correction (2026-10-12)

The first directional pass left six dedicated modules in the shared layer. Re-audit actual consumers, not hypothetical future reuse. Move the SDCP quartet to `core/multipath/`, the measured common-emitter array acquisition/projection pair to `core/spoofing/`, and the five multipath-only plotting functions to `core/multipath/plotting.py`. Keep all function bodies and experiment parameters unchanged. Update imports, documentation, recursive freezes and historical-key file location mapping together; never rewrite old evidence keys or relax byte-identity checks.

`config.py` and `models.py` remain shared contracts; `gnss.py` and `pipeline.py` are joint orchestration; `metrics.py` and the remaining `plotting.py` are common receiver/branch comparisons. A shared orchestrator may call multiple research modules; that does not make those dedicated modules shared. `configs/motion_summary.json` is an old spoofing result summary, not a configuration: preserve its bytes under `docs/spoofing/motion_summary.json`, map the old evidence key, and exclude it from new source/config manifests. Preserve the three pre-existing plotting/test/document changes without reverting them.

Validation: first reproduce missing directional ownership with layout tests, then run affected numerical/plotting/provenance tests, the full default suite, module and link checks, definition-level AST comparison against the pre-correction working files, frozen archive verification, and independent read-only review. Record outcomes in REFACTOR_VALIDATION.md. The following completed ledger describes the initial pass, not this correction.

**Follow-up outcome:** completed without staging or pushing. Two historical report-key lookup errors found by review were reproduced and fixed with strict byte checks, not fallback behavior. Final verification: 556 default tests passed (25 local-evidence tests deselected), three independent old/new numerical comparisons passed, all 124 modules imported, 909 file links resolved, and all 584 frozen original files matched. The pre-existing plotting definitions/tests and interference documentation were preserved. Details are in REFACTOR_VALIDATION.md.

### Task 1: move dedicated code and update active references

- [x] Capture the current default test baseline before editing: 528 passed and 6 Windows-sandbox atomic-replace errors; those same 6 tests passed outside the sandbox, for 534 passing default tests total. 25 local-evidence tests were deselected.
- [x] Move dedicated core modules, directional experiment entries, and specialized diagnostic tools; add package initializers. Keep shared RF/DFE/tracking/GPS primitives shared where used across directions. The array pair was initially misclassified; see the follow-up ownership correction above.
- [x] Move dedicated echo/interferer scene helpers and their small result dataclasses out of shared `gnss.py`; move blind-null combining and multipath-specific LCMV combining out of shared `diversity.py`. Preserve function bodies and shared scenario/pipeline orchestration; do not add old-path re-export wrappers.
- [x] Update absolute/relative imports, dynamic imports, script bootstraps, test paths, monkeypatches, and provenance to recursively cover the maintained nested code. No old import compatibility shims.
- [x] Classify new outputs and known existing result trees by direction; update current consumers. Move existing results without editing frozen contents; leave unclassified and general positioning inputs in place.
- [x] Organize current and previously temporary research Markdown under matching `docs/` directories, repair active links and commands. Keep historical commands/source maps explicitly historical.
- [x] Update README, local project instructions and existing refactor documentation to describe the final paths.
- [x] Run targeted layout/import/path tests, then the default suite; run practical smoke checks in temporary outputs without overwriting saved research evidence.

### Task 2: independent review and final verification

- [x] Reviewer checks import boundaries, output reset paths, moved report links, current/legacy provenance separation, and source freeze completeness.
- [x] Correct substantive findings and rerun affected checks; Main runs final verification and reports any evidence-only limitations.

**Review focus:** nested source files omitted from freeze; wrong `__file__` parent after a move; reset deleting a direction directory instead of one experiment; historical manifests silently rewritten; replay input moved without consumer updates; existing source-byte gates accidentally weakened; default tests depending on ignored local evidence.

**Progress:** completed without staging, committing, or pushing. Initial workspace clean at `326d3f7`; two read-only audits and independent review completed. Main took over after the implementation agent became unavailable, corrected the review findings, and verified 539 default tests plus three independent old/new numerical comparisons. All 77 existing result trees retain their metadata fingerprints and selected root summary/manifest/plan/snapshot byte hashes; 37 trees moved. All maintained modules import and checked document links resolve. Deleted validation evidence is described explicitly as historical, with no substitute results. Exact commands and verification limits are in REFACTOR_VALIDATION.md. The earlier sections below describe the previous completed refactor and remain historical.

> For agentic workers: use subagent-driven-development for independent implementation and review; preserve the approved numerical behavior.

**Goal:** one maintained core package and one independent entry per current experiment, with the original research tree frozen once.

**Architecture:** `core/` owns physical models, RF/DFE, acquisition/tracking, observations, detection and positioning. `experiments/` owns scenarios, CLI, evaluation and plots. Neither core nor current tests depend on frozen research drivers. `archive/pre_refactor_2026-09-16/` preserves the original top-level files and tests byte-for-byte; existing large results/validation directories stay in place.

**Tech stack:** existing Python, NumPy, SciPy, matplotlib and pytest only.

**Spec:** user-approved scope in this conversation: RF/DFE/acquisition/tracking, SDCP/multipath, latest single-source receiver, GPS positioning and recorded IQ. Keep algorithms and parameters unchanged. Daily runs do not duplicate source; explicit final freeze remains available. No additional defensive branches.

## Execution and verification

- [x] Freeze 584 original files including untracked document builder; verify SHA256 against every source.
- [x] Capture original `python -m pytest tests -q` baseline: 401 passed, 1 xfailed. Unqualified pytest originally had 139 historical-copy collection errors; fixed discovery in new tree.
- [x] Extract common numerical code with relative package imports; separate GPS simulation/receiver functions from command drivers.
- [x] Extract current single-source receiver and its actual dependencies; frontend/navigate and generated RF match frozen numerical outputs.
- [x] Migrate SDCP baseline/multipath/window-change experiments and independent verification; source copies are explicit opt-in.
- [x] Provide independent GPS SPP, L1/L5, recorded IQ, full-chain and single-source experiment CLIs; retain result arrays, summaries and plots.
- [x] Remove inactive original Python files from active root after snapshot verification. Preserve historical Markdown locations and original results/validation evidence.
- [x] Port applicable existing tests, add package-boundary and old/new numerical comparisons. Current discovery contains 158 tests.
- [x] Independent review completed; full-chain output overwrite risk corrected. Final main suite 149 passed plus 9 additional migrated tests passed. See REFACTOR_VALIDATION.md.

## Decisions / ledger

- Work in the authorized shared project rather than duplicate large local RF archives into a separate checkout. All implementation is confined to python_sim_simplified; the verified pre-change snapshot is the rollback reference. No commits or shared-branch operations are necessary.
- Follow-up authorization: commit and push the completed refactor; archive stays local and ignored. Default tests must not depend on local archive/results; explicit local_evidence tests retain the old/new comparisons.
- Preserve existing algorithm modules where coherent; do not combine unrelated methods just to achieve the illustrative eight-file diagram.
- Frozen legacy execution is explicit, separate from active pytest discovery. The snapshot includes historical tests and report builders for restoration; historical imports must never leak into the active core.
- Shared interfaces: core.gps_simulation.receiver_reference/build_scene/satellite_setup/run_record; core.single_source.frontend/navigate; core.single_source_scene.Case/generate; core.provenance.source_hashes/freeze_source; core.serialization.serializable.
- Parallel ownership: receiver task owns six receiver-specific core modules and receiver regression test; SDCP task owns its two experiment drivers, two verification tools and core.sdcp_observation; coordinator owns all other core modules, packaging, freeze, remaining experiments and integration.
