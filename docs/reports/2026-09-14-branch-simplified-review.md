# 全分支精简版合规与问题审查

日期：2026-09-14。依据：当前根目录 [AGENTS.md](../../AGENTS.md) 的“精简版开发规则”。结论：**当前 main 及未提交工作区存在明确不符合项，问题不只在最近的 sensor_contract。** 早期精简分支也有部分参数防御层；最初 simplified-python-copy 分支最接近直接研究流程。

本轮只审查、运行测试、记录证据，没有切换现有工作树、删除功能、改写算法、修改历史归档或推送。该规则是当前整改依据；未假定每段历史代码写入时就已经存在同一规则。

## 1. 范围与方法

已执行 `git fetch --all`。枚举 4 个本地分支、9 个实际远端分支，共 13 个引用、11 个不同提交；`origin/HEAD` 是符号引用，不另计分支。检查两个现有工作树，第二工作树无未提交改动；主工作树保留原有未提交修改，并额外审查三个新增 Python 文件。

含精简版的提交按 Python blob 去重扫描，共 178 个不同 Python 文件版本（包括测试）；全部语法解析通过。检索异常捕获、参数校验、默认回退、空值分支，随后按职责人工分类。候选命中数不等于违规数；没有声称逐条分支路径都获得动态覆盖。

不含精简版的分支只对历史 `python_sim/` 做存在性与语法检查，所有分支该目录合计 151 个不同 Python 文件版本，未发现语法错误；没有对历史完整版做完整功能审查或要求同步整改。

对 3 个早期精简版独立代码版本，在临时目录按 Git blob 导出 Python 源码及测试执行回归；main 工作区直接回归。main 与 origin/main 的精简版已提交代码相同，仅文档提交不同；重复引用不重复测试。快照方式不影响现有分支/工作树，也没有运行生成完整 RF 归档的长实验。

| 分支引用 | 提交 | 精简版源文件／测试文件 | 审查结论 |
|---|---|---|---|
| main | 68e38158 | 85／44；工作区另增 2 源文件＋1 测试文件 | 多处明确不符合；新增契约层最集中 |
| origin/main | 04ce4557 | 85／44 | 已提交精简版与 main 相同，不含未提交契约层 |
| codex/multipath-detection-suppression | 4f465158 | 18／14 | 宽带干扰、分数延迟等参数防御层 |
| origin/codex/multipath-detection-suppression | 4f465158 | 18／14 | 同一提交，结论相同 |
| origin/codex/simplified-research-refactor | 4799c9e7 | 16／12 | 分数延迟参数校验；主体较直接 |
| origin/codex/simplified-python-copy | 0505387c | 12／10 | 本轮未确认需整改的通用防御层；不能据此保证无其他问题 |
| codex/tracking-fixed-nco | 014f5b68 | 无精简目录 | 历史完整版；不适用本轮精简版合规判断 |
| origin/codex/tracking-fixed-nco | faa24ba0 | 无精简目录 | 同上，单独检查不同提交的目录/语法 |
| feature/antenna-diversity | f04d5924 | 无精简目录 | 历史完整版 |
| origin/feature/antenna-diversity | f04d5924 | 无精简目录 | 同一历史提交 |
| origin/codex/antenna-realism | 1b40abb9 | 无精简目录 | 历史完整版 |
| origin/codex/pre-acquisition-blind-nulling | 6fe6f93d | 无精简目录 | 历史完整版 |
| origin/codex/spatial-research | f640b9ee | 无精简目录 | 历史完整版 |

“源文件”包括实验/audit 脚本，不仅接收机运行模块。完整文件、blob 与候选行号见 [inventory.json](data/2026-09-14-branch-simplified-review/inventory.json)。

## 2. 已确认发现（按整改优先顺序）

### F01 — 高：新契约层整体偏离精简开发要求

位置：[sensor_contract.py](../../python_sim_simplified/sensor_contract.py)，主要为第 11、20、28、33、40、64、74、149、213 行起的异常包装、字段/类型/数组检查及适配包装；245 行内有 47 处 `raise`、2 处异常捕获。对应 [test_sensor_contract.py](../../python_sim_simplified/tests/test_sensor_contract.py) 的多项测试固化了错误输入拒收行为。

影响范围：仅 main 未提交工作区，不在任何已提交分支中。功能本身就是防误用校验层，与 AGENTS 明文要求冲突；先前称其为首阶段必要工作，是本轮代理对开发范围的误判。不能以“19 项新测试通过”证明它符合精简规则。

建议：移除通用 ContractError、字段/类型/空值/可用时间包装等防御层及专门验证这些包装的测试；如果继续保留研究数据适配，只留下必要字段到数组的直接转换、误差模型/真实实验所需的明确输出。不要把完整导航准入系统塞进这次精简版数据准备。

### F02 — 高：部分实验脚本丢失求解失败原因

位置：[audit_antenna_selection.py](../../python_sim_simplified/audit_antenna_selection.py) 第 40–41 行；[audit_gf_pvt_lifecycle.py](../../python_sim_simplified/audit_gf_pvt_lifecycle.py) 第 58–61 行；[audit_sdcp_positioning.py](../../python_sim_simplified/audit_sdcp_positioning.py) 第 99–102 行。

这些位置对求解器的所有 `ValueError` 统一转 NaN 或 `pass`。`solved` 等统计保留了失败数量，**因此不能说它们把所有失败伪装成成功**；但配置错误、数值异常和预期的观测不足被混成同一结果，具体原因消失，不符合“不能宽泛捕获／静默跳过”的规则，也妨碍解释研究结果。

已做最小复现：仅在内存中将 `audit_antenna_selection.solve_position` 替换为抛出 `ValueError('injected unexpected solver regression')`，运行真实 LOS 归档。三种模式都正常返回 `solved=0, epochs=40`，异常原因没有传出或记入结果。见 [reproduction.json](data/2026-09-14-branch-simplified-review/reproduction.json)。这证明会丢失原因，**不证明历史实验实际发生过该注入错误**。

影响：main 与 origin/main。建议直接暴露非模型定义的求解错误；如果实验明确研究观测不足/秩亏，按算法定义产生明确状态及原因，不能使用通吃 ValueError 的回退。

### F03 — 中：宽带波形入口存在完整通用参数防御层

位置：[interference.py](../../python_sim_simplified/interference.py) 第 9–49 行 `_validate_wideband`，调用在第 63 行附近；包括样本数类型、正值、有限性、布尔/整数区别及随机种子检查。测试 [test_wideband_interference.py](../../python_sim_simplified/tests/test_wideband_interference.py) 第 70 行起有非法配置测试。

影响：本地/远端 multipath-detection-suppression、main、origin/main，相同实现继承。通用参数检查可删除；波形功率归一化、滤波、带宽定义本身必须保留。Nyquist 与滤波器条件应写明模型前提；不把相关数值风险通过默认值修复。

### F04 — 中：基础计算函数重复验证类型、形状、空值和有限性

位置及代表行：

- [channel_delay.py](../../python_sim_simplified/channel_delay.py)：19–22、70–71，延迟/滤波长度/通道形状检查；出现在 refactor、multipath、main 相关引用。
- [positioning.py](../../python_sim_simplified/positioning.py)：10–15 `_sigmas` 异常再包装；28–31、61–64、73–77、96–109 等输入契约检查；main 相关引用。
- [sdcp.py](../../python_sim_simplified/sdcp.py)：26、34、36、52、69、86 附近的输入验证；main 相关引用。
- [spatial_spoofing.py](../../python_sim_simplified/spatial_spoofing.py)：22–26、44–47、60–72 等；[spatial_direction.py](../../python_sim_simplified/spatial_direction.py)：55–57、79–82 等。应分别审查通用参数拒收与核心不可观状态，不整体删除算法。
- [recorded_receiver.py](../../python_sim_simplified/recorded_receiver.py)：20–23、31–34、49、67 的文件窗口/截断/维度保护；属于当前规则明确禁止主动扩展的范围。真实样本截断问题应由实验前提或实际证据说明，不能默默补齐。

建议按函数清理包装和重复检查，保留计算、明确实验状态与数学条件。不能机械删除全部 `if` 或 `raise`。

### F05 — 中：缺失关键配置时静默选择另一种研究条件

位置：[audit_spatial_diversity_rf.py](../../python_sim_simplified/audit_spatial_diversity_rf.py) 第 145 行 `payload.get('positions_m', mate70_phase_centers())`。缺少坐标即自动用原手机阵列；这是历史兼容意图，但不符合当前“不增加默认值回退”要求。对未明确记录几何的输入，可能在未显式确认实验条件的情况下使用另一阵列。

建议把旧格式实验的尺寸在调用配置中明确提供；保留原实验的可重现性，不修改冻结数据。另见 [gps_observables.py](../../python_sim_simplified/gps_observables.py) 第 52 行 `eph.get('health',0)`：字段缺失默认健康，属于应去掉的字段回退；这不意味着当前正常 LNAV 解码实际漏了该字段。

影响：main、origin/main。显式可选参数的默认值（例如 `cfg=None` 表示使用指定默认配置）与缺字段回退不同，不一并判违规。

### F06 — 低：跟踪中存在不可达的重复防御判断

位置：[tracking.py](../../python_sim_simplified/tracking.py) 第 193 行先对 `n_epochs < 1` 返回；第 249–250 行在没有改变 `n_epochs` 的路径上再次检查同条件并抛异常，后者不可达。

影响：main、origin/main。建议去掉重复判断。保留 L1/L5 处理差异、Costas/atan2 选择、有效观测时间和真实失锁实验的语义，不因发现重复判断删掉整个跟踪状态流程。

### F07 — 中：部分测试将防御性接口当成精简版必须保留的功能

除 F01/F03 外，[test_spatial_spoofing.py](../../python_sim_simplified/tests/test_spatial_spoofing.py) 第 60–79 行混合了时间缺口实验与形状/类型/非法参数拒收；[test_tracking_replica.py](../../python_sim_simplified/tests/test_tracking_replica.py) 第 34 行测试带宽拒收。

建议清理代码时同步区分：直接删除只为通用参数保护而存在的测试；保留会影响研究结论的物理/数学实验，例如功率、频谱、相位、时延、同源关系、检测率、失锁与恢复状态。不能因为某个包装已有测试就继续保留，也不能把所有异常测试一律删除。

### F08 — 中：项目规则仅存在于当前未跟踪文件中

根目录 `git status` 显示 `?? AGENTS.md`，所有 13 个分支引用的根目录均没有已跟踪的 AGENTS.md。当前任务应遵守已读取的文件，但新克隆不会自动获得它；切换到另一既有工作树也不能假定已有同份文件。

建议将根目录规则作为项目协作文件纳入版本控制，之后让实际开发分支获得该规则；不重写历史分支来制造“过去已经包含”的记录。`docs/knowledge/AGENTS.md` 只对其子目录生效，不能替代根目录项目规则。本轮没有擅自提交这份用户文件。

## 3. 明确不应一刀切删除的内容

- RF/ADC 的饱和、削顶、量化、定点截位，载波取模，以及有限长度延迟模型规定的零延拓。这些属于模型定义。
- `channel_delay.py` 的零延迟直接复制、整采样延迟分支和超出有限记录的零输出，是已声明有限序列模型的一部分，不等同缺字段兜底。
- 导航几何秩亏、不收敛、NH 同步模糊、有效相位弧及可信参考缺失等会影响研究结论的状态，应先明确算法前提，再决定最小必要表达；不能全部改成看似正常结果。
- `detection_calibration.py` 中样本数不足以支持指定尾概率的统计限制，以及零试验数无法给出置信区间的事实；这些应明示，不能用默认数值补齐。
- `cfg=None` 的公开配置选择、按干扰类型选择波形、关闭某实验分支时绘图不包含该分支，是功能选择，不仅因使用 `isinstance`、`.get` 或 `if` 就判违规。
- 先前用户明确要求的告警、输出保护、unknown 和恢复研究功能，与“开发时不增加通用防御代码”是不同层面；现有规则要求保留明确请求的研究功能。
- 输出归档的不可覆盖操作应尊重冻结证据要求。可优先用明确的新输出目录与文件创建语义，而不是为通用保护增加包装；不能为了精简破坏已保存实验。

## 4. 验证结果与限制

| 实际运行对象 | 结果 | 说明 |
|---|---|---|
| main 当前工作区 | 216 passed，115.61 s | 1 条既有 tracking.py 无噪声 C/N0 除零警告；包含未提交的新契约测试 |
| simplified-python-copy 快照 | 23 passed，24.02 s | 从 0505387c 导出代码和测试 |
| simplified-research-refactor 快照 | 37 passed，59.55 s | 从 4799c9e7 导出代码和测试 |
| multipath-detection-suppression 快照 | 67 passed，63.62 s | 从 4f465158 导出；对应本地和远端同一提交 |

测试结果见 [branch_tests.json](data/2026-09-14-branch-simplified-review/branch_tests.json) 和对应日志；main 工作区结果见 [summary.json](data/2026-09-14-branch-simplified-review/summary.json)。通过测试只能说明既有行为符合当前测试，不能证明符合 AGENTS 精简规则。

原目录所有算法代码本轮均保持原样；也没有添加新的防御性实现或用新测试固化这些不符合项。本次注入复现只在独立 Python 进程内替换函数引用。测试产生的临时快照仅用于查看历史分支，不作为新开发副本或新分支。

建议整改顺序：先处理未提交的 F01 → 去掉 F02 的原因吞噬 → 清理 F03/F04/F05 的通用保护和隐式回退 → 同步精简 F07 测试 → 去掉 F06 重复项 → 单独版本化 F08 项目规则。每一批只改当前精简版源码，重跑对应物理/算法对照；历史归档保留原字节，已有研究结论若受影响应另发对照报告。
