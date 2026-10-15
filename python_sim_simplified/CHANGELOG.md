# 开发迭代与验证记录

记录每次修改、实验迭代和验证的目的、实际改动、默认行为、实验条件、验证结果、结论与限制，不等到上库时才补。新记录放在最前面，旧记录及其条件保留；同一迭代的失败、修正和复验归入同一条记录。详细研究说明复用各方向既有文档，并在记录中链接，不另建重复报告。每条记录与对应修改一起提交；提交编号可通过本文件的 Git 历史查询。此前提交保留在原有 Git 历史中。

## 2026-10-14～15 通用验证框架

- 目的：按用户选择，将日常可改条件集中在短 Python 配置文件，运行、独立评分、配对核验和绘图长期共用；不继续为每轮条件变化新建实验脚本。
- 修改后代码位置：维护根目录为 `F:\antenna_diversity\python_sim_simplified\`；日常修改 `experiments\interference\validation_config.py`，固定运行入口为 `experiments\interference\validation.py`，共享实验处理位于 `experiments\interference\validation_support.py`，公共接收阶段位于 `core\pipeline.py`。以上相对路径均以该维护根目录为起点。
- 实际改动：新增 `experiments/interference/validation_config.py`（日常条件）、`validation.py`（固定入口）、`validation_support.py`（共享实验运行/独立预算/核验/绘图）；`core/pipeline.py` 提取 `run_frontend` 与 `run_receiver`，原 `run_end_to_end` 继续组合这两阶段。具名条件支持单点、相位/方位/阵元增益扫描、两至四天线、真实/名义参数分离、自动纳入参考条件，以及各启用干扰源的独立功率预算。增加 `tests/test_validation.py` 并补充 pipeline 阶段对照。保留已有抗干扰收口修改、四个旧实验及历史记录。
- 默认行为：新入口默认 TL/BR，42/43/44 三个 seed，100 ms，功率统计两端各去掉 10 ms，0/30/60/90/120/180°，无干扰/单 CW，基准/未补/准确补偿，SINGLE/EGC/MVDR；36 条具名条件、324 行方法结果。旧完整接收入口默认值不变。`SELECTED_CASES` 可只选某一补偿条件，自动纳入其基准；真实方向与名义方向分别编辑。
- 改动前数值基准：20 ms、seed 42、默认四天线和 CW＋宽带，跟踪上限 12 历元，记录场景/ADC/DFE、四个方法的权值、合并序列、捕获及跟踪哈希，供本轮提取公共阶段后逐项比较。首次检查脚本使用 JSON 序列化复数数组失败，改用数组字节和 pickle 序列化后基准记录成功；这是检查脚本问题，没有修改仿真算法。
- 修正与复验：检查实现时修正 NumPy 布尔值可能掩盖核验失败、理论 10log10(N) 前提不足、参考条件顺序依赖、补偿字段导致 CSV 列不一致、不同扫描因素混轴，以及无 SINGLE 接收分支时仍需计算物理参考评分的问题。双方 C/N₀ 缺测标为未评价，单侧缺测或有限样本数变化拒绝配对通过；整段 DFE、合并输出与权值也参与闭环核验。保存图 QA 后拆出每个干扰源的抑制图，功率/SINR 使用线性比显示抵消，保留 CSV 中的 dB 数值；缺测子图保留实际横轴并明确无估计。
- 实际测试：Coder 报告公共阶段相关 20 项测试通过。Main 执行 `python -B -m pytest -p no:cacheprovider -q tests/test_validation.py --basetemp=results/_pytest_validation_finish`：16 passed；随后全套 `python -B -m pytest -p no:cacheprovider -q --basetemp=results/_pytest_validation_framework_final`：**159 passed、2 deselected，207.97 s**。最后仅调整绘图和注释后执行 `python -B -m pytest -p no:cacheprovider -q tests/test_validation.py --basetemp=results/_pytest_validation_plot_final`：**17 passed，25.22 s**。新旧相位入口的 0/90/180°、seed 42、20 ms、无干扰/单 CW 共 54 行对照通过；测试覆盖仅 CW、仅宽带、两者同时及无干扰、四天线反序、仅运行 MVDR、真实/名义条件的前端复用、负核验与导出失败状态保存。
- 已核验结果：提取阶段前后所有基准哈希一致。初次默认批次 324 行、450 个逐行/配对/理论检查通过；DFE、合并输出、权值恢复最大相对误差分别为 `1.0552e-16`、`8.7018e-16`、0，无干扰 EGC 热噪声收益 2.99863～3.00730 dB。方向/增益示例使用 seed 42、20 ms、trim 2 ms、真实方位 30/40/50°、BR 增益 0/6.020599913279624 dB，共 30 行通过；Main 从几何坐标独立复算 EGC 目标功率比，与数据最大绝对差 `2.22e-16`。结果分别在 `results/interference/validation/`、`results/interference/validation_examples/`；绘图调整后正在用最终源码刷新这两批结果。
- 最终输出复核：上述两批已用最终源码重新完成，分别为 324 行/450 项检查和 30 行/30 项检查，全部通过；分别保存 4 张和 8 张图，两个 manifest 的 47 个源码哈希均与当前文件一致。已查看默认批次的目标功率/SINR、接收指标、独立干扰抑制图，缺测明确标注，未改变信号强度或 NF。450 项是逐行、理论及配对检查数，不是独立实验数。
- 说明与限制：用法及具体条件见 [抗干扰说明](docs/interference/README.md#2026-10-14-通用验证框架)，设计见 [现有计划](REFACTOR_PLAN.md)。自动运行使用 `MPLBACKEND=Agg`；默认 `SHOW_PLOTS=True` 保留桌面弹图，未实际验证 GUI 交互。没有新增自动估相、宽带时延对齐或接收算法。子 Agent 的独立审查因额度限制中断，由 Main 接手剩余检查和修改；未把中断的审查写成通过。未提交、推送或合并。
- 上库复核（基准 `156b994`）：本次按用户要求，将抗干扰收口、通用验证框架及位置说明一并整理上库；上条“未提交、推送或合并”是实现阶段的状态记录。最终执行 `python -B -m pytest -p no:cacheprovider -q --basetemp=results/_pytest_upload_interference_final`：**159 passed、2 deselected，171.71 s**。独立只读审查新框架的物理前端缓存、真实/名义参数、整通道相位、独立评分、参考关系、缺测状态与导出，未发现实质阻塞。Main 重新核对两个保存批次各 47 个源码哈希与核验通过状态，并核对 271 个备份文件哈希；理想/参数化两组 20 ms 抗干扰完整输出与清理前基线一致，全部 74 个维护源码/测试文件语法解析通过，当前 README 的 45 个本地链接有效。没有重跑完整研究批次；归档、结果和缓存均排除提交，暂存差异与远端状态另作提交前检查。

## 2026-10-14 备份并收口为抗干扰维护代码

- 目的：按用户要求先冻结当前精简版，再移除抗欺骗与多径相关实现，使后续抗干扰修改只涉及当前维护链路。
- 修改后代码位置：当前维护代码位于 `F:\antenna_diversity\python_sim_simplified\`，本次直接在这个目录内完成清理；公共源码在其 `core\`，四个抗干扰实验在其 `experiments\interference\`，后续开发继续修改这些位置。
- 归档源码位置：清理前完整精简版源码位于 `F:\antenna_diversity\python_sim_simplified\archive\pre_interference_only_2026-10-14_156b994\python_sim_simplified\`；备份清单 `FREEZE_MANIFEST.json` 位于其上一级 `pre_interference_only_2026-10-14_156b994\`。
- 本地备份：起始工作区干净，Git HEAD 为 `156b994bacc9fed03b38ae8b5b4b927e3e5bae11`；`archive/pre_interference_only_2026-10-14_156b994/` 保存精简版全部已跟踪文件及根目录协作规则、状态和导航，共 **271 个文件**，每个文件的 SHA256 记录在 `FREEZE_MANIFEST.json` 并逐项核验。不复制大型结果数据；既有归档和结果保持本地身份，备份不提交、不推送。
- 实际清理：删除 `core/spoofing/`、`core/multipath/`、`experiments/spoofing/`、`experiments/multipath/`、`tools/spoofing/` 及专用测试，共 **85 个专用源码/入口/工具文件、89 个专用测试文件**；维护源码文件数由 127 减为 42。公共配置、信号生成、链路编排、结果类型、指标、绘图与 GPS 场景移除多径字段和分支。移除仅由已删除工具使用的历史源码定位映射、专用输出路径常量和 `track(loss_quality=...)` 自定义回调；四个抗干扰实验同步去掉相关引用。通用天线、RF/DFE、GNSS L1/L5、捕获跟踪、标准失锁重捕、导航/定位和录波接收继续保留。
- 行为与说明：`full_chain` 仍比较 SINGLE、EGC、MVDR、Blind Null，没有重新开启回波/检测/多径 LCMV 的开关；旧配置或脚本中被删除的字段不再受支持，恢复对应研究需使用冻结源码。当前 README、协作规则和项目状态同步抗干扰范围；历史报告、结果、旧 CHANGELOG 条件和来源哈希不追改。本次清理没有新增研究性能结论。
- 清理前验证：`python -B -m pytest -p no:cacheprovider -q --basetemp=results/_pytest_interference_cleanup_baseline` 为 **606 passed、25 deselected，229.42 s**。另用 `python -B -` 运行理想/参数化天线各 20 ms、跟踪上限 10 历元的完整抗干扰链，保存场景、ADC、DFE、合并、捕获、跟踪、真值及指标的数值指纹至备份内 `INTERFERENCE_BASELINE.json`；仅排除配置结构和有意删除的多径键。
- 失败、修正与复验：新的移除契约测试先在旧实现上复现失败，清理后针对性测试 **48 passed**，首轮完整回归 **136 passed、2 deselected**。独立审查发现被删的抗欺骗命名测试文件中有公共 PLL 测试，将四项收敛/斜率/阶跃测试及辅助函数迁入既有 `test_receiver.py`；自定义质量回调的唯一生产调用者已删除，去掉该接口，标准失锁判据与计数不变。修正后跟踪相关测试 **12 passed**，复审无剩余实质问题。首次 GPS 对照更早的 2026-09-16 冻结版两频段均不一致；该版本跨越既有模型更新，改用本次清理前快照作为相同版本基准，没有修改定位算法或旧证据强迫相等。子解释器补 `-B`，清除本次快照内生成的三处字节码缓存，271 个原文件哈希仍全部一致。
- 最终测试：`python -m pytest -q` 为 **141 passed、2 deselected，161.44 s**；`python -m pytest -m local_evidence -q tests/test_frozen_equivalence.py` 对清理前版本的 GPS L1CA/L5Q 完整接收及定位数值比较为 **2 passed，90.24 s**。Main 另执行 `python -B -m pytest -p no:cacheprovider -q tests/test_receiver.py tests/test_autonomous_tracking.py tests/test_tracking_replica.py tests/test_core_layout.py --basetemp=results/_pytest_cleanup_main_verify`，为 **15 passed，2.30 s**，覆盖公共环路、自主失锁重捕和移除后的包边界。
- Main 独立验证：用 `python -B -` 将最终理想/参数化抗干扰输出与清理前 `INTERFERENCE_BASELINE.json` 对照，两组场景、ADC、DFE、权值/合并、捕获、跟踪、真值及指标逐项完全一致（仅排除有意删除的多径键和配置结构）；另用独立解释器比较清理前/后的 GPS L1CA/L5Q 双 PRN 20 ms 场景数组及信号设置，也完全一致。42 个维护模块逐个解析和导入成功；删除目录不再存在，无已删除包/字段/模式/回调的运行引用；当前导航 54 个本地链接有效。`git diff --check` 通过，271 项备份哈希全部匹配，归档与结果仍被忽略；历史抗欺骗、多径报告无差异。
- 结论与限制：保留链路在上述对照条件下数值不变，不据此推广新的抗干扰性能结论。历史抗欺骗/多径实验未重跑，历史结果没有覆盖或追改；测试输出只写本地临时测试目录，未重新生成现有研究批次。实施边界见 [重构计划](REFACTOR_PLAN.md#2026-10-14-interference-only-maintained-code)，当前接收链说明见 [抗干扰文档](docs/interference/README.md)。本轮未提交或推送，全部改动留在工作区。
- 路径说明补充：仅补充本条记录的当前维护代码和归档源码绝对路径，核对两处目录及其源码子目录存在，`git diff --check -- python_sim_simplified/CHANGELOG.md` 通过；没有修改代码，未重跑测试或仿真。

## 2026-10-14 相位验证入口中文注释完善

- 目的与实际改动：只补充 `experiments/interference/channel_phase_validation.py` 的中文注释和函数说明；解释输入输出、矩阵维度、码值/功率单位、整行相位旋转与逆补偿、线性 RF/DFE 分量重建、共轭合并与 `wᴴRw` 功率关系、配对判据、C/N₀ 缺测和绘图统计。明显的赋值和循环不逐行重复说明；没有改算法、参数、控制流程或增加保护分支。
- 实际验证：`python -B -c '<AST/compile 检查代码>'` 按 UTF-8 读取源文件，内存编译语法检查通过；去除文档字符串后的 AST SHA-256 注释前后均为 `c210e0bb8cde5bf67f4c480d0fe76070b51c3d8c2d10b8b696aafa23a3d5d31f`，确认可执行逻辑未变。首次用 PowerShell 文本管道读取时因编码失败，改为明确 UTF-8 读取后检查通过；`git diff --check` 无空白错误。
- 行为与范围：本轮是注释更新，没有新增实验结论，未重跑 pytest、相位扫描或完整链路；原结果及当时的来源哈希保留，不因注释变动追改旧清单。详细原理与既有验证仍见 [双天线整通道相位补偿闭环](docs/interference/README.md#2026-10-14-双天线整通道相位补偿闭环)。
- 上库复核（基准 `910fa45`）：明确 UTF-8 读取工作文件及 Git 基准，内存编译通过；去除文档字符串后的 AST 完全相同，SHA-256 为上述 `c210e0bb…a3d5d31f`。`git diff --cached --check` 通过；仅提交注释与记录，不重跑测试或仿真，结果与归档仍保留本地。

## 2026-10-14 双天线整通道相位补偿闭环

- 目的：建立理想双天线无误差基准，验证整通道静态相位误差、准确补偿与单 CW 条件下的配对恢复；与历史仅改变目标响应的扫描明确分开。
- 范围：新增独立实验和对应测试，不改变公共算法、`full_chain` 默认配置或历史扫描；误差明确放在 DFE 输出后，对目标、干扰、噪声和已有前端残差一起旋转。准确相位人为已知，不是自动校准；RF/ADC 前误差留待下一项独立验证。
- 实际改动与条件：新增 `experiments/interference/channel_phase_validation.py` 和 `tests/test_channel_phase_validation.py`；按配置、数据生成、整行旋转/补偿、接收、独立评分、核验与绘图组织，顶部参数可编辑。默认 TL/BR、扰动 BR，seeds 42/43/44、100 ms、0/30/60/90/120/180°，无干扰/单 CW，目标功率/NF/门限不变；6 份 RF/DFE 基准、36 个配对条件、324 行方法结果。运行入口及指标定义同步 README，未新增防御兜底。
- 修改前基础检查：`python -B -m pytest -p no:cacheprovider -q tests/test_rf_dfe.py tests/test_pipeline.py tests/test_antenna.py --basetemp=results/_pytest_phase_baseline` 为 **31 passed**；未据此宣称新实验通过。
- 首轮与修正：模块缺失时 RED，随后修正评分更新顺序和 Path/numpy bool JSON 序列化，针对性测试 4 passed；首次 324 分支批次的闭环核验通过，首次完整回归 **605 passed、25 deselected**。独立审查复现配对 C/N₀ 单侧缺失仍误通过，先负测试后补跟踪状态、可用性和有限计数核验；图 QA 后将热噪声收益改为线性倍率显示，表格保留 dB，增加缺测图注及统一相位横轴。针对性测试最终 `python -m pytest tests/test_channel_phase_validation.py -q` 为 **5 passed**。
- 最终验证：`MPLBACKEND=Agg` 下 `python -u -B -m experiments.interference.channel_phase_validation` 退出码 0，21 项核验通过、108 个补偿/基准配对一致；`python -B -m pytest -p no:cacheprovider -q --basetemp=results/_pytest_channel_phase_final` 为 **606 passed、25 deselected**。Main 的 `python -B -` 独立几何复算目标功率比最大差 `1.78e-15`、CW 抑制差 `4.97e-14 dB`；全部 129 个来源哈希匹配，三张保存图检查完成，`git diff --check` 通过。
- 结果与边界：无干扰 EGC/MVDR 收益约 3.00 dB；EGC 90° 未补目标功率保留 0.5、180° 理论抵消并失捕。准确补偿后 DFE/合并输出相对基准差≤`1.06e-16/8.70e-16`，单 CW 的 MVDR 抑制恢复 36.94～37.10 dB并恢复基准捕获，EGC 则即使准确补偿仍因残余干扰失捕。这里只证明 DFE 后静态相位线性闭环，不证明自动校准、RF/ADC 前误差、动态相位、时延对齐或真实手机性能。桌面弹窗按默认开启，本轮仅用 Agg 验证保存图；旧扫描与历史研究未重跑。
- 详细说明、阅读顺序及实际结果：[双天线整通道相位补偿闭环](docs/interference/README.md#2026-10-14-双天线整通道相位补偿闭环)。结果在 `results/interference/channel_phase_validation/`；沿用本地忽略规则，不复制 IQ 或归档源码。
- 上库复核（基准 `4a0f3d4`）：`python -m pytest -q tests/test_channel_phase_validation.py tests/test_rf_dfe.py tests/test_pipeline.py tests/test_antenna.py --basetemp=results/_pytest_upload_channel_phase` 为 **36 passed**，覆盖短记录闭环、单侧 C/N₀ 缺失拒绝、输出保存和配套接收模块。只读复查核对了整行旋转、分量评分、配对核验及既存 21 项核验结果，未发现阻塞问题。`git diff --cached --check` 通过；本次未另行重跑完整相位批次，只提交实验、测试和说明记录，结果及归档仍保留本地。

## 2026-10-14 无干扰 EGC/MVDR 小差异的独立控制核对

- 目的：核对用户提出的“无干扰时 EGC 也更好”，区分真正无干扰与含干扰权值的事后热噪声指标；未修改算法或持久配置。读取到保存的 `full_chain` 和 `direct_response_boundary` 清单仍有两个启用干扰，没有用户所指具体无干扰记录，未把独立控制说成该记录的复现。
- 实际条件与执行：`python_sim_simplified/` 内用 `python -u -B -c '<内存诊断代码>'` 执行一条完整接收记录；代码排版展开见下方详细说明。理想四角、真实/假设响应匹配、零时延、seed 42、100 ms，进程内 `cfg.interferers=[]` 且多径关闭；原目标功率/NF/门限不变，SINGLE/EGC/MVDR 共用同一 RF/DFE 数据。独立线性 RF/DFE 评分去掉两端各 10 ms，跟踪 C/N₀ 取后半段中位数。
- 实际结果：EGC/MVDR 真码热噪声相关 C/N₀ 为 50.326788/50.326124 dB-Hz，仅差 0.000665 dB；实际跟踪后半段估计为 51.332753/51.339575 dB-Hz，次序相反。两者权值范数平方 0.25/0.2500005234，相对权值差 0.1447%，理想白噪声损失约 `9.09e-6 dB`；导向匹配误差 0。
- 结论与限制：此控制下二者基本相同，有限样本求权和短窗统计可产生微小排名变化，不支持“EGC 明显更好”；不能据此判断用户尚未提供的运行条件。当前 EGC 在幅度不等时为 MRC 型权重，命名限制已说明。未新建结果文件/图片；未改源代码，因此未重跑 pytest、完整消融、相位扫描或标定。文档 `git diff --check` 核对无空白错误。
- 详细说明及实际诊断代码：[无干扰下 EGC/MVDR 的小差异核对](docs/interference/README.md#2026-10-14-无干扰下-egcmvdr-的小差异核对)。

## 2026-10-14 球形图内凹表达与阅读性修正

- 目的与依据：原线性半径低增益已经向内，但弱响应压缩、图标过大和背面投影遮挡使凹陷不清楚。仅修改球形图显示，不改变天线/接收算法或响应数据。
- 实际改动：显示半径改为 `clip(1-G_dB/G_floor,0,1)`，默认 0/−20/≤−40 dB 对应 1/0.5/0；增加 0 dB 参考轮廓/球心，方向图标移到外侧并引线连接精确响应半径，低冷高暖加光照网格，扩大图体并移除坐标框。首次新图 QA 后继续加入互补固定半球剖视，避免背面填住前方凹陷。
- 默认行为与限制：原球形图文件名、保存和弹窗规则保留；二维、普通三维图及接收计算不变。半径是显示刻度，不是线性电压或功率；≤−40 dB 共用半径 0，但图例保留精确 dB。固定半球剖视不随 GUI 旋转重新裁切，图标是方向辅助而非增益凸起。弱旁瓣即使相对峰值低，也可为局部凸瓣，未人为改变增益排序；未增加兜底分支。
- 条件与输出：四路参数化预设、seed 42、100 ms、目标 30°/60°、默认 CW/2 MHz 宽带、多径关闭。四张新版图及配置/权值/源码清单、接收摘要在 `results/interference/spherical_readability/`；未覆盖此前绘图或实验历史。实际渲染使用 Agg，只核验静态图。
- 失败与复验：新增显示期望在旧实现上 **3 failed**，第一版球形测试 **5 passed**；QA 发现背面仍遮蔽凹陷，再先复现背面测试 **1 failed**、修正后 `python -B -m pytest -p no:cacheprovider -q tests/test_plotting_runner.py -k spherical` 为 **6 passed、7 deselected**。默认完整回归 `python -B -m pytest -p no:cacheprovider -q --basetemp=results/_pytest_spherical_readability` 为 **601 passed、25 deselected**。
- 结论与检查：四图已逐张检查，全部权值与接收指标绘图前后完全一致；清单 128 个来源哈希匹配最终源码，MVDR 峰比仍为 12.863909，其他三分支仍失捕。只读审查无实质/阻塞问题，`git diff --check` 通过。未重跑历史消融、相位扫描或标定，不新增抗干扰性能结论。
- 上库复核（基准 `946d27d`）：`python -m pytest -q tests/test_plotting_runner.py tests/test_research_contract.py --basetemp=results/_pytest_upload_spherical_readability` 为 **16 passed**，包含 FULLCHAIN 直接运行和输出保存检查；检查了本轮测试生成的 MVDR 球形图，参考轮廓、深凹陷、方向引线与图例可读。`git diff --cached --check` 通过；只提交绘图、测试和说明记录，结果和归档仍保留本地。
- 详细说明：[球形图内凹表达与阅读性修正](docs/interference/README.md#2026-10-14-球形图内凹表达与阅读性修正)。

## 2026-10-13 源级幅相直接赋值与已知/未知相位配对扫描

- 目的与改动：新增独立入口 `experiments/interference/direct_response_boundary.py`，在理想四角几何基底上直接赋目标/CW/宽带每个天线的非几何幅相；真实目标与接收机假设分开，系数作用到实际信号、导向和分级诊断一次。保留原实际序列时延与 RF 相位处理。公共接收链增加可选 `scene` 注入，功率预算复用 `antenna_ablation.run_case`，不复制 RF/DFE/捕获跟踪。
- 默认行为与范围：只增加独立实验，`full_chain` 与原方向图模型不变；只维护精简版。字典完整列出四角，未新增配置缺失、分母、文件失败等兜底。估计时延仍仅用于窄带导向相位，不声称已实现 DFE 序列对齐。独立入口不绘制未定义的全角度方向图。
- 条件：四天线、TR 追加 0/20/40/60/90/120/180°目标相位；每点配对已知与未建模假设，seed 42、100 ms，共 14 条记录、56 分支。基础幅相与硬件时延均为零，多径关闭，目标 −128.5 dBm、CW/2 MHz 宽带各 −90 dBm，NF 和门限不改；功率去掉两端各 10 ms，接收仍用完整记录。
- 失败、修正与复验：初次测试因新模块未实现而无法导入；实现中修正时延手推值的浮点单位表达与 Path 清单序列化。初轮接口目标测试 **24 passed**。审查发现模块常量替换后，函数默认参数仍可能使用旧响应/扫描点；先以替换 3 dB 仍取旧 0 dB 的测试复现，再删除响应构造的字典默认绑定并显式传递扰动天线/绘图相位。`python -B -m pytest -p no:cacheprovider tests/test_direct_response_boundary.py -q` 最终 **9 passed**。首轮默认批次在五个 case 后中断，未记为完成；最终源码从头重跑 14 条全部完成，只读复审无未解决的实质性问题。
- 实际验证：`MPLBACKEND=Agg` 下 `python -u -B -m experiments.interference.direct_response_boundary` 完成生成、独立复算、统计及三图；`python -B -m pytest -p no:cacheprovider -q --basetemp=results/_pytest_direct_response` 为 **599 passed、25 deselected**。所有分支 ADC 独立重建误差 0，合并逐点误差最大 `8.88e-15`，功率与 `wᴴRw` 最大相对误差 `2.03e-14`；清单 128 个来源文件哈希逐项匹配，三图已检查。
- 结论与限制：七个相位点的已知/未建模 MVDR、Blind Null 都正确捕获，SINGLE/EGC 均未捕获；七点不是随机重复成功率。未建模 MVDR 在 120°、180°的峰比低于 Blind Null，180°目标线性输出损失 6.43 dB，但仍捕获成功，尚未找到失捕边界。只验证单个 seed 下单天线静态相位；未完成增益/时延全扫描、一般宽带频响或硬件验证，未重跑历史消融/噪声标定。
- 详细说明与数据：[直接响应相位扫描验证](docs/interference/README.md#2026-10-13-直接响应相位扫描验证)；输出在 `results/interference/direct_response_boundary/`，未覆盖旧实验目录。
- 上库复核（基准 `6966e99`）：`python -m pytest -q --basetemp=results/_pytest_upload_direct_response` 为 **599 passed、25 deselected**。只读审查未发现实质性或阻塞问题，核对了 14 条记录、56 分支的表格指标及全部 128 个来源哈希；本轮未另行重跑完整扫描。修改连同记录一起提交，归档和结果数据继续只保留本地。

## 2026-10-13 三维方向图由固定球面改为响应塑形

- 目的与改动：球面图不再使用固定半径 1，改为 `r=|wᴴa|/max|wᴴa|` 的归一化线性幅度半径；弱响应和零陷向原点收缩，颜色仍表示 dB。方向标记按精确方向响应定位在曲面半径外侧 3%，保留双视角、图标图例、方向角与数值。
- 默认行为与范围：仍输出 `detailed_array_response_spherical_{mode}.png`，保存/弹窗设置不变。其他图及天线、RF/DFE、合并、捕获跟踪计算不改；理想各向同性单天线仍为球形，不增加兜底分支。同步当前绘图说明与 README 记录入口名称。
- 条件与输出：四天线参数化预设、seed 42、100 ms、目标 30°/60°、默认 CW/宽带干扰、多径关闭；绘图验证输出在 `results/interference/gain_shaped_response/`，保留原 `full_chain` 和消融历史输出。
- 测试先行：`python -B -m pytest -p no:cacheprovider -q tests/test_plotting_runner.py -k 'surface_radius or direction_marker or isotropic_response'`。旧实现 2 failed、1 passed；修改后 **3 passed、8 deselected**。
- 完整回归：`python -B -m pytest -p no:cacheprovider -q --basetemp=results/_pytest_gain_shaped` 为 **590 passed、25 deselected**，历史证据测试按默认配置排除。四图检查发现反向视角标签靠近标题，完成坐标标签/标题间距微调后，`python -B -m pytest -p no:cacheprovider -q tests/test_plotting_runner.py --basetemp=results/_pytest_gain_shaped_final_plot` 为 **11 passed**，四张实际图已重新生成。
- 结论与限制：新图正确显示凸起、收缩和零陷，图例仍保留角度与 dB 值；绘图前后的全部指标和各分支合并权值逐项一致。`git diff --check` 无空白错误。未重跑 60 条消融，也未以新图追改历史抗干扰结论；半径是归一化电压响应，不是负 dB 值直接作为半径，也不是绝对功率增益。
- 详细说明：[凹凸三维方向图验证](docs/interference/README.md#2026-10-13-凹凸三维方向图验证)。

## 2026-10-13 将验证说明扩展为每次迭代必记

- 目的与改动：更新根目录 `AGENTS.md`，将“上库前记录”扩展为每次代码或配置修改、实验迭代或验证后追加记录；研究验证同步追加到对应方向的既有说明，保留实验条件、失败与复验、结论边界及结果位置。本文件改为持续的开发迭代与验证记录。
- 行为与范围：仅修改项目协作规则和记录说明，不修改仿真代码、实验配置或结果；后续未提交的迭代也须及时记录。此前天线/RF 上库条目保留原内容。
- 验证：`git diff --check` 通过，旧条目完整性核对通过（历史条目内容完全一致）；本轮未重跑 pytest 或仿真，因为没有代码或实验条件变化。
- 详细实验报告仍使用 [抗干扰与联合接收链说明](docs/interference/README.md)，本轮没有新增实验结论。

## 2026-10-13 天线模型与 RF 带宽噪声，配对消融实验

对照版本：`9d4165a`。

### 改动

- 参数化天线模型增加方位角增益变化、方向相关相位中心变化（PCV，以毫米路径长度建模）和遮挡相位；真实与假定状态改为独立深拷贝。实测方向图频率匹配使用明确的 1 Hz 绝对容差。
- RF 信号和热噪声均经过等效带宽校准的 FIR，补偿群延迟；热噪声按物理通道名称及 I/Q 分量使用固定随机流，使增减天线时保留通道的噪声一致。更新 GPS、SDCP 和联合链路调用，并记录滤波器、带宽与噪声功率谱信息。
- FULLCHAIN 顶部新增 `ANTENNA_MODEL`，默认从理想模型切换到 `parametric` 合成模型，并打印实际模型名称。
- 新增 `experiments/interference/antenna_ablation.py`：固定 3 个种子、2 个方向情境和 10 类天线条件，比较 4 个接收分支；在接收后按实际合并权值复算信号、干扰、热噪声和非线性残差功率，核对 ADC 重建与功率分解。
- 更新天线、RF 和实验回归测试及抗干扰说明；新增本记录文件、README 导航与每次上库前更新记录的协作规则。

### 默认行为与限制

- RF 滤波与噪声随机流发生变化，当前数值结果不能直接视为旧版结果的重复；多径研究总开关仍默认关闭。
- `synthetic_phone_corner_v2` 的增益、PCV、遮挡和耦合参数用于敏感性研究，不是 Mate70 实测数据或硬件性能证据。真值功率分解仅用于事后评估，不用于设计接收权值。
- 本次上库验证不等于重新完成全部历史 RF 研究；`archive/`、`results/` 和本地历史资料不上传。

### 验证

- 默认回归：`python -m pytest -q --basetemp=results/_pytest_upload_antenna_rf`，**587 passed, 25 deselected**；25 项依赖本地证据的测试按配置排除。
- 只读代码审查未发现实质性或阻塞问题；核对了模型单位、RF 带宽与噪声标度、调用迁移、输出目录和既存消融结果表。
- `git diff --cached --check` 通过。上库范围不含归档、结果数据或本地历史资料；本轮未另行重跑完整消融批次。
