# 修改记录

只记录实际改动、重要行为变化、最终验证和必要限制；实验细节见对应说明。以下路径除注明外均相对于 `python_sim_simplified/`，测试数字为各次修改当时的结果。

## 2026-10-15 用例集中构造

- `experiments/interference/validation_config.py` 统一完成选择、参考依赖及 seed/时长展开；运行器直接执行完整用例，保存入口直接记录实际配置，保留物理前端复用和评价口径。
- `ValidationSettings` 仅保留统计与输出设置；`cases.base_config` 改为逐 seed 的 `receiver_configs`，清单 `actual_receiver_configs` 路径保留。两种选择都为空时不再隐式运行全部系列。
- 完整回归 **167 passed、2 deselected**；168 行结果全部字段与完整核验报告和改前严格一致。条件与兼容性见[用例构造说明](docs/interference/README.md#2026-10-15-用例构造更新)。

## 2026-10-15 统一前端与按需分量评估

- 沿用 FULLCHAIN 的公共链路，旧相位入口也改用 `core/pipeline.py`；三份分量分析统一到 `core/evaluation.py`，实际接收与分析复用 `core/dfe.py` 的线性处理，实验保留各自配置和评分口径。
- 默认只运行实际链；精确评估通过 `retain_analysis=True` 保留本次 RF 热噪声，不再重抽噪声或重复设计 DFE 滤波器。保留原物理顺序、独立 ADC 重建、交叉项和联合非线性/量化残差，不将线性分量和冒充真实输出。
- 完整回归 **164 passed、2 deselected**；十组实际链指纹、138 行实验结果与改前完全一致，另以旧代码独立核对 16 组/48 个分量数组一致。条件、命令与本地证据见[重构验证](docs/interference/README.md#本次重构数值对照)；未重跑完整历史研究批次。

## 2026-10-15 通用验证入口中文注释完善

- 为 `experiments/interference/` 下的 `validation_config.py`、`validation.py`、`validation_support.py` 补充中文说明，解释条件选择、真实/名义参数、相位补偿、功率预算、配对核验和绘图统计。
- 三个文件内存编译通过，去除文档字符串后的 AST 与提交前一致；仅修改注释和文档字符串，未重跑测试或仿真。

## 2026-10-15 通用验证绘图可读性调整

- `experiments/interference/validation_support.py` 按指标和方法分别成图，横排比较场景；使用不同线型、空心标记及重合标注，不偏移数据。默认批次由 4 张改为 18 张，方向/增益批次由 8 张改为 36 张，文件名增加方法后缀。
- 热噪声收益、目标功率差异和 SINR 改用 dB，C/N₀ 保持 dB-Hz；捕获峰值比及门限保持原始比值。跨 seed 按所示单位求均值，不截断极低 dB 值；原始 summary 不变。
- 已有批次复用 summary 重绘，保留原仿真 manifest；条件和显示口径见[绘图说明](docs/interference/README.md#2026-10-15-绘图使用更新)。
- 上库专项验证：`python -B -m pytest -p no:cacheprovider -q tests/test_validation.py --basetemp=results/_pytest_upload_plot_readability`，**17 passed**，覆盖绘图数据、捕获门限、布局及重合标注；未重跑全套测试或完整研究批次。

## 2026-10-15 精简修改记录

- 合并重复验证，删除工具操作、Agent 分工和过期中间状态；保留历史改动、关键结论及维护/归档位置。
- 根目录 `AGENTS.md` 明确：除特别简单、含义直观的代码外，须添加必要中文注释，解释逻辑、公式及函数输入输出，并随代码维护。
- 明确各级 README 只保留必要信息，措辞简单易读，避免重复解释。
- 同步项目记录规则：按改动范围决定篇幅，不限制字数或要点数量，不遗漏必要信息。仅修改文档，核对历史条目、链接及 Git 差异，未重跑测试或仿真。

## 2026-10-14 测试文件改为仅本地保留

- `tests/` 下 30 个文件取消 Git 跟踪，本地保留；新增测试自动忽略，同步协作规则和 README。历史冻结测试、`pytest.ini` 和测试依赖保留。
- 已核对 30 个本地文件完整、忽略规则生效；运行代码未改，未重跑测试。
- 其他已有副本若需保留测试，应在首次拉取该提交前备份 `tests/`。

## 2026-10-14～15 通用验证框架

- 新增统一实验入口 `experiments/interference/validation.py`，条件集中在同目录 `validation_config.py`，共用处理在 `validation_support.py`；支持相位、方向、阵元增益扫描和真实/名义参数分离。`core/pipeline.py` 拆分前端与接收阶段，原入口行为不变。
- 默认双天线 TL/BR、3 个 seed、100 ms，比较无干扰/单 CW 下的基准、未补和准确补偿，运行 SINGLE/EGC/MVDR。配对核验拒绝单侧 C/N₀ 缺测，双方缺测标为未评价。
- 最终回归 **159 passed、2 deselected**；提取前后数值基准一致。默认批次 324 行/450 项检查、方向与增益示例 30 行/30 项检查全部通过；检查数不是独立实验次数。
- [条件、结果与用法](docs/interference/README.md#2026-10-14-通用验证框架)。输出在 `results/interference/validation/` 和 `validation_examples/`（后者同属 `results/interference/`）。补偿相位人为已知，未实现自动估相或宽带时延对齐；仅验证保存图，未验证 GUI 交互。

## 2026-10-14 备份并收口为抗干扰维护代码

- 当前维护代码：`F:/antenna_diversity/python_sim_simplified/`；公共源码在 `core/`，抗干扰实验在 `experiments/interference/`，后续开发继续修改这些位置。
- 清理前源码：`F:/antenna_diversity/python_sim_simplified/archive/pre_interference_only_2026-10-14_156b994/python_sim_simplified/`；上一级 `FREEZE_MANIFEST.json` 记录 271 个备份文件哈希，均已核验。归档仅本地保留。
- 删除抗欺骗、多径专用算法、实验、工具、测试及公共链路中的相关接口；保留通用天线、RF/DFE、GNSS L1/L5、捕获跟踪、定位和录波接收。旧配置中的已删除字段不再支持；恢复相关研究需使用冻结源码。
- 清理后回归 **141 passed、2 deselected**，GPS L1CA/L5Q 冻结对照 **2 passed**。理想/参数化天线各 20 ms、最多 10 跟踪历元的抗干扰输出与清理前一致（排除配置结构和已删除多径键）。GPS 对照使用本次清理前快照；更早冻结版存在模型差异，不能作为等价基准。
- [清理范围与计划](REFACTOR_PLAN.md#2026-10-14-interference-only-maintained-code)。历史报告与结果保留，未重跑历史研究。

## 2026-10-14 相位验证入口中文注释完善

- 补充 `experiments/interference/channel_phase_validation.py` 的单位、矩阵维度、相位补偿、功率复算和配对判据说明。
- 内存编译通过，去除文档字符串后的 AST 与修改前一致；未改执行逻辑，未重跑测试或仿真。

## 2026-10-14 双天线整通道相位补偿闭环

- 新增 `experiments/interference/channel_phase_validation.py`：在 DFE 后对目标、干扰、噪声整通道施加静态相位并准确逆补偿，独立评分和绘图；修正单侧 C/N₀ 缺测仍可能通过配对核验的问题。
- TL/BR、3 个 seed、100 ms、6 个相位点，无干扰/单 CW：324 行结果完成，108 个补偿/基准配对一致；完整回归 **606 passed、25 deselected**。
- 无干扰 EGC/MVDR 收益约 3 dB；单 CW 下准确补偿恢复 MVDR 基准捕获，EGC 仍因残余干扰失捕。仅证明 DFE 后已知静态相位的补偿闭环，不代表自动校准或 RF/ADC 前误差验证。
- [实验条件与结果](docs/interference/README.md#2026-10-14-双天线整通道相位补偿闭环)；输出 `results/interference/channel_phase_validation/`。保存图已检查，GUI 交互未验证。

## 2026-10-14 无干扰 EGC/MVDR 小差异的独立控制核对

- 理想四天线、真实/假设响应匹配、零时延、无干扰、seed 42、100 ms：EGC/MVDR 真码热噪声相关 C/N₀ 仅差 0.000665 dB，实际跟踪估计的排序相反，不支持“EGC 明显更好”。
- 本轮是独立控制，未复现用户未提供的具体运行；没有修改算法。条件、统计口径和诊断代码见[详细说明](docs/interference/README.md#2026-10-14-无干扰下-egcmvdr-的小差异核对)。

## 2026-10-14 球形图内凹表达与阅读性修正

- 球形图改用 dB 显示半径，增加参考轮廓、外置方向标记及固定半球剖视，解决弱响应压缩和背面遮挡。0/−20/≤−40 dB 对应半径 1/0.5/0；该半径不是线性功率或电压，剖视不随 GUI 旋转更新。
- 完整回归 **601 passed、25 deselected**；四张保存图已检查，绘图前后权值与接收指标一致，未验证 GUI 拖动。
- [验证条件与说明](docs/interference/README.md#2026-10-14-球形图内凹表达与阅读性修正)；输出 `results/interference/spherical_readability/`。

## 2026-10-13 源级幅相直接赋值与已知/未知相位配对扫描

- 新增 `experiments/interference/direct_response_boundary.py`，可分别设置目标/CW/宽带的逐天线幅相及真实/名义参数；公共接收链支持注入场景。修正配置常量修改后默认参数仍绑定旧值的问题。
- 四天线、TR 单天线目标相位扫描、seed 42、100 ms：14 条记录/56 分支完成，回归 **599 passed、25 deselected**。已知/未建模两组 MVDR、Blind Null 均正确捕获，SINGLE/EGC 均未捕获；尚未找到失捕边界，七个相位点不代表随机成功率。
- [完整条件与结果](docs/interference/README.md#2026-10-13-直接响应相位扫描验证)；输出 `results/interference/direct_response_boundary/`。未完成增益/时延全扫描，未实现 DFE 序列时延对齐。

## 2026-10-13 三维方向图由固定球面改为响应塑形

- 球形图半径由常数改为归一化线性幅度响应，弱响应和零陷向内收缩，颜色仍表示 dB；输出文件名与接收计算不变。
- 完整回归 **590 passed、25 deselected**，最终绘图测试 **11 passed**；四图已检查，绘图前后指标和权值一致。
- [验证条件与说明](docs/interference/README.md#2026-10-13-凹凸三维方向图验证)；输出 `results/interference/gain_shaped_response/`。

## 2026-10-13 将验证说明扩展为每次迭代必记

- 更新根目录 `AGENTS.md`：每次修改或验证后及时记录，研究细节放对应方向的既有说明。
- 仅修改文档，Git 差异检查通过；未重跑测试或仿真。

## 2026-10-13 天线模型与 RF 带宽噪声，配对消融实验

- 参数化天线增加方位增益、方向相关相位中心及遮挡相位；真实/假定状态独立。RF 信号和热噪声统一经过带宽校准 FIR，噪声随机流按物理通道固定。
- FULLCHAIN 默认天线由理想模型切换为 `parametric`；新增 `experiments/interference/antenna_ablation.py`，用 3 个 seed、2 个方向和 10 类条件比较四个接收分支，并独立复算功率。
- 对照版本 `9d4165a`，回归 **587 passed、25 deselected**。RF 滤波与噪声流已变化，结果不能直接视为旧版重复；合成天线用于敏感性研究，不是手机实测证据。
- [模型、实验条件与结果](docs/interference/README.md)；输出 `results/interference/antenna_ablation/`。
