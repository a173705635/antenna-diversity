# 开发迭代与验证记录

记录每次修改、实验迭代和验证的目的、实际改动、默认行为、实验条件、验证结果、结论与限制，不等到上库时才补。新记录放在最前面，旧记录及其条件保留；同一迭代的失败、修正和复验归入同一条记录。详细研究说明复用各方向既有文档，并在记录中链接，不另建重复报告。每条记录与对应修改一起提交；提交编号可通过本文件的 Git 历史查询。此前提交保留在原有 Git 历史中。

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
