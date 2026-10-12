# 双路多径实验代码讲解

依据：2026-09-17 当前维护代码。主文件 [sdcp_multipath.py](../../experiments/multipath/sdcp_multipath.py)。本文讲程序如何工作；数值结果见 [验证报告](SDCP_MULTIPATH_REPORT.md)。

## 1. 先看入口：运行一次做什么

在 python_sim_simplified 目录执行：

```powershell
python -m experiments.multipath.sdcp_multipath
```

入口 main() 实际做两件事：

```python
output = run_simulation()
verify_and_plot(output)
```

第一步生成信号，让两个接收链分别跟踪，计算 SDCP、门限和报警，保存结果。第二步读取保存的原始跟踪数据，重新计算并核对，输出图表及核验记录。两步已在同一脚本中，无需再运行 verify。

重跑会删除并重建 results/multipath/sdcp_multipath；不是只补画图片。本文说明不启动新的 RF 实验。

## 2. 信号和数据在程序中如何流动

```text
                     generate_scenario：直达波
                         scene[0] / scene[1]
                                  │
完整源波形 waveform ── _multipath_scene：延迟、衰减、反射相位、方向响应
                                  │
                         echo[0] / echo[1]
                                  │
                ┌─────────────────┴──────────────────┐
           scene[0]+echo[0]                     scene[1]+echo[1]
                │                                    │
           RF A → DFE A                         RF B → DFE B
                │                                    │
           捕获与跟踪 A                         捕获与跟踪 B
                └─────────────────┬──────────────────┘
                         共同时间轴的 NCO 相位
                                  │
                    载波距离单差 D − 几何单差 G
                                  │
                          原始残差 residual
                                  │
                       200 点单窗均值 score
                                  │
                     连续 3 次越限 → alarm
                                  │
                        独立重算、统计、绘图
```

A、B 分别代表 TL、TR 天线通道。两路覆盖相同的接收时间，程序按顺序计算只是执行安排，各有独立噪声、捕获结果和跟踪环路状态。

## 3. 顶部常量分别控制什么

| 常量 | 当前值 | 含义 |
|---|---|---|
| CALIBRATION_SEEDS | 11900、11902、11904 | 三组正常标定的 A 路噪声种子，B 用 seed+1 |
| TEST_SEEDS | 12000、12002、12004 | 三组测试的 A 路种子，B 用 seed+1 |
| DURATION_MS | 1500 | 唯一仿真时长设置；700 ms 时改为 700 |
| times | 从 200 ms 到结束前 10 ms，间隔 1 ms | analysis_timing() 按实际时长生成，1500 ms 时共 1291 点 |
| TAIL_DURATION_MS | 400 | 尾段定义为记录最后 400 ms，与实际可用观测取交集 |
| WINDOW | 200 | 一个平均窗口包含的观测点数；当前等于 200 ms |
| STRIDE | 10 | 每隔 10 个观测点进行一次判决；当前等于 10 ms |
| CONFIRM | 3 | 连续三次判决越限才确认报警 |
| MARGIN | 1.25 | 独立标定最大绝对均值的放大系数 |
| FREEZE_SOURCE | False | 记录源码哈希，默认不复制源码快照 |

times 是跟踪后的观测时间轴，不是 33 MHz 的 RF 采样轴。RF 一秒有 3300 万点，这里的 SDCP 一秒只有 1000 个重建点。

## 4. cases()：如何定义一个双路多径场景

例如：

```python
('mp_t600ms_p-3dB_d0p3chip_phi90_az110_el15', -3, .3, 90, .6, 110, 15)
```

从左到右是场景名、回波相对功率 dB、额外时延 chip、反射相位度、出现时刻秒、回波方位角度、仰角度。

它表示：0.6 s 开始加入一个 −3 dB、0.3 chip 延迟、反射相位 90°、方向 110°/15° 的回波。该回波同时作用于两个天线。channels='both' 是固定记录字段，不是可选的单路/双路开关，当前多径分支没有单路回波选项。

10 类多径均采用完整参数名，例如 `mp_t600ms_p-6dB_d0p3chip_phi90_az110_el15`：600 ms 进入、−6 dB、0.3 chip、反射系数相位 90°、回波方向 110°/15°。所有多径都影响两路，名称不重复写 dual/both，也不使用“强、弱、参考”代替参数。`0p05` 表示 0.05。

**代码范围与报告范围有一处需要明确：** 当前 cases() 仍包含 los、hardware、drift 三种无回波控制，完整运行共 39 条测试。按用户要求，现报告不展示单路硬件/漂移控制结果，只列 30 条双路多径和 3 条 LOS。本文不展开这些控制结果；这不是单路回波仍存在。

## 5. run_simulation() 第一阶段：准备直达波

配置采用默认接收链，限定 TL/TR、DURATION_MS 指定的时长（默认 1.5 s）、Costas 跟踪、无干扰，并清零额外通道时延。cfg.signal.multipath_paths=[] 表示此时先生成纯直达波；后面逐场景手动叠加回波，因此不等于整个实验没有多径。

生成后：

| 变量 | 形状/内容 | 用途 |
|---|---|---|
| scene | 两路 × RF 样本数 | 已带各天线直达波空间响应的信号 |
| waveform | 一维源波形 | 生成同源回波的基础 |
| bits | 8 个 ±1 重复符号 | 每 20 ms 切换一次，模拟 50 bps 数据调制 |

程序同时给 waveform 和 scene 乘数据位，随后延迟 waveform 生成回波，因此回波的数据跳变也随传播时延移动。该重复序列不是有效导航电文。

每次处理 100 万个 RF 样本只是控制临时数组内存，不是每 100 万点重启一次跟踪。33 MHz 下每块约 30.3 ms。

几何单差 geometry 是天线基线向 LOS 方向的投影，本配置约 32.866 mm。它是应有的空间相位对应距离，不是多径误差。

## 6. 先标定再测试：门限从哪里来

三组独立正常标定各运行 A、B 两条接收链。对每组依次计算：

```text
正常双路跟踪 → 原始 SDCP → 减 geometry → 200 点均值
                                         ↓
                                  最大绝对均值
```

再取三组最大值，乘 MARGIN。程序保存 threshold.json 并记录文件哈希，然后才进入多径测试；结束时确认文件没有变化。

这一步没有减去正常标定的平均相位，只是使用它确定一个报警门限。它也没有解决整周或半周偏差。

既有运行中三组最大绝对均值约 0.608、95.868、95.458 mm，因此门限约 119.835 mm。1.25 是工程裕量，不能解释成已满足某个虚警概率。

## 7. 回波究竟怎么生成、怎么加入两路

在多径分支中构建 MultipathPath，然后调用 [scene.py 的 _multipath_scene()](../../core/multipath/scene.py)。内部依次完成：

1. 时延换算：excess_delay_s = delay_chips / code_rate_hz。
2. 功率转幅度：relative_voltage = 10 ** (power_db / 20)。对复振幅使用 /20，不是 /10。
3. 乘复反射系数 exp(j·reflection_phase_rad)，以及配置的额外多普勒项。
4. 按回波方向计算两天线空间响应，得到两路波形。
5. 使用 [apply_channel_delays()](../../core/channel_delay.py) 做分数采样延迟，并补回中频表示中缺少的射频传播相位。

0.3 chip 在 1.023 MHz 码率下约为 0.293 μs，即 33 MHz 下约 9.677 个样本，所以不能只移动整数样本。序列延迟会产生中频相位变化，代码再补 carrier_hz−if_hz 对应的相位，使模型包含完整传播时延效应。

输出 echo 的第一维是天线通道。加入回波的关键代码是：

```python
signal_b[start:stop] += echo[1, start:stop] * envelope
signal_a[start:stop] += echo[0, start:stop] * envelope
```

两句始终一起执行。不能把 echo[1] 理解为“实验只有 B 加回波”；紧接着的 echo[0] 就是 A 路回波。

启动场景 envelope 恒为 1。中途场景在 0.6～0.61 s 通过升余弦从 0 变成 1，然后保持。两路 envelope 相同，但 echo[0] 与 echo[1] 的空间相位可以不同，所以造成的载波偏差不必相同。

反射相位 90°只是反射系数参数。接收点的相对相位还包含传播和几何，不能仅凭这个参数判断两路输出相位。

## 8. receive()：如何进入 RF、DFE 与跟踪

实验文件中的 receive() 主要负责调用链路和保存文件。实际处理在 [core/multipath/sdcp_observation.py](../../core/multipath/sdcp_observation.py)：

```text
process_rf → process_dfe → acquire → track
```

RF 输入是单路信号，内部增加通道维度后调用 RF 模型。DFE 将 33 MHz 转为 6.6 MHz 基带；捕获给出初始估计，track 运行该通道自己的环路。

返回 h 是跟踪历史，关键字段为：

| 字段 | 含义 |
|---|---|
| epoch_start_samples | 每个跟踪历元的起始样本位置，使用 DFE 输出采样率 |
| carrier_nco_phase_rad | 历元起点的累积 NCO 相位，单位 rad |
| carrier_nco_frequency_hz | 该历元用于相位外推的 NCO 频率 |
| prompt | Prompt 复相关输出，用于诊断 |
| phase_lock_metric / pll_error_rad | 锁定指标与鉴相误差，记录质量状态 |

这里计算 SDCP 使用的是累积 NCO 相位，不是直接对 Prompt 取 angle 作为完整载波相位。

## 9. residual()：不同历元起点怎么作差

两路捕获结果可能导致历元起点相差几个样本，因此不能直接拿两个数组相同下标的相位相减。raw_sdcp() 对每个目标时间 t 分别找覆盖它的历元，再外推相位：

```text
phase_A(t) = phase_A(epoch_start) + 2π × frequency_A × 时间差
phase_B(t) = phase_B(epoch_start) + 2π × frequency_B × 时间差
D(t) = −波长 × [phase_B(t) − phase_A(t)] / (2π)
r(t) = D(t) − geometry
```

np.searchsorted(..., side='right')−1 找到不晚于 t 的最近起点。外推使用该历元的频率，不是把两路按数组索引强行对齐。

r 的单位是米，画图和统计乘 1000 转毫米。相位符号沿用当前 NCO 定义；正负不能脱离这一定义解释。

整个过程不取波长余数、不减每条记录的均值、不使用真值补半周。geometry 虽是已知先验，但它不能自动消除独立 Costas 的半周分支。

## 10. scores() 和 confirmed()：红点如何产生

scores() 先计算累积和，再用两个累积和端点相减得到单个窗口内的总和，除以 WINDOW 得到算术平均，[::STRIDE] 每隔若干点取一次。端点相减是求一个窗口的和，不是前后两个窗口均值做差。观测不足一个完整窗口时，均值和时间索引同时为空。

第一窗用 200～399 ms 的 200 点，第二窗用 210～409 ms，第三窗用 220～419 ms。所以第一个均值在 399 ms，最早连续三次确认在 419 ms。

confirmed() 先计算 abs(score)>threshold，再通过累计计数计算连续三个判决中越限的次数，只有等于 3 才确认。例如：

```text
越限：False True True True True False
报警：False False False True True False
```

单次超限不画红点；正负方向均可超限。当前实现只要求每次绝对值超限，不额外要求三次符号相同。报警不锁存。

这里没有前后窗口做差。持续 10 mm 的偏差在单窗均值稳定后仍是 10 mm，不会自动归零；但若门限是 119.835 mm，它仍不会报警。

## 11. 结果字段怎么看

| 字段 | 含义 |
|---|---|
| detected | 事件后是否至少有一次确认报警；无事件后观测时为 null，不是检测概率 |
| assessment_status | evaluated 或 not_evaluated_event_outside_record |
| full_post_event_window | 是否存在完整的事件后平均窗口；短记录即使有判决也可能为 false |
| first_alarm_s | 首次确认报警的绝对时刻；无报警为 null |
| delay_s | 首次报警时刻减场景出现时刻；无报警不记作零延迟 |
| event_alarm_count / event_decisions | 事件后报警判决数量 / 判决总数 |
| pre_event_alarm_count | 事件前确认报警数量 |
| tail_score_mean_mm | 窗口终点位于记录最后 400 ms 的 score 平均值，1.5 s 记录对应 ≥1.1 s |
| tail_residual / all_residual | 原始 r 在尾段/完整统计区间的均值、标准差、RMS、最大绝对值 |
| quality_fraction | 两路质量同时合格的比例，不用于过滤数据 |

尾段 score 的均值与尾段原始 residual 的均值不是同一个指标：尾段窗口可能包含尾段起点之前的样本。

## 12. verify_and_plot()：独立核验与配对图

核验阶段先核对记录数量、门限哈希和源码来源。再从落盘 NCO 重建相位，逐窗 np.mean 重算均值，用显式连续条件重算报警，检查原始数组、统计和质量是否相符。它没有复用 scores()/confirmed() 的结果来证明自己正确。

需要注意，当前多径复算读取 threshold.json 中的 geometry，未独立从坐标再次验证这个投影；两阶段一致也不能证明物理模型完全正确。核验通过表示这些计算与保存数据一致，不代表多径检测性能通过。

phase_distance() 返回单路的 −λφ/(2π)，随后 b−a−geometry 得到 r。为解释物理响应，还计算：

```text
A change = 多径 A 的载波距离 − 同种子纯 LOS A 的载波距离
B change = 多径 B 的载波距离 − 同种子纯 LOS B 的载波距离
SDCP change = B change − A change
```

这属于两次配对仿真的离线比较，不是单次接收机直接能得到的多径真值，也不是已删除的前后 200 ms 窗口做差。它不进入报警。

paired_carrier_changes.png 每个场景分三层显示 A、B、B−A。两条单路曲线是在同一双路多径场景中拆开画，不代表分别只给一个通道加回波。

## 13. 文件输出和实际修改入口

| 文件/位置 | 看什么 |
|---|---|
| plan.json / config.json | 实际场景、种子、参数和来源哈希 |
| cal_*_A/B.npz | 标定原始跟踪历史 |
| seed_case_A/B.npz | 多径场景的双路跟踪历史 |
| *_decision.npz | time_s、residual_m、score_m、alarm、quality |
| threshold.json | 门限、标定最大值和几何单差 |
| results.json | 首次统计 |
| verified_results.json / result_table.md | 独立复算后的统计及配对变化 |
| calibration_diagnostics.json | 标定分支与 Prompt 诊断 |
| completion.json | 仿真阶段完成 |
| verification.json | 独立复算完成 |
| all_cases.png / representative_traces.png | 脚本完整场景的判据曲线 |
| paired_carrier_changes.png | 双路多径的 A/B 分轴离线对照 |

报告专用 named_view/report_cases.png、named_view/report_results.json 是按完整参数名和报告范围另行生成的展示文件，当前主脚本不会自动生成；完整脚本输出仍包含无回波控制。重跑后需重新生成报告专用附件，不能假定旧附件会保留。

改回波强度、时延、方向、出现时刻，看 cases()；改 RF/DFE 等看 default_config() 及本脚本覆盖设置；改观测看 raw_sdcp()；改判据看 scores()/confirmed()。

现在修改时长只需改 DURATION_MS，观测时间轴、计划中的 duration_s 和尾段起点同步生成；复算从 plan.json 读取窗口长度、步长、连续次数和尾段起点。700 ms 时观测范围为 200～690 ms，尾段起点为 300 ms。回波仍于 600 ms 进入，因此尾段包含进入前的数据，且只有约 90 ms 的进入后观测，不能解释为多径稳定后的性能。很短的记录还必须满足既定观测起点与完整窗口所需的数据长度，本次修复不保证任意时长都能构成有效研究。

## 14. 阅读时最容易混淆的四件事

1. 两路都有回波，不等于两路误差相同；方向响应决定各路复信号叠加。
2. 没有 alarm，不等于没有载波变化；变化可能不足门限或未连续越限。
3. 分轴绘制 A、B，不等于单路加回波实验；图画的是同一场景的两路输出。
4. 核验通过，不等于检测可靠；现存半周分支和高门限仍是真实限制。

当前多径参数至少需要 429 ms，才能在 200 ms 观测起点后组成一个 200 点窗口及三次间隔 10 ms 的判决，并留出 10 ms 结束裕量。低于该时长会在清理输出和 RF 仿真前明确报错。
