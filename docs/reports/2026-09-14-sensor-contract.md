# W1 首版：四路 RF／IMU 数据与时间契约

日期：2026-09-14。范围：实现[首阶段计划](../superpowers/specs/2026-09-14-commercial-antispoofing-stage1-plan.md)中 W1 的最小契约与旧记录适配；不是 B 完整系统，也不是同步硬件或 IMU 导航验证。

## 已实现

[sensor_contract.py](../../python_sim_simplified/sensor_contract.py) 提供四个独立入口，均不修改输入。校验成功返回 `None`，错误抛出带 `reason` 和 `field` 的 `ContractError`。成功只说明数据结构与已声明标签自洽，绝不等于观测真实性或导航准入。

| 入口 | 约定及检查 | 不会替调用者完成的工作 |
|---|---|---|
| `validate_iq_batch(blocks, channel_ids)` | 2–4 路有序复电压向量；同运行、时钟域、计数、首样本历元、采样率及长度；整块可用时间不早于末样本 | 跨块连续性、物理同步测量、ADC 码值到电压的换算、饱和与标定判定 |
| `validate_imu(data, run_id=..., clock_id=...)` | 单调采样时间、逐样本可用时间、三轴比力 m/s²、角速度 rad/s、瞬时采样语义和显式坐标系 ID | 时钟变换、安装矩阵、重力补偿、积分区间观测转换及真实轴向认证 |
| `validate_observations(data)` | GPS L1 C/A 波长、唯一 PRN、历元×天线×PRN 数组、布尔质量掩码；有效值必须有限，失效值必须有原因；未知证据显式列出 | 卫星真实性、标定 ID 的实际有效性、相位弧认证或滤波准入 |
| `adapt_legacy_rf(...)` | 按调用者提供的通道映射读取 `run_gps_spp` 记录，复用原载波重建及质量掩码 | 发明硬件时钟、真实发布时刻、标定、IMU 或可信相位弧 |

## 时间与缺失信息处理

新采集契约使用 `monotonic_seconds` 和显式 `clock_id`。IQ 的 `sample_time_s` 是首样本时刻，`available_time_s` 是整块就绪时刻；同一批次要求严格匹配，异步设备应由另一个经过验证的时间转换层处理，不靠放大容差装作同步。IMU 可用时刻不得早于对应采样，但允许数据到达顺序与采样顺序不同；本模块不执行重排。

旧 RF 数据使用 `legacy_record_seconds`。原记录没有真实可用时间、硬件时钟域、标定和弧准入证据，因此这些字段保持 `None`，同时列入 `evidence_gaps`；IMU 与物理通道映射缺口也保留。`authenticity_verified` 固定为 `False`，校验器拒绝在本契约内声称完成真实性验证。

`valid` 仅表示原跟踪质量掩码，不等于产品输出的 `usable`。缺失或失效样本不插值、不置零，保留原数组与原因。这个区别防止“旧数据可以回放”被误解释为“旧数据符合商用准入”。

## 验证

完整回归实跑 **216 项通过，94.73 s**，保留一条既有 `tracking.py` 无噪声 C/N0 除零警告。三份已有冻结归档的 **605 项 SHA-256** 复核通过；语法编译与文档链接检查通过。测试结果为本轮软件验证，不代表传感器误差或导航性能达标。

[test_sensor_contract.py](../../python_sim_simplified/tests/test_sensor_contract.py) 的 19 项测试通过，覆盖：错误通道标签、时钟域冲突、计数/历元错位、采样率不一致、就绪时刻早于末样本、I/Q 类型或单位错误、缺时钟字段、IMU 重复/倒序/非有限时刻、比力语义错误、缺坐标系、逐星顺序及历元不一致、有效观测含 NaN、失效原因缺失和虚假发布时刻。

适配测试还验证：不修改原始记录；删除 PVT、位置误差等评价字段后重建结果不变；缺失硬件与标定证据没有被补成有效。

[回放审计](data/2026-09-14-sensor-contract/replay_audit.json) 实际读取原尺寸归档 48 条 RF 记录，比较 2/3/4 路共 36 个批次、34560 个载波观测槽位。新适配器与原重建的历元、载波和质量掩码逐项相同；读取前后 48 条输入文件 SHA-256 相同。无效观测槽位保留，没有从计数分母删除。这里使用 TL/TR、TL/TR/BL、四路三种布局，不包含全部可能的天线子集。

复现命令（输出路径须不存在）：

```powershell
python -m pytest python_sim_simplified/tests/test_sensor_contract.py -q
python python_sim_simplified/audit_sensor_contract.py --archive docs/reports/data/2026-09-13-gpspatron-spatial --output <新目录>/replay_audit.json
python -m pytest python_sim_simplified/tests -q
```

## 边界与后续

本轮是 W1 的首个实现，不宣称 W1 全部数据类型已闭环：码/Doppler/相关器完整契约、标定事件结构及消费者输出契约还需扩展；跨块采样连续性与消费者无效传播属于下一步。模块能拒绝错误的声明，但不能发现被错误贴上正确标签的实际电缆连接、轴向或时钟。

原检测、方向候选及默认导航入口没有改动，新契约目前独立调用。尚未接入 IMU 导航、真实四路硬件、误差保护界或 C 抑制；10 s 续航、99% 可用率与商用性能没有新增证据。下一步是连续数据流状态与故障事件回放，再连接标定/弧生命周期，不能把目前的静态批次检查当作在线保护闭环。
