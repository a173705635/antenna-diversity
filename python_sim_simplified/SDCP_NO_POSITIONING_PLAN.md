# 不接定位的 SDCP 验证计划

本轮用户要求：直接验证双通道 SDCP，提供方法、过程、结果和局限，不调用定位求解。不改现有接收核心，不同步历史代码，不新增保护或默认值回退。

- [x] 新建 `audit_sdcp_no_positioning.py`，复用 `generate_scenario`、`process_rf`、`process_dfe`、`acquire`、`track`。两通道各自 RF、DFE、捕获和跟踪；保存两个通道的原始 NCO 与质量时序。
- [x] 用错开历元的解析输入检验公共时间重建，验证三周半偏差不会被取余或去均值隐藏：`python -m pytest tests/test_sdcp_no_positioning.py -q`。
- [x] 固定 1.5 秒、RF 33 MHz / DFE 5 倍抽取 / 跟踪 6.6 MHz，运行噪声对 11600/11601、11602/11603、11604/11605。三组各运行零偏差、±10°固定偏差、20°/秒斜坡、10°幅度 2 Hz 正弦、30°阶跃、150 ms 单路中断、固定天线/RF幅相偏差共八场景。
- [x] 使用现有 Costas 跟踪，并给两路注入相同的确定性 50 bps 符号，验证不依赖无电文假设；这些符号不是有效 LNAV 帧。三组之间噪声独立，同组各场景复用噪声实现配对实验。零偏差 A 路在同组复用，数值上等价于相同数据再次运行。
- [x] 固定评估时间 10～1490 ms，每 1 ms 一个值。报告完整记录、200 ms 后、事件附近和 1100 ms 后的均值、总体标准差、RMS、最大绝对误差。质量指标单列，不用于筛选好看的结果。
- [x] 仅在评估阶段比较几何真值和注入真值；原始 SDCP 不取余、不减参考常数、不修整数分支。中断期没有有效信号相位，显示 NCO 输出但不声称该段误差有物理意义。
- [x] 保存配置、源码快照、哈希、原始跟踪数组、公共时间结果、图和 JSON 汇总；复核全部统计和图后编写报告。本轮不标定检测阈值、不统计欺骗或多径检测率，不新增实测硬件结论。

主要公式：在每通道当前历元内，`phase(t)=phase_start+2*pi*frequency_start*(t-start)`；`D=-wavelength*(phase_B-phase_A)/(2*pi)`。源端采用 `exp(-j*k*b.u)`，故本地源对应的几何距离项为 `+b.u`。常数和动态注入产生 `-wavelength*phase_injection/(2*pi)`。

运行：`python audit_sdcp_no_positioning.py --output validation/sdcp_no_positioning_2026-09-16`。当前已有其他研究未提交改动，保存本次使用的精确源码，不修改这些改动。
