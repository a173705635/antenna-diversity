# 实验报告与结果

当前阶段已收口，后续不推进极化分集。完整范围、结论及未完成项见[阶段总结与交接](2026-09-12-multipath-stage-closeout.md)。

本目录保存实际验证记录、结论和固定数据快照。方案的存在不代表验证已完成；当前实现状态以 [PROJECT_STATUS.md](../../PROJECT_STATUS.md) 为准。

## 综合阅读

- [双极化原理、独立模型验证与商用边界](2026-09-12-dual-polarization.md)：共享参考结构、七类场景及反例；尚非 RF/跟踪闭环结果。
- [GF 权重映射消融诊断](2026-09-12-gf-weight-ablation.md)：同触发时刻区分误伤干净天线、码误差持续污染与条件数影响。
- [GF 接入码定位与降级回放](2026-09-12-gf-pvt-lifecycle.md)：降级符合预期，但当前条件降权使启动多径定位变差，保留负面结论。
- [GF 商用运行状态管理原型](2026-09-12-gf-runtime.md)：启动、失锁、标定失效与恢复的接口及事件验证，后续码 PVT 回放见定位启停报告。
- [商用可行性首轮评估](2026-09-12-commercial-feasibility.md)：标定与接口成本、降级行为、专利技术对应及新增 C/N0 多径选路对照。
- [双频 GF 条件三态判决](2026-09-12-gf-quality-gate.md)：区间判决代码与条件回放；参数未标定，不作检测率声明。
- [启动静态 RF 多径与双频偏置压力](2026-09-12-gf-startup-rf.md)：真实启动反射保留情况、观测域偏置抵消反例及判决准入条件。
- [双频整数分支与未知偏置审查](2026-09-12-gf-branch-audit.md)：解释 LOS 大常数、几何不敏感区与分支折返，区分 RF 回放和观测域注入。
- [分集路线首轮审查与验证](2026-09-12-diversity-route-first-pass.md)：双频四路 RF、简单选路与极化/空间模型反例，明确已完成与未完成范围。
- [天线分集多径检测与抑制：公司专利及高校研究调研](2026-09-12-antenna-diversity-multipath-survey.md)：外部证据、限制和路线建议，非本项目新增实验结果。
- [GNSS 天线分集全量整合报告](GNSS天线分集全量整合报告.docx)：历史综合结论及来源附录；不包含后续新增的全部 GF/双极化研究，最新内容见阶段总结。
- [验证方案、结果与数据对应](../validation/README.md)：区分待实施方案与已完成实验。
- [原始数据快照](data/)；[早期多径数据核验说明](data/2026-09-10-multipath/README.md)。各专题的数据位置、条件和限制以报告为准。

## 接收链路与观测可信度

- [连续信号源与单反射径理想参考](2026-09-08-continuous-source-oracle.md)
- [捕获到跟踪：数据驱动小数相位初始化](2026-09-08-tracking-initialization.md)
- [双 RF / 双 DFE / 双跟踪环路相位差评估](2026-09-10-dual-tracking-phase.md)
- [双独立跟踪链路：1 秒 × 10 组无多径基线](2026-09-11-dual-tracking-stability.md)
- [跟踪载波相位基准审计](2026-09-11-phase-origin.md)
- [单天线逐星伪距偏差诊断](2026-09-11-pseudorange-bias-diagnosis.md)
- [33 MHz RF / 5倍抽取 / 6.6 MHz跟踪](2026-09-11-sampling-33mhz.md)

## GPS 定位与双频观测

- [单天线 GPS L1 仿真定位框架验证](2026-09-11-gps-spp-validation.md)
- [单天线 GPS L1 码与载波联合浮点解验证](2026-09-11-code-carrier-float.md)
- [单天线 GPS L1 / L5 Q5 接入与定位验证](2026-09-11-gps-l5.md)
- [L1/L5 融合权重与位置基准优化](2026-09-11-dual-frequency-optimization.md)

## 多径检测与 SDCP

- [SDCP 自检、压力测试与独立RF定位对照（2026-09-12）](2026-09-12-sdcp-self-check.md)

- [纯多径研究：直达径基线自查](2026-09-08-multipath-baseline-audit.md)
- [纯多径：数据驱动检测与理想 LCMV 的差距](2026-09-08-detector-versus-oracle.md)
- [多径存在性：记录级门限校准](2026-09-08-detection-threshold-calibration.md)
- [冻结门限下的多径存在性边界](2026-09-10-detection-boundaries.md)
- [多径检测阶段归档（2026-09-10）](2026-09-10-multipath-stage-summary.md)
- [Trimble 单频双天线 SDCP：实现与验证](2026-09-11-trimble-single-frequency-validation.md)
- [不使用启动参考段的 SDCP 验证](2026-09-11-calibrated-sdcp.md)
- [SDCP 因果窗口平均对比](2026-09-11-sdcp-windows.md)
- [3秒记录的500 ms / 1 s SDCP窗口验证](2026-09-11-sdcp-long.md)
- [单频双天线 SDCP 的定位有效性评估](2026-09-11-sdcp-positioning.md)

## 外部干扰与空域合并

外部干扰研究与多径检测分别说明。[算法原理资料](../theory/README.md)、[当前仿真入口](../../python_sim_simplified/README.md)及[旧版批量研究归档](../archive/python_sim_2026-09-12/README.md)可用于追溯；不能把多径实验当作外部干扰验证。

## 历史来源与维护

[历史归档总索引](../archive/README.md)统一提供原文件备份、已整合来源与旧开发记录。后续结果报告加入对应主题，原始证据继续保存在 data；新方案放入 validation，原理说明放入 theory。
