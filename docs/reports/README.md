# 实验报告与结果

本目录保存实际验证记录、结论和固定数据快照。方案的存在不代表验证已完成；当前实现状态以 [PROJECT_STATUS.md](../../PROJECT_STATUS.md) 为准。

## 综合阅读

- [GNSS 天线分集全量整合报告](GNSS天线分集全量整合报告.docx)：综合结论及完整来源附录；来源目录页码保留原文口径。
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
