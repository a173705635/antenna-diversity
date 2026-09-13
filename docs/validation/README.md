# 验证方案与结果导航

已完成：[真实接收模块与定位集成计划](2026-09-13-recorded-receiver-plan.md)，见[40 秒真实 IQ 定位结果](../reports/2026-09-13-recorded-receiver.md)。通用处理已接通，标准大气修正和垂向偏移归因仍待推进。

已完成：[单天线真实录波验证方案](2026-09-13-real-single-antenna-plan.md)，结果见[报告](../reports/2026-09-13-real-single-antenna.md)。范围为真实采样接入、DFE、捕获、跟踪和导航比特结构；不含实测定位或双天线 SDCP。

[外部双天线信号源验证](2026-09-13-external-gnss-source-plan.md)已完成源端六项检查及六场景×两噪声对接收实验；见[结果及复现步骤](../reports/2026-09-13-external-gnss-source.md)。

[SDCP天线与RF幅相偏差方案](2026-09-13-sdcp-hardware-bias-plan.md)已完成固定参数六场景×两噪声对；见[未取余完整链路结果](../reports/2026-09-13-sdcp-hardware-bias.md)。方向变化、温漂与分级校准尚未验证。

SDCP新增[相位弧与参考生命周期原型](../reports/2026-09-13-sdcp-arc-lifecycle.md)：质量恢复不自动延续旧参考；已有RF回放验证，默认流程未接入，外部重新准入证据尚需研究。

当前优先工作：先审清SDCP观测基础，再推进识别。见[定义、误差、状态与使用条件](../reports/2026-09-13-sdcp-observation-contract.md)；本轮为源码与确定性对照审计，实际失锁重捕状态闭环仍待验证。

抗欺骗应用范围先看[完整场景范围与应用边界](2026-09-13-anti-spoofing-scenario-scope.md)：覆盖冷启动、先验可信度、失定位、失锁重捕及恢复；当前先讨论范围，尚未冻结下一阶段实验约束。

**当前新阶段：天线分集抗欺骗。** 已完成[双天线空间一致性首轮实现与验证](../reports/2026-09-12-antenna-spoofing-validation.md)：同源全星替代可见，部分欺骗漏检、真假混合有未知区间；不等于认证或导航恢复。多径阶段仍收口，极化分集暂停。

此前多径阶段已收口，后续不推进极化分集。完整范围、结论及未完成项见[阶段总结与交接](../reports/2026-09-12-multipath-stage-closeout.md)。

本目录回答“准备验证什么、如何验证、按什么指标判定”。结果与固定数据留在 reports，不因存在方案就视为已经完成验证。多径检测与外部干扰分开评估。

| 方案或主题 | 状态与结果入口 | 证据入口 |
|---|---|---|
| [SDCP 可信度自检](2026-09-11-sdcp-self-check-feasibility.md) | 已形成三态原型；[自检、压力测试与定位对照](../reports/2026-09-12-sdcp-self-check.md)；尚未完成实机在线接入 | [长记录数据](../reports/data/2026-09-11-sdcp-long/) |
| [单反射径检测](2026-09-08-multipath-detection-plan.md) | 历史检测基线；[阶段结论与边界](../reports/2026-09-10-multipath-stage-summary.md) | [数据及核验说明](../reports/data/2026-09-10-multipath/README.md) |
| GPS 定位框架 | [历史设计草案](../archive/development/specs/2026-09-11-gps-spp-design.md)范围已收窄；[实际验证](../reports/2026-09-11-gps-spp-validation.md) | [GPS SPP 数据](../reports/data/2026-09-11-gps-spp/) |
| 最终 FCW 相位重建 | 见下方专项 Word；[相位基准审计](../reports/2026-09-11-phase-origin.md)是相关仿真证据，不等同于专项验收 | [相位基准数据](../reports/data/2026-09-11-phase-origin/) |
| 跨频相位稳定性、角度差分 | 见下方专项 Word；本索引未确认对应实机验收结果 | 不用双频定位仿真代替硬件验证 |

## 专项原文

- [双天线跨频GNSS物理路径差角度差分验证方案](双天线跨频GNSS物理路径差角度差分验证方案.docx)
- [基于最终FCW的载波NCO累加相位重建方案](基于最终FCW的载波NCO累加相位重建方案.docx)
- [手机双天线跨频GNSS相位稳定性验证方案](手机双天线跨频GNSS相位稳定性验证方案.docx)

[首轮自检设计与冻结规则](2026-09-12-sdcp-self-check-design.md)，开发数据与最终RF留出分开记录。
