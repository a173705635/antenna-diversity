# 验证方案与结果导航

本目录回答“准备验证什么、如何验证、按什么指标判定”。结果与固定数据留在 reports，不因存在方案就视为已经完成验证。多径检测与外部干扰分开评估。

| 方案或主题 | 状态与结果入口 | 证据入口 |
|---|---|---|
| [SDCP 可信度自检](2026-09-11-sdcp-self-check-feasibility.md) | 待实现在线自检；[长窗口结果](../reports/2026-09-11-sdcp-long.md)仅为已有依据 | [长记录数据](../reports/data/2026-09-11-sdcp-long/) |
| [单反射径检测](2026-09-08-multipath-detection-plan.md) | 历史检测基线；[阶段结论与边界](../reports/2026-09-10-multipath-stage-summary.md) | [数据及核验说明](../reports/data/2026-09-10-multipath/README.md) |
| GPS 定位框架 | [历史设计草案](../archive/development/specs/2026-09-11-gps-spp-design.md)范围已收窄；[实际验证](../reports/2026-09-11-gps-spp-validation.md) | [GPS SPP 数据](../reports/data/2026-09-11-gps-spp/) |
| 最终 FCW 相位重建 | 见下方专项 Word；[相位基准审计](../reports/2026-09-11-phase-origin.md)是相关仿真证据，不等同于专项验收 | [相位基准数据](../reports/data/2026-09-11-phase-origin/) |
| 跨频相位稳定性、角度差分 | 见下方专项 Word；本索引未确认对应实机验收结果 | 不用双频定位仿真代替硬件验证 |

## 专项原文

- [双天线跨频GNSS物理路径差角度差分验证方案](双天线跨频GNSS物理路径差角度差分验证方案.docx)
- [基于最终FCW的载波NCO累加相位重建方案](基于最终FCW的载波NCO累加相位重建方案.docx)
- [手机双天线跨频GNSS相位稳定性验证方案](手机双天线跨频GNSS相位稳定性验证方案.docx)
