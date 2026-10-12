# 文档导航

当前维护入口：[实验与共享代码目录](../python_sim_simplified/README.md)、[抗干扰联合链路](../python_sim_simplified/docs/interference/README.md)、[单源抗欺骗研究](../python_sim_simplified/docs/spoofing/SINGLE_SOURCE_SUPPRESSION_REPORT.md)、[双路 SDCP 多径研究](../python_sim_simplified/docs/multipath/SDCP_MULTIPATH_REPORT.md)。

历史抗欺骗：[GPSPATRON公开原理启发的2–4天线本地实现](reports/2026-09-14-gpspatron-spatial-diversity.md)，包含公式、独立标定、配对统计、RF链路、输出保护和负面边界；区分本地模型与商业产品能力。

SDCP 全链路误差入口：[天线、RF、DFE、跟踪与后级偏差清单](reports/2026-09-13-sdcp-error-inventory.md)，包含影响因素、处理办法和参考数值；定位优化暂不推进。

真实接收与定位最新结果：[通用接收模块及 40 秒 HackRF 码定位](reports/2026-09-13-recorded-receiver.md)。已从录波解星历并定位，保留垂向偏移和当前适用限制。

抗欺骗应用讨论入口：[完整场景范围与应用边界](validation/2026-09-13-anti-spoofing-scenario-scope.md)。场景定义与已完成实验分别记录，不能将纳入范围视为已验证。

先看 [项目状态](../PROJECT_STATUS.md) 了解当前结论，或阅读 [全量整合报告](reports/GNSS天线分集全量整合报告.docx)。运行代码见 [项目入口](../README.md)。

| 想解决的问题 | 入口 | 收录内容 |
|---|---|---|
| 基础概念和公式是什么？ | [knowledge](knowledge/README.md) | 通用知识、教程与数学推导 |
| 本项目算法如何工作？ | [theory](theory/README.md) | 算法原理、模型假设与实现说明 |
| 准备怎样验证？ | [validation](validation/README.md) | 实验设计、指标、待实施方案及结果对应 |
| 做过哪些实验，结论是什么？ | [reports](reports/README.md) | 按研究主题组织的结果与综合报告 |
| 数据在哪里？ | [reports/data](reports/data/) | 支撑报告的固定快照；具体条件见对应报告 |
| 专利和信号规范原文在哪里？ | [references](references/README.md) | 外部原始资料与来源记录 |
| 如何查旧稿或恢复旧版本？ | [archive](archive/README.md) | 历史资料、来源备份与构建记录 |

研究结果按接收链路、定位、多径检测分别查阅；外部干扰研究另有入口。目录按资料用途分类，报告内部按研究主题分组。历史版本集中从 archive 进入，避免与当前资料混淆。
