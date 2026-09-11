# GNSS 天线分集与空域抗干扰仿真

本仓库研究 GNSS 多通道接收链路、GPS L1/L5 定位、多径检测与空域抗干扰。当前只维护 `python_sim_simplified/`。多径检测与外部干扰研究分别组织，具体已验证范围以 [项目状态](PROJECT_STATUS.md) 为准。

## 研究入口

以下入口均位于 `python_sim_simplified/`，从项目根目录执行 `python python_sim_simplified/<脚本名>`。

| 研究任务 | 入口 | 验证记录 |
|---|---|---|
| 双 RF / DFE / 独立环路相位差 | `audit_dual_tracking_phase.py` | [双通道相位](docs/reports/2026-09-10-dual-tracking-phase.md) |
| GPS 码/载波浮点定位 | `run_gps_spp.py` | [定位验证](docs/reports/2026-09-11-code-carrier-float.md) |
| GPS L1/L5 定位 | `run_gps_l1_l5.py` | [双频权重](docs/reports/2026-09-11-dual-frequency-optimization.md) |
| 多径：校准取余 SDCP | `audit_calibrated_sdcp.py` | [实现与边界](docs/reports/2026-09-11-calibrated-sdcp.md) |
| 多径：500 ms / 1 s 窗口 | `audit_sdcp_long.py` | [长记录验证](docs/reports/2026-09-11-sdcp-long.md) |
| 多径：SDCP 接入定位 | `audit_sdcp_positioning.py` | [定位效果](docs/reports/2026-09-11-sdcp-positioning.md) |
| 外部干扰：空域合并 | `run_end_to_end.py` | [配置与运行说明](python_sim_simplified/README.md) |

各实验有不同前提，不能把不同 SDCP 版本的结果混为同一流程。输入、输出和已有结果目录的处理方式见对应脚本说明与报告。

## 目录职责

| 目录或文件 | 用途 |
|---|---|
| [`python_sim_simplified/`](python_sim_simplified/) | 当前主开发版本；用于日常修改、单场景仿真和结果作图 |
| [旧版 python_sim 归档](docs/archive/python_sim_2026-09-12/README.md) | 历史源码、批量研究能力及结果备份；停止维护 |
| [`PROJECT_STATUS.md`](PROJECT_STATUS.md) | 当前进度、结论、建模边界与下一阶段 |
| [`基础知识/`](docs/knowledge) | 按学习顺序整理的 GNSS 与阵列抗干扰理论知识库 |
| [`验证方案/`](docs/validation) | 专项验证方案和评审材料 |
| [文档导航](docs/README.md) | 理论、验证、来源与归档的统一入口 |
| [`docs/reports/`](docs/reports/) | 实验报告与固定数据快照 |
| [`docs/theory/`](docs/theory/) | 原理章节与历史 Word 说明 |
| [`docs/archive/`](docs/archive/) | 仅供追溯的历史项目记录 |
| [`docs/archive/development/`](docs/archive/development) | 开发过程中的设计与实施记录，不代表当前状态 |

发生内容冲突时，以 `PROJECT_STATUS.md`、当前代码及对应 Python 目录的 README 为准。

## 快速运行当前版本

```powershell
cd F:\antenna_diversity
python -m pip install -r python_sim_simplified/requirements.txt
python python_sim_simplified/run_end_to_end.py
python -m pytest -q python_sim_simplified/tests
```

仿真结果保存到 `python_sim_simplified/results/`。默认场景包含一个 CW 干扰和一个 2 MHz 带限复高斯噪声干扰；具体配置、18 张结果图及代码阅读顺序见 [`python_sim_simplified/README.md`](python_sim_simplified/README.md)。

当前只维护 `python_sim_simplified/`。旧版批量研究框架已[归档](docs/archive/python_sim_2026-09-12/README.md)，需要追溯时可恢复；它没有同步最新多径和定位实现。

## 研究边界

当前模型用于验证算法机制和敏感性，不代表真实 Mate 70 产品性能。仓库没有 Mate 70 的真实天线位置、复数方向图、S 参数和多 RF 校准数据；当前宽带空域处理仍是一组覆盖整个 DFE 带宽的固定复权值，并非分频段或时域宽带波束形成。

旧 MATLAB 与 Galileo E5/AltBOC 实现已从当前工作树移除。如需追溯，可从 Git 提交 `6f5e686` 恢复；后续新增能力默认只维护 Python 版本。

## 综合报告

[GNSS 天线分集全量整合报告](docs/reports/GNSS天线分集全量整合报告.docx)：综合验证结论与来源全文。
