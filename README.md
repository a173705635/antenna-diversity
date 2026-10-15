# GNSS 天线分集与空域抗干扰仿真

当前维护代码聚焦抗干扰，位于 `python_sim_simplified/`。公共接收链集中在 `core/`，每个抗干扰实验在 `experiments/interference/` 有独立入口；通用 GPS L1/L5 定位和真实录波接收仍保留。2026-10-14 已备份并移除抗欺骗、多径专用代码及公共链路中的多径接口，具体改动与实际验证见 [开发记录](python_sim_simplified/CHANGELOG.md)。

## 运行入口

在 `python_sim_simplified/` 目录执行。参数和输入输出路径固定在脚本顶部，无需命令行参数；生成型实验重跑只重建自己的输出目录。

| 实验 | 命令 |
|---|---|
| 完整 RF→DFE→合并→捕获→跟踪及绘图 | `python -m experiments.interference.full_chain` |
| 天线单因素消融、噪声标定和独立功率预算 | `python -m experiments.interference.antenna_ablation` |
| 目标/CW/宽带响应直接赋值与相位扫描 | `python -m experiments.interference.direct_response_boundary` |
| 理想双天线整通道相位误差与准确补偿 | `python -m experiments.interference.channel_phase_validation` |
| GPS L1 码/载波浮点定位 | `python -m experiments.gps_spp` |
| GPS L1/L5 仿真定位 | `python -m experiments.gps_l1_l5` |
| 真实录波 GPS L1 码定位 | `python -m experiments.recorded_iq` |

`full_chain` 默认包含 CW 与 2 MHz 带限复高斯噪声干扰，比较 SINGLE、EGC、MVDR、Blind Null。配置、代码阅读顺序及独立实验说明见 [精简版 README](python_sim_simplified/README.md) 和 [抗干扰说明](python_sim_simplified/docs/interference/README.md)。

```powershell
cd F:\antenna_diversity
python -m pip install -r python_sim_simplified/requirements.txt
cd python_sim_simplified
python -m experiments.interference.full_chain
python -m pytest -q
```

## 本地备份与历史资料

清理前版本在本地 `python_sim_simplified/archive/pre_interference_only_2026-10-14_156b994/`；其 `FREEZE_MANIFEST.json` 记录源码、配置、测试和说明的 SHA256。归档与 `results/` 继续只保留在本地，不上库。

抗欺骗和多径报告保留历史条件：[单源抗欺骗报告](python_sim_simplified/docs/spoofing/SINGLE_SOURCE_SUPPRESSION_REPORT.md)、[SDCP 多径报告](python_sim_simplified/docs/multipath/SDCP_MULTIPATH_REPORT.md)、[历史阶段总结](docs/reports/2026-09-12-multipath-stage-closeout.md)。这些报告中的旧命令和源码路径不是当前入口，复现需要对应冻结源码及原始数据，不能把当前代码冒充为原版本。

| 资料 | 入口 |
|---|---|
| 当前范围与历史阶段记录 | [项目状态](PROJECT_STATUS.md) |
| 理论、验证、来源和归档导航 | [文档导航](docs/README.md) |
| 实验报告与固定数据快照 | [docs/reports/](docs/reports/) |
| GNSS 与阵列抗干扰理论 | [基础知识](docs/knowledge/README.md) |
| 停止维护的历史完整版 | [旧版归档说明](docs/archive/python_sim_2026-09-12/README.md) |

## 研究边界

当前模型验证算法机制和敏感性，不代表真实 Mate 70 产品性能。仓库没有该手机的真实相位中心、复数方向图、S 参数和多 RF 校准数据；当前空域处理使用覆盖整个 DFE 带宽的固定复权值，尚非分频段或时域宽带波束形成。历史报告及其来源哈希保持原样，本次删除不产生新的性能结论。
