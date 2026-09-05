# GNSS 天线分集与空域抗干扰仿真

本仓库以 Python 为开发基线，面向手机四角的 2–4 天线组合，研究 GPS L1 C/A 从天线端口、RF/ADC、DFE、空域合并到捕获、跟踪和 C/N₀ 评估的完整链路。

当前主要信号流为：

```text
GPS L1 C/A + CW / 带限宽带干扰
→ 2–4 个手机角点天线
→ RF / ADC
→ DFE
→ SINGLE / EGC / MVDR / Blind Nulling
→ 捕获 ACQ
→ 跟踪 TRK / C/N₀
```

## 目录职责

| 目录或文件 | 用途 |
|---|---|
| [`python_sim_simplified/`](python_sim_simplified/) | 当前主开发版本；用于日常修改、单场景仿真和结果作图 |
| [`python_sim/`](python_sim/) | 完整研究框架；保留批处理、扫描、Monte Carlo、报告和更完整的保护逻辑 |
| [`PROJECT_STATUS.md`](PROJECT_STATUS.md) | 当前进度、结论、建模边界与下一阶段 |
| [`基础知识/`](基础知识/) | 按学习顺序整理的 GNSS 与阵列抗干扰理论知识库 |
| [`验证方案/`](验证方案/) | 专项验证方案和评审材料 |
| [`docs/reports/`](docs/reports/) | 对外报告、原理章节和汇报材料 |
| [`docs/archive/`](docs/archive/) | 仅供追溯的历史项目记录 |
| [`docs/superpowers/`](docs/superpowers/) | 开发过程中的设计与实施记录，不代表当前状态 |

发生内容冲突时，以 `PROJECT_STATUS.md`、当前代码及对应 Python 目录的 README 为准。

## 快速运行当前版本

```powershell
cd F:\antenna_diversity
python -m pip install -r python_sim_simplified/requirements.txt
python python_sim_simplified/run_end_to_end.py
python -m pytest -q python_sim_simplified/tests
```

仿真结果保存到 `python_sim_simplified/results/`。默认场景包含一个 CW 干扰和一个 2 MHz 带限复高斯噪声干扰；具体配置、18 张结果图及代码阅读顺序见 [`python_sim_simplified/README.md`](python_sim_simplified/README.md)。

需要运行参数扫描、Monte Carlo 或完整研究报告流程时，再使用 [`python_sim/README.md`](python_sim/README.md)。完整版尚未同步精简版最新的宽带干扰模型，因此两套目录不是逐文件镜像。

## 研究边界

当前模型用于验证算法机制和敏感性，不代表真实 Mate 70 产品性能。仓库没有 Mate 70 的真实天线位置、复数方向图、S 参数和多 RF 校准数据；当前宽带空域处理仍是一组覆盖整个 DFE 带宽的固定复权值，并非分频段或时域宽带波束形成。

旧 MATLAB 与 Galileo E5/AltBOC 实现已从当前工作树移除。如需追溯，可从 Git 提交 `6f5e686` 恢复；后续新增能力默认只维护 Python 版本。
