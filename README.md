# GNSS 天线分集与空域抗干扰仿真

本仓库以 Python 为唯一开发基线，面向手机四角的 2–4 天线组合，研究 GPS L1 C/A 从天线端口、RF/ADC、DFE、空域处理到捕获、跟踪和 C/N0 评估的完整链路。

当前主信号流为：

```text
GPS L1 C/A + 0–3 个静态 CW 干扰
→ 2–4 个手机角点天线
→ RF / ADC
→ DFE
→ SINGLE / EGC / MVDR / Blind Nulling
→ 捕获 ACQ
→ 跟踪 TRK / C/N0
→ 扫描、Monte Carlo 与研究报告
```

## 从这里开始

| 文档 | 职责 |
|---|---|
| [`PROJECT_STATUS.md`](PROJECT_STATUS.md) | 当前进度、已经证明的结论、风险与下一阶段 |
| [`python_sim/README.md`](python_sim/README.md) | 环境安装、运行命令、配置示例和实验入口 |
| [`python_sim/DESIGN.md`](python_sim/DESIGN.md) | 数学模型、配置接口、算法公式与代码映射 |
| [`python_sim/ARCHITECTURE.md`](python_sim/ARCHITECTURE.md) | 对外架构、中央数据路径、已建模与未建模边界 |
| [`基础知识/README.md`](基础知识/README.md) | GNSS 窄带阵列抗干扰理论知识库 |

`docs/superpowers/`保存开发过程中的设计和实施记录，用于追溯决策，不代表当前项目状态。发生冲突时，以`PROJECT_STATUS.md`和当前Python文档为准。

## 快速运行

```powershell
cd F:\antenna_diversity\python_sim
python -m pip install -e ".[test]"
$env:MPLBACKEND='Agg'
python -W error -m pytest -q
python -W error run_end_to_end.py
```

当前能力包括理想、合成参数化和外部测量表格三类天线接口，以及SINGLE、EGC、loaded MVDR和捕获前Blind Nulling。具体适用条件和未建模项以架构与设计文档为准，当前结果不代表真实Mate 70产品性能。

## 历史实现

旧MATLAB与Galileo E5/AltBOC实现已经从当前工作树移除。需要追溯或重新实现时，可从Git提交`6f5e686`恢复；后续新增能力默认只维护Python版本。
