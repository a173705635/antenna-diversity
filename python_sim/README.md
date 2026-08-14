# HUAWEI Mate 70 理想角点 GPS L1 C/A 空域抗干扰仿真

本目录是 GPS L1 C/A 的 Python 原生端到端仿真：等效复数 IF → RF/ADC → DFE → SINGLE/EGC/MVDR → 捕获/跟踪。默认配置使用 **HUAWEI Mate 70 的 160.9 × 75.9 × 7.8 mm 外形尺寸**，并把四个外形角点当作理想、共面、各向同性的相位中心。它们只是首版阵列接口，**不代表已知的真实 Mate 70 天线位置、方向图或产品性能**。

![HUAWEI Mate 70 理想角点空域抗干扰仿真架构](docs/simulation_architecture.svg)

- 对外架构与模型边界：[`ARCHITECTURE.md`](ARCHITECTURE.md)
- 公式、接口和实现映射：[`DESIGN.md`](DESIGN.md)
- 可编辑矢量图：[`docs/simulation_architecture.svg`](docs/simulation_architecture.svg)
- 报告用位图：[`docs/simulation_architecture.png`](docs/simulation_architecture.png)

## 坐标、方向和数据约定

手机中心为原点，`+x` 指向屏幕右侧，`+y` 指向手机顶部，`+z` 垂直于屏幕向外。方位角从 `+x` 朝 `+y` 逆时针增加；仰角从手机平面向 `+z` 增加，范围 0°–90°。方向单位向量为

```text
u = [cos(el) cos(az), cos(el) sin(az), sin(el)]
```

阵列流形使用 `exp(-j 2π f p·u/c)`，位置是三维坐标。共面理想角点在天顶（仰角 90°）具有相同空间相位。所有多通道数组 shape 为 `(channels, samples)`；合并输出为 `(samples,)`。

## 环境与入口

要求 Python 3.11+、NumPy、SciPy、Matplotlib 和 pytest：

```powershell
cd F:\antenna_diversity\python_sim
python -m pip install -e ".[test]"
$env:MPLBACKEND='Agg'
python -W error -m pytest -q
python -W error run_end_to_end.py
python -W error run_legacy_ula.py
```

`default_config()` 是四个理想手机角点的当前默认；`legacy_ula_config()` 保留原始双阵元半波长 ULA 基线。两者共享 channels-by-samples 的 RF、DFE、合并、捕获和跟踪链路。

## 可复现空间研究套件

`run_spatial_research.py` 顺序运行解析扫描、鲁棒性研究、代表场景筛选、完整样本链验证和报告发布。它不打开 GUI；成功后会打印 manifest、报告、八张表和八幅图的绝对路径。仓库内的 [`studies/example_quick.json`](studies/example_quick.json) 是可直接编辑的完整示例，显式列出研究名称、输出目录、主种子、载波/IF/采样与 DFE 频率、目标和干扰方向/功率、扫描轴及三档误差。

```powershell
cd F:\antenna_diversity\python_sim
$env:MPLBACKEND='Agg'

# 全部 11 种稳定顺序组合
python run_spatial_research.py --preset standard --combinations all

# 单个组合
python run_spatial_research.py --preset quick --combinations TL+BR --modes egc,mvdr

# 指定组合列表；逗号分隔，组合内部用 +
python run_spatial_research.py --preset quick --combinations TL+BR,TL+TR+BR --modes egc,mvdr

# 加载可编辑 JSON；只覆盖输出位置和研究名
python run_spatial_research.py --config studies/example_quick.json `
  --output-dir results/studies --name example-quick

# 配置不变时恢复中断运行
python run_spatial_research.py --config studies/example_quick.json --resume
```

配置优先级固定为 `命令行显式覆盖 > JSON > quick/standard/extended 预设 > default_config()`。未在命令行给出的选项不会遮蔽 JSON；已有研究目录默认拒绝覆盖。`--resume` 只在 schema 与科学配置哈希一致时复用哈希校验通过的阶段，从首个缺失、失败或损坏的阶段继续。

每次研究写入 `<output_dir>/<name>/`：

```text
<output_dir>/<name>/
├── manifest.json
├── resolved-config.json
├── checkpoints/
│   ├── baseline.json ... report.json
├── tables/
│   ├── baseline_rankings.csv ... full_chain_cn0.csv
├── figures/
│   ├── baseline_rankings.png ... cn0_validation.png
├── summary.json
└── report.md
```

表中的 `output_sinr_db` / `analytic_output_sinr_db` 是由功率、阵列流形和协方差得到的解析空间指标，不生成时域 RF 样本。`acquisition_metric`、`acquired`、`tracking` 和最终 Doppler 来自 RF → ADC → DFE → 合并 → 捕获/跟踪的样本链；`final_cn0_db_hz` 仅在跟踪给出有限估计时存在，否则为 JSON `null`，原因见同一行的 `status`/`reason`。解析 SINR、捕获指标与 C/N₀不是可互换的量，报告把它们并列作为不同证据。

> **理想模型警告：** 研究套件仍使用 Mate 70 外形四角的理想、共面、各向同性相位中心。排名与报告只比较该模型中的可复现实验，不代表真实天线布局、方向图、互耦、传播环境或产品抗干扰性能。

## 可复制配置示例

任意不重复的 2–4 个角点都可激活；名称和顺序为 `TL`、`TR`、`BL`、`BR`：

```python
from antenna_diversity.config import default_config
from antenna_diversity.pipeline import run_end_to_end

cfg = default_config()
cfg.array.active_antennas = ("TL", "TR", "BR")
cfg.plot.enable = False
results = run_end_to_end(cfg)
```

手机配置支持 0–3 个分别具有方向、功率、频偏、初相和独立时间门的 `CWInterferer`：

```python
from antenna_diversity.config import CWInterferer, Direction, default_config

cfg = default_config()
cfg.interferers = ()  # 零干扰
cfg.interferers = (
    CWInterferer(Direction(120.0, 10.0), power_dbm=-90.0, offset_hz=250e3),
    CWInterferer(
        Direction(250.0, 25.0), power_dbm=-96.0, offset_hz=-180e3,
        initial_phase_rad=1.1, start_ms=2.0, stop_ms=18.0,
    ),
)
```

## 快速组合与方向扫描

解析扫描不生成时域 RF 样本，适合先筛选参数；它们会深拷贝配置，不改写调用者输入：

```python
import numpy as np
from antenna_diversity.config import default_config
from antenna_diversity.experiments import (
    compare_corner_combinations,
    scan_interferer_directions,
)

cfg = default_config()
all_subarrays = compare_corner_combinations(cfg, mode="mvdr")  # 11 个子集
direction_grid = scan_interferer_directions(
    cfg, interferer_index=0,
    azimuths_deg=np.arange(0.0, 360.0, 15.0),
    elevations_deg=np.arange(0.0, 91.0, 10.0), mode="mvdr",
)
```

同一模块还提供干扰功率、目标假设方向、位置误差和对角加载扫描。

## 顺序 Monte Carlo

本阶段故意顺序运行：随机状态和结果顺序与传入 seed 完全一致，不暴露并行参数。返回值只保留每 seed、每分支的标量记录、捕获/跟踪率以及 acquisition metric、最终 Doppler 误差、输出 SINR、最差干扰增益的分布；端到端链未提供的输出 SINR 明确为 `None`，不保存原始样本数组。

```python
import json
from antenna_diversity.config import default_config
from antenna_diversity.experiments import run_monte_carlo

cfg = default_config()
cfg.plot.enable = False
mc = run_monte_carlo(cfg, seeds=(101, 102, 103))
print(json.dumps(mc.to_dict(), ensure_ascii=False, indent=2, allow_nan=False))
```

## 当前模型边界

当前已实现 3-D 阵列流形和阵元方向图协议，但首版只使用各向同性阵元。以下内容明确未建模：真实 Mate 70 天线位置、真实测量方向图、互耦、极化、多径/NLOS、动态 DOA、非 CW 干扰、多颗卫星，以及导航数据到 PVT 的完整解算。默认结果只说明给定理想模型中 MVDR 的可重复空间抑制与后续捕获/跟踪表现，不是产品级声明。

运行脚本生成的频谱、空间响应、捕获热力图、跟踪曲线和研究报告位于忽略版本控制的 `results/` 目录。
