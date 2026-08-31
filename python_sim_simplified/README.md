# GNSS 天线分集精简教学版

本目录是独立的 RF → DFE → 天线合并 → 捕获 → 跟踪仿真副本，便于按信号流阅读和修改。它不导入 `python_sim/`，也不会替代原来的完整研究工程。

## 运行

在仓库根目录执行：

```powershell
python python_sim_simplified/run_end_to_end.py
```

程序打印 SINGLE、EGC、MVDR、Blind Null 的捕获/跟踪指标，并把七张独立图片保存到 `python_sim_simplified/results/`。

## 常用修改

在 `run_end_to_end.py` 的 `cfg = default_config()` 后直接修改，例如：

```python
cfg.signal.carrier_hz = 1_575.42e6
cfg.signal.if_hz = 4.092e6
cfg.dfe.if_hz = cfg.signal.if_hz
cfg.phone.active_antennas = ("TL", "BR")
cfg.phone.channel_phase_error_deg["BR"] = 45.0
cfg.interferers[0].power_dbm = -95.0
```

天线配置入口：

```python
from pathlib import Path
from config import parametric_config, measured_config

cfg = parametric_config(("TL", "TR", "BR"))
cfg = measured_config(Path("pattern.csv"))
```

测量 CSV 列为：

```text
element_name,frequency_hz,azimuth_deg,elevation_deg,h_az_real,h_az_imag,h_el_real,h_el_imag
```

## 推荐阅读顺序

`config.py` → `antenna.py` / `gnss.py` → `rf.py` → `dfe.py` → `diversity.py` → `acquisition.py` → `tracking.py` → `pipeline.py` → `plotting.py`

## 有意删除的内容

本副本假定输入有效，删除了配置全量校验、异常包装、文件哈希和 schema 审计、零分母与退化矩阵保护、Blind Null 安全旁路、断点恢复、研究批处理、Monte Carlo、报告系统以及定点载波 NCO 对比。遇到错误配置时，Python/NumPy 会直接报错或产生非有限值。

保留下来的是有效输入下的物理模型、算法公式、Blind Null 检测判决和捕获成功后才进入跟踪的判决。需要鲁棒研究和异常诊断时，请使用 `../python_sim/`。
