# 2–4 天线空间检测实验数据

开始于2026-09-13，验证结束于2026-09-14。报告由 `build_report.py` 从结果生成，见 [方法、验证、结果与分析](../../2026-09-14-gpspatron-spatial-diversity.md)。

## 内容与分母

- `model`：5000个正常校准事件、22场景各5000个评估事件；每事件同一四天线观测取2短基线/2对角线/3/4子集，各计算全星和三星簇判据。两个完全匹配反例与正常场景共用输入，已明确标记。
- `rf_a`：LOS、全星替代、3星替代；`rf_b`：4星替代、等功率混合、+6 dB混合。每场景两组种子、每组4条通道、600 ms；每条通道8颗GPS L1星。合计48条通道记录、384条PRN跟踪历史，非48次独立攻击。
- 每通道配置和跟踪JSON可重建相位/质量/PVT，`*_observations.npz`是提取结果。未保存巨大原始合成IQ，可用已归档代码/配置/种子重新生成；没有真实多天线IQ。
- `code`：正式RF运行时全部当前Python模块与依赖要求；`tests`：本轮新增测试。`model/code`保存正式模型运行所用的三个直接来源文件，其他依赖共用`code`。模型运行后仅修复归档Path编码，不更改统计模型。
- `regression.log`记录完整回归；`validation.json`记录数据重算和哈希验证，属于可重建输出，不放入顶层固定哈希清单。
- `environment.json`保存实际运行版本；各批`manifest.json`记录假设、种子和代码哈希。顶层`sha256.json`封存全部其他结果、日志和代码快照。
- `plot_code`、`plot_refresh.json`记录出图检查后对全unknown空面板的时间轴和文字修正，未改变接收/检测数据；`refresh_plots.py`可重复生成这些图。

## 复核

从仓库根目录运行：

```powershell
python docs/reports/data/2026-09-13-gpspatron-spatial/verify.py
python docs/reports/data/2026-09-13-gpspatron-spatial/build_report.py
```

核验器逐项重算校准分位数、每事件告警、Wilson区间、配对差异、RF观测、窗口和因果门控，随后验证原始文件和源码哈希。重建报告不会修改原始数据。

## 本地配置示例

在 `python_sim_simplified` 目录运行以下Python。`indices`支持2–4路，但所用门限必须与阵列布局、PRN数、窗口和统计量的校准一致。

```python
import json
from pathlib import Path
import numpy as np
from spatial_spoofing import detect_windows

root = Path('../docs/reports/data/2026-09-13-gpspatron-spatial')
cal = json.loads((root/'model/thresholds.json').read_text(encoding='utf-8'))
obs = np.load(root/'rf_a/partial_3_610000_observations.npz')
layout = '3'  # '2_short', '2_diagonal', '3', '4'
indices = cal['layouts'][layout]
windows = detect_windows(
    obs['time_s'], obs['carrier_m'][:, indices], obs['valid'][:, indices],
    cal['thresholds'][layout+'_cluster3'], cal['wavelength_m'], method='cluster3')
print([(r['decision_s'], r['state']) for r in windows])
```

算法输入不使用归档中的误差真值。`state=0`只表示未见同源证据；`-1`为无法判断，不能作为正常信号。输出门控仅为实验示例，不是认证或恢复导航。

## 外部来源

[GPSPATRON公开空间检测说明](https://gpspatron.com/coherent-gnss-spoofing-detection-multi-antenna/)，访问2026-09-13。公开原则用于方案定位；本文的圆统计、三星簇扫描和门限不是其公开算法或产品参数，不能用本次结果推断GP-Probe性能。
