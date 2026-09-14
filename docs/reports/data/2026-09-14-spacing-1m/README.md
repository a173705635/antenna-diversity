# 1 米阵列对照数据

2026-09-14：使用 1 × 1 m 正方形替换原 75.9 × 160.9 mm 矩形；不使用 IMU。四角坐标依次为 TL=(-0.5,0.5,0)、TR=(0.5,0.5,0)、BL=(-0.5,-0.5,0)、BR=(0.5,-0.5,0)，米为单位。信号生成时将 TL 作为局部参考。

## 设计

两尺寸使用相同随机种子、天空/姿态/噪声抽样及 global/cluster3 判据，但分别校准 1% 名义虚警阈值。2/3/4 天线由相同四通道记录抽取；两天线同时有边长、对角线配置。保留所有负面结果，不对两个判据作未经校准的融合。

旧结果来自相邻的 `2026-09-13-gpspatron-spatial`，不重新调参。新模型 22 个场景各 5000 事件；校准 5000 事件。RF 六场景、每场景两个独立噪声种子、四天线、600 ms。RF 内窗口相关，不按 8 个独立实验解释。

## 复现

在仓库根目录执行，输出目录必须尚不存在：

```powershell
python python_sim_simplified/audit_spatial_diversity.py --output <new-model-dir> --width-m 1 --height-m 1
python python_sim_simplified/audit_spatial_diversity_rf.py --output <new-rf-a-dir> --model <new-model-dir>/thresholds.json --cases los all_spoof partial_3
python python_sim_simplified/audit_spatial_diversity_rf.py --output <new-rf-b-dir> --model <new-model-dir>/thresholds.json --cases partial_4 mixture_0 mixture_6
```

RF 默认种子为 610000、620000；每条天线实际噪声种子为 seed+channel。RF 几何直接从阈值文件读取，避免使用了新阈值却仍生成旧孔径。旧格式阈值没有 positions_m 时，仅为向后兼容默认使用原手机尺寸。

`code/` 是本次运行时的完整顶层 Python 源码快照。用该目录对应脚本也可运行以上命令。`tests.log` 是本次相关 17 项测试的真实输出，包含 1 米几何的解析传播差核验。`model.log`、`rf_a.log`、`rf_b.log` 保留执行进度；各输出子目录保留自身校验和。

## 重新核验与生成报告

```powershell
python docs/reports/data/2026-09-14-spacing-1m/analyze.py
```

脚本使用 `code/` 快照，从 NPZ 重新计算全部检测计数和尺寸间配对差值，从 RF JSON 重新构建观测/判决，并检查旧归档保持原字节。生成 validation.json、model_comparison.json、图及上层中文报告。不重新生成 RF，不改变冻结阈值。

`--model-only` 可在 RF 运行期间只生成模型比较。报告中的 short_aperture 场景是在各自基准尺寸上缩小十倍；新旧绝对孔径不同。1 米正方形与原矩形的长宽比不同，本轮不是保持形状的纯尺度变化。

## 解释限制

固定电缆偏差、同步误差、天线互耦/方向图、真实多径未随孔径新增；数字 RF 仍是固定天空、固定欺骗方向和静态替代/混合。结果描述本地判据的仿真识别能力，不能代表 GPSPATRON 产品、真实硬件或恢复导航能力。
