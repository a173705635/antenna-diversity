# 阶段数据快照

从 `python_sim_simplified/results/` 复制的 2026-09-10 阶段结果，保留原始字节，不随研究脚本重跑覆盖。

- `detection_calibration/`：500 条校准和 1600 条独立评估记录。
- `detection_boundaries/`：22 个点各 100 条记录、汇总和图；跨点共享种子，不可按独立样本合并。
- `detection_only/`：早期候选和参数估计结果。
- `multipath_baseline_audit/`：此前数值自查保留下来的 JSON/PNG 文件；不是完整中间波形存档。
- `sha256.json`：上述 36 个数据/图片文件的 SHA-256。

在仓库根目录执行（使用项目已有 Python 依赖）：

```powershell
python docs/reports/data/2026-09-10-multipath/verify.py
python -m pytest python_sim_simplified/tests -q
```

核验脚本检查数据字节、重算冻结门限和独立评估/边界统计、检查种子分离，不重新生成仿真。它不验证物理模型适用性。

完整重跑入口在 `python_sim_simplified/README.md`。边界脚本会复用本地 results 下的 checkpoint；改变代码或场景后应先保留旧结果并使用干净的输出目录，不能混用旧记录。快照本身不作为运行输出目录。

本阶段收尾：85 项精简版测试通过；原始统计复核通过。该结果不等于多径检测通用可靠性验证完成。结论和文献适用范围见[阶段总结](../../2026-09-10-multipath-stage-summary.md)。
