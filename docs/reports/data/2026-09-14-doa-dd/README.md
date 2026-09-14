# 原手机阵列方向候选与星间双差归档

## 已执行

- 原尺寸 75.9 × 160.9 mm，2–4 天线，无 IMU。
- 双差检测：5000 H0 校准，22 场景各 5000 留出事件；与原判据配对。
- 测向模型：100 随机源方向，3/4 天线，极性已知、半周未解、错误零偏和已校准四类条件，共 800 个拟合结果。原始带噪相位与半周分支保存在 direction_model_inputs.json。
- 数字 RF：读取相邻 2026-09-13-gpspatron-spatial 的原尺寸观测 NPZ，校准单差后拟合所有触发窗口；不新增 RF，也不接收真实发射信号。
- 参考星双差：固定 PRN 8，包含真实参考与被污染参考两类条件，80 个窗口。对有候选的窗口额外注入公共通道偏差检查候选集合不变。

## 复现

从仓库根目录执行，`<新输出目录>` 必须不存在：

```powershell
python python_sim_simplified/audit_spatial_direction.py --output <新输出目录> --archive docs/reports/data/2026-09-13-gpspatron-spatial
python python_sim_simplified/audit_reference_direction.py --archive docs/reports/data/2026-09-13-gpspatron-spatial --output <新输出目录>/reference_rf.json
python -m pytest python_sim_simplified/tests -q
```

`code/` 为执行源码快照，也可以使用快照中的同名脚本运行。`run.log` 为主审计真实输出，`tests.log` 为完整回归输出。

```powershell
python docs/reports/data/2026-09-14-doa-dd/build_report.py
```

该命令使用源码快照，核验旧归档和本次得分/方向残差，再生成 validation.json、候选图和上层中文报告。它不改阈值，不重新拟合或选择真方向。每个输出候选都保留，所有“最小角误差”均为事后评估候选集合覆盖，不能当成算法已选中的方向精度。

## 能力边界

两个同源 PRN 的星间双差把共同方向一起消掉。只有可信参考方向或经过校准的单差才能提供绝对方向约束。参考星真实性、安装朝向、相位弧稳定性均不是本轮算法自动保证的。

原手机阵列仍存在周期多解和平面上下镜像；半周兼容拟合尤其明显。全局数值搜索不代表形式化证明候选已穷尽。8 mm 拟合残差及 2 mm 候选保留容差是研究参数，不是概率置信度。

RF global/cluster3 取 OR 仅用于扩充本轮测向分析触发范围，未经融合虚警校准，不作为新的检测性能声明。DD 检测是单独校准对照，未默默接入默认检测器或输出保护逻辑。
