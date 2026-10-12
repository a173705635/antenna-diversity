# 抗干扰与联合接收链实验

在 `python_sim_simplified/` 目录运行：

```powershell
python -m experiments.interference.full_chain
```

入口为 [full_chain.py](../../experiments/interference/full_chain.py)，输出为 `results/interference/full_chain/`。时长和绘图选项位于入口顶部，信号与阵列条件在 [presets.py](../../core/presets.py) 中。

当前默认配置同时包含 CW、带限噪声干扰和一条多径；比较 SINGLE、EGC、MVDR、Blind Null 以及多径约束 LCMV。它是联合接收链演示，不能将混合场景的收益直接作为纯抗干扰性能。

专用算法在 [interference/](../../core/interference/)：`interference.py` 生成带限噪声，`scene.py` 合成干扰到各天线的响应，`combining.py` 计算盲零陷。共享的天线、RF/DFE、捕获、跟踪、协方差合并及链路编排位于 `core/`；多径专用算法位于 `core/multipath/`。

重跑入口只重建它自己的输出目录。已冻结的历史数据和来源清单保留原内容。
