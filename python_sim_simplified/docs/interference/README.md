# 抗干扰与联合接收链实验

在 `python_sim_simplified/` 目录运行：

```powershell
python -m experiments.interference.full_chain
```

入口为 [full_chain.py](../../experiments/interference/full_chain.py)，输出为 `results/interference/full_chain/`。时长和绘图选项位于入口顶部，信号与阵列条件在 [presets.py](../../core/presets.py) 中。

当前默认配置同时包含 CW、带限噪声干扰和一条多径；比较 SINGLE、EGC、MVDR、Blind Null 以及多径约束 LCMV。它是联合接收链演示，不能将混合场景的收益直接作为纯抗干扰性能。

专用算法在 [interference/](../../core/interference/)：`interference.py` 生成带限噪声，`scene.py` 合成干扰到各天线的响应，`combining.py` 计算盲零陷。共享的天线、RF/DFE、捕获、跟踪、协方差合并及链路编排位于 `core/`；多径专用算法位于 `core/multipath/`。

重跑入口只重建它自己的输出目录。已冻结的历史数据和来源清单保留原内容。

## 阵列响应绘图

`full_chain` 保留各合并分支的二维响应图，并同时输出独立的 `detailed_array_response_3d_{mode}.png` 三维曲面图。方位角为 0～360°，仰角为 0～90°，间隔为 2°；高度与颜色均表示 `20 log10(|wᴴa| / max|wᴴa|)`，每个分支按自己的扫描最大值归一化。目标、干扰和多径方向在曲面上标出。

显示下限为 −60 dB，低于该值的响应只在绘图时截到 −60 dB；二维和三维使用相同的扫描数据。该图展示当前天线模型、目标极化及载频下的固定权值空间响应，不代表绝对增益或宽带内所有频率的抑制度。交互显示时可拖动三维视角；在入口设置 `SHOW_PLOTS = True` 并使用交互式 Matplotlib 后端。

另输出 `detailed_array_response_spherical_{mode}.png` 球面响应色图，实际扫描仰角 −90～90°，覆盖完整球面。球半径固定为 1，坐标为 `x=cos(el) cos(az)`、`y=cos(el) sin(az)`、`z=sin(el)`；颜色表示全空间峰值归一化的 `20 log10(|wᴴa|/max|wᴴa|)`。同一 figure 并排展示两个相反视角，首视角朝向第一个启用的干扰（无干扰时朝向目标方位），对面视角显示另一半球。颜色范围为 −40～0 dB，低于 −40 dB 使用最深色；下方列出的精确方向响应值不截断。编号对应目标、干扰和多径；同方向的编号合并显示，隐藏半球的标记不透过球面显示。旧二维/曲面图仍按上半空间峰值归一化；球面图按全空间峰值归一化。

固定球面用于看清不同方向的响应强弱，半径不表示增益；理想单天线应是同色球面。下半空间响应由当前模型实际计算，不把上半球人为镜像；若使用实测方向图，须确认其数据覆盖下半空间，表外插值不能视为实测结果。
