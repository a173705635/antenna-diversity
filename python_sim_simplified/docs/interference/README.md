# 抗干扰与联合接收链实验

在 `python_sim_simplified/` 目录运行：

```powershell
python -m experiments.interference.full_chain
```

入口为 [full_chain.py](../../experiments/interference/full_chain.py)，输出为 `results/interference/full_chain/`。时长和绘图选项位于入口顶部，信号与阵列条件在 [presets.py](../../core/presets.py) 中。

当前默认配置包含 CW 和带限噪声干扰，比较 SINGLE、EGC、MVDR、Blind Null。`core/config.py` 的 `ENABLE_MULTIPATH_RESEARCH = False` 默认关闭回波生成、检测、评估、多径 LCMV 与专用绘图；原有回波参数和检测/抑制子开关保留。将常量改为 `True` 可恢复默认联合场景，也可在单次配置中设置 `cfg.enable_multipath_research = True`。历史联合场景结果仍按当时配置解读。

专用算法在 [interference/](../../core/interference/)：`interference.py` 生成带限噪声，`scene.py` 合成干扰到各天线的响应，`combining.py` 计算盲零陷。共享的天线、RF/DFE、捕获、跟踪、协方差合并及链路编排位于 `core/`；多径专用算法位于 `core/multipath/`。

重跑入口只重建它自己的输出目录。已冻结的历史数据和来源清单保留原内容。

## 阵列响应绘图

`full_chain` 保留各合并分支的二维响应图，并同时输出独立的 `detailed_array_response_3d_{mode}.png` 三维曲面图。方位角为 0～360°，仰角为 0～90°，间隔为 2°；高度与颜色均表示 `20 log10(|wᴴa| / max|wᴴa|)`，每个分支按自己的扫描最大值归一化。目标和干扰方向始终标出；开启多径研究后才标出回波方向。

显示下限为 −60 dB，低于该值的响应只在绘图时截到 −60 dB；二维和三维使用相同的扫描数据。该图展示当前天线模型、目标极化及载频下的固定权值空间响应，不代表绝对增益或宽带内所有频率的抑制度。入口默认 `SHOW_PLOTS = True`，保存全部图片后同时弹窗显示，三维图可拖动视角。设置 `SHOW_PLOTS = False` 可只保存不显示；无窗口的 Agg 后端只输出文件。

另输出 `detailed_array_response_spherical_{mode}.png` 球面响应色图，实际扫描仰角 −90～90°，覆盖完整球面。球半径固定为 1，坐标为 `x=cos(el) cos(az)`、`y=cos(el) sin(az)`、`z=sin(el)`；颜色表示全空间峰值归一化的 `20 log10(|wᴴa|/max|wᴴa|)`。同一 figure 并排展示两个相反视角，首视角朝向第一个启用的干扰（无干扰时朝向目标方位），对面视角显示另一半球。颜色范围为 −40～0 dB，低于 −40 dB 使用最深色；图例列出的精确方向响应值不截断。目标用绿色星形、CW 用红色叉号、宽带干扰用品红菱形、多径用蓝色圆形，不在球面标注数字；图例同时显示方位角、仰角和具体 dB 响应。隐藏半球的标记不透过球面显示。旧二维/曲面图仍按上半空间峰值归一化；球面图按全空间峰值归一化。三维曲面图也在图例中保留方向角及响应值。

固定球面用于看清不同方向的响应强弱，半径不表示增益；理想单天线应是同色球面。下半空间响应由当前模型实际计算，不把上半球人为镜像；若使用实测方向图，须确认其数据覆盖下半空间，表外插值不能视为实测结果。
