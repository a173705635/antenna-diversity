# GNSS 天线分集多径检测与抑制设计

日期：2026-09-07

## 1. 目标

在 `python_sim_simplified/` 当前 GPS L1 C/A 端到端链路中加入可配置的静态镜面多径，并实现不读取仿真真值的码延迟–空域联合检测，以及基于估计多径方向的 LCMV 抑制。

本阶段回答三个问题：

1. 多径经过手机四角天线、RF、DFE 和现有合并算法后，如何改变相关峰、捕获和码跟踪；
2. 2～4 天线能否从 DFE 多通道相关结果中检测可分辨多径并估计其方向；
3. 在 LOS 方向已知、多径方向未知的条件下，LCMV 能否降低多径响应和码偏差。

仿真真值只用于结果评估，不得作为检测或 LCMV 权值计算的输入。

## 2. 范围

### 2.1 本阶段包含

- 任意数量的可配置静态镜面反射径，默认启用一条；
- 每条路径独立的超额时延、相对功率、DOA、反射相位、多普勒偏差和极化；
- 多径完整码包络延迟及对应 RF 载波相位；
- 各路径独立的天线方向图、空间相位、遮挡、互耦和通道响应；
- DFE 后的码延迟–空间方向联合检测；
- LOS 直达分量投影、残差延迟次峰检测和 Bartlett DOA 扫描；
- `multipath_lcmv` 合并分支；
- 多径检测、抑制、相关峰和码跟踪偏差指标及独立图片；
- 无多径、可分辨多径和不可分辨多径的自动化验证。

### 2.2 本阶段不包含

- 对 `python_sim/` 完整版的任何同步修改或功能移植；
- 随时间运动的反射点和动态多径；
- 弥散多径、统计信道模型和纯 NLOS；
- 分频段 MVDR、时域抽头波束形成或完整 STAP；
- MEDLL、SAGE 等高分辨率码域参数估计；
- 导航数据位、多星、多 PRN 和 PVT；
- 利用仿真真实多径参数直接生成抑制权值。

## 3. 多径配置

在 `config.py` 增加：

```python
@dataclass
class MultipathPath:
    direction: Direction
    enable: bool = True
    relative_power_db: float = -6.0
    excess_delay_chips: float = 0.5
    reflection_phase_rad: float = 0.0
    doppler_offset_hz: float = 0.0
    polarization: WavePolarization = field(default_factory=WavePolarization)
```

`SignalConfig` 增加：

```python
multipath_paths: list[MultipathPath] = field(default_factory=list)
```

`relative_power_db` 是相对直达径参考接收功率的功率比。`excess_delay_chips` 是相对直达径的额外传播时延，换算关系为：

```text
excess_delay_s = excess_delay_chips / code_rate_hz
excess_path_m = excess_delay_s × c
```

`reflection_phase_rad` 表示反射面除传播时延外引入的附加复反射相位。超额时延本身已经同时产生码包络延迟和 RF 载波相位，不得再用该字段重复补偿传播相位。

默认场景启用一条方位角 110°、仰角 15°、超额时延 0.5 chip、相对功率 -6 dB、反射相位 π 的反射径，以便形成可重复的演示场景。用户可将列表设为空以完全关闭多径。

## 4. 多径信号生成

直达径保持现有实现。对每条启用的反射径：

1. 从同一 PRN、码率和卫星信号生成延迟副本；
2. 路径总延迟等于公共超额传播时延与逐通道真实时延之和；
3. 对完整 IF 序列执行整数/分数样点延迟，并补齐下变频模型省略的 RF 等效相位；
4. 施加相对幅度、反射相位和路径多普勒偏差；
5. 根据该路径的方向和极化独立计算真实天线响应；
6. 将所有路径与干扰源相加后送入现有 RF/ADC 和 DFE。

数据流为：

```text
LOS ───────────────────────┐
MP1：延迟/衰减/相位/DOA ───┼→ 各路径天线响应 → 通道延迟 → 求和
MP2：延迟/衰减/相位/DOA ───┘                         │
CW / 宽带干扰 ──────────────────────────────────────┤
                                                   ↓
                                             RF/ADC → DFE
```

`ScenarioTruth` 增加每条多径的组件、有效真实导向矢量、真实超额时延、相对功率、方向和天线诊断量。这些字段只允许被指标与绘图使用。

## 5. 多径检测

增加 `MultipathDetectionConfig`：

```python
@dataclass
class MultipathDetectionConfig:
    enable: bool = True
    max_excess_delay_chips: float = 3.0
    delay_step_chips: float = 0.25
    detection_threshold_db: float = 6.0
    azimuth_step_deg: float = 5.0
    elevation_step_deg: float = 5.0
```

另增加：

```python
@dataclass
class MultipathSuppressionConfig:
    enable: bool = True
    diagonal_loading_factor: float = 1e-2
    max_steering_correlation: float = 0.95
```

`SimulationConfig` 分别通过 `multipath_detection` 和 `multipath_suppression` 持有以上配置。`multipath_lcmv` 不放入需要预先执行的基础 `diversity.modes`，而是在检测完成且抑制功能启用后由流水线追加，从而避免循环依赖。

检测流程：

1. 先运行现有 SINGLE、EGC、MVDR 和 Blind Nulling 分支；
2. 从捕获成功的基础分支中选择捕获指标最高者，使用其 Doppler 和码相位作为检测起点；
3. 对每根天线 DFE 输出分别进行载波剥离和本地码相关；
4. 在直达径之后 `0～max_excess_delay_chips` 范围内形成复数阵列相关向量 `z(τ)`；
5. 使用 assumed LOS 导向矢量构造正交投影，得到：

   ```text
   z_perp(τ) = [I - a0 a0^H / (a0^H a0)] z(τ)
   ```

6. 在 `||z_perp(τ)||²` 上按照阈值、最小峰间隔和局部极大值检测残差路径；
7. 对每个候选路径使用 assumed 天线模型执行 Bartlett 方位–仰角扫描；
8. 按检测强度排序，最多保留 `天线数−1` 条空间独立路径。

检测模块不得导入或接收 `ScenarioTruth`。真值匹配在独立的评估函数中完成。

当前 DFE 约为 4 samples/chip，默认延迟步进 0.25 chip。小于约 0.25～0.5 chip 的近端多径可能只能体现为相关峰畸变和 DLL 偏差。与 LOS 空间导向矢量高度相关的多径会被 LOS 投影同时删除，必须报告为空间不可分辨。

## 6. LCMV 多径抑制

增加 `multipath_lcmv` 分支。约束矩阵和响应向量为：

```text
C = [a_LOS, a_MP1, ..., a_MPK]
f = [1, 0, ..., 0]^T
```

使用带对角加载的样本协方差 `R_L` 计算：

```text
w = R_L^-1 C (C^H R_L^-1 C)^-1 f
```

权值满足 LOS 单位响应，并在估计的多径方向形成零陷。多径约束数 `K` 不得超过 `M−1`。若候选路径与 LOS 或已有路径的归一化导向矢量相关度过高，则将其标记为 `spatially_unresolvable`，不加入约束矩阵。

LCMV 合并后重新执行捕获和跟踪，以便与基础四分支在相同输入数据上比较。

状态处理：

- 无基础分支捕获：检测状态为 `no_pilot_acquisition`，不生成虚假的多径估计；
- 未检测到可用多径：`multipath_lcmv` 使用普通 loaded MVDR 权值，并记录 `no_multipath_detected`；
- 候选路径不可空间分辨：记录原因，不强制零陷；
- 路径过多：只使用检测强度最高且空间独立的 `M−1` 条。

### 6.1 精简版保护逻辑边界

精简版不恢复完整版的大量通用保护、异常包装和工程框架，只保留三类会直接影响研究结论的检查：

1. 物理配置有效性，例如多径时延非负、功率和角度为有限值；
2. 算法状态，例如没有初始捕获、没有检测到多径或空间不可分辨；
3. 数值稳定性，例如协方差对角加载、近共线约束剔除和自由度限制。

不增加文件打开失败包装、catch-all 异常、重复字段检查、checkpoint/schema 校验或为了极端非法调用而层层嵌套的分支。研究配置错误应在入口处直接报出清晰异常，核心数学流程保持线性、可读。

## 7. 输出与指标

检测结果包含：

- 原始码延迟功率曲线；
- 去除 LOS 后的残差延迟曲线；
- 检测状态和选用的基础捕获分支；
- 每条候选路径的估计超额时延、检测强度、估计 DOA 和可分辨状态；
- 独立的 Bartlett 方向扫描结果。

评估指标包含：

- 检测率、漏检数和虚警数；
- 超额时延误差和 DOA 夹角误差；
- 各合并分支对真实 LOS 和各真实多径的阵列增益；
- LCMV 相对输入的多径抑制度；
- 捕获码相位相对真实 LOS 的偏差；
- 跟踪码偏差的 chips 和 metres 表示；
- 原有捕获指标、DLL/PLL、Prompt 和 C/N0。

真值路径与检测路径采用时延优先、角度辅助的一对一匹配，仅用于评估。

## 8. 绘图

保留现有 18 张图，并新增独立图片：

- `multipath_delay_profile.png`：原始与 LOS 投影后的延迟功率曲线、检测阈值和路径标记；
- `multipath_angle_path_<n>.png`：每条检测路径独立的 Bartlett 方向图；
- `multipath_correlation_comparison.png`：各分支在 LOS 附近的相关函数；
- `multipath_tracking_bias.png`：各成功跟踪分支的码偏差，单位为 chips 和 metres。

不把多个方向图或已有频谱重新拼接到同一张图，避免降低可读性。检测路径数量不同，因此最终图片总数允许随场景变化。

## 9. 代码边界

- `config.py`：多径路径、检测与抑制配置；
- `gnss.py`：多径传播路径生成和真值记录；
- `multipath.py`：码延迟相关、LOS 投影、峰检测、Bartlett DOA 和真值评估；
- `diversity.py`：通用 LCMV 权值和合并；
- `pipeline.py`：基础分支、检测、LCMV 分支的顺序编排；
- `models.py`：多径真值、检测结果、LCMV 信息和指标类型；
- `metrics.py`：检测、抑制和码偏差指标；
- `plotting.py`：新增四类独立图片；
- `presets.py` / `run_end_to_end.py`：默认演示配置；
- `README.md` / `PROJECT_STATUS.md`：配置说明和项目进度。

多径数学处理集中在 `multipath.py`，不得继续扩大 `gnss.py` 或 `pipeline.py` 的职责。

## 10. 验证标准

自动化测试至少覆盖：

1. 多径列表为空时，原有场景采样保持不变；
2. 单条路径的相对功率、完整序列延迟、RF 等效相位和天线方向响应正确；
3. 多条路径能独立叠加；
4. 检测模块接口不包含 `ScenarioTruth`；
5. 可分辨的无噪或高信噪场景能检出已配置的延迟和方向；
6. 近时延或近方向场景报告不可分辨或未检出，而非给出虚假成功；
7. LCMV 数值满足 `w^H a_LOS ≈ 1` 和 `w^H a_MP ≈ 0`；
8. 默认确定性场景中，LCMV 相比 SINGLE/EGC 降低真实多径方向增益和码跟踪偏差；
9. 无检测结果时，LCMV 明确退化到 loaded MVDR；
10. 现有精简版测试继续通过，默认运行能够生成新增指标和图片。

## 11. 成功判据

本阶段完成时，应能通过一份可修改的配置复现实验：关闭多径得到原基线；开启一条或多条多径后观察相关峰畸变；检测模块只根据接收数据给出延迟与方向估计；LCMV 在空间可分辨场景保持 LOS、抑制反射方向，并减少相对 SINGLE/EGC 的码跟踪偏差。对近时延、近方向或自由度不足的场景，输出必须明确说明能力边界。
