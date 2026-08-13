# Mate 70 四角阵列空域抗干扰扩展设计

日期：2026-08-12

状态：待用户复核

范围：`python_sim` GPS L1 C/A 端到端仿真

## 1. 目标

把现有“理想双阵元 ULA + 单个静态 CW 干扰”仿真升级为适用于手机四角天线的空域抗干扰研究平台，同时保留当前双天线实验作为可复现回归基线。

新平台优先研究：

- 2～4根手机角点天线的任意组合；
- 阵元数量、角点组合和阵列几何对空间抑制能力的影响；
- 最多3个空间独立静态CW干扰；
- 目标/干扰的方位角与俯仰角；
- MVDR导向矢量失配、对角加载和协方差估计；
- 空域输出指标与完整RF→DFE→捕获→跟踪结果之间的关系。

第一阶段不追求完整电磁、传播或导航解算模型。真实天线方向图、互耦、极化、多径、动态干扰、多星和PVT属于后续扩展层。

## 2. 准确的研究定位

实现后可对外表述为：

> 已构建Python端到端手机多天线GPS L1 C/A空域抗干扰仿真平台，支持Mate 70外轮廓四角中任意2～4个理想相位中心、最多3个静态CW干扰、三维来波方向、SINGLE/EGC/MVDR合并、FFT捕获和闭环跟踪，并可评估阵元组合、空间自由度和导向矢量误差。

不得表述为：

> 已复现华为Mate 70的真实GNSS天线系统。

原因是四个相位中心仅根据机身外轮廓建立，真实天线位置、方向图、极化、互耦和平台效应均未知。

## 3. 手机几何与坐标系

### 3.1 外轮廓尺寸

采用HUAWEI Mate 70标准版官方外形尺寸：

- 长：160.9 mm；
- 宽：75.9 mm；
- 厚：7.8 mm。

规格来源：[HUAWEI Mate 70 官方规格参数](https://consumer.huawei.com/cn/phones/mate70/specs/)。官方同时说明实际尺寸可能因配置、制造工艺和测量方法略有差异。

阵列计算第一阶段仅使用长、宽；厚度作为设备元数据保留。

### 3.2 手机坐标系

手机水平放置、屏幕朝上：

- 原点：手机外轮廓几何中心；
- `+x`：从手机中心指向右侧；
- `+y`：从手机中心指向顶部；
- `+z`：垂直屏幕向上。

俯视示意：

```text
                 +y（手机顶部）
 TL ●────────────────────● TR
    │                    │
    │       原点 O        │
    │                    │
 BL ●────────────────────● BR
                 +x（手机右侧）

 +z：垂直屏幕向上
```

### 3.3 理想四角相位中心

第一阶段相位中心位于外轮廓理想四角：

```text
TL = (-0.03795, +0.08045, 0.0) m
TR = (+0.03795, +0.08045, 0.0) m
BL = (-0.03795, -0.08045, 0.0) m
BR = (+0.03795, -0.08045, 0.0) m
```

坐标不得由代码中的阵元序号隐式推导；配置必须保存角点名称与三维坐标映射。以后获得结构数据后，可分别替换四个坐标。

默认内缩量和高度偏移均为0，但数据结构允许每个角点具有独立三维偏移。

### 3.4 L1电尺寸

在GPS L1 1575.42 MHz、波长约190.29 mm时：

- 短边75.9 mm，约0.399λ；
- 长边160.9 mm，约0.846λ；
- 对角线约177.9 mm，约0.935λ。

长边与对角线基线超过λ/2，可能存在空间方向模糊或栅瓣。二维阵列响应和模糊区域必须作为研究输出，不能只绘制一维角度曲线。

## 4. 方向约定与三维阵列流形

### 4.1 方位角与俯仰角

- `azimuth = 0°`：手机右侧 `+x`；
- 方位角从 `+x` 向 `+y` 逆时针增加；
- `azimuth = 90°`：手机顶部 `+y`；
- `elevation = 0°`：位于手机平面；
- `elevation = 90°`：屏幕正上方 `+z`。

单位来波方向定义为：

\[
\mathbf u(az,el)=
\begin{bmatrix}
\cos(el)\cos(az)\\
\cos(el)\sin(az)\\
\sin(el)
\end{bmatrix}.
\]

### 4.2 阵列流形

第m个阵元的响应定义为：

\[
a_m(az,el,f)
=g_m(az,el,f)
\exp\left(-j\frac{2\pi f}{c}\mathbf r_m^T\mathbf u(az,el)\right).
\]

第一阶段采用理想点阵元：

\[
g_m(az,el,f)=1.
\]

代码仍需定义复数方向图接口，使以后可以用仿真或实测的方向相关复增益替换1，而不修改场景生成、MVDR或后级接收机接口。

### 4.3 共面阵列限制

所有理想角点满足 `z=0`，所以空间相位只依赖来波方向的水平投影：

\[
\mathbf r_m^T\mathbf u
=x_m\cos(el)\cos(az)+y_m\cos(el)\sin(az).
\]

当 `elevation → 90°` 时，`cos(elevation) → 0`，阵元间相位差趋近于0，手机共面阵列的方位分辨能力下降。这一物理限制必须通过目标俯仰角扫描展示。

## 5. 天线选择与组合

用户可从 `TL/TR/BL/BR` 中任选2～4个角点：

```python
cfg.array.active_antennas = ("TL", "BR")
```

合法组合总数为：

- 2天线：6种；
- 3天线：4种；
- 4天线：1种；
- 总计11种。

系统同时支持：

1. 单组合完整链路运行；
2. 自动遍历全部11种组合并比较。

选择子阵列时坐标始终相对于手机中心保存，不重新居中。每次结果必须记录：

- 启用的角点名称；
- 实际三维坐标；
- 阵元数量；
- 最大阵元间基线；
- 阵列流形和二维阵列响应。

预设组合：

- 2天线：`("TL", "BR")`；
- 3天线：`("TL", "TR", "BR")`；
- 4天线：`("TL", "TR", "BL", "BR")`。

新默认入口采用4角全开。

## 6. 目标信号与导向矢量失配

目标信号配置必须把真实方向与算法假设方向分开：

```python
cfg.signal.true_direction = Direction(
    azimuth_deg=30.0,
    elevation_deg=60.0,
)
cfg.signal.assumed_direction = Direction(
    azimuth_deg=30.0,
    elevation_deg=60.0,
)
```

- `true_direction`：场景生成时计算真实目标阵列响应；
- `assumed_direction`：EGC和MVDR计算权值时使用；
- 两者相等：理想已知目标方向；
- 两者不同：DOA失配实验。

通道幅相误差仍应用于场景中的有效响应，但算法可选择使用名义或校准后的阵列流形，从而研究未校准、部分校准和理想校准三种情况。

默认目标方向：

```text
azimuth = 30°
elevation = 60°
```

## 7. 多干扰场景

接收阵列模型升级为：

\[
\mathbf x[n]
=\mathbf a_s s[n]
+\sum_{k=1}^{K}\mathbf a_{j,k}j_k[n]
+\mathbf n[n],
\qquad K=0,1,2,3.
\]

每个CW干扰源独立配置：

- `enable`；
- `power_dbm`；
- `Direction(azimuth_deg, elevation_deg)`；
- 相对IF的 `offset_hz`；
- `initial_phase_rad`；
- `start_ms` 与 `stop_ms`。

示例：

```python
cfg.interferers = (
    CWInterferer(
        power_dbm=-90.0,
        direction=Direction(120.0, 10.0),
        offset_hz=250e3,
        start_ms=0.0,
        stop_ms=40.0,
    ),
    CWInterferer(
        power_dbm=-93.0,
        direction=Direction(220.0, 20.0),
        offset_hz=-400e3,
    ),
    CWInterferer(
        power_dbm=-96.0,
        direction=Direction(310.0, 15.0),
        offset_hz=700e3,
    ),
)
```

第一阶段只实现CW波形，不实现宽带、扫频、脉冲或调制干扰。

### 7.1 自由度实验

在保持一个目标单位响应约束时，M阵元阵列通常剩余M−1个空间自由度。默认重点场景：

- 2天线：1个干扰；
- 3天线：最多2个空间独立干扰；
- 4天线：最多3个空间独立干扰。

系统也允许干扰数超过理论自由度，以形成可解释的失败案例。

干扰数量不等同于协方差秩。相同波形、频率和固定相位关系的干扰可能相干，需在结果中标记干扰参数和估计协方差秩，不能只按干扰个数解释结果。

## 8. 软件组件与数据流

采用结构化扩展现有仿真，不建立过重的通用电磁框架。

```text
PhoneGeometry
├── Mate 70外轮廓尺寸
├── TL/TR/BL/BR三维相位中心
└── 任意active_antennas子阵列

Direction
└── azimuth_deg + elevation_deg

ElementPattern
└── complex_gain(direction, frequency)
    第一阶段返回1，预留实测方向图接口

DesiredSignal
├── true_direction
└── assumed_direction

CWInterferer[]
├── power / direction / offset / phase
└── start_time / stop_time

ArrayManifold
└── 方向图 × 三维空间相位

Experiment
├── 单组合完整链路
├── 全11组合比较
├── 快速二维空域扫描
└── 关键点Monte Carlo完整验证
```

公共数据流：

```text
手机几何 + 激活角点 + 目标/干扰方向
                ↓
三维阵列流形与多通道场景生成
                ↓
RF → ADC → 公共相干DDC → FIR → 抽取
                ↓
SINGLE / EGC / loaded MVDR
                ↓
空域与SINR快速指标
                ↓
关键点FFT捕获 → DLL/FLL/PLL跟踪
                ↓
单次结果 / Monte Carlo统计 / 组合比较
```

RF、DFE、捕获和跟踪继续使用“天线数 × 样点数”接口，无需为2、3、4天线分别实现处理链。

## 9. 配置接口

新默认配置：

```python
from antenna_diversity.config import (
    CWInterferer,
    Direction,
    mate70_default_config,
)

cfg = mate70_default_config()
cfg.array.active_antennas = ("TL", "TR", "BL", "BR")
cfg.signal.true_direction = Direction(30.0, 60.0)
cfg.signal.assumed_direction = Direction(30.0, 60.0)
cfg.interferers = (
    CWInterferer(
        enable=True,
        power_dbm=-90.0,
        direction=Direction(120.0, 10.0),
        offset_hz=250e3,
        initial_phase_rad=0.8,
        start_ms=0.0,
        stop_ms=40.0,
    ),
)
```

几何配置：

```python
cfg.phone.model_name       # "HUAWEI Mate 70"
cfg.phone.length_m         # 0.1609
cfg.phone.width_m          # 0.0759
cfg.phone.thickness_m      # 0.0078
cfg.phone.phase_centers_m  # TL/TR/BL/BR三维坐标映射
```

## 10. 兼容策略

### 10.1 场景入口

- `default_config()`：改为Mate 70四角全开默认配置；
- `mate70_default_config()`：显式返回相同的新默认配置；
- `legacy_ula_config()`：完整保存当前双天线ULA场景和数值参数；
- `run_end_to_end.py`：运行Mate 70新默认；
- `run_legacy_ula.py`：复现现有双天线回归用例。

### 10.2 阵列接口

- 现有一维 `steering_vector()` 暂时保留，保证旧测试和旧场景可复现；
- 新增三维 `array_manifold()`；
- 旧ULA通过明确适配器进入公共场景与合并后级，避免在新三维函数中散布兼容条件。

### 10.3 结果兼容

现有 `SimulationResults` 和 `BranchResult` 主结构保留。真值信息扩展为：

- 激活角点与坐标；
- 目标真实/假设方向与导向矢量；
- 每个干扰的方向、波形分量和导向矢量；
- 阵列流形类型和方向图模型；
- 干扰协方差秩等空间信息。

## 11. 算法范围

第一阶段保留：

- SINGLE；
- EGC；
- 带对角加载MVDR。

不在本阶段增加LCMV、MUSIC、ESPRIT、GSC或鲁棒Capon等算法，以控制范围。

MVDR仍使用：

\[
\hat{\mathbf R}=\frac{1}{N}\mathbf X\mathbf X^H,
\qquad
\delta=\alpha\frac{\operatorname{Re}\{\operatorname{tr}(\hat{\mathbf R})\}}{M},
\]

\[
\mathbf w=
\frac{(\hat{\mathbf R}+\delta\mathbf I)^{-1}\mathbf a_{assumed}}
{\mathbf a_{assumed}^H(\hat{\mathbf R}+\delta\mathbf I)^{-1}\mathbf a_{assumed}}.
\]

第一阶段保留整段数据离线一次计算的基线，同时把协方差窗口和更新时间定义为后续实时扩展点。本阶段至少必须让指标明确记录估计样点数和权值是否为离线固定。

## 12. 分层实验与评价指标

### 12.1 快速空域扫描

大规模角度、阵元组合和参数扫描不运行完整RF与跟踪，而是用阵列流形、协方差或解析信号功率快速计算：

- 目标方向响应；
- 每个干扰方向响应；
- 零陷深度；
- 输出目标功率、总干扰功率、噪声功率；
- 输出SINR；
- 权值范数与白噪声增益；
- 协方差条件数、秩和对角加载量；
- 二维方位角–俯仰角响应。

### 12.2 关键点完整链路

对有代表性的组合和方向运行完整：

```text
场景 → RF → ADC → DFE → 合并 → 捕获 → 跟踪
```

输出：

- 捕获比值与成功标志；
- Doppler误差与码相位误差；
- 跟踪成功；
- PLL RMS、DLL RMS、Prompt幅度和C/N₀；
- 合并前后频谱和捕获热图。

### 12.3 Monte Carlo

- 单次默认运行固定种子，保证复现；
- 批量研究的种子数量可配置；
- 快速开发默认建议20次；
- 正式统计建议50～100次；
- 汇总捕获率、跟踪率、SINR分布和估计误差分布。

## 13. 默认方向与扫描范围

默认目标：

```text
az = 30°, el = 60°
```

默认三个干扰方向预设：

```text
J1: az = 120°, el = 10°
J2: az = 220°, el = 20°
J3: az = 310°, el = 15°
```

重点扫描：

- 目标俯仰角：10°～90°；
- 干扰方位角：0°～360°；
- 干扰俯仰角：0°～60°；
- 目标与干扰空间方向间隔；
- 目标真实方向与假设方向误差；
- 对角加载因子；
- 阵元数量和全部11种角点组合；
- 干扰功率与干扰数量。

二维方向图：

- 横轴：方位角；
- 纵轴：俯仰角；
- 颜色：合并响应dB；
- 标记：目标方向、各干扰方向、零陷和高响应模糊区域。

## 14. 校验与错误处理

配置校验必须拒绝：

- 少于2或多于4个启用角点；
- 重复或未知角点名称；
- 候选坐标不是有限三维向量；
- 方位角或俯仰角不是有限值；
- 俯仰角超出 `[0°, 90°]`；
- 超过3个干扰配置；
- 干扰结束时间早于开始时间；
- 干扰时间超出仿真时间且未经过明确裁剪策略；
- 通道幅相误差长度与激活天线数不一致；
- 阵列流形输出长度与激活天线数不一致；
- 与现有DFE整数samples/chip等链路约束冲突的采样配置。

错误信息必须包含具体配置字段和值，便于批量实验定位失败组合。

## 15. 测试策略

### 15.1 几何与方向

- Mate 70四角坐标与尺寸一致；
- 11种组合枚举无重复、无遗漏；
- 任意角点顺序能够被规范化或明确保留；
- `Direction`边界值和非法值校验；
- `az=0/el=0`、`az=90/el=0`、`el=90`的单位方向正确。

### 15.2 阵列流形

- 原点阵元响应为1；
- 理想点阵元响应幅度为1；
- 天顶方向下共面阵元响应相同；
- 三维流形在特定一维几何与方向约定下可与旧ULA结果对照；
- 激活子阵列结果等于全阵列对应索引切片；
- 复数方向图接口可改变幅相且不改变后级接口。

### 15.3 多干扰场景

- 0、1、2、3个干扰均能生成；
- 每个干扰的时间门控、频率、功率和方向独立生效；
- 场景总和等于目标与各干扰分量逐项相加；
- 真值记录完整且维度与激活天线一致。

### 15.4 合并与自由度

- SINGLE/EGC/MVDR对2、3、4通道均可运行；
- 单位目标响应约束成立；
- 理想可分场景中M阵元可对最多M−1个独立干扰形成预期抑制；
- 超自由度场景不假定成功，但必须数值稳定并给出可解释指标；
- DOA失配和对角加载变化会反映到目标增益和干扰增益；
- 组合扫描只使用对应的通道子集。

### 15.5 端到端与回归

- Mate 70四角新默认完成RF→DFE→三分支→捕获/跟踪；
- 至少一个合理基线场景能产生已定义、可重复的预期结论；
- `legacy_ula_config()`保持当前双天线默认参数与关键结果容差；
- 现有31项测试全部保留或等价迁移；
- 图形函数支持二维阵列响应和任意数量干扰标记；
- 批量组合与Monte Carlo结果可序列化和复现。

## 16. 交付物

实施完成后应包含：

- Mate 70手机几何、方向、阵列流形和方向图接口；
- 任意2～4角点组合；
- 最多3个静态CW干扰；
- 新手机默认和旧ULA回归配置；
- SINGLE/EGC/MVDR在2～4通道上的公共实现；
- 快速空域扫描；
- 单组合与11组合比较；
- 关键点完整端到端运行；
- Monte Carlo框架；
- 二维方位角–俯仰角响应图；
- 更新后的架构图、README和详细设计说明；
- 自动测试与复现基线。

## 17. 明确不在第一阶段实施

- Mate 70真实天线位置；
- 实测或全波仿真的天线复方向图；
- 极化和互耦；
- 多径、Rayleigh/Rician衰落和NLOS；
- 时变手机姿态与动态DOA；
- 宽带、扫频、脉冲或调制干扰；
- 多星、导航数据、位/帧同步和PVT；
- 在线DOA估计；
- LCMV、MUSIC、ESPRIT、GSC或鲁棒Capon；
- 完整模拟RF逐级非线性与硬件实测校准。

这些内容应在理想阵列能力边界、导向失配和自由度实验稳定后逐层加入。
