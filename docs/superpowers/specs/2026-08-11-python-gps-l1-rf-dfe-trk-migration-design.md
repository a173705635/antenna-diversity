# Python GPS L1 C/A RF→DFE→TRK 迁移设计

## 1. 目标

在不修改 `matlab_sim` 的前提下，新建独立的 `python_sim`，用 Python 原生科学计算工具完整复现当前 GPS L1 C/A 双天线仿真链路：

```text
GPS L1 C/A + 连续波干扰
→ 双天线空间响应与通道幅相误差
→ RF 级联增益、Friis 噪声、限幅、复数 ADC
→ 公共 NCO 下变频、低通 FIR、4 倍抽取、输出量化
→ single / EGC / MVDR 合并
→ FFT 并行码相位捕获
→ DLL + FLL 辅助 PLL 跟踪
→ 指标汇总和捕获、频谱、方向图、跟踪结果作图
```

迁移是“算法与物理行为等价”，不是 MATLAB 语法的机械逐行翻译。Python 代码应便于后续开展天线分集参数扫描和算法扩展。

## 2. 已确认范围

### 包含

- 当前 `matlab_sim/default_config.m` 中的 GPS L1 C/A 默认场景与配置参数。
- PRN 1～32 的 GPS L1 C/A 码生成。
- 双天线 ULA 导向矢量、通道幅相误差、目标信号和单音窄带干扰。
- RF 级联、热噪声、软/硬限幅、I/Q ADC 量化。
- DDC、窗函数 sinc FIR、群延迟对齐、抽取和低位宽量化。
- single、EGC、带对角加载的 MVDR。
- 多普勒 × 码相位二维捕获、主峰/第二峰与峰值比判决。
- 1 ms Early/Prompt/Late DLL、FLL 辅助 PLL 跟踪。
- 与 MATLAB 版本表达相同物理含义的指标和图形。
- Python 单元测试和端到端回归测试。

### 不包含

- `matlab_sim/+galileo` 和 Galileo E5/AltBOC 全带宽跟踪；它将在 GPS 迁移验证后独立设计。
- 多星、多径、导航电文、伪距、定位、STAP 或自适应导向矢量估计。
- MATLAB 与 Python 随机噪声逐样点相同。
- GPU、JAX、Numba 或实时流式处理。
- 修改或删除现有 MATLAB 文件。

## 3. 方案选择

采用 Python 原生并行实现：保留 MATLAB 的算法边界和变量物理含义，使用 Python 命名、类型标注、`dataclass` 配置和 pytest 测试。

未采用以下方案：

- 逐行翻译：短期看似直接，但会把 MATLAB 的一基索引、结构体和隐式行列规则带进 Python，后续维护困难。
- 一开始建立 MATLAB/Python 逐样点联合回归：两个运行时的随机数和舍入细节不同，成本高于当前收益。第一阶段采用确定性模块对照与端到端物理结论对照。

## 4. 技术栈和运行约束

- Python 3.11 或更高版本。
- NumPy：数组、复数运算、FFT 和线性代数。
- SciPy：FIR 因果滤波。
- Matplotlib：频谱、方向图、捕获热力图和跟踪曲线。
- pytest：单元测试与端到端测试。
- 不引入 pandas；指标表使用轻量的记录列表和文本格式化，避免为一张小表增加依赖。
- 默认使用 `complex128`/`float64`，量化结果仍存为复数数组，但 I/Q 分量必须为整数值。

## 5. 目录和模块边界

```text
python_sim/
├── README.md
├── pyproject.toml
├── run_end_to_end.py
├── antenna_diversity/
│   ├── __init__.py
│   ├── config.py
│   ├── models.py
│   ├── numeric.py
│   ├── pipeline.py
│   ├── gnss/
│   │   ├── __init__.py
│   │   ├── ca_code.py
│   │   └── scenario.py
│   ├── array/
│   │   ├── __init__.py
│   │   └── steering.py
│   ├── rf/
│   │   ├── __init__.py
│   │   ├── cascade.py
│   │   └── frontend.py
│   ├── dfe/
│   │   ├── __init__.py
│   │   ├── decimator.py
│   │   └── processing.py
│   ├── diversity/
│   │   ├── __init__.py
│   │   └── combiner.py
│   ├── receiver/
│   │   ├── __init__.py
│   │   ├── acquisition.py
│   │   └── tracking.py
│   └── analysis/
│       ├── __init__.py
│       ├── metrics.py
│       └── plotting.py
└── tests/
    ├── test_core.py
    ├── test_rf.py
    ├── test_dfe.py
    ├── test_diversity.py
    ├── test_acquisition.py
    ├── test_tracking.py
    ├── test_plotting.py
    └── test_end_to_end.py
```

每个文件只承担一个清晰职责。`pipeline.py` 只负责串联各阶段，不复制阶段内部算法；`run_end_to_end.py` 是面向使用者的脚本入口。

## 6. 配置与结果接口

### 配置

`config.py` 使用嵌套 `dataclass` 表达：

- `SignalConfig`
- `ArrayConfig`
- `JammerConfig`
- `RFConfig` 与 `RFStage`
- `DFEConfig`
- `DiversityConfig`
- `AcquisitionConfig`
- `TrackingConfig`
- `PlotConfig`
- `SimulationConfig`

`default_config()` 返回与 MATLAB 默认值相同的配置。配置保持可修改，方便如下参数扫描：

```python
cfg = default_config()
cfg.jammer.power_dbm = -95.0
cfg.diversity.diagonal_loading_factor = 1e-3
results = run_end_to_end(cfg)
```

`default_config()` 和各处理函数都执行必要的输入校验。非法采样率、偶数 FIR 长度、维度不一致、无效 PRN、位宽或空输入使用带明确说明的 `ValueError`。

### 结果

`models.py` 定义少量顶层结果数据类：

- `BranchResult`：`combined`、`combining`、`acquisition`、`tracking`。
- `SimulationResults`：配置、真值、RF/DFE 中间数据、各分支和指标。

每个算法函数返回 `numpy.ndarray` 与描述该阶段的字典；避免为每个中间标量建立大量数据类。分支通过 `results.branches["mvdr"]` 访问。

## 7. 数据约定和跨语言差异

### 数组维度

- 多通道信号统一为 `(n_antennas, n_samples)`。
- 单路合并信号统一为一维 `(n_samples,)`。
- 导向矢量和权值统一为一维 `(n_antennas,)`。
- 捕获搜索功率为 `(n_doppler_bins, coherent_samples)`。
- 合并采用 `y = w.conj() @ X`，协方差采用 `R = X @ X.conj().T / n_samples`。

所有公开函数都验证维度，禁止依赖 NumPy 广播来掩盖天线数或采样数错误。

### 索引和相位

- Python 内部统一零基索引。
- `code_phase_samples` 和 `epoch_start_samples` 对外也保持零基，这与 MATLAB 结果中显式减一后的语义相同。
- MATLAB 的 `firstValid = groupDelay + 1` 翻译为 Python 的 `filtered[:, group_delay::decimation]`。
- 角度仍以阵列宽侧为零度，导向矢量相位符号保持不变。

### 舍入和随机数

- `numeric.py` 提供 MATLAB 风格的“中点远离零”舍入函数，用于 ADC 和 DFE 量化，避免 NumPy 银行家舍入带来的边界差异。
- 随机噪声使用局部 `numpy.random.Generator`，不污染全局随机状态；固定种子必须在 Python 内部可复现。
- 不要求 NumPy 与 MATLAB `twister` 产生相同噪声样本。因此跨语言验收比较统计量、估计结果和算法结论，不比较含噪数组的逐元素相等性。

## 8. 处理流程

`run_end_to_end(cfg)` 顺序执行：

1. 生成目标 PRN 波形、阵列响应、CW 干扰和真值。
2. 计算 Friis 级联噪声系数，加入输入等效热噪声，施加电压增益、限幅和 ADC 量化。
3. 所有通道使用同一 NCO 搬移中频，经过同一 FIR，再统一丢弃群延迟并抽取。
4. 对同一份 DFE 数据依次执行 single、EGC、MVDR 合并。
5. 每个分支独立捕获；只有捕获成功才进入跟踪，否则返回 `status="acquisition_failed"`。
6. 汇总干扰方向增益、捕获结果和稳态跟踪误差。
7. 根据绘图配置保存或显示图形。

默认配置下，single 和 EGC 受强窄带干扰影响而捕获失败；MVDR 应抑制干扰方向并成功捕获和跟踪。这是当前演示场景的预期结果，不被解释为所有场景下 MVDR 必然优于 EGC。

## 9. 绘图设计

`analysis/plotting.py` 提供互相独立的函数：

- RF 输入、ADC 输出、DFE 输出频谱。
- single/EGC/MVDR 阵列方向响应，并标出目标和干扰方向。
- 每个分支的二维捕获热力图和最佳 Doppler 下的码相位切片；标出主峰、第二峰和 PASS/FAIL。
- 成功分支的 Doppler 跟踪和 Prompt 幅值。

绘图函数返回 Matplotlib `Figure`，便于测试和二次编辑。默认端到端入口将图片保存到 `python_sim/results/`；`cfg.plot.show` 控制是否调用 `plt.show()`。测试使用非交互式 `Agg` 后端，不弹窗。

## 10. 测试与验收

### 单元测试

Python 测试逐项覆盖 MATLAB 当前八类回归行为：

- C/A 码长度、取值和 Gold 码自相关集合。
- 导向矢量相位约定。
- 场景维度、码 Doppler 和天线间复响应。
- Friis 级联、固定种子可复现、ADC 范围与整数性。
- DDC 归零、FIR 输出长度和跨通道相位保持。
- EGC/MVDR 目标响应为一，MVDR 显著降低确定性干扰方向增益。
- 高信噪比信号的捕获 Doppler 和码相位误差。
- 跟踪输出有限、稳态 Prompt、DLL/PLL RMS 和最终 Doppler 误差。
- 捕获图包含热力图、切片和峰值标记。

### 端到端验收

以 2026-08-11 通过的 MATLAB R2023a 基线为参照：

- MATLAB 当前八项测试全部通过。
- 默认场景中 single 与 EGC 捕获失败，MVDR 捕获成功。
- MATLAB MVDR 基线：捕获 Doppler `1500 Hz`，最终 Doppler 约 `1498.4 Hz`，干扰方向增益约 `0.00450`。

Python 端到端测试不锁死随机噪声导致的全部小数，而要求：

- 三个合并分支均被执行。
- 默认种子下 single、EGC 捕获失败，MVDR 捕获及跟踪成功。
- MVDR 捕获 Doppler 不超过一个搜索 bin 的误差。
- MVDR 最终 Doppler 与真实值误差不超过 `250 Hz`。
- MVDR 干扰方向增益小于 single，且确定性合并测试中小于 single 的 10%。
- 所有跟踪状态量为有限值，稳态 DLL/PLL RMS 均小于 `0.5`。
- `pytest` 全部通过，端到端脚本能够生成结果表和图片。

如果 Python 随机样本令默认场景处在阈值边缘，应先核查算法、量化、索引和归一化差异；不得仅通过放宽捕获阈值来制造通过结果。

## 11. 文档与使用体验

`python_sim/README.md` 说明：

- 安装依赖和运行测试的命令。
- 运行默认端到端仿真的命令。
- 各阶段的物理意义、主要数组维度和关键配置入口。
- MATLAB 与 Python 舍入、随机数及索引差异。
- 如何修改干扰功率/角度、阵元间距、通道幅相误差和 MVDR 对角加载。
- Galileo E5 尚未迁移，避免用户误以为两条链路已同时覆盖。

## 12. 完成标准

当且仅当以下条件全部满足，GPS 第一阶段迁移完成：

1. `matlab_sim` 未被修改。
2. `python_sim` 可以独立安装/运行。
3. 全部 Python 单元和端到端测试通过。
4. 默认场景复现“single/EGC 失败、MVDR 成功”的物理结论。
5. 捕获热力图、方向图、频谱图和跟踪图可以保存并正常打开。
6. README 足以让用户修改关键参数并理解 RF→DFE→分集合并→捕获→跟踪的数据流。
