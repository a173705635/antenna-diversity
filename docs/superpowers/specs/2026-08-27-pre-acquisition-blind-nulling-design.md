# 捕获前盲抗干扰 V1 设计

## 1. 目标

在现有 GPS L1 C/A Python 端到端仿真中新增 `blind_null` 分支。该分支直接根据 DFE 多通道 IQ 的样本协方差检测并估计一个主导干扰空间，在 GNSS 捕获和跟踪之前形成零陷。

算法不得使用目标真实/假设方向、目标导向矢量、PRN、捕获结果或跟踪结果。仿真流水线可以在权值形成之后使用场景真值计算评价指标，但真值不能反向进入算法。

V1 研究范围固定为：

- 2–4 个激活天线；
- 一个占主导的静态相关干扰源；
- 整段 DFE 数据估计一次协方差和一组固定权值；
- 干扰检测门控、盲零陷和安全旁路；
- 端到端 ACQ/TRK/C/N₀ 验证；
- 与 `single`、`egc`、`mvdr` 同场景对比。

多干扰只能作为诊断输入，V1 不承诺同时消除多个干扰。实时滑窗、权值平滑、检测迟滞和动态相位漂移不属于本阶段。

## 2. 架构

新增独立模块：

```text
antenna_diversity/diversity/blind_null.py
```

该模块的公开接口只接收 DFE 多通道样本和盲零陷配置，不接收目标导向矢量：

```text
DFE channels + BlindNullConfig
        │
        ▼
sample covariance
        │
        ▼
normalized-correlation eigenspectrum
        │
        ├── no dominant interference ──→ reference-channel bypass
        │
        └── dominant interference
                 │
                 ▼
       principal interference vector
                 │
                 ▼
       orthogonal projection weights
                 │
                 ├── safe ──→ blind-null output
                 └── unsafe → reference-channel bypass
```

端到端数据路径为：

```text
DFE channels
    ├─→ SINGLE ─────┐
    ├─→ EGC ────────┤
    ├─→ MVDR ───────┼─→ ACQ → gated TRK → metrics/plots
    └─→ BLIND_NULL ─┘
```

`single`、`egc`、`mvdr` 继续使用现有 `combiner.py`。pipeline 对 `blind_null` 做显式分派，调用不含目标导向参数的独立接口。不得为了复用现有函数而把目标导向矢量传入盲模块后“忽略”。

## 3. 配置

新增 `BlindNullConfig`，并作为 `DiversityConfig` 的嵌套配置：

```python
@dataclass
class BlindNullConfig:
    detection_threshold_db: float = 6.0
    reference_channel: int = 0
    min_reference_projection_power: float = 0.01
```

字段语义：

- `detection_threshold_db`：归一化相关矩阵主模态优势门限；6 dB 只是研究起点，不是工程定值；
- `reference_channel`：旁路输出和盲权值归一化所使用的激活通道索引；
- `min_reference_projection_power`：投影后参考向量剩余功率的安全下限，防止噪声增益过大。

默认 `DiversityConfig.modes` 增加 `blind_null`，使普通端到端入口默认产生四个分支。配置校验要求：

- 门限是有限实数；
- 参考通道是非负整数，运行时还必须小于实际通道数；
- 最小投影功率是 `(0, 1]` 内的有限实数；
- 模式只允许 `single`、`egc`、`mvdr`、`blind_null`。

研究配置中：

- `full_chain_modes` 允许 `blind_null`；
- `spatial_modes` 不允许 `blind_null`，因为解析路径没有 DFE 样本协方差；
- 错误信息必须说明 `blind_null` 仅支持完整样本链，不能只说“unknown mode”。

## 4. 干扰检测

输入 `X` 的 shape 固定为 `(M, N)`，其中 `M` 为 2–4，`N` 为 DFE 样点数。样本协方差为：

\[
\widehat{R}=\frac{XX^H}{N}.
\]

数值实现将协方差显式 Hermitian 化：

\[
R_H=\frac{\widehat{R}+\widehat{R}^H}{2}.
\]

令通道功率矩阵为：

\[
D=\operatorname{diag}(R_H).
\]

检测使用功率归一化相关矩阵：

\[
C=D^{-1/2}R_HD^{-1/2}.
\]

这样固定通道增益差不会单独制造主特征值。对 `C` 使用 Hermitian 特征分解，并按升序得到：

\[
\lambda_1\le \lambda_2\le\cdots\le\lambda_M.
\]

主模态优势分数为：

\[
S_{det}=10\log_{10}
\left(
\frac{\lambda_M}
{\frac{1}{M-1}\sum_{m=1}^{M-1}\lambda_m}
\right).
\]

若分母为零且最大特征值大于零，分数为正无穷并判定存在主导模态；全零输入不判定为干扰并走旁路。只有 `S_det >= detection_threshold_db` 时才尝试形成零陷。

检测器同时报告：

- 每通道功率；
- 样本协方差和归一化相关矩阵；
- 升序特征值；
- 主模态优势分数和配置门限；
- 最大通道间归一化相关系数；
- `interference_detected`。

最大相关系数只作为诊断量，不作为 V1 的第二个门限，避免引入未经验证的联合判决逻辑。

## 5. 主干扰向量和盲权值

设 `u_max` 是 `C` 最大特征值对应的单位特征向量。将它映射回原始通道尺度：

\[
\widetilde{v}_j=D^{1/2}u_{max},
\qquad
v_j=\frac{\widetilde{v}_j}{\|\widetilde{v}_j\|_2}.
\]

`v_j` 只由 DFE 样本得到，是主导相关空间模态的估计，不声称是物理 DOA。

构造干扰正交投影：

\[
P_\perp=I-v_jv_j^H.
\]

令 `q` 是 `reference_channel` 位置为 1、其他位置为 0 的单位基向量，定义：

\[
\gamma=q^HP_\perp q.
\]

若 `gamma >= min_reference_projection_power`，使用：

\[
w=\frac{P_\perp q}{\gamma}.
\]

这保证：

\[
w^Hq=1,
\qquad
w^Hv_j=0
\]

（数值误差范围内）。合并输出沿用现有约定：

\[
y=w^HX.
\]

该归一化只固定参考通道响应，不保护未知 GNSS 方向。如果卫星空间响应接近估计干扰空间，盲零陷可能同时衰减卫星信号；这是必须报告的失效条件，不得用场景真值修正权值。

## 6. 旁路和错误行为

旁路权值是 `w=q`，输出必须与选择的参考通道逐样点一致。

合法旁路原因包括：

- `no_interference_detected`：检测分数低于门限；
- `zero_input_power`：所有通道功率为零；
- `unsafe_reference_projection`：`gamma` 小于安全下限；
- `degenerate_interference_vector`：映射后的干扰向量无法安全归一化。

旁路不是异常。结果必须记录：

- `nulling_enabled=False`；
- `bypass_reason`；
- 参考通道索引；
- 实际旁路权值。

以下情况属于调用或配置错误，必须抛出明确异常：

- 输入不是非空二维 channels-by-samples 数组；
- 通道数少于 2；
- 输入含 NaN 或无穷；
- 参考通道越界；
- 配置字段类型、范围或有限性不合法。

不允许返回 NaN、无穷权值或静默截断非法通道索引。

## 7. 结果和事后评价

`blind_null` 的 `combining` 字典保持与其他分支共有的字段：

- `mode`；
- `weights`；
- `weight_norm`；
- `sample_covariance`；
- `condition_number`；
- `covariance_snapshots`；
- `weights_are_fixed=True`。

另增加：

- `channel_powers`；
- `normalized_correlation`；
- `correlation_eigenvalues`；
- `detection_score_db`；
- `detection_threshold_db`；
- `maximum_pairwise_coherence`；
- `interference_detected`；
- `nulling_enabled`；
- `bypass_reason`；
- `reference_channel`；
- `reference_projection_power`；
- `estimated_interference_vector`；
- `estimated_interference_response`；
- `reference_response`。

pipeline 在权值形成后继续利用 truth 计算：

- 实际目标方向增益；
- 每个实际干扰方向增益；
- 最差干扰增益。

这些字段只用于评价和画图。测试必须证明改变 assumed target steering 不会改变 `blind_null` 权值或输出。

## 8. 绘图和研究框架

普通端到端绘图将已存在的 `blind_null` 分支追加到固定显示顺序：

```text
single, egc, mvdr, blind_null
```

至少以下图支持第四分支：

- 合并后频谱；
- 手机阵列空间响应；
- 捕获热力图；
- 跟踪曲线；
- C/N₀。

图形布局根据实际分支数生成，不能继续假设固定三张子图。现有三分支自定义配置仍必须可运行。

七阶段研究允许 `blind_null` 出现在 `full_chain_modes`、完整链表格、捕获/跟踪率和相应图中。解析扫描、解析排名和 `spatial_modes` 保持现有算法集合，不伪造盲零陷的解析结果。

## 9. 相位一致性边界

V1 不要求预先知道固定的逐通道相位偏置。固定偏置已经包含在 DFE 样本协方差中，因此估计出的主干扰向量会观察到该偏置。

V1 要求在“协方差估计窗口 + 固定权值使用窗口”期间，多通道相对相位基本稳定。如果相对相位持续漂移：

- 干扰空间会在估计窗口内扩散；
- 固定零陷方向会随时间失配；
- 抑制度下降，严重时可能失效。

因此本阶段只能证明静态或准静态相位条件下的算法能力。多 RF 差分频偏、动态相位漂移、校准周期、分块协方差、权值更新和平滑必须作为下一阶段单独建模。

## 10. 验收标准

### 10.1 单元测试

1. 无干扰或独立噪声输入低于门限，输出与参考通道逐样点一致；
2. 单个强相关干扰产生高于门限的主模态分数；
3. 启用零陷后，对估计主干扰向量的响应显著小于参考通道响应；
4. 权值满足参考通道单位响应；
5. 2、3、4 通道均能运行；
6. 固定通道相位偏置下仍能估计并压制主相关模态；
7. 投影退化时安全旁路且权值有限；
8. 非有限输入、错误 shape 和参考通道越界被拒绝；
9. 改变目标 assumed steering 不改变盲权值和盲输出。

### 10.2 集成测试

1. 默认端到端结果包含 `single`、`egc`、`mvdr`、`blind_null`；
2. 构造一个代表场景，使 SINGLE 捕获失败而 `blind_null` 恢复捕获和跟踪；
3. `blind_null` 的捕获、跟踪、C/N₀、频谱和空间响应均可生成；
4. 自定义只运行原有三分支时行为保持兼容；
5. 七阶段完整样本链接受 `blind_null`，解析扫描明确拒绝；
6. 现有 SINGLE、EGC、MVDR 回归结论不退化。

### 10.3 完成判据

- 新增测试和完整现有测试集通过；
- 主要入口在 warnings-as-errors 模式下运行通过；
- README、DESIGN、ARCHITECTURE 和 PROJECT_STATUS 与实现一致；
- 结果说明中明确区分“盲算法输入”和“仿真事后 truth 评价”；
- 不宣称已解决多干扰、动态相位或真实硬件校准问题。

## 11. 非目标

本阶段不实现：

- 多干扰数目估计和多维干扰子空间投影；
- MUSIC/ESPRIT 或干扰 DOA 输出；
- GNSS 目标导向保护或鲁棒约束；
- 在线滑窗、重叠分块、检测迟滞、权值平滑；
- 动态相位漂移补偿；
- 多 RF 硬件校准算法；
- 定点 EVD、定点协方差或定点盲权值；
- 非 CW、动态、多径和多星场景扩展。
