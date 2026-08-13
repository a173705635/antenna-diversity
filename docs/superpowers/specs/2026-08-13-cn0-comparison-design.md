# 独立 C/N₀ 对比图设计

## 目标

在现有 RF → DFE → SINGLE/EGC/MVDR → ACQ/TRK 仿真链路中，保留当前基于跟踪 Prompt 的 C/N₀ 估计，并新增一张独立的 C/N₀ 随时间对比图。图中只展示成功进入跟踪的分支；捕获失败的分支明确标注为无法估计，不生成理论值或伪造数据。

## 当前状态

跟踪器已经为每个跟踪历元计算 `cn0_db_hz`。现有估计使用长度由 `TrackingConfig.cn0_window_epochs` 控制的 Prompt 滑动窗口：

\[
P_{\mathrm{coherent}}=\left|\operatorname{mean}(P_k)\right|^2
\]

\[
P_{\mathrm{noise}}=\max\left(\operatorname{mean}(|P_k|^2)-P_{\mathrm{coherent}},\epsilon\right)
\]

\[
C/N_0=10\log_{10}\left(\frac{P_{\mathrm{coherent}}}{P_{\mathrm{noise}}T_{\mathrm{coh}}}\right)
\quad\text{dB-Hz}
\]

其中 `T_coh` 是一个跟踪历元的相干积分时间。窗口不足两个 Prompt 样本时结果保持 `NaN`。

## 设计

### 计算

- 不新增第二套 C/N₀ 算法，继续以 `receiver/tracking.py` 产生的 `cn0_db_hz` 作为唯一测量来源。
- 将窗口估计提取为可独立测试的小函数；跟踪循环调用该函数，公式及数值保护保持一致。
- 输入必须是一维复数 Prompt 序列，积分时间必须为有限正数。
- 少于两个样本时返回 `NaN`；有效输入返回 dB-Hz 标量。

### 绘图

新增 `plot_cn0(results)`，创建一张独立图：

- 横轴：Tracking time (ms)。时间由每个分支的 `epoch_start_samples` 和 DFE 输出采样率换算；若历元为固定 1 ms，显示仍来自真实采样位置，而不是写死序号。
- 纵轴：Estimated C/N₀ (dB-Hz)。
- 每个成功跟踪且包含有限 C/N₀ 样本的分支绘制一条曲线，图例沿用 `SINGLE`、`EGC`、`MVDR`。
- 捕获或跟踪失败的分支不绘制无效曲线，并在图内标注 `<MODE>: acquisition failed — C/N₀ unavailable` 或 `<MODE>: tracking unavailable — C/N₀ unavailable`。
- 图包含网格、图例、明确单位和标题，不人为平滑或裁剪数值。
- 若所有分支都没有有效 C/N₀，仍返回带状态说明的有效 Figure，而不是抛出绘图错误。

### 输出集成

- `plot_results()` 新增键 `cn0_comparison`，对应 `plot_cn0()` 返回的 Figure。
- 两条 CLI 自动保存 `cn0_comparison.png`，标准输出图数由 5 张变为 6 张。
- 现有指标表继续使用最后一个有限 C/N₀；列标题从 `FinalCn0` 改为 `FinalCn0DbHz`，明确单位。
- 其他五个图的键名和行为保持不变。

## 错误与边界

- 捕获失败不是异常，是“无法从跟踪 Prompt 估计 C/N₀”的正常状态。
- 跟踪成功但早期窗口不足时保留 `NaN`，绘图只画有限点。
- 分支的跟踪数组长度不一致时，每条曲线独立使用自己的历元时间。
- `epoch_start_samples` 与 `cn0_db_hz` 长度不一致属于结果结构错误，`plot_cn0()` 抛出带分支名的 `ValueError`。

## 测试与验收

- 单元测试验证 C/N₀ 小函数：已知 Prompt 窗口、少于两个样本、非法积分时间及非一维输入。
- 绘图测试验证：成功分支曲线、失败 SINGLE 状态标注、坐标单位、全失败场景、数组长度不一致错误。
- 集成测试验证 `plot_results()` 保留原有五个键并新增 `cn0_comparison`。
- CLI 测试更新为检查两条入口均保存 6 张图，且在 `MPLBACKEND=Agg` 与 `-W error` 下无警告。
- 完成后运行相关定向测试及完整 `python -W error -m pytest -q`。

## 非目标

- 不计算捕获前或理论 C/N₀。
- 不为失败的 SINGLE 分支补造数值。
- 不改变捕获门限、跟踪环路、组合权重或 RF/DFE 建模。
- 不将 C/N₀ 合并进现有 tracking comparison 图。
