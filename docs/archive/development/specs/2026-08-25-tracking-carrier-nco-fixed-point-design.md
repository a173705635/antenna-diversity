# TRK Carrier NCO 定点化设计

日期：2026-08-25

## 1. 背景

当前 Python 仿真中的跟踪模块使用浮点载波频率和相位状态。项目实测背景显示，L1/L5 的积分 Doppler 与载波相位变化存在稳定的跨频差异；30-bit Carrier NCO 的频率步进值得单独验证，但现有证据不足以把观测到的约 2.8 mm/s 偏差归因于 NCO 量化。

本设计在不改变多天线 RF → DFE → 空间合并 → ACQ/TRK 主架构的前提下，为 TRK 增加可选的 Carrier NCO 定点模型。第一阶段只隔离 Carrier NCO 的量化影响，不把相关器、鉴相器、环路滤波器、Code NCO 或 DFE 内部运算改成定点。

当前 DFE 边界保持不变：ADC 输出先经过浮点 DDC、浮点 FIR 和抽取，随后 I/Q 分量按配置进行 8-bit 有符号饱和量化。TRK 接收整数码值，但数组容器仍为 `complex128`。

## 2. 目标

1. 保留现有浮点 TRK 作为严格参考模式。
2. 增加硬件语义明确的有符号 FCW 和无符号相位累加器模型。
3. 支持在相同输入、捕获结果和环路参数下比较 30-bit 与 43-bit Carrier NCO。
4. 记录 FCW、量化频率、量化误差和相位码字，支持长期平均与码字跳变分析。
5. 提供可直接运行的 L1 对比脚本、CSV 和图形证据。
6. 保持 SINGLE、EGC、MVDR、C/N0、研究套件和默认 CLI 的兼容性。

## 3. 非目标

- 不实现完整的 TRK 全定点化。
- 不定点化 E/P/L 相关累加器、FLL/PLL/DLL 鉴相器或环路滤波器。
- 不定点化 Code NCO。
- 不修改 DFE 的 DDC NCO、FIR 系数或 FIR 累加器字长。
- 不模拟异步 NCO 时钟与 4.092 Msps 样点之间的重采样、保持效应或 DDS 杂散。
- 不实现 L5 信号、L5 码或 L5 端到端链路；33 MHz 仅作为未来可配置时钟使用。
- 不把任何仿真差异直接解释为实测跨频偏差的根因。

## 4. 配置接口

`TrackingConfig` 增加三个扁平字段：

```python
carrier_nco_mode: str = "float"
carrier_nco_clock_hz: float = 6.6e6
carrier_nco_word_bits: int = 30
```

选择扁平字段是为了保持现有研究 JSON 和 dataclass 解析方式简单。默认模式为 `float`，因此已有配置不需要修改。

验证规则：

- `carrier_nco_mode` 只能为 `"float"` 或 `"fixed"`。
- `carrier_nco_clock_hz` 必须是有限正实数，不能是 bool、复数或数组。
- `carrier_nco_word_bits` 必须是 2–62 的非 bool 整数；当前验收重点为 30 和 43。
- 固定模式要求 `carrier_nco_clock_hz * coherent_interval_s` 在数值容差内为整数。
- 频率命令超出有符号 FCW 可表示区间时抛出 `ValueError`，不静默饱和或回绕。

## 5. Carrier NCO 数值模型

新增 `antenna_diversity/receiver/carrier_nco.py`，封装可独立测试的 NCO 量化与状态更新。

设：

- NCO 时钟为 `f_clk`
- 字长为 `N`
- 相位模数为 `M = 2^N`
- 浮点环路频率命令为 `f_cmd`

频率控制字为：

```text
FCW = round_away_from_zero(f_cmd / f_clk * M)
```

使用项目现有的 MATLAB 对齐舍入规则。FCW 为 N 位有符号二进制补码整数，其有效范围为：

```text
-2^(N-1) <= FCW <= 2^(N-1)-1
```

量化后的实际频率为：

```text
f_actual = FCW * f_clk / M
```

频率 LSB 为：

```text
delta_f = f_clk / M
```

频率量化误差定义为：

```text
e_f = f_actual - f_cmd
```

对于 L1，距离速度分辨率为：

```text
delta_v_mm_s = 1000 * c / f_L1 * delta_f
```

相位累加器为 N 位无符号码字。一个相干积分历元包含：

```text
n_ticks = f_clk * coherent_interval_s
```

个整数 NCO 时钟。PLL 比例项 `kp * phase_error_rad` 转换成相位码字后，相位更新为：

```text
phase_correction_word = round_away_from_zero(
    kp * phase_error_rad / (2*pi) * M
)

phase_word = (
    phase_word
    + FCW * n_ticks
    + phase_correction_word
) mod M
```

固定模式的历元内载波副本使用历元起始相位码字和量化后的实际频率：

```text
phase[n] = 2*pi*phase_word/M + 2*pi*f_actual*n/fs
```

这一等效模型准确隔离 FCW 和历元边界相位量化，不声称复现异步硬件 DDS 的所有频谱细节。

## 6. TRK 数据流

浮点和定点模式共用同一套：

- C/A 码采样
- Early/Prompt/Late 相关
- DLL 归一化 Early-minus-Late 鉴相器
- Prompt `atan2` PLL 鉴相器
- 相邻 Prompt FLL 鉴相器
- FLL-assisted PLL 和 DLL 环路更新
- 锁定指标与 C/N0 估计

固定模式每个历元执行：

1. 使用当前量化频率和相位码字生成载波副本。
2. 计算 E/P/L、DLL、PLL 和 FLL 误差。
3. 以浮点执行现有环路滤波，得到新的 `f_cmd`。
4. 将 `f_cmd` 量化成下一历元 FCW 和 `f_actual`。
5. 将 PLL 比例相位修正量化，并更新相位码字。
6. 记录历元后的命令、FCW、实际频率、量化误差和相位码字。

浮点模式继续使用现有频率和相位更新公式，不经过整数 NCO。默认配置的数值结果必须保持现有回归容差。

## 7. 跟踪输出

所有成功跟踪结果增加：

- `carrier_nco_mode`
- `carrier_frequency_command_hz`：环路滤波器输出的量化前频率命令
- `carrier_frequency_quantization_error_hz`：`f_actual - f_cmd`；浮点模式为零
- `carrier_nco_frequency_lsb_hz`

保留现有：

- `carrier_frequency_hz`

其含义为实际用于 NCO 的频率；浮点模式等于命令频率，固定模式等于 FCW 反算频率。

固定模式额外输出：

- `carrier_fcw`：每历元更新后的有符号整数 FCW
- `carrier_phase_word`：每历元更新后的无符号相位累加器码字

数组长度均与跟踪历元数一致。获取失败时仍返回现有 `acquisition_failed` 结构，不伪造 NCO 历史。

## 8. 对比工具

新增：

```text
python_sim/run_tracking_nco_comparison.py
```

默认比较：

- 浮点 NCO
- 30-bit、6.6 MHz NCO
- 43-bit、6.6 MHz NCO

三个运行使用相同仿真配置和随机种子。当前仿真是确定性的，因此 RF、DFE 和捕获输入保持一致；测试会显式验证这一前提。

对 SINGLE、EGC、MVDR 分别汇总：

- 字长、时钟、频率 LSB 和 L1 距离速度 LSB
- 最终 FCW
- FCW 不同码字数量与跳变次数
- 最终频率命令和实际频率
- 平均、RMS、最大绝对频率量化误差
- 最终 Doppler 误差
- PLL RMS、DLL RMS
- Prompt 平均幅度
- 最终 C/N0

工具把结果写入用户指定或默认结果目录：

- `tracking_nco_comparison.csv`
- `tracking_nco_frequency.png`
- `tracking_nco_quantization_error.png`

结果目录不纳入 Git。

## 9. 错误处理

以下情况以稳定 `ValueError` 失败：

- 非法模式、时钟或字长
- 非整数历元 NCO 时钟数
- FCW 溢出
- 非有限频率命令或相位修正
- 定点状态产生非有限等效频率

禁止静默裁剪 FCW，因为裁剪会把环路失稳伪装成可用的饱和输出。

## 10. 测试策略

### NCO 单元测试

- 30-bit、6.6 MHz LSB 等于 `6.6e6 / 2^30`。
- 43-bit LSB 比 30-bit 小 `8192` 倍。
- 正负频率使用对称的半值远离零舍入。
- FCW 反算频率和量化误差公式正确。
- 相位字在正向、负向 FCW 和 PLL 相位修正下正确回绕。
- 超范围、非有限输入、非法字长和非整数 ticks 被拒绝。

### TRK 回归测试

- 默认浮点模式继续通过现有收敛测试。
- 浮点模式频率命令等于实际频率，量化误差为零。
- 30-bit 与 43-bit 固定模式均能完成当前确定性 L1 跟踪。
- FCW 和相位字是合法整数且长度与历元一致。
- `carrier_frequency_hz` 严格等于 FCW 反算值。
- 43-bit 最大量化误差不大于 30-bit 对应上界。
- 获取失败路径不进入 NCO 循环。

### 端到端和 CLI 测试

- SINGLE、EGC、MVDR 在浮点、30-bit 和 43-bit 模式下均保留原分支结构。
- 对比工具产生固定 CSV 列和两张非空 PNG。
- 三种模式使用相同仿真种子与输入配置。
- 输出不含非有限 JSON/CSV 标量；不可用 C/N0 按现有规则处理。
- 不增加 warning filter。

## 11. 验收标准

1. 默认端到端、legacy ULA、参数化天线研究和全测试套件保持通过。
2. 30-bit 与 43-bit 模式可通过配置单独运行。
3. 每个成功固定模式分支都能导出整数 FCW 和相位字历史。
4. 对比脚本能生成可复查的 CSV 和图，不修改仿真源配置。
5. README 和 DESIGN 明确写出定点边界及未建模部分。
6. 结果只能用于隔离 NCO 量化影响，不作未经数据支持的实测根因结论。
