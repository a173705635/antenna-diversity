# Codex Project Memory — GNSS 多天线 / L1-L5 跨频载波稳定性 / 分集与抗干扰

> 用途：将本窗口中的关键上下文、实验结论、代码约定、公式、架构思路和后续任务沉淀为 Codex 可直接读取的项目记忆。
> 建议放置位置：项目根目录 `CODEX_MEMORY_GNSS_ANTENNA_DIVERSITY.md`，或将其中“Codex 工作约束”部分摘入 `AGENTS.md`。

# 1. 项目背景

当前项目是手机 GNSS 2~4 天线/天线分集可行性探索。主要方向包括：

- 天线选择 / 抗握持遮挡
- 弱信号分集增益，提高跟踪稳定性
- 多径 / NLOS 检测与选路
- 双路测量一致性与异常检测
- 干扰检测
- 后续空间抗干扰（Blind Nulling / MVDR / LCMV）
- 更远期 spoofing / common-source anomaly 检测

前期仍以仿真和离线数据分析为主。

# 2. 当前 L1/L5 实验背景

GPS：

- L1 C/A：`type = 0`
- L5：`type = 9`

频率：

- `f_L1 = 1575.42 MHz`
- `f_L5 = 1176.45 MHz`

波长：

- `lambda_L1 ≈ 0.190293672798 m`
- `lambda_L5 ≈ 0.254828048791 m`

使用 GSS7000 / SimGEN 类模拟器。暗室里手机和发射天线物理上可以固定，但模拟卫星仍在运动，因此 pseudorange、carrier phase、Doppler 都会随时间变化。

# 3. 输入数据格式

默认输入文件：

```text
rover_raw_info.csv
```

主要字段：

```text
week, seconds, type, svid, cn0, pr, cp, lli, dpl, dpli, azi, ele, clkbias, clkdrift, ion, trop
```

含义：

- `cp`：载波周 / carrier phase cycles
- `dpl`：瞬时 Doppler
- `dpli`：积分 Doppler
- `cn0`：C/N0
- `ion`：电离层字段
- `trop`：对流层字段

当前代码假设：

```python
DPLI_MODE = "per_epoch"
DPLI_DISTANCE_SIGN = -1.0
```

即：

`dpli[k]` 表示该历元对应的一秒积分 Doppler，单位 cycle，距离变化按 `-lambda * DPLI` 计算。

# 4. L1/L5 对齐与清洗

必须按：

```text
week + seconds + svid
```

严格做 L1/L5 inner join，不能按行号对齐。

规则：

- `cp = 0` 删除
- 只有 L1 或只有 L5 的历元删除
- 不随意删除大跳点
- 1-s 差分只允许 `dt ≈ 1 s`

## 连续段筛选

出现过：

```text
elapsed_s = 0 ... 9
然后直接跳到
elapsed_s = 49 ...
```

中间有 40 s 断档。

当前增加：

```python
KEEP_LAST_CONTINUOUS_SEGMENT = True
CONTINUOUS_GAP_THRESHOLD_S = 1.5
```

逻辑：

1. merge 后按 `gps_time_s` 排序
2. 相邻 `dt > 1.5 s` 视为断段
3. 开关打开时，只保留最后一段连续数据
4. 重新计算 `elapsed_s`
5. 不跨 gap 做差分

# 5. 当前推荐命名

不要使用：

- `Relative`
- `Derived`
- 过长的 `Accumulated ... Difference`

03~06 推荐：

```text
03  L1/L5 1-s Carrier Distance Change
04  L5-L1 1-s Carrier Distance Change
05  L1/L5 Carrier Distance
06  L5-L1 Carrier Distance Difference
```

中文：

- 03：L1/L5 每秒载波距离变化
- 04：L5-L1 每秒载波距离变化
- 05：L1/L5 载波距离
- 06：L5-L1 载波距离差

# 6. 核心公式

载波周转距离：

`R_i[k] = CP_i[k] * lambda_i`

每秒载波周变化：

`DeltaCP_i[k] = CP_i[k] - CP_i[k-1]`

每秒载波距离变化：

`DeltaR_i[k] = lambda_i * DeltaCP_i[k]`

图 04：

`e[k] = DeltaR_L5[k] - DeltaR_L1[k]`

单位 mm/s。

图 06：

`D[k] = R_L5[k] - R_L1[k]`

单位 mm。

严格关系：

`D[k] - D[k-1] = e[k]`

即：

- 04 = 06 的一阶差分
- 06 = 04 的累计

如果 `Mean(04) ≈ 2.8 mm/s`，则 600 s 后 06 累计约 1680 mm。

# 7. 数学一致性检查

程序验证：

```text
carrier_distance_change_1s_diff_mm
==
diff(carrier_distance_diff_mm)
```

实测：

```text
Mean ~ 1e-9 mm
RMS  ~ 2e-6 mm
Max  ~ 6e-6 mm
```

结论：

**CP 后处理数学链路是自洽的。**

# 8. DPLI 与 CP

CP 的每秒变化本质上是频率积分，所以应使用积分 Doppler 对比，而不是直接用某一时刻的瞬时 Doppler。

DPLI 距离变化：

`DeltaR_DPLI,i[k] = -lambda_i * DPLI_i[k]`

跨频：

`e_DPLI[k] = DeltaR_DPLI,L5[k] - DeltaR_DPLI,L1[k]`

CP：

`e_CP[k] = DeltaR_CP,L5[k] - DeltaR_CP,L1[k]`

残差：

`e_res[k] = e_CP[k] - e_DPLI[k]`

# 9. 当前最重要的实测结论

常规信号强度下，多颗星：

```text
G01  CP Mean 2.853559 mm/s, DPLI Mean 2.848643 mm/s
G03  CP Mean 2.505271 mm/s, DPLI Mean 2.484532 mm/s
G08  CP Mean 2.678536 mm/s, DPLI Mean 2.684295 mm/s
G11  CP Mean 2.779180 mm/s, DPLI Mean 2.779115 mm/s
G28  CP Mean 3.007758 mm/s, DPLI Mean 3.006239 mm/s
G32  CP Mean 3.027588 mm/s, DPLI Mean 3.021408 mm/s
```

总体：

- `Mean(e_CP) ≈ 2.8 mm/s`
- `Mean(e_DPLI) ≈ 2.8 mm/s`
- `Mean(CP-DPLI residual) ≈ 0`

结论：

**CP 中长期跨频偏差在积分 Doppler / Tracking 频率层就已经存在，不是后处理产生。**

# 10. -140 dBm 结果

有效双频星主要：

```text
G01, G08, G11, G28, G32
```

CP Mean：

```text
G01 2.826414
G08 2.553016
G11 2.704459
G28 2.920809
G32 2.946491 mm/s
```

平均约 2.79 mm/s。

DPLI Mean：

```text
G01 2.844332
G08 2.592218
G11 2.710417
G28 2.899147
G32 2.955989 mm/s
```

平均约 2.80 mm/s。

强信号时：

- CP STD ≈ 8.3 mm/s
- DPLI STD ≈ 2.5 mm/s

-140 dBm：

- CP STD ≈ 11.0 mm/s
- DPLI STD ≈ 7.6 mm/s

Residual Mean 仍接近 0，Residual STD 约 7.8 mm/s。

结论：

**弱信号主要增加随机跟踪抖动，但没有解释约 2.8 mm/s 的长期平均偏差。**

# 11. R²

线性拟合：

`y(t) = a*t + b`

当前很多 slope 的 `R² ≈ 0`。

因此不要把这些小 slope 解释成真实慢漂。

当前更合理模型：

`e(t) = B + n(t)`

其中：

`B ≈ 2.8 mm/s`

是稳定平均偏置，`n(t)` 是随机噪声。

# 12. NCO 位宽与量化

当前 Carrier NCO：

```text
30 bit
```

L1 NCO 时钟：

```text
6.6 MHz
```

L5 NCO 时钟：

```text
33 MHz
```

频率步进：

`Deltaf = f_clk / 2^N`

L1 30 bit：

- `Deltaf ≈ 0.006147 Hz`
- 距离速度 ≈ `1.17 mm/s/LSB`

L5 30 bit：

- `Deltaf ≈ 0.030734 Hz`
- 距离速度 ≈ `7.83 mm/s/LSB`

若提升到 43 bit，精度提升 8192 倍。

L1 43 bit：

- 距离速度分辨率 ≈ `0.000143 mm/s`

L5 43 bit：

- 距离速度分辨率 ≈ `0.000956 mm/s`

结论：

**30 bit NCO 特别是 L5 的量化精度值得怀疑，但当前还不能直接证明 2.8 mm/s 就是 NCO 量化导致。**

最佳验证：

- 抓最终 Carrier FCW
- 看 FCW 整数序列和码字跳变
- 计算长期平均
- 离线比较 30 bit vs 43 bit

# 13. 电离层跨频效应

一阶电离层项：

`I_i(t) = 40.3 * TEC(t) / f_i^2`

载波相位里近似：

`R_i(t) = rho(t) - I_i(t) + ...`

因此：

`e_ion = dI_L1/dt - dI_L5/dt`

即使 Tracking 完全正确，L1/L5 也可能因电离层色散产生非零距离速度差。

约：

- L1：0.162 m/TECU
- L5：0.291 m/TECU
- 差：0.129 m/TECU = 129 mm/TECU

若 2.8 mm/s 全由 ion 解释，需要约：

`dTEC/dt ≈ 0.0217 TECU/s ≈ 1.3 TECU/min`

模拟器中不能凭经验排除，因为卫星、穿刺点、elevation 和模拟斜向 ion delay 都可能变化。

当前 CSV 有 `ion` 列，下一步优先计算：

`e_ion_mm_s = 1000 * (DeltaI_L1 - DeltaI_L5)`

并与 CP、DPLI 直接比较。

# 14. 双 L1 / 同频双天线

理想情况下，同一颗卫星、同一个 L1、两根仅相隔几厘米的天线，真实物理 Doppler 应几乎相同：

`f_D1_true ≈ f_D2_true`

但实际 TRK 输出可能：

`fhat_D1 = f_D + b1 + n1`

`fhat_D2 = f_D + b2 + n2`

因此：

`Delta fhat_D = (b2-b1) + (n2-n1)`

长期平均可能不为 0。

若长期存在 differential frequency：

`Delta f != 0`

则：

`Delta phi(t) = Delta phi_0 + 2*pi*Delta f*t`

所以双 L1 也可能出现载波距离差持续漂移。

例如：

`Delta f = 0.01 Hz`

L1 对应约：

`1.9 mm/s`

10 分钟可累计约 1.14 m。

# 15. 相位差对分集的影响

固定相位差：

- Selection 基本不受影响
- EGC/MRC/Beamforming 可校准

动态相位漂移：

`Delta phi(t) = Delta phi_0 + 2*pi*Delta f*t`

一次静态校准不够。

若差频约 0.0147 Hz，相位约每秒转 5.3°，约 68 s 转一圈。

对 EGC/MRC/Nulling/MVDR 都有影响。

# 16. 分集 vs 抗干扰架构

## 分集增益

如果两路都能 Tracking：

```text
RF1 -> TRK1
RF2 -> TRK2
```

之后可做：

- selection
- measurement-level combining
- coherent combining（需相位跟踪）

适用于：

- 弱信号
- 抗遮挡
- 提升跟踪稳定性
- 减少失锁

## 强抗干扰

若两路单独都已无法 ACQ/TRK，则不能靠“先 Tracking 再合并”。

正确方向：

```text
多天线 IQ
   ↓
空间处理 / 干扰抑制
   ↓
单路输出
   ↓
ACQ / TRK
```

即：

`RF/ADC -> DFE -> Multi-Antenna Spatial Processing -> ACQ/TRK`

# 17. 多天线干扰检测模型

接收模型：

`x[n] = a_s*s[n] + sum(a_jk*j_k[n]) + n[n]`

核心：

`R_x = E[x*x^H]`

有限窗口：

`Rhat_x = (1/N) * sum(x[n]*x[n]^H)`

两天线只需：

- `R11 = E[|x1|^2]`
- `R22 = E[|x2|^2]`
- `R12 = E[x1*x2*]`

因为 `R21 = R12*`。

硬件只需累计：

```text
R11
R22
Re(R12)
Im(R12)
```

窗口可用 `N = 2^K`。

# 18. 干扰检测指标

总功率：

`P = R11 + R22`

空间相关系数：

`rho = |R12| / sqrt(R11*R22)`

特征值：

`lambda_1,2 = (R11+R22 ± sqrt((R11-R22)^2 + 4|R12|^2))/2`

特征值比：

`T_lambda = lambda_max / lambda_min`

含义：

- 白噪声：两个特征值接近
- 强单主导空间干扰：最大特征值明显变大

频域还可做 FFT/PSD，识别：

- CW
- 窄带
- 多音
- 扫频
- 宽带噪声

# 19. 硬件模块位置

推荐：

```text
Antenna
  ↓
RF
  ↓
ADC
  ↓
DFE
  ↓
Channel Alignment / Calibration
  ↓
Multi-Antenna Detection / Spatial Processing
  ↓
ACQ / TRK
```

第一版建议做旁路 detector：

```text
DFE IQ ─────→ 原 ACQ/TRK
   │
   └────────→ Interference Detector
```

先不改主链路。

DFE 后处理的原因：

- 已完成 DDC / FIR / Decimation
- 数据率降低
- 协方差计算硬件代价更小

# 20. 多 RF 相位同步

至少三层：

## 采样同步

最好多 ADC 共 sampling clock。

否则时间偏差 `Delta t` 会形成：

`Delta phi = 2*pi*f*Delta t`

宽带下还会变成群时延差。

## 频率同步

最好：

- 共 reference clock
- 共 LO 或相干 LO

否则相对相位持续旋转。

## 固定幅相校准

LNA、Mixer、Filter、PCB、ADC、DDC、FIR 等会带来：

`H2/H1 = G * exp(j*phi)`

固定项可校准。

# 21. 没 Tracking 时如何做相位相关处理

关键结论：

**没跟踪上，不妨碍做 Blind Interference Detection 和 Blind Nulling。**

若强干扰：

`x[n] = h_j*j[n] + n[n]`

其中 `h_j` 已包含：

- 天线空间相位
- RF 固定相位
- 固定增益差

从实际 IQ 估计协方差矩阵，最大特征向量可近似得到实际干扰空间响应。

因此 Blind Nulling 不要求先知道 RF 固定 phase bias。

但要求：

**估计窗口内 relative phase 要稳定。**

# 22. 固定 phase bias vs 动态 phase drift

固定相位差只改变 `R12` 相角，不影响 `|R12|`，所以基于：

`rho = |R12| / sqrt(R11*R22)`

的 detector 对固定 phase bias 不敏感。

真正麻烦的是 differential frequency：

`Delta f != 0`

则：

`Delta phi(t) = Delta phi_0 + 2*pi*Delta f*t`

长窗口互相关会发生平均抵消，近似 sinc 衰减：

`|R12| ∝ |sin(pi*Delta f*T)/(pi*Delta f*T)|`

所以真正关键的是：

- differential frequency
- relative phase stability

# 23. Blind Nulling vs MVDR

Blind Nulling：

- 直接从强干扰估计实际空间向量
- 不要求 GNSS 已经跟踪
- 固定 RF phase bias 会包含在估计向量里

标准 MVDR：

`min w^H R w`

约束：

`w^H a_s = 1`

需要知道目标 GNSS 的实际 steering vector。

若只使用理论几何 steering vector而没有包含 RF 幅相误差，可能压制目标。

因此标准 MVDR / LCMV 对通道校准要求更高。

# 24. 独立于 GNSS Tracking 的校准

可考虑内部校准信号：

同时向多 RF 注入已知 `s_cal(t)`。

若：

`Y1 = H1*S_cal`
`Y2 = H2*S_cal`

则：

`Y2/Y1 = H2/H1 = G*exp(j*phi)`

不依赖 ACQ/TRK。

也可做出厂/暗室校准，但需验证：

- 温度
- AGC gain state
- 频点
- 重启
- 长时间 phase drift

# 25. Differential Doppler 下的更新

若两路存在：

`Delta f = f2 - f1 != 0`

则一次静态校准不够。

需要：

- 实时 differential frequency / phase tracking
- 或周期性重新估计 covariance / interference vector / weights

若允许最大 phase error 为 `phi_max`：

`T_update < phi_max / (2*pi*|Delta f|)`

例如允许 10°：

- Deltaf = 0.01 Hz → T_update < 2.78 s
- Deltaf = 0.1 Hz → T_update < 0.278 s

# 26. 当前第一版硬件建议

不建议一上来做完整 MVDR。

第一阶段：

```text
DFE
 ↓
Covariance Engine
 ↓
Power / rho / Eigenvalue Ratio
 ↓
Jam Flag
```

验证：

- 无干扰
- CW
- 窄带
- 宽带噪声
- 不同 J/S
- 不同入射方向
- 不同天线间距
- 2 天线，后续扩 3/4 天线

先回答：

> 相比单天线能量检测，多天线空间相关性是否能更可靠地检测干扰？

若成立，再扩展：

```text
Jam Detection
 ↓
Interference Subspace
 ↓
Blind Nulling / MVDR
 ↓
ACQ/TRK
```

# 27. 当前代码输出建议

明细 CSV 保持简洁，建议核心字段：

```text
week
seconds
elapsed_s
svid
dt_s

cp_l1_cycle
cp_l5_cycle
delta_cp_l1_cycle
delta_cp_l5_cycle

carrier_distance_change_1s_l1_m
carrier_distance_change_1s_l5_m
carrier_distance_change_1s_diff_mm

carrier_distance_l1_m
carrier_distance_l5_m
carrier_distance_diff_mm

dpli_l1_cycle
dpli_l5_cycle
dpli_distance_change_1s_diff_mm

cp_dpli_residual_1s_mm

doppler_l1_hz
doppler_l5_hz
doppler_diff_hz

cn0_l1
cn0_l5
azi
ele
```

一致性检查量建议内部计算或 summary，不要每行重复占列。

# 28. 当前脚本功能要求

最终脚本应保留：

1. 默认读取 `rover_raw_info.csv`
2. L1/L5 按 week + seconds + svid 严格对齐
3. `cp=0` 删除
4. 单频缺失历元删除
5. 连续段筛选开关
6. 只对 `dt≈1s` 做 1-s 差分
7. 03/04/05/06 图
8. DPLI 图
9. CP vs DPLI residual
10. 瞬时 Doppler
11. C/N0
12. 数学一致性检查
13. TXT 日志
14. 单星 CSV
15. all satellites CSV
16. satellite summary

# 29. 下一步优先级

## P0：电离层贡献

利用 `ion` 列计算：

`e_ion = 1000*(DeltaI_L1 - DeltaI_L5)`

与：

- CP L5-L1 1-s Carrier Distance Change
- DPLI L5-L1 1-s Distance Change

比较 Mean / STD / correlation。

## P0：双 L1 同频双通道基线

测试：

```text
同一 L1
-> 两根天线 / 两条 RF
-> 同频 Tracking
```

重点观察：

- differential Doppler mean
- differential carrier phase
- phase drift
- carrier distance difference
- relative phase stability

目标不是假设均值一定为 0，而是测出系统本身 differential bias baseline。

## P0：多 RF 相干性

重点测：

`Delta phi_12(t)`

以及：

`d(Delta phi_12)/dt`

验证：

- 固定 phase offset
- differential frequency offset
- 温度
- AGC 切档
- 重启
- 长时间稳定性

## P1：30 bit vs 43 bit NCO

抓最终 Carrier FCW：

- L1 FCW
- L5 FCW
- 量化码字跳变
- 长期平均
- 30/43 bit 离线仿真

## P1：干扰检测最小可行算法

基于 DFE IQ：

`R11, R22, R12`

计算：

- Power
- |R12|
- rho
- lambda_max/lambda_min

先做 detector。

## P2：Blind Nulling

若 detector 有效：

```text
Covariance
-> Dominant Interference Eigenvector
-> Null Weight
-> y = w^H x
-> ACQ/TRK
```

重点验证单天线已经无法 ACQ/TRK 时，多天线能否先压干扰再恢复跟踪。

# 30. 当前阶段核心结论

1. L1/L5 CP 换算和 04/06 数学链路没有错误。
2. 04 与 06 严格是差分 vs 累计关系。
3. 多星稳定存在约 2.5~3.0 mm/s 的 L5-L1 每秒载波距离变化均值。
4. DPLI 几乎完整复现该均值。
5. 因此长期跨频偏差已经存在于 Tracking / Doppler 上游。
6. -140 dBm 主要放大随机抖动，但平均 bias 基本不变。
7. 当前 slope 的 R²≈0，不支持“偏差随时间线性增长”的解释。
8. 更合理模型是 `e(t)=B+n(t)`，`B≈2.8 mm/s`。
9. 30 bit NCO 特别是 L5 的 7.83 mm/s/LSB 值得怀疑，但未证实。
10. 电离层色散是必须优先排查的跨频物理来源。
11. 双 L1 实验可以建立接收机本身的 differential phase/frequency baseline。
12. 动态 phase drift 会影响 coherent combining。
13. 强抗干扰不能依赖两路先 Tracking 再合并。
14. 更合适架构是 `DFE -> Spatial Processing -> ACQ/TRK`。
15. 第一版多天线抗干扰建议先做旁路 covariance-based interference detector。
16. Blind Nulling 不要求 GNSS 已经跟踪，但要求多 RF 同步且短时间 relative phase 稳定。
17. 标准 MVDR 若要保护目标 steering vector，需要更严格的幅相校准机制。

# 31. Codex 工作约束

Codex 继续项目时：

1. 不要把现有项目改造成纯 Tracking 学习项目。
2. Tracking / NCO / carrier phase 分析只是服务于手机多天线可行性的子任务。
3. 优先复用现有 Python 分析脚本，不要无必要重写。
4. L1/L5 必须按时间和 SVID 严格对齐。
5. 任何跨 gap 的 `.diff()` 必须禁止。
6. 输出命名要短、物理意义直接。
7. 不要重新引入 `Relative Carrier Distance`、`CP-Derived...` 等绕的命名。
8. 04/06 必须保持严格数学一致性。
9. 结论必须区分 fixed bias / drift / random noise。
10. 不要仅凭 NCO LSB 大小就断言根因是 NCO 量化。
11. ionosphere、NCO、aiding、RF/TRK differential bias 应分项验证。
12. 多天线抗干扰设计必须优先考虑共时钟、同步采样、relative phase stability、channel alignment。
13. 第一阶段硬件应轻量、旁路、可验证，避免直接上复杂 MVDR。
14. 两天线优先利用 2x2 covariance 的解析结构，不必实现通用 EVD。
15. 扩到 3~4 天线后再考虑通用矩阵/子空间模块复用。

# 32. 建议 Codex 接手后的第一个任务

建议第一步直接在当前分析脚本中加入 `ion` 分析链：

```text
ion_l1
ion_l5
delta_ion_l1
delta_ion_l5
ion_predicted_l5_l1_1s_mm
```

核心公式：

`ion_predicted_l5_l1_1s_mm = 1000*(DeltaI_L1 - DeltaI_L5)`

新增一张对比图：

```text
CP L5-L1 1-s Carrier Distance Change
DPLI L5-L1 1-s Distance Change
Ionosphere-predicted L5-L1 1-s Change
```

并统计三者 Mean / STD / correlation。

这一步最有可能快速判断当前约 2.8 mm/s bias 的物理来源。
