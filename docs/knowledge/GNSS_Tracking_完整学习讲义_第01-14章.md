# GNSS Tracking 系统学习完整讲义

> 从 Tracking 基础数据流、相关器、DLL/PLL/FLL、Loop Filter、NCO、Carrier Phase/ADR，
> 到弱信号、AltBOC、Vector Tracking / Deep Coupling 的完整学习笔记。

本讲义由 14 个章节整合而成，适合作为后续 TRK 实战、芯片模块理解和多天线 GNSS 架构研究的理论参考。

## 目录

1. [第 1 章｜Tracking 模块整体数据流](#chapter-01)
2. [第 2 章｜Tracking 信号模型与 I/Q 解调](#chapter-02)
3. [第 3 章｜相关器、自相关函数与 Early-Prompt-Late](#chapter-03)
4. [第 4 章｜DLL 鉴别器、S 曲线与码跟踪环](#chapter-04)
5. [第 5 章｜PLL 载波相位鉴别器与 Costas Loop](#chapter-05)
6. [第 6 章｜FLL 频率鉴别器与 FLL-Assisted PLL](#chapter-06)
7. [第 7 章｜Loop Filter：环路带宽、自然频率与数字实现](#chapter-07)
8. [第 8 章｜Carrier NCO、Phase Accumulator、FCW 与连续载波相位](#chapter-08)
9. [第 9 章｜Carrier Phase、ADR、Integrated Doppler 与 Measurement](#chapter-09)
10. [第 10 章｜Tracking Noise、Jitter、Dynamic Stress 与 Lock Threshold](#chapter-10)
11. [第 11 章｜实际 TRK Channel 数据流：从 Sample 到 Measurement](#chapter-11)
12. [第 12 章｜弱信号与长积分 Tracking](#chapter-12)
13. [第 13 章｜现代 GNSS Tracking：BOC / AltBOC / Pilot / Secondary Code / 多峰](#chapter-13)
14. [第 14 章｜Vector Tracking / Deep Coupling](#chapter-14)


---

<a id="chapter-01"></a>

## 第 1 章｜GNSS Tracking 模块整体数据流

### 1. 本章目标

理解 Acquisition 将卫星交给 Tracking 以后，Tracking Channel 内部的完整处理流程，并建立 DLL、PLL、FLL、Correlator、Loop Filter 和 NCO 之间的整体关系。

---

### 2. Tracking 的作用

Acquisition 通常只能获得：

- 粗码相位
- 粗 Doppler
- 卫星 PRN
- 信号类型

例如：

```text
PRN = 3
粗码相位 = 152.3 chip
粗 Doppler = -2500 Hz
```

真实参数可能是：

$$
\tau=152.36\text{ chip}
$$

$$
f_D=-2473.6\text{ Hz}
$$

因此仍然存在：

$$
\Delta\tau=\tau-\hat\tau
$$

$$
\Delta f=f_D-\hat f_D
$$

同时由于卫星和接收机持续运动，这些参数不断变化，因此必须由 Tracking 持续跟踪。

Tracking 的本质是：

> 不断测量真实信号和本地复制信号之间的误差，并不断修正本地信号。

即一个闭环反馈系统：

```text
测量误差
   ↓
修正本地信号
   ↓
再次测量
   ↓
再次修正
   ↓
……
```

---

### 3. Tracking 信号模型

GNSS 信号可以简化为：

$$
r(t)
=
A c(t-\tau)d(t)
\cos(2\pi(f_{IF}+f_D)t+\phi)
+n(t)
$$

其中：

- $A$：信号幅度
- $c(t)$：PRN 码
- $d(t)$：导航数据
- $\tau$：码延迟
- $f_D$：Doppler
- $\phi$：载波相位

Tracking 内部需要产生两个本地副本。

#### 本地载波

$$
e^{-j(2\pi\hat f t+\hat\phi)}
$$

#### 本地码

$$
c(t-\hat\tau)
$$

Tracking 的目标为：

$$
\Delta f\rightarrow0
$$

$$
\Delta\phi\rightarrow0
$$

$$
\Delta\tau\rightarrow0
$$

---

### 4. Tracking Channel 整体结构

```text
                     ┌──────── Carrier NCO ◄─────────────┐
                     │                                    │
                     ▼                                    │
输入采样 ───────► Carrier Wipeoff                         │
                     │                                    │
                     ▼                                    │
                Code Correlator                           │
                     │                                    │
          ┌──────────┼──────────┐                         │
          ▼          ▼          ▼                         │
        Early      Prompt      Late                       │
         E           P           L                        │
          │           │           │                        │
          └───────────┴───────────┘                        │
                     │                                    │
          ┌──────────┴─────────────┐                      │
          ▼                        ▼                      │
     Code Discriminator     Carrier Discriminator         │
          │                        │                      │
          ▼                        ▼                      │
     DLL Loop Filter        PLL/FLL Loop Filter           │
          │                        │                      │
          ▼                        │                      │
      Code NCO                      └──────────────────────┘
          │
          ▼
       PRN Generator
```

---

### 5. Carrier Wipeoff

输入载波：

$$
\cos(2\pi f_rt+\phi)
$$

本地载波：

$$
\cos(2\pi f_lt+\hat\phi)
$$

两者相乘：

$$
\cos A\cos B
=
\frac12[\cos(A-B)+\cos(A+B)]
$$

低通或积分后主要保留：

$$
f_r-f_l
$$

即 residual carrier frequency：

$$
\Delta f=f_r-f_l
$$

例如：

$$
f_D=-2473.6\text{ Hz}
$$

而本地估计：

$$
\hat f_D=-2500\text{ Hz}
$$

则残余频率为：

$$
\Delta f=26.4\text{ Hz}
$$

Tracking 后续继续估计并消除该残余频率。

---

### 6. I/Q 信号

接收机通常同时产生正交本地载波：

$$
\cos(\hat\phi)
$$

和：

$$
-\sin(\hat\phi)
$$

理想情况下：

$$
I=A\cos(\Delta\phi)
$$

$$
Q=A\sin(\Delta\phi)
$$

其中：

$$
\Delta\phi=\phi-\hat\phi
$$

如果：

$$
\Delta\phi=0
$$

则：

$$
I=A
$$

$$
Q=0
$$

因此 I/Q 可以用于估计载波相位误差，是 PLL 工作的基础。

---

### 7. Code Correlation

Carrier wipeoff 后，还需要将输入信号与本地 PRN 码进行相关。

如果：

$$
\hat\tau=\tau
$$

则相关值最大。

因此相关值可以反映本地码和真实码之间的时间偏差。

---

### 8. Early / Prompt / Late

以 Prompt 为中心：

$$
E=\hat\tau-\frac d2
$$

$$
P=\hat\tau
$$

$$
L=\hat\tau+\frac d2
$$

基本结构：

```text
------ E ------ P ------ L ------>
```

如果真实相关峰位于 Prompt 的右侧，一般：

$$
L>E
$$

如果真实相关峰位于 Prompt 左侧，一般：

$$
E>L
$$

因此 DLL 可以利用 E/L 的差异估计码相位误差。

---

### 9. Correlator 输出

基本 E/P/L Tracking 会得到六路积分结果：

$$
I_E,\quad Q_E
$$

$$
I_P,\quad Q_P
$$

$$
I_L,\quad Q_L
$$

其中：

- Prompt 主要用于 Carrier Tracking
- Early/Late 主要用于 Code Tracking

---

### 10. Discriminator

Discriminator（鉴别器）的作用是：

> 根据相关器输出估计 Tracking error。

不同环路对应不同误差：

| 环路 | 主要估计量 |
|---|---|
| DLL | 码相位误差 |
| PLL | 载波相位误差 |
| FLL | 载波频率误差 |

可以近似理解为：

$$
e_{DLL}\approx\Delta\tau
$$

$$
e_{PLL}\approx\Delta\phi
$$

$$
e_{FLL}\approx\Delta f
$$

---

### 11. PLL 鉴别器基本思想

Prompt 输出近似为：

$$
I_P=A\cos(\Delta\phi)
$$

$$
Q_P=A\sin(\Delta\phi)
$$

因此可以使用：

$$
e_{PLL}
=
\operatorname{atan2}(Q_P,I_P)
$$

估计：

$$
\Delta\phi
$$

---

### 12. DLL 鉴别器基本思想

首先计算 Early 和 Late 幅度：

$$
E=\sqrt{I_E^2+Q_E^2}
$$

$$
L=\sqrt{I_L^2+Q_L^2}
$$

典型鉴别器：

$$
e_{DLL}
=
\frac{E-L}{E+L}
$$

其输出反映 Code phase error。

---

### 13. Loop Filter

Discriminator 的输出中不仅包含真实误差，还包含：

- 热噪声
- 多径
- 干扰
- 量化误差

因此不能直接控制 NCO。

基本结构为：

```text
Discriminator
      ↓
Loop Filter
      ↓
NCO
```

Loop Filter 的核心作用是在以下两项之间进行权衡：

```text
动态跟踪能力
        ↕
噪声抑制能力
```

---

### 14. Carrier NCO

Carrier NCO 用于产生本地载波。

数字相位累加器可以表示为：

$$
\Phi[k+1]
=
\Phi[k]+\Delta\Phi
$$

其中：

$$
\Delta\Phi
=
2\pi\frac{f}{f_s}
$$

因此 NCO 通过控制相位增量产生指定频率。

实际数字芯片通常使用 FCW：

$$
FCW
=
\frac{f}{f_s}2^N
$$

---

### 15. Code NCO

Code NCO 控制本地 PRN 码：

- 当前位置
- 码速率

基本反馈链：

```text
DLL discriminator
      ↓
DLL Loop Filter
      ↓
Code FCW
      ↓
Code NCO
      ↓
PRN Generator
```

---

### 16. Tracking 的不同时间尺度

Tracking 并不是每个 ADC sample 都更新一次环路。

例如：

$$
f_s=20\text{ MHz}
$$

若：

$$
T_{coh}=1\text{ ms}
$$

则每次积分包含：

$$
20\,000
$$

个 sample。

完整过程为：

```text
20000 samples
      ↓
Correlation + Accumulation
      ↓
IE QE IP QP IL QL
      ↓
Discriminator
      ↓
Loop Filter
      ↓
更新 NCO
```

Tracking 中至少存在三个时间尺度：

#### Sample 级

Carrier NCO、Code NCO 按 sample 持续工作。

#### Integration 级

例如每 1 ms 产生一次相关器积分结果。

#### Loop Update 级

例如每 1 ms、10 ms 等更新一次 Tracking loop。

---

### 17. Tracking 完整闭环

核心流程：

```text
Acquisition
    ↓
初始化 Code NCO / Carrier NCO
    ↓
生成本地 Carrier + PRN
    ↓
与输入信号相关
    ↓
得到 IE/QE、IP/QP、IL/QL
    ↓
DLL / PLL / FLL 鉴别器
    ↓
获得 Code / Carrier error
    ↓
Loop Filter
    ↓
修改 Code FCW / Carrier FCW
    ↓
NCO 更新
    ↓
下一次积分
    ↓
重复
```

---

### 18. Lock 的含义

Tracking lock 并不意味着 NCO 参数停止变化。

由于卫星和接收机一直运动：

$$
f_D=f_D(t)
$$

$$
\tau=\tau(t)
$$

Tracking 仍需要不断调整 NCO。

因此 Lock 更准确的含义是：

> Tracking loop 已进入稳定跟随状态，误差保持在较小范围内。

---

### 19. Tracking 输出的测量信息

Tracking 不仅用于保持信号锁定，还产生大量 GNSS 测量信息。

#### Doppler

Carrier NCO 频率可以形成 Doppler measurement。

#### Carrier Phase

Carrier NCO 的 Phase Accumulator 可用于重建累计载波相位，并进一步形成 ADR。

#### Code Phase

Code NCO 提供码相位信息，可进一步用于形成 pseudorange。

#### CN0

可以根据 Prompt I/Q 等相关输出估计：

$$
C/N_0
$$

#### Navigation Data

Prompt 信号还可用于恢复：

- navigation bit
- symbol
- secondary code

---

### 20. Tracking 的两个核心闭环

#### Code Loop

```text
Code NCO
   ↓
Local Code
   ↓
E/P/L Correlation
   ↓
DLL Discriminator
   ↓
DLL Loop Filter
   ↓
Code NCO
```

#### Carrier Loop

```text
Carrier NCO
     ↓
Local Carrier
     ↓
I/Q Correlation
     ↓
PLL / FLL Discriminator
     ↓
Loop Filter
     ↓
Carrier NCO
```

因此 Tracking 的核心结构可以统一表示为：

$$
\boxed{
\text{NCO}
\rightarrow
\text{Correlation}
\rightarrow
\text{Error Estimation}
\rightarrow
\text{Loop Filter}
\rightarrow
\text{NCO}
}
$$

---

### 21. Carrier FCW 与 ADR 的基本关系

基本数据链：

```text
PLL / FLL
    ↓
Loop Filter
    ↓
Carrier FCW
    ↓
Carrier NCO
    ↓
Phase Accumulator
    ↓
累计载波相位
    ↓
Carrier Phase / ADR
```

因此 Tracking 是 Carrier Phase 和 ADR 的底层信息来源之一。

---

### 22. 本章核心变量

| 变量 | 含义 |
|---|---|
| $\hat\tau$ | 本地估计码相位 |
| $\hat f$ | 本地载波频率 |
| $\hat\phi$ | 本地载波相位 |
| $\Delta\tau$ | 码相位跟踪误差 |
| $\Delta f$ | 载波频率跟踪误差 |
| $\Delta\phi$ | 载波相位跟踪误差 |

Tracking 的目标为：

$$
\Delta\tau\rightarrow0
$$

$$
\Delta f\rightarrow0
$$

$$
\Delta\phi\rightarrow0
$$

---

### 23. 本章关键结论

1. Acquisition 只负责提供粗码相位和粗 Doppler，Tracking 负责持续精确跟踪。
2. Tracking 本质是一个闭环控制系统。
3. Tracking 内部同时存在 Code Loop 和 Carrier Loop。
4. Correlator 是 Tracking 的误差观测基础。
5. E/P/L 用于判断码位置，其中 Early/Late 主要服务于 DLL。
6. Prompt I/Q 包含载波相位信息，是 PLL/FLL 的核心输入。
7. Discriminator 将 Correlator 输出转换为 Tracking error。
8. Loop Filter 对误差进行滤波并生成 NCO 修正量。
9. NCO 根据 FCW 持续产生本地 Carrier 和 Code。
10. Tracking 同时是 Doppler、Code Phase、Carrier Phase、ADR 等 GNSS 测量量的重要底层来源。


---

<a id="chapter-02"></a>

## 第 2 章｜Tracking 信号模型与 I/Q 解调

### 1. 本章目标

本章重点建立从接收 GNSS 信号到 Tracking 相关器输出之间的完整数学链路，理解：

- 接收信号中包含哪些未知量；
- Carrier wipeoff 为什么会产生 I/Q；
- I/Q 与残余载波相位、残余 Doppler 的关系；
- PRN 解扩为什么能够把扩频信号“还原”；
- 相干积分为什么能够提升相关输出 SNR；
- 最终为什么会得到 `IE/QE/IP/QP/IL/QL`。

---

### 2. 接收 GNSS 信号模型

先采用简化的实信号模型：

$$
r(t)
=
A\,d(t)c(t-\tau)
\cos\left(2\pi(f_{IF}+f_D)t+\phi\right)
+n(t)
$$

其中：

- $A$：接收信号幅度；
- $d(t)$：导航数据；
- $c(t-\tau)$：延迟后的 PRN 码；
- $\tau$：真实码延迟；
- $f_{IF}$：中频；
- $f_D$：Doppler；
- $\phi$：载波初相位；
- $n(t)$：噪声。

Tracking 的核心目标是估计：

$$
\tau,\quad f_D,\quad \phi
$$

接收机本地产生估计量：

$$
\hat\tau,\quad \hat f,\quad \hat\phi
$$

对应误差：

$$
\Delta\tau=\tau-\hat\tau
$$

$$
\Delta f=(f_{IF}+f_D)-\hat f
$$

$$
\Delta\phi=\phi-\hat\phi
$$

---

### 3. 为什么需要 Carrier Wipeoff

原始信号仍然在中频或残余载波上振荡，不能直接进行长时间相关积分。

本地 Carrier NCO 产生正交载波：

$$
\cos(2\pi\hat f t+\hat\phi)
$$

以及：

$$
-\sin(2\pi\hat f t+\hat\phi)
$$

输入信号分别与两路本地载波相乘，形成 I/Q 两路。

---

### 4. I 路推导

I 路：

$$
r_I(t)
=
r(t)\cos(2\pi\hat f t+\hat\phi)
$$

忽略 PRN、数据和噪声，先看载波部分：

$$
\cos(2\pi f_rt+\phi)
\cos(2\pi\hat f t+\hat\phi)
$$

利用：

$$
\cos A\cos B
=
\frac12[\cos(A-B)+\cos(A+B)]
$$

得到：

$$
\frac12
\cos\left(2\pi(f_r-\hat f)t+\phi-\hat\phi\right)
+
\frac12
\cos\left(2\pi(f_r+\hat f)t+\phi+\hat\phi\right)
$$

定义：

$$
\Delta f=f_r-\hat f
$$

$$
\Delta\phi=\phi-\hat\phi
$$

则低频部分为：

$$
\frac12
\cos(2\pi\Delta f t+\Delta\phi)
$$

高频和频项会在后续积分中被强烈抑制。

因此 I 路基带信号近似为：

$$
I(t)
\propto
A\,d(t)c(t-\tau)
\cos(2\pi\Delta f t+\Delta\phi)
$$

---

### 5. Q 路推导

Q 路使用正交本地载波。

根据具体符号定义，最终通常可写为：

$$
Q(t)
\propto
A\,d(t)c(t-\tau)
\sin(2\pi\Delta f t+\Delta\phi)
$$

因此 I/Q 可组成复基带：

$$
z(t)
=
I(t)+jQ(t)
$$

近似表示为：

$$
z(t)
\propto
A\,d(t)c(t-\tau)
e^{j(2\pi\Delta f t+\Delta\phi)}
$$

这是 Tracking 中非常重要的表达。

Carrier wipeoff 后，原本的高频载波被去除，只剩：

- residual Doppler $\Delta f$；
- residual phase $\Delta\phi$；
- PRN；
- data；
- noise。

---

### 6. 完全锁定时的 I/Q

如果：

$$
\Delta f=0
$$

并且：

$$
\Delta\phi=0
$$

则：

$$
I\propto A
$$

$$
Q\approx0
$$

因此理想 PLL lock 状态下：

> 信号能量主要集中在 I 路，Q 路接近 0。

如果：

$$
\Delta\phi=30^\circ
$$

则：

$$
I=A\cos30^\circ
$$

$$
Q=A\sin30^\circ
$$

即：

$$
I\approx0.866A
$$

$$
Q=0.5A
$$

因此：

$$
\frac{Q}{I}=\tan(\Delta\phi)
$$

这正是 PLL 相位鉴别器能够工作的基础。

---

### 7. residual Doppler 的作用

如果载波频率没有完全对准：

$$
\Delta f\neq0
$$

则：

$$
I(t)\propto\cos(2\pi\Delta f t+\Delta\phi)
$$

$$
Q(t)\propto\sin(2\pi\Delta f t+\Delta\phi)
$$

也就是说 I/Q 向量会在复平面不断旋转。

定义：

$$
P(t)=I(t)+jQ(t)
$$

则：

$$
P(t)\propto e^{j(2\pi\Delta f t+\Delta\phi)}
$$

其相位随时间：

$$
\theta(t)=2\pi\Delta f t+\Delta\phi
$$

因此 residual Doppler 本质上就是：

> I/Q 复向量的旋转速度。

---

### 8. 频率误差和相位误差的关系

相位对时间求导：

$$
\frac{d\theta(t)}{dt}
=
2\pi\Delta f
$$

因此：

$$
\Delta f
=
\frac{1}{2\pi}
\frac{d\theta}{dt}
$$

离散情况下，如果两个积分结果间隔为 $T$：

$$
\Delta f
\approx
\frac{\theta_k-\theta_{k-1}}
{2\pi T}
$$

这就是 FLL 基本原理的来源。

---

### 9. PRN 解扩

Carrier wipeoff 后仍然存在：

$$
c(t-\tau)
$$

Tracking 本地产生：

$$
c(t-\hat\tau)
$$

两者相乘：

$$
c(t-\tau)c(t-\hat\tau)
$$

如果完全对齐：

$$
\hat\tau=\tau
$$

由于 PRN chip 取值通常为 $\pm1$：

$$
c^2(t)=1
$$

所以：

$$
c(t-\tau)c(t-\tau)=1
$$

扩频调制被消除。

原本快速正负变化的 GNSS 信号在解扩后变成可以稳定积累的基带信号。

---

### 10. 码没有完全对准时

如果：

$$
\Delta\tau=\tau-\hat\tau\neq0
$$

则本地 PRN 和接收 PRN 不能完全相消。

积分后的信号幅度由 PRN 自相关函数决定：

$$
R_c(\Delta\tau)
$$

因此相关器输出近似可写为：

$$
P
\propto
A\,R_c(\Delta\tau)
$$

当：

$$
\Delta\tau=0
$$

时：

$$
R_c(0)
$$

达到最大值。

这就是 DLL 可以利用相关峰定位码相位的基础。

---

### 11. Prompt 相关器

Prompt 使用当前估计码相位：

$$
\hat\tau_P=\hat\tau
$$

因此 Prompt 相关输出可简化为：

$$
P
\propto
A\,R_c(\Delta\tau)
e^{j\Delta\phi}
$$

在没有 residual Doppler、导航 bit 在积分期间不翻转的简化情况下：

$$
I_P
\propto
A\,R_c(\Delta\tau)\cos(\Delta\phi)
$$

$$
Q_P
\propto
A\,R_c(\Delta\tau)\sin(\Delta\phi)
$$

这个公式非常重要。

它说明 Prompt 输出同时包含两类误差：

- 幅度受到码相位误差影响；
- I/Q 方向受到载波相位误差影响。

---

### 12. Early / Late 相关器

设 Early-Late 总间距为 $d$。

则：

$$
\hat\tau_E=\hat\tau-\frac d2
$$

$$
\hat\tau_L=\hat\tau+\frac d2
$$

相对于真实码位置，其误差分别约为：

$$
\Delta\tau_E
=
\Delta\tau+\frac d2
$$

$$
\Delta\tau_L
=
\Delta\tau-\frac d2
$$

因此：

$$
E
\propto
R_c\left(\Delta\tau+\frac d2\right)
$$

$$
L
\propto
R_c\left(\Delta\tau-\frac d2\right)
$$

当 Prompt 对准相关峰：

$$
\Delta\tau=0
$$

则：

$$
E=L
$$

当 Prompt 偏离峰值：

$$
E\neq L
$$

DLL 就可以利用两者差异估计码相位误差。

---

### 13. 相干积分

实际 Tracking 不会只看某一个 sample，而是在长度 $T_{coh}$ 内进行累加：

$$
I_P
=
\int_0^{T_{coh}}
r(t)c_P(t)
\cos(\hat\theta(t))
dt
$$

$$
Q_P
=
\int_0^{T_{coh}}
r(t)c_P(t)
\sin(\hat\theta(t))
dt
$$

数字实现为：

$$
I_P
=
\sum_{n=0}^{N_s-1}
x[n]c_P[n]\cos\hat\theta[n]
$$

$$
Q_P
=
\sum_{n=0}^{N_s-1}
x[n]c_P[n]\sin\hat\theta[n]
$$

其中：

$$
N_s=f_sT_{coh}
$$

注意：

- $N_s$ 是 sample 数；
- PRN chip 数由 $R_cT_{coh}$ 决定；
- sample 数不等于 chip 数。

---

### 14. 相干积分的主要作用

相干积分主要有三方面作用：

1. 抑制混频后的高频和频项；
2. 让目标 GNSS 信号同相累积；
3. 对随机噪声和非相关成分进行平均。

若目标信号保持相干，积分时间从 $T_1$ 延长到 $T_2$，理论额外 SNR 收益约为：

$$
\Delta G
=
10\log_{10}
\frac{T_2}{T_1}
$$

例如：

$$
1\text{ ms}\rightarrow10\text{ ms}
$$

理论增益为：

$$
10\log_{10}(10)=10\text{ dB}
$$

---

### 15. residual Doppler 为什么会造成相干积分损失

积分期间若仍存在：

$$
\Delta f\neq0
$$

则复基带信号旋转：

$$
e^{j2\pi\Delta f t}
$$

积分结果：

$$
\int_0^T
e^{j2\pi\Delta f t}dt
$$

其幅度相对于理想值会出现 sinc 型衰减：

$$
L_f
=
\left|
\frac{\sin(\pi\Delta fT)}
{\pi\Delta fT}
\right|
$$

即：

$$
L_f
=
|\operatorname{sinc}(\Delta fT)|
$$

因此只要：

$$
\Delta fT
$$

变大，相干积分就会出现明显损失。

---

### 16. 一个直观例子

若：

$$
T=10\text{ ms}
$$

且 residual Doppler：

$$
\Delta f=50\text{ Hz}
$$

则积分期间总相位旋转：

$$
\Delta\theta
=
2\pi\Delta fT
$$

即：

$$
\Delta\theta
=
2\pi\times50\times0.01
=
\pi
$$

也就是：

$$
180^\circ
$$

前半段和后半段信号会明显互相抵消。

因此长相干积分要求更精确的 carrier wipeoff。

---

### 17. 导航 bit 翻转为什么也会破坏相干积分

若导航 bit 从：

$$
+1
$$

变成：

$$
-1
$$

那么相关结果符号也翻转。

如果积分窗口跨越未知 bit 边界：

$$
P_1+(-P_2)
$$

可能出现明显抵消。

因此数据通道的长相干积分受到导航 bit 限制。

如果 data bit 已知，可以执行 data wipeoff：

$$
P'[k]=P[k]d[k]
$$

从而恢复同符号累积。

---

### 18. 最终为什么得到六路 EPL I/Q

Tracking 同时需要：

- Carrier phase/frequency 信息；
- Code phase 信息。

Carrier 信息来自 I/Q：

$$
I=A\cos\Delta\phi
$$

$$
Q=A\sin\Delta\phi
$$

Code 信息来自 E/P/L 不同码延迟上的相关幅度。

因此最基本的 Tracking correlator 会形成：

$$
I_E,\quad Q_E
$$

$$
I_P,\quad Q_P
$$

$$
I_L,\quad Q_L
$$

即六路相关积分输出。

---

### 19. 整体信号链

```text
ADC / DFE samples
        ↓
Carrier NCO
        ↓
Carrier wipeoff
        ↓
I/Q baseband samples
        ↓
Local PRN E/P/L
        ↓
Despreading
        ↓
Coherent Integration
        ↓
IE/QE
IP/QP
IL/QL
        ↓
Discriminator
        ↓
DLL / PLL / FLL
```

---

### 20. 本章关键公式

接收信号：

$$
r(t)
=
A d(t)c(t-\tau)
\cos(2\pi(f_{IF}+f_D)t+\phi)
+n(t)
$$

残余频率：

$$
\Delta f
=
(f_{IF}+f_D)-\hat f
$$

残余相位：

$$
\Delta\phi
=
\phi-\hat\phi
$$

I/Q：

$$
I\propto A\cos(\Delta\phi)
$$

$$
Q\propto A\sin(\Delta\phi)
$$

复基带：

$$
P(t)
=
I+jQ
\propto
e^{j(2\pi\Delta ft+\Delta\phi)}
$$

频率与相位：

$$
\Delta f
=
\frac{1}{2\pi}
\frac{d\theta}{dt}
$$

Prompt：

$$
P
\propto
A R_c(\Delta\tau)e^{j\Delta\phi}
$$

相干积分额外增益：

$$
\Delta G
=
10\log_{10}
\frac{T_2}{T_1}
$$

频率误差造成的积分损失：

$$
L_f
=
\left|
\frac{\sin(\pi\Delta fT)}
{\pi\Delta fT}
\right|
$$

---

### 21. 本章关键结论

1. Carrier wipeoff 的目标不是一次完全消除载波，而是把大载波频率降成较小 residual carrier。
2. I/Q 本质上记录了 residual carrier 的二维相量信息。
3. residual phase 决定 I/Q 向量的方向。
4. residual Doppler 决定 I/Q 向量的旋转速度。
5. PRN 解扩通过本地码与接收码相乘恢复目标信号的相干性。
6. 码相位误差通过 PRN 自相关函数反映到相关器幅度。
7. Prompt 主要用于载波跟踪，Early/Late 主要用于码跟踪。
8. 相干积分可以提升 SNR，但受到 residual Doppler、相位动态和 bit 翻转限制。
9. Tracking 最基本的六路观测量为 `IE/QE/IP/QP/IL/QL`。
10. DLL、PLL、FLL 的鉴别器都建立在这些相关输出之上。

---

### 22. 下一章

下一章进入：

## 第 3 章｜相关器、自相关函数与 Early / Prompt / Late

重点分析：

- PRN 自相关峰为什么是三角形；
- E/P/L 为什么能判断码偏早还是偏晚；
- Early-Late spacing 为什么会影响噪声和多径性能；
- DLL 的 S 曲线是怎么形成的。


---

<a id="chapter-03"></a>

## 第 3 章｜相关器、自相关函数与 Early / Prompt / Late

### 1. 本章目标

理解：

- PRN 相关到底在计算什么；
- 为什么 BPSK 类 PRN 信号的自相关峰近似三角形；
- Early / Prompt / Late 为什么能估计码相位；
- DLL 的 S 曲线如何形成；
- Early-Late spacing 为什么影响噪声和多径性能。

---

### 2. 相关的本质

设接收到的 PRN 为：

$$
c(t-\tau)
$$

本地产生的 PRN 为：

$$
c(t-\hat\tau)
$$

相关函数可以写成：

$$
R(\Delta\tau)
=
\int
c(t-\tau)c(t-\hat\tau)\,dt
$$

其中：

$$
\Delta\tau=\tau-\hat\tau
$$

如果：

$$
\Delta\tau=0
$$

两路码完全对齐，相关值最大。

如果：

$$
|\Delta\tau|
$$

增大，则两路码的匹配程度降低，相关值下降。

所以相关函数本质上描述：

> 本地码相对于真实码移动时，两者的匹配程度如何变化。

---

### 3. 为什么自相关主峰近似三角形

先只看一个矩形 chip。

设：

$$
p(t)
=
\begin{cases}
1,&0\le t<T_c\\
0,&\text{otherwise}
\end{cases}
$$

把另一个相同矩形平移：

$$
p(t-\tau)
$$

两者相关：

$$
R_p(\tau)
=
\int p(t)p(t-\tau)dt
$$

相关值实际上就是两个矩形的重叠长度。

当：

$$
\tau=0
$$

完全重叠。

当：

$$
|\tau|=\frac{T_c}{2}
$$

只剩一半重叠。

当：

$$
|\tau|=T_c
$$

完全错开。

归一化以后：

$$
R_p(\tau)
=
\begin{cases}
1-\dfrac{|\tau|}{T_c},&|\tau|<T_c\\
0,&|\tau|\ge T_c
\end{cases}
$$

因此得到三角形主峰。

对理想 BPSK PRN，可以把单位换成 chip：

$$
R(\epsilon)
=
\begin{cases}
1-|\epsilon|,&|\epsilon|<1\\
0,&|\epsilon|\ge1
\end{cases}
$$

其中：

$$
R(0)=1
$$

$$
R(\pm0.5)=0.5
$$

$$
R(\pm1)=0
$$

示意：

```text
相关值
 1.0 |           /\
     |          /  \
 0.5 |         /    \
     |        /      \
 0.0 |_______/________\_______
            -1   0    +1
             Code Error / chip
```

实际系统受到前端带宽、滤波和调制方式影响，相关峰不会完全是理想三角形。

---

### 4. Prompt 相关器

Prompt 使用当前估计码位置：

$$
\hat\tau
$$

其输出近似：

$$
P=A R(\Delta\tau)
$$

当：

$$
\Delta\tau=0
$$

相关值最大。

但因为：

$$
R(+\epsilon)=R(-\epsilon)
$$

只看 Prompt 幅度无法判断本地码到底偏早还是偏晚。

这就是为什么需要 Early 和 Late。

---

### 5. Early / Prompt / Late

设 Early-Late 总间距为：

$$
d
$$

则：

$$
\tau_E=\hat\tau-\frac d2
$$

$$
\tau_P=\hat\tau
$$

$$
\tau_L=\hat\tau+\frac d2
$$

对应输出：

$$
E
=
A R\left(\Delta\tau+\frac d2\right)
$$

$$
P
=
A R(\Delta\tau)
$$

$$
L
=
A R\left(\Delta\tau-\frac d2\right)
$$

---

### 6. Prompt 对准时

当：

$$
\Delta\tau=0
$$

有：

$$
E
=
A R\left(\frac d2\right)
$$

$$
L
=
A R\left(-\frac d2\right)
$$

由于理想相关函数左右对称：

$$
R(x)=R(-x)
$$

因此：

$$
\boxed{E=L}
$$

这就是 DLL 的平衡点。

---

### 7. Prompt 偏早时

定义：

$$
\Delta\tau=\tau-\hat\tau
$$

若本地 Prompt 比真实码早：

$$
\hat\tau<\tau
$$

所以：

$$
\Delta\tau>0
$$

真实峰位于本地 Prompt 右侧，Late 更接近真实峰，因此：

$$
L>E
$$

此时：

$$
E-L<0
$$

---

### 8. Prompt 偏晚时

若：

$$
\hat\tau>\tau
$$

则：

$$
\Delta\tau<0
$$

Early 更接近真实峰：

$$
E>L
$$

所以：

$$
E-L>0
$$

注意：实际工程中鉴别器正负方向由误差定义、代码实现和 NCO 符号约定决定，不能只死记正负号。

---

### 9. DLL 鉴别器

最简单的鉴别器：

$$
D(\Delta\tau)=E-L
$$

代入：

$$
D(\Delta\tau)
=
A
R\left(\Delta\tau+\frac d2\right)
-
A
R\left(\Delta\tau-\frac d2\right)
$$

在：

$$
\Delta\tau=0
$$

处：

$$
D(0)=0
$$

即锁点对应鉴别器零交叉。

---

### 10. S 曲线

当 Prompt 从相关峰一侧扫到另一侧：

```text
Prompt 偏早
    ↓
L > E
    ↓
D < 0

接近中心
    ↓
E ≈ L
    ↓
D ≈ 0

Prompt 偏晚
    ↓
E > L
    ↓
D > 0
```

所以：

```text
D
↑
|           ______
|          /
|         /
|--------/-----------> Δτ
|       /
|      /
|_____/
```

这就是 DLL discriminator S-curve。

---

### 11. 为什么要归一化

如果直接用：

$$
D=E-L
$$

鉴别器输出会随信号幅度变化。

因此常用：

$$
D
=
\frac{E-L}{E+L}
$$

例如：

$$
E=900,\quad L=700
$$

则：

$$
D=\frac{900-700}{900+700}=0.125
$$

如果整个信号衰减 10 倍：

$$
E=90,\quad L=70
$$

仍然：

$$
D=0.125
$$

因此归一化可以降低信号幅度变化的影响。

---

### 12. 非相干 DLL 鉴别器

实际 E/L 是复相关量：

$$
E=I_E+jQ_E
$$

$$
L=I_L+jQ_L
$$

可以先计算幅度：

$$
|E|=\sqrt{I_E^2+Q_E^2}
$$

$$
|L|=\sqrt{I_L^2+Q_L^2}
$$

再构造：

$$
D
=
\frac{|E|-|L|}
{|E|+|L|}
$$

也可使用功率形式：

$$
D
=
\frac{I_E^2+Q_E^2-I_L^2-Q_L^2}
{I_E^2+Q_E^2+I_L^2+Q_L^2}
$$

这种形式对载波相位误差更不敏感。

---

### 13. Early-Late spacing

若：

$$
d=1\text{ chip}
$$

则：

$$
E=P-0.5\text{ chip}
$$

$$
L=P+0.5\text{ chip}
$$

如果：

$$
d=0.2\text{ chip}
$$

则：

$$
E=P-0.1\text{ chip}
$$

$$
L=P+0.1\text{ chip}
$$

较小 spacing 常称为 narrow correlator。

---

### 14. spacing 为什么影响性能

#### 多径

多径会扭曲相关峰。

减小 E/L spacing，可让 E/L 更靠近主峰中心，一般能降低部分多径偏差。

#### 噪声与实现

spacing 不能无限小，因为实际系统还受到：

- 前端带宽；
- 采样率；
- Code NCO 分辨率；
- E/L 噪声相关性；
- 量化误差；
- 相关峰失真。

因此 narrow correlator 不是 spacing 越小越好。

---

### 15. 前端带宽与 spacing

实际前端带宽有限，会把理想相关峰“磨圆、变宽”。

如果 spacing 很窄，而前端带宽不够：

- E/L 相关值差异可能变小；
- 鉴别器灵敏度下降；
- narrow correlator 优势无法充分发挥。

所以：

$$
\boxed{\text{Early-Late spacing}}
$$

必须结合：

$$
\boxed{\text{frontend bandwidth}}
$$

一起设计。

采样率高并不等价于前端带宽宽。

---

### 16. EPL 不一定只有三路

高性能接收机可能使用更多相关器：

```text
VE   E   P   L   VL
```

甚至更多 tap，用于：

- 多径检测；
- 相关峰形状监测；
- BOC 多峰消歧；
- 干扰或欺骗检测；
- 高级多径抑制算法。

---

### 17. EPL 本质是三个复相关观测

每个码位置都同时有 I/Q：

$$
E=I_E+jQ_E
$$

$$
P=I_P+jQ_P
$$

$$
L=I_L+jQ_L
$$

所以：

```text
Code delay
   ↓

Early          Prompt          Late
  │              │              │
IE+jQE         IP+jQP         IL+jQL
```

Carrier loop 主要观察复向量方向。

Code loop 主要观察不同 code delay 上的幅度差。

---

### 18. 一个数值例子

假设：

$$
R(\epsilon)=1-|\epsilon|
$$

Early-Late spacing：

$$
d=0.5\text{ chip}
$$

于是：

$$
E=P-0.25\text{ chip}
$$

$$
L=P+0.25\text{ chip}
$$

假设 Prompt 偏早：

$$
\Delta\tau=+0.1\text{ chip}
$$

那么：

$$
E
=
R(0.1+0.25)
=
R(0.35)
=
0.65
$$

$$
L
=
R(0.1-0.25)
=
R(-0.15)
=
0.85
$$

所以：

$$
L>E
$$

归一化鉴别器：

$$
D
=
\frac{0.65-0.85}{0.65+0.85}
\approx-0.133
$$

当：

$$
\Delta\tau\rightarrow0
$$

有：

$$
E\rightarrow L
$$

$$
D\rightarrow0
$$

---

### 19. DLL 不是每次重新找最大峰

DLL 并不是反复：

> 搜索 Prompt 最大值。

它真正做的是：

$$
E-L\rightarrow0
$$

流程：

```text
E/L 相关值
   ↓
DLL discriminator
   ↓
得到码误差
   ↓
Loop Filter
   ↓
Code NCO
   ↓
移动本地 PRN
   ↓
E ≈ L
```

这就是闭环跟踪，而不是 Acquisition 式搜索。

---

### 20. 本章关键公式

PRN 自相关：

$$
R(\Delta\tau)
=
\int
c(t)c(t-\Delta\tau)dt
$$

理想 BPSK 主峰：

$$
R(\epsilon)
=
\begin{cases}
1-|\epsilon|,&|\epsilon|<1\\
0,&|\epsilon|\ge1
\end{cases}
$$

E/P/L：

$$
E
=
A R\left(\Delta\tau+\frac d2\right)
$$

$$
P
=
A R(\Delta\tau)
$$

$$
L
=
A R\left(\Delta\tau-\frac d2\right)
$$

锁定平衡条件：

$$
E=L
$$

简单 DLL：

$$
D=E-L
$$

归一化 DLL：

$$
D
=
\frac{E-L}{E+L}
$$

非相干幅度：

$$
|E|=\sqrt{I_E^2+Q_E^2}
$$

$$
|L|=\sqrt{I_L^2+Q_L^2}
$$

---

### 21. 本章关键结论

1. 相关函数表示本地 PRN 与接收 PRN 在不同时间偏移下的匹配程度。
2. 理想 BPSK 矩形 chip 的自相关主峰近似为三角形。
3. Prompt 单独不能可靠区分偏早还是偏晚。
4. Early / Late 通过左右两侧相关幅度差提供码误差方向。
5. Prompt 位于相关峰中心时，理想情况下 $E=L$。
6. DLL 的目标不是最大化 Prompt，而是让鉴别器输出趋近零。
7. $E-L$ 随码误差形成典型 S 曲线。
8. 归一化鉴别器可降低信号幅度变化影响。
9. Early-Late spacing 影响多径、噪声、鉴别器灵敏度和硬件实现。
10. Narrow correlator 通常能改善部分多径性能，但 spacing 不能脱离前端带宽、采样率和 NCO 分辨率设计。
11. EPL 本质上是三个不同 code delay 上的复相关观测。

---

### 22. 下一章

下一章进入：

## 第 4 章｜DLL 鉴别器、S 曲线与码跟踪环

重点包括：

- DLL 鉴别器数学推导；
- S 曲线斜率的物理意义；
- coherent / non-coherent DLL；
- DLL noise jitter；
- DLL bandwidth 与动态性能；
- Code NCO 如何根据 DLL 输出持续修正码相位。


---

<a id="chapter-04"></a>

## 第 4 章｜DLL 鉴别器、S 曲线与码跟踪环

### 1. 本章目标

本章重点理解：

- DLL 鉴别器如何把 Early/Late 相关值转换成码误差；
- S 曲线的零点、斜率和线性区分别意味着什么；
- coherent / non-coherent DLL 的区别；
- Loop Filter 在 DLL 中到底做什么；
- DLL bandwidth 为什么影响噪声与动态性能；
- Code NCO 如何根据 DLL 输出持续修正本地码；
- DLL noise jitter 和 dynamic stress error 的基本来源。

### 2. DLL 的完整闭环

```text
Code NCO
   ↓
Local PRN
   ↓
E / P / L Correlator
   ↓
DLL Discriminator
   ↓
Loop Filter
   ↓
Code NCO correction
   ↓
继续下一次相关
```

目标：

$$
\Delta\tau=\tau-\hat\tau\rightarrow0
$$

### 3. Early-Minus-Late 鉴别器

设 Early-Late 总间距为 $d$：

$$
E=A R\left(\Delta\tau+\frac d2\right)
$$

$$
L=A R\left(\Delta\tau-\frac d2\right)
$$

最简单鉴别器：

$$
D(\Delta\tau)=E-L
$$

当：

$$
\Delta\tau=0
$$

由于相关函数左右对称：

$$
E=L
$$

所以：

$$
D(0)=0
$$

DLL 的理想锁点就是鉴别器零交叉。

### 4. S 曲线与鉴别器增益

在锁点附近，可把鉴别器近似线性化：

$$
D(\Delta\tau)\approx K_D\Delta\tau
$$

其中：

$$
K_D=
\left.
\frac{dD}{d\Delta\tau}
\right|_{\Delta\tau=0}
$$

称为鉴别器增益或 S 曲线斜率。

斜率越大，意味着相同的小码误差能产生更明显的鉴别器输出。

需要注意：

- 零点：候选锁点；
- 斜率：锁点附近灵敏度；
- 线性区：只有在该区域内，$D$ 才近似正比于 $\Delta\tau$。

偏离过大后，S 曲线会弯曲，鉴别器不再近似线性。

### 5. 为什么要归一化

直接使用：

$$
D=E-L
$$

会受到信号幅度变化影响。

因此常用：

$$
D=
\frac{E-L}{E+L}
$$

例如：

$$
E=900,\quad L=700
$$

则：

$$
D=0.125
$$

整体衰减 10 倍后：

$$
E=90,\quad L=70
$$

仍然：

$$
D=0.125
$$

因此归一化可以降低整体信号强度变化的影响。

### 6. Coherent DLL

如果 Carrier Loop 已经很好锁定，信号主要落在 I 路：

$$
Q\approx0
$$

可以直接使用：

$$
D_{coh}=I_E-I_L
$$

或者：

$$
D_{coh}=
\frac{I_E-I_L}{I_E+I_L}
$$

优点：

- 结构简单；
- 载波锁定良好时噪声性能较好。

缺点：

- 对 Carrier Phase Error 敏感。

### 7. Non-Coherent DLL

实际工程中常先计算：

$$
E_m=\sqrt{I_E^2+Q_E^2}
$$

$$
L_m=\sqrt{I_L^2+Q_L^2}
$$

然后：

$$
D_{nc}=
\frac{E_m-L_m}{E_m+L_m}
$$

也可以使用功率形式：

$$
D=
\frac{I_E^2+Q_E^2-I_L^2-Q_L^2}
{I_E^2+Q_E^2+I_L^2+Q_L^2}
$$

因为：

$$
\sqrt{I^2+Q^2}
$$

不依赖 I/Q 向量方向，所以 Non-Coherent DLL 对载波相位误差更鲁棒。

### 8. 为什么 PLL 没完全锁定时 DLL 仍可工作

设：

$$
E=A_Ee^{j\phi}
$$

那么：

$$
I_E=A_E\cos\phi
$$

$$
Q_E=A_E\sin\phi
$$

所以：

$$
\sqrt{I_E^2+Q_E^2}=A_E
$$

载波相位 $\phi$ 被消掉。

Late 同理，因此 Non-Coherent DLL 在 PLL 还没有完全锁相时仍能获取码误差信息。

### 9. Early-Late spacing

例如：

$$
d=1\text{ chip}
$$

表示：

$$
E=P-0.5\text{ chip}
$$

$$
L=P+0.5\text{ chip}
$$

如果：

$$
d=0.2\text{ chip}
$$

则：

$$
E=P-0.1\text{ chip}
$$

$$
L=P+0.1\text{ chip}
$$

spacing 会影响：

- S 曲线形状；
- 锁点附近斜率；
- E/L 噪声相关性；
- 多径偏差；
- 前端带宽要求。

### 10. Loop Filter 为什么存在

实际鉴别器输出可以粗略写成：

$$
D[k]
=
K_D\Delta\tau[k]
+n_D[k]
+b_D[k]
$$

其中：

- $K_D\Delta\tau$：真实码误差；
- $n_D$：热噪声造成的随机抖动；
- $b_D$：多径、干扰等造成的偏置。

如果直接把鉴别器输出全部送给 Code NCO，NCO 会跟着噪声快速抖动。

因此需要：

```text
DLL discriminator
      ↓
Loop Filter
      ↓
平滑后的修正量
      ↓
Code NCO
```

Loop Filter 决定 DLL 可以跟随多快的真实码动态，同时允许多少测量噪声进入 NCO。

### 11. DLL Bandwidth

DLL 常用 $B_n$ 表示等效噪声带宽。

#### 带宽较大

$$
B_n\uparrow
$$

意味着：

- 响应更快；
- 动态能力更强；
- 更多鉴别器噪声进入 NCO；
- 码相位抖动更大。

#### 带宽较小

$$
B_n\downarrow
$$

意味着：

- 噪声更小；
- 码跟踪更平稳；
- 环路响应更慢；
- 快速动态可能跟不上。

核心折中：

$$
\boxed{
\text{动态跟踪能力}
\leftrightarrow
\text{噪声抑制能力}
}
$$

### 12. 为什么 DLL 通常比 PLL 带宽窄

Carrier Doppler 和 Code Doppler 都来自相对径向运动。

近似：

$$
\frac{\Delta f_{code}}{f_{code}}
\approx
\frac{\Delta f_{carrier}}{f_{carrier}}
$$

GPS L1 C/A：

$$
f_{carrier}\approx1.57542\text{ GHz}
$$

$$
f_{code}=1.023\text{ MHz}
$$

所以即使 Carrier Doppler 是几 kHz，Code Doppler 通常也只是几个 Hz 量级。

因此 Code Loop 的动态通常比 Carrier Loop 慢，DLL 可以使用更窄的带宽。

### 13. Carrier Aiding

利用 Carrier Doppler 预测 Code Doppler：

$$
\Delta f_{code}
\approx
f_{code}
\frac{\Delta f_{carrier}}{f_{carrier}}
$$

这样大部分动态由 Carrier Loop 提供。

DLL 只需跟踪较小残差，因此可以：

- 使用更窄 DLL bandwidth；
- 降低码跟踪噪声；
- 提高弱信号能力。

这称为 Carrier Aiding。

### 14. Code NCO 在修什么

Code NCO 控制：

- 本地码相位；
- 本地码速率。

名义码率：

$$
f_{code,0}
$$

实际 NCO：

$$
\hat f_{code}
=
f_{code,0}
+
\Delta\hat f_{code}
$$

如果 Code NCO 频率存在误差，则码相位误差会随时间累积。

例如 Code NCO 每秒多跑：

$$
0.5\text{ chip/s}
$$

100 ms 后就会积累：

$$
0.05\text{ chip}
$$

误差。

所以 DLL 不只是“一次移动 Prompt”，而是在持续修正本地码推进速度。

### 15. 一个闭环例子

假设：

$$
\Delta\tau=+0.10\text{ chip}
$$

得到：

$$
E=0.65
$$

$$
L=0.85
$$

归一化鉴别器：

$$
D=
\frac{0.65-0.85}{0.65+0.85}
\approx-0.133
$$

Loop Filter 不会一次把全部误差修完，而是逐步改变 Code NCO。

例如：

```text
初始误差：0.10 chip
   ↓
第一次修正
   ↓
误差：0.04 chip
   ↓
再次鉴别
   ↓
继续修正
   ↓
误差 → 0
```

最终：

$$
E\rightarrow L
$$

$$
D\rightarrow0
$$

### 16. DLL Noise Jitter

即使真实码误差：

$$
\Delta\tau=0
$$

由于噪声，瞬时仍可能：

$$
E\neq L
$$

因此鉴别器输出会在 0 附近抖动，Code NCO 也会出现小范围随机抖动。

这称为：

> DLL tracking jitter

或：

> Code tracking noise jitter

总体趋势：

$$
C/N_0\uparrow
\Rightarrow
\sigma_{DLL}\downarrow
$$

$$
T_{coh}\uparrow
\Rightarrow
\sigma_{DLL}\downarrow
$$

$$
B_n\uparrow
\Rightarrow
\sigma_{DLL}\uparrow
$$

常见近似趋势：

$$
\sigma_{DLL}
\propto
\sqrt{
\frac{B_n}{C/N_0}
}
$$

实际还受到：

- Early-Late spacing；
- 积分时间；
- 前端带宽；
- 鉴别器类型

等影响。

### 17. Dynamic Stress Error

如果真实码相位变化过快，而 DLL bandwidth 太窄，环路会跟不上。

此时即使没有热噪声：

$$
\Delta\tau\neq0
$$

仍然会产生持续跟踪误差。

这种误差称为：

> Dynamic Stress Error

所以 DLL 总误差可以粗略理解为：

$$
\text{Total Error}
=
\text{Noise Jitter}
+
\text{Dynamic Error}
+
\text{Bias}
$$

其中 Bias 可能来自：

- 多径；
- 干扰；
- 前端失真；
- 鉴别器非理想。

### 18. DLL bandwidth 的核心矛盾

```text
Bn 太大
 ↓
动态跟踪快
但
Noise Jitter 大

Bn 太小
 ↓
Noise Jitter 小
但
Dynamic Stress Error 大
```

因此 DLL bandwidth 不是越窄越好，而是 Noise 和 Dynamics 的折中。

### 19. DLL 与 Multipath

Loop Filter 可以抑制快速随机噪声。

但如果多径造成的是稳定偏置，S 曲线的零点本身会移动。

例如真实码位置为：

$$
0
$$

但多径使鉴别器零点移动到：

$$
0.08\text{ chip}
$$

那么 DLL 会稳定锁在错误位置附近。

Loop Filter 无法判断这是多径造成的 bias。

因此：

> Loop Filter 可以压随机噪声，但不能自动消除稳定的多径或干扰偏置。

### 20. DLL 状态机

实际接收机不一定始终使用同一组参数。

可能：

```text
Acquisition
   ↓
DLL Pull-In
   ↓
较大 bandwidth
   ↓
快速消除初始码误差
   ↓
DLL Lock
   ↓
减小 bandwidth
   ↓
降低稳态噪声
```

弱信号情况下还可能：

```text
C/N0 下降
   ↓
延长积分
   ↓
减小 bandwidth
   ↓
依赖 Carrier Aiding
```

### 21. DLL 工程数据流

```text
IE/QE
IP/QP
IL/QL
   ↓
计算 |E|、|L|
   ↓
DLL Discriminator
   ↓
Raw Code Error
   ↓
Loop Filter
   ↓
Code NCO Correction
   ↓
Code FCW / Phase
   ↓
PRN Generator
   ↓
下一周期 EPL
```

### 22. 本章关键公式

码误差：

$$
\Delta\tau=\tau-\hat\tau
$$

E/L：

$$
E
=
A R\left(\Delta\tau+\frac d2\right)
$$

$$
L
=
A R\left(\Delta\tau-\frac d2\right)
$$

简单鉴别器：

$$
D=E-L
$$

归一化鉴别器：

$$
D=
\frac{E-L}{E+L}
$$

Non-Coherent DLL：

$$
D=
\frac{\sqrt{I_E^2+Q_E^2}-\sqrt{I_L^2+Q_L^2}}
{\sqrt{I_E^2+Q_E^2}+\sqrt{I_L^2+Q_L^2}}
$$

锁点附近：

$$
D\approx K_D\Delta\tau
$$

Carrier Aiding：

$$
\Delta f_{code}
\approx
f_{code}
\frac{\Delta f_{carrier}}{f_{carrier}}
$$

DLL jitter 趋势：

$$
\sigma_{DLL}
\propto
\sqrt{
\frac{B_n}{C/N_0}
}
$$

### 23. 本章关键结论

1. DLL 的目标是让码误差 $\Delta\tau$ 趋近 0。
2. Early/Late 幅度差形成 DLL 鉴别器输出。
3. S 曲线零交叉是 DLL 的候选锁点。
4. S 曲线锁点附近斜率决定鉴别器对小码误差的灵敏度。
5. 鉴别器只有在一定范围内近似线性。
6. 归一化可以降低信号幅度变化的影响。
7. Coherent DLL 依赖较好的 Carrier Lock。
8. Non-Coherent DLL 使用复相关幅度，对载波相位误差更鲁棒。
9. Loop Filter 决定 DLL 的噪声与动态折中。
10. DLL bandwidth 越宽，动态能力越强，但 noise jitter 越大。
11. DLL bandwidth 越窄，噪声越低，但 dynamic stress error 越大。
12. Code NCO 持续控制本地码速率和码相位。
13. Carrier Aiding 可以利用 Carrier Doppler 预测大部分 Code Doppler。
14. Loop Filter 无法消除稳定的多径或干扰 bias。
15. 实际接收机通常会根据 Tracking 状态动态调整积分时间和 DLL bandwidth。

### 24. 下一章

下一章进入：

## 第 5 章｜PLL：载波相位鉴别器与 Costas Loop

重点包括：

- 为什么 $I/Q$ 能直接估计载波相位；
- `atan2(Q,I)` 和 `atan(Q/I)` 的区别；
- 导航 bit 为什么会造成 180° 相位翻转；
- Costas Loop 为什么能消除 data bit 的影响；
- PLL 的相位锁定范围；
- cycle slip；
- PLL bandwidth 与 phase jitter。


---

<a id="chapter-05"></a>

## 第 5 章｜PLL：载波相位鉴别器与 Costas Loop

### 1. 本章目标

本章重点理解：

- 为什么 Prompt I/Q 可以估计载波相位误差；
- `atan2(Q,I)` 和 `atan(Q/I)` 的区别；
- 导航 bit 为什么会造成 180° 相位翻转；
- Costas Loop 为什么可以消除未知导航 bit 的影响；
- PLL 如何通过 Loop Filter 和 Carrier NCO 修正载波；
- PLL bandwidth、phase jitter、dynamic stress error 与 cycle slip；
- PLL 和 FLL 的分工。

---

### 2. PLL 的完整闭环

```text
Carrier NCO
     ↓
Local Carrier
     ↓
Carrier Wipeoff
     ↓
Prompt I/Q
     ↓
PLL Discriminator
     ↓
Loop Filter
     ↓
Carrier NCO Correction
     ↓
下一次积分
```

目标：

$$
\Delta\phi=\phi-\hat\phi\rightarrow0
$$

并持续使：

$$
\Delta f=f-\hat f\rightarrow0
$$

---

### 3. Prompt I/Q 与载波相位误差

在码已经基本对准、积分期间 residual Doppler 较小时：

$$
I_P=A d\cos(\Delta\phi)
$$

$$
Q_P=A d\sin(\Delta\phi)
$$

其中：

- $A$：相关后幅度；
- $d\in\{+1,-1\}$：导航数据 bit；
- $\Delta\phi$：载波相位误差。

若暂时忽略 data bit：

$$
d=+1
$$

则：

$$
\tan(\Delta\phi)=\frac{Q_P}{I_P}
$$

所以：

$$
\Delta\phi=\operatorname{atan2}(Q_P,I_P)
$$

---

### 4. I/Q 的几何意义

定义：

$$
P=I_P+jQ_P
$$

则：

$$
P=Ae^{j\Delta\phi}
$$

因此：

- $|P|$ 主要反映相关能量；
- $\arg(P)$ 反映 Carrier Phase Error。

理想锁定：

$$
\Delta\phi\rightarrow0
$$

因此：

$$
Q_P\rightarrow0
$$

信号主要落在 I 轴上。

---

### 5. `atan2(Q,I)` 鉴别器

最直接：

$$
D_{PLL}=\operatorname{atan2}(Q_P,I_P)
$$

它可以区分四象限，输出范围通常为：

$$
-\pi<D_{PLL}\le\pi
$$

即：

$$
-180^\circ\sim180^\circ
$$

因此 `atan2` 能恢复完整的复向量方向。

---

### 6. `atan(Q/I)` 与 `atan2(Q,I)` 的区别

使用：

$$
D=\arctan\left(\frac{Q}{I}\right)
$$

时，无法区分：

$$
(I,Q)
$$

和：

$$
(-I,-Q)
$$

因为：

$$
\frac{-Q}{-I}=\frac{Q}{I}
$$

所以它把相差：

$$
180^\circ
$$

的两个方向视为相同。

这恰好适合 BPSK Data Channel。

---

### 7. 导航 bit 为什么造成 180° 翻转

BPSK 数据：

$$
d\in\{+1,-1\}
$$

Prompt：

$$
P=Ade^{j\Delta\phi}
$$

如果：

$$
d=-1
$$

由于：

$$
-1=e^{j\pi}
$$

所以：

$$
P=Ae^{j(\Delta\phi+\pi)}
$$

即 data bit 翻转会让 I/Q 向量整体旋转：

$$
180^\circ
$$

---

### 8. 为什么普通 `atan2` PLL 会受 data bit 影响

若：

$$
\Delta\phi=0
$$

且：

$$
d=+1
$$

则：

$$
I=A,\quad Q=0
$$

鉴别器输出 0。

如果 bit 变成：

$$
d=-1
$$

则：

$$
I=-A,\quad Q=0
$$

此时 `atan2` 会认为相位发生约：

$$
180^\circ
$$

变化。

所以普通完整相位 PLL 会被未知导航数据扰动。

---

### 9. Costas Loop

Costas Loop 的核心目标：

> 让相差 $180^\circ$ 的两个 BPSK 状态对 PLL 来说等价。

典型鉴别器：

$$
D_{Costas}
=
\arctan\left(\frac{Q}{I}\right)
$$

当：

$$
(I,Q)\rightarrow(-I,-Q)
$$

时：

$$
\frac{Q}{I}
$$

不变。

因此导航 bit 被自然消除。

代价是：

> Costas PLL 只能获得 modulo $\pi$ 的载波相位，存在 $180^\circ$ 相位模糊。

---

### 10. 其他 Costas 鉴别器

锁点附近：

$$
I\approx A
$$

$$
Q\approx A\Delta\phi
$$

因此：

$$
IQ\approx A^2\Delta\phi
$$

所以：

$$
D=IQ
$$

也能近似作为相位鉴别器。

常见 data-insensitive 形式还包括：

$$
D=\operatorname{sign}(I)Q
$$

因为 bit 翻转时 I/Q 同时变号，但该结果保持不变。

---

### 11. PLL 的 S 曲线

锁点附近：

$$
D_{PLL}(\Delta\phi)\approx K_D\Delta\phi
$$

其中：

$$
K_D
$$

是 Carrier phase discriminator gain。

与 DLL 一样：

- 零交叉：候选锁点；
- 斜率：对小相位误差的灵敏度；
- 线性区：正常小误差工作区域。

---

### 12. PLL 为什么通过调频来修相位

相位与频率满足：

$$
\frac{d\phi}{dt}=2\pi f
$$

所以对误差：

$$
\frac{d\Delta\phi}{dt}=2\pi\Delta f
$$

因此：

$$
\Delta f
=
\frac{1}{2\pi}
\frac{d\Delta\phi}{dt}
$$

PLL 虽然直接观测 Phase Error，但通常通过 Loop Filter 输出 frequency/FCW correction。

改变 Carrier NCO frequency 后，相位会逐渐被拉回。

---

### 13. PLL Loop Filter

鉴别器输出可以粗略写成：

$$
D[k]
=
K_D\Delta\phi[k]
+n_\phi[k]
+b_\phi[k]
$$

其中：

- $K_D\Delta\phi$：真实相位误差；
- $n_\phi$：热噪声；
- $b_\phi$：干扰、多径等偏置。

Loop Filter 决定：

$$
\boxed{
\text{动态跟踪能力}
\leftrightarrow
\text{相位噪声抑制能力}
}
$$

---

### 14. PLL Bandwidth

#### Bandwidth 较大

$$
B_n\uparrow
$$

意味着：

- 高频动态跟得更快；
- pull-in/响应更快；
- 更多 phase noise 进入 NCO；
- phase jitter 增大。

#### Bandwidth 较小

$$
B_n\downarrow
$$

意味着：

- phase jitter 更小；
- 输出更平稳；
- 高动态时更容易跟不上。

---

### 15. PLL Phase Jitter

即使真实：

$$
\Delta\phi=0
$$

Prompt I/Q 仍受热噪声影响。

因此鉴别器输出会在 0 附近波动。

这种稳态随机误差称为：

> PLL Phase Jitter

总体趋势：

$$
C/N_0\uparrow
\Rightarrow
\sigma_\phi\downarrow
$$

$$
B_n\uparrow
\Rightarrow
\sigma_\phi\uparrow
$$

常见近似趋势：

$$
\sigma_\phi
\propto
\sqrt{\frac{B_n}{C/N_0}}
$$

---

### 16. 相干积分时间与 PLL

延长：

$$
T_{coh}
$$

通常能提高单次 Prompt I/Q 的 SNR。

但同时：

- 更新率可能下降；
- 对 residual Doppler 更敏感；
- 对动态变化更敏感；
- Data Channel 还会受 bit 翻转限制。

所以：

$$
T_{coh},\quad T_{update},\quad B_n
$$

需要联合设计。

---

### 17. 一阶、二阶、三阶 PLL 的直觉

#### 一阶 PLL

适合跟踪近似恒定相位。

#### 二阶 PLL

可以较好跟踪恒定频率偏差：

$$
\Delta f=\text{constant}
$$

即相位线性变化。

#### 三阶 PLL

可以进一步跟踪持续变化的频率：

$$
\frac{d\Delta f}{dt}\neq0
$$

适合更强的加速度动态。

GNSS Carrier Tracking 中常见二阶或三阶环。

---

### 18. Dynamic Stress Error

如果真实 Carrier 动态变化太快，而 PLL bandwidth 太窄，环路会跟不上。

即使没有热噪声，也会产生持续：

$$
\Delta\phi\neq0
$$

这就是 Dynamic Stress Error。

所以 PLL 总误差可以粗略理解为：

$$
\text{PLL Error}
=
\text{Phase Jitter}
+
\text{Dynamic Error}
+
\text{Bias}
$$

---

### 19. Cycle Slip

PLL 正常锁定时，相位误差只在锁点附近小幅波动。

如果：

- $C/N_0$ 太低；
- 干扰突然增强；
- 动态太大；
- bandwidth 太窄；
- residual Doppler 太大；

相位误差可能跨出稳定区域。

环路重新锁定到相差整数个周期的位置，称为：

> Cycle Slip

对于 Carrier Phase / ADR 来说非常关键。

一次 cycle slip 会使累计载波相位跳变：

$$
N\times2\pi
$$

即整数个 carrier cycle。

---

### 20. Costas Loop 的 180° 模糊

BPSK Costas Loop 本身具有 $\pi$ 周期性，因此可能出现：

$$
180^\circ
$$

状态翻转。

对 Costas Loop 来说可能仍然是“锁定”的，但 data 符号会整体翻转。

因此实际接收机常结合：

- bit synchronization；
- navigation data；
- carrier phase continuity；
- measurement consistency

进行检测和处理。

---

### 21. PLL 与 FLL

PLL 直接估计：

$$
\Delta\phi
$$

FLL 更直接估计：

$$
\Delta f
$$

#### FLL

优点：

- 对较大的 frequency error 更鲁棒；
- pull-in range 更大。

缺点：

- 最终相位精度不如 PLL。

#### PLL

优点：

- 相位精度高；
- 适合高质量 Carrier Phase measurement。

缺点：

- 对大 frequency error、高动态更敏感。

因此常见流程：

```text
Acquisition
   ↓
FLL Pull-In
   ↓
FLL-Assisted PLL
   ↓
PLL Lock
```

---

### 22. PLL 与 Carrier NCO / FCW

N 位 Carrier NCO：

$$
\Phi[k+1]
=
\Phi[k]+FCW
\pmod{2^N}
$$

FCW：

$$
FCW=
\frac{f}{f_s}2^N
$$

PLL 数据链：

```text
IP/QP
  ↓
PLL Discriminator
  ↓
Phase Error
  ↓
Loop Filter
  ↓
Frequency / FCW Correction
  ↓
Carrier NCO
  ↓
Local Carrier
```

---

### 23. 一个简单例子

假设 residual frequency：

$$
\Delta f=20\text{ Hz}
$$

相干积分周期：

$$
T=1\text{ ms}
$$

那么每 1 ms 相位增长：

$$
\Delta\theta
=
360^\circ\times20\times0.001
=
7.2^\circ
$$

所以可能观察到：

```text
第1ms：  7.2°
第2ms： 14.4°
第3ms： 21.6°
...
```

PLL 看到相位持续向同一方向漂移，就会调整 Carrier NCO frequency。

最终：

$$
\Delta f\rightarrow0
$$

相位不再持续漂移，只在锁点附近小幅波动。

---

### 24. Data Channel 与 Pilot Channel

Data Channel 存在未知：

$$
d(t)\in\{\pm1\}
$$

因此通常使用：

- Costas discriminator；
- 或 Data Wipeoff。

Pilot Channel 没有未知导航 bit，或只存在已知 Secondary Code。

完成 Secondary Code Wipeoff 后，可以使用完整：

$$
\operatorname{atan2}(Q,I)
$$

并支持更长相干积分。

---

### 25. 本章完整 Carrier Loop

```text
Input Samples
      ↓
Carrier NCO
      ↓
Carrier Wipeoff
      ↓
Prompt I/Q
      ↓
PLL / Costas Discriminator
      ↓
Raw Phase Error
      ↓
Loop Filter
      ↓
Carrier Frequency / FCW Correction
      ↓
Carrier NCO
      ↓
下一次积分
```

最终：

$$
\boxed{\Delta f\rightarrow0}
$$

以及：

$$
\boxed{\Delta\phi\text{ 保持在锁点附近}}
$$

---

### 26. 本章关键公式

Prompt：

$$
I_P=A d\cos(\Delta\phi)
$$

$$
Q_P=A d\sin(\Delta\phi)
$$

完整相位鉴别器：

$$
D=\operatorname{atan2}(Q_P,I_P)
$$

Costas 鉴别器：

$$
D=\arctan\left(\frac{Q_P}{I_P}\right)
$$

锁点附近：

$$
D\approx K_D\Delta\phi
$$

相位与频率：

$$
\frac{d\Delta\phi}{dt}=2\pi\Delta f
$$

因此：

$$
\Delta f=
\frac{1}{2\pi}\frac{d\Delta\phi}{dt}
$$

PLL Jitter 趋势：

$$
\sigma_\phi
\propto
\sqrt{\frac{B_n}{C/N_0}}
$$

Carrier NCO：

$$
FCW=
\frac{f}{f_s}2^N
$$

---

### 27. 本章关键结论

1. Prompt I/Q 的复向量角度反映 Carrier Phase Error。
2. `atan2(Q,I)` 可以恢复完整四象限相位。
3. BPSK data bit 翻转会使 I/Q 向量旋转 180°。
4. 普通 `atan2` PLL 会受未知 data bit 翻转影响。
5. Costas Loop 利用 $\pi$ 周期对称性消除未知 data bit 的影响。
6. Costas PLL 的代价是存在 180° 相位模糊。
7. PLL 通常通过修改 Carrier NCO frequency / FCW 逐渐修正 Phase Error。
8. Frequency Error 是 Phase Error 的时间变化率。
9. PLL bandwidth 越宽，动态能力越强，但 phase jitter 越大。
10. PLL bandwidth 越窄，噪声越小，但 Dynamic Stress Error 越大。
11. Cycle Slip 会破坏 Carrier Phase / ADR 连续性。
12. FLL 适合较大 Frequency Error 拉入，PLL 适合高精度相位跟踪。
13. Data Channel 常使用 Costas Loop；Pilot Channel 更适合完整相位跟踪和长相干积分。

---

### 28. 下一章

下一章进入：

## 第 6 章｜FLL：频率鉴别器、叉积/点积与 FLL-Assisted PLL

重点包括：

- 为什么相邻两个 Prompt I/Q 可以估计 Frequency Error；
- 复共轭乘积为什么能得到相位增量；
- Cross / Dot FLL discriminator；
- FLL 的 frequency pull-in range；
- FLL bandwidth；
- FLL、PLL 如何切换和联合工作；
- 为什么 Acquisition 后通常先 FLL 再 PLL。


---

<a id="chapter-06"></a>

## 第 6 章｜FLL：频率鉴别器、叉积/点积与 FLL-Assisted PLL

### 1. 本章目标

本章重点理解：

- FLL 为什么可以根据相邻两个 Prompt I/Q 估计载波频率误差；
- 复共轭乘积、点积和叉积之间的关系；
- 四象限 `atan2` FLL 与两象限 `atan` FLL 的区别；
- BPSK data bit 翻转为什么会影响 FLL，以及如何降低这种影响；
- FLL 的无模糊频率范围为什么与更新周期有关；
- FLL bandwidth 的作用；
- Acquisition、FLL、PLL 为什么通常按“粗到细”的方式衔接；
- FLL-Assisted PLL 是怎么工作的。

---

### 2. FLL 到底跟踪什么

PLL 直接观察的是：

$$
\Delta\phi
$$

也就是 Carrier Phase Error。

FLL 更直接关注：

$$
\Delta f
$$

也就是 Carrier Frequency Error / Residual Doppler。

两者关系：

$$
\frac{d\Delta\phi}{dt}
=
2\pi\Delta f
$$

所以：

$$
\Delta f
=
\frac{1}{2\pi}
\frac{d\Delta\phi}{dt}
$$

因此，只要观察一段时间内 Carrier Phase 变化了多少，就能估计平均 Frequency Error。

---

### 3. 为什么需要两个相邻 Prompt

设第 $k-1$ 个 Prompt 复相关结果：

$$
P_{k-1}
=
I_{k-1}+jQ_{k-1}
=
A_{k-1}e^{j\theta_{k-1}}
$$

第 $k$ 个：

$$
P_k
=
I_k+jQ_k
=
A_ke^{j\theta_k}
$$

两个积分结果之间间隔为：

$$
T
$$

如果 residual frequency 在这段时间内近似恒定：

$$
\theta_k-\theta_{k-1}
\approx
2\pi\Delta f T
$$

因此：

$$
\boxed{
\Delta f
\approx
\frac{\theta_k-\theta_{k-1}}
{2\pi T}
}
$$

这就是 FLL 最基本的原理。

---

### 4. 相邻 Prompt 向量图

示意图文件：

```text
GNSS_Tracking_第06章_FLL_相邻Prompt向量.png
```

图中：

- $P_{k-1}$ 角度为 $20^\circ$；
- $P_k$ 角度为 $65^\circ$；
- 相位增量：

$$
\Delta\theta=45^\circ
$$

如果两个 Prompt 间隔：

$$
T=1\text{ ms}
$$

则：

$$
\Delta f
=
\frac{45^\circ}{360^\circ\times0.001}
=
125\text{ Hz}
$$

所以这两个 Prompt 告诉 FLL：

> 本地 Carrier NCO 仍然存在约 125 Hz 的 residual frequency。

---

### 5. 为什么用复共轭乘积

直接分别计算两个 `atan2` 再相减当然可以：

$$
\theta_k
=
\operatorname{atan2}(Q_k,I_k)
$$

$$
\theta_{k-1}
=
\operatorname{atan2}(Q_{k-1},I_{k-1})
$$

再计算：

$$
\Delta\theta
=
\theta_k-\theta_{k-1}
$$

但更自然的方法是计算：

$$
P_kP_{k-1}^*
$$

因为：

$$
P_k
=
A_ke^{j\theta_k}
$$

$$
P_{k-1}^*
=
A_{k-1}e^{-j\theta_{k-1}}
$$

所以：

$$
P_kP_{k-1}^*
=
A_kA_{k-1}
e^{j(\theta_k-\theta_{k-1})}
$$

其相位直接就是：

$$
\boxed{
\Delta\theta
=
\theta_k-\theta_{k-1}
}
$$

因此：

$$
\boxed{
\Delta f
=
\frac{
\arg(P_kP_{k-1}^*)
}
{2\pi T}
}
$$

---

### 6. 点积和叉积怎么出现的

展开：

$$
P_kP_{k-1}^*
=
(I_k+jQ_k)
(I_{k-1}-jQ_{k-1})
$$

得到：

$$
P_kP_{k-1}^*
=
I_kI_{k-1}
+
Q_kQ_{k-1}
+
j
\left(
Q_kI_{k-1}
-
I_kQ_{k-1}
\right)
$$

定义：

#### Dot

$$
Dot
=
I_kI_{k-1}
+
Q_kQ_{k-1}
$$

#### Cross

$$
Cross
=
Q_kI_{k-1}
-
I_kQ_{k-1}
$$

于是：

$$
P_kP_{k-1}^*
=
Dot+jCross
$$

并且：

$$
Dot
=
A_kA_{k-1}\cos(\Delta\theta)
$$

$$
Cross
=
A_kA_{k-1}\sin(\Delta\theta)
$$

所以：

$$
\boxed{
\Delta\theta
=
\operatorname{atan2}(Cross,Dot)
}
$$

进一步：

$$
\boxed{
\Delta f
=
\frac{
\operatorname{atan2}(Cross,Dot)
}
{2\pi T}
}
$$

这就是典型的四象限 FLL 鉴别器。

---

### 7. Dot 和 Cross 的几何意义

两个 Prompt 向量：

$$
\vec P_{k-1}
$$

和：

$$
\vec P_k
$$

Dot 反映两个向量“有多同向”：

$$
Dot\propto\cos\Delta\theta
$$

Cross 反映从前一个向量转到后一个向量的旋转方向和大小：

$$
Cross\propto\sin\Delta\theta
$$

所以联合使用：

$$
\operatorname{atan2}(Cross,Dot)
$$

就能恢复完整的相对旋转角。

---

### 8. 一个数值例子

假设：

$$
P_{k-1}
$$

角度：

$$
20^\circ
$$

而：

$$
P_k
$$

角度：

$$
65^\circ
$$

所以：

$$
\Delta\theta=45^\circ
$$

若：

$$
T=1\text{ ms}
$$

则：

$$
\Delta f
=
\frac{45}{360\times0.001}
=
125\text{ Hz}
$$

因此 FLL discriminator 输出可以理解为：

```text
Residual Frequency ≈ +125 Hz
```

Loop Filter 再根据这个结果修改 Carrier NCO FCW。

---

### 9. 为什么 FLL 比 PLL 更适合较大的 Frequency Error

PLL 直接看当前相位：

$$
\Delta\phi
$$

如果 residual frequency 很大，Prompt 向量每个积分周期都会转很多角度，PLL 很容易跑出稳定线性区。

FLL 则主要看：

$$
\Delta\theta
=
\theta_k-\theta_{k-1}
$$

所以只要相邻两次相位变化没有发生不可辨识的绕圈，FLL 就可以直接估计频率偏差。

因此：

> FLL 更适合 Acquisition 后 residual Doppler 仍然比较大的拉入阶段。

---

### 10. FLL 的无模糊频率范围

四象限：

$$
\operatorname{atan2}(Cross,Dot)
$$

只能返回：

$$
-\pi<\Delta\theta\le\pi
$$

因此要求相邻两次 Prompt 的真实相位变化满足：

$$
|\Delta\theta|<\pi
$$

又因为：

$$
\Delta\theta
=
2\pi\Delta fT
$$

所以：

$$
|2\pi\Delta fT|<\pi
$$

得到：

$$
\boxed{
|\Delta f|
<
\frac{1}{2T}
}
$$

如果：

$$
T=1\text{ ms}
$$

则：

$$
|\Delta f|<500\text{ Hz}
$$

所以四象限 FLL 的理论无模糊范围约为：

$$
-500\sim+500\text{ Hz}
$$

---

### 11. 为什么更新周期越长，FLL 范围越小

因为：

$$
|\Delta f|_{max}
=
\frac{1}{2T}
$$

例如：

#### $T=1$ ms

$$
|\Delta f|_{max}=500\text{ Hz}
$$

#### $T=5$ ms

$$
|\Delta f|_{max}=100\text{ Hz}
$$

#### $T=10$ ms

$$
|\Delta f|_{max}=50\text{ Hz}
$$

所以：

> 长积分虽然提高 SNR，但同时缩小 FLL 的无模糊 Frequency Error 范围。

这和前面学习相干积分时看到的“噪声性能和动态能力折中”是同一条逻辑。

---

### 12. 相位增量和 residual frequency 图

图文件：

```text
GNSS_Tracking_第06章_FLL_频率与相位增量.png
```

对于：

$$
T=1\text{ ms}
$$

有：

$$
\Delta\theta[^\circ]
=
360^\circ\Delta fT
$$

所以：

- $+125$ Hz → $+45^\circ$
- $+250$ Hz → $+90^\circ$
- $+500$ Hz → $+180^\circ$

超过 $\pm180^\circ$ 后，四象限相位差会发生 wrapping，频率估计出现模糊。

---

### 13. BPSK Data Bit 对 FLL 的影响

Data Channel：

$$
P_k
=
A_kd_ke^{j\theta_k}
$$

其中：

$$
d_k\in\{+1,-1\}
$$

如果相邻两个 Prompt 之间 data bit 没有翻转：

$$
d_kd_{k-1}=+1
$$

没有问题。

如果发生 bit 翻转：

$$
d_kd_{k-1}=-1
$$

相当于复向量额外增加：

$$
\pi
$$

相位。

因此四象限 `atan2` FLL 可能把 data bit 翻转误认为巨大的 Frequency Error。

---

### 14. 两象限 FLL

可以改用：

$$
D
=
\arctan\left(
\frac{Cross}{Dot}
\right)
$$

因为 data bit 翻转时：

$$
Dot\rightarrow-Dot
$$

同时：

$$
Cross\rightarrow-Cross
$$

所以：

$$
\frac{-Cross}{-Dot}
=
\frac{Cross}{Dot}
$$

因此 BPSK data bit 的符号被消掉。

这与 Costas PLL 使用：

$$
\arctan(Q/I)
$$

消除 $180^\circ$ data 翻转，本质完全相同。

---

### 15. 两象限 FLL 的代价

`atan(Cross/Dot)` 的输出只能唯一表示：

$$
-\frac{\pi}{2}
<
\Delta\theta
<
\frac{\pi}{2}
$$

因此：

$$
|2\pi\Delta fT|
<
\frac{\pi}{2}
$$

得到：

$$
\boxed{
|\Delta f|
<
\frac{1}{4T}
}
$$

如果：

$$
T=1\text{ ms}
$$

则：

$$
|\Delta f|<250\text{ Hz}
$$

所以：

| FLL 鉴别器 | 是否抗未知 BPSK bit | 理论相位范围 | $T=1$ ms 时频率范围 |
|---|---|---:|---:|
| `atan2(Cross,Dot)` | 否 | $\pm180^\circ$ | $\pm500$ Hz |
| `atan(Cross/Dot)` | 是 | $\pm90^\circ$ | $\pm250$ Hz |

核心折中：

> 抗 data bit 翻转，代价是 Frequency Pull-In Range 变小。

---

### 16. Bit 翻转发生在积分窗口内部怎么办

上面讨论的是：

> 两个完整 Prompt 之间符号不同。

但如果导航 bit 恰好在一个 coherent integration 窗口内部翻转：

```text
前半段：+1
后半段：-1
```

那么积分本身就会出现信号抵消。

这不是 FLL discriminator 之后再处理能够完全恢复的。

因此：

- Data bit synchronization；
- data wipeoff；
- Pilot Channel；
- 合理的 coherent integration boundary

都非常重要。

---

### 17. FLL Loop Filter

FLL discriminator 输出本质上已经是：

$$
\Delta f
$$

或与 Frequency Error 成比例的量。

但它仍然包含：

- Thermal Noise；
- Multipath；
- Interference；
- Discriminator Nonlinearity。

所以仍需要：

```text
FLL Discriminator
      ↓
Loop Filter
      ↓
Frequency Correction
      ↓
Carrier NCO
```

FLL bandwidth 仍然控制：

$$
\boxed{
\text{动态响应}
\leftrightarrow
\text{频率噪声}
}
$$

---

### 18. FLL Bandwidth

#### 较宽 FLL

优点：

- Frequency Pull-In 快；
- 能跟踪较快 Doppler 变化；
- Acquisition 后收敛更快。

缺点：

- Frequency Estimate 更抖；
- 更多 noise 进入 Carrier NCO。

#### 较窄 FLL

优点：

- Frequency Estimate 更稳定；
- 噪声更低。

缺点：

- 拉入速度慢；
- 高频动态跟不上。

因此 Tracking 启动阶段可能使用较宽 FLL，稳定后逐步减小带宽或切换到 PLL。

---

### 19. Acquisition 为什么不能直接完全替代 FLL

Acquisition 通常通过频率搜索得到粗 Doppler：

$$
\hat f_{acq}
$$

例如搜索步长：

$$
500\text{ Hz}
$$

捕获得到：

$$
-2500\text{ Hz}
$$

但真实可能：

$$
-2375\text{ Hz}
$$

则 residual：

$$
+125\text{ Hz}
$$

Acquisition 已经足够告诉 Tracking：

> 卫星大概在哪个 Doppler bin。

FLL 接下来负责：

> 把 residual Doppler 从几百/几十 Hz 继续拉到几 Hz 甚至更小。

然后 PLL 才适合进行高精度 Phase Lock。

---

### 20. Acquisition → FLL → PLL

典型过程：

```text
Acquisition
    ↓
粗 Doppler
例如 residual = 150 Hz
    ↓
FLL
    ↓
residual = 20 Hz
    ↓
FLL-Assisted PLL
    ↓
residual = 2 Hz
    ↓
PLL
    ↓
高精度 Carrier Phase Tracking
```

可以把三者理解为：

#### Acquisition

找到大概位置。

#### FLL

把 Carrier 的“转速”拉准。

#### PLL

把 Carrier 的“角度”精确锁准。

---

### 21. FLL-Assisted PLL

FLL 和 PLL 不一定必须完全先后切换。

也可以同时工作：

```text
                ┌→ FLL Discriminator → FLL Filter ─┐
Prompt I/Q ─────┤                                  ├→ Carrier NCO
                └→ PLL Discriminator → PLL Filter ─┘
```

直觉上：

- FLL 负责较大的 Frequency/Dynamic Error；
- PLL 负责精细 Phase Error。

因此在高动态条件下：

> FLL 帮 PLL 把大部分频率动态承担掉，PLL 不容易因为快速相位变化而失锁。

这就是 FLL-Assisted PLL。

---

### 22. FLL 与 PLL 的作用区间

可以粗略理解：

```text
Residual Frequency 很大
          ↓
         FLL
          ↓
Residual Frequency 变小
          ↓
    FLL + PLL
          ↓
Phase Error 足够小
          ↓
         PLL
```

实际接收机会根据：

- Frequency discriminator variance；
- Phase discriminator variance；
- $C/N_0$；
- Lock indicator；
- 动态状态

决定何时切换或如何组合。

---

### 23. 为什么 FLL 最终不能替代 PLL

FLL 主要保证：

$$
\Delta f\approx0
$$

但即使：

$$
\Delta f=0
$$

仍可能存在一个固定：

$$
\Delta\phi=30^\circ
$$

因为：

$$
\frac{d\Delta\phi}{dt}=0
$$

FLL 会认为：

> Frequency 已经完全正确。

但 Carrier Phase 仍然偏了 30°。

所以：

> FLL 能锁 Frequency，但不能保证高精度 Phase Lock。

要得到：

- 高质量 Carrier Phase；
- ADR；
- 相位差测量；

最终仍然需要 PLL。

---

### 24. FLL 与 Carrier Phase Measurement

FLL 拉频阶段，Carrier NCO phase 仍然在持续累加。

但由于 Phase Error 可能较大、噪声较高，Carrier Phase Measurement 精度通常不如稳定 PLL 状态。

因此高精度测量通常要求：

```text
FLL Pull-In 完成
      ↓
PLL Stable Lock
      ↓
Carrier Phase / ADR Reliable
```

---

### 25. FLL 一个完整数值流程

假设 Acquisition 给：

$$
\hat f=-2500\text{ Hz}
$$

真实：

$$
f=-2375\text{ Hz}
$$

Residual：

$$
\Delta f=125\text{ Hz}
$$

若：

$$
T=1\text{ ms}
$$

相邻 Prompt 相位变化：

$$
\Delta\theta
=
360^\circ\times125\times0.001
=
45^\circ
$$

于是：

$$
Dot
\propto
\cos45^\circ
$$

$$
Cross
\propto
\sin45^\circ
$$

所以：

$$
\operatorname{atan2}(Cross,Dot)=45^\circ
$$

再换算：

$$
\hat{\Delta f}
=
\frac{45^\circ}
{360^\circ\times0.001}
=
125\text{ Hz}
$$

Loop Filter 输出修正量，例如：

$$
+90\text{ Hz}
$$

新 NCO：

$$
-2500+90
=
-2410\text{ Hz}
$$

下一次 residual：

$$
35\text{ Hz}
$$

FLL 再继续修正。

最终：

$$
\Delta f\rightarrow0
$$

---

### 26. 本章完整数据流

```text
Prompt P[k-1]
      +
Prompt P[k]
      ↓
Dot / Cross
      ↓
Phase Increment Δθ
      ↓
Frequency Discriminator
      ↓
Residual Frequency Δf
      ↓
FLL Loop Filter
      ↓
Carrier Frequency / FCW Correction
      ↓
Carrier NCO
      ↓
下一组 Prompt
```

---

### 27. 本章关键公式

相邻 Prompt：

$$
P_k=I_k+jQ_k
$$

$$
P_{k-1}=I_{k-1}+jQ_{k-1}
$$

复共轭乘积：

$$
P_kP_{k-1}^*
=
Dot+jCross
$$

Dot：

$$
Dot
=
I_kI_{k-1}
+
Q_kQ_{k-1}
$$

Cross：

$$
Cross
=
Q_kI_{k-1}
-
I_kQ_{k-1}
$$

相位增量：

$$
\Delta\theta
=
\operatorname{atan2}(Cross,Dot)
$$

频率误差：

$$
\Delta f
=
\frac{\Delta\theta}{2\pi T}
$$

四象限 FLL 无模糊范围：

$$
|\Delta f|
<
\frac{1}{2T}
$$

两象限 data-insensitive FLL：

$$
D
=
\arctan\left(\frac{Cross}{Dot}\right)
$$

其无模糊范围：

$$
|\Delta f|
<
\frac{1}{4T}
$$

---

### 28. 本章关键结论

1. FLL 本质上通过相邻 Prompt 的相位变化估计 residual frequency。
2. Frequency Error 是 Phase Error 对时间的变化率。
3. 复共轭乘积 $P_kP_{k-1}^*$ 的相位直接等于两个 Prompt 的相位差。
4. Dot 对应 $\cos\Delta\theta$，Cross 对应 $\sin\Delta\theta$。
5. `atan2(Cross,Dot)` 可以恢复四象限相位增量。
6. FLL 的无模糊 Frequency Range 与 Prompt 间隔 $T$ 成反比。
7. 积分/更新周期越长，FLL 的频率无模糊范围越小。
8. 四象限 FLL 会受到未知 BPSK data bit 翻转影响。
9. 两象限 `atan(Cross/Dot)` 可以消除 180° data bit 翻转，但无模糊频率范围缩小一半。
10. FLL 更适合 Acquisition 后较大的 residual Doppler 拉入。
11. PLL 更适合最终高精度 Carrier Phase Tracking。
12. FLL-Assisted PLL 可以让 FLL 承担较大频率动态，让 PLL 专注精细相位跟踪。
13. FLL 最终不能替代 PLL，因为频率锁定并不等于相位误差为零。
14. Carrier Phase / ADR 的高质量输出通常依赖稳定 PLL Lock。

---

### 29. 下一章

下一章进入：

## 第 7 章｜Loop Filter：环路阶数、Noise Bandwidth、Natural Frequency 与数字实现

重点包括：

- 为什么鉴别器后面一定需要 Loop Filter；
- 一阶、二阶、三阶环到底差在哪里；
- Noise Bandwidth $B_n$ 与 Natural Frequency $\omega_n$ 的关系；
- Damping Ratio $\zeta$ 是什么；
- 如何从 `1 ms update + 指定 bandwidth` 算数字滤波器系数；
- PLL / DLL / FLL 的 bandwidth 为什么不同；
- Loop Filter 输出如何最终变成 NCO FCW。


---

<a id="chapter-07"></a>

## 第 7 章｜Loop Filter：环路阶数、Noise Bandwidth、Natural Frequency 与数字实现

### 1. 本章目标

本章重点理解：

- Loop Filter 为什么是 Tracking 闭环的核心；
- Noise Bandwidth $B_n$、Natural Frequency $\omega_n$、Damping Ratio $\zeta$ 分别表示什么；
- 一阶、二阶、三阶环路的区别；
- 为什么 $B_n$ 不是简单的“截止频率”；
- $T_{coh}$、$T_{update}$ 与 $B_n$ 如何共同约束环路；
- 如何从指定的 $B_n$、$\zeta$ 得到二阶环路参数；
- 数字 Loop Filter 如何把鉴别器误差转换成 NCO frequency / FCW correction。

---

### 2. Loop Filter 在 Tracking 中的位置

无论 DLL、PLL 还是 FLL，结构都可以抽象成：

```text
真实信号
   ↓
Correlator
   ↓
Discriminator
   ↓
Raw Tracking Error
   ↓
Loop Filter
   ↓
NCO Control
   ↓
NCO
   ↓
Local Replica
   └──────────────反馈
```

鉴别器输出可以粗略表示成：

$$
e[k]
=
e_{\text{true}}[k]
+
n[k]
+
b[k]
$$

其中：

- $e_{\text{true}}$：真实跟踪误差；
- $n[k]$：热噪声等随机扰动；
- $b[k]$：多径、干扰、模型误差等偏置。

Loop Filter 的任务不是简单“把变化变平滑”，而是决定：

> 哪些变化应该由 NCO 跟随，哪些变化应该被视为噪声而尽量忽略。

---

### 3. Noise Bandwidth $B_n$

Tracking 中常用：

$$
B_n
$$

表示 Loop Noise Bandwidth（等效噪声带宽）。

它描述的是：

> 环路对输入噪声允许通过多少。

概念上可以写成：

$$
B_n
\propto
\int_0^\infty
|H_n(j2\pi f)|^2df
$$

其中：

$$
H_n
$$

是噪声到输出之间的闭环传递函数。

因此 $B_n$ 并不等同于普通低通滤波器的：

$$
f_{-3dB}
$$

两者有关联，但不是同一个定义。

#### $B_n$ 较大

意味着：

- 更多高频误差可以进入控制量；
- 环路响应更快；
- 动态跟踪能力更强；
- Noise Jitter 更大。

#### $B_n$ 较小

意味着：

- 更多快速随机扰动被抑制；
- 稳态测量更平稳；
- 但环路响应更慢；
- Dynamic Stress Error 更容易增大。

核心折中：

$$
\boxed{
\text{Dynamic Performance}
\leftrightarrow
\text{Noise Performance}
}
$$

---

### 4. Natural Frequency $\omega_n$

对于标准二阶系统，闭环特征多项式通常写成：

$$
s^2
+
2\zeta\omega_n s
+
\omega_n^2
$$

其中：

$$
\omega_n
$$

为 Natural Frequency（自然频率）。

直觉上：

> $\omega_n$ 主要决定系统“动作有多快”。

$\omega_n$ 越大：

- 收敛更快；
- 环路能追踪更快变化；
- 通常对应更大的 Noise Bandwidth。

---

### 5. Damping Ratio $\zeta$

$\zeta$ 为阻尼比。

它主要决定瞬态过程：

#### $\zeta$ 很小

环路欠阻尼明显：

- 响应快；
- 但容易振荡；
- 超调明显。

#### $\zeta\approx0.707$

是经典二阶系统中常见的折中设置：

- 响应较快；
- 超调可控；
- 不容易产生明显振荡。

#### $\zeta>1$

过阻尼：

- 基本不振荡；
- 但响应较慢。

配套图：

```text
GNSS_Tracking_第07章_阻尼系数响应.png
```

---

### 6. $B_n$、$\omega_n$、$\zeta$ 的关系

对于常见标准二阶 GNSS Tracking Loop，常使用：

$$
\boxed{
\omega_n
=
B_n
\frac{8\zeta}
{4\zeta^2+1}
}
$$

等价地：

$$
\boxed{
B_n
=
\omega_n
\frac{4\zeta^2+1}
{8\zeta}
}
$$

注意：

- $B_n$ 是 Noise Bandwidth；
- $\omega_n$ 是自然频率参数；
- 二者不是同一个物理量；
- 给定 $\zeta$ 后，两者可以互相换算。

当：

$$
\zeta=0.707
$$

有：

$$
\omega_n
\approx
1.886B_n
$$

例如：

$$
B_n=10\text{ Hz}
$$

则：

$$
\omega_n
\approx18.86\text{ rad/s}
$$

对应自然频率：

$$
f_n
=
\frac{\omega_n}{2\pi}
\approx3.0\text{ Hz}
$$

所以不要把：

$$
B_n=10\text{ Hz}
$$

理解成：

> 系统的自然频率就是 10 Hz。

---

### 7. Bandwidth 对响应速度的影响

在相同阻尼下：

$$
B_n\uparrow
\Rightarrow
\omega_n\uparrow
$$

所以环路响应会变快。

配套图：

```text
GNSS_Tracking_第07章_带宽响应对比.png
```

图中展示相同：

$$
\zeta=0.707
$$

但不同 $B_n$ 的标准二阶响应。

总体趋势：

```text
Bn 小
 ↓
响应慢
噪声低

Bn 大
 ↓
响应快
噪声高
```

---

### 8. 一阶、二阶、三阶到底是什么

Tracking Loop 的“阶数”通常指整个闭环系统的动态阶数，而不是只看某个软件 Filter 有几个系数。

对于 PLL 可以先用下面的直觉理解。

#### 一阶 PLL

主要可以对固定 Phase Error 建立稳态跟踪。

如果输入存在持续 Frequency Error，相位会持续变化，一阶环难以做到零稳态误差。

#### 二阶 PLL

内部多一个积分状态。

因此可以对恒定 Frequency Offset：

$$
\Delta f=\text{constant}
$$

保持零稳态 Phase Tracking Error。

#### 三阶 PLL

进一步增加动态状态，可以更好地跟踪：

$$
\frac{d\Delta f}{dt}
=
\text{constant}
$$

即持续 Frequency Ramp，对应更强的加速度动态。

所以可以粗略记成：

```text
1阶：Phase
2阶：Phase + Frequency
3阶：Phase + Frequency + Frequency Rate
```

这是理解环路阶数最重要的直觉。

---

### 9. 为什么高阶环不是一定更好

增加阶数意味着：

- 能跟踪更高阶动态；
- 但状态更多；
- 稳定性设计更复杂；
- 对参数误差更敏感；
- Noise Response 也更复杂。

所以实际设计应根据目标动态选择最低够用的阶数。

例如：

- DLL 常使用较低阶；
- Carrier PLL 常见二阶或三阶；
- FLL 常见一阶或二阶。

---

### 10. 二阶 PLL 为什么通常是 PI 结构

把 PLL 简化为：

```text
Phase Error
    ↓
Kd
    ↓
Loop Filter
    ↓
K0 / s
    ↓
Carrier Phase
```

NCO 从 Frequency 到 Phase 本身就是一个积分器：

$$
\frac{K_0}{s}
$$

若 Loop Filter 采用 PI：

$$
F(s)
=
K_p+\frac{K_i}{s}
$$

则开环中包含足够的积分能力，可以形成典型二阶 Tracking Loop。

闭环特征多项式：

$$
s^2
+
K_dK_0K_p s
+
K_dK_0K_i
$$

与标准二阶形式：

$$
s^2
+
2\zeta\omega_n s
+
\omega_n^2
$$

对比可得：

$$
\boxed{
K_p
=
\frac{2\zeta\omega_n}
{K_dK_0}
}
$$

$$
\boxed{
K_i
=
\frac{\omega_n^2}
{K_dK_0}
}
$$

这一步非常重要：

> 一旦 $B_n$ 和 $\zeta$ 定下来，就可以先得到 $\omega_n$，再得到 Loop Filter 的比例和积分参数。

---

### 11. 为什么还要考虑 $K_d$ 和 $K_0$

前面的公式不是直接拿来写寄存器。

因为实际系统有：

#### Discriminator Gain

$$
K_d
$$

例如鉴别器输出单位可能是：

- rad；
- cycle；
- chip；
- 归一化无量纲值。

#### NCO Gain

$$
K_0
$$

NCO 输入也可能使用：

- Hz；
- rad/s；
- FCW；
- 定点 LSB。

所以真正的 Loop Filter coefficient 必须把：

$$
K_dK_0
$$

包含进去。

如果算法中已经做了单位归一化：

$$
K_dK_0=1
$$

公式才会简化成：

$$
K_p=2\zeta\omega_n
$$

$$
K_i=\omega_n^2
$$

---

### 12. 一个 10 Hz PLL 的例子

假设：

$$
B_n=10\text{ Hz}
$$

$$
\zeta=0.707
$$

使用标准二阶换算：

$$
\omega_n
=
10
\times
\frac{8\times0.707}
{4\times0.707^2+1}
$$

得到约：

$$
\omega_n\approx18.86\text{ rad/s}
$$

若暂时使用归一化：

$$
K_dK_0=1
$$

则：

$$
K_p
=
2\times0.707\times18.86
\approx26.7
$$

$$
K_i
=
18.86^2
\approx355.7
$$

这里的数字是归一化控制系统参数。

真实芯片实现还必须继续映射到：

- Discriminator output scale；
- Carrier frequency unit；
- FCW LSB；
- 固定位宽。

---

### 13. 数字 Loop Filter 怎么实现

假设环路每：

$$
T_{update}
$$

更新一次。

一个最直观的离散 PI 实现：

#### 积分状态

$$
u_I[k]
=
u_I[k-1]
+
K_iT_{update}e[k]
$$

#### 总输出

$$
u[k]
=
K_pe[k]
+
u_I[k]
$$

然后：

$$
u[k]
$$

作为 Carrier Frequency Correction。

数据流：

```text
e[k]
 │
 ├──────────×Kp────────────┐
 │                         │
 └─×Ki×Tupdate─→ Σ state ──┤
                           ↓
                          u[k]
                           ↓
                       Carrier NCO
```

生产实现可能采用更精确的离散化方式，例如双线性变换等，但核心 PI 状态思想相同。

---

### 14. $T_{update}$ 为什么进入数字系数

连续积分：

$$
\int e(t)dt
$$

数字系统要变成求和：

$$
\sum e[k]T_{update}
$$

所以积分支路中自然出现：

$$
K_iT_{update}
$$

这说明：

> 即使 $B_n$ 不变，只要 Loop Update Period 改变，数字 Filter coefficient 也必须重新计算。

例如：

$$
T_{update}:1\text{ ms}\rightarrow20\text{ ms}
$$

不能直接继续沿用原来的数字系数。

---

### 15. $T_{coh}$、$T_{disc}$ 和 $T_{update}$ 再区分一次

#### $T_{coh}$

一次相干积分多长。

决定单次 Correlator Measurement 的 SNR 和动态损失。

#### $T_{disc}$

多久产生一次 Discriminator Measurement。

#### $T_{update}$

多久真正更新一次 Loop Filter / NCO control。

简单实现可能：

$$
T_{coh}
=
T_{disc}
=
T_{update}
$$

但复杂系统完全可以：

```text
每 1 ms：
Correlator output

每 1 ms：
Discriminator measurement

累计 / Filter 20 次

每 20 ms：
更新 NCO control
```

因此这三个时间参数不能混为一谈。

---

### 16. $B_n$ 与 $T_{update}$ 的约束

数字环路的误差观测采样率：

$$
f_{update}
=
\frac{1}{T_{update}}
$$

理论上首先受到离散采样约束。

工程上还希望：

$$
\boxed{
B_nT_{update}\ll1
}
$$

也就是：

> 一个 Loop time constant 内应该有足够多次控制更新。

如果：

$$
T_{update}
$$

太长，而：

$$
B_n
$$

又设计得很宽，则离散化效应会非常明显，甚至破坏设计的稳定性和瞬态性能。

---

### 17. 为什么长积分时常配合 Narrow Bandwidth

弱信号时常：

$$
T_{coh}\uparrow
$$

这样：

- Correlator SNR 提高；
- Measurement Noise 降低。

如果同时：

$$
T_{update}\uparrow
$$

那么环路反应速度也下降。

因此实际 Tracking 状态机里经常看到：

```text
强信号 / Pull-in
    ↓
短积分
较宽 Loop Bandwidth

稳定 Tracking
    ↓
中等积分
中等 Bandwidth

弱信号
    ↓
长积分
Narrow Bandwidth
```

但这只是常见联动趋势，不是固定的一一对应关系。

真正决定 Bandwidth 的核心仍然是：

$$
\text{Noise Requirement}
+
\text{Dynamic Requirement}
$$

---

### 18. DLL、PLL、FLL 为什么 Bandwidth 不一样

#### DLL

Code Doppler 绝对变化较慢，因此通常可使用较窄 bandwidth。

#### PLL

Carrier phase 对 Doppler / Dynamic 更敏感，因此 bandwidth 通常比 DLL 宽。

#### FLL

Pull-in 阶段需要快速压低较大的 residual Doppler，常使用相对较宽 bandwidth。

所以常见直觉：

```text
DLL：窄
PLL：中等
FLL Pull-in：较宽
```

但具体数值必须由：

- Signal Dynamics；
- $C/N_0$；
- Coherent Integration；
- Aiding；
- Loop Order

共同决定。

---

### 19. Carrier Aiding 为什么能让 DLL 变窄

Code Doppler 可以利用 Carrier Doppler 预测：

$$
\Delta f_{code}
\approx
f_{code}
\frac{\Delta f_{carrier}}
{f_{carrier}}
$$

如果大部分 Code Dynamics 已经通过 Carrier Aiding 补偿，DLL 只需要跟 residual code error。

因此：

$$
B_{DLL}
$$

可以进一步减小，从而降低 Code Tracking Jitter。

---

### 20. Loop Filter 输出如何变成 FCW

假设 Loop Filter 最终输出的是频率修正：

$$
\Delta f
$$

N 位 Carrier NCO：

$$
FCW
=
\frac{f}{f_s}2^N
$$

所以：

$$
\boxed{
\Delta FCW
=
\frac{\Delta f}{f_s}2^N
}
$$

例如：

```text
Discriminator
   ↓
Phase Error
   ↓
PI Loop Filter
   ↓
Δf = +3.2 Hz
   ↓
换算 ΔFCW
   ↓
Carrier NCO
```

因此 Loop Filter 的算法单位与 NCO 硬件单位之间必须有明确的 scaling。

---

### 21. Loop Filter 与“抗干扰”再澄清一次

Loop Filter 能抑制：

- Discriminator output 中的高频随机 noise；
- 干扰造成的一部分快速 error fluctuation。

但如果干扰造成的是稳定低频 bias：

$$
e[k]\approx\text{constant bias}
$$

且位于 Loop passband 内，

Loop Filter 会把它当成真实 Tracking Error。

所以 Loop Filter 不能替代：

- 前端抗干扰；
- 空域 Beamforming / Nulling；
- 干扰检测；
- Multipath mitigation。

---

### 22. 图：阻尼比对响应的影响

配套图片：

```text
GNSS_Tracking_第07章_阻尼系数响应.png
```

这是标准二阶系统的示意：

- $\zeta=0.3$：振荡和 overshoot 明显；
- $\zeta=0.707$：较均衡；
- $\zeta=1.2$：响应平稳但较慢。

该图用于帮助理解 $\zeta$，并不是某一颗具体 GNSS 芯片的实测响应。

---

### 23. 图：Bandwidth 对响应的影响

配套图片：

```text
GNSS_Tracking_第07章_带宽响应对比.png
```

保持：

$$
\zeta=0.707
$$

改变 $B_n$。

可以看到：

$$
B_n\uparrow
$$

时响应明显加快。

这就是更宽 Tracking Loop 可以提高动态响应速度的直观表现。

---

### 24. 本章完整参数关系

可以把设计流程整理成：

```text
目标动态
   +
目标 Noise Jitter
   ↓
选择 Loop Order
   ↓
选择 Bn
   ↓
选择 ζ
   ↓
计算 ωn
   ↓
结合 Kd / K0
   ↓
得到连续 Loop Filter 参数
   ↓
结合 Tupdate
   ↓
离散化
   ↓
数字 Filter Coefficients
   ↓
Frequency Correction
   ↓
换算 FCW
   ↓
NCO
```

---

### 25. 本章关键公式

标准二阶特征方程：

$$
s^2
+
2\zeta\omega_n s
+
\omega_n^2
$$

Noise Bandwidth 与 Natural Frequency：

$$
\omega_n
=
B_n
\frac{8\zeta}
{4\zeta^2+1}
$$

PI Filter：

$$
F(s)
=
K_p+\frac{K_i}{s}
$$

参数匹配：

$$
K_p
=
\frac{2\zeta\omega_n}
{K_dK_0}
$$

$$
K_i
=
\frac{\omega_n^2}
{K_dK_0}
$$

数字积分状态：

$$
u_I[k]
=
u_I[k-1]
+
K_iT_{update}e[k]
$$

总控制：

$$
u[k]
=
K_pe[k]+u_I[k]
$$

NCO：

$$
FCW
=
\frac{f}{f_s}2^N
$$

因此：

$$
\Delta FCW
=
\frac{\Delta f}{f_s}2^N
$$

---

### 26. 本章关键结论

1. Loop Filter 决定 Tracking Loop 对动态和噪声的取舍。
2. $B_n$ 是等效噪声带宽，不等同于普通滤波器的 -3 dB 截止频率。
3. $\omega_n$ 主要决定环路响应速度。
4. $\zeta$ 主要决定瞬态的振荡与超调特性。
5. 给定 $\zeta$ 后，$B_n$ 和 $\omega_n$ 可以进行标准二阶换算。
6. 二阶 PLL 常可理解为 PI Loop Filter 加 NCO 积分器。
7. $K_d$ 和 $K_0$ 决定理论控制参数如何映射到实际数字单位。
8. $T_{update}$ 改变后，数字 Loop Filter coefficient 必须重新设计。
9. $T_{coh}$、$T_{disc}$、$T_{update}$ 不一定相等。
10. 工程上通常希望 $B_nT_{update}\ll1$，使离散控制更新足够快。
11. Loop Order 越高，能跟踪更高阶动态，但实现与稳定性设计更复杂。
12. DLL、PLL、FLL 的合适 bandwidth 不同，因为它们面对的动态和噪声不同。
13. Carrier Aiding 可以显著降低 DLL 自身需要承担的动态。
14. Loop Filter 输出最终需要经过 scaling 映射为 NCO frequency / FCW。
15. Loop Filter 只能抑制一部分误差扰动，不能替代真正的抗干扰算法。

---

### 27. 下一章

下一章进入：

## 第 8 章｜Carrier NCO、Phase Accumulator、FCW 与连续载波相位

重点包括：

- NCO 如何真正生成数字 Carrier；
- Phase Accumulator 为什么会不断 wrap；
- 30-bit / 32-bit NCO 的 FCW 如何计算；
- Phase Wrap 为什么不会导致 Tracking 丢掉整周；
- Fractional Phase 与 Integer Cycle Counter 的关系；
- Carrier FCW 如何累加得到连续 Carrier Phase；
- Doppler、Carrier Phase、ADR 三者如何在 TRK 内部真正连接起来。


---

<a id="chapter-08"></a>

## 第 8 章｜Carrier NCO、Phase Accumulator、FCW 与连续载波相位

### 1. 本章目标

本章重点理解：

- Carrier NCO 在 Tracking 中到底做什么；
- FCW 如何决定数字载波频率；
- Phase Accumulator 为什么每个 sample 都在累加；
- N-bit Phase Accumulator 为什么必然会 wrap；
- NCO wrap 为什么不会导致载波生成错误；
- 为什么“wrapped phase”本身不能表示连续 Carrier Phase；
- Integer Cycle Counter / Unwrapped Phase 如何维护整数周；
- PLL Loop Filter 输出如何修改 FCW；
- FCW、Doppler、Carrier Phase、ADR 之间如何连接；
- NCO wrap 与真正的 Cycle Slip 有什么区别。

---

### 2. Carrier NCO 在 Tracking 中的位置

Carrier Tracking 的简化链路：

```text
Input Samples
      ↓
Local Carrier cos/sin  ← Carrier NCO
      ↓
Carrier Wipeoff
      ↓
Prompt I/Q
      ↓
PLL / FLL Discriminator
      ↓
Loop Filter
      ↓
Frequency / FCW Correction
      ↓
Carrier NCO
```

所以 Carrier NCO 的任务是：

> 根据当前 Tracking 估计的载波频率和相位，连续产生本地 `cos` / `sin`，用于 Carrier Wipeoff。

---

### 3. NCO 的核心组成

典型数字 NCO / DDS 可以抽象为：

```text
       FCW
        ↓
Phase Accumulator
        ↓
Phase Word
        ↓
LUT / CORDIC
        ↓
cos(phase), sin(phase)
```

其中最关键的是：

> Phase Accumulator。

因为数字正弦波本质上是不断推进相位，再对相位取 `sin` / `cos`。

---

### 4. Phase Accumulator

如果使用“cycle”为单位，则可以写：

$$
\phi[n+1]
=
\phi[n]
+
\frac{f}{f_s}
$$

其中：

- $f$：NCO 输出频率；
- $f_s$：NCO 时钟 / sample rate；
- $\phi$：以 cycle 为单位的相位。

如果使用 rad：

$$
\theta[n+1]
=
\theta[n]
+
2\pi\frac{f}{f_s}
$$

也就是说每个 sample 都增加固定相位步进。

---

### 5. N-bit NCO 与 FCW

实际硬件通常不会直接存浮点：

$$
\frac{f}{f_s}
$$

而是使用 N-bit Phase Accumulator。

定义：

$$
2^N
$$

代表一个完整的：

$$
2\pi
$$

或者：

$$
1\text{ cycle}
$$

那么：

$$
\boxed{
FCW
=
\frac{f}{f_s}2^N
}
$$

Phase Accumulator：

$$
\boxed{
PHA[n+1]
=
PHA[n]+FCW
\pmod{2^N}
}
$$

其中：

- `PHA`：Phase Accumulator；
- `FCW`：Frequency Control Word。

---

### 6. 为什么 FCW 决定频率

每个 sample：

$$
PHA
$$

增加：

$$
FCW
$$

一个完整 cycle 需要：

$$
2^N
$$

个 phase units。

所以每个 sample 前进的 cycle 数：

$$
\frac{FCW}{2^N}
$$

每秒有：

$$
f_s
$$

个 sample，因此每秒走的 cycle 数：

$$
f
=
f_s\frac{FCW}{2^N}
$$

即：

$$
\boxed{
f
=
\frac{FCW}{2^N}f_s
}
$$

反推就是：

$$
\boxed{
FCW
=
\frac{f}{f_s}2^N
}
$$

---

### 7. 30-bit NCO 的频率分辨率

对于 N-bit NCO，FCW 改变 1 LSB，对应频率变化：

$$
\boxed{
\Delta f_{LSB}
=
\frac{f_s}{2^N}
}
$$

例如：

$$
N=30
$$

$$
f_s=20\text{ MHz}
$$

则：

$$
\Delta f_{LSB}
=
\frac{20\times10^6}{2^{30}}
\approx0.0186\text{ Hz}
$$

即一个 FCW LSB 约对应：

$$
0.0186\text{ Hz}
$$

所以 30-bit NCO 在 20 MHz 时钟下已经有很细的频率分辨率。

---

### 8. FCW 与频率是线性关系

配套图：

```text
GNSS_Tracking_第08章_30bit_FCW频率关系.png
```

在：

$$
N=30,\quad f_s=20\text{ MHz}
$$

条件下：

$$
FCW
=
\frac{f}{20\times10^6}2^{30}
$$

因此 Frequency 与 FCW 是严格线性映射。

---

### 9. 为什么 Phase Accumulator 会 Wrap

假设 Phase Accumulator 只有：

$$
N
$$

bit。

最大值：

$$
2^N-1
$$

再加 FCW 超过：

$$
2^N
$$

后就会回到低位：

```text
... → 2^N-200 → 2^N-100 → 20 → 120 → ...
```

数学上：

$$
PHA[n]
=
\left(
PHA[0]
+
\sum_{i=0}^{n-1}FCW[i]
\right)
\bmod 2^N
$$

所以 wrap 是 NCO 正常工作的一部分。

---

### 10. 图：Wrapped Phase 与 Continuous Phase

配套图：

```text
GNSS_Tracking_第08章_NCO相位累加与Wrap.png
```

图里两条线分别代表：

- `Unwrapped phase`：真实累计走过多少 cycle；
- `Wrapped phase`：N-bit Phase Accumulator 实际存储的 modulo-one-cycle phase。

Wrapped phase 每到一个完整周期都会回到 0。

Continuous phase 则持续增长。

---

### 11. 为什么 Wrap 不影响 sin / cos

因为：

$$
\sin(\theta+2\pi N)
=
\sin\theta
$$

$$
\cos(\theta+2\pi N)
=
\cos\theta
$$

因此对于本地载波生成：

$$
10^\circ
$$

和：

$$
370^\circ
$$

完全等价。

所以 NCO 只需要知道：

$$
\boxed{
\theta\bmod2\pi
}
$$

就足够生成正确的 `cos` / `sin`。

因此：

> Phase Accumulator wrap 不意味着 Tracking 丢了一周。

它只是利用了正弦函数的周期性。

---

### 12. 但做 Carrier Phase / ADR 时就不够了

如果只保存：

$$
\theta\bmod2\pi
$$

那么：

$$
10^\circ
$$

和：

$$
370^\circ
$$

无法区分。

但 Carrier Phase Measurement 需要知道：

> 从开始到现在总共累计走过多少个 cycle。

因此需要：

$$
\boxed{
\Phi_{\text{continuous}}
=
N_{\text{cycle}}
+
\phi_{\text{fractional}}
}
$$

其中：

- $N_{\text{cycle}}$：整数周；
- $\phi_{\text{fractional}}\in[0,1)$：小数周相位。

---

### 13. 图：Integer Cycle + Fractional Phase

配套图：

```text
GNSS_Tracking_第08章_整数周与小数相位.png
```

图中：

- Fractional Phase 周期性从接近 1 回到 0；
- Integer Cycle Counter 每次增加 1；
- 两者相加得到 Continuous Phase。

即：

$$
\boxed{
\Phi=N+\phi_f
}
$$

---

### 14. Integer Cycle Counter 怎么得到

一种最直接的硬件实现是：

```text
Phase Accumulator
      ↓ overflow / carry
Integer Cycle Counter
```

当 Phase Accumulator 从：

$$
2^N-1
$$

跨回：

$$
0
$$

时，产生一次 overflow。

正向频率：

$$
N_{\text{cycle}}\leftarrow N_{\text{cycle}}+1
$$

负向频率则相反。

也可以不显式使用 overflow，而维护一个更宽位宽的：

> Extended / Unwrapped Phase Accumulator。

低 N bit 用来生成 sin/cos，高位自然表示累计整数周。

---

### 15. 一个更完整的数字表示

例如内部维护：

```text
| integer cycles | N-bit fractional phase |
```

那么可以理解为一个很宽的定点数：

$$
\Phi_{\text{ext}}
$$

低 N bit：

$$
\Phi_{\text{ext}}\bmod2^N
$$

送给 LUT / CORDIC 生成载波。

高位：

$$
\left\lfloor
\frac{\Phi_{\text{ext}}}{2^N}
\right\rfloor
$$

表示累计整数周。

这是一种非常自然的实现。

---

### 16. FCW 如果不断变化怎么办

Tracking 中 FCW 并不是恒定值。

PLL / FLL 每次更新都会改变：

$$
FCW[k]
$$

所以 Continuous Phase 是：

$$
\boxed{
\Phi[n]
=
\Phi[0]
+
\frac{1}{2^N}
\sum_{i=0}^{n-1}FCW[i]
}
$$

单位为 cycle。

这就是数字系统里：

> 对 Frequency 进行积分得到 Phase。

因此上一章说 NCO 是积分器，不只是控制理论上的抽象，它在硬件里就是一个真实的累加器。

---

### 17. Loop Filter 到 NCO

假设 Acquisition 给一个初始频率：

$$
f_{\text{acq}}
$$

PLL/FLL Loop Filter 给出修正：

$$
\Delta f[k]
$$

那么 NCO 可以使用：

$$
f_{\text{NCO}}[k]
=
f_{\text{acq}}
+
\Delta f[k]
$$

再换算：

$$
FCW[k]
=
\frac{
f_{\text{NCO}}[k]
}{f_s}2^N
$$

实际架构也可能把：

- nominal IF；
- Acquisition Doppler；
- Tracking correction

分别保存，最后再相加。

---

### 18. Nominal Frequency 与 Residual Frequency

这一点非常重要。

如果接收机有 IF：

$$
f_{IF}
$$

真实信号数字域频率可能为：

$$
f_{\text{sig}}
=
f_{IF}+f_D
$$

Carrier NCO 需要生成约：

$$
\hat f_{\text{NCO}}
=
f_{IF}+\hat f_D
$$

做 wipeoff。

因此 NCO FCW 可能包含：

$$
\boxed{
\text{Nominal IF}
+
\text{Estimated Doppler}
}
$$

而 PLL discriminator 真正纠正的通常只是：

$$
\boxed{
Residual Frequency Error
}
$$

所以看到一个很大的 Carrier FCW 时：

> 不能直接把整个 FCW 都理解成 Doppler。

必须先看芯片架构和 FCW 定义。

---

### 19. Baseband 架构又不同

如果 DFE 已经把 IF 完全搬到接近 0 Hz：

$$
f_{IF}\approx0
$$

那么 Tracking Carrier NCO 可能主要补偿：

$$
f_D
$$

以及 residual carrier error。

所以：

> Carrier FCW 的物理意义依赖前级频率规划。

读 TRK LOG 时一定要先搞清楚这一点。

---

### 20. PLL 为什么修改 Frequency 而不是直接重置 Phase

假设 PLL 发现：

$$
\Delta\phi=20^\circ
$$

可以理论上直接：

$$
PHA\leftarrow PHA+20^\circ
$$

但频繁直接跳 Phase 会破坏：

- Carrier Phase Continuity；
- ADR Continuity；
- 相位测量质量。

更常见的 Tracking 思路是：

> Loop Filter 调整 NCO frequency，让相位逐渐追上真实信号。

即：

$$
\Delta f
\rightarrow
\Delta\phi\text{ 的变化率}
$$

NCO 持续累加后，Phase Error 被逐渐消除。

有些实现也会存在显式 phase correction / phase steering，但必须非常清楚它对连续相位测量的影响。

---

### 21. NCO Wrap 与 Cycle Slip 完全不是一回事

#### NCO Wrap

例如：

$$
0.98\text{ cycle}
\rightarrow
0.03\text{ cycle}
$$

这是：

$$
\bmod 1
$$

运算。

如果 Integer Counter 正确：

$$
100.98
\rightarrow
101.03
$$

Continuous Phase 完全连续。

这是正常现象。

#### Cycle Slip

Cycle Slip 是 Tracking 环路真正失去相位连续性。

例如真实应为：

$$
101.03
$$

但接收机重新锁定后变成：

$$
102.03
$$

多了一整周。

因此：

$$
\boxed{
NCO\ Wrap\neq Cycle\ Slip
}
$$

---

### 22. FCW 与 Doppler 的关系

如果 Carrier NCO 中：

$$
f_{\text{NCO}}
=
f_{\text{nominal}}+\hat f_D
$$

那么：

$$
\hat f_D
=
f_{\text{NCO}}-f_{\text{nominal}}
$$

如果直接从 FCW 得频率：

$$
f_{\text{NCO}}
=
\frac{FCW}{2^N}f_s
$$

所以：

$$
\boxed{
\hat f_D
=
\frac{FCW}{2^N}f_s
-
f_{\text{nominal}}
}
$$

具体正负号取决于：

- mixer 定义；
- complex exponential 的正负号；
- high-side / low-side mixing；
- 芯片内部 Doppler sign convention。

因此工程分析必须以实际 TRK 定义为准。

---

### 23. Doppler 积分与 Carrier Phase

因为：

$$
\frac{d\phi}{dt}
=
f
$$

如果相位使用 cycle 为单位，则：

$$
\boxed{
\Delta\Phi
=
\int_{t_1}^{t_2} f_D(t)dt
}
$$

单位直接是：

$$
cycle
$$

离散情况下：

$$
\boxed{
\Delta\Phi
\approx
\sum_k
f_D[k]T_k
}
$$

所以：

> Integrated Doppler 本质上就是 Carrier Phase Change。

这也是 Doppler、FCW、Carrier Phase 三者能够相互转换的根本原因。

---

### 24. Carrier Phase 与距离

载波波长：

$$
\lambda
=
\frac{c}{f_c}
$$

若 Carrier Phase Change 为：

$$
\Delta\Phi
$$

cycle，则对应距离变化：

$$
\boxed{
\Delta\rho
=
\lambda\Delta\Phi
}
$$

因此：

```text
FCW / Doppler
     ↓ 积分
Carrier Cycles
     ↓ × wavelength
Carrier-phase distance / ADR change
```

---

### 25. ADR 是不是直接等于 NCO Phase

不能简单这么说。

NCO Continuous Phase 是 ADR 的重要底层信息来源，但实际 GNSS measurement 还要考虑：

- NCO sign convention；
- nominal IF 是否已经包含在 FCW 中；
- Doppler sign convention；
- initial carrier phase；
- receiver clock；
- cycle ambiguity；
- cycle slip；
- measurement epoch；
- data/pilot channel handling；
- hardware/software phase corrections。

因此更准确地说：

> Carrier NCO / integrated Doppler 提供了构造连续 Carrier Phase / ADR 的核心 Tracking 信息，但最终 measurement 需要按照接收机定义进行 reference、sign、scaling 和 continuity 处理。

---

### 26. 初始整数周模糊度仍然存在

即使接收机从 Tracking 开始以后完美统计每一周：

$$
N_{\text{cycle}}
$$

它仍然不知道在 Tracking 开始之前，从卫星到接收机的整条传播路径包含了多少个 Carrier Cycle。

所以 GNSS Carrier Phase Observation 通常写为：

$$
\Phi
=
\frac{\rho}{\lambda}
+
N
+
\text{clock / atmosphere / errors}
$$

其中：

$$
N
$$

是 Integer Ambiguity。

因此：

> “NCO 从锁定以后没有丢周”与“知道绝对整周模糊度”是两个完全不同的问题。

---

### 27. Tracking 内部需要维护的两个层次

可以区分：

#### 层次 1：Local Oscillator Phase

为了生成正确本地载波，只需要：

$$
\phi\bmod2\pi
$$

#### 层次 2：Continuous Carrier Phase

为了 ADR / Carrier Phase Measurement，需要：

$$
N+\phi_f
$$

也就是不能丢掉 Integer Cycle Continuity。

所以实际芯片里可能同时存在：

- N-bit phase accumulator；
- carry / overflow；
- extended phase accumulator；
- carrier cycle counter；
- integrated Doppler accumulator。

具体名称会因实现不同而变化。

---

### 28. 一个 30-bit NCO 数值例子

假设：

$$
N=30
$$

$$
f_s=20\text{ MHz}
$$

希望 NCO 产生：

$$
f=2500\text{ Hz}
$$

则：

$$
FCW
=
\frac{2500}{20\times10^6}2^{30}
$$

约为：

$$
134218
$$

每个 sample 相位前进：

$$
\frac{134218}{2^{30}}
\approx1.25\times10^{-4}\text{ cycle}
$$

而每秒：

$$
20\times10^6
$$

个 sample，

总计约：

$$
2500\text{ cycle/s}
$$

也就是：

$$
2500\text{ Hz}
$$

---

### 29. 如果 FCW 是负数

数字 NCO 也可以支持负频率。

例如采用 signed / two's-complement FCW。

负 FCW：

$$
PHA[n+1]
=
PHA[n]+FCW
\pmod{2^N}
$$

相位就反方向旋转。

因此 complex Carrier NCO 可以自然生成：

$$
e^{+j\theta}
$$

或：

$$
e^{-j\theta}
$$

具体符号仍由 mixer convention 决定。

---

### 30. Sample 级 NCO 与 ms 级 Loop Update

这是理解 TRK 硬件很重要的一点。

例如：

#### 每个 sample

$$
PHA[n+1]
=
PHA[n]+FCW
$$

NCO 连续运行。

#### 每 1 ms

Correlator 输出：

$$
I_P,Q_P
$$

#### 每 1 ms 或若干 ms

PLL/FLL Discriminator 输出误差。

#### Loop Update 时刻

Loop Filter 计算新的：

$$
FCW
$$

然后 NCO 从这个时刻开始使用新 FCW。

因此：

> Loop 20 ms 更新一次，不代表 NCO 20 ms 才动一次。

NCO 始终在 sample 级连续累加。

---

### 31. 本章完整数据流

```text
Acquisition Doppler / Nominal Frequency
                ↓
             Base FCW
                ↓
PLL/FLL → Loop Filter → ΔFCW
                ↓
             Total FCW
                ↓
        Phase Accumulator
                ↓
       ┌────────┴─────────┐
       ↓                  ↓
Wrapped Phase       Integer / Extended
       ↓             Phase Continuity
LUT / CORDIC             ↓
       ↓          Carrier Phase / ADR
   cos / sin
       ↓
Carrier Wipeoff
```

---

### 32. 本章关键公式

FCW：

$$
FCW
=
\frac{f}{f_s}2^N
$$

频率：

$$
f
=
\frac{FCW}{2^N}f_s
$$

NCO frequency resolution：

$$
\Delta f_{LSB}
=
\frac{f_s}{2^N}
$$

Phase Accumulator：

$$
PHA[n+1]
=
PHA[n]+FCW
\pmod{2^N}
$$

Continuous Phase：

$$
\Phi
=
N_{\text{cycle}}
+
\phi_{\text{fractional}}
$$

变 FCW 时：

$$
\Phi[n]
=
\Phi[0]
+
\frac{1}{2^N}
\sum_iFCW[i]
$$

Doppler Integration：

$$
\Delta\Phi
=
\int f_D(t)dt
$$

Carrier phase distance：

$$
\Delta\rho
=
\lambda\Delta\Phi
$$

---

### 33. 本章关键结论

1. NCO 的核心是 Phase Accumulator。
2. FCW 决定每个 sample 的相位步进，因此决定输出频率。
3. N-bit NCO 的频率分辨率为 $f_s/2^N$。
4. Phase Accumulator wrap 是正常 modulo 运算，不是 Tracking 丢周。
5. 对生成 sin/cos 来说，只需要 modulo-$2\pi$ phase。
6. 对 Carrier Phase / ADR 来说，必须维护 integer-cycle continuity。
7. Continuous Phase 可以看成 integer cycles + fractional phase。
8. 也可以使用更宽的 extended phase accumulator 直接维护 unwrapped phase。
9. PLL/FLL 改的是 NCO frequency / FCW，而 NCO 在 sample 级持续积分为 phase。
10. Loop Update 很慢，不意味着 NCO 本身运行很慢。
11. NCO FCW 中可能同时包含 nominal IF 与 Doppler，不能看到 FCW 就直接当 Doppler。
12. Integrated Doppler 本质上等于 Carrier Phase Change。
13. Carrier Phase Change 乘波长即可得到相位距离变化。
14. NCO Continuous Phase 是 ADR 的重要底层来源，但最终 ADR 还需要 sign、reference、ambiguity、cycle-slip 等处理。
15. 从 Tracking 开始后维护整数周连续性，不代表已经解决 GNSS 的绝对 Integer Ambiguity。
16. NCO Wrap 与真正的 Cycle Slip 是完全不同的概念。

---

### 34. 下一章

下一章进入：

## 第 9 章｜Carrier Phase、ADR、Integrated Doppler 与 Measurement

重点包括：

- Carrier Phase Measurement 到底定义的是什么；
- ADR 与 Integrated Doppler 是不是同一个东西；
- Doppler 的正负号为什么经常容易混；
- 载波周数如何换成距离；
- 为什么 Carrier Phase 有 Integer Ambiguity；
- Cycle Slip 如何在 ADR / Carrier Phase LOG 中表现；
- L1 / L5 Carrier Phase Change 为什么可以转成距离后比较；
- Tracking 内部 measurement epoch、NCO phase 与上层 GNSS measurement 如何对齐。


---

<a id="chapter-09"></a>

## 第 9 章｜Carrier Phase、ADR、Integrated Doppler 与 Measurement

### 1. 本章目标

本章重点理解：

- Doppler、Integrated Doppler、Carrier Phase、ADR 分别是什么；
- 为什么“积分 Doppler”会得到 Carrier Phase Change；
- 为什么 Carrier Phase Change 乘波长后可以得到距离变化；
- 为什么 Carrier Phase 有整数模糊度；
- 为什么整数模糊度通常不会影响短时间差分；
- Cycle Slip 和正常 NCO Wrap 有什么区别；
- Carrier Phase / ADR 的正负号为什么经常让人混淆；
- L1、L5 载波周变化如何转成距离后比较；
- Measurement Epoch 为什么必须严格对齐。

---

### 2. 先把四个量分开

#### 2.1 Doppler

Doppler 是频率变化量，单位：

$$
Hz = cycle/s
$$

记作：

$$
f_D(t)
$$

它反映载波相位变化的速度。

---

#### 2.2 Integrated Doppler

对 Doppler 随时间积分：

$$
\Delta\Phi
=
\int_{t_1}^{t_2} f_D(t)dt
$$

因为：

$$
Hz\times s = cycle
$$

所以积分结果单位是：

$$
cycle
$$

因此 Integrated Doppler 本质上得到的是：

> 某一段时间内累计变化了多少载波周。

---

#### 2.3 Carrier Phase

Carrier Phase 可以用 rad 或 cycle 表示。

若用 cycle：

$$
\Phi(t)
$$

表示连续载波相位。

相邻两个 epoch：

$$
\Delta\Phi
=
\Phi(t_2)-\Phi(t_1)
$$

与 Doppler 的积分存在直接关系。

---

#### 2.4 ADR

ADR 常展开为：

> Accumulated Delta Range

也就是累计距离变化量。

如果 Carrier Phase Change 为：

$$
\Delta\Phi
$$

cycle，载波波长为：

$$
\lambda
$$

则对应距离变化：

$$
\Delta\rho
=
\lambda\Delta\Phi
$$

在很多接收机里，ADR 就是由连续 Carrier Tracking / Integrated Doppler 构造出的相位距离量。

但必须注意：

> ADR 的最终符号、参考时刻、是否包含某些 correction，取决于接收机和接口定义。

---

### 3. Doppler 为什么能积分成 Carrier Phase

载波可以写成：

$$
s(t)
=
A\cos(2\pi f_ct+\phi_0)
$$

若瞬时频率变化为：

$$
f(t)
$$

那么相位满足：

$$
\frac{d\Phi(t)}{dt}
=
f(t)
$$

这里 $\Phi$ 用 cycle 表示。

所以：

$$
\Phi(t_2)-\Phi(t_1)
=
\int_{t_1}^{t_2}f(t)dt
$$

如果只考虑 Doppler 部分：

$$
\boxed{
\Delta\Phi_D
=
\int f_D(t)dt
}
$$

所以：

> Integrated Doppler 与 Carrier Phase Change，本质上是同一个物理关系的频率域和相位域表达。

---

### 4. 离散系统里怎么做

如果每隔：

$$
T
$$

得到一次 Doppler Estimate：

$$
f_D[k]
$$

那么：

$$
\Delta\Phi
\approx
\sum_k f_D[k]T
$$

如果 $T$ 不固定：

$$
\Delta\Phi
\approx
\sum_k f_D[k]\Delta t_k
$$

这就是离散 Integrated Doppler。

---

### 5. FCW 和 Integrated Doppler

上一章有：

$$
f[k]
=
\frac{FCW[k]}{2^N}f_s
$$

每个 sample 的 phase increment：

$$
\Delta\Phi_{\text{sample}}
=
\frac{FCW[k]}{2^N}
$$

cycle。

所以连续累加 FCW：

$$
\Phi[n]
=
\Phi[0]
+
\frac{1}{2^N}
\sum_iFCW[i]
$$

本质上就是：

$$
\int fdt
$$

因此：

> Phase Accumulator、Integrated Doppler 和 Carrier Phase Continuity，本质上是同一件事在不同模块里的表达。

---

### 6. Carrier Phase Change 怎么转距离

载波波长：

$$
\lambda
=
\frac{c}{f_c}
$$

如果相位变化：

$$
\Delta\Phi
$$

cycle，

则对应传播距离变化量的绝对尺度为：

$$
\boxed{
|\Delta\rho|
=
\lambda|\Delta\Phi|
}
$$

带符号关系则要结合 Carrier Phase / Doppler 的 sign convention。

例如 GPS L1：

$$
f_{L1}=1575.42\text{ MHz}
$$

波长约：

$$
\lambda_{L1}\approx0.1903\text{ m}
$$

所以：

$$
1\text{ cycle}
\leftrightarrow
19.03\text{ cm}
$$

而：

$$
0.01\text{ cycle}
\leftrightarrow
1.903\text{ mm}
$$

这也是 Carrier Phase 能做到毫米级变化敏感度的原因。

---

### 7. 图：Integrated Doppler → Phase → Distance

配套图：

```text
GNSS_Tracking_第09章_积分多普勒与载波距离.png
```

图中展示：

$$
f_D(t)
$$

积分后变成累计 cycle，

再乘：

$$
\lambda
$$

变成等效 Carrier-Phase Distance Change。

---

### 8. Doppler 与几何距离变化的符号关系

这是最容易混乱的部分之一。

如果几何距离：

$$
\rho(t)
$$

增加，表示卫星和接收机在远离。

其 range rate：

$$
\dot\rho>0
$$

对于常见物理 Doppler 定义，接收频率降低：

$$
f_D
\approx
-\frac{f_c}{c}\dot\rho
$$

也就是：

$$
\boxed{
f_D
\approx
-\frac{\dot\rho}{\lambda}
}
$$

所以：

$$
\dot\rho>0
\Rightarrow
f_D<0
$$

而距离减小时：

$$
\dot\rho<0
$$

通常：

$$
f_D>0
$$

但注意：

> 芯片内部 Carrier NCO、Mixer、软件 LOG 可能采用相反符号定义。

因此看实际 LOG 时，必须先确认：

- Doppler 是 signal frequency offset 还是 NCO correction；
- Mixer 使用 $e^{+j\theta}$ 还是 $e^{-j\theta}$；
- 上层 API 是否对 Doppler 又做了一次取反。

---

### 9. 为什么 Carrier Phase 的符号也容易混

如果采用经典 GNSS Carrier Phase Observation 约定，常可写成类似：

$$
\Phi
\approx
-\frac{\rho}{\lambda}
+
N
+
\text{clock / atmosphere / bias}
$$

于是距离增加：

$$
\rho\uparrow
$$

Carrier Phase Observation 可能减少。

但有些内部 Tracking Counter 是：

> 本地 NCO 累计 phase

它可能随时间正向增加。

所以：

$$
\text{NCO accumulated phase}
$$

和：

$$
\text{standard GNSS carrier phase observable}
$$

可能差一个符号、常数 offset 或 reference。

因此：

> 不要仅凭“都叫 Carrier Phase”就默认数值符号一致。

---

### 10. ADR 与 Carrier Phase 是不是同一个东西

物理上高度相关，但工程上不要直接完全等同。

可以粗略理解：

```text
Carrier Tracking
      ↓
Continuous Phase / Integrated Doppler
      ↓
Sign / Scale / Reference / Correction
      ↓
ADR Measurement
```

如果 ADR 单位是 m：

$$
ADR
\sim
\lambda\Phi
$$

如果 Carrier Phase 单位是 cycle：

$$
\Phi
\sim
\frac{ADR}{\lambda}
$$

但具体是否带负号、初值怎么定义、reset 时如何处理，必须看接口定义。

---

### 11. Integer Ambiguity 从哪里来

即使 Tracking 从现在开始能够完美统计：

$$
\Delta\Phi
$$

它还是不知道锁定之前已经经历了多少个完整 Carrier Cycle。

因此完整 Carrier Phase 常写成：

$$
\Phi
=
\text{fractional phase}
+
N
$$

其中：

$$
N\in\mathbb Z
$$

就是 Integer Ambiguity。

GNSS 高精度载波相位模型里，常进一步包含：

$$
\rho/\lambda,\quad
\text{clock},\quad
\text{ionosphere},\quad
\text{troposphere},\quad
\text{bias}
$$

---

### 12. 为什么差分以后 Integer Ambiguity 可以消掉

同一条连续跟踪弧段内，如果没有 Cycle Slip：

$$
N
$$

保持不变。

例如：

$$
\Phi_1
=
N+\phi_1
$$

$$
\Phi_2
=
N+\phi_2
$$

做时间差分：

$$
\Phi_2-\Phi_1
=
\phi_2-\phi_1
$$

整数模糊度：

$$
N
$$

直接消掉。

所以：

> 如果只研究一小段时间内的 Carrier Phase Change / Distance Change，通常不需要先解绝对 Integer Ambiguity。

这也是用载波相位差分析短时间距离变化的基础。

---

### 13. 图：整数模糊度只改变绝对 Offset

配套图：

```text
GNSS_Tracking_第09章_整数模糊度示意.png
```

两条曲线可能相差：

$$
37\text{ cycles}
$$

但它们随时间的变化量完全一样。

因此：

$$
\Delta\Phi
$$

不受固定整数 offset 影响。

---

### 14. Cycle Slip 为什么麻烦

如果 Tracking 失锁，整数周连续性可能发生跳变：

$$
N
\rightarrow
N+m
$$

其中：

$$
m\in\mathbb Z
$$

那么 Carrier Phase 会突然跳：

$$
m\text{ cycles}
$$

此时再做差分：

$$
\Delta\Phi
$$

就会多出：

$$
m
$$

因此：

> Integer Ambiguity 本身不是短时间差分的问题；Cycle Slip 导致 Integer Ambiguity 在中途发生变化，才是真正破坏连续性的原因。

---

### 15. 图：Cycle Slip

配套图：

```text
GNSS_Tracking_第09章_CycleSlip示意.png
```

在某个 epoch 后，测量突然整体偏移：

$$
+1\text{ cycle}
$$

这种跳变不是正常运动产生的连续相位变化，而是 Tracking Continuity 被破坏。

---

### 16. NCO Wrap 与 Cycle Slip 再对比

#### NCO Wrap

低位 phase：

$$
0.99
\rightarrow
0.02
$$

但 integer counter：

$$
100
\rightarrow
101
$$

所以 continuous phase：

$$
100.99
\rightarrow
101.02
$$

完全连续。

#### Cycle Slip

本应：

$$
101.02
$$

实际突然：

$$
102.02
$$

这才是真正多了一周。

所以：

$$
\boxed{
Wrap\neq Slip
}
$$

---

### 17. Integrated Doppler 与 Cycle Slip 的关系

如果直接对连续 Doppler Estimate 做积分：

$$
\sum f_D[k]T
$$

理论上得到连续 Phase Change。

但如果 Tracking 突然失锁并重新锁定：

- Doppler Estimate 可能短暂异常；
- Phase reference 可能重置；
- integer cycle continuity 可能断裂。

所以只积分 Doppler 并不能天然保证：

> 绝对 Carrier Phase Continuity 永远正确。

实际 Measurement 通常还需要：

- Lock state；
- Cycle-slip detection；
- Reset flag；
- Half-cycle ambiguity；
- phase-valid indicator

等状态一起判断。

---

### 18. L1 / L5 为什么要先乘各自波长再比较

L1 和 L5 频率不同：

$$
f_{L1}\neq f_{L5}
$$

所以：

$$
\lambda_{L1}\neq\lambda_{L5}
$$

相同几何距离变化：

$$
\Delta\rho
$$

对应的 Carrier Cycle Change 不同：

$$
\Delta\Phi_{L1}
=
\frac{\Delta\rho}{\lambda_{L1}}
$$

$$
\Delta\Phi_{L5}
=
\frac{\Delta\rho}{\lambda_{L5}}
$$

所以直接比较：

$$
\Delta\Phi_{L1}
$$

和：

$$
\Delta\Phi_{L5}
$$

没有意义。

必须分别乘各自波长：

$$
\Delta\rho_{L1}
=
\lambda_{L1}\Delta\Phi_{L1}
$$

$$
\Delta\rho_{L5}
=
\lambda_{L5}\Delta\Phi_{L5}
$$

再比较：

$$
e
=
\Delta\rho_{L1}
-
\Delta\rho_{L5}
$$

---

### 19. 为什么理论上 L1 / L5 Distance Change 应接近

同一颗卫星、同一个接收机，在相同 measurement interval 内，主要几何距离变化：

$$
\Delta\rho
$$

应该是共同的。

所以理想情况下：

$$
\lambda_{L1}\Delta\Phi_{L1}
\approx
\lambda_{L5}\Delta\Phi_{L5}
$$

差异可能来自：

- 两路 measurement epoch 不完全一致；
- Tracking noise；
- 不同 multipath；
- ionosphere frequency dependence；
- hardware group/phase delay variation；
- independent clock / channel timing；
- cycle slip；
- phase correction / sign convention；
- finite precision / FCW quantization；
- measurement generation logic。

短时间 carrier-change comparison 中，很多公共误差会抵消，但并不是所有频点相关误差都完全相同。

---

### 20. Measurement Epoch 为什么极其重要

假设 L1 Measurement 是：

$$
t=1.000\text{ s}
$$

而 L5 Measurement 是：

$$
t=1.001\text{ s}
$$

如果 range rate：

$$
100\text{ m/s}
$$

仅：

$$
1\text{ ms}
$$

时间错位就会产生：

$$
0.1\text{ m}
$$

即：

$$
100\text{ mm}
$$

的距离差。

所以如果你要比较毫米级 L1/L5 Carrier Distance Change：

> Measurement Epoch Alignment 必须非常严格。

即使两路都显示“1 s 输出一次”，也要确认：

- accumulator 起点；
- accumulator 终点；
- latch 时刻；
- software timestamp；
- channel latency

是否真正一致。

---

### 21. Doppler 与相位差的离散关系

如果相邻 measurement interval 为：

$$
T
$$

可以写：

$$
\Delta\Phi[k]
\approx
f_D[k]T
$$

但更准确是：

$$
\Delta\Phi[k]
=
\int_{t_k}^{t_{k+1}}f_D(t)dt
$$

如果 Doppler 在区间内变化明显，使用单点：

$$
f_D[k]T
$$

只是近似。

可以使用：

- interval average Doppler；
- trapezoidal integration；
- sample-level FCW accumulation

提高一致性。

---

### 22. 为什么 Sample-level FCW Accumulation 更自然

如果硬件 NCO 每个 sample 都知道：

$$
FCW[n]
$$

那么直接累加：

$$
\sum_n\frac{FCW[n]}{2^N}
$$

得到的 phase change，天然包含 interval 内的所有 frequency update。

相比：

$$
f_D(t_k)\times T
$$

这种单点近似，更准确地反映实际 local carrier phase evolution。

---

### 23. ADR 的 Reset

有些接收机在以下情况可能 reset / invalidate ADR：

- Tracking channel reset；
- signal reacquisition；
- PLL loss of lock；
- cycle slip；
- clock discontinuity；
- measurement engine reset。

所以 ADR LOG 中如果突然大跳，不能立即认定为真实距离变化。

首先应该检查：

```text
Lock state
ADR valid state
Reset flag
Cycle-slip flag
Clock discontinuity
```

---

### 24. Carrier Phase Measurement 的完整直觉

可以把它理解成：

```text
真实传播距离变化
       ↓
Carrier phase rotates
       ↓
PLL / FLL Tracking
       ↓
Carrier NCO / Integrated Doppler
       ↓
Continuous cycle count
       ↓
Sign / Reference / Correction
       ↓
Carrier Phase / ADR Measurement
```

---

### 25. 一个简单数值例子

假设 L1 一秒内 Carrier Phase Change：

$$
\Delta\Phi_{L1}=620.000\text{ cycles}
$$

L1 波长：

$$
\lambda_{L1}\approx0.1902937\text{ m}
$$

则距离变化量绝对尺度：

$$
\Delta\rho_{L1}
\approx
620\times0.1902937
$$

约：

$$
117.982\text{ m}
$$

假设 L5 对应：

$$
\Delta\Phi_{L5}
$$

乘 L5 波长后得到：

$$
117.985\text{ m}
$$

则两路距离变化差：

$$
e
=
117.985-117.982
=
0.003\text{ m}
$$

即：

$$
3\text{ mm}
$$

这类误差分析真正应该研究的是：

- 是否有稳定 bias；
- 标准差多大；
- 是否随 range rate / Doppler 相关；
- 是否随 time offset 相关；
- 是否存在周期项；
- 是否与 Tracking state / C/N0 相关。

---

### 26. Signed Mean 和 Standard Deviation 要分开看

如果误差：

$$
e_k
=
\Delta\rho_{L1,k}
-
\Delta\rho_{L5,k}
$$

则：

#### Mean

$$
\bar e
$$

反映稳定 bias。

例如：

$$
\bar e=+2.8\text{ mm}
$$

表示 L1/L5 之间存在平均约 2.8 mm 的有符号偏差。

#### Standard Deviation

$$
\sigma_e
$$

反映随机波动。

例如：

$$
\sigma_e=2.6\text{ mm}
$$

表示每个 epoch 之间还有毫米级随机变化。

两者物理意义完全不同。

---

### 27. 累计差分可以暴露很小的长期 Bias

如果每个 interval 的距离差：

$$
e[k]
$$

很小，但有稳定均值：

$$
E[e]=b
$$

累计：

$$
D[M]
=
\sum_{k=1}^{M}e[k]
$$

那么：

$$
E[D[M]]
=
Mb
$$

所以一个每秒只有：

$$
0.2\text{ mm}
$$

的稳定 bias，

1000 s 后会累计成：

$$
200\text{ mm}
$$

因此累计差分特别适合暴露长期比例误差、时钟差、频率 scaling error 等系统性问题。

---

### 28. 本章关键公式

Doppler integration：

$$
\Delta\Phi
=
\int f_D(t)dt
$$

离散形式：

$$
\Delta\Phi
\approx
\sum_kf_D[k]\Delta t_k
$$

FCW accumulation：

$$
\Delta\Phi
=
\frac{1}{2^N}
\sum_nFCW[n]
$$

Carrier distance change：

$$
\Delta\rho
=
\lambda\Delta\Phi
$$

波长：

$$
\lambda
=
\frac{c}{f_c}
$$

常见物理 range-rate / Doppler relation：

$$
f_D
\approx
-\frac{\dot\rho}{\lambda}
$$

Integer Ambiguity：

$$
\Phi
=
N+\phi_f
$$

同弧段时间差分：

$$
\Delta\Phi
=
\Phi_2-\Phi_1
$$

固定 $N$ 被消掉。

L1/L5 comparison：

$$
e
=
\lambda_{L1}\Delta\Phi_{L1}
-
\lambda_{L5}\Delta\Phi_{L5}
$$

---

### 29. 本章关键结论

1. Doppler 的单位是 cycle/s，对时间积分后自然得到 cycle。
2. Integrated Doppler 本质上就是 Carrier Phase Change。
3. Carrier Phase Change 乘波长得到 Carrier-Phase Distance Change。
4. FCW sample-level accumulation 是 Integrated Doppler 的数字硬件实现之一。
5. ADR 与 Carrier Phase 高度相关，但最终 sign、reference、scale 依赖接收机接口定义。
6. Carrier Phase 的绝对值存在 Integer Ambiguity。
7. 同一连续 Tracking Arc 内做时间差分时，固定 Integer Ambiguity 会消掉。
8. Cycle Slip 会让 Integer Ambiguity 在中途改变，从而破坏差分连续性。
9. NCO Wrap 是正常 modulo 运算，和 Cycle Slip 完全不同。
10. Doppler、NCO phase、标准 Carrier Phase Observation 的符号不一定一致，必须查实际实现定义。
11. L1/L5 Cycle Change 必须先分别乘各自波长，再比较距离变化。
12. 毫米级 L1/L5 Comparison 对 Measurement Epoch Alignment 非常敏感。
13. 同样“每秒输出一次”并不能保证两个通道 measurement interval 真正一致。
14. Sample-level FCW accumulation 比单点 Doppler × interval 更接近真实 Carrier Phase evolution。
15. ADR 大跳时必须同时检查 lock、reset、cycle-slip、clock discontinuity 等状态。
16. Signed Mean 主要反映 bias，Standard Deviation 主要反映随机 jitter。
17. 累计差分可以把很小的长期 bias 放大出来，适合定位 scaling / timing / systematic error。

---

### 30. 下一章

下一章进入：

## 第 10 章｜Tracking Noise、Jitter、Dynamic Stress 与 Lock Threshold

重点包括：

- PLL / DLL Thermal Noise Jitter；
- $C/N_0$、$T_{coh}$、$B_n$ 如何定量影响 Tracking Error；
- 为什么弱信号要增加积分但又会损失动态；
- Dynamic Stress Error 怎么产生；
- Loop Order 与 acceleration / jerk 的关系；
- PLL 什么时候算失锁；
- Cycle Slip Probability 为什么在低 $C/N_0$ 时急剧增加；
- 如何从性能角度设计 Tracking 参数。


---

<a id="chapter-10"></a>

## 第 10 章｜Tracking Noise、Jitter、Dynamic Stress 与 Lock Threshold

### 1. 本章目标

本章重点理解：

- 为什么 Tracking 已经“锁住”以后仍然存在随机误差；
- PLL / DLL Jitter 是什么；
- $C/N_0$、$T_{coh}$、$B_n$ 如何共同决定 Tracking Noise；
- 为什么延长相干积分可以改善弱信号 Tracking；
- 为什么长积分又会牺牲动态能力；
- Dynamic Stress Error 是怎么产生的；
- Loop Order 为什么决定能跟踪哪一类动态；
- Cycle Slip 为什么在弱信号时会突然增多；
- Lock Threshold / Lock Indicator 应该怎么看；
- Tracking 参数为什么本质上是 Noise 与 Dynamics 的联合设计。

---

### 2. Tracking Error 的三个主要来源

可以粗略写成：

$$
e_{\text{tracking}}
=
e_{\text{thermal}}
+
e_{\text{dynamic}}
+
e_{\text{bias}}
$$

其中：

#### Thermal / Random Noise

来自：

- RF / DFE thermal noise；
- correlator noise；
- quantization；
- discriminator noise。

表现为随机 Jitter。

#### Dynamic Stress Error

来自：

> 真实信号变化速度超过环路的跟踪能力。

例如：

- Doppler 变化太快；
- acceleration 太大；
- jerk 太大；
- loop bandwidth 太窄。

#### Bias

来自：

- multipath；
- interference；
- frontend distortion；
- discriminator asymmetry；
- implementation bias。

Loop Filter 能降低随机高频部分，但不能自动消除稳定 bias。

---

### 3. PLL Thermal Phase Jitter

PLL 已经锁定时：

$$
\Delta\phi_{\text{true}}\approx0
$$

但 Prompt 仍然是：

$$
I=A+n_I
$$

$$
Q=n_Q
$$

所以：

$$
\hat{\Delta\phi}
=
\operatorname{atan2}(Q,I)
$$

仍然会在 0 附近随机波动。

这就是：

> PLL Phase Jitter。

---

### 4. PLL Jitter 的常用近似趋势

对于常见 arctangent / Costas 型 PLL，在一定模型假设下常见近似：

$$
\boxed{
\sigma_\phi
\approx
\sqrt{
\frac{B_n}{C/N_0}
\left(
1+
\frac{1}
{2T_{coh}(C/N_0)}
\right)
}
}
$$

单位为 rad。

注意：

> 不同教材和 discriminator / data-channel 模型下，系数可能不同。

真正最重要的是趋势：

$$
\boxed{
\sigma_\phi
\uparrow
\quad\text{when}\quad
B_n\uparrow
}
$$

$$
\boxed{
\sigma_\phi
\downarrow
\quad\text{when}\quad
C/N_0\uparrow
}
$$

以及在低 $C/N_0$ 时：

$$
T_{coh}\uparrow
\Rightarrow
\text{single-measurement quality improves}
$$

---

### 5. 图：PLL Jitter 与 $C/N_0$

配套图片：

```text
GNSS_Tracking_第10章_PLL热噪声抖动_vs_CN0.png
```

图中采用：

$$
T_{coh}=1\text{ ms}
$$

比较：

$$
B_n=5,\ 10,\ 20\text{ Hz}
$$

可以看到：

#### $C/N_0$ 高

三种 bandwidth 的 phase jitter 都较小。

#### $C/N_0$ 下降

phase jitter 非线性增大。

#### 同样 $C/N_0$

$$
B_n=20\text{ Hz}
$$

比：

$$
B_n=5\text{ Hz}
$$

明显更抖。

---

### 6. 一个数值例子

假设：

$$
C/N_0=30\text{ dB-Hz}
$$

即：

$$
C/N_0=1000\text{ Hz}
$$

取：

$$
B_n=10\text{ Hz}
$$

$$
T_{coh}=1\text{ ms}
$$

代入上述近似：

$$
\sigma_\phi
\approx
\sqrt{
\frac{10}{1000}
\left(
1+
\frac{1}{2\times0.001\times1000}
\right)
}
$$

$$
=
\sqrt{0.015}
$$

约：

$$
0.122\text{ rad}
$$

即：

$$
\boxed{
\sigma_\phi\approx7.0^\circ
}
$$

所以 30 dB-Hz + 10 Hz PLL + 1 ms integration 时，Phase Noise 已经不能忽略。

---

### 7. $C/N_0$ 为什么是 Tracking 的核心指标

$C/N_0$ 单位：

$$
dB-Hz
$$

转成线性：

$$
C/N_0
$$

单位可以理解成：

$$
Hz
$$

一次相干积分长度：

$$
T
$$

那么相关输出可用的一个核心量是：

$$
\boxed{
(C/N_0)T
}
$$

它是无量纲量。

因此：

$$
T\uparrow
$$

意味着单次相关 measurement 的有效 SNR 提高。

这就是弱信号时延长 coherent integration 的根本原因。

---

### 8. 相干积分增加多少收益

理想情况下：

$$
T_1\rightarrow T_2
$$

相干积分 SNR 增益：

$$
\boxed{
G
=
10\log_{10}
\frac{T_2}{T_1}
}
$$

例如：

#### 1 ms → 10 ms

$$
+10\text{ dB}
$$

#### 1 ms → 20 ms

$$
+13.0\text{ dB}
$$

前提是：

- carrier sufficiently wiped off；
- code aligned；
- data / secondary-code transition handled；
- interval 内动态没有造成严重相干损失。

---

### 9. 图：$T_{coh}$ 对 PLL Jitter 的影响

配套图片：

```text
GNSS_Tracking_第10章_Tcoh与PLL抖动.png
```

图中固定：

$$
C/N_0=30\text{ dB-Hz}
$$

$$
B_n=10\text{ Hz}
$$

可以看到增加：

$$
T_{coh}
$$

能够降低低 $C/N_0$ 下额外的 measurement noise 项。

但曲线最终会逐渐趋于：

$$
\sqrt{\frac{B_n}{C/N_0}}
$$

对应的底噪水平。

---

### 10. 为什么不能无限延长 $T_{coh}$

因为长积分还有代价。

#### Residual Doppler

积分期间 phase rotation：

$$
\Delta\theta
=
2\pi\Delta fT_{coh}
$$

过大时发生 coherent cancellation。

#### Dynamic

真实 Doppler / phase 在一个长窗口内变化，local replica 不能完全匹配。

#### Data Bit

未知 bit transition 会造成：

$$
+1\rightarrow-1
$$

使前后积分互相抵消。

#### Secondary Code

如果未知或未 wipeoff，也会限制 coherent integration。

#### Update Rate

如果：

$$
T_{update}=T_{coh}
$$

则：

$$
T_{coh}\uparrow
\Rightarrow
f_{update}\downarrow
$$

环路动态能力下降。

---

### 11. DLL Jitter

DLL 也存在类似问题。

理想锁定：

$$
E=L
$$

但有噪声时：

$$
E+n_E
$$

和：

$$
L+n_L
$$

不会完全相等。

因此 DLL discriminator：

$$
D
=
\frac{E-L}{E+L}
$$

会在 0 周围随机波动。

Code NCO 因此产生：

> Code Tracking Jitter。

总体趋势同样：

$$
\boxed{
\sigma_{DLL}
\propto
\sqrt{
\frac{B_n}{C/N_0}
}
}
$$

但实际 DLL Jitter 还明显依赖：

- Early-Late spacing；
- Frontend bandwidth；
- code waveform；
- discriminator type；
- coherent integration time。

因此 DLL 的完整公式比 PLL 更依赖接收机具体实现。

---

### 12. Narrow Correlator 为什么可以改善 DLL

DLL error 是通过 E/L 对相关峰斜率的采样得到。

在一定条件下减小：

$$
d_{EL}
$$

可以提高 multipath performance 和局部 delay sensitivity。

但不能无限减小，因为受：

- frontend bandwidth；
- sample / interpolation resolution；
- E/L noise correlation；
- code NCO precision

限制。

所以 DLL Jitter 设计不能只看：

$$
B_n
$$

还必须看 correlator geometry。

---

### 13. Dynamic Stress Error

即使：

$$
C/N_0\rightarrow\infty
$$

完全没有 thermal noise，

如果真实信号动态太快：

$$
\text{Loop 仍然可能跟不上}
$$

这部分误差就是：

> Dynamic Stress Error。

例如 PLL：

真实 Doppler：

```text
100 Hz
105 Hz
110 Hz
115 Hz
...
```

而 NCO 因 loop 太慢：

```text
100 Hz
102 Hz
105 Hz
109 Hz
...
```

始终存在 lag。

于是 residual frequency：

$$
\Delta f\neq0
$$

继续积分成 phase error。

---

### 14. Bandwidth 与 Dynamic Stress

一般：

$$
B_n\uparrow
$$

环路响应更快，所以：

$$
e_{\text{dynamic}}\downarrow
$$

但：

$$
e_{\text{noise}}\uparrow
$$

反过来：

$$
B_n\downarrow
$$

Noise Jitter 减小，

但：

$$
Dynamic Stress Error
$$

增加。

所以总误差往往存在一个折中点。

---

### 15. 图：Bandwidth 的 Noise / Dynamic Tradeoff

配套图片：

```text
GNSS_Tracking_第10章_带宽噪声动态折中.png
```

该图只是定性示意，不代表某一具体接收机数值。

它表达的是：

```text
Bandwidth 太小：
Dynamic Error 主导

Bandwidth 适中：
Total Error 最小

Bandwidth 太大：
Noise Jitter 主导
```

所以：

> 最优 bandwidth 不是“越窄越好”或者“越宽越好”。

---

### 16. Loop Order 与 Dynamic Stress

#### 一阶 PLL

主要能无稳态误差跟踪：

$$
\text{Constant Phase}
$$

持续 Frequency Offset 会产生明显跟踪误差。

#### 二阶 PLL

理想情况下可以无稳态误差跟踪：

$$
\text{Constant Frequency}
$$

即：

$$
\phi(t)\propto t
$$

但如果存在持续：

$$
\dot f=\text{constant}
$$

也就是 frequency ramp / acceleration-induced Doppler ramp，

会产生有限 Dynamic Stress Error。

#### 三阶 PLL

可以进一步理想跟踪：

$$
\dot f=\text{constant}
$$

其主要动态压力进一步转向更高阶的：

$$
\ddot f
$$

也就是 jerk 类变化。

---

### 17. 为什么高动态常用三阶 Carrier Loop

车辆、无人机、高动态平台中：

$$
f_D(t)
$$

并不是常数。

如果 acceleration 明显：

$$
\dot f_D\neq0
$$

二阶 PLL 会产生持续 phase lag。

三阶 PLL 可以进一步建模这种 frequency ramp。

代价是：

- 参数更多；
- Noise Response 更复杂；
- 稳定性更难设计；
- 弱信号条件下不一定比低阶环更好。

---

### 18. Carrier Aiding 与 Dynamic Stress

DLL 的 code dynamics 可以从 Carrier Doppler 预测：

$$
f_{code,dyn}
\approx
f_{code}
\frac{f_{carrier,dyn}}{f_{carrier}}
$$

因此 Carrier Aiding 可以先补掉绝大部分动态。

剩余 DLL 只跟踪 residual。

这样：

$$
B_{DLL}
$$

可以做得更窄，

同时：

$$
Dynamic Stress
$$

仍然可控。

---

### 19. Lock Threshold 是什么

Tracking 不是“误差等于 0 才叫锁定”。

实际是：

> Phase / Frequency / Code Error 落在环路能够稳定工作的范围，并且误差统计满足一定条件。

因此芯片会设计：

- PLL lock indicator；
- FLL lock indicator；
- DLL lock indicator；
- C/N0 threshold；
- discriminator variance threshold。

---

### 20. PLL 的一个常见工程经验

GNSS Tracking 教材里经常使用类似：

> 总 Phase Error 要明显小于鉴别器非线性/失锁边界。

例如一些工程分析会拿：

$$
15^\circ
$$

附近作为较保守的 phase-jitter reference。

因此可能检查类似：

$$
3\sigma_\phi
+
|\phi_{\text{dynamic}}|
$$

是否仍然保持在较小角度范围内。

但要注意：

$$
15^\circ
$$

不是所有芯片统一的“失锁标准”。

实际 lock threshold 与：

- Costas / atan2 discriminator；
- loop order；
- data/pilot；
- integration；
- implementation lock detector

有关。

---

### 21. 为什么弱信号会突然 Cycle Slip

当：

$$
C/N_0
$$

逐渐下降，

phase jitter 并不是“只是稍微大一点”。

一旦随机 phase error 经常接近：

- discriminator 非线性区域；
- Costas $\pm90^\circ$ ambiguity boundary；
- PLL cycle boundary；

就容易发生：

$$
Cycle\ Slip
$$

因此低 $C/N_0$ 下 Cycle Slip Probability 会急剧增加。

表现常常不是：

```text
慢慢变差
```

而是：

```text
开始还正常
↓
jitter变大
↓
偶尔大误差
↓
突然 slip / loss of lock
```

---

### 22. 为什么 3σ 经常被拿来判断裕量

如果 phase error 近似 Gaussian：

$$
\phi_e\sim N(0,\sigma_\phi^2)
$$

那么：

$$
\pm3\sigma
$$

覆盖大部分普通随机波动。

因此工程上会用：

$$
3\sigma_\phi
$$

作为一个方便的随机噪声裕量估计，

再与：

$$
Dynamic Stress Error
$$

一起检查。

这不是严格的 cycle-slip probability 计算，但非常适合早期设计。

---

### 23. 一个例子

前面：

$$
C/N_0=30\text{ dB-Hz}
$$

$$
B_n=10\text{ Hz}
$$

$$
T_{coh}=1\text{ ms}
$$

得到近似：

$$
\sigma_\phi\approx7^\circ
$$

那么：

$$
3\sigma_\phi\approx21^\circ
$$

已经比较大。

如果同时还有：

$$
5^\circ
$$

Dynamic Stress Error，

总 phase margin 就会更紧张。

如果提高到：

$$
C/N_0=35\text{ dB-Hz}
$$

同样参数下，近似 thermal jitter 会明显降低到几度量级。

---

### 24. 弱信号 Tracking 通常怎么调

当：

$$
C/N_0\downarrow
$$

常见策略：

#### 增大 $T_{coh}$

提高每次 correlator measurement SNR。

#### 减小 $B_n$

减少 noise 进入 NCO。

#### 使用 Pilot / Data Wipeoff

支持更长 coherent integration。

#### Carrier Aiding

减轻 DLL dynamics。

#### FLL-assisted PLL

在高动态或较大 residual frequency 时帮助 PLL。

#### 使用更强的 state / vector aiding

减少单通道完全依靠 noisy measurement 的程度。

---

### 25. 但为什么不能简单“弱信号就全部变窄”

因为如果设备同时：

$$
Dynamic\uparrow
$$

那么：

$$
B_n\downarrow
$$

会导致：

$$
Dynamic Stress\uparrow
$$

最后可能反而更容易失锁。

所以 Tracking 参数设计一定是联合考虑：

$$
\boxed{
C/N_0
+
Dynamics
+
T_{coh}
+
B_n
+
Loop\ Order
}
$$

---

### 26. Tracking State Machine 的典型思想

实际接收机很可能不是固定参数。

例如：

```text
Acquisition
   ↓
Pull-in
Short Tcoh
Wide FLL / PLL
   ↓
Stable lock
Medium bandwidth
   ↓
Weak-signal mode
Longer Tcoh
Narrower bandwidth
```

如果检测到：

- dynamics increase；
- phase variance increase；
- FLL error increase；

又可能重新：

```text
widen bandwidth
or
enable FLL aiding
```

所以现代 Tracking 是状态相关、条件相关的。

---

### 27. Lock Indicator 可能看哪些量

实际实现可以综合：

#### Prompt Power

$$
I_P^2+Q_P^2
$$

#### Phase Error Variance

$$
Var(\hat\phi)
$$

#### I / Q Distribution

PLL lock 后通常：

$$
|I|\gg|Q|
$$

#### FLL Error

Residual frequency 是否足够小。

#### DLL discriminator

是否长期在 zero-crossing 附近。

#### $C/N_0$

是否高于当前 mode 的可用门限。

---

### 28. Lock 和 Measurement Valid 是两回事

有时 Tracking channel 仍然输出：

$$
I/Q,\ FCW,\ Phase
$$

但 measurement engine 可能已经把：

$$
ADR\ Valid=0
$$

因为：

- phase continuity 不可信；
- cycle slip；
- channel reset；
- clock discontinuity。

因此分析 LOG 时：

> “TRK 没完全掉”并不等于 Carrier Phase Measurement 仍然可靠。

---

### 29. PLL、DLL Error 对 Measurement 的影响不同

#### PLL Error

直接影响：

- Carrier Phase；
- ADR；
- Doppler；
- coherent integration loss。

#### DLL Error

直接影响：

- code phase；
- pseudorange；
- correlator alignment。

如果 DLL 有小量 meter-level noise，

而 PLL 可以做到 degree-level phase noise，

对应 Carrier Distance 可以做到：

$$
\text{mm～cm}
$$

级。

这就是 code measurement 和 carrier measurement 精度差别巨大的原因之一。

---

### 30. Phase Jitter 怎么换成距离 Jitter

如果 PLL phase jitter：

$$
\sigma_\phi
$$

单位 rad，

换成 cycle：

$$
\sigma_{\Phi}
=
\frac{\sigma_\phi}{2\pi}
$$

再乘波长：

$$
\boxed{
\sigma_\rho
=
\lambda
\frac{\sigma_\phi}{2\pi}
}
$$

例如 L1：

$$
\lambda\approx0.1903\text{ m}
$$

如果：

$$
\sigma_\phi=5^\circ
$$

则：

$$
\sigma_\Phi
=
\frac{5}{360}
$$

因此相位距离 jitter：

$$
\sigma_\rho
\approx
0.1903\times\frac{5}{360}
$$

约：

$$
2.64\text{ mm}
$$

这条关系对于分析 Carrier Distance LOG 非常有用。

---

### 31. 本章完整设计关系

可以总结成：

```text
C/N0
  │
  ├──→ Measurement Noise
  │
Tcoh
  │
  └──→ Correlator SNR
          ↓
      Discriminator Jitter
          ↓
           +             Dynamics
           │                │
           ↓                ↓
         Loop Filter / Bandwidth
                  ↓
          Tracking Error
                  ↓
       Lock / Slip / Measurement
```

---

### 32. 本章关键公式

PLL thermal jitter 常用近似：

$$
\sigma_\phi
\approx
\sqrt{
\frac{B_n}{C/N_0}
\left(
1+
\frac{1}
{2T_{coh}(C/N_0)}
\right)
}
$$

Coherent integration gain：

$$
G
=
10\log_{10}
\frac{T_2}{T_1}
$$

相关 SNR 核心尺度：

$$
(C/N_0)T
$$

PLL Jitter distance：

$$
\sigma_\rho
=
\lambda
\frac{\sigma_\phi}{2\pi}
$$

总体误差概念：

$$
e_{\text{total}}
\sim
e_{\text{noise}}
+
e_{\text{dynamic}}
+
e_{\text{bias}}
$$

---

### 33. 本章关键结论

1. Tracking lock 后仍存在随机 Jitter。
2. PLL Phase Jitter 随 $B_n$ 增大而增加，随 $C/N_0$ 增大而降低。
3. 增大 $T_{coh}$ 可以改善低 $C/N_0$ 下的单次 measurement quality。
4. 长积分受 residual Doppler、dynamic、data bit、secondary code 和 update rate 限制。
5. DLL Jitter 同样受 $B_n$、$C/N_0$、correlator spacing 和 frontend bandwidth 影响。
6. Dynamic Stress Error 即使在无 thermal noise 条件下也会存在。
7. Narrow bandwidth 降低 Noise Jitter，但增加 Dynamic Stress。
8. Wide bandwidth 提高 Dynamics Capability，但增加 Noise Jitter。
9. 总体最优 bandwidth 通常存在折中点。
10. 二阶 PLL 理想跟踪 constant frequency；三阶 PLL 更适合持续 frequency ramp / acceleration dynamics。
11. Carrier Aiding 可以显著减小 DLL 需要承担的 dynamics。
12. Lock Threshold 不是一个所有接收机统一的固定角度。
13. 低 $C/N_0$ 下 Cycle Slip Probability 会快速增加，而不是线性缓慢恶化。
14. $3\sigma$ 常用于早期 phase margin / jitter budget 评估，但不是严格失锁概率公式。
15. 弱信号 Tracking 通常联合使用 longer integration、narrower bandwidth、pilot/data wipeoff 和 aiding。
16. 弱信号并不意味着 bandwidth 可以无限缩窄，还必须满足 Dynamics Requirement。
17. Tracking Channel Active 和 Carrier Measurement Valid 是两个不同概念。
18. PLL phase jitter 可以直接通过 $\lambda\sigma_\phi/(2\pi)$ 转换为 Carrier Distance Jitter。

---

### 34. 下一章

下一章进入：

## 第 11 章｜实际 TRK Channel 数据流：从 Sample 到 Measurement

重点包括：

- 一个真实 Tracking Channel 内部到底有哪些硬件模块；
- Carrier / Code NCO 在 sample 级怎么运行；
- Correlator / Accumulator 如何输出 E/P/L I/Q；
- Hardware 和 Software / Firmware 如何分工；
- Discriminator、Loop Filter、NCO Update 的 timing；
- 1 ms、20 ms 等多时间尺度如何共存；
- Carrier Phase / Doppler / ADR / C/N0 最后从哪里输出；
- 多天线方案会新增多少 correlator、NCO、memory 与 backend resource。


---

<a id="chapter-11"></a>

## 第 11 章｜实际 TRK Channel 数据流：从 Sample 到 Measurement

### 1. 本章目标

本章把前 10 章的模块真正串成一条 Tracking Channel：

- Sample 级 Carrier / Code NCO 如何运行；
- Carrier Wipeoff、PRN 相关、E/P/L 如何形成；
- Accumulator 如何在积分窗口内工作；
- Discriminator、Loop Filter、NCO Update 如何衔接；
- Sample、1 ms、20 ms、Measurement Epoch 等多时间尺度如何共存；
- Hardware / Firmware / Software 如何分工；
- Doppler、Carrier Phase、ADR、C/N0、Pseudorange 从哪里来；
- 多天线方案会增加哪些 TRK 资源。

### 2. 一条完整 Tracking Channel

```text
                 ┌──── Carrier NCO ←──── PLL/FLL Loop
                 │
ADC / DFE Sample ─┼─ Carrier Wipeoff
                 │
                 └──── Code NCO ←────── DLL Loop
                          ↓
                    Local PRN Code
                          ↓
                  E / P / L Correlator
                          ↓
                   I/Q Accumulator
                          ↓
                 IE QE / IP QP / IL QL
                          ↓
         ┌────────────────┼────────────────┐
         ↓                ↓                ↓
        DLL              PLL              FLL
         ↓                ↓                ↓
      Loop Filter      Loop Filter      Loop Filter
         ↓                └──────┬─────────┘
     Code FCW                 Carrier FCW
         ↓                       ↓
     Code NCO              Carrier NCO
```

同时从 Tracking 状态生成：

```text
Pseudorange
Doppler
Carrier Phase
ADR
C/N0
Lock State
```

### 3. Sample 级高速数据流

假设：

$$
f_s = 20\text{ MHz}
$$

每个 sample 基本都需要执行：

```text
Carrier NCO phase update
        ↓
Local cos / sin
        ↓
Carrier Wipeoff
        ↓
Code NCO phase update
        ↓
Early / Prompt / Late PRN
        ↓
Complex correlation
        ↓
Accumulator += current sample
```

真正高速运行的核心模块是：

$$
\boxed{NCO + Mixer + Correlator + Accumulator}
$$

### 4. Carrier NCO

每个 sample：

$$
PHA_c[n+1]=PHA_c[n]+FCW_c
$$

根据 phase 产生本地 `cos/sin`，完成 Carrier Wipeoff：

$$
x_{bb}[n]=x[n]e^{-j\hat\theta[n]}
$$

Tracking 较准时：

$$
f_D-\hat f_D\approx0
$$

残余 phasor 不再快速旋转。

### 5. Code NCO

Code NCO 控制 Code Phase。GPS L1 C/A：

$$
R_c=1.023\text{ Mcps}
$$

若：

$$
f_s=20\text{ MHz}
$$

每个 chip 约：

$$
\frac{20}{1.023}\approx19.55
$$

个 samples。

因此一个 sample 绝不等于一个 chip。

### 6. E / P / L 的产生

设总 correlator spacing 为：

$$
d
$$

则：

$$
E=\hat\tau-\frac d2,\quad
P=\hat\tau,\quad
L=\hat\tau+\frac d2
$$

通常可以用一个主 Code NCO 加固定 offset 产生多个 correlator。

### 7. Correlator 与 Accumulator

Carrier wipeoff 后：

$$
z[n]=I[n]+jQ[n]
$$

Prompt：

$$
z_P[n]=z[n]c_P[n]
$$

一个积分窗口：

$$
I_P=\sum_n\Re\{z_P[n]\}
$$

$$
Q_P=\sum_n\Im\{z_P[n]\}
$$

同理得到：

$$
I_E,Q_E,I_L,Q_L
$$

传统 E/P/L Channel 基础输出：

$$
\boxed{I_E,Q_E,I_P,Q_P,I_L,Q_L}
$$

### 8. Integration Epoch

例如：

$$
T_{coh}=1\text{ ms}
$$

在：

$$
f_s=20\text{ MHz}
$$

条件下，一次积分包含：

$$
N=20000
$$

个 samples。

```text
20,000 samples
      ↓
Accumulator Freeze / Latch
      ↓
IE QE IP QP IL QL
      ↓
Accumulator Clear
      ↓
Next 1 ms
```

### 9. Discriminator

DLL：

$$
D_{DLL}=\frac{|E|-|L|}{|E|+|L|}
$$

PLL：

$$
D_{PLL}=\arctan\left(\frac{Q_P}{I_P}\right)
$$

Pilot / Data-wiped：

$$
D_{PLL}=\operatorname{atan2}(Q_P,I_P)
$$

FLL：

$$
Dot=I_kI_{k-1}+Q_kQ_{k-1}
$$

$$
Cross=Q_kI_{k-1}-I_kQ_{k-1}
$$

$$
D_{FLL}=\frac{\operatorname{atan2}(Cross,Dot)}{2\pi T}
$$

### 10. Hardware / Firmware 分工

典型高速 Hardware：

```text
ADC Sample → Carrier NCO → Mixer → Code NCO → PRN → Correlator → Accumulator
```

典型低速 Hardware / DSP / Firmware：

```text
IE/QE/IP/QP/IL/QL
      ↓
DLL / PLL / FLL Discriminator
      ↓
Loop Filter
      ↓
Carrier / Code FCW
      ↓
Write NCO
```

### 11. Loop Update 与 NCO 连续运行

不一定有：

$$
T_{coh}=T_{update}
$$

例如：

$$
T_{coh}=1\text{ ms},\quad
T_{disc}=1\text{ ms},\quad
T_{update}=20\text{ ms}
$$

但 NCO 仍然每个 sample 连续累加：

$$
PHA[n+1]=PHA[n]+FCW
$$

20 ms 到时只是在某个时刻更新新的 FCW。

### 12. Tracking = Predict + Measure + Correct

- NCO：Predict；
- Correlator + Discriminator：Measure；
- Loop Filter：Correct。

```text
Predict → Measure → Filter → Correct → Predict
```

### 13. Measurement 来源

```text
Code Phase → Pseudorange

Carrier FCW → Doppler

Carrier Accumulated Phase → Carrier Phase / ADR
```

C/N0 通常来自 correlator statistics。

### 14. Measurement Epoch

Measurement Engine 在统一 epoch latch：

- Code Phase；
- Carrier Phase；
- Carrier FCW；
- C/N0；
- Lock / ADR Valid。

如果两路只差：

$$
1\text{ ms}
$$

而：

$$
\dot\rho=100\text{ m/s}
$$

就可能产生：

$$
100\text{ mm}
$$

距离差。

因此毫米级多频 / 多天线比较必须严格做 epoch alignment。

### 15. 三层时间尺度

#### Sample Level

- Carrier NCO；
- Code NCO；
- Mixer；
- PRN；
- Correlator。

#### Tracking Level

- Accumulator latch；
- Discriminator；
- Loop Filter；
- NCO update；
- C/N0；
- Lock Detection。

#### Measurement Level

- Pseudorange；
- Doppler；
- Carrier Phase / ADR；
- Flags。

### 16. 1 ms × 20 与 20 ms Coherent Integration 不同

前者保留：

$$
P_1,\ldots,P_{20}
$$

可以继续做 FLL、bit wipeoff、phase correction、secondary-code wipeoff。

后者只得到一个 20 ms Prompt。

### 17. 多天线 TRK 资源

若单天线有 3 个 complex correlators，2 天线完全复制：

$$
3\times2=6
$$

4 天线：

$$
3\times4=12
$$

还可能增加：

- Carrier Mixer；
- Carrier NCO；
- Code NCO；
- Accumulator；
- Loop State；
- Memory / bandwidth。

小型阵列中 Code Phase 差异极小，因此 Code NCO / Code Tracking 具有一定共享可能；但 Carrier Relative Phase 必须保留。

### 18. 本章关键结论

1. Sample-level NCO / Mixer / Correlator / Accumulator 是 TRK 高速数据面。
2. Discriminator / Loop Filter 属于低速 Tracking Control Domain。
3. NCO 始终 sample-level 连续运行。
4. $T_{coh}$、$T_{disc}$、$T_{update}$、Measurement Rate 必须分开。
5. Tracking 本质是 Predict → Measure → Correct。
6. Carrier FCW 是 Doppler 的重要底层来源。
7. Carrier accumulated phase 是 Carrier Phase / ADR 的重要底层来源。
8. Code Phase 是 Pseudorange 的核心底层来源。
9. Measurement Engine 必须统一 epoch latch。
10. 多天线 TRK 不一定需要完整复制所有资源。
11. Code dynamics 有共享可能，但 antenna relative carrier phase 必须保留。


---

<a id="chapter-12"></a>

## 第 12 章｜弱信号与长积分 Tracking

### 1. 本章目标

本章重点理解：

- 为什么弱信号首先想到延长积分；
- 真正 20 ms coherent integration 与 1 ms × 20 个 Prompt 的区别；
- Residual Doppler、Dynamics、Data Bit、Secondary Code 为什么限制长积分；
- Data Wipeoff / Secondary-code Wipeoff；
- Coherent / Noncoherent Integration；
- Weak Tracking 为什么需要 Narrow Loop；
- Aiding 为什么重要；
- Multi-Antenna Combining 如何改善 Tracking。

### 2. 为什么延长积分

若每 1 ms：

$$
P_k=I_k+jQ_k
$$

若相位保持一致：

$$
P_{sum}=\sum_{k=0}^{N-1}P_k
$$

Signal amplitude：

$$
\propto N
$$

Noise RMS：

$$
\propto\sqrt N
$$

因此：

$$
SNR\propto N
$$

理想相干积分增益：

$$
\boxed{G=10\log_{10}N}
$$

### 3. 1 ms → 20 ms

$$
N=20
$$

因此：

$$
G=10\log_{10}(20)\approx13\text{ dB}
$$

### 4. 两种 20 ms 不同

#### 20 ms Coherent Integration

```text
Sample stream
   ↓
One accumulator for 20 ms
   ↓
One Prompt
```

#### 1 ms × 20 Prompt

```text
P1 P2 P3 ... P20
```

可继续做：

- phase alignment；
- bit wipeoff；
- secondary-code wipeoff；
- dynamic compensation；
- coherent combination。

### 5. 长积分第一限制：Residual Doppler

积分期间：

$$
\Delta\theta=2\pi\Delta fT
$$

相干幅度损失：

$$
\boxed{
L_f=
\left|
\frac{\sin(\pi\Delta fT)}
{\pi\Delta fT}
\right|
}
$$

所以：

$$
T\uparrow
$$

意味着 residual frequency tolerance 迅速下降。

### 6. 例子

若：

$$
\Delta f=20\text{ Hz}
$$

1 ms：

$$
\Delta\theta=7.2^\circ
$$

20 ms：

$$
\Delta\theta=144^\circ
$$

因此长 coherent integration 要求 Carrier Tracking 非常准。

### 7. 第二限制：Data Bit

GPS L1 C/A 导航数据：

$$
50\text{ bps}
$$

一个 bit：

$$
20\text{ ms}
$$

若跨 bit boundary，$+1/-1$ 会导致前后 coherent cancellation。

### 8. Data Wipeoff

如果已知：

$$
d[k]\in\{\pm1\}
$$

可做：

$$
P'_k=d[k]P_k
$$

把不同 bit sign 的 Prompt 重新旋到相同方向，再继续 coherent accumulation。

### 9. Pilot Channel

Pilot 没有未知 navigation data bit，因此更适合：

$$
Long\ Coherent\ Integration
$$

但很多 Pilot 仍有 Secondary Code。

### 10. Secondary Code Wipeoff

若 secondary code：

$$
s[k]\in\{\pm1\}
$$

则：

$$
P'_k=s[k]P_k
$$

可以消除 secondary-code sign，再继续长时间 coherent sum。

### 11. 第三限制：Dynamics

若：

$$
f_D(t)=f_0+\dot ft
$$

则：

$$
\phi(t)
=
2\pi
\left(
f_0t+\frac12\dot ft^2
\right)
$$

即使补掉 $f_0$，仍有：

$$
\phi_e(t)=\pi\dot ft^2
$$

因此 long coherent integration 对 frequency rate / acceleration 也敏感。

### 12. 高动态例子

若：

$$
\dot f=100\text{ Hz/s}
$$

$$
T=0.1s
$$

则：

$$
\Delta\phi\approx\pi\dot fT^2=\pi
$$

即约：

$$
180^\circ
$$

100 ms coherent integration 已可能严重失配。

### 13. Dynamic Compensation

若能够预测：

$$
f,\quad\dot f
$$

可生成：

$$
\hat\phi(t)
=
2\pi
\left(
\hat ft+
\frac12\hat{\dot f}t^2
\right)
$$

信息可来自：

- PLL / FLL；
- INS；
- PVT aiding；
- Vector Tracking。

### 14. Coherent Integration

直接复数相加：

$$
P_{coh}=\sum_kP_k
$$

优点：理论增益最高。

要求：

- Carrier phase alignment；
- Data sign known；
- Secondary code known；
- Dynamics compensated。

### 15. Noncoherent Integration

先取：

$$
M_k=|P_k|
$$

或：

$$
M_k=I_k^2+Q_k^2
$$

再：

$$
M_{sum}=\sum_kM_k
$$

优点：

- 不要求 phase coherence；
- 对 bit / phase flip 更鲁棒。

代价：

- squaring loss；
- noncoherent loss；
- noise bias。

### 16. Weak Tracking 为什么需要 Narrow Loop

弱信号：

$$
C/N_0\downarrow
$$

Discriminator Noise 上升。

若 $B_n$ 太宽，更多 noise 进入 NCO。

所以常见：

$$
\boxed{B_n\downarrow}
$$

以降低 Tracking Jitter。

但：

$$
B_n\downarrow
\Rightarrow
Dynamic\ Stress\uparrow
$$

所以不能无限缩窄。

### 17. Aiding

如果外部能预测 Doppler：

```text
INS / PVT / Other Channels
        ↓
Predicted Doppler
        ↓
Carrier NCO
        ↓
PLL tracks residual only
```

则 residual dynamics 减小，PLL 可以使用更窄 bandwidth。

### 18. Carrier Aiding DLL

Carrier Doppler 可以预测 Code Doppler：

$$
f_{D,code}
\approx
f_{code}
\frac{f_{D,carrier}}
{f_{carrier}}
$$

DLL 因此只需要跟 residual code error。

### 19. Weak Tracking State Machine

```text
Acquisition
   ↓
Short Tcoh + Wide FLL
   ↓
Carrier Pull-in
   ↓
PLL Stable
   ↓
Longer Tcoh + Narrower PLL
   ↓
Bit / Secondary-code Sync
   ↓
Long Coherent Integration
   ↓
Weak Tracking
```

若 dynamics 变大，可退回：

```text
Shorter Integration
+
Wider Loop
+
FLL Aiding
```

### 20. Integration Bandwidth 与 Loop Bandwidth 不同

Integration filter bandwidth 大致由：

$$
T_{coh}
$$

决定，量级：

$$
\sim\frac1{T_{coh}}
$$

发生在 Signal / Correlator Domain。

Loop Noise Bandwidth：

$$
B_n
$$

发生在 Discriminator / Control Domain。

因此：

```text
Tcoh → 单次 measurement 的 SNR / frequency selectivity

Bn   → 连续 measurement 驱动 NCO 的速度
```

### 21. Multi-Antenna 对 Weak Tracking 的帮助

若多个 antenna 可以正确相位对齐：

$$
y=\sum_mw_mx_m
$$

有效 $C/N_0$ 可以提高。

结果：

- PLL Jitter 降低；
- DLL Jitter 降低；
- Cycle-slip probability 降低；
- Weak tracking threshold 改善。

### 22. Tracking 前 Combine

```text
Ant1 ─┐
Ant2 ─┼→ Beamformer → One Tracking Channel
Ant3 ─┤
Ant4 ─┘
```

优点：

- 后级 TRK resource 少；
- Effective C/N0 高。

问题：

- 需要较早知道 antenna phase / weight；
- initialization 更复杂。

### 23. Correlator 后 Combine

```text
Ant1 → Correlator ─┐
Ant2 → Correlator ─┤
Ant3 → Correlator ─┼→ Combine
Ant4 → Correlator ─┘
```

优点：

- 保留每路 complex phase；
- 低速域易做 Beamforming；
- 适合阵列算法。

代价：

- Mixer / Correlator / Accumulator resource 增加。

### 24. 本章关键结论

1. Longer coherent integration 是 Weak Tracking 的核心手段之一。
2. 理想 coherent gain 为 $10\log_{10}N$。
3. 长积分要求更小 residual Doppler。
4. Unknown Data Bit 会限制 Data Channel integration。
5. Data Wipeoff 可以消除 bit sign。
6. Pilot 更适合 Long Coherent Integration。
7. Secondary Code 需要同步与 wipeoff。
8. Frequency Rate / Acceleration 会造成 long-window phase curvature。
9. 高动态下需要更强 prediction / aiding。
10. Coherent Integration 性能优于 Noncoherent，但要求更严格。
11. Weak Signal 通常需要 Narrower Loop，但不能忽略 Dynamic Stress。
12. Carrier Aiding 可让 DLL 使用更窄 bandwidth。
13. Weak Tracking 通常依赖动态 State Machine。
14. Integration Bandwidth 与 Loop Noise Bandwidth 是两个不同概念。
15. Multi-Antenna Combining 能通过提高有效 C/N0 改善 Weak Tracking。
16. Correlator 后 Combine 是阵列 GNSS 后级处理中非常值得评估的架构。

### 25. 下一章

## 第 13 章｜现代 GNSS Tracking：BOC / AltBOC / Pilot / Secondary Code / 多峰

重点：

- BOC 为什么会产生多峰；
- 为什么传统 E/P/L 可能 false lock；
- BOC ambiguity mitigation；
- Pilot / Data component；
- Secondary Code；
- Galileo E5 AltBOC；
- E5a / E5b 单边带与完整 AltBOC Tracking；
- 多 correlator 与资源开销。


---

<a id="chapter-13"></a>

## 第 13 章｜现代 GNSS Tracking：BOC / AltBOC / Pilot / Secondary Code / 多峰

### 1. 本章目标

这一章从传统 BPSK Tracking 进入现代 GNSS 信号，重点理解：

- BOC 为什么要引入副载波；
- BOC 为什么会让相关函数出现更多结构和多零点；
- 为什么传统 E/P/L DLL 可能发生 False Lock；
- BOC ambiguity mitigation 的基本思路；
- Data / Pilot 为什么要分开；
- Secondary Code 为什么影响长积分；
- Galileo E5 AltBOC(15,10) 的频率结构；
- E5a / E5b 单边带 Tracking 与 Full-band AltBOC Tracking 的区别；
- 为什么 Full AltBOC 能得到更尖锐的相关峰；
- 为什么 Full AltBOC 对 RF/DFE bandwidth、sample rate、local replica 和 correlator resources 要求更高；
- 多 correlator 在现代 GNSS Tracking 中为什么越来越常见。

---

## 2. 从 BPSK 到 BOC

传统 BPSK spreading 可以粗略写成：

$$
s(t)=c(t)d(t)\cos(2\pi f_ct)
$$

其中：

- $c(t)$：PRN code；
- $d(t)$：navigation data / pilot sign；
- $f_c$：RF carrier。

BOC（Binary Offset Carrier）在 PRN Code 外又乘了一个副载波：

$$
\boxed{
s(t)
=
c(t)d(t)s_{sc}(t)\cos(2\pi f_ct)
}
$$

其中：

$$
s_{sc}(t)\in\{+1,-1\}
$$

以比导航数据高得多的速率翻转。

---

## 3. BOC(m,n) 中的 m 和 n

常见定义：

$$
f_{sc}=m\times1.023\text{ MHz}
$$

$$
f_{code}=n\times1.023\text{ MHz}
$$

例如 BOC(1,1)：

$$
f_{sc}=1.023\text{ MHz}
$$

$$
f_{code}=1.023\text{ Mcps}
$$

所以一个 PRN chip 内，副载波还会发生符号翻转。

---

## 4. 为什么要加副载波

最直接的结果是：

> 把原来集中在载波附近的频谱能量搬到载波两侧。

概念上：

```text
传统 BPSK：

             main lobe
                ↓
-------------█████-------------
                fc


BOC：

       lobe                 lobe
        ↓                    ↓
------█████----------------█████------
              carrier center
```

这样做有几个重要意义：

- 与其它 GNSS 信号做频谱兼容；
- 利用更宽有效带宽；
- 获得更陡的 correlation feature；
- 提高 code-delay resolution；
- 改善某些 multipath / ranging performance。

---

## 5. BPSK 与 BOC 自相关函数的区别

传统理想 BPSK chip 的 ACF 近似三角形：

$$
R(\tau)\approx1-|\tau|
$$

在主峰附近只有一个非常直观的大峰。

BOC 因为一个 chip 内还有副载波翻转，所以相关函数会出现额外的：

- 正峰；
- 负峰；
- 局部结构。

配套图：

```text
GNSS_Tracking_第13章_BPSK与BOC自相关对比.png
```

图中 BOC 曲线为简化的 BOC(1,1)-like chip pulse 数值示意，主要用于理解多结构，不代表某一实际卫星完整 PRN 的精确 ACF。

---

## 6. 为什么 BOC 主峰会更尖

副载波提高了信号中的高频成分。

从 delay estimation 角度：

> Effective bandwidth 越宽，通常对微小 delay change 越敏感。

所以 BOC / AltBOC 往往能形成比低 chip-rate BPSK 更陡、更窄的主相关特征。

这也是现代 GNSS 广带信号能够获得高 code ranging precision 的重要原因。

但：

> 更尖的峰不是免费得到的，它需要前端真正保留这些带宽。

如果 RF / DFE bandwidth 太窄，原本的高频结构会被滤掉，相关峰也会变钝。

---

## 7. BOC 最大的问题：相关函数有多峰 / 多零点

传统 DLL：

$$
D(\Delta\tau)=E-L
$$

或者：

$$
D(\Delta\tau)
=
\frac{E-L}{E+L}
$$

依靠：

$$
D=0
$$

来判断 Code Lock Point。

对于传统 BPSK：

> 主峰附近通常只有目标 zero crossing。

但 BOC 相关函数内部结构更多，所以 E-L discriminator 也可能产生多个：

$$
D=0
$$

的位置。

---

## 8. False Lock 是怎么来的

DLL 并不知道：

> “哪个 zero crossing 是真正的主峰。”

它只知道：

$$
D\rightarrow0
$$

如果 Acquisition 给出的初始 code phase 偏到某个 side peak 附近，DLL 可能稳定在错误的 zero crossing。

也就是：

$$
\boxed{
False\ Lock
}
$$

配套图：

```text
GNSS_Tracking_第13章_BOC多零点与FalseLock.png
```

图中可以看到目标：

$$
\Delta\tau=0
$$

之外，还可能存在其它 discriminator zero crossing。

---

## 9. False Lock 为什么危险

假设真正 delay：

$$
\tau
$$

但 DLL 锁到了：

$$
\tau+\Delta\tau_{false}
$$

那么：

- Tracking Channel 看起来“锁住了”；
- E-L discriminator 可能接近 0；
- Prompt 甚至仍然存在不小能量；

但 Code Measurement 已经有固定 bias。

因此：

> Lock Detector 只看 DLL discriminator 是否接近 0 是不够的。

---

## 10. BOC Ambiguity Mitigation 的基本思想

具体算法很多，但可以先理解成四大类。

### 10.1 更好的 Acquisition 初始化

尽量在 Acquisition 阶段就确认：

> 当前峰是否是 central main peak。

这样 DLL 一开始就进入正确 attraction region。

---

### 10.2 Bump Jumping

Tracking 时除了 E/P/L 再看更外侧 correlator：

```text
VE   E   P   L   VL
```

如果发现：

$$
|VE|>|P|
$$

或者：

$$
|VL|>|P|
$$

等异常模式，

说明可能锁在 side peak。

然后主动把 Code Phase：

> jump 到相邻更强 peak。

---

### 10.3 Unambiguous Discriminator

利用多个 correlator、不同组合或经过处理后的相关函数，构造一个：

> 主峰附近只有一个稳定 zero crossing

的 discriminator。

目标不是改变实际信号，而是重新设计：

$$
D(\tau)
$$

让 ambiguity 消失或大幅减弱。

---

### 10.4 两级 / 双估计器 Tracking

同时维护：

- coarse / unambiguous delay estimator；
- fine / narrow peak estimator。

粗估计保证不锁错峰；

细估计利用窄峰获得高 precision。

这是一种非常常见的现代 Tracking 思路：

$$
\boxed{
Coarse\ Robustness
+
Fine\ Precision
}
$$

---

## 11. 为什么现代 GNSS 越来越喜欢多 correlator

传统 GPS L1 C/A：

```text
E P L
```

已经能工作。

但 BOC / AltBOC / multipath / spoofing 场景中，仅三个点越来越不够。

可能扩展成：

```text
VE  E  P  L  VL
```

甚至：

```text
C[-7] ... C[-1] P C[+1] ... C[+7]
```

例如 15 路 correlator。

多 correlator 能观察到：

> 整个 correlation shape

而不是只看：

$$
E,\ P,\ L
$$

三个点。

---

## 12. 多 correlator 能做什么

不仅是 DLL：

- BOC false-lock detection；
- Multipath mitigation；
- Signal Quality Monitoring；
- Spoofing detection；
- Interference distortion detection；
- Full correlation-function reconstruction；
- Advanced discriminator；
- Antenna / channel combining。

所以现代 TRK 的 Correlator Bank 往往也是非常重要的硬件资源。

---

## 13. Data 与 Pilot

现代 GNSS 常把功率分给两个 component：

```text
Data Component
+
Pilot Component
```

Data 上：

$$
PRN\times Navigation\ Data
$$

Pilot 上通常没有未知 Navigation Data：

$$
PRN\times Secondary\ Code
$$

或者类似结构。

---

## 14. 为什么 Pilot 特别适合 Tracking

Data Bit 未知时：

$$
+1\leftrightarrow-1
$$

会导致 phasor：

$$
180^\circ
$$

翻转。

因此：

- PLL 需要 Costas；
- Long Coherent Integration 需要 Data Wipeoff。

Pilot 没有未知 nav-data bit，所以：

- Carrier Phase 更容易连续跟踪；
- 可以使用 full-angle PLL；
- 更容易做 Long Coherent Integration；
- Weak Tracking 性能通常更好。

---

## 15. 但 Pilot 为什么还有 Secondary Code

如果 Pilot 完全不调制任何东西：

> 结构可能不利于码设计、同步或系统复用。

因此很多 Pilot 会再乘一个已知 Secondary Code：

$$
p(t)=c_{primary}(t)c_{secondary}(t)
$$

Primary Code 可能每：

$$
1\text{ ms}
$$

重复，

但 Secondary Code 每个 primary-code epoch 改一次 sign。

所以：

```text
P1  P2  -P3  P4 ...
```

要做超长 coherent integration，仍需要先知道 secondary-code position。

---

## 16. Secondary-Code Tracking 可以看成更慢一级 Code Tracking

可以把时间尺度理解成：

```text
Sample Level
    ↓
Primary PRN Code
    ↓
1 ms Prompt
    ↓
Secondary Code Sequence
    ↓
10/20/... ms long integration
```

因此现代 Tracking 本身就天然具有多层时间尺度。

---

## 17. 进入 Galileo E5

Galileo E5 是一个非常典型的宽带现代 GNSS Signal。

其中心频率：

$$
f_{E5}=1191.795\text{ MHz}
$$

两侧分别是：

$$
f_{E5a}=1176.45\text{ MHz}
$$

$$
f_{E5b}=1207.14\text{ MHz}
$$

两者相对 E5 center 的 offset：

$$
\pm15.345\text{ MHz}
$$

配套图：

```text
GNSS_Tracking_第13章_GalileoE5_AltBOC频谱布局.png
```

---

## 18. Galileo E5 的四个主要 component

经典 E5 structure 中包含：

```text
E5a-I   Data
E5a-Q   Pilot

E5b-I   Data
E5b-Q   Pilot
```

这些 component 的 ranging-code chip rate 都是：

$$
10.230\text{ Mchip/s}
$$

经典 Open Service 定义中：

- E5a-I 为 Data；
- E5a-Q 为 Pilot；
- E5b-I 为 Data；
- E5b-Q 为 Pilot。

---

## 19. AltBOC(15,10)

Galileo E5 wideband 使用：

$$
\boxed{
AltBOC(15,10)
}
$$

其：

$$
f_{sc}=15\times1.023
=
15.345\text{ MHz}
$$

而：

$$
f_{code}
=
10\times1.023
=
10.23\text{ MHz}
$$

因此：

$$
AltBOC(15,10)
$$

中的 15 和 10 就分别对应：

- subcarrier rate；
- primary ranging-code rate。

---

## 20. 为什么 E5a / E5b 分别落在两侧

由于 AltBOC subcarrier：

$$
15.345\text{ MHz}
$$

所以 E5 center：

$$
1191.795\text{ MHz}
$$

两侧形成：

$$
1191.795-15.345
=
1176.45\text{ MHz}
$$

即 E5a。

以及：

$$
1191.795+15.345
=
1207.14\text{ MHz}
$$

即 E5b。

这不是巧合，而就是 AltBOC frequency structure 的核心。

---

## 21. E5a / E5b 可以单独 Tracking

如果 RF / DFE 只保留一侧，例如 E5a，

那么 receiver 可以把它看成一个独立的宽带 GNSS component 来处理。

概念上：

```text
Full E5 AltBOC
      ↓ DFE Filter
Only E5a
      ↓
E5a-I / E5a-Q Tracking
```

E5b 同理。

---

## 22. 单边 Tracking 的优势

只跟 E5a 或 E5b：

- RF bandwidth 要求较低；
- ADC / DFE sample rate 要求降低；
- local replica 简单；
- correlator complexity 低；
- power / area 更低。

对于手机 receiver：

> 这是非常现实的架构优势。

---

## 23. 但单边 Tracking 丢掉了什么

如果只使用 E5a：

> E5b 的频谱信息就没有参与 delay estimation。

相当于没有利用完整 E5 的：

$$
\sim30.69\text{ MHz}
$$

两个主要 sideband center separation。

因此 Full-band AltBOC 可以利用更大的 effective bandwidth。

更大的 effective bandwidth：

$$
\Rightarrow
\text{Sharper correlation structure}
$$

$$
\Rightarrow
\text{Potentially better code-delay precision}
$$

$$
\Rightarrow
\text{Better multipath resolution potential}
$$

---

## 24. Full AltBOC Tracking 在干什么

Full-band receiver 同时保留：

```text
E5a
+
E5b
```

并使用符合 AltBOC modulation structure 的本地 replica。

因此 correlation 不再只是：

$$
r(t)\times c_{E5a}(t)
$$

或者：

$$
r(t)\times c_{E5b}(t)
$$

而是对完整 wideband waveform 做 matched correlation。

概念上：

$$
\boxed{
R_{AltBOC}(\tau)
=
\int
r(t)s^*_{AltBOC}(t-\tau)dt
}
$$

---

## 25. 为什么 Full AltBOC Local Replica 更复杂

AltBOC 不只是：

$$
PRN\times简单方波副载波
$$

它要同时组合：

- E5a-I；
- E5a-Q；
- E5b-I；
- E5b-Q；

以及 AltBOC multiplexing 中的 subcarrier / product-term structure。

所以硬件实现经常采用：

- LUT；
- 分段 subcarrier state；
- finite complex phase states；
- I/Q sign / rotation combination；

而不是每个 sample 真去计算复杂三角函数。

---

## 26. 为什么会看到 8-state / LUT 类实现

AltBOC waveform 的复包络只会落在有限的复相位状态集合中。

因此工程上很自然：

```text
Code states
+
Subcarrier state
      ↓
LUT Address
      ↓
Local I/Q Replica
```

相较直接实时计算：

> LUT 对芯片更友好。

所以如果 TRK RTL 中出现：

- phase segment；
- subcarrier column；
- 8-state；
- LUT；

通常是在把 AltBOC 数学表达式压缩成低成本的数字逻辑。

---

## 27. Full AltBOC 为什么需要更高 Sample Rate

如果要利用完整 wideband waveform，就必须在 ADC / DFE / TRK 中保留足够带宽。

Nyquist 的最基本要求：

$$
f_s>2B
$$

但真实 receiver 还需要：

- transition band；
- anti-alias margin；
- digital filter implementation；
- correlator timing resolution。

所以实际 sample rate 往往明显高于最低理论边界。

---

## 28. Sample Rate 不足会怎么样

主要有两种情况。

### 情况 A：RF/DFE Bandwidth 已经把高频信息滤掉

此时：

> 后面再提高 correlator 数量也救不回来。

因为信息已经不存在。

### 情况 B：带宽保留，但 Sampling 太粗

会产生：

- code-delay discretization；
- local replica approximation error；
- correlation-shape distortion；
- poorer interpolation。

可以通过：

- fractional Code NCO；
- interpolation；
- polyphase；
- oversampling

改善，但前提仍是前端频谱没丢。

---

## 29. 为什么 61 MHz 与 122 MHz 这种 Sample Rate 会有差异

对于宽带 AltBOC：

更高 sample rate 可以：

- 更细地表示副载波 transition；
- 更细地表示 correlation peak；
- 降低 local-replica quantization；
- 给 fractional correlator spacing 更多实现余量。

但：

> Sample Rate 翻倍并不意味着 Tracking Gain 自动增加 3 dB。

真正性能由：

- preserved RF bandwidth；
- quantization；
- replica accuracy；
- discriminator；
- correlator spacing；
- integration architecture

共同决定。

---

## 30. Correlator Spacing 为什么在 AltBOC 中更敏感

传统 BPSK correlation peak 相对宽。

AltBOC full-band correlation feature 更尖。

所以：

$$
d_{EL}
$$

选得太大：

> 可能没有充分利用 steep slope。

选得太小：

- Noise correlation 增大；
- sample/interpolation resolution 要求高；
- fixed-point / timing error 影响更明显。

所以最佳 spacing 必须和实际：

$$
RF\ Bandwidth
+
Sample\ Rate
+
Discriminator
$$

一起设计。

---

## 31. 为什么 Full AltBOC 可能需要更多 correlator

因为它同时存在：

-复杂 correlation shape；
- wideband timing sensitivity；
- possible ambiguity / side structure；
- Data / Pilot components；
- E5a / E5b information。

因此工程实现可能使用：

$$
>3
$$

个 taps。

例如每侧：

$$
15
$$

路 correlator，

或者针对完整 waveform 设计 dense correlator bank。

这会直接影响：

- multiplier resources；
- accumulator width；
- memory；
- power；
- bus bandwidth；
- firmware computation。

---

## 32. E5a/E5b 两边独立相关之后还能怎么合并

假设得到：

$$
P_a
$$

与：

$$
P_b
$$

可以有多种思路。

### 非相干合并

$$
M
=
|P_a|^2+|P_b|^2
$$

优点：

- 不需要知道两边绝对 phase relation；
- 实现简单。

缺点：

> 丢掉两边之间的 phase information。

---

### 相干合并

如果两边 phase / modulation relation 已知并校准：

$$
P
=
P_a e^{-j\phi_a}
+
P_b e^{-j\phi_b}
$$

理论上可以利用更多 signal information。

但要求：

- inter-band phase relationship；
- RF chain phase；
- group delay；
- local replica phase；

都必须处理正确。

---

## 33. 为什么合并收益可能超过“简单 3 dB”

如果两个完全独立、等功率、等质量的 measurement 只做标准功率意义的 combine，

最熟悉的结果是：

$$
3\text{ dB}
$$

但实际比较中如果看到：

$$
>3\text{ dB}
$$

不一定说明物理定律被突破。

因为你比较的量可能不是：

> 两个完全等功率独立信号的输入 SNR。

例如：

- 单边相关模板并非完整 matched filter；
- Full-band template 利用了 cross-sideband structure；
- baseline noise estimation 不同；
- correlation peak sampling 不同；
- 单边带 bandwidth 不同；
- normalization 不一致；
- sidelobe 被计入 noise；
- Data/Pilot energy composition 不一致。

所以必须首先明确：

$$
\boxed{
Gain\ Definition
}
$$

到底是什么。

---

## 34. Full AltBOC vs 两边独立 Tracking

可以粗略总结：

| 项目 | E5a/E5b 单边处理 | Full AltBOC |
|---|---|---|
| RF/DFE Bandwidth | 较低 | 很高 |
| Sample Rate | 较低 | 较高 |
| Local Replica | 简单 | 复杂 |
| Correlator Resource | 较少 | 更多 |
| Power | 较低 | 较高 |
| Code Precision Potential | 高 | 更高 |
| Wideband Multipath Resolution | 较弱 | 更好潜力 |
| 手机落地难度 | 较低 | 较高 |

因此：

> Full AltBOC 并不是“必然更划算”，而是用 Hardware Cost 换更完整的 Wideband Information。

---

## 35. 手机 receiver 为什么特别需要做 Tradeoff

手机环境非常受限：

- ADC / DFE power；
- memory bandwidth；
- TRK multiplier count；
- accumulator area；
- clock frequency；
- always-on power；
- thermal budget。

因此 Full AltBOC 是否值得做，不能只问：

> Tracking 性能是不是更好？

还必须问：

$$
\boxed{
Performance\ Gain / Hardware\ Cost
}
$$

---

## 36. 对后级 TRK 资源评估应看什么

建议拆成：

#### Correlator

$$
N_{antenna}
\times
N_{satellite}
\times
N_{signal}
\times
N_{tap}
$$

#### Complex Accumulators

每个 correlator 至少有：

$$
I+Q
$$

两路 accumulation。

#### Integration Width

长积分：

$$
T_{coh}\uparrow
$$

Accumulator bit width 也要增加。

#### Replica Generation

- PRN generators；
- Secondary-code；
- AltBOC LUT；
- subcarrier state。

#### NCO

- Carrier NCO；
- Code NCO；
- possible shared NCO。

#### Firmware

- discriminator；
- ambiguity detection；
- combining；
- lock state；
- phase calibration。

---

## 37. 多天线 + AltBOC 为什么资源增长特别快

假设：

- 4 antennas；
- 15 correlators；
- complex I/Q；

仅一个 signal / satellite：

$$
4\times15=60
$$

个 complex correlator outputs。

也就是：

$$
120
$$

个 I/Q accumulators。

再乘：

- satellites；
- E5a/E5b；
- Data/Pilot；

资源会很快爆炸。

所以阵列方案几乎一定要考虑：

- 什么能共享；
- 什么必须每 antenna 保留；
- 在 sample domain combine 还是 correlator domain combine；
- 是否所有 tap 都需要每 antenna 复制。

---

## 38. 一个非常重要的架构选择

### 方案 A：每 antenna 完整 Wideband Tracking

```text
Ant1 → Full AltBOC TRK
Ant2 → Full AltBOC TRK
Ant3 → Full AltBOC TRK
Ant4 → Full AltBOC TRK
```

最灵活，但资源最大。

---

### 方案 B：共享 Code / Frequency，保留 antenna phase

```text
            Common Code / Doppler
                   ↓
       ┌───────────┼───────────┐
       ↓           ↓           ↓
     Ant1        Ant2        Ant3...
       ↓           ↓           ↓
 Relative Complex Correlation
       ↓           ↓           ↓
       └──── Array Processing ───┘
```

更适合阵列场景。

---

### 方案 C：先 Beamforming，再 Full TRK

```text
Ant1 ─┐
Ant2 ─┼→ Beamforming → AltBOC TRK
Ant3 ─┤
Ant4 ─┘
```

TRK 后级资源最省，

但 beamformer 必须提前获得可靠：

- direction / phase；
- calibration；
- weight。

因此 initialization 很难。

---

## 39. 本章最核心的认识

传统 Tracking 的目标是：

> 找到一个 correlation peak，并守住它。

现代 GNSS Tracking 则变成：

> 在复杂 multi-component / multi-peak / wideband waveform 中，既保证不锁错峰，又尽量利用完整 bandwidth 获得更高 precision，同时控制 hardware cost。

也就是：

$$
\boxed{
Robustness
+
Precision
+
Complexity
}
$$

三者之间的 Tradeoff。

---

## 40. 本章关键结论

1. BOC 在 PRN Code 外增加 Subcarrier，因此频谱能量分布到 Carrier 两侧。
2. 更宽 Effective Bandwidth 可以提高 Delay Sensitivity。
3. BOC ACF 比传统 BPSK 更复杂，存在多个局部结构。
4. E-L Discriminator 因此可能出现多个 Zero Crossing。
5. DLL 可能稳定在错误 Zero Crossing，形成 False Lock。
6. BOC Tracking 通常需要 ambiguity detection / mitigation。
7. Extra Correlators 可以观察 correlation shape，而不仅是 E/P/L 三点。
8. Pilot 没有未知 Navigation Data，更适合 Carrier Tracking 与 Long Coherent Integration。
9. Secondary Code 仍需要同步和 Wipeoff，才能继续延长 Coherent Integration。
10. Galileo E5 由 E5a / E5b 组成，中心分别为 1176.45 MHz 与 1207.14 MHz。
11. E5 wideband center 为 1191.795 MHz，两边相距中心 ±15.345 MHz。
12. AltBOC(15,10) 的 subcarrier rate 为 15.345 MHz，ranging-code chip rate 为 10.23 Mcps。
13. E5a-I / E5b-I 是 Data component，E5a-Q / E5b-Q 是 Pilot component。
14. E5a / E5b 可以单边独立 Tracking，Hardware Cost 较低。
15. Full AltBOC Tracking 能利用更大的 Effective Bandwidth，Code Precision Potential 更高。
16. Full AltBOC 需要更宽 RF/DFE bandwidth、更高 sample rate、更复杂 local replica 和更多 correlator resources。
17. 提高 Sample Rate 本身不会自动带来对应的 dB Tracking Gain，关键是保留了多少有效 Signal Bandwidth。
18. Correlator spacing 必须结合 RF bandwidth、sample rate 与 actual ACF 一起设计。
19. E5a/E5b 可以非相干或相干合并，但两者需要的 phase knowledge 不同。
20. 看到超过 3 dB 的“合并收益”时，首先检查 Gain Definition、Normalization、Noise Estimation 和 Template 是否一致。
21. 多天线 × 多 correlator × Full AltBOC 会导致 TRK 资源快速增长。
22. 多天线架构必须评估 Code/Doppler sharing、relative phase preservation 和 combining domain。
23. Full AltBOC 是否值得手机落地，本质是 Performance Gain / Hardware Cost 的工程 Tradeoff。

---

## 41. 下一章

## 第 14 章｜Vector Tracking / Deep Coupling

最后一章重点：

- Scalar Tracking Loop 的局限；
- Vector Tracking 为什么把多个 satellite channel 联合起来；
- PVT 如何反过来预测各 satellite Doppler / Code Rate；
- Channel measurement residual 如何进入 Navigation Filter；
- INS aiding 与 Deep Coupling；
- Weak / Blocked satellite 如何被其它 satellites / INS 帮助；
- Vector Tracking 与多天线阵列处理如何结合；
- 手机 GNSS 中什么时候值得使用 Vector / Deep Coupling。


---

<a id="chapter-14"></a>

## 第 14 章｜Vector Tracking / Deep Coupling

### 1. 本章目标

本章是整个 GNSS Tracking 课程的收尾，重点理解：

- Scalar Tracking 与 Vector Tracking 的本质区别；
- 为什么传统每星独立 DLL / PLL 在弱信号和遮挡下容易失败；
- Vector Tracking 如何利用所有卫星共同估计接收机状态；
- PVT / Navigation Filter 如何反过来预测每颗卫星的 Code Rate 与 Doppler；
- Channel residual 如何进入 EKF / Navigation Filter；
- 为什么 Vector Tracking 能帮助弱星维持 Tracking；
- Vector Tracking 与 Carrier Aiding、FLL/PLL 的关系；
- INS aiding、Tightly Coupled、Deep Coupling 分别是什么；
- Deep Coupling 为什么直接使用 Tracking residual / correlator information；
- Vector Tracking 与多天线阵列处理如何结合；
- 手机 GNSS 中什么时候值得采用 Vector / Deep Coupling。

---

## 2. 先回顾 Scalar Tracking

我们前面 13 章讲的基本都是 Scalar Tracking。

每一颗卫星拥有自己独立的：

```text
Code NCO
Carrier NCO
DLL
PLL / FLL
Loop Filter
```

例如：

```text
SV1 → DLL / PLL → Measurement ┐
SV2 → DLL / PLL → Measurement ├→ PVT
SV3 → DLL / PLL → Measurement ┤
SV4 → DLL / PLL → Measurement ┘
```

每颗卫星先独立保证：

$$
\Delta\tau_i\rightarrow0
$$

$$
\Delta f_i\rightarrow0
$$

然后才把：

- Pseudorange；
- Doppler；
- Carrier Phase

交给 PVT。

这就是：

$$
\boxed{
Tracking\ First
\rightarrow
Navigation\ Later
}
$$

---

## 3. Scalar Tracking 的优点

Scalar 架构非常成熟。

优点：

- Channel 之间相互独立；
- 实现简单；
- Debug 容易；
- 某颗卫星坏了不容易污染其它 Channel；
- Loop 参数可以按每颗星单独调；
- Hardware / Firmware 架构清晰。

所以绝大多数传统接收机都会以 Scalar Tracking 为基础。

---

## 4. Scalar Tracking 的局限

问题在于：

> 每颗卫星必须依靠自己的 correlator measurement 维持自己的 Tracking。

如果某颗星：

$$
C/N_0\downarrow
$$

或者被：

- 建筑遮挡；
- 树叶衰减；
- 干扰；
- 天线方向图 null；
- multipath

影响，

它自己的：

$$
PLL/DLL\ discriminator
$$

会迅速变吵。

即使其它 10 颗卫星都非常强：

> 传统 Scalar Loop 也不会直接用其它星帮助它。

---

## 5. 但所有卫星其实不是彼此独立的

因为所有 satellite measurements 都来自同一个：

$$
\boxed{
Receiver\ Position,\ Velocity,\ Clock
}
$$

例如卫星 $i$ 的 Range Rate：

$$
\dot\rho_i
=
\mathbf u_i^T
(\mathbf v_r-\mathbf v_{s,i})
$$

其中：

- $\mathbf u_i$：Satellite-Receiver LOS unit vector；
- $\mathbf v_r$：Receiver Velocity；
- $\mathbf v_{s,i}$：Satellite Velocity。

Carrier Doppler approximately：

$$
f_{D,i}
\approx
-\frac{1}{\lambda_i}
\dot\rho_i
+
f_{clk}
$$

所以所有卫星 Doppler 并不是 20 个完全独立的随机量。

它们背后共享：

$$
\boxed{
Receiver\ Velocity + Clock\ Drift
}
$$

---

## 6. Code Rate 也是共同状态决定的

同理，Code Doppler：

$$
f_{code,i}
\approx
f_{code,0}
\left(
1-\frac{\dot\rho_i}{c}
\right)
$$

因此：

- Carrier Doppler；
- Code Rate；
- Pseudorange Rate

都和同一个 Receiver Motion 有关。

这就提供了一个机会：

> 为什么一定要每颗星独立估 Doppler？

可以先估：

$$
Receiver\ Velocity
$$

然后预测每颗星应该看到多少 Doppler。

---

## 7. 这就是 Vector Tracking

Vector Tracking 的核心变化：

```text
SV1 Correlator ─┐
SV2 Correlator ─┤
SV3 Correlator ─┼→ Navigation Filter
SV4 Correlator ─┘
                     ↓
              Receiver State
                     ↓
       ┌─────────────┼─────────────┐
       ↓             ↓             ↓
  Predict SV1    Predict SV2    Predict SV3
 Code/Doppler    Code/Doppler    Code/Doppler
```

所以它不是：

$$
\text{每星独立估计频率}
$$

而是：

$$
\boxed{
所有 Channel 共同估计 Receiver State
}
$$

再从 Receiver State 预测每颗星的：

- Code Phase Rate；
- Carrier Doppler；
- possible Carrier Phase evolution。

---

## 8. Scalar vs Vector 架构

配套图：

```text
GNSS_Tracking_第14章_Scalar与VectorTracking架构对比.png
```

Scalar：

```text
Channel Tracking
      ↓
Measurement
      ↓
PVT
```

Vector：

```text
Correlator Residual
      ↓
Navigation Filter
      ↓
Receiver State
      ↓
Predict each channel
      ↓
NCO
```

最大的区别是：

$$
\boxed{
PVT\ feedback\ enters\ Tracking
}
$$

---

## 9. Vector Tracking 中 Navigation Filter 在估什么

一个基础 state vector 可以包含：

$$
\mathbf x
=
[
x,y,z,
v_x,v_y,v_z,
b,\dot b
]^T
$$

即：

- Receiver Position；
- Receiver Velocity；
- Clock Bias；
- Clock Drift。

更复杂系统还可能增加：

- Acceleration；
- Oscillator model；
- Troposphere；
- IMU bias；
- Attitude。

---

## 10. 从 State Prediction 到 Satellite Doppler

Navigation Filter 给出预测：

$$
\hat{\mathbf x}_{k|k-1}
$$

因此可以计算：

$$
\hat{\rho}_i
$$

和：

$$
\hat{\dot\rho}_i
$$

于是：

$$
\boxed{
\hat f_{D,i}
\approx
-\frac{\hat{\dot\rho}_i}{\lambda_i}
+
\hat f_{clk}
}
$$

再换成：

$$
FCW_i
=
\frac{
f_{nom,i}+\hat f_{D,i}
}{f_s}2^N
$$

写入 Carrier NCO。

所以 Carrier NCO 现在主要依赖：

> Navigation Prediction。

---

## 11. Code NCO 同样可以预测

根据：

$$
\hat{\dot\rho}_i
$$

可以预测 Code Rate：

$$
\hat f_{code,i}
$$

从而驱动：

$$
Code\ NCO_i
$$

所以：

```text
Navigation Filter
      ├→ Carrier Doppler Prediction → Carrier NCO
      └→ Code Rate Prediction       → Code NCO
```

---

## 12. 那 DLL / PLL 还要不要？

要。

Vector Tracking 并不是：

> 不再需要 correlator / discriminator。

而是这些 discriminator 输出的角色变了。

Scalar Tracking：

$$
D_i
\rightarrow
LoopFilter_i
\rightarrow
NCO_i
$$

Vector Tracking：

$$
D_1,D_2,\ldots,D_N
\rightarrow
Navigation\ Filter
\rightarrow
\mathbf x
\rightarrow
NCO_1,\ldots,NCO_N
$$

所以原来的多个独立 Loop Filter：

> 被一个联合 Navigation Filter 部分取代或增强。

---

## 13. Channel Residual 是什么

假设 Navigation Filter 预测某颗星应该：

$$
\hat f_D=1200\text{ Hz}
$$

但 correlator / FLL measurement 认为 residual：

$$
+3\text{ Hz}
$$

这个：

$$
3\text{ Hz}
$$

就是一个：

$$
\boxed{
Innovation / Residual
}
$$

它告诉 Navigation Filter：

> 你预测的 Receiver State 还有一点错误。

Navigation Filter 再综合其它卫星 residual 修正：

- Velocity；
- Clock Drift。

---

## 14. 为什么弱星能被强星帮助

假设：

```text
SV1 强
SV2 强
SV3 强
SV4 很弱
```

SV1～SV3 已经可以很好估计：

$$
Receiver\ Velocity + Clock\ Drift
$$

那么即使 SV4 的 FLL / PLL measurement 非常 noisy，

Navigation Filter 仍然可以预测：

$$
f_{D,4}
$$

大概应该是多少。

于是 SV4 NCO 不会因为自己的 noise 就乱跑。

这就是：

$$
\boxed{
Cross-channel aiding
}
$$

---

## 15. 图：弱星得到 Vector Aiding

配套图：

```text
GNSS_Tracking_第14章_VectorAiding弱信号示意.png
```

图是概念示意：

- Weak Scalar Estimate 在弱信号区间噪声迅速增加；
- Vector-aided estimate 由于其它 channels 提供 Receiver-State Constraint，仍然可以保持更平稳。

这不是某一实际 receiver 的性能曲线。

---

## 16. Vector Tracking 为什么可以用更窄的 Channel Loop

Scalar PLL 必须自己承担：

- Receiver dynamics；
- oscillator drift；
- measurement noise。

Vector Tracking 已经通过 Navigation Filter 预测了绝大部分 common dynamics。

Channel residual 变成：

$$
\text{small correction}
$$

于是每个 Channel 不需要那么大的：

$$
B_n
$$

这和前面讲的 Aiding 思想完全一致。

---

## 17. Vector Tracking 与 Carrier Aiding 的关系

Carrier Aiding DLL：

```text
Carrier Loop
    ↓
Code Doppler Prediction
    ↓
DLL
```

Vector Tracking：

```text
Navigation State
     ↓
Carrier Doppler Prediction
     ↓
Code Doppler Prediction
     ↓
All Channels
```

所以 Vector Tracking 可以看成：

> 把 Aiding 从“单通道内部”提升到了“全接收机联合状态”。

---

## 18. Vector Tracking 的主要收益

理论和工程上常期待：

#### Weak Signal Robustness

弱星可以依赖其它强星的 navigation state。

#### Short Blockage Bridging

短时间 signal blocked 时：

$$
NCO
$$

还能根据 state prediction 继续运行。

#### High Dynamics

Navigation motion model 可以提前预测 Doppler。

#### Improved Reacquisition

信号回来时，Code / Doppler uncertainty 较小。

#### Consistency

所有 Channel 被同一个 Receiver Motion Model 约束。

---

## 19. 但 Vector Tracking 也有明显风险

Scalar 中某颗星坏了：

> 主要影响自己。

Vector 中某颗星出现错误 measurement：

$$
bad\ residual
$$

可能进入 Navigation Filter。

如果 filter 没有 robust detection：

> 错误可能传播到其它 channels。

所以 Vector Tracking 非常依赖：

- Measurement Quality Control；
- Innovation Check；
- Outlier Rejection；
- C/N0 weighting；
- Multipath detection；
- Fault detection。

---

## 20. 为什么 Multipath 对 Vector Tracking 仍然危险

Navigation Filter 能很好预测：

$$
Common\ Motion
$$

但 Multipath 是：

$$
Channel\ Specific\ Bias
$$

例如某颗星 Code Delay 被反射路径偏移：

$$
+10\text{ m}
$$

这个不是 Receiver Motion 造成的。

如果 filter 把它错误解释成：

$$
Position\ Change
$$

反而会影响所有 Channel。

所以：

> Vector Tracking 不等于自动消除 Multipath。

它甚至更需要 measurement integrity。

---

## 21. Vector Tracking 常见的 EKF 结构

状态：

$$
\mathbf x_k
$$

Prediction：

$$
\mathbf x_{k|k-1}
=
F\mathbf x_{k-1|k-1}
+
w_k
$$

所有 Tracking measurements：

$$
\mathbf z_k
$$

预测 measurement：

$$
\hat{\mathbf z}_k
=
h(\mathbf x_{k|k-1})
$$

Innovation：

$$
\mathbf y_k
=
\mathbf z_k-\hat{\mathbf z}_k
$$

EKF update：

$$
\mathbf x_{k|k}
=
\mathbf x_{k|k-1}
+
K_k\mathbf y_k
$$

然后更新后的 state 再预测各 Channel NCO。

---

## 22. Measurement Noise 如何体现强星 / 弱星差异

EKF 有 measurement covariance：

$$
R
$$

强星：

$$
C/N_0\uparrow
$$

Tracking jitter 小，

对应：

$$
R_i\downarrow
$$

Navigation Filter 更相信它。

弱星：

$$
C/N_0\downarrow
$$

对应：

$$
R_i\uparrow
$$

Filter 自动降低其权重。

所以理论上：

> Vector Tracking 可以自然进行 cross-channel weighted fusion。

---

## 23. 遮挡以后能撑多久

这取决于 Navigation Prediction Error。

没有 measurement 更新时：

$$
P_{k|k-1}
=
FPF^T+Q
$$

uncertainty 会不断增长。

所以 Vector Tracking 不是：

> 卫星消失后永远知道它在哪里。

而是：

> 在短暂 blockage 内，依靠 motion/clock model 继续预测。

时间越长：

- velocity error 积累；
- oscillator drift 积累；
- phase uncertainty 增加。

最终仍会需要 Reacquisition。

---

## 24. 这时 INS 就非常有用了

如果加入 IMU：

- Accelerometer；
- Gyroscope；

可以高频预测：

- Position；
- Velocity；
- Attitude。

因此短时 GNSS blockage 中：

$$
\hat{\mathbf v}
$$

仍然能保持较好。

于是 Doppler prediction 也会更准。

---

## 25. GNSS + INS 有不同耦合层级

常见可以粗略分成：

### Loose Coupling

GNSS：

$$
PVT
$$

INS：

$$
PVT
$$

最后融合：

```text
GNSS PVT ─┐
          ├→ Navigation Filter
INS PVT ──┘
```

Tracking Loop 本身没有被 INS 直接帮助。

---

### Tight Coupling

GNSS 提供：

- Pseudorange；
- Doppler；

而不是完整 PVT。

```text
GNSS Measurements ─┐
                   ├→ EKF
IMU ────────────────┘
```

在少于 4 星时也可能继续利用 GNSS measurements。

---

## 26. Deep Coupling

进一步：

> 不等 GNSS Tracking 先生成完整 Pseudorange / Doppler measurement。

而是把更底层：

- DLL discriminator；
- PLL/FLL discriminator；
- Correlator I/Q；
- Tracking residual

直接进入联合 filter。

然后 Navigation / INS state：

> 直接反馈到 Carrier / Code NCO。

这就是：

$$
\boxed{
Deep\ Coupling
}
$$

---

## 27. Deep Coupling 架构

配套图：

```text
GNSS_Tracking_第14章_DeepCoupling架构.png
```

核心闭环：

```text
GNSS Correlator
      ↓
Tracking Residual
      ↓
Navigation / INS Filter
      ↑
IMU
      ↓
Predicted Code / Doppler
      ↓
NCO
      ↓
GNSS Correlator
```

相当于：

> Tracking Loop 和 Navigation Filter 真正合成了一套大闭环。

---

## 28. Deep Coupling 为什么对弱信号特别有潜力

传统 PLL 在弱信号下只看到 noisy：

$$
Q/I
$$

Deep Coupling 同时知道：

- IMU 预测的 acceleration；
- Receiver velocity；
- Other satellites；
- Clock model。

所以 Carrier NCO 的主要 motion term 已经预测出来。

PLL residual 只需要估：

$$
small\ error
$$

因此可以：

- narrower loop；
- longer integration；
- lower C/N0 tracking。

---

## 29. 但 Deep Coupling 很难

原因很多：

#### Filter State 很多

可能包含：

- Position；
- Velocity；
- Attitude；
- Clock；
- Accelerometer Bias；
- Gyro Bias；
- Tracking errors。

#### Update Rates 不同

IMU：

$$
100\sim1000\text{ Hz}
$$

GNSS Correlator：

$$
1\text{ kHz or other}
$$

Navigation：

$$
10\sim100\text{ Hz}
$$

必须正确做 multi-rate system design。

#### Error Model 复杂

- IMU Bias；
- oscillator；
- multipath；
- tracking jitter；
- antenna phase；
- clock。

#### Stability 更难 Debug

一个模块错误可能影响整套 loop。

---

## 30. Vector Tracking 和多天线有什么关系

这是非常值得关注的地方。

多天线接收同一颗卫星：

$$
x_m
=
Ae^{j(\phi_{common}+\phi_{array,m})}
$$

其中：

$$
\phi_{common}
$$

主要包含：

- Common Doppler；
- Receiver Clock；
- Satellite Motion。

而：

$$
\phi_{array,m}
$$

主要反映：

- Antenna Position；
- Arrival Direction；
- RF Phase Bias。

所以可以自然分成：

$$
\boxed{
Common\ Tracking\ State
+
Antenna\ Relative\ Phase
}
$$

---

## 31. 这意味着不一定每根天线都需要完整 PLL

一种可能的思想：

```text
              Common Navigation / Doppler
                       ↓
                 Common Prediction
            ┌──────────┼──────────┐
            ↓          ↓          ↓
          Ant1       Ant2       Ant3
            ↓          ↓          ↓
      Relative Phase Residuals
            ↓          ↓          ↓
             Array Processing
```

也就是：

- Common Doppler / Code Rate 共享；
- 每根天线只维护 relative phase / gain state。

相比：

$$
M
$$

根天线复制：

$$
M
$$

套完整 PLL/DLL，

资源可能明显减少。

---

## 32. 多天线 Beamforming 与 Vector Tracking 可以互相帮助

方向一：

```text
Tracking / Navigation
       ↓
Predict satellite LOS
       ↓
Array Steering Vector
       ↓
Beamforming
       ↓
Higher C/N0
       ↓
Better Tracking
```

反过来：

```text
Multi-antenna Relative Phase
       ↓
AoA / Interference Information
       ↓
Better Beamforming / Integrity
       ↓
Better GNSS Measurements
```

所以最终可以形成：

$$
\boxed{
Navigation
\leftrightarrow
Tracking
\leftrightarrow
Array
}
$$

三层互相反馈。

---

## 33. LOS 已知时为什么特别有价值

如果 PVT 已经知道：

- Receiver position；
- Satellite position；
- Device attitude（如果可得）；

那么可以预测 satellite arrival direction：

$$
\mathbf u_i
$$

阵列 steering vector：

$$
\mathbf a(\mathbf u_i)
$$

就有了很好的 prior。

这可以减少 Beamforming 完全 blind estimation 的压力。

---

## 34. 但手机阵列还有 RF Calibration 问题

理论 steering phase：

$$
\phi_{geom,m}
$$

实际测到：

$$
\phi_m
=
\phi_{geom,m}
+
\phi_{RF,m}
+
\phi_{antenna,m}
+
noise
$$

所以多天线 Vector / Beamforming 架构还必须处理：

- RF chain phase bias；
- frequency-dependent phase；
- antenna pattern；
- mutual coupling；
- device orientation；
- calibration drift。

这部分不是 Vector Tracking 自己能自动解决的。

---

## 35. Scalar、Vector、Deep Coupling 可以理解成三层升级

#### Scalar

$$
\boxed{
Channel\ independently\ tracks
}
$$

```text
Tracking → Measurement → PVT
```

#### Vector

$$
\boxed{
All\ GNSS\ channels\ jointly\ track
}
$$

```text
Correlators ↔ Navigation Filter
```

#### Deep Coupling

$$
\boxed{
GNSS\ Tracking + Navigation + INS\ jointly\ close\ the\ loop
}
$$

```text
GNSS Correlator ↔ Navigation/INS Filter ↔ NCO
```

---

## 36. 什么时候没必要上 Vector Tracking

如果场景：

- C/N0 很高；
- Dynamics 不大；
- Satellite visibility 好；
- Scalar Loop 已经非常稳定；
- Power / Hardware resource 很紧；

那么 Vector Tracking 的收益可能不值得其复杂度。

因为 Scalar：

> 简单、稳定、成熟。

工程设计不能为了“高级”而 Vector。

---

## 37. 什么时候 Vector / Deep Coupling 更有价值

典型：

- Urban Canyon；
- Weak GNSS；
- Frequent short blockage；
- High Dynamics；
- Strong IMU available；
- Automotive；
- UAV；
- Robust positioning；
- Anti-jam / degraded GNSS。

如果系统本来就有高质量 IMU / navigation filter：

> Deep Coupling 的边际价值会更高。

---

## 38. 对手机 GNSS 的现实判断

手机里有：

- IMU；
- 多频 GNSS；
- PVT Engine；
- potentially multiple antennas / RF paths。

所以从算法角度：

> Vector / Tight / Deep Coupling 很有吸引力。

但真正落地还要看：

- Power；
- Firmware Complexity；
- State Estimation Cost；
- IMU quality；
- antenna calibration；
- real-time scheduling；
- failure isolation。

因此手机方案通常更合理的路线是：

```text
Scalar Baseline
     ↓
Carrier / Code Aiding
     ↓
Cross-channel Doppler Aiding
     ↓
Vector Tracking
     ↓
GNSS/INS Deep Coupling
```

逐层验证收益，而不是一开始直接上最复杂结构。

---

## 39. 整个 14 章课程最后串起来

现在可以把整个 GNSS Tracking 理解成：

```text
                  RF / DFE Samples
                         ↓
                Carrier / Code NCO
                         ↓
                   Correlators
                         ↓
                    I/Q Accum
                         ↓
            DLL / PLL / FLL Residual
                         ↓
               Tracking Estimator
                         ↓
               Code / Carrier State
                         ↓
          Pseudorange / Doppler / ADR
                         ↓
                      PVT
```

传统 Scalar 到这里结束。

Vector Tracking 再增加：

```text
PVT / Navigation State
          ↓
Predict Doppler / Code Rate
          ↓
Tracking NCO
```

Deep Coupling 再增加：

```text
IMU
 ↓
Navigation / Tracking Joint Filter
 ↓
NCO
```

多天线则进一步加入：

```text
Antenna Relative Phase
          ↓
Array Processing
          ↓
Beamforming / Interference Suppression
```

最终形成：

$$
\boxed{
Signal\ Processing
+
Tracking
+
Navigation
+
Array
}
$$

完整系统。

---

## 40. 全课程核心知识地图

### 第一阶段：Tracking 基础

1. Tracking Module Overall Flow
2. Signal Model & I/Q Demodulation
3. Correlator / ACF / E-P-L
4. DLL / S-Curve / Code Loop

### 第二阶段：Carrier Loop

5. PLL / Costas
6. FLL
7. Loop Filter
8. Carrier NCO / FCW / Phase Accumulator

### 第三阶段：Measurement

9. Carrier Phase / ADR / Integrated Doppler
10. Noise / Jitter / Dynamic Stress
11. Real TRK Channel Data Flow

### 第四阶段：Advanced Tracking

12. Weak Signal / Long Integration
13. BOC / AltBOC / Pilot / Secondary Code
14. Vector Tracking / Deep Coupling

---

## 41. 学完以后应该具备什么能力

如果真正理解这 14 章，应该能够：

- 看懂一个典型 GNSS TRK Block Diagram；
- 理解 E/P/L Correlator 数据；
- 理解 DLL / PLL / FLL discriminator；
- 根据 $C/N_0$ / dynamics 思考 Loop Bandwidth；
- 理解 Carrier FCW / NCO / Continuous Phase；
- 分析 ADR / Integrated Doppler / Carrier Phase；
- 区分 NCO Wrap / Integer Ambiguity / Cycle Slip；
- 理解 1 ms / 20 ms / Measurement Epoch；
- 设计 Weak Tracking / Long Integration；
- 理解 BOC / AltBOC 的 multi-peak problem；
- 评估 E5a/E5b vs Full AltBOC；
- 估算 multi-correlator hardware cost；
- 理解 Scalar / Vector / Deep Coupling；
- 开始设计 Multi-Antenna GNSS Tracking Architecture。

---

## 42. 本章关键结论

1. Scalar Tracking 中每颗卫星独立闭环，然后再进行 PVT。
2. 所有 GNSS Channels 实际共享 Receiver Position / Velocity / Clock 等物理状态。
3. Vector Tracking 利用联合 Navigation Filter 估计这些公共状态。
4. Navigation State 可以预测每颗卫星的 Carrier Doppler 和 Code Rate。
5. Correlator / discriminator residual 用来修正 Navigation State。
6. 弱卫星因此能够被强卫星通过公共 Receiver State 间接帮助。
7. Vector Tracking 的核心是 Cross-channel Aiding。
8. Vector Tracking 并不会自动消除 Multipath / Bias，错误 measurement 反而可能污染其它 channels。
9. 必须配套 Quality Weighting、Innovation Check 与 Outlier Rejection。
10. EKF measurement covariance 可以根据 C/N0 / Tracking Jitter 调节不同 satellite weight。
11. Vector Prediction 可以帮助短时间 signal blockage，但 uncertainty 会随时间增加。
12. Tight Coupling 使用 GNSS raw measurements 与 INS 融合。
13. Deep Coupling 进一步直接使用 discriminator / correlator-level information。
14. Deep Coupling 把 Tracking、Navigation 与 INS 变成一套联合闭环。
15. Deep Coupling 对 Weak Signal / Dynamics 很有潜力，但实现与稳定性明显更复杂。
16. 多天线 GNSS 可以把状态拆成 Common Doppler / Code 与 Antenna Relative Phase。
17. Common Tracking State 可以具有共享潜力，而 Relative Carrier Phase 必须保留给阵列处理。
18. Navigation LOS Prediction 可以帮助 Beamforming，Beamforming 提高 C/N0 后又能帮助 Tracking。
19. Multi-Antenna + Vector Tracking 最终可以形成 Navigation ↔ Tracking ↔ Array 的联合架构。
20. 对实际手机方案，更合理的是从 Scalar Baseline 开始逐步加入 Aiding / Vector / Deep Coupling，而不是直接采用最高复杂度架构。

---

## 43. 课程结束

至此 GNSS Tracking 基础到进阶课程：

$$
\boxed{
14/14
}
$$

已经完整结束。

下一步最适合做的不是继续增加理论章节，而是进入：

## TRK 实战阶段

建议依次做：

1. 选一颗实际卫星，画出真实 `IP/QP/IE/QE/IL/QL`；
2. 从 Prompt 重建 PLL/FLL discriminator；
3. 从 E/L 重建 DLL discriminator；
4. 从 Carrier FCW 重建 Doppler；
5. 从 FCW / phase accumulator 重建 Carrier Phase；
6. 和芯片输出 ADR / Doppler 做一致性验证；
7. 建一个简化的软件 Scalar Tracking Loop；
8. 再尝试 2 天线 common-state + relative-phase tracking；
9. 最后评估 Beamforming / Interference Detection 所需要的 TRK 接口和硬件资源。
