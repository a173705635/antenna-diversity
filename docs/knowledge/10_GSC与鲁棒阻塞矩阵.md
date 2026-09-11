---
tags:
  - 阵列信号处理
  - 波束形成
  - GSC
  - LCMV
  - 自适应滤波
aliases:
  - GSC
  - 广义旁瓣相消器
  - Generalized Sidelobe Canceller
created: 2026-07-19
---

# 第10章：GSC 与鲁棒阻塞矩阵

> GSC（Generalized Sidelobe Canceller）不是一个新的优化目标，而是将 LCMV 的受约束优化问题分解为**固定保护支路 + 自适应干扰相消支路**，更适合 LMS、RLS 等自适应迭代实现。

前置阅读：[[09_LCMV与GNSS多星自由度]]

---

## 1. 为什么要把 LCMV 改写成 GSC

LCMV 的最优权值为：

$$
\mathbf{w}_{\mathrm{LCMV}}
=
\mathbf{R}^{-1}\mathbf{C}
\left(\mathbf{C}^{H}\mathbf{R}^{-1}\mathbf{C}\right)^{-1}\mathbf{f}
$$

直接实现 LCMV 时需要估计协方差矩阵、求解矩阵方程、数据环境变化后重新更新权值。

GSC 的思路是：

> 先通过结构保证约束始终成立，再把剩余问题变成普通的无约束自适应滤波问题。

这样就可以使用 LMS、NLMS、RLS、维纳滤波等自适应算法。

---

## 2. GSC 的基本结构

GSC 由三部分组成：**固定波束形成器、阻塞矩阵、自适应相消器。**

```text
阵列数据 x[k]
      │
      ├──────── 固定波束形成器 w_q
      │                  │
      │                  ▼
      │                 d[k]
      │                  │
      │                  ├──────── 减法器 ───────► y[k]
      │                  │
      ▼                  ▲
   阻塞矩阵 B            │
      │                  │
      ▼                  │
    u[k] ─── 自适应滤波器 g ───► 干扰估计
```

- **固定支路：** 负责保护目标方向
- **阻塞矩阵：** 尽量去除目标信号
- **自适应滤波器：** 从阻塞支路中学习干扰
- 最后从固定支路输出中减去干扰估计

---

## 3. GSC 的总权向量

GSC 将总权向量写成：

$$
\boxed{
\mathbf{w} = \mathbf{w}_{q} - \mathbf{B}\mathbf{g}
}
$$

其中：

- $\mathbf{w}_q$ — 固定权向量
- $\mathbf{B}$ — 阻塞矩阵
- $\mathbf{g}$ — 自适应权向量

减号表示：自适应支路估计出干扰后，从固定支路输出中减掉。

---

## 4. 固定权向量的作用

固定权向量必须满足原来的 LCMV 约束：

$$
\mathbf{C}^{H}\mathbf{w}_{q} = \mathbf{f}
$$

一种常见的固定权值为：

$$
\mathbf{w}_{q}
=
\mathbf{C}
\left(\mathbf{C}^{H}\mathbf{C}\right)^{-1}
\mathbf{f}
$$

它先满足约束，后续自适应过程不会破坏这些约束。

---

## 5. 阻塞矩阵的作用

阻塞矩阵需要满足：

$$
\boxed{\mathbf{C}^{H}\mathbf{B} = \mathbf{0}}
$$

这表示阻塞矩阵的列位于约束矩阵的**零空间**中。

阻塞支路输出为：

$$
\mathbf{u}[k] = \mathbf{B}^{H}\mathbf{x}[k]
$$

理想情况下，阻塞支路主要包含干扰和噪声，目标信号被消除。

---

### 5.1 $v_B[k]$ 表示什么——阻塞支路中的信号成分分析

为了更清楚地说明阻塞支路里到底有什么，有时把阻塞支路中的干扰和噪声成分记为 $\mathbf{v}_B[k]$。它不是新的算法变量，只是一个便于讨论的简写。

假设阵列接收信号可以分解为目标信号部分和干扰噪声部分：

$$
\mathbf{x}[k] = \mathbf{x}_s[k] + \mathbf{x}_v[k]
$$

其中 $\mathbf{x}_s[k]$ 是目标信号部分，$\mathbf{x}_v[k]$ 是干扰和噪声部分。

经过阻塞矩阵后：

$$
\mathbf{u}[k] = \mathbf{B}^{H}\mathbf{x}_s[k] + \mathbf{B}^{H}\mathbf{x}_v[k]
$$

理想情况下阻塞矩阵完美阻塞目标：$\mathbf{B}^{H}\mathbf{x}_s[k] = \mathbf{0}$。因此：

$$
\mathbf{u}[k] = \mathbf{B}^{H}\mathbf{x}_v[k]
$$

把右侧简写为：

$$
\mathbf{v}_B[k] = \mathbf{B}^{H}\mathbf{x}_v[k]
$$

所以：

$$
\mathbf{u}[k] \approx \mathbf{v}_B[k]
$$

> $\mathbf{v}_B[k]$ 的含义就是：阻塞矩阵输出中的干扰和噪声成分。

这个简写有助于后续理解——自适应滤波器正是利用 $\mathbf{v}_B[k]$（阻塞支路中的干扰）去估计并抵消固定支路中的干扰分量 $v_0[k]$。

---

## 6. 为什么总权值仍然满足约束

GSC 总权向量为 $\mathbf{w} = \mathbf{w}_{q} - \mathbf{B}\mathbf{g}$。

左乘约束矩阵的共轭转置：

$$
\mathbf{C}^{H}\mathbf{w}
=
\mathbf{C}^{H}\mathbf{w}_{q} - \mathbf{C}^{H}\mathbf{B}\mathbf{g}
=
\mathbf{f} - \mathbf{0} \cdot \mathbf{g}
=
\mathbf{f}
$$

> 无论自适应权向量 $\mathbf{g}$ 如何变化，总权向量**始终满足**原来的 LCMV 约束。这是 GSC 最关键的结构性质。

---

## 7. 两条支路的输出

固定支路输出：$d[k] = \mathbf{w}_{q}^{H}\mathbf{x}[k]$

阻塞支路输出：$\mathbf{u}[k] = \mathbf{B}^{H}\mathbf{x}[k]$

自适应滤波器输出：$\hat{v}[k] = \mathbf{g}^{H}\mathbf{u}[k]$

最终输出：

$$
y[k] = d[k] - \hat{v}[k]
$$

代入各部分，可得总权向量 $\mathbf{w} = \mathbf{w}_{q} - \mathbf{B}\mathbf{g}$。

---

## 8. 自适应支路真正做什么

固定支路输出可以写成：

$$
d[k]
=
s[k]
+
v_{0}[k]
$$

其中：

- `s[k]` 是目标信号；
- `v_0[k]` 是固定支路中的干扰和噪声。

阻塞支路理想情况下不包含目标信号：

$$
\mathbf{u}[k]
\approx
\mathbf{v}_{B}[k]
$$

自适应滤波器利用阻塞支路估计固定支路中的干扰：

$$
\hat{v}[k]
=
\mathbf{g}^{H}
\mathbf{u}[k]
$$

如果估计准确：

$$
\hat{v}[k]
\approx
v_{0}[k]
$$

那么最终输出为：

$$
y[k]
=
s[k]
+
v_{0}[k]
-
\hat{v}[k]
$$

于是：

$$
y[k]
\approx
s[k]
$$

可以把它理解为：

> 用辅助支路观察到的干扰，去抵消主支路中的干扰。

---

## 9. GSC 的优化目标

GSC 不再直接优化带约束的总权向量，而是优化自适应权向量 $\mathbf{g}$：

$$
\min_{\mathbf{g}}
E\left[\left|d[k] - \mathbf{g}^{H}\mathbf{u}[k]\right|^{2}\right]
$$

这是**标准的最小均方误差问题**。约束已由固定权值和阻塞矩阵自动保证。

---

## 10. 最优自适应权值

定义：

$$
\mathbf{R}_{uu} = E\left[\mathbf{u}[k]\mathbf{u}^{H}[k]\right],\quad
\mathbf{r}_{ud} = E\left[\mathbf{u}[k]d^{*}[k]\right]
$$

最优自适应权值（维纳滤波器解）：

$$
\boxed{\mathbf{g}_{\mathrm{opt}} = \mathbf{R}_{uu}^{-1}\mathbf{r}_{ud}}
$$

最终 GSC 权值：

$$
\mathbf{w}_{\mathrm{GSC}} = \mathbf{w}_{q} - \mathbf{B}\mathbf{g}_{\mathrm{opt}}
$$

理想条件下 $\mathbf{w}_{\mathrm{GSC}} = \mathbf{w}_{\mathrm{LCMV}}$。

> GSC 和 LCMV 在**数学上等价**，只是实现结构不同。

---

### 10.1 GSC 与 LCMV 的等价关系——为什么数学上一致

LCMV 直接在整个 $N$ 维权空间中寻找最优权值，但要求满足 $\mathbf{C}^{H}\mathbf{w} = \mathbf{f}$。

GSC 把所有满足约束的权向量写成一种特殊形式：

$$
\mathbf{w} = \mathbf{w}_{q} - \mathbf{B}\mathbf{g}
$$

其中 $\mathbf{w}_q$ 是满足约束的一个**特解**，而 $\mathbf{B}\mathbf{g}$ 是约束零空间内的**调整量**。

因为 $\mathbf{C}^{H}\mathbf{B} = \mathbf{0}$，所以 $\mathbf{B}\mathbf{g}$ 这个调整量无论怎么变化都不会改变约束响应：

$$
\mathbf{C}^{H}(\mathbf{B}\mathbf{g}) = (\mathbf{C}^{H}\mathbf{B})\mathbf{g} = \mathbf{0} \cdot \mathbf{g} = \mathbf{0}
$$

令 $r=\operatorname{rank}(\mathbf C)$。GSC 实际上是在 $N-r$ 维零空间中寻找最优调整量。当自适应权值 $\mathbf{g}$ 取最优维纳解时，得到的 $\mathbf{w}_{\mathrm{GSC}}$ 就等于 $\mathbf{w}_{\mathrm{LCMV}}$。

> 从数学角度看，LCMV 直接求解带约束的优化问题；GSC 把问题分解成"先满足约束（特解）→ 再在零空间中优化（调整量）"。两种方式的解空间完全相同，因此最终结果一致。

---

### 10.2 为什么叫"旁瓣相消器"

固定波束形成器通常把主瓣对准目标方向，目标信号主要从主瓣进入。干扰通常从旁瓣、后瓣或其他非目标方向进入。

阻塞支路去掉目标后，主要保留这些非目标方向信号。自适应支路再将其从主支路输出中减去——相当于把从旁瓣进来的干扰"消掉"。

因此最初称为 **Sidelobe Canceller（旁瓣相消器）**。

后来推广到多约束、一般阵列、一般阻塞矩阵和更复杂的自适应结构，所以称为 **Generalized Sidelobe Canceller（广义旁瓣相消器）**。

---

## 11. 阻塞矩阵的维度

设阵元数为 $N$，约束矩阵共有 $Q$ 列，有效约束秩为 $r=\operatorname{rank}(\mathbf C)$。

约束矩阵 $\mathbf{C} \in \mathbb{C}^{N \times Q}$，阻塞矩阵 $\mathbf{B} \in \mathbb{C}^{N \times (N-r)}$。

阻塞支路输出维度为 $\mathbf{u}[k] \in \mathbb{C}^{N-r}$，这与剩余自由度 $D_{\mathrm{remain}} = N-r$ 一致。只有当所有 $Q$ 列约束线性独立时，才有 $r=Q$。

> 阻塞矩阵相当于把原始阵列数据投影到**不影响约束的剩余自由度空间**。

---

## 12. 单约束情况下的 GSC

单目标方向时，$\mathbf{C} = \mathbf{a}_0$，约束为 $\mathbf{a}_0^{H}\mathbf{w} = 1$。

固定权值可以选择：

$$
\mathbf{w}_{q} = \frac{\mathbf{a}_0}{\mathbf{a}_0^{H}\mathbf{a}_0}
$$

阻塞矩阵需满足 $\mathbf{B}^{H}\mathbf{a}_0 = \mathbf{0}$，尺寸为 $\mathbf{B} \in \mathbb{C}^{N \times (N-1)}$，提供 $N-1$ 个自适应通道。

---

## 13. 宽边方向示例

宽边方向（Broadside）是垂直于阵列排列方向的方向。对于均匀线阵，宽边方向 $\theta = 0^\circ$ 的导向矢量为：

$$
\mathbf{a}_0 = \begin{bmatrix} 1 \\ 1 \\ 1 \\ 1 \end{bmatrix}
$$

一种简单的阻塞矩阵由相邻阵元差分构成：

$$
\mathbf{B}^{H}
=
\begin{bmatrix}
1 & -1 & 0 & 0 \\
0 & 1 & -1 & 0 \\
0 & 0 & 1 & -1
\end{bmatrix}
$$

检查：$\mathbf{B}^{H}\mathbf{a}_0 = \mathbf{0}$。相同相位、相同幅度的宽边目标信号会被相邻差分消除。

来自其他方向的干扰在相邻阵元之间通常存在相位差，不会被完全抵消，因此阻塞支路保留干扰信息。

---

## 14. 非宽边目标怎么办

如果目标不在宽边方向，不能直接使用普通相邻差分。可采用：

1. 先对目标方向进行相位补偿，使目标在各阵元上重新同相，再做差分
2. 使用 QR 分解、SVD 或零空间计算，直接构造满足 $\mathbf{B}^{H}\mathbf{a}_0 = \mathbf{0}$ 的阻塞矩阵

---

## 15. 使用 SVD 构造阻塞矩阵

对于一般约束矩阵 $\mathbf{C}$，需要找到 $\mathbf{C}^{H}$ 的零空间。阻塞矩阵的列就是该零空间的一组正交基。

```python
def blocking_matrix(C, tolerance=1e-10):
    """构造满足 C^H B = 0 的阻塞矩阵。"""
    _, singular_values, Vh = np.linalg.svd(
        C.conj().T, full_matrices=True
    )

    if singular_values.size == 0:
        rank = 0
    else:
        threshold = tolerance * singular_values[0]
        rank = np.sum(singular_values > threshold)

    V = Vh.conj().T
    B = V[:, rank:]
    return B
```

应检查：

```python
blocking_residual = C.conj().T @ B
orthogonal_residual = B.conj().T @ B - np.eye(B.shape[1])
```

两者都应接近零矩阵。SVD 给出的正交阻塞基通常比任意零空间基有更好的数值性质。

### 15.1 批量数据下的 GSC 实现

设阵列快拍矩阵 $\mathbf X\in\mathbb C^{N\times K}$：

```python
# 固定支路与阻塞支路
d = w_q.conj().T @ X
U_block = B.conj().T @ X

# 辅助协方差和互相关
R_uu = U_block @ U_block.conj().T / K
r_ud = U_block @ d.conj().T / K

# 正则化维纳解
loading = 1e-3 * np.trace(R_uu).real / R_uu.shape[0]
R_uu_loaded = R_uu + loading * np.eye(R_uu.shape[0])
g = np.linalg.solve(R_uu_loaded, r_ud)

# 输出和等效总权值
y = d - g.conj().T @ U_block
w_gsc = w_q - B @ g
```

至少验证：

```python
assert np.allclose(C.conj().T @ w_gsc, f)
```

在相同协方差、约束和数值秩设置下，批量 GSC 总权值应与加载 LCMV 解一致到数值误差范围。

---

## 16. GSC 最大的问题：目标泄漏

GSC 正常工作的关键是假设：

$$
\mathbf{B}^{H}\mathbf{a}_{\mathrm{true}} = \mathbf{0}
$$

但实际中使用的导向矢量是 $\mathbf{a}_{\mathrm{model}}$（模型值），而真实导向矢量是 $\mathbf{a}_{\mathrm{true}}$。如果两者不完全一致：

$$
\mathbf{B}^{H}\mathbf{a}_{\mathrm{true}} \neq \mathbf{0}
$$

目标信号泄漏到阻塞支路。

对于多目标保护，设计子空间为 $\operatorname{col}(\mathbf C)$，真实目标子空间为 $\operatorname{col}(\mathbf C_{\mathrm{true}})$。定义泄漏矩阵：

$$
\boxed{
\mathbf L
=
\mathbf B^H\mathbf C_{\mathrm{true}}}
$$

可用：

$$
\epsilon_{\mathrm{leak}}
=
\|\mathbf L\|_F
$$

衡量整个保护子空间向辅助支路泄漏的程度。阻塞矩阵阻塞的是**设计模型中的子空间**，不会自动识别真实目标。

**导致目标泄漏的原因：**

阵元位置误差、通道相位误差、通道增益误差、阵元互耦、设备姿态误差、星历方向误差、天线方向图差异、多径、机体遮挡、宽带频率失配、极化失配。

---

### 16.1 目标泄漏为什么会导致自消除

假设阻塞支路中包含少量目标信号：

$$
\mathbf{u}[k] = \mathbf{u}_s[k] + \mathbf{u}_v[k]
$$

其中 $\mathbf{u}_s[k]$ 是泄漏进来的目标信号成分，$\mathbf{u}_v[k]$ 是干扰和噪声。

自适应滤波器的目标是最小化输出功率 $E[|y[k]|^2]$。只要泄漏的目标信号 $\mathbf{u}_s[k]$ 与固定支路中的目标信号 $s[k]$（包含在 $d[k]$ 里）**相关**，自适应滤波器就会学习这个关系——它能"看到"阻塞支路里有一份目标信号的副本。

于是自适应支路输出 $\hat{v}[k] = \mathbf{g}^{H}\mathbf{u}[k]$ 中也会包含目标信号成分的估计。

当这个估计从固定支路输出中减去时：

$$
y[k] = d[k] - \hat{v}[k]
$$

目标信号就被部分抵消了。最终可能导致：

- 目标信号衰减
- 卫星信号自消除
- 输出载噪比下降
- 跟踪环失锁

> 本质原因是：GSC 依赖阻塞矩阵的准确性，而实际系统中导向矢量误差不可避免。一旦阻塞矩阵不能完美去除目标，自适应滤波器就会把泄漏的目标当作"干扰"来处理。

---

---

## 17. 如何提高 GSC 稳健性

| 方法 | 说明 |
|------|------|
| **对角加载** | $\mathbf{R}_{uu,\mathrm{DL}} = \mathbf{R}_{uu} + \delta\mathbf{I}$，降低权值过度放大 |
| **多点约束** | 在目标方向附近设置多个保护点，扩大保护区域 |
| **导数约束** | 限制目标方向附近的响应变化率 |
| **鲁棒阻塞矩阵** | 根据阵列误差模型构造更加稳健的阻塞矩阵 |
| **权值范数约束** | 限制 $\|\mathbf{g}\|^2$，避免自适应权值过大 |
| **后相关处理** | 对每颗卫星解扩后再进行 GSC |

### 17.1 多点保护和导数保护为什么有效

小角度误差下：

$$
\mathbf a(\theta_0+\Delta\theta)
\approx
\mathbf a(\theta_0)
+\Delta\theta
\frac{\partial\mathbf a}{\partial\theta}
\bigg|_{\theta_0}
$$

若阻塞矩阵同时满足：

$$
\mathbf B^H\mathbf a(\theta_0)=0
$$

$$
\mathbf B^H
\frac{\partial\mathbf a}{\partial\theta}
\bigg|_{\theta_0}=0
$$

则一阶角度误差引起的泄漏会被抑制。多点保护用离散角度采样扩展保护子空间，导数保护则直接约束局部变化率；两者都会增加保护秩、减少辅助自由度。

### 17.2 辅助支路正则化

$$
\boxed{
\mathbf g_\delta
=(\mathbf R_u+\delta\mathbf I)^{-1}\mathbf p_{ud}}
$$

它限制辅助权值范数，降低病态协方差和目标泄漏的放大。在线实现对应泄漏 LMS：

$$
\mathbf g[k+1]
=(1-\mu\gamma)\mathbf g[k]
+\mu\mathbf u[k]y^*[k]
$$

---

## 18. 使用 LMS 更新自适应权值

GSC 可以逐 Snapshot 更新自适应权值，不需要一次性计算维纳解。

阻塞支路输出为 $\mathbf{u}[k] = \mathbf{B}^{H}\mathbf{x}[k]$，GSC 输出为 $y[k] = d[k] - \mathbf{g}^{H}[k]\mathbf{u}[k]$。

一种 LMS 更新形式为：

$$
\mathbf{g}[k+1] = \mathbf{g}[k] + \mu\,\mathbf{u}[k]\,y^{*}[k]
$$

其中 $\mu$ 是步长。步长太小则收敛慢，步长太大则可能发散。

> 不同误差定义和符号约定可能导致更新式中的正负号不同，只要与输出定义保持一致即可。

---

### 18.1 LMS 的直观含义

LMS 更新式可以从"当前做得怎么样"来理解。

如果当前输出误差较大，说明 $\mathbf{g}^{H}[k]\mathbf{u}[k]$ 还没有很好地估计固定支路中的干扰。LMS 根据两个量来调整 $\mathbf{g}$：

- **$\mathbf{u}[k]$（阻塞支路数据）：** 告诉我们干扰当前在哪些空间方向上
- **$y^{*}[k]$（输出误差的共轭）：** 告诉我们当前估计偏差有多大、朝哪个方向偏

更新式 $\mu\,\mathbf{u}[k]\,y^{*}[k]$ 的含义是：

> 沿着干扰当前的方向 $\mathbf{u}[k]$，按照误差 $y^{*}[k]$ 给出的幅度和方向，微调自适应权值 $\mathbf{g}$。

随着迭代进行，$\mathbf{g}^{H}[k]\mathbf{u}[k]$ 会逐渐逼近固定支路中的干扰分量 $v_0[k]$，输出 $y[k]$ 中的干扰成分越来越小，最终输出逼近纯净的目标信号。

### 18.2 时变约束和阻塞基变化

卫星方向和平台姿态变化会使：

$$
\mathbf C=\mathbf C[k],
\qquad
\mathbf B=\mathbf B[k]
$$

阻塞基不唯一。若 $\mathbf T$ 可逆，$\mathbf B'=\mathbf B\mathbf T$ 仍张成同一零空间；但辅助坐标会改变。因而直接保留旧的 $\mathbf g[k]$，可能使总权值突然跳变。

实际系统需要在阻塞基更新时选择：

- 将旧总权值投影到新的约束零空间；
- 把旧辅助权值映射到新阻塞基；
- 平滑更新 $\mathbf C$、$\mathbf w_q$ 和 $\mathbf B$；
- 在变化过大时冻结或重置辅助权值。

---

---

## 19. GSC 在 GNSS 中的定位

### 前相关 GSC
所有卫星共用同一套输出。如果只把某一颗卫星作为目标约束，其他卫星可能进入阻塞支路被抵消。可能需要多卫星约束、天空区域约束或功率反演结构。

### 后相关 GSC
每颗卫星单独建立 GSC：
- 固定支路保护该卫星
- 阻塞矩阵阻止该卫星进入辅助支路
- 自适应支路抵消相关输出中的干扰和多径

这种方式更适合多卫星、多方向场景。

---

## 20. GSC、MVDR 和 LCMV 的区别

| 方法 | 约束数量 | 主要实现形式 | 适合自适应迭代 |
|------|:-------:|------------|:------------:|
| MVDR | 一个 | 协方差矩阵求解 | 可以 |
| LCMV | 多个 | 受约束矩阵求解 | 可以 |
| GSC | 一个或多个 | 固定支路 + 阻塞支路 | **非常适合** |

> 理想条件下，单约束 GSC ≈ MVDR，多约束 GSC ≈ LCMV。

---

---

### 20.1 GSC 与前面算法的关系

```text
阵列接收数据
    │
    ▼
协方差矩阵
    │
    ├── MUSIC
    │     利用噪声子空间估计信号方向
    │
    ├── ESPRIT
    │     利用信号子空间旋转关系估计信号方向
    │
    ├── 空间平滑
    │     处理相干信号导致的秩亏
    │
    ├── MVDR
    │     单约束最小方差波束形成
    │
    ├── LCMV
    │     多约束最小方差波束形成
    │
    └── GSC
          将 LCMV 分解为
          固定保护支路
          加自适应相消支路
```

GSC 是 LCMV 的等价实现结构。它不改变优化目标，而是改变了实现方式——把抽象的矩阵约束优化，拆成了物理意义更直观、更适合自适应迭代的两个支路。

---

## 21. 核心公式汇总

GSC 总权向量：

$$
\mathbf{w} = \mathbf{w}_{q} - \mathbf{B}\mathbf{g}
$$

固定权值约束：$\mathbf{C}^{H}\mathbf{w}_{q} = \mathbf{f}$

阻塞矩阵条件：$\mathbf{C}^{H}\mathbf{B} = \mathbf{0}$

最终输出：

$$
y[k] = d[k] - \mathbf{g}^{H}\mathbf{u}[k]
$$

最优自适应权值（维纳解）：

$$
\mathbf{g}_{\mathrm{opt}} = \mathbf{R}_{uu}^{-1}\mathbf{r}_{ud}
$$

---

## 22. 最核心的理解

```
LCMV
    ↓
分解约束空间和自由空间
    ↓
固定波束形成器满足约束
    ↓
阻塞矩阵保护目标
    ↓
自适应滤波器学习干扰
    ↓
干扰相消
```

**LCMV 问：** 怎样找到一组满足约束并使输出功率最小的总权值？

**GSC 答：** 分解为四步——固定权值保护目标 → 阻塞矩阵去掉目标 → 自适应滤波器学习干扰 → 从固定支路减去干扰估计。

---

## 下一步建议

完整仿真流程：

- 四阵元或八阵元 ULA
- 一个目标信号 + 两个强干扰
- 构造固定权值
- 使用 SVD 生成阻塞矩阵
- 计算维纳解
- 比较 LCMV 与 GSC 的最终权值和波束图
- 人为加入导向矢量误差，观察目标泄漏和自消除

下一步学习 [[11_自适应滤波与功率反演]]，系统理解 LMS、NLMS、RLS，以及目标方向未知时的功率反演结构。仿真代码参见 [[15_仿真代码与数学推导]]。
