# 第二十六章：GSC 广义旁瓣相消器

GSC 的英文全称为：

**Generalized Sidelobe Canceller**

中文通常称为：

**广义旁瓣相消器**

GSC 不是一个全新的优化目标。

它本质上是把 LCMV 的受约束优化问题，重新分解成两个更容易理解和实现的部分：

```text
固定保护支路
    +
自适应干扰相消支路
```

GSC 的核心思想是：

> 先用固定支路保护目标信号，再让自适应支路只负责学习和抵消干扰。

---

## 1. 为什么要把 LCMV 改写成 GSC

LCMV 的目标函数为：

$$
\min_{\mathbf{w}}
\mathbf{w}^{H}
\mathbf{R}
\mathbf{w}
$$

约束条件为：

$$
\mathbf{C}^{H}
\mathbf{w}
=
\mathbf{f}
$$

LCMV 的最优权值为：

$$
\mathbf{w}_{\mathrm{LCMV}}
=
\mathbf{R}^{-1}
\mathbf{C}
\left(
\mathbf{C}^{H}
\mathbf{R}^{-1}
\mathbf{C}
\right)^{-1}
\mathbf{f}
$$

直接实现 LCMV 时，通常需要：

- 估计协方差矩阵；
- 求解矩阵方程；
- 数据环境变化后重新更新权值；
- 同时处理约束和自适应优化。

GSC 的思路是：

> 先通过结构保证约束始终成立，再把剩余问题变成普通的无约束自适应滤波问题。

这样就可以使用：

- LMS；
- NLMS；
- RLS；
- 维纳滤波；
- 其他自适应滤波算法。

---

## 2. GSC 的基本结构

GSC 通常由三部分组成：

1. 固定波束形成器；
2. 阻塞矩阵；
3. 自适应相消器。

整体结构如下：

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

其中：

- 固定支路负责保护目标方向；
- 阻塞矩阵尽量去除目标信号；
- 自适应滤波器从阻塞支路中学习干扰；
- 最后从固定支路输出中减去干扰估计。

---

## 3. GSC 的总权向量

GSC 将总权向量写成：

$$
\mathbf{w}
=
\mathbf{w}_{q}
-
\mathbf{B}
\mathbf{g}
$$

其中：

- `w_q` 是固定权向量；
- `B` 是阻塞矩阵；
- `g` 是自适应权向量。

这里的减号表示：

> 自适应支路估计出干扰后，从固定支路输出中减掉。

---

## 4. 固定权向量的作用

固定权向量必须满足原来的 LCMV 约束：

$$
\mathbf{C}^{H}
\mathbf{w}_{q}
=
\mathbf{f}
$$

因此，固定支路负责保证：

- 目标方向得到指定响应；
- 受保护方向不会被固定支路压制；
- 后续自适应过程不会破坏这些约束。

一种常见的固定权值为：

$$
\mathbf{w}_{q}
=
\mathbf{C}
\left(
\mathbf{C}^{H}
\mathbf{C}
\right)^{-1}
\mathbf{f}
$$

它不一定是最终最优权值，但它可以先满足：

$$
\mathbf{C}^{H}
\mathbf{w}_{q}
=
\mathbf{f}
$$

---

## 5. 阻塞矩阵的作用

阻塞矩阵需要满足：

$$
\mathbf{C}^{H}
\mathbf{B}
=
\mathbf{0}
$$

等价地：

$$
\mathbf{B}^{H}
\mathbf{C}
=
\mathbf{0}
$$

这表示阻塞矩阵的列位于约束矩阵的零空间中。

如果约束矩阵包含目标导向矢量，那么经过阻塞矩阵后，目标信号理想情况下会被消除。

阻塞支路输出为：

$$
\mathbf{u}[k]
=
\mathbf{B}^{H}
\mathbf{x}[k]
$$

理想情况下，阻塞支路主要包含：

- 干扰；
- 噪声；
- 非受保护方向信号。

---

## 6. `v_B[k]` 表示什么

有时为了说明方便，会把阻塞支路中的干扰和噪声记为：

$$
\mathbf{v}_{B}[k]
$$

它不是新的算法变量，只是一个简写。

假设阵列接收信号可以分成：

$$
\mathbf{x}[k]
=
\mathbf{x}_{s}[k]
+
\mathbf{x}_{v}[k]
$$

其中：

- `x_s[k]` 是目标信号部分；
- `x_v[k]` 是干扰和噪声部分。

经过阻塞矩阵后：

$$
\mathbf{u}[k]
=
\mathbf{B}^{H}
\mathbf{x}_{s}[k]
+
\mathbf{B}^{H}
\mathbf{x}_{v}[k]
$$

理想情况下：

$$
\mathbf{B}^{H}
\mathbf{x}_{s}[k]
=
\mathbf{0}
$$

因此：

$$
\mathbf{u}[k]
=
\mathbf{B}^{H}
\mathbf{x}_{v}[k]
$$

把右侧简写为：

$$
\mathbf{v}_{B}[k]
=
\mathbf{B}^{H}
\mathbf{x}_{v}[k]
$$

所以：

$$
\mathbf{u}[k]
\approx
\mathbf{v}_{B}[k]
$$

它的含义就是：

> 阻塞矩阵输出中的干扰和噪声。

---

## 7. 为什么总权值仍然满足约束

GSC 总权向量为：

$$
\mathbf{w}
=
\mathbf{w}_{q}
-
\mathbf{B}
\mathbf{g}
$$

左乘约束矩阵的共轭转置：

$$
\mathbf{C}^{H}
\mathbf{w}
=
\mathbf{C}^{H}
\mathbf{w}_{q}
-
\mathbf{C}^{H}
\mathbf{B}
\mathbf{g}
$$

因为：

$$
\mathbf{C}^{H}
\mathbf{w}_{q}
=
\mathbf{f}
$$

并且：

$$
\mathbf{C}^{H}
\mathbf{B}
=
\mathbf{0}
$$

所以：

$$
\mathbf{C}^{H}
\mathbf{w}
=
\mathbf{f}
$$

这说明：

> 无论自适应权向量 `g` 如何变化，总权向量始终满足原来的 LCMV 约束。

这是 GSC 最关键的结构性质。

---

## 8. 两条支路的输出

固定支路输出为：

$$
d[k]
=
\mathbf{w}_{q}^{H}
\mathbf{x}[k]
$$

阻塞支路输出为：

$$
\mathbf{u}[k]
=
\mathbf{B}^{H}
\mathbf{x}[k]
$$

自适应滤波器输出为：

$$
\hat{v}[k]
=
\mathbf{g}^{H}
\mathbf{u}[k]
$$

最终输出为：

$$
y[k]
=
d[k]
-
\hat{v}[k]
$$

代入各部分：

$$
y[k]
=
\mathbf{w}_{q}^{H}
\mathbf{x}[k]
-
\mathbf{g}^{H}
\mathbf{B}^{H}
\mathbf{x}[k]
$$

整理后：

$$
y[k]
=
\left(
\mathbf{w}_{q}
-
\mathbf{B}
\mathbf{g}
\right)^{H}
\mathbf{x}[k]
$$

因此总权向量为：

$$
\mathbf{w}
=
\mathbf{w}_{q}
-
\mathbf{B}
\mathbf{g}
$$

---

## 9. 自适应支路真正做什么

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

## 10. GSC 的优化目标

GSC 不再直接优化带约束的总权向量，而是优化自适应权向量 `g`。

目标函数为：

$$
\min_{\mathbf{g}}
E
\left[
\left|
d[k]
-
\mathbf{g}^{H}
\mathbf{u}[k]
\right|^{2}
\right]
$$

这是标准的最小均方误差问题。

约束已经由固定权值和阻塞矩阵自动保证，所以目标函数中不再显式出现约束。

---

## 11. 最优自适应权值

定义阻塞支路协方差矩阵：

$$
\mathbf{R}_{uu}
=
E
\left[
\mathbf{u}[k]
\mathbf{u}^{H}[k]
\right]
$$

定义阻塞支路与固定支路输出之间的互相关向量：

$$
\mathbf{r}_{ud}
=
E
\left[
\mathbf{u}[k]
d^{*}[k]
\right]
$$

最优自适应权值为：

$$
\mathbf{g}_{\mathrm{opt}}
=
\mathbf{R}_{uu}^{-1}
\mathbf{r}_{ud}
$$

这就是维纳滤波器解。

最终 GSC 权值为：

$$
\mathbf{w}_{\mathrm{GSC}}
=
\mathbf{w}_{q}
-
\mathbf{B}
\mathbf{g}_{\mathrm{opt}}
$$

理想条件下：

$$
\mathbf{w}_{\mathrm{GSC}}
=
\mathbf{w}_{\mathrm{LCMV}}
$$

因此：

> GSC 和 LCMV 在数学上等价，只是实现结构不同。

---

## 12. 阻塞矩阵的维度

假设：

- 阵元数为 `N`；
- 独立约束数量为 `Q`。

约束矩阵尺寸为：

$$
\mathbf{C}
\in
\mathbb{C}^{N \times Q}
$$

阻塞矩阵尺寸通常为：

$$
\mathbf{B}
\in
\mathbb{C}^{N \times (N-Q)}
$$

阻塞支路输出维度为：

$$
\mathbf{u}[k]
\in
\mathbb{C}^{N-Q}
$$

这与前面讲的剩余自由度一致：

$$
D_{\mathrm{remain}}
=
N-Q
$$

阻塞矩阵相当于把原始阵列数据投影到不影响约束的剩余自由度空间。

---

## 13. 单约束情况下的 GSC

假设只有一个目标方向。

约束矩阵退化为：

$$
\mathbf{C}
=
\mathbf{a}_{0}
$$

约束条件为：

$$
\mathbf{a}_{0}^{H}
\mathbf{w}
=
1
$$

固定权值可以选择：

$$
\mathbf{w}_{q}
=
\frac{
\mathbf{a}_{0}
}{
\mathbf{a}_{0}^{H}
\mathbf{a}_{0}
}
$$

阻塞矩阵需要满足：

$$
\mathbf{B}^{H}
\mathbf{a}_{0}
=
\mathbf{0}
$$

如果阵元数为 `N`，则：

$$
\mathbf{B}
\in
\mathbb{C}^{N \times (N-1)}
$$

阻塞支路提供 `N-1` 个自适应通道。

---

## 14. 宽边方向是什么

宽边方向的英文是：

**Broadside**

对于均匀线阵，宽边方向就是：

> 垂直于阵列排列方向的方向。

例如阵元沿水平方向排列：

```text
●——●——●——●
```

信号从阵列正前方垂直入射：

```text
        ↓ 信号

●——●——●——●
```

这个方向就是宽边方向。

如果把宽边方向定义为 `0°`，那么：

$$
\theta
=
0^\circ
$$

此时信号到达所有阵元的传播距离相同，因此各阵元接收到的相位相同。

导向矢量为：

$$
\mathbf{a}(0^\circ)
=
\begin{bmatrix}
1 \\
1 \\
\vdots \\
1
\end{bmatrix}
$$

与宽边方向相对的是端射方向。

端射方向的英文是：

**Endfire**

端射方向平行于阵列排列方向：

```text
信号 →  ●——●——●——●
```

通常记为：

$$
\theta
=
\pm 90^\circ
$$

简单记忆：

```text
宽边：垂直阵列

端射：平行阵列
```

---

## 15. 四阵元宽边入射例子

假设四阵元 ULA，目标来自宽边方向。

目标导向矢量为：

$$
\mathbf{a}_{0}
=
\begin{bmatrix}
1 \\
1 \\
1 \\
1
\end{bmatrix}
$$

固定权值可以选择：

$$
\mathbf{w}_{q}
=
\frac{1}{4}
\begin{bmatrix}
1 \\
1 \\
1 \\
1
\end{bmatrix}
$$

一种简单的阻塞矩阵可以由相邻阵元差分构成。

阻塞矩阵的共轭转置为：

$$
\mathbf{B}^{H}
=
\begin{bmatrix}
1 & -1 & 0 & 0 \\
0 & 1 & -1 & 0 \\
0 & 0 & 1 & -1
\end{bmatrix}
$$

检查目标是否被阻塞：

$$
\mathbf{B}^{H}
\mathbf{a}_{0}
=
\begin{bmatrix}
1-1 \\
1-1 \\
1-1
\end{bmatrix}
$$

因此：

$$
\mathbf{B}^{H}
\mathbf{a}_{0}
=
\mathbf{0}
$$

这说明相同相位、相同幅度的宽边目标信号会被相邻差分消除。

---

## 16. 为什么相邻差分能去除宽边目标

宽边目标在所有阵元上的信号近似相同：

$$
x_{1,s}[k]
=
x_{2,s}[k]
=
x_{3,s}[k]
=
x_{4,s}[k]
$$

第一路阻塞输出为：

$$
u_{1}[k]
=
x_{1}[k]
-
x_{2}[k]
$$

对于目标信号部分：

$$
u_{1,s}[k]
=
x_{1,s}[k]
-
x_{2,s}[k]
$$

所以：

$$
u_{1,s}[k]
=
0
$$

同理：

$$
u_{2,s}[k]
=
0
$$

$$
u_{3,s}[k]
=
0
$$

但来自其他方向的干扰，在相邻阵元之间通常存在相位差，所以不会被完全抵消。

因此，阻塞支路会保留干扰信息。

---

## 17. 非宽边目标怎么办

如果目标不在宽边方向，各阵元上的目标相位不同。

例如：

$$
\mathbf{a}_{0}
=
\begin{bmatrix}
1 \\
e^{-j\mu_{0}} \\
e^{-j2\mu_{0}} \\
e^{-j3\mu_{0}}
\end{bmatrix}
$$

这时不能直接使用普通相邻差分。

可以采用两种方法：

1. 先对目标方向进行相位补偿，使目标在各阵元上重新同相，再做差分；
2. 使用 QR 分解、SVD 或零空间计算，直接构造满足下式的阻塞矩阵：

$$
\mathbf{B}^{H}
\mathbf{a}_{0}
=
\mathbf{0}
$$

---

## 18. 使用 SVD 构造阻塞矩阵

对于一般约束矩阵 `C`，需要找到：

$$
\mathbf{C}^{H}
$$

的零空间。

阻塞矩阵的列就是该零空间的一组正交基。

Python 示例：

```python
import numpy as np


def blocking_matrix(C, tolerance=1e-10):
    """
    构造满足 C^H B = 0 的阻塞矩阵。

    参数：
        C: 约束矩阵，尺寸为 N × Q

    返回：
        B: 阻塞矩阵，尺寸约为 N × (N-Q)
    """

    _, singular_values, Vh = np.linalg.svd(
        C.conj().T,
        full_matrices=True
    )

    if singular_values.size == 0:
        rank = 0
    else:
        threshold = (
            tolerance
            * singular_values[0]
        )

        rank = np.sum(
            singular_values > threshold
        )

    V = Vh.conj().T

    B = V[:, rank:]

    return B
```

检查阻塞条件：

```python
B = blocking_matrix(C)

error = C.conj().T @ B

print(error)
```

理论上，结果应接近零矩阵。

---

## 19. 固定权值的 Python 实现

可以使用：

```python
w_q = C @ np.linalg.solve(
    C.conj().T @ C,
    f
)
```

检查约束：

```python
response = C.conj().T @ w_q

print(response)
```

理论上结果应接近：

```text
f
```

---

## 20. 批量数据下的 GSC 实现

假设阵列数据矩阵为：

$$
\mathbf{X}
\in
\mathbb{C}^{N \times K}
$$

其中 `K` 是 Snapshot 数量。

固定支路输出：

```python
d = w_q.conj().T @ X
```

阻塞支路输出：

```python
U_block = B.conj().T @ X
```

阻塞支路协方差矩阵：

```python
R_uu = (
    U_block
    @ U_block.conj().T
    / K
)
```

阻塞支路与固定支路输出之间的互相关向量：

```python
r_ud = (
    U_block
    @ d.conj().T
    / K
)
```

为了提高稳定性，可以进行对角加载：

```python
loading = (
    1e-3
    * np.trace(R_uu).real
    / R_uu.shape[0]
)

R_uu_loaded = (
    R_uu
    + loading
    * np.eye(R_uu.shape[0])
)
```

求解最优自适应权值：

```python
g = np.linalg.solve(
    R_uu_loaded,
    r_ud
)
```

最终输出：

```python
y = (
    d
    - g.conj().T @ U_block
)
```

等效总权值：

```python
w_gsc = (
    w_q
    - B @ g
)
```

---

## 21. GSC 与 LCMV 的等价关系

LCMV 直接在整个 `N` 维权空间中寻找最优权值，但要求满足：

$$
\mathbf{C}^{H}
\mathbf{w}
=
\mathbf{f}
$$

GSC 把所有满足约束的权向量写成：

$$
\mathbf{w}
=
\mathbf{w}_{q}
-
\mathbf{B}
\mathbf{g}
$$

其中：

- `w_q` 是满足约束的一个特解；
- `B g` 是约束零空间内的调整量。

因为：

$$
\mathbf{C}^{H}
\mathbf{B}
=
\mathbf{0}
$$

所以：

$$
\mathbf{B}
\mathbf{g}
$$

不会改变约束响应。

GSC 实际上是在：

$$
N-Q
$$

维零空间中寻找最优调整量。

这就是 GSC 与 LCMV 数学等价的根本原因。

---

## 22. 为什么叫旁瓣相消器

固定波束形成器通常把主瓣对准目标方向。

目标信号主要从主瓣进入。

干扰通常从：

- 旁瓣；
- 后瓣；
- 非目标方向；

进入。

阻塞支路去掉目标后，主要保留这些非目标方向信号。

自适应支路再将其从主支路输出中减去。

因此最初称为：

**Sidelobe Canceller**

后来推广到：

- 多约束；
- 一般阵列；
- 一般阻塞矩阵；
- 更复杂的自适应结构；

所以称为：

**Generalized Sidelobe Canceller**

---

## 23. GSC 最大的问题：目标泄漏

GSC 能正常工作的关键是假设：

$$
\mathbf{B}^{H}
\mathbf{a}_{\mathrm{true}}
=
\mathbf{0}
$$

但实际算法中使用的导向矢量可能是：

$$
\mathbf{a}_{\mathrm{model}}
$$

真实导向矢量是：

$$
\mathbf{a}_{\mathrm{true}}
$$

如果两者不完全一致，则可能出现：

$$
\mathbf{B}^{H}
\mathbf{a}_{\mathrm{true}}
\neq
\mathbf{0}
$$

这表示目标信号泄漏到阻塞支路。

自适应滤波器可能把泄漏的目标信号误认为干扰，并尝试将其抵消。

最终可能导致：

- 目标信号衰减；
- 卫星信号自消除；
- 输出载噪比下降；
- 跟踪环失锁。

这个问题称为：

**Signal Leakage**

或：

**Target Leakage**

---

## 24. 导致目标泄漏的原因

GNSS 阵列中，目标泄漏可能来自：

- 阵元位置误差；
- 通道相位误差；
- 通道增益误差；
- 阵元互耦；
- 设备姿态误差；
- 星历方向误差；
- 天线方向图差异；
- 多径；
- 机体遮挡；
- 宽带频率失配；
- 极化失配。

GSC 对阻塞矩阵的准确性非常敏感。

---

## 25. 目标泄漏为什么会导致自消除

假设阻塞支路中包含少量目标信号：

$$
\mathbf{u}[k]
=
\mathbf{u}_{s}[k]
+
\mathbf{u}_{v}[k]
$$

其中：

- `u_s[k]` 是泄漏目标；
- `u_v[k]` 是干扰和噪声。

自适应滤波器的目标是最小化输出功率：

$$
E
\left[
|y[k]|^{2}
\right]
$$

只要泄漏目标与固定支路中的目标相关，自适应滤波器就会学习它。

于是自适应支路输出中也会包含目标信号副本。

当它从固定支路中减去时，就会发生目标自消除。

---

## 26. 如何提高 GSC 稳健性

常见方法包括：

### 26.1 对角加载

对阻塞支路协方差矩阵进行加载：

$$
\mathbf{R}_{uu,\mathrm{DL}}
=
\mathbf{R}_{uu}
+
\delta
\mathbf{I}
$$

可以降低权值过度放大。

### 26.2 多点约束

在目标方向附近设置多个保护点，扩大保护区域。

### 26.3 导数约束

限制目标方向附近的响应变化率。

### 26.4 鲁棒阻塞矩阵

根据阵列误差模型构造更加稳健的阻塞矩阵。

### 26.5 权值范数约束

限制自适应权值能量：

$$
\left\|
\mathbf{g}
\right\|^{2}
$$

避免自适应权值过大。

### 26.6 后相关处理

对每颗卫星解扩后再进行 GSC，可以提高目标与干扰的可分离性。

---

## 27. 使用 LMS 更新自适应权值

GSC 不一定要一次性计算维纳解。

也可以逐 Snapshot 更新自适应权值。

阻塞支路输出为：

$$
\mathbf{u}[k]
=
\mathbf{B}^{H}
\mathbf{x}[k]
$$

GSC 输出为：

$$
y[k]
=
d[k]
-
\mathbf{g}^{H}[k]
\mathbf{u}[k]
$$

一种 LMS 更新形式为：

$$
\mathbf{g}[k+1]
=
\mathbf{g}[k]
+
\mu
\mathbf{u}[k]
y^{*}[k]
$$

其中：

- `mu` 是步长；
- 步长太小，收敛慢；
- 步长太大，可能发散。

不同误差定义和符号约定可能导致更新式中的正负号不同，只要与输出定义保持一致即可。

---

## 28. LMS 的直观含义

如果当前输出误差较大，说明：

$$
\mathbf{g}^{H}[k]
\mathbf{u}[k]
$$

还没有很好地估计固定支路中的干扰。

LMS 根据：

- 当前阻塞支路数据；
- 当前输出误差；

不断调整自适应权值。

随着迭代进行：

$$
\mathbf{g}^{H}[k]
\mathbf{u}[k]
$$

会逐渐逼近固定支路中的干扰分量。

---

## 29. GSC 在 GNSS 前相关处理中的问题

如果 GSC 位于相关器之前，那么所有卫星共用同一套输出：

$$
y[k]
=
\mathbf{w}^{H}
\mathbf{x}[k]
$$

如果只把某一颗卫星作为目标约束，其他卫星可能进入阻塞支路，并被当作非目标信号抵消。

因此，前相关 GSC 通常不能简单地只保护一颗卫星。

可能需要：

- 多卫星约束；
- 天空区域约束；
- 只阻塞干扰子空间；
- 功率反演结构；
- 接受部分卫星衰减。

---

## 30. GSC 在 GNSS 后相关处理中的优势

后相关处理中，可以针对每颗卫星单独建立 GSC。

对于第 `i` 颗卫星：

- 固定支路保护第 `i` 颗卫星；
- 阻塞矩阵阻止该卫星进入辅助支路；
- 自适应支路抵消相关输出中的干扰和多径。

每颗卫星可以使用自己的：

$$
\mathbf{w}_{q,i}
$$

$$
\mathbf{B}_{i}
$$

$$
\mathbf{g}_{i}
$$

最终输出为：

$$
y_{i}[k]
=
\mathbf{w}_{q,i}^{H}
\mathbf{z}_{i}[k]
-
\mathbf{g}_{i}^{H}
\mathbf{B}_{i}^{H}
\mathbf{z}_{i}[k]
$$

这种方式更适合多卫星、多方向场景。

---

## 31. GSC 与前面算法的关系

```text
阵列接收数据
    │
    ▼
协方差矩阵
    │
    ├── MUSIC
    │     估计信号方向
    │
    ├── ESPRIT
    │     估计信号方向
    │
    ├── 空间平滑
    │     处理相干信号秩亏
    │
    ├── MVDR
    │     单约束最小方差
    │
    ├── LCMV
    │     多约束最小方差
    │
    └── GSC
          将 LCMV 分解为
          固定保护支路
          加自适应相消支路
```

---

## 32. GSC、MVDR 和 LCMV 的区别

| 方法 |   约束数量 | 主要实现形式       | 是否适合自适应迭代 |
| ---- | ---------: | ------------------ | -----------------: |
| MVDR |       一个 | 协方差矩阵求解     |               可以 |
| LCMV |       多个 | 受约束矩阵求解     |               可以 |
| GSC  | 一个或多个 | 固定支路加阻塞支路 |           非常适合 |

理想条件下，单约束 GSC 可以实现 MVDR。

多约束 GSC 可以实现 LCMV。

---

## 33. 本节核心公式

GSC 总权向量：

$$
\mathbf{w}
=
\mathbf{w}_{q}
-
\mathbf{B}
\mathbf{g}
$$

固定权值约束：

$$
\mathbf{C}^{H}
\mathbf{w}_{q}
=
\mathbf{f}
$$

阻塞矩阵条件：

$$
\mathbf{C}^{H}
\mathbf{B}
=
\mathbf{0}
$$

固定支路输出：

$$
d[k]
=
\mathbf{w}_{q}^{H}
\mathbf{x}[k]
$$

阻塞支路输出：

$$
\mathbf{u}[k]
=
\mathbf{B}^{H}
\mathbf{x}[k]
$$

最终输出：

$$
y[k]
=
d[k]
-
\mathbf{g}^{H}
\mathbf{u}[k]
$$

最优自适应权值：

$$
\mathbf{g}_{\mathrm{opt}}
=
\mathbf{R}_{uu}^{-1}
\mathbf{r}_{ud}
$$

最终总权值：

$$
\mathbf{w}_{\mathrm{GSC}}
=
\mathbf{w}_{q}
-
\mathbf{B}
\mathbf{g}_{\mathrm{opt}}
$$

---

## 34. 本节最核心的理解

LCMV 直接解决：

> 怎样找到一组满足约束并使输出功率最小的总权值。

GSC 将这个问题拆成：

```text
第一步：
固定权值负责保护目标

第二步：
阻塞矩阵去掉目标

第三步：
自适应滤波器学习干扰

第四步：
从固定支路中减去干扰估计
```

整体逻辑可以记成：

```text
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

---

## 35. 下一步

下一步可以做一个完整 GSC 仿真：

- 四阵元或八阵元 ULA；
- 一个目标信号；
- 两个强干扰；
- 构造固定权值；
- 使用 SVD 生成阻塞矩阵；
- 计算维纳解；
- 比较 LCMV 与 GSC 的最终权值；
- 验证两种方法的波束图基本一致；
- 人为加入导向矢量误差；
- 观察目标泄漏和目标自消除。
