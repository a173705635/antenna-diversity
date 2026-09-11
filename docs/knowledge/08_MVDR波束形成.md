---
tags:
  - 阵列信号处理
  - 波束形成
  - MVDR
  - 抗干扰
  - Capon
aliases:
  - MVDR波束形成
  - 最小方差无失真响应
  - Capon波束形成
created: 2026-07-19
---

# 第08章：MVDR 最小方差无失真响应波束形成

> MVDR（Minimum Variance Distortionless Response）是最经典的 adaptive 波束形成算法。它在保证目标方向无失真通过的约束下，最小化阵列输出总功率，从而自适应地在干扰方向形成零陷。

前置阅读：[[03_有限快拍协方差估计与正则化]]、[[07_DOA功率与SINR估计]]

---

## 1. 学习目标

- 常规波束形成为什么不能自动抑制干扰
- MVDR 的优化目标是什么
- 为什么需要无失真约束
- MVDR 权值如何推导
- 协方差矩阵为什么能够使波束零陷指向干扰
- 为什么导向矢量失配会造成目标自消除
- 对角加载有什么作用

---

## 2. 波束形成输出

阵列接收向量为 $\mathbf{x}[k]$，波束形成权向量为 $\mathbf{w}$，输出为：

$$
y[k] = \mathbf{w}^{H}\mathbf{x}[k]
$$

输出功率为：

$$
\boxed{
P_{\mathrm{out}} = \mathbf{w}^{H}\mathbf{R}_x\mathbf{w}
}
$$

---

## 3. 常规波束形成

若目标方向为 $\theta_0$，对应导向矢量为 $\mathbf{a}_0 = \mathbf{a}(\theta_0)$，常规波束形成权值常取：

$$
\mathbf{w}_{\mathrm{CBF}} = \frac{\mathbf{a}_0}{\mathbf{a}_0^{H}\mathbf{a}_0}
$$

此时 $\mathbf{w}_{\mathrm{CBF}}^{H}\mathbf{a}_0 = 1$。

> 常规波束形成只根据目标方向设计，不利用实际干扰协方差。因此会在目标方向形成主瓣，但**不一定在干扰方向形成深零陷**。

---

## 4. MVDR 优化问题

MVDR 希望输出总功率尽可能小，同时保持目标方向增益为 1：

$$
\boxed{
\min_{\mathbf{w}} \mathbf{w}^{H}\mathbf{R}_x\mathbf{w}
\quad \text{s.t.} \quad
\mathbf{w}^{H}\mathbf{a}_0 = 1
}
$$

- **目标函数** 负责抑制干扰和噪声
- **约束条件** 负责保护目标信号

---

## 5. MVDR 权值推导

构造拉格朗日函数：

$$
\mathcal{L}
=
\mathbf{w}^{H}\mathbf{R}_x\mathbf{w}
+
\lambda^{*}
\left(
\mathbf{w}^{H}\mathbf{a}_0-1
\right)
+
\lambda
\left(
\mathbf{a}_0^{H}\mathbf{w}-1
\right)
$$

对复权向量求导并令其为零：

$$
\mathbf{R}_x\mathbf{w}
+
\lambda\mathbf{a}_0
=
\mathbf{0}
$$

因此：

$$
\mathbf{w}
=
-\lambda\mathbf{R}_x^{-1}\mathbf{a}_0
$$

代入无失真约束，最终得到：

$$
\boxed{
\mathbf{w}_{\mathrm{MVDR}}
=
\frac{
\mathbf{R}_x^{-1}\mathbf{a}_0
}{
\mathbf{a}_0^{H}\mathbf{R}_x^{-1}\mathbf{a}_0
}
}
$$

---

## 6. MVDR 的物理理解

协方差矩阵描述不同空间方向上的能量结构。高功率干扰方向在协方差矩阵中对应较大的能量模式。逆协方差矩阵会相对压低这些高能量模式。

因此 $\mathbf{R}_x^{-1}\mathbf{a}_0$ 可以理解为：

> 在保留目标方向信息的同时，对高干扰能量方向进行反向加权。

- **分子：** 根据接收数据的协方差结构，对目标方向导向矢量进行自适应调整，倾向于抑制协方差矩阵中能量较强的空间模式
- **分母：** 归一化因子，保证约束为 1

---

## 7. 协方差和逆协方差的空间作用（深入理解）

在理解 MVDR 时，经常会看到这样的说法：

> 协方差矩阵会对高能量空间模式进行更大的缩放，而逆协方差矩阵会对高能量空间模式进行更强的抑制。

这个结论并不是额外规定的，而是可以从以下三个概念直接推导出来：

1. 向量在某个方向上的投影
2. 投影信号的平均功率
3. 协方差矩阵的特征值和特征向量

下面逐步推导这一完整逻辑链。

---

### 7.1 阵列数据向某个方向投影

设阵列在第 $k$ 个快拍的接收数据为 $\mathbf{x}[k]$。选择一个单位向量 $\mathbf{u}$（$\mathbf{u}^{H}\mathbf{u} = 1$），并将阵列数据投影到该方向上。

投影结果为：

$$
z[k] = \mathbf{u}^{H}\mathbf{x}[k]
$$

- $\mathbf{x}[k]$ 是多维阵列数据
- $\mathbf{u}$ 表示一个空间方向或权向量
- $z[k]$ 是投影后得到的一个复数标量

$\mathbf{u}$ 是单位向量的约束保证了：它只表示方向，不会通过人为放大自身长度来放大投影结果。

---

### 7.2 投影信号的平均功率

投影信号的瞬时功率为 $|z[k]|^2$。平均功率为：

$$
P_{\mathbf{u}} = E\left[|z[k]|^2\right]
$$

代入投影表达式：

$$
P_{\mathbf{u}} = E\left[(\mathbf{u}^{H}\mathbf{x}[k])(\mathbf{u}^{H}\mathbf{x}[k])^{*}\right]
$$

由于 $(\mathbf{u}^{H}\mathbf{x}[k])^{*} = \mathbf{x}^{H}[k]\mathbf{u}$：

$$
P_{\mathbf{u}} = E\left[\mathbf{u}^{H}\mathbf{x}[k]\mathbf{x}^{H}[k]\mathbf{u}\right]
$$

向量 $\mathbf{u}$ 与随机变量无关，可以移到期望运算外面：

$$
P_{\mathbf{u}} = \mathbf{u}^{H}E\left[\mathbf{x}[k]\mathbf{x}^{H}[k]\right]\mathbf{u}
$$

阵列数据协方差矩阵定义为 $\mathbf{R} = E[\mathbf{x}[k]\mathbf{x}^{H}[k]]$，因此：

$$
\boxed{P_{\mathbf{u}} = \mathbf{u}^{H}\mathbf{R}\mathbf{u}}
$$

> **二次型 $\mathbf{u}^{H}\mathbf{R}\mathbf{u}$ 表示阵列数据在方向 $\mathbf{u}$ 上的平均功率。**

---

### 7.3 当投影方向是协方差矩阵的特征向量

设 $\mathbf{u}_i$ 是协方差矩阵 $\mathbf{R}$ 的一个单位特征向量，对应特征值为 $\lambda_i$。特征值方程为：

$$
\mathbf{R}\mathbf{u}_i = \lambda_i\mathbf{u}_i
$$

数据在特征向量方向 $\mathbf{u}_i$ 上的平均功率为：

$$
P_i = \mathbf{u}_i^{H}\mathbf{R}\mathbf{u}_i
$$

利用特征值方程 $\mathbf{R}\mathbf{u}_i = \lambda_i\mathbf{u}_i$：

$$
P_i = \mathbf{u}_i^{H}\lambda_i\mathbf{u}_i = \lambda_i\mathbf{u}_i^{H}\mathbf{u}_i = \lambda_i
$$

特征值是标量，特征向量已归一化（$\mathbf{u}_i^{H}\mathbf{u}_i = 1$），因此：

$$
\boxed{P_i = \lambda_i}
$$

> **协方差矩阵的特征值，等于数据在对应单位特征向量方向上的平均功率。**
>
> - 特征值越大，对应方向上的数据能量越强
> - 特征值越小，对应方向上的数据能量越弱

---

### 7.4 协方差矩阵如何作用于特征方向

特征值方程 $\mathbf{R}\mathbf{u}_i = \lambda_i\mathbf{u}_i$ 说明协方差矩阵作用在特征向量上时：

- 特征向量方向不变
- 特征向量长度被乘以特征值

例如 $\lambda_1 = 100$，则 $\mathbf{R}\mathbf{u}_1 = 100\mathbf{u}_1$，协方差矩阵将该方向放大 100 倍。如果 $\lambda_2 = 2$，则 $\mathbf{R}\mathbf{u}_2 = 2\mathbf{u}_2$，该方向只被放大 2 倍。

前面已经证明 $\lambda_i = P_i$，所以：

> **数据能量越大的特征方向，对应特征值越大，协方差矩阵对该方向的缩放倍数也越大。**

---

### 7.5 协方差矩阵如何作用于任意向量

协方差矩阵是厄米矩阵，它的特征向量可以构成一组正交完备基。任意向量 $\mathbf{v}$ 都可以在这些特征向量上展开：

$$
\mathbf{v} = \sum_{i=1}^{N} c_i\mathbf{u}_i,\quad c_i = \mathbf{u}_i^{H}\mathbf{v}
$$

使用协方差矩阵作用于该向量，利用线性性质和特征值关系：

$$
\boxed{\mathbf{R}\mathbf{v} = \sum_{i=1}^{N} c_i\lambda_i\mathbf{u}_i}
$$

也可以写成：

$$
\boxed{\mathbf{R}\mathbf{v} = \sum_{i=1}^{N} \lambda_i(\mathbf{u}_i^{H}\mathbf{v})\mathbf{u}_i}
$$

这说明协方差矩阵会**把任意向量拆分到各个特征方向上，然后分别乘以对应特征值**。因此：

- 高能量方向对应较大特征值，分量被放大更多
- 低能量方向对应较小特征值，分量被放大更少

---

### 7.6 二维数值例子

假设协方差矩阵已经处于其特征向量坐标系中：

$$
\mathbf{R} = \begin{bmatrix} 100 & 0 \\ 0 & 2 \end{bmatrix}
$$

这表示第一个特征方向上的平均功率为 100，第二个特征方向上的平均功率为 2。

取一个向量 $\mathbf{v} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$，原向量在两个特征方向上的分量都是 1。协方差矩阵作用在该向量上：

$$
\mathbf{R}\mathbf{v} = \begin{bmatrix} 100 & 0 \\ 0 & 2 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 100 \\ 2 \end{bmatrix}
$$

- 第一个方向的分量从 1 变成 100
- 第二个方向的分量从 1 变成 2

结果明显偏向第一个高能量方向。

---

### 7.7 为什么协方差矩阵会积累出高能量方向

协方差矩阵定义为 $\mathbf{R} = E[\mathbf{x}[k]\mathbf{x}^{H}[k]]$，它是大量快拍外积的平均。

对于单个快拍，外积 $\mathbf{x}[k]\mathbf{x}^{H}[k]$ 表示该快拍沿各个方向的幅度和相关性结构。如果大量数据都主要沿某个方向变化，那么每次外积都会在该方向积累较大的数值。经过大量快拍平均后，该方向就会形成一个较大的特征值。

---

### 7.8 一个简单的数据分布例子

假设二维数据主要沿第一个坐标轴变化：

$$
\mathbf{x}[k] \approx \begin{bmatrix} s[k] \\ 0 \end{bmatrix}
$$

单个快拍的外积为：

$$
\mathbf{x}[k]\mathbf{x}^{H}[k] \approx \begin{bmatrix} |s[k]|^2 & 0 \\ 0 & 0 \end{bmatrix}
$$

取平均后：

$$
\mathbf{R} \approx \begin{bmatrix} E[|s[k]|^2] & 0 \\ 0 & 0 \end{bmatrix}
$$

第一个方向的特征值为 $E[|s[k]|^2]$，第二个方向的特征值接近 0。这说明：

> 数据主要变化的方向形成大特征值，数据几乎没有变化的方向形成小特征值。协方差矩阵正是通过大量数据外积的平均，把数据中的高能量方向积累成较大的特征值。

---

### 7.9 强干扰为什么会形成大特征值

假设阵列接收到一个强干扰：

$$
\mathbf{x}[k] = \mathbf{a}_i s_i[k] + \mathbf{n}[k]
$$

其中 $\mathbf{a}_i$ 是干扰导向矢量，$s_i[k]$ 是干扰信号，$\mathbf{n}[k]$ 是噪声。

协方差矩阵为：

$$
\mathbf{R} = \sigma_i^2\mathbf{a}_i\mathbf{a}_i^{H} + \sigma_n^2\mathbf{I}
$$

其中 $\sigma_i^2 = E[|s_i[k]|^2]$。如果干扰很强（$\sigma_i^2 \gg \sigma_n^2$），那么干扰导向矢量所张成的空间方向会对应一个很大的特征值。

> 因此，强干扰会在协方差矩阵中表现为高能量特征模式。

---

### 7.10 逆协方差矩阵的作用

对协方差矩阵进行特征分解：

$$
\mathbf{R} = \mathbf{U}\boldsymbol{\Lambda}\mathbf{U}^{H},\quad
\boldsymbol{\Lambda} = \operatorname{diag}(\lambda_1, \lambda_2, \ldots, \lambda_N)
$$

逆协方差矩阵为：

$$
\mathbf{R}^{-1} = \mathbf{U}\boldsymbol{\Lambda}^{-1}\mathbf{U}^{H},\quad
\boldsymbol{\Lambda}^{-1} = \operatorname{diag}\left(\frac{1}{\lambda_1}, \frac{1}{\lambda_2}, \ldots, \frac{1}{\lambda_N}\right)
$$

因此逆协方差矩阵对特征方向执行相反的缩放：

$$
\mathbf{R}^{-1}\mathbf{u}_i = \frac{1}{\lambda_i}\mathbf{u}_i
$$

>
> **特征值越大，逆缩放系数越小。** 即 $\lambda_i$ 越大 $\Longrightarrow \frac{1}{\lambda_i}$ 越小。

---

### 7.11 逆协方差矩阵作用于任意向量

任意向量 $\mathbf{v} = \sum_{i=1}^{N} c_i\mathbf{u}_i$，逆协方差矩阵作用后：

$$
\boxed{\mathbf{R}^{-1}\mathbf{v} = \sum_{i=1}^{N} \frac{c_i}{\lambda_i}\mathbf{u}_i}
$$

也可以写成：

$$
\boxed{\mathbf{R}^{-1}\mathbf{v} = \sum_{i=1}^{N} \frac{\mathbf{u}_i^{H}\mathbf{v}}{\lambda_i}\mathbf{u}_i}
$$

因此：**高能量模式对应的大特征值会出现在分母中，使该方向的分量被明显压低。**

---

### 7.12 与 MVDR 的联系

MVDR 权值为：

$$
\mathbf{w}_{\mathrm{MVDR}} = \frac{\mathbf{R}^{-1}\mathbf{a}_0}{\mathbf{a}_0^{H}\mathbf{R}^{-1}\mathbf{a}_0}
$$

将目标导向矢量 $\mathbf{a}_0$ 展开到协方差矩阵的特征向量上：

$$
\mathbf{a}_0 = \sum_{i=1}^{N} c_i\mathbf{u}_i,\quad c_i = \mathbf{u}_i^{H}\mathbf{a}_0
$$

经过逆协方差矩阵加权后：

$$
\boxed{\mathbf{R}^{-1}\mathbf{a}_0 = \sum_{i=1}^{N} \frac{\mathbf{u}_i^{H}\mathbf{a}_0}{\lambda_i}\mathbf{u}_i}
$$

因此：

- 强干扰模式的特征值很大 → 对应分量除以大特征值 → 该分量被明显压低
- 最终权向量会尽量避开强干扰空间模式

之后再通过分母进行归一化，使目标方向响应满足无失真约束。

---

### 7.13 为什么还需要归一化

仅使用 $\mathbf{R}^{-1}\mathbf{a}_0$ 虽然能够降低高能量空间模式，但不能保证目标方向增益为 1。

因此需要除以 $\mathbf{a}_0^{H}\mathbf{R}^{-1}\mathbf{a}_0$，最终权值为：

$$
\mathbf{w}_{\mathrm{MVDR}} = \frac{\mathbf{R}^{-1}\mathbf{a}_0}{\mathbf{a}_0^{H}\mathbf{R}^{-1}\mathbf{a}_0}
$$

检查目标方向响应：

$$
\mathbf{w}_{\mathrm{MVDR}}^{H}\mathbf{a}_0 = \frac{\mathbf{a}_0^{H}\mathbf{R}^{-1}\mathbf{a}_0}{\mathbf{a}_0^{H}\mathbf{R}^{-1}\mathbf{a}_0} = 1
$$

所以 MVDR 同时实现：

1. 利用逆协方差矩阵压低高能量干扰模式
2. 利用归一化约束保持目标方向单位响应

---

### 7.14 为什么协方差矩阵作用后的结果会偏向高能量方向

设 $\mathbf{v} = c_1\mathbf{u}_1 + c_2\mathbf{u}_2$，对应特征值 $\lambda_1 \gg \lambda_2$。

协方差矩阵作用后：

$$
\mathbf{R}\mathbf{v} = c_1\lambda_1\mathbf{u}_1 + c_2\lambda_2\mathbf{u}_2
$$

即使原始系数 $c_1$ 和 $c_2$ 大小接近，由于 $\lambda_1 \gg \lambda_2$，处理后的第一项仍然可能远大于第二项。所以结果方向会更接近 $\mathbf{u}_1$。

> 这也是反复使用协方差矩阵进行幂迭代时，可以逐渐找到最大特征值对应特征向量的原因。

---

### 7.15 一个容易混淆的问题

需要区分两个不同结论。

**结论一：特征值表示投影功率。** 当方向为单位特征向量时：

$$
\mathbf{u}_i^{H}\mathbf{R}\mathbf{u}_i = \lambda_i
$$

这表示数据在该方向上的平均功率为特征值。

**结论二：协方差矩阵对特征方向进行缩放。** 特征值方程为：

$$
\mathbf{R}\mathbf{u}_i = \lambda_i\mathbf{u}_i
$$

这表示协方差矩阵对该特征方向的缩放倍数为特征值。

这两个结论中的特征值相同，不是巧合。原因是协方差矩阵的特征方向正是数据能量分布的自然主轴。

---

### 7.16 对角加载后的变化

工程中常使用对角加载协方差矩阵：

$$
\mathbf{R}_{\mathrm{L}} = \mathbf{R} + \delta\mathbf{I}
$$

若原协方差矩阵特征分解为 $\mathbf{R} = \mathbf{U}\boldsymbol{\Lambda}\mathbf{U}^{H}$，则：

$$
\mathbf{R}_{\mathrm{L}} = \mathbf{U}(\boldsymbol{\Lambda} + \delta\mathbf{I})\mathbf{U}^{H}
$$

加载后的特征值变为 $\lambda_i + \delta$，逆协方差缩放系数变为 $\frac{1}{\lambda_i + \delta}$。

对角加载**不会改变理想情况下的特征向量方向**，但会减小不同特征模式逆权重之间的差异：

- 改善数值稳定性
- 降低协方差估计误差的影响
- 减少权值过度放大
- 提高对导向矢量失配的稳健性

但加载过大也会削弱自适应干扰抑制能力。

---

### 7.17 核心推导总结

从投影信号 $z[k] = \mathbf{u}^{H}\mathbf{x}[k]$ 开始：

**投影功率：**

$$
E[|z[k]|^2] = \mathbf{u}^{H}\mathbf{R}\mathbf{u}
$$

**当投影方向是单位特征向量时：**

$$
\mathbf{u}_i^{H}\mathbf{R}\mathbf{u}_i = \lambda_i
$$

> 特征值就是该方向上的平均功率。

**协方差矩阵对特征方向的作用：**

$$
\mathbf{R}\mathbf{u}_i = \lambda_i\mathbf{u}_i
$$

> 高能量方向对应大特征值，会被协方差矩阵放大更多。

**逆协方差矩阵的作用：**

$$
\mathbf{R}^{-1}\mathbf{u}_i = \frac{1}{\lambda_i}\mathbf{u}_i
$$

> 高能量方向会被逆协方差矩阵压低更多。

**MVDR 中最关键的关系：**

$$
\mathbf{R}^{-1}\mathbf{a}_0 = \sum_{i=1}^{N} \frac{\mathbf{u}_i^{H}\mathbf{a}_0}{\lambda_i}\mathbf{u}_i
$$

> 强干扰对应的大特征值出现在分母中，使对应空间模式受到明显抑制。

---

### 7.18 最终结论

协方差矩阵之所以能够描述数据的能量方向，是因为：

$$
\mathbf{R} = E[\mathbf{x}\mathbf{x}^{H}]
$$

它通过对大量数据外积求平均，把数据长期存在的高能量方向积累成较大的特征值。因此：

$$
\boxed{\text{特征值} = \text{对应特征方向上的平均功率}}
$$

同时：

$$
\boxed{\mathbf{R}\text{ 对该方向的缩放倍数} = \lambda_i}
$$

而逆协方差矩阵满足：

$$
\boxed{\mathbf{R}^{-1}\text{ 对该方向的缩放倍数} = \frac{1}{\lambda_i}}
$$

> **协方差矩阵会突出高能量空间模式，而逆协方差矩阵会抑制高能量空间模式。这正是 MVDR 能够自适应抑制强干扰的数学基础。**

---

## 8. 为什么 MVDR 会形成零陷

设强干扰方向为 $\theta_i$，干扰协方差中包含 $\sigma_i^2\mathbf{a}_i\mathbf{a}_i^{H}$。

当干扰功率很大时，优化问题会强烈惩罚权向量在干扰空间方向上的响应。因此最优权值会尽量满足：

$$
\mathbf{w}^{H}\mathbf{a}_i \approx 0
$$

> **零陷不是人工指定的，而是最小化输出功率自动产生的。**

---

## 9. 样本协方差矩阵

实际系统通过 $K$ 个快拍估计：

$$
\widehat{\mathbf{R}}_x = \frac{1}{K}\mathbf{X}\mathbf{X}^{H}
$$

代码实现中**不建议显式求逆**，应使用线性方程求解：

```python
v = np.linalg.solve(R_loaded, a0)
w = v / (a0.conj().T @ v)
```

---

## 10. 对角加载

有限快拍、模型误差和强相关信号可能使样本协方差矩阵病态。

对角加载后的协方差矩阵：

$$
\boxed{
\mathbf{R}_{\mathrm{DL}} = \widehat{\mathbf{R}}_x + \delta\mathbf{I}
},\quad \delta > 0
$$

> 对角加载不是越大越好，而是**稳健性与自适应能力之间的折中**：
> - 加载较小：零陷更深，但可能不稳定
> - 加载较大：更稳健，但零陷可能变浅
> - 加载很大：趋近常规波束形成

---

## 11. 目标自消除与训练协方差选择

MVDR 使用的保护导向矢量为 $\widehat{\mathbf{a}}_0$，若与真实导向矢量 $\mathbf{a}_0$ 不一致，MVDR 只保证 $\mathbf{w}^{H}\widehat{\mathbf{a}}_0 = 1$，不保证 $\mathbf{w}^{H}\mathbf{a}_0 = 1$。

如果目标信号被包含在样本协方差矩阵中，优化器可能把未被正确约束的目标成分误认为需要压低的输出能量，最终在真实目标方向附近形成零陷。

**这称为目标自消除（Signal Self-Cancellation）。**

### 11.1 目标究竟能不能进入协方差

理论 MVDR 可以直接使用总接收协方差 $\mathbf R_x$，其中同时包含目标、干扰和噪声。若真实导向矢量与约束完全一致，目标分量虽然进入目标函数，仍无法在受保护方向被压低。

工程风险来自两个条件同时出现：

1. 目标能量进入训练协方差；
2. 保护导向矢量与真实目标存在失配。

此时算法只保护模型方向，真实目标中落在约束外的分量会被当成可降低的功率。

因此协方差选择分为两类：

| 选择 | 优点 | 风险或代价 |
|---|---|---|
| 总接收协方差 $\mathbf R_x$ | 容易在线估计，包含当前真实环境 | 导向失配时更容易自消除 |
| 干扰加噪声协方差 $\mathbf R_{i+n}$ | 不把目标能量纳入最小化目标 | 需要目标缺失训练、码域抑制或额外分离手段 |

在 GNSS 中，后相关处理可以利用不同 PRN 码为每颗卫星构造更有针对性的训练或独立权值；前相关公共权值则必须更重视多星保护、加载和校准。

导致导向矢量失配的原因包括：
- 阵元位置误差
- 通道相位误差
- 互耦
- 姿态误差
- 天线方向图误差

工程上的应对策略：
- 使用目标缺失时的数据估计干扰加噪声协方差
- 对角加载
- 导向矢量不确定集
- 多点保护约束
- 宽主瓣约束
- 后相关后为每颗卫星独立设计权值

---

## 12. 输出 SINR 评估

输出目标功率：$P_s = \mathbf{w}^{H}\mathbf{R}_s\mathbf{w}$

输出干扰加噪声功率：$P_{i+n} = \mathbf{w}^{H}(\mathbf{R}_i + \mathbf{R}_n)\mathbf{w}$

$$
\boxed{
\mathrm{SINR}_{\mathrm{out}}
=
\frac{\mathbf{w}^{H}\mathbf{R}_s\mathbf{w}}
{\mathbf{w}^{H}(\mathbf{R}_i + \mathbf{R}_n)\mathbf{w}}
}
$$

> 仿真中单独构造理论目标、干扰和噪声协方差，通常只是为了评估输出 SINR，不一定参与 MVDR 权值设计。

---

## 13. MVDR 常见疑惑

| 疑惑 | 解答 |
|------|------|
| MVDR 是否需要提前知道干扰方向？ | 不一定。标准 MVDR 通过样本协方差矩阵学习高能量干扰结构 |
| 为什么强干扰方向会自动出现零陷？ | 输出功率目标函数对强干扰方向产生很大惩罚 |
| MVDR 与 MUSIC 是否相同？ | 不同。MUSIC 用于估计方向，MVDR 用于设计波束形成权值 |
| 为什么不用显式矩阵逆？ | 线性方程求解通常比显式矩阵求逆更稳定 |
| 对角加载是不是越大越好？ | 不是。加载太大，自适应能力减弱，结果趋近常规波束形成 |
| 为什么最小输出功率能够抑制干扰？ | 目标信号无法被压低（受约束保护），优化器只能尽量降低干扰和噪声功率 |

---

## 下一步

MVDR 只有一个约束。当需要同时保护多个方向或显式设置零陷时，进入 [[09_LCMV与GNSS多星自由度|LCMV 与 GNSS 多星自由度]]。
