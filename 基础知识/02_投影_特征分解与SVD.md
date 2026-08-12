---
tags:
  - 阵列信号处理
  - 数学基础
  - 特征分解
  - 奇异值分解
  - SVD
  - 协方差矩阵
  - 二次型
aliases:
  - 协方差矩阵与特征分解
  - 投影与特征值
created: 2026-07-19
---

# 第02章：投影、特征分解与 SVD

> 本章建立从向量投影到特征分解的完整数学链条，是理解信号子空间理论的核心基础。

前置阅读：[[01_阵列几何传播与窄带模型]]

---

## 1. 向量投影

### 1.1 单位方向向量

二维空间中，一个角度为 $\theta$ 的单位方向向量可以写为：

$$
\mathbf{u}
=
\begin{bmatrix}
\cos\theta \\
\sin\theta
\end{bmatrix}
$$

它满足：

$$
\mathbf{u}^T\mathbf{u}=1
$$

单位长度约束保证 $\mathbf{u}$ 只表示方向，不通过人为放大自身长度来放大投影结果。

---

### 1.2 投影坐标

设二维点为：

$$
\mathbf{x}
=
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
$$

它在单位方向 $\mathbf{u}$ 上的投影坐标为：

$$
z=\mathbf{u}^T\mathbf{x}
$$

展开得到：

$$
z=x_1\cos\theta+x_2\sin\theta
$$

$z$ 是一个标量，表示点在这条方向轴上的一维坐标。

> 这里是内积，而"向量长度乘以它与投影轴夹角的余弦"，正是几何上的投影长度。

---

### 1.3 投影向量

投影后的完整向量为：

$$
\mathbf{x}_{\mathrm{proj}}
=
(\mathbf{u}^T\mathbf{x})\mathbf{u}
$$

也可以写成：

$$
\mathbf{x}_{\mathrm{proj}}
=
\mathbf{u}\mathbf{u}^T\mathbf{x}
$$

对应的投影矩阵为：

$$
\mathbf{P}=\mathbf{u}\mathbf{u}^T
$$

所有点投影后都会落在方向 $\mathbf{u}$ 所在的同一条直线上。

---

## 2. 投影方差与二次型

### 2.1 投影后的方差

设数据已经去均值，投影结果为：

$$
z=\mathbf{u}^T\mathbf{x}
$$

则投影后的方差为：

$$
\operatorname{Var}(z)=E[z^2]
$$

代入投影公式：

$$
\operatorname{Var}(z)
=
E[(\mathbf{u}^T\mathbf{x})^2]
=
\mathbf{u}^T
E[\mathbf{x}\mathbf{x}^T]
\mathbf{u}
$$

利用协方差矩阵定义：

$$
\operatorname{Var}(z)
=
\mathbf{u}^T\mathbf{R}\mathbf{u}
$$

对于复数阵列信号，若输出为 $z=\mathbf{u}^H\mathbf{x}$，则平均输出功率为：

$$
E[|z|^2]
=
\mathbf{u}^H\mathbf{R}\mathbf{u}
$$

---

### 2.2 二次型的含义

表达式：

$$
\mathbf{u}^H\mathbf{R}\mathbf{u}
$$

称为**二次型**。

在协方差矩阵语境中，它表示数据沿方向 $\mathbf{u}$ 投影后的平均能量或方差。

可以把协方差矩阵理解为一个**方向测试器：**

1. 输入一个单位方向 $\mathbf{u}$；
2. 计算二次型；
3. 得到数据沿该方向有多分散或有多少能量。

---

## 3. 特征值和特征向量

### 3.1 特征值方程

如果存在非零向量 $\mathbf{v}$，满足：

$$
\mathbf{A}\mathbf{v}
=
\lambda\mathbf{v}
$$

则 $\mathbf{v}$ 是矩阵 $\mathbf{A}$ 的特征向量，$\lambda$ 是对应的特征值。

其几何含义是：矩阵作用在该向量上以后，向量方向不变，只发生比例缩放；当特征值为负数时还会发生方向翻转。

---

### 3.2 具体数值例子

取矩阵：

$$
\mathbf{A}
=
\begin{bmatrix}
2 & 1 \\
1 & 2
\end{bmatrix}
$$

特征值方程等价于：

$$
(\mathbf{A}-\lambda\mathbf{I})\mathbf{v}=0
$$

要使方程存在非零解，需要：

$$
\det(\mathbf{A}-\lambda\mathbf{I})=0
$$

计算得到特征值：

$$
\lambda_1=3,\quad \lambda_2=1
$$

当特征值为 $3$ 时，可以取特征向量 $\mathbf{v}_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$：

$$
\mathbf{A}\mathbf{v}_1
=
\begin{bmatrix} 3 \\ 3 \end{bmatrix}
=
3\mathbf{v}_1
$$

当特征值为 $1$ 时，可以取特征向量 $\mathbf{v}_2 = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$：

$$
\mathbf{A}\mathbf{v}_2
=
\begin{bmatrix} 1 \\ -1 \end{bmatrix}
=
\mathbf{v}_2
$$

---

### 3.3 普通矩阵与协方差矩阵中的含义区别

**对于一般线性变换矩阵：**

- 特征向量表示作用后方向保持不变的特殊方向；
- 特征值表示沿该方向的缩放倍数。

**对于协方差矩阵：**

- 特征向量表示数据点云的主轴方向；
- 特征值表示数据沿该主轴方向的方差或能量。

> 这是因为协方差矩阵本身编码的是数据在各方向上的分散程度。

---

## 4. 用具体协方差矩阵理解特征分解

### 4.1 示例协方差矩阵

考虑：

$$
\mathbf{R}
=
\begin{bmatrix}
5 & 4 \\
4 & 5
\end{bmatrix}
$$

取任意单位方向：

$$
\mathbf{u}(\theta)
=
\begin{bmatrix}
\cos\theta \\
\sin\theta
\end{bmatrix}
$$

该方向上的投影方差为：

$$
\mathbf{u}^T\mathbf{R}\mathbf{u}
=
5\cos^2\theta
+
8\sin\theta\cos\theta
+
5\sin^2\theta
$$

利用三角恒等式：

$$
\mathbf{u}^T\mathbf{R}\mathbf{u}
=
5+4\sin 2\theta
$$

---

### 4.2 不同方向上的方差

| 方向 $\theta$ | 投影方差 |
|:-------------:|:--------:|
| $0^\circ$ | $5$ |
| $45^\circ$ | $9$ |
| $90^\circ$ | $5$ |
| $135^\circ$ | $1$ |

因此点云沿 $45^\circ$ 方向分布最宽，沿 $135^\circ$ 方向分布最窄。

对应的单位特征向量和特征值为：

$$
\mathbf{v}_1
=
\frac{1}{\sqrt{2}}
\begin{bmatrix} 1 \\ 1 \end{bmatrix},\quad
\lambda_1=9
$$

$$
\mathbf{v}_2
=
\frac{1}{\sqrt{2}}
\begin{bmatrix} 1 \\ -1 \end{bmatrix},\quad
\lambda_2=1
$$

> 所以，协方差矩阵的特征向量正是点云的主轴，而特征值正是对应主轴上的方差。

---

## 5. 拉格朗日乘子与特征值方程

### 5.1 为什么需要约束

如果直接最大化二次型，而不限制向量长度，只要不断放大 $\mathbf{u}$，目标函数就会无限增大。

因此要求 $\mathbf{u}$ 为单位向量：

$$
\mathbf{u}^T\mathbf{u}=1
$$

优化问题为：

$$
\max_{\mathbf{u}}
\mathbf{u}^T\mathbf{R}\mathbf{u}
\quad \text{s.t.} \quad
\mathbf{u}^T\mathbf{u}=1
$$

---

### 5.2 拉格朗日乘子的几何含义

带约束极值点处，目标函数的等值面和约束曲面相切。

相切意味着两者的法向方向平行，而梯度就是等值面的法向量。因此：

$$
\nabla f
=
\lambda\nabla g
$$

$\lambda$ 用于补偿两个梯度向量长度的差异。

---

### 5.3 构造拉格朗日函数

构造：

$$
\mathcal{L}(\mathbf{u},\lambda)
=
\mathbf{u}^T\mathbf{R}\mathbf{u}
-
\lambda(\mathbf{u}^T\mathbf{u}-1)
$$

如果 R 为实对称矩阵，则：

$$
\frac{\partial}
{\partial\mathbf{u}}
(\mathbf{u}^T\mathbf{R}\mathbf{u})
=
2\mathbf{R}\mathbf{u}
$$

同时：

$$
\frac{\partial}
{\partial\mathbf{u}}
(\mathbf{u}^T\mathbf{u})
=
2\mathbf{u}
$$

令梯度为零：

$$
2\mathbf{R}\mathbf{u}
-
2\lambda\mathbf{u}
=
0
$$

整理得到：

$$
\mathbf{R}\mathbf{u}
=
\lambda\mathbf{u}
$$

这就是特征值方程。

---

### 5.4 为什么特征值等于投影方差

从特征值方程出发，左乘转置向量：

$$
\mathbf{u}^T\mathbf{R}\mathbf{u}
=
\lambda\mathbf{u}^T\mathbf{u}
=
\lambda
$$

左边是投影方差，所以**特征值就是对应特征方向上的方差**。

- 最大特征值对应最大方差方向
- 最小特征值对应最小方差方向

---

## 6. 协方差、SVD、秩与零空间

### 6.1 单个向量的外积

设 $\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$，外积为：

$$
\mathbf{x}\mathbf{x}^T
=
\begin{bmatrix}
x_1^2 & x_1x_2 \\
x_2x_1 & x_2^2
\end{bmatrix}
$$

- 对角元素记录各维度的瞬时平方能量；
- 非对角元素记录两个维度在该样本中的共同变化关系。

对大量样本的外积求平均，就能保留反复出现的稳定关系。

---

### 6.2 同方向数据的例子

假设所有样本都沿同一方向：

$$
\mathbf{x}[k]
=
\alpha_k
\begin{bmatrix} 1 \\ 1 \end{bmatrix}
$$

则：

$$
\mathbf{R}
=
E[\alpha_k^2]
\begin{bmatrix}
1 & 1 \\
1 & 1
\end{bmatrix}
$$

虽然单个样本的长度系数被平均掉了，但共同方向仍然保存在矩阵结构中。最大特征向量正是 $\begin{bmatrix} 1 \\ 1 \end{bmatrix}$。

---

### 6.3 协方差矩阵不是恢复原始样本

特征值分解不会恢复每一个快拍，也无法知道每个样本出现的先后顺序。

它提取的是：

- 哪些空间模式在大量数据中持续存在；
- 每个模式的平均能量有多大；
- 模式之间是否正交或相关。

> EVD 的作用是提取**统计结构**，而不是重建原始数据。

---

### 6.4 奇异值分解（SVD）

> 特征值分解（EVD）要求矩阵是方阵，且通常要求可对角化。奇异值分解（SVD）对**任意矩阵**都成立——无论方阵还是矩形、满秩还是秩亏、实数还是复数。它是比 EVD 更通用的矩阵结构分析工具。

---

#### 6.4.1 SVD 的基本形式

对于任意复矩阵 $\mathbf{A} \in \mathbb{C}^{m \times n}$，都可以进行奇异值分解：

$$
\boxed{
\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^H
}
$$

其中：

- $\mathbf{U} \in \mathbb{C}^{m \times m}$，$\mathbf{U}^H\mathbf{U} = \mathbf{I}_m$（左奇异向量组成的酉矩阵）
- $\mathbf{V} \in \mathbb{C}^{n \times n}$，$\mathbf{V}^H\mathbf{V} = \mathbf{I}_n$（右奇异向量组成的酉矩阵）
- $\boldsymbol{\Sigma} \in \mathbb{R}^{m \times n}$ 是一个矩形对角矩阵

$\boldsymbol{\Sigma}$ 的结构如下（以 $m \ge n$ 为例）：

$$
\boldsymbol{\Sigma}
=
\begin{bmatrix}
\sigma_1 & 0 & \cdots & 0 \\
0 & \sigma_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \sigma_r \\
& & & \mathbf{0}
\end{bmatrix}
$$

奇异值满足 $\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_r > 0$，其中 $r = \operatorname{rank}(\mathbf{A})$。剩余奇异值为零。

---

#### 6.4.2 SVD 最核心的直观理解

把矩阵映射 $\mathbf{y} = \mathbf{A}\mathbf{x}$ 拆成三个连续步骤，从右向左读：

```text
输入向量 x
    ↓
V^H：把 x 分解到一组特殊的输入方向上（右奇异向量）
    ↓
Σ：分别缩放这些方向上的坐标（乘以奇异值）
    ↓
U：把缩放后的坐标重新组合到输出空间（左奇异向量）
    ↓
输出向量 y
```

> $\mathbf{V}^H$ 负责换输入坐标系，$\boldsymbol{\Sigma}$ 负责伸缩，$\mathbf{U}$ 负责换到输出坐标系。

---

#### 6.4.3 单个奇异方向的映射——理解 SVD 的关键

设 $\mathbf{V}$ 的第 $i$ 列为 $\mathbf{v}_i$，$\mathbf{U}$ 的第 $i$ 列为 $\mathbf{u}_i$，那么：

$$
\boxed{\mathbf{A}\mathbf{v}_i = \sigma_i\mathbf{u}_i}
$$

这句话的含义是：

> 如果输入刚好沿着右奇异向量 $\mathbf{v}_i$，矩阵 $\mathbf{A}$ 不会把它变成任意混合方向，而是把它映射到左奇异向量 $\mathbf{u}_i$，并将长度缩放 $\sigma_i$ 倍。

```text
输入方向 v_i
    ↓ A
输出方向 u_i
长度乘以 σ_i
```

对于一般输入向量，先在右奇异向量基底下展开：

$$
\mathbf{x} = \sum_{i=1}^{n} \alpha_i\mathbf{v}_i,\quad
\alpha_i = \mathbf{v}_i^H\mathbf{x}
$$

利用线性性：

$$
\mathbf{A}\mathbf{x}
=
\sum_{i=1}^{n} \alpha_i \mathbf{A}\mathbf{v}_i
=
\sum_{i=1}^{n} \alpha_i\sigma_i\mathbf{u}_i
$$

代入 $\alpha_i$：

$$
\boxed{
\mathbf{A}\mathbf{x}
=
\sum_{i=1}^{n}
\sigma_i (\mathbf{v}_i^H\mathbf{x}) \mathbf{u}_i
}
$$

这就是矩阵作用过程的完整展开形式。

---

#### 6.4.4 为什么是 $\mathbf{V}^H\mathbf{x}$

$\mathbf{V}$ 的列向量组成一组正交归一基。要计算 $\mathbf{x}$ 在每个 $\mathbf{v}_i$ 方向上的坐标，需要做内积 $\mathbf{v}_i^H\mathbf{x}$。

把所有坐标放在一起：

$$
\begin{bmatrix}
\alpha_1 \\ \alpha_2 \\ \vdots \\ \alpha_n
\end{bmatrix}
=
\begin{bmatrix}
\mathbf{v}_1^H \\ \mathbf{v}_2^H \\ \vdots \\ \mathbf{v}_n^H
\end{bmatrix}
\mathbf{x}
=
\mathbf{V}^H\mathbf{x}
$$

所以 $\mathbf{V}^H\mathbf{x}$ 就是 $\mathbf{x}$ 在右奇异向量坐标系中的坐标。

---

#### 6.4.5 $\mathbf{U}$、$\boldsymbol{\Sigma}$、$\mathbf{V}$ 分别表示什么

**右奇异向量 $\mathbf{V}$（输入空间）：** 描述输入空间中的特殊正交方向 $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n$。矩阵 $\mathbf{A}$ 对每个方向的作用都可以单独描述，不会再相互混合。

**奇异值矩阵 $\boldsymbol{\Sigma}$（伸缩系数）：** 描述每个输入奇异方向被放大或压缩多少。

- $\sigma_i > 1$：该方向被放大
- $0 < \sigma_i < 1$：该方向被压缩
- $\sigma_i = 1$：该方向长度不变
- $\sigma_i = 0$：该方向被完全压缩为零——信息丢失

**左奇异向量 $\mathbf{U}$（输出空间）：** 描述输出空间中的正交方向 $\mathbf{u}_1, \mathbf{u}_2, \ldots, \mathbf{u}_m$。输入沿 $\mathbf{v}_i$ 的分量经过 $\sigma_i$ 缩放后，会沿着 $\mathbf{u}_i$ 输出。

---

#### 6.4.6 为什么可以直接计算 $\mathbf{A}\mathbf{x}$，还需要 SVD

直接计算 $\mathbf{y} = \mathbf{A}\mathbf{x}$ 只给出结果。SVD 告诉我们**结果为什么会这样**，并揭示矩阵的内部结构：

- 哪些输入方向最重要
- 哪些方向被强烈放大，哪些被压缩
- 哪些方向完全映射到零
- 矩阵的秩是多少
- 矩阵是否病态
- 零空间在哪里
- 如何做降维、构造伪逆
- 如何提取信号子空间和噪声子空间

> 直接乘法给出结果，SVD 解释结果为什么会这样，并揭示矩阵的空间结构。

---

#### 6.4.7 SVD 与特征值分解的关系

由 $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^H$ 可得：

$$
\mathbf{A}^H\mathbf{A}
=
\mathbf{V}\boldsymbol{\Sigma}^H\boldsymbol{\Sigma}\mathbf{V}^H
$$

所以 $\mathbf{V}$ 的列是 $\mathbf{A}^H\mathbf{A}$ 的特征向量，$\sigma_i^2$ 是 $\mathbf{A}^H\mathbf{A}$ 的特征值。

同理：

$$
\mathbf{A}\mathbf{A}^H
=
\mathbf{U}\boldsymbol{\Sigma}\boldsymbol{\Sigma}^H\mathbf{U}^H
$$

所以 $\mathbf{U}$ 的列是 $\mathbf{A}\mathbf{A}^H$ 的特征向量，$\sigma_i^2$ 也是 $\mathbf{A}\mathbf{A}^H$ 的非零特征值。

因此：

$$
\boxed{\sigma_i = \sqrt{\lambda_i(\mathbf{A}^H\mathbf{A})}}
$$

> **关键区别：** 奇异值通常不是 $\mathbf{A}$ 自身的特征值，而是 $\mathbf{A}^H\mathbf{A}$ 特征值的非负平方根。

对于数据矩阵 $\mathbf{X} \in \mathbb{C}^{N \times K}$：

$$
\mathbf{X}\mathbf{X}^H = \mathbf{U}\boldsymbol{\Sigma}^2\mathbf{U}^H
$$

- 协方差矩阵的特征向量 = 数据矩阵的左奇异向量
- 协方差矩阵的特征值 = 奇异值的平方除以 $K$
- EVD 与 SVD 对数据主要空间方向的描述一致

---

#### 6.4.8 为什么称为"奇异值"

"奇异"不是指它奇怪，而是和矩阵的**奇异性**有关。

当某个奇异值为零（$\sigma_i = 0$），就有 $\mathbf{A}\mathbf{v}_i = \mathbf{0}$，说明存在非零向量被矩阵映射成零。对于方阵，如果至少一个奇异值为零，则矩阵不可逆，也就是**奇异矩阵**。

> 奇异值直接反映矩阵在哪些方向上接近或达到奇异状态。

但 SVD 不只适用于奇异矩阵——它对**任意矩阵**（可逆方阵、不可逆方阵、高矩阵、宽矩阵、实矩阵、复矩阵）都成立。

---

#### 6.4.9 矩形矩阵为什么也能做 SVD

特征值分解通常要求矩阵是方阵。SVD 不要求方阵。

如果 $\mathbf{A} \in \mathbb{C}^{m \times n}$，那么：

- $\mathbf{A}^H\mathbf{A} \in \mathbb{C}^{n \times n}$（Hermitian 半正定）
- $\mathbf{A}\mathbf{A}^H \in \mathbb{C}^{m \times m}$（Hermitian 半正定）

这两个矩阵都可以进行稳定的特征值分解。通过它们可以得到输入空间的右奇异向量、输出空间的左奇异向量以及非负奇异值。

---

#### 6.4.10 完整 SVD 与经济型 SVD

**完整 SVD（full_matrices=True）：** 保留所有正交方向，包括对应零奇异值的方向。$\mathbf{U} \in \mathbb{C}^{m \times m}$，$\mathbf{V} \in \mathbb{C}^{n \times n}$。在寻找零空间时（如 GSC 阻塞矩阵）必须使用完整 SVD。

**经济型 SVD：** 只保留必要部分，更节省存储。但如果要获得完整零空间基底，经济型 SVD 可能不够。

```python
# 完整 SVD
U, s, Vh = np.linalg.svd(A, full_matrices=True)

# V 需要从 Vh 恢复（Vh = V^H）
V = Vh.conj().T
```

> 如果要通过 SVD 构造完整零空间或 GSC 阻塞矩阵，必须使用完整 SVD。

---

#### 6.4.11 秩与非零奇异值

矩阵的秩等于非零奇异值的数量：

$$
\operatorname{rank}(\mathbf{A}) = \#\{\sigma_i > 0\}
$$

实际数值计算中，奇异值可能不会精确等于零，通常使用阈值判断：

```python
rank = np.linalg.matrix_rank(A)
# 或手动：
tolerance = max(A.shape) * np.finfo(float).eps * s[0]
rank = np.sum(s > tolerance)
```

---

#### 6.4.12 零空间与零奇异值

矩阵的零空间定义为：

$$
\operatorname{null}(\mathbf{A}) = \{\mathbf{z} \mid \mathbf{A}\mathbf{z} = \mathbf{0}\}
$$

如果 $\sigma_i = 0$，那么 $\mathbf{A}\mathbf{v}_i = \sigma_i\mathbf{u}_i = \mathbf{0}$，所以 $\mathbf{v}_i \in \operatorname{null}(\mathbf{A})$。

> **与零奇异值对应的右奇异向量，构成矩阵零空间的一组正交基。**

如果 $\mathbf{A} \in \mathbb{C}^{m \times n}$ 且 $\operatorname{rank}(\mathbf{A}) = r$，则零空间维数为 $n - r$，右奇异向量 $\mathbf{v}_{r+1}, \ldots, \mathbf{v}_n$ 构成零空间基底。

**秩与零度定理：**

$$
\operatorname{rank}(\mathbf{A}) + \operatorname{nullity}(\mathbf{A}) = n
$$

- 秩：有多少个输入方向能产生独立输出
- 零度：有多少个输入方向被映射为零
- 两类方向合起来覆盖完整的 $n$ 维输入空间

---

#### 6.4.13 SVD 与 PCA

设数据矩阵 $\mathbf{X} \in \mathbb{C}^{N \times K}$，对其做 SVD：$\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^H$。样本协方差矩阵为：

$$
\widehat{\mathbf{R}} = \frac{1}{K}\mathbf{X}\mathbf{X}^H
= \frac{1}{K}\mathbf{U}\boldsymbol{\Sigma}\boldsymbol{\Sigma}^H\mathbf{U}^H
$$

因此：

- $\mathbf{U}$ 给出数据空间中的主方向
- $\sigma_i^2 / K$ 给出各主方向上的样本功率
- 大奇异值方向对应主要数据结构
- 小奇异值方向通常对应弱成分或噪声

---

#### 6.4.14 SVD 与 MUSIC

阵列快拍矩阵 $\mathbf{X} = \mathbf{A}_s\mathbf{S} + \mathbf{N}$。对协方差矩阵做特征分解，或者直接对数据矩阵做 SVD，都可以提取主要子空间。

若存在 $L$ 个独立信号，理想情况下前 $L$ 个大奇异值对应信号子空间，剩余较小奇异值对应噪声子空间。MUSIC 利用导向矢量与噪声子空间的正交性构造空间谱。详见 [[06_DOA估计_MUSIC_ESPRIT与空间平滑|MUSIC 算法]]。

---

#### 6.4.15 二维数值例子

**例 1：可逆矩阵**

$$
\mathbf{A} = \begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix}
$$

SVD 为 $\mathbf{U} = \mathbf{I}$，$\mathbf{V} = \mathbf{I}$，$\boldsymbol{\Sigma} = \operatorname{diag}(3, 1)$。

两个奇异方向分别为 $\mathbf{v}_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ 和 $\mathbf{v}_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$。对应映射：

$$
\mathbf{A}\mathbf{v}_1 = 3\mathbf{v}_1,\quad
\mathbf{A}\mathbf{v}_2 = 1\mathbf{v}_2
$$

水平方向被放大 3 倍，垂直方向长度不变。

**例 2：秩亏矩阵**

$$
\mathbf{A} = \begin{bmatrix} 1 & 1 \\ 2 & 2 \end{bmatrix},\quad \operatorname{rank}(\mathbf{A}) = 1
$$

存在非零向量 $\mathbf{v}_0 = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ -1 \end{bmatrix}$ 满足 $\mathbf{A}\mathbf{v}_0 = \mathbf{0}$，对应零奇异值。$\mathbf{v}_0$ 就是零空间方向。

---

#### 6.4.16 SVD 与 GSC 阻塞矩阵（预览）

GSC 中阻塞矩阵 $\mathbf{B}$ 需要满足 $\mathbf{C}^H\mathbf{B} = \mathbf{0}$，即 $\mathbf{B}$ 的每一列必须位于 $\mathbf{C}^H$ 的**零空间**中。这正是 SVD 最直接的应用：对 $\mathbf{C}^H$ 做完整 SVD，取零奇异值对应的右奇异向量即可得到 $\mathbf{B}$。

阻塞矩阵尺寸为 $N \times (N - \operatorname{rank}(\mathbf{C}^H))$——这正是秩与零度定理的直接推论。详细的构造方法和 Python 实现见 [[10_GSC与鲁棒阻塞矩阵|GSC 与鲁棒阻塞矩阵]] 的第 15 节。

---

#### 6.4.17 条件数与数值稳定性

矩阵的二范数条件数为最大奇异值与最小非零奇异值之比：

$$
\kappa_2(\mathbf{A}) = \frac{\sigma_{\max}}{\sigma_{\min}}
$$

如果 $\sigma_{\min} \approx 0$ 则 $\kappa_2 \gg 1$，矩阵病态。含义是：**输入或数据中的很小误差，经过求逆后可能被严重放大。** 这与 MVDR 中"小特征值被逆矩阵放大"的现象一致，也是对角加载的理论依据。

---

## 下一步

掌握这些数学工具后，先学习 [[03_有限快拍协方差估计与正则化]]，理解理论协方差在有限数据下会遇到的问题，再进入 [[04_信号子空间与噪声子空间]]。
