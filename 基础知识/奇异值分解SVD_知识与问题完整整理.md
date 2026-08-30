# 奇异值分解 SVD：从直观理解到 GSC 阻塞矩阵构造

本文整理奇异值分解相关的核心知识，并集中回答此前学习过程中出现过的问题，包括：

- 奇异值分解到底在做什么；
- 为什么已经可以直接计算 $\mathbf A\mathbf x$，还需要 SVD；
- $\mathbf U$、$\mathbf \Sigma$、$\mathbf V$ 分别表示什么；
- 为什么是 $\mathbf V^H\mathbf x$；
- 奇异值与特征值是什么关系；
- 为什么称为“奇异值”；
- 矩形矩阵怎样进行分解；
- 秩、零空间和零奇异值之间是什么关系；
- 为什么 GSC 阻塞矩阵尺寸是 $N\times(N-Q)$；
- 怎样通过 SVD 得到阻塞矩阵；
- Python 中 `Vh`、`V` 和 `B` 应该怎样取。

---

## 1. SVD 的基本形式

对于任意复矩阵：

$$
\mathbf A\in\mathbb C^{m\times n}
$$

都可以进行奇异值分解：

$$
\mathbf A
=
\mathbf U
\mathbf \Sigma
\mathbf V^H
$$

其中：

$$
\mathbf U\in\mathbb C^{m\times m}
$$

$$
\mathbf V\in\mathbb C^{n\times n}
$$

并且：

$$
\mathbf U^H\mathbf U
=
\mathbf I_m
$$

$$
\mathbf V^H\mathbf V
=
\mathbf I_n
$$

也就是说，$\mathbf U$ 和 $\mathbf V$ 都是酉矩阵。

$\mathbf\Sigma$ 是一个 $m\times n$ 的矩形对角矩阵：

$$
\mathbf\Sigma
=
\begin{bmatrix}
\sigma_1 & 0 & \cdots & 0\\
0 & \sigma_2 & \cdots & 0\\
\vdots & \vdots & \ddots & \vdots\\
0 & 0 & \cdots & \sigma_r\\
& & \mathbf 0 &
\end{bmatrix}
$$

奇异值满足：

$$
\sigma_1
\ge
\sigma_2
\ge
\cdots
\ge
\sigma_r
>
0
$$

其中：

$$
r
=
\operatorname{rank}(\mathbf A)
$$

剩余奇异值为零。

---

## 2. SVD 最核心的直观理解

SVD 把矩阵映射：

$$
\mathbf y
=
\mathbf A\mathbf x
$$

拆成三个连续步骤：

$$
\mathbf y
=
\mathbf U
\mathbf\Sigma
\mathbf V^H
\mathbf x
$$

按照从右向左的顺序理解：

```text
输入向量 x
    ↓
V^H：把 x 分解到一组特殊的输入方向上
    ↓
Σ：分别缩放这些方向上的坐标
    ↓
U：把缩放后的坐标重新组合到输出空间
    ↓
输出向量 y
```

因此可以记成：

> $\mathbf V^H$ 负责换输入坐标系，$\mathbf\Sigma$ 负责伸缩，$\mathbf U$ 负责换到输出坐标系。

---

## 3. 单个奇异方向的映射

设 $\mathbf V$ 的第 $i$ 列为：

$$
\mathbf v_i
$$

$\mathbf U$ 的第 $i$ 列为：

$$
\mathbf u_i
$$

那么：

$$
\mathbf A\mathbf v_i
=
\sigma_i\mathbf u_i
$$

这句话是理解 SVD 的关键：

> 如果输入刚好沿着右奇异向量 $\mathbf v_i$，矩阵 $\mathbf A$ 不会把它变成任意混合方向，而是把它映射到左奇异向量 $\mathbf u_i$，并将长度缩放 $\sigma_i$ 倍。

即：

```text
输入方向 v_i
    ↓ A
输出方向 u_i
长度乘以 σ_i
```

---

## 4. 一般输入向量怎样通过 SVD

任意输入向量都可以在右奇异向量基底下展开：

$$
\mathbf x
=
\sum_{i=1}^{n}
\alpha_i\mathbf v_i
$$

其中坐标为：

$$
\alpha_i
=
\mathbf v_i^H\mathbf x
$$

所以：

$$
\mathbf A\mathbf x
=
\mathbf A
\sum_{i=1}^{n}
\alpha_i\mathbf v_i
$$

利用线性性：

$$
\mathbf A\mathbf x
=
\sum_{i=1}^{n}
\alpha_i\mathbf A\mathbf v_i
$$

再利用：

$$
\mathbf A\mathbf v_i
=
\sigma_i\mathbf u_i
$$

得到：

$$
\mathbf A\mathbf x
=
\sum_{i=1}^{n}
\alpha_i\sigma_i\mathbf u_i
$$

代入 $\alpha_i$：

$$
\mathbf A\mathbf x
=
\sum_{i=1}^{n}
\sigma_i
\left(
\mathbf v_i^H\mathbf x
\right)
\mathbf u_i
$$

这就是矩阵作用过程的展开形式。

---

## 5. 为什么是 $\mathbf V^H\mathbf x$

$\mathbf V$ 的列向量组成一组正交归一基：

$$
\mathbf V
=
\begin{bmatrix}
\mathbf v_1 &
\mathbf v_2 &
\cdots &
\mathbf v_n
\end{bmatrix}
$$

要计算 $\mathbf x$ 在每个 $\mathbf v_i$ 方向上的坐标，需要做内积：

$$
\alpha_i
=
\mathbf v_i^H\mathbf x
$$

把所有坐标放在一起：

$$
\begin{bmatrix}
\alpha_1\\
\alpha_2\\
\vdots\\
\alpha_n
\end{bmatrix}
=
\begin{bmatrix}
\mathbf v_1^H\\
\mathbf v_2^H\\
\vdots\\
\mathbf v_n^H
\end{bmatrix}
\mathbf x
$$

而：

$$
\begin{bmatrix}
\mathbf v_1^H\\
\mathbf v_2^H\\
\vdots\\
\mathbf v_n^H
\end{bmatrix}
=
\mathbf V^H
$$

所以：

$$
\mathbf V^H\mathbf x
$$

就是 $\mathbf x$ 在右奇异向量坐标系中的坐标。

---

## 6. $\mathbf U$、$\mathbf\Sigma$、$\mathbf V$ 分别表示什么

### 6.1 右奇异向量 $\mathbf V$

$\mathbf V$ 描述输入空间中的特殊正交方向：

$$
\mathbf v_1,\mathbf v_2,\ldots,\mathbf v_n
$$

这些方向具有一个特点：

> 矩阵 $\mathbf A$ 对每个方向的作用都可以单独描述，不会再相互混合。

### 6.2 奇异值矩阵 $\mathbf\Sigma$

$\mathbf\Sigma$ 描述每个输入奇异方向被放大或压缩多少。

对于第 $i$ 个方向：

$$
\mathbf v_i
\longrightarrow
\sigma_i\mathbf u_i
$$

因此：

- $\sigma_i>1$：该方向被放大；
- $0<\sigma_i<1$：该方向被压缩；
- $\sigma_i=1$：该方向长度不变；
- $\sigma_i=0$：该方向被完全压缩为零。

### 6.3 左奇异向量 $\mathbf U$

$\mathbf U$ 描述输出空间中的正交方向：

$$
\mathbf u_1,\mathbf u_2,\ldots,\mathbf u_m
$$

输入沿 $\mathbf v_i$ 的分量经过缩放后，会沿着 $\mathbf u_i$ 输出。

---

## 7. 为什么可以直接计算 $\mathbf A\mathbf x$，还需要 SVD

数值上当然可以直接计算：

$$
\mathbf y
=
\mathbf A\mathbf x
$$

SVD 并不是为了让一次矩阵乘法更简单，而是为了暴露矩阵内部结构。

直接计算只能得到：

$$
\mathbf y
$$

但 SVD 还能告诉我们：

- 哪些输入方向最重要；
- 哪些方向被强烈放大；
- 哪些方向被压缩；
- 哪些方向完全映射到零；
- 矩阵的秩是多少；
- 矩阵是否病态；
- 零空间在哪里；
- 如何做降维；
- 如何构造伪逆；
- 如何提取信号子空间和噪声子空间。

所以：

> 直接乘法给出结果，SVD 解释结果为什么会这样，并揭示矩阵的空间结构。

---

## 8. SVD 与特征值分解的关系

SVD 与以下两个矩阵的特征值分解直接相关：

$$
\mathbf A^H\mathbf A
$$

和：

$$
\mathbf A\mathbf A^H
$$

由：

$$
\mathbf A
=
\mathbf U\mathbf\Sigma\mathbf V^H
$$

可得：

$$
\mathbf A^H\mathbf A
=
\mathbf V
\mathbf\Sigma^H\mathbf\Sigma
\mathbf V^H
$$

所以：

- $\mathbf V$ 的列是 $\mathbf A^H\mathbf A$ 的特征向量；
- $\sigma_i^2$ 是 $\mathbf A^H\mathbf A$ 的特征值。

同理：

$$
\mathbf A\mathbf A^H
=
\mathbf U
\mathbf\Sigma\mathbf\Sigma^H
\mathbf U^H
$$

所以：

- $\mathbf U$ 的列是 $\mathbf A\mathbf A^H$ 的特征向量；
- $\sigma_i^2$ 也是 $\mathbf A\mathbf A^H$ 的非零特征值。

因此：

$$
\sigma_i
=
\sqrt{\lambda_i(\mathbf A^H\mathbf A)}
$$

需要特别注意：

> 奇异值通常不是 $\mathbf A$ 自身的特征值，而是 $\mathbf A^H\mathbf A$ 特征值的非负平方根。

---

## 9. 为什么称为“奇异值”

“奇异”不是指它奇怪，而是和矩阵的奇异性有关。

当某个奇异值为零：

$$
\sigma_i=0
$$

就有：

$$
\mathbf A\mathbf v_i
=
\mathbf0
$$

说明存在非零向量 $\mathbf v_i$，被矩阵映射成零。

这意味着：

$$
\mathbf A
$$

丢失了该方向的信息。

对于方阵，如果至少一个奇异值为零，则矩阵不可逆，也就是奇异矩阵。

因此可以理解为：

> 奇异值直接反映矩阵在哪些方向上接近或达到奇异状态。

但 SVD 不只适用于奇异矩阵，也适用于任意矩阵，包括：

- 可逆方阵；
- 不可逆方阵；
- 高矩阵；
- 宽矩阵；
- 实矩阵；
- 复矩阵。

---

## 10. 矩形矩阵为什么也能做 SVD

特征值分解通常要求矩阵是方阵：

$$
\mathbf A\in\mathbb C^{n\times n}
$$

但 SVD 不要求方阵。

如果：

$$
\mathbf A\in\mathbb C^{m\times n}
$$

那么：

$$
\mathbf A^H\mathbf A
\in
\mathbb C^{n\times n}
$$

$$
\mathbf A\mathbf A^H
\in
\mathbb C^{m\times m}
$$

这两个矩阵都是 Hermitian 半正定矩阵，因此可以进行稳定的特征值分解。

所以通过它们可以得到：

- 输入空间的右奇异向量；
- 输出空间的左奇异向量；
- 非负奇异值。

---

## 11. 完整 SVD 与经济型 SVD

假设：

$$
\mathbf A\in\mathbb C^{m\times n}
$$

且秩为：

$$
r
$$

### 11.1 完整 SVD

完整形式：

$$
\mathbf A
=
\mathbf U
\mathbf\Sigma
\mathbf V^H
$$

其中：

$$
\mathbf U\in\mathbb C^{m\times m}
$$

$$
\mathbf V\in\mathbb C^{n\times n}
$$

完整 SVD 会保留所有正交方向，包括对应零奇异值的方向。

这在寻找零空间时非常重要。

### 11.2 经济型 SVD

经济型 SVD 只保留必要部分，因此更节省存储。

但如果要获得完整零空间基底，经济型 SVD 可能不够。

所以：

> 如果要通过 SVD 构造完整零空间或 GSC 阻塞矩阵，应使用完整 SVD。

Python 中需要：

```python
U, s, Vh = np.linalg.svd(
    A,
    full_matrices=True
)
```

---

## 12. 秩与非零奇异值

矩阵的秩等于非零奇异值的数量：

$$
\operatorname{rank}(\mathbf A)
=
\#\left\{
\sigma_i>0
\right\}
$$

实际数值计算中，奇异值可能不会精确等于零，因此通常使用阈值判断。

例如：

```python
rank = np.linalg.matrix_rank(A)
```

或者：

```python
tolerance = (
    max(A.shape)
    * np.finfo(float).eps
    * s[0]
)

rank = np.sum(s > tolerance)
```

---

## 13. 零空间与零奇异值

矩阵的零空间定义为：

$$
\operatorname{null}(\mathbf A)
=
\left\{
\mathbf z
\mid
\mathbf A\mathbf z
=
\mathbf0
\right\}
$$

如果：

$$
\sigma_i=0
$$

那么：

$$
\mathbf A\mathbf v_i
=
\sigma_i\mathbf u_i
=
\mathbf0
$$

所以：

$$
\mathbf v_i
\in
\operatorname{null}(\mathbf A)
$$

因此：

> 与零奇异值对应的右奇异向量，构成矩阵零空间的一组正交基。

如果：

$$
\mathbf A\in\mathbb C^{m\times n}
$$

且：

$$
\operatorname{rank}(\mathbf A)=r
$$

那么零空间维数为：

$$
n-r
$$

右奇异向量：

$$
\mathbf v_{r+1},
\ldots,
\mathbf v_n
$$

构成零空间基底。

---

## 14. 秩与零度定理

对于：

$$
\mathbf A\in\mathbb C^{m\times n}
$$

有：

$$
\operatorname{rank}(\mathbf A)
+
\operatorname{nullity}(\mathbf A)
=
n
$$

注意右边是列数 $n$。

这是因为：

- 秩表示有多少个输入方向能产生独立输出；
- 零度表示有多少个输入方向被映射为零；
- 两类方向合起来覆盖完整的 $n$ 维输入空间。

---

## 15. SVD 与 GSC 阻塞矩阵

GSC 中约束矩阵为：

$$
\mathbf C\in\mathbb C^{N\times Q}
$$

约束为：

$$
\mathbf C^H\mathbf w
=
\mathbf f
$$

阻塞矩阵需要满足：

$$
\mathbf C^H\mathbf B
=
\mathbf0
$$

因此，$\mathbf B$ 的每一列都必须位于：

$$
\operatorname{null}(\mathbf C^H)
$$

也就是说：

> 阻塞矩阵的列向量，就是 $\mathbf C^H$ 零空间的一组基。

---

## 16. 为什么阻塞矩阵尺寸是 $N\times(N-Q)$

假设：

$$
\mathbf C\in\mathbb C^{N\times Q}
$$

且 $Q$ 个约束线性独立。

那么：

$$
\mathbf C^H
\in
\mathbb C^{Q\times N}
$$

并且：

$$
\operatorname{rank}(\mathbf C^H)=Q
$$

由秩与零度定理：

$$
\operatorname{nullity}(\mathbf C^H)
=
N-Q
$$

所以零空间有：

$$
N-Q
$$

个独立基向量。

每个基向量都是 $N$ 维列向量，因此把它们排成矩阵：

$$
\mathbf B
=
\begin{bmatrix}
\mathbf b_1 &
\mathbf b_2 &
\cdots &
\mathbf b_{N-Q}
\end{bmatrix}
$$

得到：

$$
\mathbf B
\in
\mathbb C^{N\times(N-Q)}
$$

可以记成：

```text
B 的行数 N：
每个阻塞方向都是 N 维权值调整向量

B 的列数 N-Q：
零空间中有 N-Q 个独立方向
```

---

## 17. 怎样通过 SVD 得到阻塞矩阵

令：

$$
\mathbf A
=
\mathbf C^H
$$

对 $\mathbf A$ 做完整 SVD：

$$
\mathbf C^H
=
\mathbf U
\mathbf\Sigma
\mathbf V^H
$$

假设：

$$
\operatorname{rank}(\mathbf C^H)=r
$$

那么：

$$
\mathbf V
=
\begin{bmatrix}
\mathbf V_r &
\mathbf V_0
\end{bmatrix}
$$

其中：

- $\mathbf V_r$ 对应非零奇异值；
- $\mathbf V_0$ 对应零奇异值。

因为：

$$
\mathbf C^H\mathbf V_0
=
\mathbf0
$$

所以可以取：

$$
\mathbf B
=
\mathbf V_0
$$

如果约束满列秩，即 $r=Q$，则：

$$
\mathbf B
=
\mathbf V[:,Q:]
$$

更稳健的写法是根据实际秩切片：

$$
\mathbf B
=
\mathbf V[:,r:]
$$

---

## 18. Python 中怎样正确构造阻塞矩阵

```python
import numpy as np


def blocking_matrix_from_svd(C):
    # 根据约束矩阵 C 构造 GSC 阻塞矩阵。

    C = np.asarray(C, dtype=complex)

    if C.ndim != 2:
        raise ValueError(
            "C 必须是二维矩阵"
        )

    A = C.conj().T

    U, s, Vh = np.linalg.svd(
        A,
        full_matrices=True
    )

    V = Vh.conj().T

    rank = np.linalg.matrix_rank(A)

    B = V[:, rank:]

    return B
```

验证：

```python
blocking_error = (
    C.conj().T
    @ B
)

print(blocking_error)
```

理论上应满足：

$$
\mathbf C^H\mathbf B
=
\mathbf0
$$

考虑浮点误差，通常检查：

```python
print(
    np.linalg.norm(
        C.conj().T @ B
    )
)
```

结果应接近零。

---

## 19. 为什么 NumPy 返回的是 `Vh`

NumPy 使用：

```python
U, s, Vh = np.linalg.svd(A)
```

这里：

```python
Vh
```

表示：

$$
\mathbf V^H
$$

而不是 $\mathbf V$。

所以需要：

```python
V = Vh.conj().T
```

复数情况下不能只写：

```python
V = Vh.T
```

因为还必须做共轭。

---

## 20. 一个 GSC 数值例子

假设：

$$
N=4
$$

只有一个约束：

$$
Q=1
$$

宽边导向矢量为：

$$
\mathbf C
=
\begin{bmatrix}
1\\
1\\
1\\
1
\end{bmatrix}
$$

则：

$$
\mathbf C^H
=
\begin{bmatrix}
1&1&1&1
\end{bmatrix}
$$

它的秩为：

$$
r=1
$$

所以零空间维数为：

$$
4-1=3
$$

因此阻塞矩阵尺寸为：

$$
\mathbf B\in\mathbb C^{4\times3}
$$

并且：

$$
\mathbf C^H\mathbf B
=
\mathbf0
$$

阻塞矩阵的每一列都表示一种权值变化方向，这种变化不会改变宽边约束。

---

## 21. 阻塞矩阵为什么不是唯一的

只要矩阵 $\mathbf B$ 的列空间等于：

$$
\operatorname{null}(\mathbf C^H)
$$

并满足：

$$
\mathbf C^H\mathbf B
=
\mathbf0
$$

它就是合法阻塞矩阵。

因此阻塞矩阵不唯一。

例如，如果 $\mathbf B$ 是一个阻塞矩阵，而 $\mathbf T$ 是任意可逆矩阵，那么：

$$
\widetilde{\mathbf B}
=
\mathbf B\mathbf T
$$

仍满足：

$$
\mathbf C^H\widetilde{\mathbf B}
=
\mathbf C^H\mathbf B\mathbf T
=
\mathbf0
$$

SVD 的优点是得到一组正交归一的零空间基：

$$
\mathbf B^H\mathbf B
=
\mathbf I
$$

这通常有利于数值稳定性和自适应算法收敛。

---

## 22. SVD、特征分解和协方差矩阵

对于 Hermitian 协方差矩阵：

$$
\mathbf R
=
\mathbf R^H
$$

并且：

$$
\mathbf R
\succeq
0
$$

它可以做特征值分解：

$$
\mathbf R
=
\mathbf Q
\mathbf\Lambda
\mathbf Q^H
$$

也可以做 SVD：

$$
\mathbf R
=
\mathbf U
\mathbf\Sigma
\mathbf V^H
$$

由于 $\mathbf R$ 是 Hermitian 半正定矩阵：

$$
\mathbf U
=
\mathbf V
=
\mathbf Q
$$

并且：

$$
\mathbf\Sigma
=
\mathbf\Lambda
$$

所以对协方差矩阵而言，SVD 与特征值分解结果基本一致。

但对于一般矩形矩阵，不能直接做普通特征值分解，此时 SVD 更通用。

---

## 23. SVD 与 PCA

设数据矩阵为：

$$
\mathbf X
\in
\mathbb C^{N\times K}
$$

对其做 SVD：

$$
\mathbf X
=
\mathbf U
\mathbf\Sigma
\mathbf V^H
$$

样本协方差矩阵为：

$$
\widehat{\mathbf R}
=
\frac{1}{K}
\mathbf X\mathbf X^H
$$

代入 SVD：

$$
\widehat{\mathbf R}
=
\frac{1}{K}
\mathbf U
\mathbf\Sigma\mathbf\Sigma^H
\mathbf U^H
$$

因此：

- $\mathbf U$ 给出数据空间中的主方向；
- $\sigma_i^2/K$ 给出各主方向上的样本功率；
- 大奇异值方向对应主要数据结构；
- 小奇异值方向通常对应弱成分或噪声。

这就是 SVD 与 PCA 的直接关系。

---

## 24. SVD 与 MUSIC

阵列快拍矩阵为：

$$
\mathbf X
=
\mathbf A_s\mathbf S
+
\mathbf N
$$

对协方差矩阵做特征分解，或者直接对数据矩阵做 SVD，都可以提取主要子空间。

若存在 $L$ 个独立信号，理想情况下：

- 前 $L$ 个大奇异值对应信号子空间；
- 剩余较小奇异值对应噪声子空间。

MUSIC 使用导向矢量与噪声子空间的正交性：

$$
\mathbf E_n^H
\mathbf a(\theta_{\mathrm{true}})
\approx
\mathbf0
$$

因此构造空间谱：

$$
P_{\mathrm{MUSIC}}(\theta)
=
\frac{
1
}{
\mathbf a^H(\theta)
\mathbf E_n
\mathbf E_n^H
\mathbf a(\theta)
}
$$

---

## 25. SVD 与伪逆

如果：

$$
\mathbf A
=
\mathbf U
\mathbf\Sigma
\mathbf V^H
$$

则 Moore-Penrose 伪逆为：

$$
\mathbf A^+
=
\mathbf V
\mathbf\Sigma^+
\mathbf U^H
$$

其中 $\mathbf\Sigma^+$ 的构造方式是：

- 对非零奇异值取倒数；
- 对零奇异值仍然保留为零；
- 再进行矩形转置。

即：

$$
\sigma_i
\longrightarrow
\frac{1}{\sigma_i}
$$

但只有：

$$
\sigma_i>0
$$

时才取倒数。

伪逆可用于：

- 超定方程最小二乘；
- 欠定方程最小范数解；
- 秩亏矩阵求解；
- 约束权值计算。

---

## 26. 条件数与数值稳定性

矩阵的二范数条件数为：

$$
\kappa_2(\mathbf A)
=
\frac{
\sigma_{\max}
}{
\sigma_{\min}
}
$$

如果最小奇异值非常小：

$$
\sigma_{\min}
\approx0
$$

则：

$$
\kappa_2(\mathbf A)
\gg1
$$

矩阵病态。

含义是：

> 输入或数据中的很小误差，经过求逆后可能被严重放大。

这与鲁棒 MVDR 中“小特征值被逆矩阵放大”的现象一致。

---

## 27. 一个简单数值例子

设：

$$
\mathbf A
=
\begin{bmatrix}
3&0\\
0&1
\end{bmatrix}
$$

那么：

$$
\mathbf U
=
\mathbf I
$$

$$
\mathbf V
=
\mathbf I
$$

$$
\mathbf\Sigma
=
\begin{bmatrix}
3&0\\
0&1
\end{bmatrix}
$$

两个奇异方向分别为：

$$
\mathbf v_1
=
\begin{bmatrix}
1\\
0
\end{bmatrix}
$$

$$
\mathbf v_2
=
\begin{bmatrix}
0\\
1
\end{bmatrix}
$$

对应映射：

$$
\mathbf A\mathbf v_1
=
3\mathbf v_1
$$

$$
\mathbf A\mathbf v_2
=
1\mathbf v_2
$$

所以水平方向被放大 3 倍，垂直方向长度不变。

对于：

$$
\mathbf x
=
\begin{bmatrix}
2\\
4
\end{bmatrix}
$$

最终：

$$
\mathbf A\mathbf x
=
\begin{bmatrix}
6\\
4
\end{bmatrix}
$$

---

## 28. 零奇异值数值例子

设：

$$
\mathbf A
=
\begin{bmatrix}
1&1\\
2&2
\end{bmatrix}
$$

第二列和第一列相同，因此：

$$
\operatorname{rank}(\mathbf A)=1
$$

存在非零向量：

$$
\mathbf v_0
=
\frac{1}{\sqrt2}
\begin{bmatrix}
1\\
-1
\end{bmatrix}
$$

满足：

$$
\mathbf A\mathbf v_0
=
\mathbf0
$$

因此 $\mathbf v_0$ 是零空间方向，并对应零奇异值。

---

## 29. 此前问题集中解答

### 29.1 为什么不直接使用 $\mathbf A\mathbf x$

直接计算没有问题。

但 SVD 能把结果解释成：

$$
\mathbf x
\longrightarrow
\mathbf V^H\mathbf x
\longrightarrow
\mathbf\Sigma\mathbf V^H\mathbf x
\longrightarrow
\mathbf U\mathbf\Sigma\mathbf V^H\mathbf x
$$

它揭示了方向、缩放、秩和零空间。

### 29.2 $\mathbf V^H$ 是不是在做投影

是。

如果 $\mathbf V$ 是酉矩阵，$\mathbf V^H\mathbf x$ 计算的是 $\mathbf x$ 在所有右奇异向量上的坐标。

每个元素都是：

$$
\mathbf v_i^H\mathbf x
$$

### 29.3 $\mathbf\Sigma$ 是不是只负责乘数字

是。

它对不同奇异方向上的坐标分别乘以对应奇异值：

$$
\alpha_i
\longrightarrow
\sigma_i\alpha_i
$$

### 29.4 $\mathbf U$ 是不是把结果旋转出去

可以这样直观理解。

更准确地说，$\mathbf U$ 把缩放后的坐标按照左奇异向量重新合成为输出向量。

### 29.5 奇异值是不是矩阵的特征值

一般不是。

正确关系为：

$$
\sigma_i^2
=
\lambda_i(\mathbf A^H\mathbf A)
$$

### 29.6 为什么矩阵是复数时使用 $H$

因为复数内积必须使用共轭转置：

$$
\mathbf x^H\mathbf y
$$

这样才能正确描述复向量的长度、正交性和投影。

### 29.7 为什么从 $\mathbf V$ 中取零空间，而不是从 $\mathbf U$ 中取

因为：

$$
\mathbf A\mathbf z
=
\mathbf0
$$

中的 $\mathbf z$ 位于 $\mathbf A$ 的输入空间。

右奇异向量 $\mathbf V$ 描述输入空间方向，所以 $\mathbf A$ 的零空间来自 $\mathbf V$。

左奇异向量 $\mathbf U$ 描述输出空间，它们与 $\mathbf A^H$ 的零空间相关。

### 29.8 为什么取 `V[:, rank:]`

因为前 `rank` 个右奇异向量对应非零奇异值。

从 `rank` 开始的剩余右奇异向量对应零奇异值，因此构成零空间基。

### 29.9 如果约束不独立怎么办

如果：

$$
\operatorname{rank}(\mathbf C)<Q
$$

不能直接认为零空间维数是：

$$
N-Q
$$

而应该使用实际秩：

$$
r
=
\operatorname{rank}(\mathbf C)
$$

于是：

$$
\dim
\operatorname{null}(\mathbf C^H)
=
N-r
$$

阻塞矩阵尺寸应为：

$$
N\times(N-r)
$$

### 29.10 阻塞矩阵是不是把干扰阻塞掉

不是。

阻塞矩阵首先阻塞的是约束子空间，也就是目标保护方向：

$$
\mathbf C^H\mathbf B
=
\mathbf0
$$

因此：

$$
\mathbf B^H\mathbf x
$$

理想情况下不含受保护目标，但仍保留干扰和噪声，供自适应支路学习。

---

## 30. SVD 的完整概念关系

```text
矩阵 A
    ↓ SVD
U Σ V^H
    ↓
V：输入空间特殊方向
Σ：每个方向的伸缩系数
U：输出空间对应方向

非零奇异值数量
    ↓
矩阵秩

零奇异值对应的右奇异向量
    ↓
A 的零空间

A = C^H
    ↓
null(C^H)
    ↓
GSC 阻塞矩阵 B

数据矩阵 X 的大奇异值方向
    ↓
主要数据结构或信号子空间

数据矩阵 X 的小奇异值方向
    ↓
弱成分或噪声子空间
```

---

## 31. 最需要记住的公式

SVD：

$$
\mathbf A
=
\mathbf U
\mathbf\Sigma
\mathbf V^H
$$

单个奇异方向：

$$
\mathbf A\mathbf v_i
=
\sigma_i\mathbf u_i
$$

一般向量映射：

$$
\mathbf A\mathbf x
=
\sum_i
\sigma_i
\left(
\mathbf v_i^H\mathbf x
\right)
\mathbf u_i
$$

奇异值与特征值：

$$
\sigma_i
=
\sqrt{
\lambda_i(
\mathbf A^H\mathbf A
)
}
$$

秩：

$$
\operatorname{rank}(\mathbf A)
=
\#\{\sigma_i>0\}
$$

零空间：

$$
\operatorname{null}(\mathbf A)
=
\operatorname{span}
\left\{
\mathbf v_i
\mid
\sigma_i=0
\right\}
$$

秩与零度：

$$
\operatorname{rank}(\mathbf A)
+
\operatorname{nullity}(\mathbf A)
=
n
$$

GSC 阻塞条件：

$$
\mathbf C^H\mathbf B
=
\mathbf0
$$

通过 SVD 构造：

$$
\mathbf C^H
=
\mathbf U
\mathbf\Sigma
\mathbf V^H
$$

$$
\mathbf B
=
\mathbf V[:,r:]
$$

其中：

$$
r
=
\operatorname{rank}(\mathbf C^H)
$$

---

## 32. 最终总结

SVD 最重要的理解不是记住三个矩阵的名称，而是看懂矩阵如何处理空间方向：

> 任意输入先分解到右奇异方向，在每个方向上乘以奇异值，再沿左奇异方向合成为输出。

其中：

- $\mathbf V$ 告诉我们输入空间有哪些关键方向；
- $\mathbf\Sigma$ 告诉我们这些方向被保留、放大、压缩还是完全消除；
- $\mathbf U$ 告诉我们这些方向最终映射到输出空间的哪里；
- 零奇异值对应的信息丢失方向；
- 这些方向正是零空间；
- 对 $\mathbf C^H$ 做完整 SVD，就可以从右奇异向量中提取 GSC 阻塞矩阵。

一句话概括：

> SVD 是把一个复杂矩阵映射，拆解成“输入方向分解、逐方向缩放、输出方向合成”，并由此揭示秩、子空间、零空间和数值稳定性。
