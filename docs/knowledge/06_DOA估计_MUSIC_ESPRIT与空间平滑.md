---
tags:
  - 阵列信号处理
  - DOA估计
  - MUSIC
  - ESPRIT
  - 空间平滑
  - 相干信号
aliases:
  - 测向算法
  - MUSIC算法
  - ESPRIT算法
created: 2026-07-19
---

# 第06章：DOA 估计——MUSIC、ESPRIT 与空间平滑

> DOA（Direction of Arrival）估计是阵列信号处理的核心任务之一。MUSIC 和 ESPRIT 是两种最经典的子空间测向算法，空间平滑则解决了相干信号导致的算法失效问题。

前置阅读：[[04_信号子空间与噪声子空间]]、[[05_信号源数量与干扰维数估计]]

---

## 1. MUSIC 空间谱估计

### 1.1 学习目标

- 为什么可以利用协方差矩阵估计信号来波方向
- 什么是信号子空间和噪声子空间
- 为什么真实导向矢量与噪声子空间正交
- MUSIC 为什么需要扫描角度
- MUSIC 谱峰为什么对应信号方向
- 多信号情况下，特征向量为什么通常不直接等于各个导向矢量

---

### 1.2 多信号阵列模型

假设阵列有 $N$ 个阵元，同时接收到 $M$ 个窄带远场信号。

$$
\mathbf{x}[k]
=
\sum_{m=1}^{M}
\mathbf{a}(\theta_m)s_m[k]
+
\mathbf{n}[k]
=
\mathbf{A}\mathbf{s}[k]
+
\mathbf{n}[k]
$$

其中：

$$
\mathbf{A}
=
\begin{bmatrix}
\mathbf{a}(\theta_1) & \mathbf{a}(\theta_2) & \cdots & \mathbf{a}(\theta_M)
\end{bmatrix}
\in \mathbb{C}^{N\times M}
$$

$$
\mathbf{s}[k]
=
\begin{bmatrix}
s_1[k] \\ s_2[k] \\ \vdots \\ s_M[k]
\end{bmatrix}
\in \mathbb{C}^{M\times 1}
$$

---

### 1.3 信号向量的物理意义

$s_m[k]$ 表示第 $m$ 个来波信号在第 $k$ 个快拍时的瞬时复包络。第 $m$ 个信号对整个阵列的贡献为：

$$
\mathbf{a}(\theta_m)s_m[k]
$$

可以理解为：

$$
\text{阵列上的瞬时信号} = \text{空间结构} \times \text{当前时刻信号值}
$$

对于 GNSS，第 $m$ 颗卫星的复包络可以粗略写成：

$$
s_m[k]
=
A_m\,
c_m[k-\tau_m]\,
d_m[k-\tau_m]\,
e^{j(2\pi f_{D,m}kT+\phi_m)}
$$

---

### 1.4 协方差矩阵

阵列数据协方差矩阵定义为：

$$
\mathbf{R}_x
=
E\left[\mathbf{x}[k]\mathbf{x}^{H}[k]\right]
$$

假设信号与噪声互不相关，噪声为空间白噪声：

$$
\boxed{
\mathbf{R}_x
=
\mathbf{A}\mathbf{R}_s\mathbf{A}^{H}
+
\sigma_n^2\mathbf{I}
}
$$

---

### 1.5 信号子空间与噪声子空间

对协方差矩阵进行特征分解，按特征值从大到小排列：

$$
\mathbf{R}_x
=
\mathbf{E}_s\mathbf{\Lambda}_s\mathbf{E}_s^{H}
+
\mathbf{E}_n\mathbf{\Lambda}_n\mathbf{E}_n^{H}
$$

- 较大的 $M$ 个特征值对应**信号子空间** $\mathbf{E}_s \in \mathbb{C}^{N\times M}$
- 剩余 $N-M$ 个较小特征值对应**噪声子空间** $\mathbf{E}_n \in \mathbb{C}^{N\times(N-M)}$

理想白噪声条件下，噪声特征值相等：$\lambda_{M+1} = \lambda_{M+2} = \cdots = \lambda_N = \sigma_n^2$

---

### 1.6 为什么导向矢量属于信号子空间

信号部分协方差矩阵为 $\mathbf{R}_{\mathrm{sig}} = \mathbf{A}\mathbf{R}_s\mathbf{A}^{H}$。

当信号协方差矩阵满秩且各导向矢量线性独立时：

$$
\boxed{
\operatorname{span}(\mathbf{E}_s)
=
\operatorname{span}(\mathbf{A})
}
$$

---

### 1.7 重要疑惑：特征向量是否直接等于导向矢量

**一般情况下不等于。**

多信号情况下，信号子空间的每个特征向量通常是多个导向矢量的线性组合，而不是某一个导向矢量本身。

正确关系是 $\operatorname{span}(\mathbf{E}_s) = \operatorname{span}(\mathbf{A})$，而不是 $\mathbf{e}_m = \mathbf{a}(\theta_m)$。

只有在单信号、白噪声和理想模型等特殊条件下，最大特征值对应的特征向量才与导向矢量成比例。

---

### 1.8 为什么真实导向矢量与噪声子空间正交

协方差矩阵是Hermitian Matrix（**共轭转置**等于它本身），不同特征值对应的特征向量相互正交。因此：

$$
\mathbf{E}_n^{H}\mathbf{E}_s = \mathbf{0}
$$

真实导向矢量属于信号子空间，所以：

$$
\boxed{
\mathbf{E}_n^{H}\mathbf{a}(\theta_m) = \mathbf{0}
}
$$

---

### 1.9 MUSIC 空间谱

候选导向矢量在噪声子空间中的投影能量为：

$$
D_{\mathrm{MUSIC}}(\theta)
=
\mathbf{a}^{H}(\theta)\mathbf{E}_n\mathbf{E}_n^{H}\mathbf{a}(\theta)
$$

真实方向处，该值接近零。定义 MUSIC 伪谱：

$$
\boxed{
P_{\mathrm{MUSIC}}(\theta)
=
\frac{1}{
\mathbf{a}^{H}(\theta)\mathbf{E}_n\mathbf{E}_n^{H}\mathbf{a}(\theta)
}
}
$$

实际绘图时常归一化为：

$$
P_{\mathrm{dB}}(\theta)
=
10\log_{10}
\left(
\frac{P_{\mathrm{MUSIC}}(\theta)}
{\max P_{\mathrm{MUSIC}}(\theta)}
\right)
$$

---

### 1.10 为什么 MUSIC 必须扫描角度

协方差矩阵的特征分解只能给出信号子空间和噪声子空间，不会直接给出角度。

MUSIC 的本质是：

> 在阵列流形中寻找最接近信号子空间的方向。

步骤：假设候选角度 → 生成候选导向矢量 → 计算其在噪声子空间中的投影 → 扫描全部候选角度 → 找出投影最小、伪谱最大的方向。

---

### 1.11 MUSIC 的基本步骤

1. 收集快拍：$\mathbf{X} = [\mathbf{x}[1], \mathbf{x}[2], \cdots, \mathbf{x}[K]]$
2. 估计协方差矩阵：$\widehat{\mathbf{R}}_x = \frac{1}{K}\mathbf{X}\mathbf{X}^{H}$
3. 特征分解：$\widehat{\mathbf{R}}_x = \widehat{\mathbf{E}}\widehat{\mathbf{\Lambda}}\widehat{\mathbf{E}}^{H}$
4. 提取噪声子空间 $\widehat{\mathbf{E}}_n$（取最小的 $N-M$ 个特征值对应的特征向量）
5. 扫描角度计算伪谱
6. 寻找谱峰 — 最明显的 $M$ 个峰值对应估计方向

---

### 1.12 MUSIC 的条件与限制

1. 阵元数大于信号数：$N > M$
2. 信号协方差矩阵满秩：$\operatorname{rank}(\mathbf{R}_s) = M$
3. 不同信号的导向矢量线性独立
4. 噪声近似为空间白噪声
5. 导向矢量模型较准确
6. 快拍数足够
7. 信号数估计正确

> 若信号数估计过小，部分信号会被放入噪声子空间；若过大，部分噪声方向会被误认为信号子空间。

---

### 1.13 MUSIC 常见疑惑

| 疑惑 | 解答 |
|------|------|
| 为什么不直接看最大特征向量？ | 多信号时特征向量是多个导向矢量的混合，不能直接读取角度 |
| MUSIC 谱是真实功率谱吗？ | 不是。是伪谱，峰值高度不直接等于信号真实功率 |
| 为什么谱峰可能很尖？ | 真实导向矢量与噪声子空间近似正交，分母接近零 |
| 为什么两个接近的方向可能分不开？ | 对应导向矢量高度相关，有限快拍/低信噪比下谱峰可能合并 |

---

## 2. ESPRIT 旋转不变子空间法

### 2.1 学习目标

- ESPRIT 为什么不需要角度扫描
- 什么是阵列平移不变性
- 为什么两个子阵的信号子空间之间存在旋转关系
- 旋转矩阵的特征值如何包含方向信息

---

### 2.2 平移不变阵列结构

考虑均匀线阵。从完整阵列中选取两个形状相同、位置相差一个阵元间距的子阵：

- 子阵 1：阵元 $1,2,\cdots,N-1$
- 子阵 2：阵元 $2,3,\cdots,N$

对于第 $m$ 个信号，相邻阵元之间的相位因子为：

$$
\phi_m = e^{-j\mu_m},\quad
\mu_m = \frac{2\pi d}{\lambda}\sin\theta_m
$$

---

### 2.3 两个子阵的导向矢量关系

$$
\boxed{
\mathbf{A}_2 = \mathbf{A}_1\mathbf{\Phi}
}
$$

其中：

$$
\mathbf{\Phi}
=
\operatorname{diag}
\left(
e^{-j\mu_1}, e^{-j\mu_2}, \cdots, e^{-j\mu_M}
\right)
$$

---

### 2.4 为什么称为旋转不变性

$e^{-j\mu_m}$ 模长为 $1$，只改变复数相位。在复平面中，乘以该复数相当于旋转相位。这里的旋转主要指**复平面相位旋转**，不是阵列在物理空间中真的旋转。

---

### 2.5 信号子空间关系

信号子空间与阵列流形张成相同空间，因此存在可逆矩阵 $\mathbf{T}$ 使得 $\mathbf{E}_s = \mathbf{A}\mathbf{T}$。

取信号子空间的前 $N-1$ 行和后 $N-1$ 行：

$$
\mathbf{E}_{s1} = \mathbf{A}_1\mathbf{T},\quad
\mathbf{E}_{s2} = \mathbf{A}_2\mathbf{T}
$$

代入 $\mathbf{A}_2 = \mathbf{A}_1\mathbf{\Phi}$，得到：

$$
\boxed{
\mathbf{E}_{s2} = \mathbf{E}_{s1}\mathbf{\Psi}
},\quad
\mathbf{\Psi} = \mathbf{T}^{-1}\mathbf{\Phi}\mathbf{T}
$$

---

### 2.6 为什么可以从特征值恢复角度

$\mathbf{\Psi} = \mathbf{T}^{-1}\mathbf{\Phi}\mathbf{T}$ 与 $\mathbf{\Phi}$ **相似**，具有相同特征值 $e^{-j\mu_m}$。

设第 $m$ 个特征值为 $\lambda_m = e^{-j\mu_m}$，则：

$$
\mu_m = -\arg(\lambda_m)
$$

再利用 $\mu_m = \frac{2\pi d}{\lambda}\sin\theta_m$，得到：

$$
\boxed{
\theta_m = \arcsin\left(\frac{\lambda}{2\pi d}\mu_m\right)
}
$$

---

### 2.7 ESPRIT 的计算步骤

1. 估计协方差矩阵
2. 提取信号子空间 $\widehat{\mathbf{E}}_s$（最大的 $M$ 个特征值对应的特征向量）
3. 构造两个子空间矩阵：$\widehat{\mathbf{E}}_{s1} = \widehat{\mathbf{E}}_s(1:N-1,:)$，$\widehat{\mathbf{E}}_{s2} = \widehat{\mathbf{E}}_s(2:N,:)$
4. 最小二乘估计：$\boxed{\widehat{\mathbf{\Psi}} = \widehat{\mathbf{E}}_{s1}^{\dagger}\widehat{\mathbf{E}}_{s2}}$
5. 求特征值并恢复空间频率：$\widehat{\mu}_m = -\arg(\widehat{\lambda}_m)$
6. 恢复角度：$\widehat{\theta}_m = \arcsin\left(\frac{\lambda}{2\pi d}\widehat{\mu}_m\right)$

---

### 2.8 MUSIC 与 ESPRIT 比较

| | MUSIC | ESPRIT |
|------|-------|--------|
| **优点** | 适用多种阵列几何；空间谱直观 | 不需角度扫描；计算速度快 |
| **缺点** | 需要扫描角度；存在网格步长误差 | 要求平移不变结构；对子阵误差敏感 |

---

### 2.9 ESPRIT 常见疑惑

| 疑惑 | 解答 |
|------|------|
| $\mathbf{\Psi}$ 是否就是物理旋转矩阵？ | 不是。$\mathbf{\Psi}$ 是信号子空间坐标系中的线性变换，与 $\mathbf{\Phi}$ 相似 |
| 为什么使用特征值而不是 $\mathbf{\Psi}$ 的矩阵元素？ | $\mathbf{\Psi}$ 受未知基变换 $\mathbf{T}$ 影响，相似变换不改变特征值 |
| 为什么复指数被称为旋转？ | $e^{j\theta} = \cos\theta + j\sin\theta$，乘以单位模复指数保持幅度、改变相位 |

---

## 3. 相干信号与空间平滑

### 3.1 学习目标

- 什么是独立信号、相关信号和完全相干信号
- 为什么相干信号会使协方差矩阵秩下降
- 为什么经典 MUSIC 和 ESPRIT 会失效
- 空间平滑如何恢复协方差矩阵秩
- 双阵元、三阵元和四阵元有什么能力差异

---

### 3.2 独立信号与相干信号

两个信号不相关时：$E[s_1[k]s_2^{*}[k]] = 0$

若两个信号完全相干（例如 $s_2[k] = \beta s_1[k]$），则：

$$
\mathbf{s}[k] = \begin{bmatrix} 1 \\ \beta \end{bmatrix} s_1[k]
$$

$$
\mathbf{R}_s = E[|s_1[k]|^2] \begin{bmatrix} 1 \\ \beta \end{bmatrix} \begin{bmatrix} 1 & \beta^{*} \end{bmatrix}
$$

因此 $\operatorname{rank}(\mathbf{R}_s) = 1$。

> 尽管物理上存在两个来波方向，统计上却只剩一个独立波形维度。

---

### 3.3 相干信号为什么导致 MUSIC 失效

信号部分协方差矩阵为：

$$
\mathbf{R}_{\mathrm{sig}}
=
\mathbf{A}\mathbf{R}_s\mathbf{A}^{H}
$$

其秩满足：

$$
\operatorname{rank}
\left(
\mathbf{R}_{\mathrm{sig}}
\right)
\le
\operatorname{rank}
\left(
\mathbf{R}_s
\right)
$$

两个信号完全相干时：

$$
\operatorname{rank}
\left(
\mathbf{R}_s
\right)
=
1
$$

信号子空间可能只有一维。

经典 MUSIC 若要分辨两个方向，需要二维信号子空间，因此会失效或出现严重偏差。

---

### 3.4 相干信号的典型来源

常见来源包括：

- 同一发射源经过多个反射路径到达阵列
- 转发式欺骗信号
- 同一干扰源经过多径传播
- 重复器或中继产生的复制信号
- 同一个基带波形从多个方向到达

> 不同真实 GNSS 卫星的 PRN 码通常不同，不会天然完全相干。但多径和转发式欺骗可能产生高度相关或相干分量。

---

### 3.5 空间平滑基本思想

将长度为 N 的均匀线阵划分为多个长度为 L 的重叠子阵。

子阵数量为：

$$
P
=
N-L+1
$$

第 p 个子阵数据记为：

$$
\mathbf{x}_p[k]
$$

其协方差矩阵为：

$$
\mathbf{R}_p
=
E\left[
\mathbf{x}_p[k]\mathbf{x}_p^{H}[k]
\right]
$$

前向空间平滑协方差矩阵定义为：

$$
\boxed{
\mathbf{R}_{\mathrm{SS}}
=
\frac{1}{P}
\sum_{p=1}^{P}
\mathbf{R}_p
}
$$

不同子阵对相干信号具有不同空间相位。

将多个子阵协方差平均后，可破坏完全相干关系，使信号协方差恢复更高的有效秩。

---

### 3.6 空间平滑的代价

空间平滑使用长度为 L 的子阵，而不是原始 N 阵元完整阵列。

代价包括：

- 有效孔径减小
- 角度分辨率下降
- 可用噪声子空间维数变化
- 可分辨信号数量受到子阵长度限制

若需要分辨 M 个信号，子阵长度至少要满足：

$$
L>M
$$

为了通过平滑恢复 M 个相干信号的秩，通常还要求：

$$
P\ge M
$$

由于：

$$
P=N-L+1
$$

经典前向空间平滑常见最低阵元数条件为：

$$
N\ge 2M
$$

所以分辨两个完全相干信号时，通常至少需要四个阵元。

### 3.7 双阵元、三阵元和四阵元能力边界

| 阵元数 | 能力 |
|:------:|------|
| $N=2$ | $N-M=0$，没有噪声子空间，经典 MUSIC 无法工作 |
| $N=3$ | 选择 $L=2$ 得两个重叠子阵，但 $L-M=0$ 仍无噪声子空间 |
| $N=4$ | 选择 $L=3$，$P=2$，$L-M=1$，保留一维噪声子空间 |

> 四阵元通常是经典前向空间平滑分辨两个完全相干信号的最低规模。

---

### 3.8 前后向空间平滑

对于中心对称阵列，前后向平均协方差矩阵为：

$$
\boxed{
\mathbf{R}_{\mathrm{FB}}
=
\frac{1}{2}
\left(
\mathbf{R}_{\mathrm{SS}}
+
\mathbf{J}\mathbf{R}_{\mathrm{SS}}^{*}\mathbf{J}
\right)
}
$$

前后向平均利用阵列共轭对称结构，可改善协方差估计，但不能突破所有维数限制。

---

### 3.9 空间平滑常见疑惑

| 疑惑 | 解答 |
|------|------|
| 为什么时间平均不能解决完全相干问题？ | 完全相干信号的固定比例关系无论多少快拍都存在 |
| 空间平滑为什么有效？ | 不同子阵位置使相干信号具有不同空间相位因子，破坏了完全线性依赖 |
| 空间平滑是否总能恢复所有信号？ | 不能。受子阵数量、长度、信噪比等条件限制 |
| 为什么空间平滑会损失分辨率？ | 最终有效阵列只有 $L$ 个阵元，有效孔径小于原始 $N$ 阵元阵列 |

---

## 下一步

完成测向后，先学习 [[07_DOA功率与SINR估计]]，把方向进一步转化为功率、噪声和 SINR 估计；随后进入 [[08_MVDR波束形成]] 设计空间权值。
