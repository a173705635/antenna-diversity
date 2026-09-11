以下内容由chatgpt生成，不涉及公司内部信息
# 第十五章 MUSIC 空间谱估计

## 15.1 学习目标

本章解决以下问题：

- 为什么可以利用协方差矩阵估计信号来波方向
- 什么是信号子空间和噪声子空间
- 为什么真实导向矢量与噪声子空间正交
- MUSIC 为什么需要扫描角度
- MUSIC 谱峰为什么对应信号方向
- 多信号情况下，特征向量为什么通常不直接等于各个导向矢量
- 阵元数、信号数和快拍数如何限制 MUSIC 的能力

---

## 15.2 多信号阵列模型

假设阵列有 N 个阵元，同时接收到 M 个窄带远场信号。

第 k 个快拍的阵列接收向量为：

$$
\mathbf{x}[k]
=
\sum_{m=1}^{M}
\mathbf{a}(\theta_m)s_m[k]
+
\mathbf{n}[k]
$$

写成矩阵形式：

$$
\mathbf{x}[k]
=
\mathbf{A}\mathbf{s}[k]
+
\mathbf{n}[k]
$$

阵列流形矩阵为：

$$
\mathbf{A}
=
\begin{bmatrix}
\mathbf{a}(\theta_1)
&
\mathbf{a}(\theta_2)
&
\cdots
&
\mathbf{a}(\theta_M)
\end{bmatrix}
$$

信号向量为：

$$
\mathbf{s}[k]
=
\begin{bmatrix}
s_1[k] \\
s_2[k] \\
\vdots \\
s_M[k]
\end{bmatrix}
$$

维度关系为：

$$
\mathbf{A}
\in
\mathbb{C}^{N\times M}
$$

$$
\mathbf{s}[k]
\in
\mathbb{C}^{M\times 1}
$$

$$
\mathbf{x}[k]
\in
\mathbb{C}^{N\times 1}
$$

---

## 15.3 信号向量的物理意义

信号向量中的第 m 个元素：

$$
s_m[k]
$$

表示第 m 个来波信号在第 k 个快拍时的瞬时复包络。

导向矢量：

$$
\mathbf{a}(\theta_m)
$$

描述该信号到达阵列后，在不同阵元之间形成的相对幅度和相位结构。

因此，第 m 个信号对整个阵列的贡献为：

$$
\mathbf{a}(\theta_m)s_m[k]
$$

可以理解为：

$$
\text{阵列上的瞬时信号}
=
\text{空间结构}
\times
\text{当前时刻信号值}
$$

对于 GNSS，第 m 颗卫星的复包络可以粗略写成：

$$
s_m[k]
=
A_m
c_m[k-\tau_m]
d_m[k-\tau_m]
e^{j(2\pi f_{D,m}kT+\phi_m)}
$$

其中可能包含：

- 信号幅度
- PRN 码
- 导航数据
- 码延迟
- 多普勒
- 初始载波相位

在窄带阵列模型中，阵元之间的空间差异由导向矢量描述，时间变化统一包含在信号复包络中。

---

## 15.4 协方差矩阵

阵列数据协方差矩阵定义为：

$$
\mathbf{R}_x
=
E\left[
\mathbf{x}[k]\mathbf{x}^{H}[k]
\right]
$$

将阵列模型代入，并假设信号与噪声互不相关：

$$
E\left[
\mathbf{s}[k]\mathbf{n}^{H}[k]
\right]
=
\mathbf{0}
$$

定义信号协方差矩阵：

$$
\mathbf{R}_s
=
E\left[
\mathbf{s}[k]\mathbf{s}^{H}[k]
\right]
$$

若噪声为空间白噪声：

$$
E\left[
\mathbf{n}[k]\mathbf{n}^{H}[k]
\right]
=
\sigma_n^2\mathbf{I}
$$

则：

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

## 15.5 信号子空间与噪声子空间

对协方差矩阵进行特征分解：

$$
\mathbf{R}_x
=
\mathbf{E}\mathbf{\Lambda}\mathbf{E}^{H}
$$

按照特征值从大到小排列后，可写成：

$$
\mathbf{R}_x
=
\mathbf{E}_s\mathbf{\Lambda}_s\mathbf{E}_s^{H}
+
\mathbf{E}_n\mathbf{\Lambda}_n\mathbf{E}_n^{H}
$$

其中：

- 较大的 M 个特征值对应信号子空间
- 剩余 N 减 M 个较小特征值对应噪声子空间

维度为：

$$
\mathbf{E}_s
\in
\mathbb{C}^{N\times M}
$$

$$
\mathbf{E}_n
\in
\mathbb{C}^{N\times(N-M)}
$$

理想白噪声条件下，噪声特征值相等：

$$
\lambda_{M+1}
=
\lambda_{M+2}
=
\cdots
=
\lambda_N
=
\sigma_n^2
$$

---

## 15.6 为什么导向矢量属于信号子空间

信号部分协方差矩阵为：

$$
\mathbf{R}_{\mathrm{sig}}
=
\mathbf{A}\mathbf{R}_s\mathbf{A}^{H}
$$

对任意向量进行作用：

$$
\mathbf{R}_{\mathrm{sig}}\mathbf{v}
=
\mathbf{A}
\left(
\mathbf{R}_s\mathbf{A}^{H}\mathbf{v}
\right)
$$

右侧一定是阵列流形矩阵各列的线性组合，因此：

$$
\operatorname{range}
\left(
\mathbf{R}_{\mathrm{sig}}
\right)
\subseteq
\operatorname{span}
\left(
\mathbf{A}
\right)
$$

当信号协方差矩阵满秩，且各导向矢量线性独立时：

$$
\operatorname{range}
\left(
\mathbf{R}_{\mathrm{sig}}
\right)
=
\operatorname{span}
\left(
\mathbf{A}
\right)
$$

信号特征向量张成信号部分协方差矩阵的值域，所以：

$$
\boxed{
\operatorname{span}
\left(
\mathbf{E}_s
\right)
=
\operatorname{span}
\left(
\mathbf{A}
\right)
}
$$

---

## 15.7 重要疑惑：特征向量是否直接等于导向矢量

一般情况下不等于。

多信号情况下，信号子空间的每个特征向量通常是多个导向矢量的线性组合，而不是某一个导向矢量本身。

正确关系是：

$$
\boxed{
\operatorname{span}
\left(
\mathbf{E}_s
\right)
=
\operatorname{span}
\left(
\mathbf{A}
\right)
}
$$

而不是：

$$
\mathbf{e}_m
=
\mathbf{a}(\theta_m)
$$

只有在单信号、白噪声和理想模型等特殊条件下，最大特征值对应的特征向量才与导向矢量成比例：

$$
\mathbf{e}_1
\propto
\mathbf{a}(\theta_1)
$$

---

## 15.8 为什么真实导向矢量与噪声子空间正交

协方差矩阵是厄米矩阵，不同特征值对应的特征向量相互正交。

因此：

$$
\mathbf{E}_n^{H}\mathbf{E}_s
=
\mathbf{0}
$$

真实导向矢量属于信号子空间，所以：

$$
\boxed{
\mathbf{E}_n^{H}\mathbf{a}(\theta_m)
=
\mathbf{0}
}
$$

于是：

$$
\mathbf{a}^{H}(\theta_m)
\mathbf{E}_n\mathbf{E}_n^{H}
\mathbf{a}(\theta_m)
=
0
$$

实际数据中，由于噪声、有限快拍和模型误差，该值不会严格为零，但通常会出现明显极小值。

---

## 15.9 MUSIC 空间谱

候选导向矢量在噪声子空间中的投影能量为：

$$
D_{\mathrm{MUSIC}}(\theta)
=
\mathbf{a}^{H}(\theta)
\mathbf{E}_n\mathbf{E}_n^{H}
\mathbf{a}(\theta)
$$

真实方向处，该值接近零。

定义 MUSIC 伪谱：

$$
\boxed{
P_{\mathrm{MUSIC}}(\theta)
=
\frac{1}{
\mathbf{a}^{H}(\theta)
\mathbf{E}_n\mathbf{E}_n^{H}
\mathbf{a}(\theta)
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

## 15.10 为什么 MUSIC 必须扫描角度

协方差矩阵的特征分解只能给出信号子空间和噪声子空间，不会直接给出角度。

角度与导向矢量之间的关系由阵列几何决定：

$$
\theta
\longrightarrow
\mathbf{a}(\theta)
$$

因此 MUSIC 需要：

1. 假设一个候选角度
2. 根据阵列几何生成候选导向矢量
3. 计算其在噪声子空间中的投影
4. 扫描全部候选角度
5. 找出投影最小、伪谱最大的方向

MUSIC 的本质是：

> 在阵列流形中寻找最接近信号子空间的方向。

---

## 15.11 MUSIC 的基本步骤

### 步骤一：收集快拍

$$
\mathbf{X}
=
\begin{bmatrix}
\mathbf{x}[1]
&
\mathbf{x}[2]
&
\cdots
&
\mathbf{x}[K]
\end{bmatrix}
$$

### 步骤二：估计协方差矩阵

$$
\widehat{\mathbf{R}}_x
=
\frac{1}{K}
\mathbf{X}\mathbf{X}^{H}
$$

### 步骤三：特征分解

$$
\widehat{\mathbf{R}}_x
=
\widehat{\mathbf{E}}
\widehat{\mathbf{\Lambda}}
\widehat{\mathbf{E}}^{H}
$$

### 步骤四：提取噪声子空间

若信号数为 M，则取最小的 N 减 M 个特征值对应的特征向量：

$$
\widehat{\mathbf{E}}_n
$$

### 步骤五：扫描角度

$$
P_{\mathrm{MUSIC}}(\theta)
=
\frac{1}{
\mathbf{a}^{H}(\theta)
\widehat{\mathbf{E}}_n
\widehat{\mathbf{E}}_n^{H}
\mathbf{a}(\theta)
}
$$

### 步骤六：寻找谱峰

空间谱中最明显的 M 个峰值对应估计方向。

---

## 15.12 MUSIC 的条件与限制

经典 MUSIC 通常要求：

1. 阵元数大于信号数

$$
N>M
$$

2. 信号协方差矩阵满秩

$$
\operatorname{rank}
\left(
\mathbf{R}_s
\right)
=M
$$

3. 不同信号的导向矢量线性独立
4. 噪声近似为空间白噪声
5. 导向矢量模型较准确
6. 快拍数足够
7. 信号数估计正确

若信号数估计过小，部分信号会被放入噪声子空间。

若信号数估计过大，部分噪声方向会被误认为信号子空间。

---

## 15.13 MUSIC 常见疑惑

### 疑惑一：为什么不直接看最大特征向量

单信号情况下，最大特征向量可能与导向矢量接近。

多信号情况下，信号特征向量通常是多个导向矢量的混合，因此不能直接读取角度。

### 疑惑二：MUSIC 谱是真实功率谱吗

不是。

MUSIC 谱是伪谱，其峰值高度不直接等于信号真实功率，主要用于判断方向位置。

### 疑惑三：为什么谱峰可能很尖

真实导向矢量与噪声子空间近似正交，使分母接近零，取倒数后形成尖峰。

### 疑惑四：为什么两个很接近的方向可能分不开

两个方向很接近时，对应导向矢量高度相关。

有限快拍、低信噪比和模型误差会使子空间估计不准确，两个谱峰可能合并。

---

# 第十六章 ESPRIT 旋转不变子空间法

## 16.1 学习目标

本章解决以下问题：

- ESPRIT 为什么不需要角度扫描
- 什么是阵列平移不变性
- 为什么两个子阵的信号子空间之间存在旋转关系
- 旋转矩阵的特征值如何包含方向信息
- ESPRIT 与 MUSIC 的优缺点有什么不同

---

## 16.2 平移不变阵列结构

考虑均匀线阵。

从完整阵列中选取两个形状相同、位置相差一个阵元间距的子阵。

第一个子阵包含阵元：

$$
1,2,\cdots,N-1
$$

第二个子阵包含阵元：

$$
2,3,\cdots,N
$$

两个子阵几何结构相同，只存在固定平移。

对于第 m 个信号，相邻阵元之间的相位因子为：

$$
\phi_m
=
e^{-j\mu_m}
$$

空间频率为：

$$
\mu_m
=
\frac{2\pi d}{\lambda}
\sin\theta_m
$$

具体使用正弦还是余弦，取决于角度定义。

---

## 16.3 两个子阵的导向矢量关系

第一个子阵导向矢量为：

$$
\mathbf{a}_1(\theta_m)
=
\begin{bmatrix}
1 \\
e^{-j\mu_m} \\
\vdots \\
e^{-j(N-2)\mu_m}
\end{bmatrix}
$$

第二个子阵导向矢量为：

$$
\mathbf{a}_2(\theta_m)
=
\begin{bmatrix}
e^{-j\mu_m} \\
e^{-j2\mu_m} \\
\vdots \\
e^{-j(N-1)\mu_m}
\end{bmatrix}
$$

因此：

$$
\mathbf{a}_2(\theta_m)
=
\mathbf{a}_1(\theta_m)e^{-j\mu_m}
$$

多个信号同时存在时：

$$
\boxed{
\mathbf{A}_2
=
\mathbf{A}_1\mathbf{\Phi}
}
$$

其中：

$$
\mathbf{\Phi}
=
\operatorname{diag}
\left(
e^{-j\mu_1},
e^{-j\mu_2},
\cdots,
e^{-j\mu_M}
\right)
$$

---

## 16.4 为什么称为旋转不变性

复数：

$$
e^{-j\mu_m}
$$

模长为 1，只改变复数相位，不改变幅度。

在复平面中，乘以该复数相当于旋转相位：

$$
z
\longrightarrow
ze^{-j\mu_m}
$$

因此，第二个子阵相对于第一个子阵，每个信号分量发生固定的复相位旋转。

这里的旋转主要指复平面相位旋转，不是阵列在物理空间中真的旋转。

---

## 16.5 信号子空间关系

信号子空间与阵列流形张成相同空间，因此存在可逆矩阵：

$$
\mathbf{T}
$$

使得：

$$
\mathbf{E}_s
=
\mathbf{A}\mathbf{T}
$$

取信号子空间前 N 减 1 行：

$$
\mathbf{E}_{s1}
=
\mathbf{A}_1\mathbf{T}
$$

取后 N 减 1 行：

$$
\mathbf{E}_{s2}
=
\mathbf{A}_2\mathbf{T}
$$

代入：

$$
\mathbf{A}_2
=
\mathbf{A}_1\mathbf{\Phi}
$$

得到：

$$
\mathbf{E}_{s2}
=
\mathbf{A}_1\mathbf{\Phi}\mathbf{T}
$$

又因为：

$$
\mathbf{E}_{s1}
=
\mathbf{A}_1\mathbf{T}
$$

所以：

$$
\mathbf{E}_{s2}
=
\mathbf{E}_{s1}
\mathbf{T}^{-1}\mathbf{\Phi}\mathbf{T}
$$

定义：

$$
\mathbf{\Psi}
=
\mathbf{T}^{-1}\mathbf{\Phi}\mathbf{T}
$$

于是：

$$
\boxed{
\mathbf{E}_{s2}
=
\mathbf{E}_{s1}\mathbf{\Psi}
}
$$

---

## 16.6 为什么可以从特征值恢复角度

矩阵：

$$
\mathbf{\Psi}
=
\mathbf{T}^{-1}\mathbf{\Phi}\mathbf{T}
$$

与矩阵 Phi 相似。

相似矩阵具有相同特征值，所以 Psi 的特征值为：

$$
e^{-j\mu_1},
e^{-j\mu_2},
\cdots,
e^{-j\mu_M}
$$

设第 m 个特征值为：

$$
\lambda_m
=
e^{-j\mu_m}
$$

则：

$$
\mu_m
=
-\arg(\lambda_m)
$$

再利用：

$$
\mu_m
=
\frac{2\pi d}{\lambda}
\sin\theta_m
$$

得到：

$$
\boxed{
\theta_m
=
\arcsin
\left(
\frac{\lambda}{2\pi d}\mu_m
\right)
}
$$

---

## 16.7 ESPRIT 的计算步骤

### 步骤一：估计协方差矩阵

$$
\widehat{\mathbf{R}}_x
=
\frac{1}{K}\mathbf{X}\mathbf{X}^{H}
$$

### 步骤二：提取信号子空间

$$
\widehat{\mathbf{E}}_s
$$

由最大的 M 个特征值对应的特征向量组成。

### 步骤三：构造两个子空间矩阵

$$
\widehat{\mathbf{E}}_{s1}
=
\widehat{\mathbf{E}}_s(1:N-1,:)
$$

$$
\widehat{\mathbf{E}}_{s2}
=
\widehat{\mathbf{E}}_s(2:N,:)
$$

### 步骤四：最小二乘估计旋转矩阵

$$
\widehat{\mathbf{E}}_{s2}
\approx
\widehat{\mathbf{E}}_{s1}\mathbf{\Psi}
$$

最小二乘解为：

$$
\boxed{
\widehat{\mathbf{\Psi}}
=
\widehat{\mathbf{E}}_{s1}^{\dagger}
\widehat{\mathbf{E}}_{s2}
}
$$

### 步骤五：求特征值并恢复空间频率

$$
\widehat{\mu}_m
=
-\arg
\left(
\widehat{\lambda}_m
\right)
$$

### 步骤六：恢复角度

$$
\widehat{\theta}_m
=
\arcsin
\left(
\frac{\lambda}{2\pi d}
\widehat{\mu}_m
\right)
$$

---

## 16.8 MUSIC 与 ESPRIT 比较

### MUSIC

优点：

- 可适用于多种阵列几何
- 空间谱直观
- 便于观察峰值结构

缺点：

- 需要扫描角度
- 计算量受扫描网格影响
- 存在网格步长误差

### ESPRIT

优点：

- 不需要角度扫描
- 直接从特征值恢复方向
- 通常计算速度较快

缺点：

- 要求阵列具有平移不变结构
- 对子阵结构和阵列误差敏感
- 角度恢复依赖明确的阵列模型

---

## 16.9 ESPRIT 常见疑惑

### 疑惑一：Psi 是否就是物理阵列旋转矩阵

不是。

Psi 是信号子空间坐标系中的线性变换。

它与包含真实相位因子的 Phi 相似，因此具有相同特征值。

### 疑惑二：为什么使用特征值而不是 Psi 的矩阵元素

Psi 受到未知基变换 T 的影响，其矩阵元素不是直接的物理相位因子。

相似变换不改变特征值，所以特征值仍保留真实空间相位信息。

### 疑惑三：为什么复指数被称为旋转

根据欧拉公式：

$$
e^{j\theta}
=
\cos\theta
+
j\sin\theta
$$

复数乘以单位模复指数，会保持幅度并改变相位。

---

# 第十七章 相干信号与空间平滑

## 17.1 学习目标

本章解决以下问题：

- 什么是独立信号、相关信号和完全相干信号
- 为什么相干信号会使协方差矩阵秩下降
- 为什么经典 MUSIC 和 ESPRIT 会失效
- 空间平滑如何恢复协方差矩阵秩
- 双阵元、三阵元和四阵元有什么能力差异

---

## 17.2 独立信号与相干信号

两个信号不相关时：

$$
E\left[
s_1[k]s_2^{*}[k]
\right]
=
0
$$

若两个信号完全相干，例如：

$$
s_2[k]
=
\beta s_1[k]
$$

则信号向量可以写成：

$$
\mathbf{s}[k]
=
\begin{bmatrix}
1 \\
\beta
\end{bmatrix}
s_1[k]
$$

信号协方差矩阵为：

$$
\mathbf{R}_s
=
E\left[
|s_1[k]|^2
\right]
\begin{bmatrix}
1 \\
\beta
\end{bmatrix}
\begin{bmatrix}
1 & \beta^{*}
\end{bmatrix}
$$

因此：

$$
\operatorname{rank}
\left(
\mathbf{R}_s
\right)
=
1
$$

尽管物理上存在两个来波方向，统计上却只剩一个独立波形维度。

---

## 17.3 相干信号为什么导致 MUSIC 失效

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

## 17.4 相干信号的典型来源

常见来源包括：

- 同一发射源经过多个反射路径到达阵列
- 转发式欺骗信号
- 同一干扰源经过多径传播
- 重复器或中继产生的复制信号
- 同一个基带波形从多个方向到达

不同真实 GNSS 卫星的 PRN 码通常不同，不会天然完全相干。

但多径和转发式欺骗可能产生高度相关或相干分量。

---

## 17.5 空间平滑基本思想

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

## 17.6 空间平滑的代价

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

---

## 17.7 双阵元、三阵元和四阵元能力边界

### 双阵元

若：

$$
N=2
$$

并希望分辨两个信号：

$$
M=2
$$

则：

$$
N-M=0
$$

没有噪声子空间，经典 MUSIC 无法工作。

### 三阵元

若：

$$
N=3
$$

选择子阵长度：

$$
L=2
$$

可以得到两个重叠子阵。

但分辨两个信号时：

$$
L-M=0
$$

仍没有噪声子空间。

前后向平均在部分理想场景可能改善结果，但三阵元不是稳定分辨两个完全相干信号的通用方案。

### 四阵元

若：

$$
N=4
$$

选择：

$$
L=3
$$

可得到：

$$
P=2
$$

个重叠子阵。

此时：

$$
L-M=1
$$

保留一维噪声子空间，并且子阵数量满足恢复两个相干信号秩的要求。

因此四阵元通常是经典前向空间平滑分辨两个完全相干信号的最低规模。

---

## 17.8 前后向空间平滑

对于中心对称阵列，可以定义反交换矩阵：

$$
\mathbf{J}
$$

前后向平均协方差矩阵为：

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

前后向平均利用阵列共轭对称结构，可改善协方差估计，并在某些相干场景中进一步改善秩恢复效果。

但它不能突破所有维数限制。

---

## 17.9 空间平滑常见疑惑

### 疑惑一：为什么时间平均不能解决完全相干问题

完全相干信号满足固定比例关系：

$$
s_2[k]
=
\beta s_1[k]
$$

无论收集多少时间快拍，这种线性依赖关系仍然存在。

### 疑惑二：空间平滑为什么有效

不同子阵位置不同，相干信号在不同子阵上具有不同空间相位因子。

对子阵协方差进行平均，相当于引入多个不同空间观测结构，从而降低信号之间的完全线性依赖。

### 疑惑三：空间平滑是否总能恢复所有信号

不能。

它受到以下条件限制：

- 阵列必须适合划分规则子阵
- 子阵数量要足够
- 子阵长度要大于信号数
- 信噪比和快拍数要足够
- 导向矢量要线性独立

### 疑惑四：为什么空间平滑会损失分辨率

最终进行 MUSIC 的有效阵列只有 L 个阵元，有效孔径小于原始 N 阵元阵列。

---

# 第十八章 MVDR 最小方差无失真响应波束形成

## 18.1 学习目标

本章解决以下问题：

- 常规波束形成为什么不能自动抑制干扰
- MVDR 的优化目标是什么
- 为什么需要无失真约束
- MVDR 权值如何推导
- 协方差矩阵为什么能够使波束零陷指向干扰
- 为什么导向矢量失配会造成目标自消除
- 对角加载有什么作用

---

## 18.2 波束形成输出

阵列接收向量为：

$$
\mathbf{x}[k]
$$

波束形成权向量为：

$$
\mathbf{w}
$$

输出为：

$$
y[k]
=
\mathbf{w}^{H}\mathbf{x}[k]
$$

输出功率为：

$$
E\left[|y[k]|^2\right]
=
E\left[
\mathbf{w}^{H}
\mathbf{x}[k]\mathbf{x}^{H}[k]
\mathbf{w}
\right]
$$

因此：

$$
\boxed{
P_{\mathrm{out}}
=
\mathbf{w}^{H}\mathbf{R}_x\mathbf{w}
}
$$

---

## 18.3 常规波束形成

若目标方向为：

$$
\theta_0
$$

对应导向矢量为：

$$
\mathbf{a}_0
=
\mathbf{a}(\theta_0)
$$

常规波束形成权值常取：

$$
\mathbf{w}_{\mathrm{CBF}}
=
\frac{\mathbf{a}_0}
{\mathbf{a}_0^{H}\mathbf{a}_0}
$$

此时：

$$
\mathbf{w}_{\mathrm{CBF}}^{H}\mathbf{a}_0
=
1
$$

常规波束形成只根据目标方向设计，不利用实际干扰协方差。

因此它会在目标方向形成主瓣，但不一定在干扰方向形成深零陷。

---

## 18.4 MVDR 优化问题

MVDR 希望输出总功率尽可能小，同时保持目标方向增益为 1。

优化问题为：

$$
\boxed{
\min_{\mathbf{w}}
\mathbf{w}^{H}\mathbf{R}_x\mathbf{w}
}
$$

约束为：

$$
\boxed{
\mathbf{w}^{H}\mathbf{a}_0
=
1
}
$$

目标函数负责抑制干扰和噪声。

约束条件负责保护目标信号。

---

## 18.5 MVDR 权值推导

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

## 18.6 MVDR 的物理理解

协方差矩阵描述不同空间方向上的能量结构。

高功率干扰方向在协方差矩阵中对应较大的能量模式。

逆协方差矩阵会相对压低这些高能量模式。

因此：

$$
\mathbf{R}_x^{-1}\mathbf{a}_0
$$

可以理解为：

> 在保留目标方向信息的同时，对高干扰能量方向进行反向加权。

最后通过分母归一化，使目标方向响应严格等于 1。

分子：根据接收数据的协方差结构，对目标方向导向矢量进行自适应调整。它会倾向于抑制协方差矩阵中能量较强的空间模式。
分母：是归一化因子，用于保证约束为1
简单理解为：分子负责抗干扰，分母保证目标不失真
---
```markdown
# 协方差矩阵为什么会放大高能量方向

## 1. 问题说明

在理解 MVDR 时，经常会看到下面的说法：

> 协方差矩阵会对高能量空间模式进行更大的缩放，而逆协方差矩阵会对高能量空间模式进行更强的抑制。

这个结论并不是额外规定的，而是可以从以下三个概念直接推导出来：

1. 向量在某个方向上的投影
2. 投影信号的平均功率
3. 协方差矩阵的特征值和特征向量

---

## 2. 阵列数据向某个方向投影

设阵列在第 k 个快拍的接收数据为：

$$
\mathbf{x}[k]
$$

选择一个单位向量：

$$
\mathbf{u}
$$

并将阵列数据投影到该方向上。

投影结果为：

$$
z[k]
=
\mathbf{u}^{H}\mathbf{x}[k]
$$

这里：

- \(\mathbf{x}[k]\) 是多维阵列数据
- \(\mathbf{u}\) 表示一个空间方向或权向量
- \(z[k]\) 是投影后得到的一个复数标量

因为：

$$
\mathbf{u}^{H}\mathbf{u}
=
1
$$

所以 \(\mathbf{u}\) 是单位向量。

---

## 3. 投影信号的平均功率

投影信号为：

$$
z[k]
=
\mathbf{u}^{H}\mathbf{x}[k]
$$

它的瞬时功率为：

$$
|z[k]|^2
$$

平均功率为：

$$
P_{\mathbf{u}}
=
E
\left[
|z[k]|^2
\right]
$$

代入投影表达式：

$$
P_{\mathbf{u}}
=
E
\left[
\left(
\mathbf{u}^{H}\mathbf{x}[k]
\right)
\left(
\mathbf{u}^{H}\mathbf{x}[k]
\right)^*
\right]
$$

由于：

$$
\left(
\mathbf{u}^{H}\mathbf{x}[k]
\right)^*
=
\mathbf{x}^{H}[k]\mathbf{u}
$$

因此：

$$
P_{\mathbf{u}}
=
E
\left[
\mathbf{u}^{H}
\mathbf{x}[k]
\mathbf{x}^{H}[k]
\mathbf{u}
\right]
$$

向量 \(\mathbf{u}\) 与随机变量无关，可以移到期望运算外面：

$$
P_{\mathbf{u}}
=
\mathbf{u}^{H}
E
\left[
\mathbf{x}[k]
\mathbf{x}^{H}[k]
\right]
\mathbf{u}
$$

阵列数据协方差矩阵定义为：

$$
\mathbf{R}
=
E
\left[
\mathbf{x}[k]
\mathbf{x}^{H}[k]
\right]
$$

因此：

$$
\boxed{
P_{\mathbf{u}}
=
\mathbf{u}^{H}
\mathbf{R}
\mathbf{u}
}
$$

这说明：

> 二次型 \(\mathbf{u}^{H}\mathbf{R}\mathbf{u}\) 表示阵列数据在方向 \(\mathbf{u}\) 上的平均功率。

---

## 4. 当投影方向是协方差矩阵的特征向量

设：

$$
\mathbf{u}_i
$$

是协方差矩阵 \(\mathbf{R}\) 的一个单位特征向量，对应特征值为：

$$
\lambda_i
$$

特征值方程为：

$$
\mathbf{R}\mathbf{u}_i
=
\lambda_i\mathbf{u}_i
$$

数据在特征向量方向 \(\mathbf{u}_i\) 上的平均功率为：

$$
P_i
=
\mathbf{u}_i^{H}
\mathbf{R}
\mathbf{u}_i
$$

利用特征值方程：

$$
\mathbf{R}\mathbf{u}_i
=
\lambda_i\mathbf{u}_i
$$

代入得到：

$$
P_i
=
\mathbf{u}_i^{H}
\lambda_i
\mathbf{u}_i
$$

特征值是标量，可以移到外面：

$$
P_i
=
\lambda_i
\mathbf{u}_i^{H}
\mathbf{u}_i
$$

由于特征向量已经归一化：

$$
\mathbf{u}_i^{H}
\mathbf{u}_i
=
1
$$

因此：

$$
\boxed{
P_i
=
\lambda_i
}
$$

这个结论非常重要：

> 协方差矩阵的特征值，等于数据在对应单位特征向量方向上的平均功率。

因此：

- 特征值越大，对应方向上的数据能量越强
- 特征值越小，对应方向上的数据能量越弱

---

## 5. 协方差矩阵如何作用于特征方向

特征值方程为：

$$
\mathbf{R}\mathbf{u}_i
=
\lambda_i\mathbf{u}_i
$$

它说明协方差矩阵作用在特征向量上时：

- 特征向量方向不变
- 特征向量长度被乘以特征值

例如：

$$
\lambda_1
=
100
$$

则：

$$
\mathbf{R}\mathbf{u}_1
=
100\mathbf{u}_1
$$

协方差矩阵将该方向放大 100 倍。

如果：

$$
\lambda_2
=
2
$$

则：

$$
\mathbf{R}\mathbf{u}_2
=
2\mathbf{u}_2
$$

该方向只被放大 2 倍。

前面已经证明：

$$
\lambda_i
=
P_i
$$

所以可以得到：

> 数据能量越大的特征方向，对应特征值越大，协方差矩阵对该方向的缩放倍数也越大。

---

## 6. 协方差矩阵如何作用于任意向量

协方差矩阵是厄米矩阵，因此它的特征向量可以构成一组正交完备基。

任意向量：

$$
\mathbf{v}
$$

都可以在这些特征向量上展开：

$$
\mathbf{v}
=
\sum_{i=1}^{N}
c_i\mathbf{u}_i
$$

其中展开系数为：

$$
c_i
=
\mathbf{u}_i^{H}
\mathbf{v}
$$

因此：

$$
\mathbf{v}
=
\sum_{i=1}^{N}
\left(
\mathbf{u}_i^{H}\mathbf{v}
\right)
\mathbf{u}_i
$$

现在使用协方差矩阵作用于该向量：

$$
\mathbf{R}\mathbf{v}
=
\mathbf{R}
\sum_{i=1}^{N}
c_i\mathbf{u}_i
$$

利用矩阵乘法的线性性质：

$$
\mathbf{R}\mathbf{v}
=
\sum_{i=1}^{N}
c_i
\mathbf{R}\mathbf{u}_i
$$

利用特征值关系：

$$
\mathbf{R}\mathbf{u}_i
=
\lambda_i\mathbf{u}_i
$$

得到：

$$
\boxed{
\mathbf{R}\mathbf{v}
=
\sum_{i=1}^{N}
c_i\lambda_i\mathbf{u}_i
}
$$

也可以写成：

$$
\boxed{
\mathbf{R}\mathbf{v}
=
\sum_{i=1}^{N}
\lambda_i
\left(
\mathbf{u}_i^{H}\mathbf{v}
\right)
\mathbf{u}_i
}
$$

这说明协方差矩阵会把任意向量拆分到各个特征方向上，然后分别乘以对应特征值。

因此：

- 高能量方向对应较大特征值，分量被放大更多
- 低能量方向对应较小特征值，分量被放大更少

---

## 7. 二维数值例子

假设协方差矩阵已经处于其特征向量坐标系中：

$$
\mathbf{R}
=
\begin{bmatrix}
100 & 0 \\
0 & 2
\end{bmatrix}
$$

这表示：

- 第一个特征方向上的平均功率为 100
- 第二个特征方向上的平均功率为 2

取一个向量：

$$
\mathbf{v}
=
\begin{bmatrix}
1 \\
1
\end{bmatrix}
$$

协方差矩阵作用在该向量上：

$$
\mathbf{R}\mathbf{v}
=
\begin{bmatrix}
100 & 0 \\
0 & 2
\end{bmatrix}
\begin{bmatrix}
1 \\
1
\end{bmatrix}
$$

得到：

$$
\mathbf{R}\mathbf{v}
=
\begin{bmatrix}
100 \\
2
\end{bmatrix}
$$

原向量在两个特征方向上的分量都是 1。

经过协方差矩阵作用后：

- 第一个方向的分量从 1 变成 100
- 第二个方向的分量从 1 变成 2

所以结果明显偏向第一个高能量方向。

---

## 8. 为什么协方差矩阵会积累出高能量方向

协方差矩阵定义为：

$$
\mathbf{R}
=
E
\left[
\mathbf{x}[k]
\mathbf{x}^{H}[k]
\right]
$$

它可以理解为大量快拍外积的平均。

对于单个快拍：

$$
\mathbf{x}[k]
\mathbf{x}^{H}[k]
$$

表示该快拍沿各个方向的幅度和相关性结构。

如果大量数据都主要沿某个方向变化，那么每次外积都会在该方向积累较大的数值。

经过大量快拍平均后，该方向就会形成一个较大的特征值。

---

## 9. 一个简单的数据分布例子

假设二维数据主要沿第一个坐标轴变化：

$$
\mathbf{x}[k]
\approx
\begin{bmatrix}
s[k] \\
0
\end{bmatrix}
$$

那么单个快拍的外积为：

$$
\mathbf{x}[k]
\mathbf{x}^{H}[k]
\approx
\begin{bmatrix}
|s[k]|^2 & 0 \\
0 & 0
\end{bmatrix}
$$

取平均后：

$$
\mathbf{R}
\approx
\begin{bmatrix}
E
\left[
|s[k]|^2
\right]
&
0
\\
0
&
0
\end{bmatrix}
$$

第一个方向的特征值为：

$$
\lambda_1
=
E
\left[
|s[k]|^2
\right]
$$

第二个方向的特征值接近：

$$
\lambda_2
=
0
$$

这说明：

- 数据主要变化的方向形成大特征值
- 数据几乎没有变化的方向形成小特征值

协方差矩阵正是通过大量数据外积的平均，把数据中的高能量方向积累成较大的特征值。

---

## 10. 强干扰为什么会形成大特征值

假设阵列接收到一个强干扰：

$$
\mathbf{x}[k]
=
\mathbf{a}_i s_i[k]
+
\mathbf{n}[k]
$$

其中：

- \(\mathbf{a}_i\) 是干扰导向矢量
- \(s_i[k]\) 是干扰信号
- \(\mathbf{n}[k]\) 是噪声

协方差矩阵为：

$$
\mathbf{R}
=
\sigma_i^2
\mathbf{a}_i
\mathbf{a}_i^{H}
+
\sigma_n^2
\mathbf{I}
$$

其中：

$$
\sigma_i^2
=
E
\left[
|s_i[k]|^2
\right]
$$

如果干扰很强：

$$
\sigma_i^2
\gg
\sigma_n^2
$$

那么干扰导向矢量所张成的空间方向会对应一个很大的特征值。

因此，强干扰会在协方差矩阵中表现为高能量特征模式。

---

## 11. 逆协方差矩阵的作用

对协方差矩阵进行特征分解：

$$
\mathbf{R}
=
\mathbf{U}
\mathbf{\Lambda}
\mathbf{U}^{H}
$$

其中：

$$
\mathbf{\Lambda}
=
\operatorname{diag}
\left(
\lambda_1,
\lambda_2,
\ldots,
\lambda_N
\right)
$$

逆协方差矩阵为：

$$
\mathbf{R}^{-1}
=
\mathbf{U}
\mathbf{\Lambda}^{-1}
\mathbf{U}^{H}
$$

其中：

$$
\mathbf{\Lambda}^{-1}
=
\operatorname{diag}
\left(
\frac{1}{\lambda_1},
\frac{1}{\lambda_2},
\ldots,
\frac{1}{\lambda_N}
\right)
$$

因此：

$$
\mathbf{R}^{-1}
\mathbf{u}_i
=
\frac{1}{\lambda_i}
\mathbf{u}_i
$$

这说明逆协方差矩阵会执行相反的缩放：

- 特征值越大，逆缩放系数越小
- 特征值越小，逆缩放系数越大

即：

$$
\boxed{
\lambda_i
\text{ 越大}
\quad
\Longrightarrow
\quad
\frac{1}{\lambda_i}
\text{ 越小}
}
$$

所以逆协方差矩阵会对高能量方向进行更强的压低。

---

## 12. 逆协方差矩阵作用于任意向量

任意向量可以展开为：

$$
\mathbf{v}
=
\sum_{i=1}^{N}
c_i\mathbf{u}_i
$$

逆协方差矩阵作用后：

$$
\mathbf{R}^{-1}\mathbf{v}
=
\sum_{i=1}^{N}
c_i
\mathbf{R}^{-1}\mathbf{u}_i
$$

代入：

$$
\mathbf{R}^{-1}\mathbf{u}_i
=
\frac{1}{\lambda_i}
\mathbf{u}_i
$$

得到：

$$
\boxed{
\mathbf{R}^{-1}\mathbf{v}
=
\sum_{i=1}^{N}
\frac{c_i}{\lambda_i}
\mathbf{u}_i
}
$$

也可以写成：

$$
\boxed{
\mathbf{R}^{-1}\mathbf{v}
=
\sum_{i=1}^{N}
\frac{
\mathbf{u}_i^{H}\mathbf{v}
}{
\lambda_i
}
\mathbf{u}_i
}
$$

因此，高能量模式对应的大特征值会出现在分母中，使该方向的分量被明显压低。

---

## 13. 与 MVDR 的联系

MVDR 权值为：

$$
\mathbf{w}_{\mathrm{MVDR}}
=
\frac{
\mathbf{R}^{-1}\mathbf{a}_0
}{
\mathbf{a}_0^{H}
\mathbf{R}^{-1}
\mathbf{a}_0
}
$$

其中：

$$
\mathbf{a}_0
$$

是目标方向导向矢量。

将目标导向矢量展开到协方差矩阵的特征向量上：

$$
\mathbf{a}_0
=
\sum_{i=1}^{N}
c_i\mathbf{u}_i
$$

其中：

$$
c_i
=
\mathbf{u}_i^{H}\mathbf{a}_0
$$

经过逆协方差矩阵加权后：

$$
\boxed{
\mathbf{R}^{-1}\mathbf{a}_0
=
\sum_{i=1}^{N}
\frac{
\mathbf{u}_i^{H}\mathbf{a}_0
}{
\lambda_i
}
\mathbf{u}_i
}
$$

因此：

- 强干扰模式的特征值很大
- 对应分量除以大特征值
- 该分量被明显压低
- 最终权向量会尽量避开强干扰空间模式

之后再通过分母进行归一化，使目标方向响应满足无失真约束。

---

## 14. 为什么还需要归一化

仅使用：

$$
\mathbf{R}^{-1}\mathbf{a}_0
$$

虽然能够降低高能量空间模式，但不能保证目标方向增益为 1。

因此需要除以：

$$
\mathbf{a}_0^{H}
\mathbf{R}^{-1}
\mathbf{a}_0
$$

最终权值为：

$$
\mathbf{w}_{\mathrm{MVDR}}
=
\frac{
\mathbf{R}^{-1}\mathbf{a}_0
}{
\mathbf{a}_0^{H}
\mathbf{R}^{-1}\mathbf{a}_0
}
$$

检查目标方向响应：

$$
\mathbf{w}_{\mathrm{MVDR}}^{H}
\mathbf{a}_0
=
1
$$

所以 MVDR 同时实现：

1. 利用逆协方差矩阵压低高能量干扰模式
2. 利用归一化约束保持目标方向单位响应

---

## 15. 为什么协方差矩阵作用后的结果会偏向高能量方向

设：

$$
\mathbf{v}
=
c_1\mathbf{u}_1
+
c_2\mathbf{u}_2
$$

对应特征值满足：

$$
\lambda_1
\gg
\lambda_2
$$

协方差矩阵作用后：

$$
\mathbf{R}\mathbf{v}
=
c_1\lambda_1\mathbf{u}_1
+
c_2\lambda_2\mathbf{u}_2
$$

即使原始系数：

$$
c_1
$$

和：

$$
c_2
$$

大小接近，由于：

$$
\lambda_1
\gg
\lambda_2
$$

处理后的第一项仍然可能远大于第二项。

所以结果方向会更接近：

$$
\mathbf{u}_1
$$

这也是反复使用协方差矩阵进行幂迭代时，可以逐渐找到最大特征值对应特征向量的原因。

---

## 16. 一个容易混淆的问题

需要区分两个不同结论。

### 结论一：特征值表示投影功率

当方向为单位特征向量时：

$$
\mathbf{u}_i^{H}
\mathbf{R}
\mathbf{u}_i
=
\lambda_i
$$

这表示数据在该方向上的平均功率为特征值。

### 结论二：协方差矩阵对特征方向进行缩放

特征值方程为：

$$
\mathbf{R}\mathbf{u}_i
=
\lambda_i\mathbf{u}_i
$$

这表示协方差矩阵对该特征方向的缩放倍数为特征值。

这两个结论中的特征值相同，不是巧合。

原因是协方差矩阵的特征方向正是数据能量分布的自然主轴。

---

## 17. 对角加载后的变化

工程中常使用对角加载协方差矩阵：

$$
\mathbf{R}_{\mathrm{L}}
=
\mathbf{R}
+
\delta\mathbf{I}
$$

若原协方差矩阵特征分解为：

$$
\mathbf{R}
=
\mathbf{U}
\mathbf{\Lambda}
\mathbf{U}^{H}
$$

则：

$$
\mathbf{R}_{\mathrm{L}}
=
\mathbf{U}
\left(
\mathbf{\Lambda}
+
\delta\mathbf{I}
\right)
\mathbf{U}^{H}
$$

加载后的特征值变为：

$$
\lambda_i+\delta
$$

逆协方差缩放系数变为：

$$
\frac{1}{\lambda_i+\delta}
$$

因此，对角加载不会改变理想情况下的特征向量方向，但会减小不同特征模式逆权重之间的差异。

这可以：

- 改善数值稳定性
- 降低协方差估计误差的影响
- 减少权值过度放大
- 提高对导向矢量失配的稳健性

但加载过大也会削弱自适应干扰抑制能力。

---

## 18. 核心推导总结

投影信号为：

$$
z[k]
=
\mathbf{u}^{H}\mathbf{x}[k]
$$

投影功率为：

$$
\boxed{
E
\left[
|z[k]|^2
\right]
=
\mathbf{u}^{H}
\mathbf{R}
\mathbf{u}
}
$$

当投影方向是单位特征向量时：

$$
\boxed{
\mathbf{u}_i^{H}
\mathbf{R}
\mathbf{u}_i
=
\lambda_i
}
$$

说明特征值就是该方向上的平均功率。

协方差矩阵对特征方向的作用为：

$$
\boxed{
\mathbf{R}\mathbf{u}_i
=
\lambda_i\mathbf{u}_i
}
$$

说明高能量方向对应大特征值，会被协方差矩阵放大更多。

逆协方差矩阵的作用为：

$$
\boxed{
\mathbf{R}^{-1}\mathbf{u}_i
=
\frac{1}{\lambda_i}
\mathbf{u}_i
}
$$

说明高能量方向会被逆协方差矩阵压低更多。

MVDR 中最关键的关系为：

$$
\boxed{
\mathbf{R}^{-1}\mathbf{a}_0
=
\sum_{i=1}^{N}
\frac{
\mathbf{u}_i^{H}\mathbf{a}_0
}{
\lambda_i
}
\mathbf{u}_i
}
$$

因此，强干扰对应的大特征值会出现在分母中，使对应空间模式受到明显抑制。

---

## 19. 最终结论

协方差矩阵之所以能够描述数据的能量方向，是因为：

$$
\mathbf{R}
=
E
\left[
\mathbf{x}\mathbf{x}^{H}
\right]
$$

它通过对大量数据外积求平均，把数据长期存在的高能量方向积累成较大的特征值。

因此：

$$
\boxed{
\text{特征值}
=
\text{对应特征方向上的平均功率}
}
$$

同时：

$$
\boxed{
\mathbf{R}
\text{ 对该方向的缩放倍数}
=
\lambda_i
}
$$

而逆协方差矩阵满足：

$$
\boxed{
\mathbf{R}^{-1}
\text{ 对该方向的缩放倍数}
=
\frac{1}{\lambda_i}
}
$$

所以：

> 协方差矩阵会突出高能量空间模式，而逆协方差矩阵会抑制高能量空间模式。这正是 MVDR 能够自适应抑制强干扰的数学基础。
```

## 18.7 为什么 MVDR 会形成零陷

设强干扰方向为：

$$
\theta_i
$$

对应导向矢量为：

$$
\mathbf{a}_i
$$

干扰协方差中包含：

$$
\sigma_i^2
\mathbf{a}_i\mathbf{a}_i^{H}
$$

当干扰功率很大时，优化问题会强烈惩罚权向量在干扰空间方向上的响应。

因此最优权值会尽量满足：

$$
\mathbf{w}^{H}\mathbf{a}_i
\approx
0
$$

也就是在干扰方向形成零陷。

该零陷不是人工指定的，而是最小化输出功率自动产生的。

---

## 18.8 样本协方差矩阵

实际系统无法直接得到精确协方差矩阵，只能通过 K 个快拍估计：

$$
\widehat{\mathbf{R}}_x
=
\frac{1}{K}
\mathbf{X}\mathbf{X}^{H}
$$

实际 MVDR 权值为：

$$
\widehat{\mathbf{w}}_{\mathrm{MVDR}}
=
\frac{
\widehat{\mathbf{R}}_x^{-1}\mathbf{a}_0
}{
\mathbf{a}_0^{H}
\widehat{\mathbf{R}}_x^{-1}\mathbf{a}_0
}
$$

代码实现中不建议显式求逆。

应使用线性方程求解：

```python
v = np.linalg.solve(R_loaded, a0)
w = v / (a0.conj().T @ v)
```

这通常比下面的写法更稳定：

```python
w = np.linalg.inv(R_loaded) @ a0
```

---

## 18.9 对角加载

有限快拍、模型误差和强相关信号可能使样本协方差矩阵病态。

对角加载后的协方差矩阵为：

$$
\boxed{
\mathbf{R}_{\mathrm{DL}}
=
\widehat{\mathbf{R}}_x
+
\delta\mathbf{I}
}
$$

其中：

$$
\delta>0
$$

对角加载的作用包括：

- 改善矩阵条件数
- 防止数值求解不稳定
- 降低对有限快拍误差的敏感性
- 降低对导向矢量失配的敏感性
- 避免权值幅度过大

加载量很大时，MVDR 会逐渐接近常规波束形成。

因此对角加载不是越大越好，而是稳健性与自适应能力之间的折中。

---

## 18.10 目标自消除问题

MVDR 使用的保护导向矢量可能为：

$$
\widehat{\mathbf{a}}_0
$$

真实目标导向矢量为：

$$
\mathbf{a}_0
$$

若：

$$
\widehat{\mathbf{a}}_0
\ne
\mathbf{a}_0
$$

MVDR 只保证：

$$
\mathbf{w}^{H}\widehat{\mathbf{a}}_0
=
1
$$

并不保证：

$$
\mathbf{w}^{H}\mathbf{a}_0
=
1
$$

如果目标信号被包含在样本协方差矩阵中，优化器可能把未被正确约束的目标成分误认为需要压低的输出能量。

最终可能在真实目标方向附近形成零陷，称为：

- 目标自消除
- 信号相消
- 自零陷

实际系统中的阵元位置误差、通道相位误差、互耦、姿态误差和天线方向图误差都可能引起导向矢量失配。

---

## 18.11 目标是否应该包含在协方差矩阵中

理论 MVDR 常直接使用总接收协方差：

$$
\mathbf{R}_x
$$

其中可能包含目标、干扰和噪声。

在目标导向矢量完全准确的理想条件下，约束可以保护目标。

实际系统中，如果导向矢量存在误差，目标被包含在协方差矩阵中会增加自消除风险。

工程上可能采用：

- 使用目标缺失时的数据估计干扰加噪声协方差
- 使用训练快拍
- 对角加载
- 导向矢量不确定集
- 多点保护约束
- 宽主瓣约束
- 后相关后为每颗卫星独立设计权值

---

## 18.12 输出 SINR 评估

目标、干扰和噪声协方差分别为：

$$
\mathbf{R}_s
$$

$$
\mathbf{R}_i
$$

$$
\mathbf{R}_n
$$

输出目标功率为：

$$
P_s
=
\mathbf{w}^{H}\mathbf{R}_s\mathbf{w}
$$

输出干扰加噪声功率为：

$$
P_{i+n}
=
\mathbf{w}^{H}
\left(
\mathbf{R}_i+\mathbf{R}_n
\right)
\mathbf{w}
$$

输出 SINR 为：

$$
\boxed{
\mathrm{SINR}_{\mathrm{out}}
=
\frac{
\mathbf{w}^{H}\mathbf{R}_s\mathbf{w}
}{
\mathbf{w}^{H}
\left(
\mathbf{R}_i+\mathbf{R}_n
\right)
\mathbf{w}
}
}
$$

仿真中单独构造理论目标、干扰和噪声协方差，通常只是为了评估输出 SINR。

这些理论分量不一定参与 MVDR 权值设计。

---

## 18.13 MVDR 常见疑惑

### 疑惑一：MVDR 是否需要提前知道干扰方向

不一定。

标准 MVDR 通过样本协方差矩阵学习高能量干扰结构，通常不需要显式输入干扰方向。

但必须知道或估计目标方向的导向矢量。

### 疑惑二：为什么强干扰方向会自动出现零陷

输出功率目标函数会对强干扰方向产生很大惩罚。

只要目标约束允许，优化器就会降低权值对干扰空间模式的响应。

### 疑惑三：MVDR 与 MUSIC 是否相同

不同。

MUSIC 用于估计来波方向。

MVDR 用于设计波束形成权值并抑制干扰。

二者都使用协方差矩阵，但目标不同。

### 疑惑四：为什么不用显式矩阵逆

求解：

$$
\mathbf{R}\mathbf{v}
=
\mathbf{a}_0
$$

即可得到：

$$
\mathbf{v}
=
\mathbf{R}^{-1}\mathbf{a}_0
$$

线性方程求解通常比显式矩阵求逆更稳定。

### 疑惑五：对角加载是不是越大越好

不是。

加载太小，系统可能不稳健。

加载太大，自适应能力减弱，零陷变浅，结果逐渐接近常规波束形成。

### 疑惑六：为什么最小输出功率能够抑制干扰
接收数据由目标信号 干扰 噪声组成，由于目标信号无法被压低，而优化器要求降低总输出功率，因此只能尽量降低干扰和噪声的功率。
当某个干扰方向很强，协方差矩阵会明显记录该方向的能量。
MVDR会自动调整权值，使该方向的阵列响应变小，甚至形成零陷。

---

# 第十九章 LCMV 线性约束最小方差波束形成

## 19.1 学习目标

本章解决以下问题：

- 为什么 MVDR 只能表达一个约束
- 如何同时保护多个方向
- 如何显式规定零陷方向
- LCMV 的矩阵约束如何构造
- LCMV 权值如何推导
- 多约束为什么会消耗阵列自由度
- GNSS 多卫星为什么难以共用一套小阵列权值

---

## 19.2 从 MVDR 到 LCMV

MVDR 只有一个无失真约束：

$$
\mathbf{w}^{H}\mathbf{a}_0
=
1
$$

如果希望同时满足多个线性约束，可以写成：

$$
\boxed{
\mathbf{C}^{H}\mathbf{w}
=
\mathbf{f}
}
$$

约束矩阵为：

$$
\mathbf{C}
=
\begin{bmatrix}
\mathbf{c}_1
&
\mathbf{c}_2
&
\cdots
&
\mathbf{c}_Q
\end{bmatrix}
$$

约束响应向量为：

$$
\mathbf{f}
=
\begin{bmatrix}
f_1 \\
f_2 \\
\vdots \\
f_Q
\end{bmatrix}
$$

---

## 19.3 LCMV 优化问题

LCMV 的优化问题为：

$$
\boxed{
\min_{\mathbf{w}}
\mathbf{w}^{H}\mathbf{R}_x\mathbf{w}
}
$$

约束为：

$$
\boxed{
\mathbf{C}^{H}\mathbf{w}
=
\mathbf{f}
}
$$

MVDR 是 LCMV 的单约束特例：

$$
\mathbf{C}
=
\mathbf{a}_0
$$

$$
\mathbf{f}
=
1
$$

---

## 19.4 LCMV 权值推导

构造拉格朗日函数：

$$
\mathcal{L}
=
\mathbf{w}^{H}\mathbf{R}_x\mathbf{w}
+
\boldsymbol{\lambda}^{H}
\left(
\mathbf{C}^{H}\mathbf{w}-\mathbf{f}
\right)
+
\left(
\mathbf{w}^{H}\mathbf{C}-\mathbf{f}^{H}
\right)
\boldsymbol{\lambda}
$$

对权向量求导并令其为零：

$$
\mathbf{R}_x\mathbf{w}
+
\mathbf{C}\boldsymbol{\lambda}
=
\mathbf{0}
$$

因此：

$$
\mathbf{w}
=
-\mathbf{R}_x^{-1}
\mathbf{C}\boldsymbol{\lambda}
$$

代入约束：

$$
\mathbf{C}^{H}\mathbf{w}
=
\mathbf{f}
$$

得到：

$$
\boldsymbol{\lambda}
=
-
\left(
\mathbf{C}^{H}\mathbf{R}_x^{-1}\mathbf{C}
\right)^{-1}
\mathbf{f}
$$

最终：

$$
\boxed{
\mathbf{w}_{\mathrm{LCMV}}
=
\mathbf{R}_x^{-1}\mathbf{C}
\left(
\mathbf{C}^{H}\mathbf{R}_x^{-1}\mathbf{C}
\right)^{-1}
\mathbf{f}
}
$$

---

## 19.5 多目标保护约束

若需要同时保护两个方向：

$$
\theta_1
$$

和：

$$
\theta_2
$$

可构造：

$$
\mathbf{C}
=
\begin{bmatrix}
\mathbf{a}(\theta_1)
&
\mathbf{a}(\theta_2)
\end{bmatrix}
$$

$$
\mathbf{f}
=
\begin{bmatrix}
1 \\
1
\end{bmatrix}
$$

则 LCMV 保证：

$$
\mathbf{w}^{H}\mathbf{a}(\theta_1)
=
1
$$

$$
\mathbf{w}^{H}\mathbf{a}(\theta_2)
=
1
$$

同时在剩余自由度中最小化输出功率。

---

## 19.6 显式零陷约束

若希望保护目标方向：

$$
\theta_0
$$

并在已知干扰方向：

$$
\theta_i
$$

形成严格零陷，可构造：

$$
\mathbf{C}
=
\begin{bmatrix}
\mathbf{a}(\theta_0)
&
\mathbf{a}(\theta_i)
\end{bmatrix}
$$

$$
\mathbf{f}
=
\begin{bmatrix}
1 \\
0
\end{bmatrix}
$$

于是：

$$
\mathbf{w}^{H}\mathbf{a}(\theta_0)
=
1
$$

$$
\mathbf{w}^{H}\mathbf{a}(\theta_i)
=
0
$$

这种方式要求干扰方向已知或已估计。

---

## 19.7 主瓣宽化与导数约束

若只约束一个精确方向，目标方向稍有误差就可能出现增益下降。

可以在目标附近增加多个保护点：

$$
\theta_0-\Delta
$$

$$
\theta_0
$$

$$
\theta_0+\Delta
$$

构造：

$$
\mathbf{C}
=
\begin{bmatrix}
\mathbf{a}(\theta_0-\Delta)
&
\mathbf{a}(\theta_0)
&
\mathbf{a}(\theta_0+\Delta)
\end{bmatrix}
$$

并令：

$$
\mathbf{f}
=
\begin{bmatrix}
1 \\
1 \\
1
\end{bmatrix}
$$

也可以使用导向矢量对角度的导数约束：

$$
\mathbf{w}^{H}
\frac{\partial\mathbf{a}(\theta)}
{\partial\theta}
\bigg|_{\theta=\theta_0}
=
0
$$

它表示目标方向附近的一阶响应变化较小，从而提高方向误差容忍度。

---

## 19.8 自由度与约束数量

N 阵元复权向量具有 N 个复数维度。

若有 Q 个线性独立约束，满足约束后的可调整自由度大约为：

$$
\boxed{
N-Q
}
$$

因此：

- 约束越多，可用于自适应抑制干扰的自由度越少
- 约束数量接近阵元数时，权值几乎被完全确定
- 独立约束数量超过阵元数时，通常无法同时满足全部约束

一个目标无失真约束时：

$$
Q=1
$$

剩余自适应自由度约为：

$$
N-1
$$

这也是常说 N 阵元阵列通常最多具有约 N 减 1 个独立零陷自由度的原因之一。

---

## 19.9 GNSS 多卫星场景中的问题

GNSS 接收机同时可见多颗卫星。

每颗卫星方向不同，对应不同导向矢量：

$$
\mathbf{a}(\theta_1),
\mathbf{a}(\theta_2),
\cdots,
\mathbf{a}(\theta_S)
$$

如果使用一套公共阵列权值，并要求所有卫星方向均保持单位增益，则：

$$
\mathbf{C}
=
\begin{bmatrix}
\mathbf{a}(\theta_1)
&
\mathbf{a}(\theta_2)
&
\cdots
&
\mathbf{a}(\theta_S)
\end{bmatrix}
$$

$$
\mathbf{f}
=
\mathbf{1}
$$

当卫星数量大于或接近阵元数时，小阵列无法严格满足全部独立约束，同时也几乎没有剩余自由度抑制干扰。

更实际的方案包括：

- 每颗卫星使用独立权值
- 后相关波束形成
- 只保护少数关键卫星或方向区域
- 使用宽主瓣或不确定集约束
- 在阵列自由度和保护目标之间折中

---

## 19.10 数值实现注意事项

LCMV 公式中包含逆矩阵。

代码中应使用线性方程求解。

先计算：

$$
\mathbf{V}
=
\mathbf{R}^{-1}\mathbf{C}
$$

代码：

```python
V = np.linalg.solve(R_loaded, C)
```

再计算：

$$
\mathbf{G}
=
\mathbf{C}^{H}\mathbf{V}
$$

求解：

$$
\mathbf{G}\mathbf{q}
=
\mathbf{f}
$$

最后：

$$
\mathbf{w}
=
\mathbf{V}\mathbf{q}
$$

代码：

```python
V = np.linalg.solve(R_loaded, C)
q = np.linalg.solve(C.conj().T @ V, f)
w = V @ q
```

还需要检查：

- 约束矩阵是否列满秩
- 约束方向是否过于接近
- 约束数量是否超过阵元能力
- 协方差矩阵条件数是否过大
- 是否需要对角加载

---

## 19.11 LCMV 常见疑惑

### 疑惑一：LCMV 是否必须知道干扰方向

不一定。

若只给出保护约束，LCMV 可以利用协方差自动抑制其他高能量方向。

只有设置显式零陷约束时，才需要输入对应干扰方向。

### 疑惑二：LCMV 与 MVDR 的区别是什么

MVDR 只有一个保护约束。

LCMV 可以设置多个线性约束。

MVDR 是 LCMV 的特例。

### 疑惑三：约束越多是否越安全

不一定。

更多约束可以保护更多方向，但会消耗更多自由度，可能使干扰抑制能力下降，也可能使矩阵求解变得不稳定。

### 疑惑四：四阵元能否保护四颗卫星并同时抑制干扰

若四个保护约束线性独立：

$$
N=4
$$

$$
Q=4
$$

剩余自由度约为零。

权值基本被约束完全确定，几乎没有额外自由度自适应抑制未知干扰。

### 疑惑五：多个保护方向很接近会怎样

对应导向矢量可能高度相关，使约束矩阵条件数变差。

小误差会导致很大的权值变化，需要使用加载、约束合并或宽主瓣设计。

---

# 第二十章 本阶段概念关系总结

## 20.1 从测向到抗干扰的逻辑链

```text
阵列接收数据
    |
    v
样本协方差矩阵
    |
    +-----------------------------+
    |                             |
    v                             v
特征分解与子空间              最小输出功率优化
    |                             |
    v                             v
MUSIC / ESPRIT              MVDR / LCMV
    |                             |
    v                             v
估计来波方向                 形成波束和零陷
```

MUSIC 和 ESPRIT 主要回答：

> 信号从什么方向来。

MVDR 和 LCMV 主要回答：

> 已知需要保护的方向后，怎样组合阵元信号以抑制干扰。

---

## 20.2 MUSIC 与 ESPRIT 的共同基础

二者都依赖：

- 多信号阵列模型
- 样本协方差矩阵
- 特征值分解
- 信号子空间
- 阵列流形结构
- 信号数估计

MUSIC 利用：

$$
\mathbf{E}_n^{H}\mathbf{a}(\theta)
\approx
\mathbf{0}
$$

ESPRIT 利用：

$$
\mathbf{E}_{s2}
\approx
\mathbf{E}_{s1}\mathbf{\Psi}
$$

MUSIC 使用噪声子空间正交性。

ESPRIT 使用信号子空间平移不变性。

---

## 20.3 相干信号位于测向算法与实际传播之间

经典 MUSIC 和 ESPRIT 默认信号协方差矩阵满秩。

多径和转发复制会使信号相干：

$$
\operatorname{rank}
\left(
\mathbf{R}_s
\right)
<M
$$

这会导致信号子空间维数小于物理来波数量。

空间平滑通过重叠子阵恢复有效秩，但需要牺牲孔径和阵列自由度。

---

## 20.4 MVDR 与 LCMV 的共同基础

二者都最小化输出功率：

$$
\mathbf{w}^{H}\mathbf{R}\mathbf{w}
$$

区别在于约束。

MVDR：

$$
\mathbf{w}^{H}\mathbf{a}_0
=
1
$$

LCMV：

$$
\mathbf{C}^{H}\mathbf{w}
=
\mathbf{f}
$$

因此：

$$
\boxed{
\text{MVDR}
\subset
\text{LCMV}
}
$$

---

## 20.5 测向与波束形成的协作关系

典型流程可以是：

1. 使用 MUSIC 或 ESPRIT 估计干扰方向
2. 根据卫星星历、姿态和阵列模型得到卫星导向矢量
3. 使用 MVDR 或 LCMV 设计权值
4. 对多阵元数据进行加权合并
5. 评估输出 SINR、零陷深度和卫星信号损失

但标准 MVDR 不一定需要先显式估计干扰方向。

它可以直接从协方差矩阵学习干扰空间。

MUSIC 或 ESPRIT 更适合需要显示干扰方向、进行态势感知或设置显式零陷约束的场景。

---

## 20.6 本阶段最关键的能力边界

### 阵元数必须大于信号数

MUSIC 需要噪声子空间：

$$
N-M>0
$$

### 相干信号会降低有效信号秩

即使物理上存在多个方向，统计上也可能只有一个独立信号维度。

### 空间平滑需要额外阵元

经典前向空间平滑分辨 M 个完全相干信号时，常见最低条件为：

$$
N\ge 2M
$$

### 多个约束会消耗波束形成自由度

剩余自适应自由度约为：

$$
N-Q
$$

### 导向矢量误差可能导致自消除

工程系统中的阵元位置误差、通道相位误差、互耦和姿态误差都会影响结果。

---

## 20.7 本阶段疑惑点汇总

### 疑惑一：多个信号时，特征向量是否一一对应导向矢量

通常不对应。

正确关系是信号子空间与阵列流形空间相同。

### 疑惑二：MUSIC 为什么要扫描

特征分解给出子空间，不直接给角度。

需要将候选角度映射为导向矢量，再检测其与噪声子空间的正交性。

### 疑惑三：ESPRIT 中的旋转是什么

是两个平移子阵之间的复相位变换。

Psi 不是直接的物理旋转矩阵，但其特征值保留真实空间相位。

### 疑惑四：为什么相干信号导致秩下降

多个信号波形之间存在固定线性关系，无法形成多个独立统计维度。

### 疑惑五：两个相干信号为什么通常至少需要四阵元

空间平滑既需要足够长的子阵保留噪声子空间，也需要足够多的重叠子阵恢复信号秩。

### 疑惑六：MVDR 为什么能自动形成零陷

强干扰在协方差矩阵中贡献较大。

最小输出功率优化会降低权值对这些空间模式的响应。

### 疑惑七：MVDR 为什么可能把目标压掉

约束保护的是假设导向矢量。

若真实导向矢量与假设不一致，目标能量可能被优化器当成需要抑制的输出功率。

### 疑惑八：仿真中为什么还要构造理论目标和干扰协方差

这些矩阵可用于独立计算输出目标功率、干扰功率和 SINR。

它们不一定参与 MVDR 权值设计。

### 疑惑九：LCMV 为什么不能无限增加保护方向

每个独立约束都会占用一个自由度。

约束过多后，阵列没有足够自由度自适应抑制未知干扰。

### 疑惑十：投影公式为什么是加号

方向单位向量为：

$$
\mathbf{u}
=
\begin{bmatrix}
\cos\theta \\
\sin\theta
\end{bmatrix}
$$

二维向量为：

$$
\mathbf{x}
=
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
$$

标量投影为内积：

$$
z
=
\mathbf{u}^{T}\mathbf{x}
$$

所以：

$$
\boxed{
z
=
x_1\cos\theta
+
x_2\sin\theta
}
$$

这与主动旋转向量不是同一个操作。

---

## 20.8 下一阶段衔接

下一阶段适合继续学习：

1. GSC 广义旁瓣相消器
2. LMS 和 NLMS 自适应更新
3. RLS 和递推协方差估计
4. 宽带阵列模型
5. 空时自适应处理
6. GNSS 前相关与后相关抗干扰
7. 阵列标定
8. 稳健波束形成

其中 GSC 是 LCMV 的等价实现结构，可以把抽象约束优化拆分成固定保护支路、阻塞矩阵和自适应相消支路。

---

# 附录 A MUSIC 仿真代码

下面代码模拟八阵元均匀线阵接收两个独立窄带信号，并使用 MUSIC 估计来波方向。

```python
import numpy as np
import matplotlib.pyplot as plt


def steering_vector(
    angle_deg: float,
    num_elements: int,
    spacing_wavelength: float = 0.5,
) -> np.ndarray:
    """
    均匀线阵导向矢量。

    角度相对于阵列法线定义。
    阵元间距使用波长作为单位。
    """
    theta = np.deg2rad(angle_deg)
    element_index = np.arange(num_elements)

    return np.exp(
        -1j
        * 2.0
        * np.pi
        * spacing_wavelength
        * element_index
        * np.sin(theta)
    )


def estimate_music_spectrum(
    x: np.ndarray,
    num_sources: int,
    scan_angles_deg: np.ndarray,
    spacing_wavelength: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    根据阵列快拍数据计算 MUSIC 归一化空间谱。

    x 的形状为：阵元数乘快拍数。
    """
    num_elements, num_snapshots = x.shape

    covariance = x @ x.conj().T / num_snapshots

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    noise_subspace = eigenvectors[:, : num_elements - num_sources]

    spectrum = np.zeros(scan_angles_deg.size, dtype=float)

    for index, angle_deg in enumerate(scan_angles_deg):
        a = steering_vector(
            angle_deg,
            num_elements,
            spacing_wavelength,
        )

        denominator = (
            a.conj().T
            @ noise_subspace
            @ noise_subspace.conj().T
            @ a
        )

        spectrum[index] = 1.0 / max(
            np.abs(denominator),
            1e-12,
        )

    spectrum_db = 10.0 * np.log10(
        spectrum / np.max(spectrum)
    )

    return spectrum_db, eigenvalues


def main() -> None:
    rng = np.random.default_rng(2026)

    num_elements = 8
    num_snapshots = 2000
    spacing_wavelength = 0.5

    true_angles_deg = np.array([-20.0, 30.0])
    num_sources = true_angles_deg.size

    source_power = np.array([1.0, 0.8])
    noise_power = 0.1

    source_signals = (
        rng.standard_normal((num_sources, num_snapshots))
        + 1j
        * rng.standard_normal((num_sources, num_snapshots))
    ) / np.sqrt(2.0)

    source_signals *= np.sqrt(source_power)[:, None]

    array_manifold = np.column_stack(
        [
            steering_vector(
                angle,
                num_elements,
                spacing_wavelength,
            )
            for angle in true_angles_deg
        ]
    )

    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal((num_elements, num_snapshots))
        + 1j
        * rng.standard_normal((num_elements, num_snapshots))
    )

    x = array_manifold @ source_signals + noise

    scan_angles_deg = np.linspace(-90.0, 90.0, 1801)

    spectrum_db, eigenvalues = estimate_music_spectrum(
        x,
        num_sources,
        scan_angles_deg,
        spacing_wavelength,
    )

    print("按从小到大排列的协方差特征值：")
    print(eigenvalues)

    plt.figure()
    plt.plot(scan_angles_deg, spectrum_db)

    for angle in true_angles_deg:
        plt.axvline(angle, linestyle="--")

    plt.xlabel("角度 / 度")
    plt.ylabel("归一化 MUSIC 谱 / dB")
    plt.title("MUSIC 来波方向估计")
    plt.grid(True)
    plt.ylim(-60.0, 2.0)

    plt.show()


if __name__ == "__main__":
    main()
```

---

# 附录 B MUSIC 与 ESPRIT 对比仿真代码

下面代码使用同一组阵列数据，分别通过 MUSIC 和 ESPRIT 估计方向。

```python
import numpy as np
import matplotlib.pyplot as plt


def steering_vector(
    angle_deg: float,
    num_elements: int,
    spacing_wavelength: float = 0.5,
) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    element_index = np.arange(num_elements)

    return np.exp(
        -1j
        * 2.0
        * np.pi
        * spacing_wavelength
        * element_index
        * np.sin(theta)
    )


def covariance_matrix(x: np.ndarray) -> np.ndarray:
    num_snapshots = x.shape[1]
    return x @ x.conj().T / num_snapshots


def signal_subspace(
    covariance: np.ndarray,
    num_sources: int,
) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    return eigenvectors[:, :num_sources], eigenvalues


def music_spectrum(
    covariance: np.ndarray,
    num_sources: int,
    scan_angles_deg: np.ndarray,
    spacing_wavelength: float,
) -> np.ndarray:
    num_elements = covariance.shape[0]

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, order]

    noise_subspace = eigenvectors[:, : num_elements - num_sources]

    spectrum = []

    for angle_deg in scan_angles_deg:
        a = steering_vector(
            angle_deg,
            num_elements,
            spacing_wavelength,
        )

        denominator = (
            a.conj().T
            @ noise_subspace
            @ noise_subspace.conj().T
            @ a
        )

        spectrum.append(
            1.0 / max(np.abs(denominator), 1e-12)
        )

    spectrum = np.asarray(spectrum)

    return 10.0 * np.log10(
        spectrum / np.max(spectrum)
    )


def esprit_angles(
    covariance: np.ndarray,
    num_sources: int,
    spacing_wavelength: float,
) -> np.ndarray:
    es, _ = signal_subspace(
        covariance,
        num_sources,
    )

    es1 = es[:-1, :]
    es2 = es[1:, :]

    psi = np.linalg.pinv(es1) @ es2
    rotation_eigenvalues = np.linalg.eigvals(psi)

    spatial_frequency = -np.angle(
        rotation_eigenvalues
    )

    sin_theta = spatial_frequency / (
        2.0
        * np.pi
        * spacing_wavelength
    )

    sin_theta = np.clip(
        sin_theta.real,
        -1.0,
        1.0,
    )

    return np.sort(
        np.rad2deg(np.arcsin(sin_theta))
    )


def main() -> None:
    rng = np.random.default_rng(2026)

    num_elements = 8
    num_snapshots = 3000
    spacing_wavelength = 0.5

    true_angles_deg = np.array([-25.0, 18.0])
    num_sources = true_angles_deg.size

    source_power = np.array([1.0, 0.7])
    noise_power = 0.05

    source_signals = (
        rng.standard_normal((num_sources, num_snapshots))
        + 1j
        * rng.standard_normal((num_sources, num_snapshots))
    ) / np.sqrt(2.0)

    source_signals *= np.sqrt(source_power)[:, None]

    manifold = np.column_stack(
        [
            steering_vector(
                angle,
                num_elements,
                spacing_wavelength,
            )
            for angle in true_angles_deg
        ]
    )

    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal((num_elements, num_snapshots))
        + 1j
        * rng.standard_normal((num_elements, num_snapshots))
    )

    x = manifold @ source_signals + noise
    covariance = covariance_matrix(x)

    estimated_esprit_deg = esprit_angles(
        covariance,
        num_sources,
        spacing_wavelength,
    )

    print("真实角度：")
    print(np.sort(true_angles_deg))

    print("ESPRIT 估计角度：")
    print(estimated_esprit_deg)

    scan_angles_deg = np.linspace(-90.0, 90.0, 1801)

    spectrum_db = music_spectrum(
        covariance,
        num_sources,
        scan_angles_deg,
        spacing_wavelength,
    )

    plt.figure()
    plt.plot(scan_angles_deg, spectrum_db)

    for angle in true_angles_deg:
        plt.axvline(angle, linestyle="--")

    for angle in estimated_esprit_deg:
        plt.axvline(angle, linestyle=":")

    plt.xlabel("角度 / 度")
    plt.ylabel("归一化 MUSIC 谱 / dB")
    plt.title("MUSIC 与 ESPRIT 结果对比")
    plt.grid(True)
    plt.ylim(-60.0, 2.0)

    plt.show()


if __name__ == "__main__":
    main()
```

---

# 附录 C 相干信号与空间平滑仿真代码

下面代码构造两个完全相干信号，比较未平滑 MUSIC 和前向空间平滑 MUSIC。

```python
import numpy as np
import matplotlib.pyplot as plt


def steering_vector(
    angle_deg: float,
    num_elements: int,
    spacing_wavelength: float = 0.5,
) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    element_index = np.arange(num_elements)

    return np.exp(
        -1j
        * 2.0
        * np.pi
        * spacing_wavelength
        * element_index
        * np.sin(theta)
    )


def sample_covariance(x: np.ndarray) -> np.ndarray:
    return x @ x.conj().T / x.shape[1]


def forward_spatial_smoothing(
    x: np.ndarray,
    subarray_length: int,
) -> np.ndarray:
    num_elements, num_snapshots = x.shape

    if not 1 <= subarray_length <= num_elements:
        raise ValueError("子阵长度必须位于 1 到阵元数之间。")

    num_subarrays = num_elements - subarray_length + 1

    smoothed_covariance = np.zeros(
        (subarray_length, subarray_length),
        dtype=complex,
    )

    for start in range(num_subarrays):
        subarray_data = x[
            start : start + subarray_length,
            :,
        ]

        smoothed_covariance += (
            subarray_data
            @ subarray_data.conj().T
            / num_snapshots
        )

    return smoothed_covariance / num_subarrays


def music_spectrum_from_covariance(
    covariance: np.ndarray,
    num_sources: int,
    scan_angles_deg: np.ndarray,
    spacing_wavelength: float,
) -> np.ndarray:
    num_elements = covariance.shape[0]

    if num_sources >= num_elements:
        raise ValueError(
            "MUSIC 要求信号数小于有效阵元数。"
        )

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, order]

    noise_subspace = eigenvectors[
        :,
        : num_elements - num_sources,
    ]

    spectrum = np.zeros(
        scan_angles_deg.size,
        dtype=float,
    )

    for index, angle_deg in enumerate(scan_angles_deg):
        a = steering_vector(
            angle_deg,
            num_elements,
            spacing_wavelength,
        )

        denominator = (
            a.conj().T
            @ noise_subspace
            @ noise_subspace.conj().T
            @ a
        )

        spectrum[index] = 1.0 / max(
            np.abs(denominator),
            1e-12,
        )

    return 10.0 * np.log10(
        spectrum / np.max(spectrum)
    )


def main() -> None:
    rng = np.random.default_rng(2026)

    num_elements = 8
    subarray_length = 6
    num_snapshots = 4000
    spacing_wavelength = 0.5

    true_angles_deg = np.array([-18.0, 24.0])
    num_sources = true_angles_deg.size

    noise_power = 0.02

    common_waveform = (
        rng.standard_normal(num_snapshots)
        + 1j * rng.standard_normal(num_snapshots)
    ) / np.sqrt(2.0)

    source_signals = np.vstack(
        [
            common_waveform,
            0.8
            * np.exp(1j * 0.7)
            * common_waveform,
        ]
    )

    manifold = np.column_stack(
        [
            steering_vector(
                angle,
                num_elements,
                spacing_wavelength,
            )
            for angle in true_angles_deg
        ]
    )

    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal((num_elements, num_snapshots))
        + 1j
        * rng.standard_normal((num_elements, num_snapshots))
    )

    x = manifold @ source_signals + noise

    covariance_full = sample_covariance(x)

    covariance_smoothed = forward_spatial_smoothing(
        x,
        subarray_length,
    )

    scan_angles_deg = np.linspace(-90.0, 90.0, 1801)

    spectrum_full_db = music_spectrum_from_covariance(
        covariance_full,
        num_sources,
        scan_angles_deg,
        spacing_wavelength,
    )

    spectrum_smoothed_db = music_spectrum_from_covariance(
        covariance_smoothed,
        num_sources,
        scan_angles_deg,
        spacing_wavelength,
    )

    plt.figure()
    plt.plot(
        scan_angles_deg,
        spectrum_full_db,
        label="未做空间平滑",
    )

    plt.plot(
        scan_angles_deg,
        spectrum_smoothed_db,
        label="前向空间平滑",
    )

    for angle in true_angles_deg:
        plt.axvline(angle, linestyle="--")

    plt.xlabel("角度 / 度")
    plt.ylabel("归一化 MUSIC 谱 / dB")
    plt.title("相干信号场景下的空间平滑")
    plt.grid(True)
    plt.legend()
    plt.ylim(-60.0, 2.0)

    plt.show()


if __name__ == "__main__":
    main()
```

---

# 附录 D CBF 与 MVDR 抗干扰仿真代码

下面代码比较常规波束形成和 MVDR 的方向响应及输出 SINR。

```python
import numpy as np
import matplotlib.pyplot as plt


def steering_vector(
    angle_deg: float,
    num_elements: int,
    spacing_wavelength: float = 0.5,
) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    element_index = np.arange(num_elements)

    return np.exp(
        -1j
        * 2.0
        * np.pi
        * spacing_wavelength
        * element_index
        * np.sin(theta)
    )


def db(value: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(
        np.maximum(value, 1e-15)
    )


def beam_response(
    weights: np.ndarray,
    scan_angles_deg: np.ndarray,
    num_elements: int,
    spacing_wavelength: float,
) -> np.ndarray:
    response = np.zeros(
        scan_angles_deg.size,
        dtype=float,
    )

    for index, angle_deg in enumerate(scan_angles_deg):
        a = steering_vector(
            angle_deg,
            num_elements,
            spacing_wavelength,
        )

        response[index] = np.abs(
            weights.conj().T @ a
        ) ** 2

    response /= np.max(response)
    return db(response)


def output_sinr(
    weights: np.ndarray,
    target_covariance: np.ndarray,
    interference_covariance: np.ndarray,
    noise_covariance: np.ndarray,
) -> float:
    target_power = np.real(
        weights.conj().T
        @ target_covariance
        @ weights
    )

    interference_noise_power = np.real(
        weights.conj().T
        @ (
            interference_covariance
            + noise_covariance
        )
        @ weights
    )

    return float(
        target_power / interference_noise_power
    )


def main() -> None:
    rng = np.random.default_rng(2026)

    num_elements = 8
    num_snapshots = 4000
    spacing_wavelength = 0.5

    target_angle_deg = 10.0
    interference_angles_deg = np.array([-35.0, 42.0])

    target_power = 1.0
    interference_powers = np.array([100.0, 50.0])
    noise_power = 1.0

    a_target = steering_vector(
        target_angle_deg,
        num_elements,
        spacing_wavelength,
    )

    interference_manifold = np.column_stack(
        [
            steering_vector(
                angle,
                num_elements,
                spacing_wavelength,
            )
            for angle in interference_angles_deg
        ]
    )

    target_signal = np.sqrt(target_power / 2.0) * (
        rng.standard_normal(num_snapshots)
        + 1j * rng.standard_normal(num_snapshots)
    )

    interference_signals = (
        rng.standard_normal(
            (
                interference_angles_deg.size,
                num_snapshots,
            )
        )
        + 1j
        * rng.standard_normal(
            (
                interference_angles_deg.size,
                num_snapshots,
            )
        )
    ) / np.sqrt(2.0)

    interference_signals *= np.sqrt(
        interference_powers
    )[:, None]

    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(
            (num_elements, num_snapshots)
        )
        + 1j
        * rng.standard_normal(
            (num_elements, num_snapshots)
        )
    )

    x = (
        a_target[:, None] * target_signal[None, :]
        + interference_manifold @ interference_signals
        + noise
    )

    sample_covariance = (
        x @ x.conj().T / num_snapshots
    )

    loading_factor = 1e-2
    loading = (
        loading_factor
        * np.trace(sample_covariance).real
        / num_elements
    )

    loaded_covariance = (
        sample_covariance
        + loading * np.eye(num_elements)
    )

    w_cbf = a_target / (
        a_target.conj().T @ a_target
    )

    v = np.linalg.solve(
        loaded_covariance,
        a_target,
    )

    w_mvdr = v / (
        a_target.conj().T @ v
    )

    target_covariance = (
        target_power
        * np.outer(
            a_target,
            a_target.conj(),
        )
    )

    interference_covariance = np.zeros(
        (num_elements, num_elements),
        dtype=complex,
    )

    for power, angle_deg in zip(
        interference_powers,
        interference_angles_deg,
    ):
        a_interference = steering_vector(
            angle_deg,
            num_elements,
            spacing_wavelength,
        )

        interference_covariance += (
            power
            * np.outer(
                a_interference,
                a_interference.conj(),
            )
        )

    noise_covariance = (
        noise_power * np.eye(num_elements)
    )

    sinr_cbf = output_sinr(
        w_cbf,
        target_covariance,
        interference_covariance,
        noise_covariance,
    )

    sinr_mvdr = output_sinr(
        w_mvdr,
        target_covariance,
        interference_covariance,
        noise_covariance,
    )

    print(
        "CBF 输出 SINR / dB：",
        10.0 * np.log10(sinr_cbf),
    )

    print(
        "MVDR 输出 SINR / dB：",
        10.0 * np.log10(sinr_mvdr),
    )

    scan_angles_deg = np.linspace(
        -90.0,
        90.0,
        1801,
    )

    response_cbf_db = beam_response(
        w_cbf,
        scan_angles_deg,
        num_elements,
        spacing_wavelength,
    )

    response_mvdr_db = beam_response(
        w_mvdr,
        scan_angles_deg,
        num_elements,
        spacing_wavelength,
    )

    plt.figure()
    plt.plot(
        scan_angles_deg,
        response_cbf_db,
        label="CBF",
    )

    plt.plot(
        scan_angles_deg,
        response_mvdr_db,
        label="MVDR",
    )

    plt.axvline(
        target_angle_deg,
        linestyle="--",
        label="目标方向",
    )

    for angle in interference_angles_deg:
        plt.axvline(
            angle,
            linestyle=":",
        )

    plt.xlabel("角度 / 度")
    plt.ylabel("归一化方向响应 / dB")
    plt.title("CBF 与 MVDR 方向响应")
    plt.grid(True)
    plt.legend()
    plt.ylim(-80.0, 5.0)

    plt.show()


if __name__ == "__main__":
    main()
```

---

# 附录 E LCMV 多约束波束形成仿真代码

下面代码比较单目标 MVDR 和多约束 LCMV。

LCMV 同时保护两个方向，并对一个已知干扰方向施加显式零陷约束。

```python
import numpy as np
import matplotlib.pyplot as plt


def steering_vector(
    angle_deg: float,
    num_elements: int,
    spacing_wavelength: float = 0.5,
) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    element_index = np.arange(num_elements)

    return np.exp(
        -1j
        * 2.0
        * np.pi
        * spacing_wavelength
        * element_index
        * np.sin(theta)
    )


def calculate_mvdr_weights(
    covariance: np.ndarray,
    desired_vector: np.ndarray,
) -> np.ndarray:
    v = np.linalg.solve(
        covariance,
        desired_vector,
    )

    return v / (
        desired_vector.conj().T @ v
    )


def calculate_lcmv_weights(
    covariance: np.ndarray,
    constraint_matrix: np.ndarray,
    response_vector: np.ndarray,
) -> np.ndarray:
    v = np.linalg.solve(
        covariance,
        constraint_matrix,
    )

    gram_matrix = (
        constraint_matrix.conj().T @ v
    )

    q = np.linalg.solve(
        gram_matrix,
        response_vector,
    )

    return v @ q


def array_response_db(
    weights: np.ndarray,
    scan_angles_deg: np.ndarray,
    num_elements: int,
    spacing_wavelength: float,
) -> np.ndarray:
    response = []

    for angle_deg in scan_angles_deg:
        a = steering_vector(
            angle_deg,
            num_elements,
            spacing_wavelength,
        )

        response.append(
            np.abs(
                weights.conj().T @ a
            ) ** 2
        )

    response = np.asarray(response)
    response /= np.max(response)

    return 10.0 * np.log10(
        np.maximum(response, 1e-15)
    )


def main() -> None:
    rng = np.random.default_rng(2026)

    num_elements = 8
    num_snapshots = 5000
    spacing_wavelength = 0.5

    desired_angles_deg = np.array([-10.0, 22.0])
    interference_angles_deg = np.array([-40.0, 48.0])

    desired_powers = np.array([1.0, 0.8])
    interference_powers = np.array([80.0, 40.0])
    noise_power = 1.0

    desired_manifold = np.column_stack(
        [
            steering_vector(
                angle,
                num_elements,
                spacing_wavelength,
            )
            for angle in desired_angles_deg
        ]
    )

    interference_manifold = np.column_stack(
        [
            steering_vector(
                angle,
                num_elements,
                spacing_wavelength,
            )
            for angle in interference_angles_deg
        ]
    )

    desired_signals = (
        rng.standard_normal(
            (desired_angles_deg.size, num_snapshots)
        )
        + 1j
        * rng.standard_normal(
            (desired_angles_deg.size, num_snapshots)
        )
    ) / np.sqrt(2.0)

    desired_signals *= np.sqrt(
        desired_powers
    )[:, None]

    interference_signals = (
        rng.standard_normal(
            (
                interference_angles_deg.size,
                num_snapshots,
            )
        )
        + 1j
        * rng.standard_normal(
            (
                interference_angles_deg.size,
                num_snapshots,
            )
        )
    ) / np.sqrt(2.0)

    interference_signals *= np.sqrt(
        interference_powers
    )[:, None]

    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(
            (num_elements, num_snapshots)
        )
        + 1j
        * rng.standard_normal(
            (num_elements, num_snapshots)
        )
    )

    x = (
        desired_manifold @ desired_signals
        + interference_manifold @ interference_signals
        + noise
    )

    sample_covariance = (
        x @ x.conj().T / num_snapshots
    )

    loading_factor = 1e-2

    loading = (
        loading_factor
        * np.trace(sample_covariance).real
        / num_elements
    )

    loaded_covariance = (
        sample_covariance
        + loading * np.eye(num_elements)
    )

    a_first_desired = desired_manifold[:, 0]

    w_mvdr = calculate_mvdr_weights(
        loaded_covariance,
        a_first_desired,
    )

    explicit_null_vector = steering_vector(
        interference_angles_deg[0],
        num_elements,
        spacing_wavelength,
    )

    constraint_matrix = np.column_stack(
        [
            desired_manifold[:, 0],
            desired_manifold[:, 1],
            explicit_null_vector,
        ]
    )

    response_vector = np.array(
        [1.0, 1.0, 0.0],
        dtype=complex,
    )

    w_lcmv = calculate_lcmv_weights(
        loaded_covariance,
        constraint_matrix,
        response_vector,
    )

    print("LCMV 约束响应：")
    print(
        constraint_matrix.conj().T @ w_lcmv
    )

    scan_angles_deg = np.linspace(
        -90.0,
        90.0,
        1801,
    )

    response_mvdr_db = array_response_db(
        w_mvdr,
        scan_angles_deg,
        num_elements,
        spacing_wavelength,
    )

    response_lcmv_db = array_response_db(
        w_lcmv,
        scan_angles_deg,
        num_elements,
        spacing_wavelength,
    )

    plt.figure()
    plt.plot(
        scan_angles_deg,
        response_mvdr_db,
        label="单目标 MVDR",
    )

    plt.plot(
        scan_angles_deg,
        response_lcmv_db,
        label="多约束 LCMV",
    )

    for angle in desired_angles_deg:
        plt.axvline(
            angle,
            linestyle="--",
        )

    for angle in interference_angles_deg:
        plt.axvline(
            angle,
            linestyle=":",
        )

    plt.xlabel("角度 / 度")
    plt.ylabel("归一化方向响应 / dB")
    plt.title("MVDR 与 LCMV 方向响应")
    plt.grid(True)
    plt.legend()
    plt.ylim(-80.0, 5.0)

    plt.show()


if __name__ == "__main__":
    main()
```

---

# 附录 F 二维投影、旋转与复相位旋转补充

## F.1 二维方向单位向量

与 x 轴夹角为 theta 的单位方向向量为：

$$
\mathbf{u}
=
\begin{bmatrix}
\cos\theta \\
\sin\theta
\end{bmatrix}
$$

二维向量为：

$$
\mathbf{x}
=
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
$$

向量在方向 u 上的标量投影为：

$$
z
=
\mathbf{u}^{T}\mathbf{x}
$$

展开：

$$
\boxed{
z
=
x_1\cos\theta
+
x_2\sin\theta
}
$$

这是内积，不是主动旋转向量。

---

## F.2 向量投影

标量投影表示沿单位方向的坐标。

真正的二维投影向量为：

$$
\mathbf{x}_{\mathrm{proj}}
=
z\mathbf{u}
$$

代入标量投影：

$$
\mathbf{x}_{\mathrm{proj}}
=
\mathbf{u}\mathbf{u}^{T}\mathbf{x}
$$

因此投影矩阵为：

$$
\boxed{
\mathbf{P}
=
\mathbf{u}\mathbf{u}^{T}
}
$$

---

## F.3 二维主动旋转

将二维向量逆时针旋转 theta：

$$
\mathbf{x}'
=
\mathbf{R}(\theta)\mathbf{x}
$$

旋转矩阵为：

$$
\boxed{
\mathbf{R}(\theta)
=
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
}
$$

此时：

$$
x_1'
=
x_1\cos\theta
-
x_2\sin\theta
$$

$$
x_2'
=
x_1\sin\theta
+
x_2\cos\theta
$$

投影公式中的加号与旋转矩阵中的减号不同，是因为二者执行的操作不同。

---

## F.4 坐标轴旋转

若向量保持不动，而坐标轴逆时针旋转 theta，则新坐标为：

$$
\mathbf{z}
=
\mathbf{R}^{T}(\theta)\mathbf{x}
$$

因为：

$$
\mathbf{R}^{-1}(\theta)
=
\mathbf{R}^{T}(\theta)
=
\mathbf{R}(-\theta)
$$

所以：

$$
\begin{bmatrix}
z_1 \\
z_2
\end{bmatrix}
=
\begin{bmatrix}
\cos\theta & \sin\theta \\
-\sin\theta & \cos\theta
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
$$

第一个新坐标为：

$$
z_1
=
x_1\cos\theta
+
x_2\sin\theta
$$

它正是原向量在新坐标轴第一基向量方向上的投影。

---

## F.5 复数相位旋转

根据欧拉公式：

$$
e^{j\theta}
=
\cos\theta
+
j\sin\theta
$$

设：

$$
z
=
x+jy
$$

则：

$$
ze^{j\theta}
=
(x+jy)
(\cos\theta+j\sin\theta)
$$

展开：

$$
ze^{j\theta}
=
(x\cos\theta-y\sin\theta)
+
j(x\sin\theta+y\cos\theta)
$$

因此，复数乘以单位模复指数，与二维实平面旋转等价：

$$
\boxed{
e^{j\theta}
\quad
\Longleftrightarrow
\quad
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
}
$$

这解释了 ESPRIT 中“旋转不变性”的名称来源。

---

# 附录 G 代码运行环境与角度约定

建议使用以下环境：

```text
Python 3.10 或更高版本
NumPy
Matplotlib
```

安装命令：

```bash
pip install numpy matplotlib
```

代码中的角度约定为：

- 均匀线阵沿阵元索引方向排列
- 角度相对于阵列法线
- 阵元间距默认是半波长
- 导向矢量空间相位使用负号

导向矢量为：

$$
\mathbf{a}(\theta)
=
\begin{bmatrix}
1 \\
e^{-j2\pi(d/\lambda)\sin\theta} \\
\vdots \\
e^{-j2\pi(N-1)(d/\lambda)\sin\theta}
\end{bmatrix}
$$

若现有笔记使用相对于阵列轴方向定义角度，则可能需要把正弦关系改成余弦关系。

不同教材中的相位正负号也可能不同。

只要以下内容保持一致，算法结果就是一致的：

- 数据生成时的导向矢量
- MUSIC 扫描时的导向矢量
- ESPRIT 的角度恢复公式
- 波束响应计算时的导向矢量

---

# 附录 H 使用说明

本文件从第十五章开始编号，可直接追加在已经整理到第十四章的现有 Markdown 文档之后。

为提高 Typora、Obsidian 和 GitHub 兼容性，本文件采用：

- UTF-8 编码
- 标准 Markdown 标题
- 独立的双美元符号公式块
- 公式块前后留空行
- 不使用 HTML 锚点
- 不使用平台专属提示块
- 不使用行内 LaTeX 作为主要表达方式

将内容复制到现有文件时，应复制原始 Markdown 源码，而不是复制软件渲染后的页面内容。
