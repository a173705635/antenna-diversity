# 阵列信号处理学习笔记

> 学习目标：建立从电磁波传播到现代阵列信号处理（MUSIC、MVDR、CRPA）的完整知识体系。

---

# 整体知识框架

```text
            
```

---

# 第一章 阵列几何

## 1. 阵元（Element）

阵元是阵列中能够独立接收或发射信号的基本单元。
在典型接收阵列中，每个阵元通常对应：

- 一个物理天线单元；
- 一条射频前端链路；
- 一路 ADC 或数字中频数据；
- 一个随时间变化的复数采样序列。


对于 N 阵元阵列，在第 k 个采样时刻得到的接收向量为：

$$
\mathbf{x}[k]
=
\begin{bmatrix}
x_1[k] \\
x_2[k] \\
\vdots \\
x_N[k]
\end{bmatrix}
$$

这里每个 `x_i[k]` 是一个复数采样值，不是整段时间序列。

例如四阵元：

```
O   O   O   O
```

第 i 个阵元的完整时间序列则是：

$$
x_i[0],\;x_i[1],\;x_i[2],\;\ldots
$$

---

## 2. 阵列类型

按照阵元在空间中的排布方式，可以分为：

- 线性阵列：阵元沿一条直线排列；
- 平面阵列：阵元分布在一个平面内；
- 圆阵：阵元沿圆周排列；
- 立体阵列：阵元分布在三维空间中；
- 共形阵列：阵元贴合在曲面载体上。

线性阵列通常只能利用一个空间方向上的相位变化。平面阵列可以同时利用两个方向上的空间相位变化，因此能够估计方位角和俯仰角。

---

## 3. 阵元间距

均匀线阵中，相邻阵元间距通常记为 d。

为了避免扫描区域内出现空间混叠，常采用：

$$
d\leq \frac{\lambda}{2}
$$

其中 lambda 是载波波长。

当阵元间距过大时，不同入射方向可能产生相同的阵元间相位差，从而出现栅瓣。

---

## 4. 阵列孔径（Aperture）

对于 N 阵元均匀线阵，如果相邻阵元间距为 d，则首尾阵元中心之间的几何孔径为：

$$
L=(N-1)d
$$

孔径主要决定主瓣宽度和角度分辨能力。

需要区分：

- 固定阵元间距时，增加阵元数会同时增大孔径，所以主瓣通常变窄；
- 固定孔径时，增加阵元数主要提高空间采样密度、旁瓣控制能力和自由度；
- 对阵元数较少的离散阵列，即使孔径相同，主瓣宽度也可能存在一定差别；
- 当阵元足够密集时，离散阵列逐渐接近连续孔径模型。
核心理解：

> 决定角分辨率的是孔径，而不是阵元数量。**对于离散阵列，主瓣宽度主要由有效孔径决定，但当阵元数较少时，离散采样效应也会影响方向图；随着阵元数增加、阵元间距减小，离散阵列逐渐逼近连续孔径，此时主瓣宽度主要由孔径决定。**

---

# 第二章 电磁波传播

## 5. 波速、频率和波长

波速、频率与波长的关系为：

$$
v=f\lambda
$$

因此：

$$
\lambda=\frac{v}{f}
$$

GPS L1 载波频率为 1575.42 MHz，在真空中的波长约为：

$$
\lambda_{\mathrm{L1}}
=
\frac{299792458}{1575.42\times 10^6}
\approx 0.1903\;\mathrm{m}
$$

即约 19.03 cm。

## 6. 波数

波数表示波每传播单位距离所积累的相位：

$$
k=\frac{2\pi}{\lambda}
$$

其单位通常为 rad/m。

因此，传播距离差与相位差之间满足：

$$
\Delta\phi=k\Delta l
$$

---
## 7. 远场和平面波近似

严格地说，点源发出的波是球面波。对于阵列中第 n 个阵元，路径差应由各阵元到信号源的实际距离计算。

当信号源距离远大于阵列孔径时，阵列局部看到的球面波可以近似为平面波。

常见远场判据为：

$$
R\gg \frac{2D^2}{\lambda}
$$

其中 D 是阵列最大尺寸。

GNSS 卫星距离远大于普通接收阵列尺寸，因此平面波近似通常非常准确。


## 8. 路径差
对于阵元坐标沿 x 轴排列的均匀线阵，设入射角定义使正侧向为 0 度，则第 n 个阵元相对于参考阵元的远场路径差可写为：

$$
\Delta l_n=nd\sin\theta
$$

角度定义不同，公式中也可能出现 cos。关键是必须明确角度相对于阵列轴还是阵列法线定义。

---

## 9. 时间差与相位差

路径差对应的传播时间差为：

$$
\Delta\tau_n=\frac{\Delta l_n}{c}
$$

相位差为：

$$
\Delta\phi_n
=
2\pi f_c\Delta\tau_n
$$

利用波长关系，可以写成：

$$
\Delta\phi_n
=
knd\sin\theta
$$

阵列真正利用的是：

> **不同阵元之间的载波相位差。**

---


# 第三章 窄带阵列模型与导向矢量
## 10. GNSS 信号为什么可以写成同一个信号乘相位项

GPS L1 C/A 信号可以简化理解为幅度、扩频码、导航电文和载波的乘积。

真实情况下，第 n 个阵元收到的是延迟后的信号：

$$
x_n(t)=s(t-\tau_n)
$$

窄带假设认为阵元间传播延迟相对于信号包络变化时间非常小，因此包络近似不变，主要差别体现为载波相位变化：

$$
s(t-\tau_n)
\approx
s(t)e^{-j2\pi f_c\tau_n}
$$

对于普通小孔径 GNSS 阵列，阵元间时延通常远小于 C/A 码片持续时间，所以该近似非常适用。

## 10. 导向矢量

对于 N 阵元均匀线阵，导向矢量可以写为：

$$
\mathbf{a}(\theta)
=
\begin{bmatrix}
1 \\
e^{-jkd\sin\theta} \\
e^{-j2kd\sin\theta} \\
\vdots \\
e^{-j(N-1)kd\sin\theta}
\end{bmatrix}
$$

导向矢量描述来自某个方向的平面波在各阵元上形成的相对复数幅相关系。

它可以理解为该方向的空间指纹。

---

## 12. 单信号阵列模型

单个窄带信号的阵列模型为：

$$
\mathbf{x}[k]
=
\mathbf{a}(\theta)s[k]
+
\mathbf{n}[k]
$$

其中：

- `x[k]` 是第 k 个快拍；
- `a(theta)` 是导向矢量；
- `s[k]` 是信号在第 k 个采样时刻的复包络样本；
- `n[k]` 是阵列噪声向量。

---

## 13. 多信号阵列模型

如果有 M 个独立来波信号，则：

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

其中阵列流形矩阵为：

$$
\mathbf{A}
=
\begin{bmatrix}
\mathbf{a}(\theta_1) &
\mathbf{a}(\theta_2) &
\cdots &
\mathbf{a}(\theta_M)
\end{bmatrix}
$$

其中：

$$
\mathbf{s}[k]
$$

表示**第 k 个时刻，所有 M 个信号源的复数样本组成的列向量**。

具体写成：

$$
\mathbf{s}[k]=
\begin{bmatrix}
s_1[k]\\
s_2[k]\\
\vdots\\
s_M[k]
\end{bmatrix}
$$

其中：

- s₁[k]：第 1 个来波信号在时刻 k 的复数样本
- s₂[k]：第 2 个来波信号在时刻 k 的复数样本
- …
- s_M[k]：第 M 个来波信号在时刻 k 的复数样本

因此：

$$
\mathbf{s}[k]\in\mathbb{C}^{M\times1}
$$

---

## 代入矩阵乘法理解

阵列流形矩阵：

$$
\mathbf{A}=
\begin{bmatrix}
\mathbf{a}(\theta_1)&
\mathbf{a}(\theta_2)&
\cdots&
\mathbf{a}(\theta_M)
\end{bmatrix}
$$

乘上：

$$
\mathbf{s}[k]=
\begin{bmatrix}
s_1[k]\\
s_2[k]\\
\vdots\\
s_M[k]
\end{bmatrix}
$$

得到：

$$
\mathbf{A}\mathbf{s}[k]
=
\mathbf{a}(\theta_1)s_1[k]
+
\mathbf{a}(\theta_2)s_2[k]
+
\cdots
+
\mathbf{a}(\theta_M)s_M[k]
$$

这正好就是：

$$
\mathbf{x}[k]
=
\sum_{m=1}^{M}
\mathbf{a}(\theta_m)s_m[k]
+
\mathbf{n}[k]
$$

---


# 第四章 波束形成与阵列因子

## 14. 波束形成的本质

接收波束形成的基本表达式为：

$$
y[k]=\mathbf{w}^H\mathbf{x}[k]
$$

展开后为：

$$
y[k]
=
\sum_{n=1}^{N}w_n^*x_n[k]
$$

波束形成的本质是：

1. 对每个阵元通道进行复数加权；
2. 补偿目标方向上的相位差；
3. 将各阵元信号相干叠加。

如果接收模型采用前述导向矢量定义，则传统波束形成可以选择：

$$
\mathbf{w}
=
\frac{\mathbf{a}(\theta_0)}{N}
$$

因为实际输出使用的是 `w` 的共轭转置，所以会自动形成与导向矢量相反的相位补偿。

## 15. 阵列因子

波束固定指向 theta0 时，对任意测试方向 theta 的响应为：

$$
AF(\theta)
=
\mathbf{w}^H\mathbf{a}(\theta)
$$

归一化幅度方向图可写为：

$$
G(\theta)
=
\frac{|AF(\theta)|}
{\max_{\theta}|AF(\theta)|}
$$

阵列因子表示：如果信号从某个方向来，经过当前权值合并后能够得到多大的响应。

## 16. 主瓣、旁瓣和零点

- 主瓣：波束指向附近的主要高增益区域；
- 旁瓣：主瓣之外的局部响应峰值；
- 零点：多个阵元输出完全抵消的位置；
- 半功率波束宽度：功率下降到主瓣峰值一半时的两个角度之间的宽度。

功率下降到一半，对应幅度下降到：

$$
\frac{1}{\sqrt{2}}
$$

在分贝表示中约为 -3 dB。

## 17. 栅瓣

均匀线阵相邻阵元的空间相位差为：

$$
\psi(\theta)=kd\sin\theta
$$

如果两个方向满足：

$$
kd\sin\theta_2
=
kd\sin\theta_1+2\pi m
$$

则它们在均匀线阵中可能形成相同的相位序列。

整理得到：

$$
\sin\theta_2
=
\sin\theta_1
+
m\frac{\lambda}{d}
$$

这就是方向混叠条件。

栅瓣的本质是空间采样不足，与时间采样中的频谱混叠相对应。

# 第五章 快拍、数据矩阵和协方差矩阵

## 18. 快拍

在第 k 个 ADC 采样时刻，同时取得全部阵元数据，构成一个快拍：

$$
\mathbf{x}[k]
=
\begin{bmatrix}
x_1[k] \\
x_2[k] \\
\vdots \\
x_N[k]
\end{bmatrix}
$$

快拍描述的是空间维度：同一时刻不同阵元的采样。

单个阵元的序列描述的是时间维度：同一阵元在不同时刻的采样。

---

## 19. 数据矩阵

将 K 个快拍按列排列：

$$
\mathbf{X}
=
\begin{bmatrix}
| & | & & | \\
\mathbf{x}[1] &
\mathbf{x}[2] &
\cdots &
\mathbf{x}[K] \\
| & | & & |
\end{bmatrix}
$$

数据矩阵维度为：

$$
\mathbf{X}\in\mathbb{C}^{N\times K}
$$

---

## 20. 均值和去均值

严格的协方差矩阵描述数据相对于均值的变化。

均值向量为：

$$
\boldsymbol{\mu}
=
E[\mathbf{x}]
$$

实数随机向量的协方差矩阵为：

$$
\mathbf{R}
=
E\left[
(\mathbf{x}-\boldsymbol{\mu})
(\mathbf{x}-\boldsymbol{\mu})^T
\right]
$$

复数阵列信号的协方差矩阵为：

$$
\mathbf{R}
=
E\left[
(\mathbf{x}-\boldsymbol{\mu})
(\mathbf{x}-\boldsymbol{\mu})^H
\right]
$$

如果数据已经零均值，可以简化为：

$$
\mathbf{R}=E[\mathbf{x}\mathbf{x}^H]
$$

在通信与阵列处理中，载波下变频后的随机复包络通常按零均值模型处理。

---

## 21. 样本协方差矩阵

有限快拍下，用样本平均估计协方差矩阵：

$$
\hat{\mathbf{R}}
=
\frac{1}{K}
\sum_{k=1}^{K}
\mathbf{x}[k]\mathbf{x}^H[k]
$$

矩阵形式为：

$$
\hat{\mathbf{R}}
=
\frac{1}{K}
\mathbf{X}\mathbf{X}^H
$$

其维度关系为：

$$
(N\times K)(K\times N)=N\times N
$$

---

## 22. 协方差矩阵元素的含义

协方差矩阵可写成：

$$
\mathbf{R}
=
\begin{bmatrix}
R_{11} & R_{12} & \cdots & R_{1N} \\
R_{21} & R_{22} & \cdots & R_{2N} \\
\vdots & \vdots & \ddots & \vdots \\
R_{N1} & R_{N2} & \cdots & R_{NN}
\end{bmatrix}
$$

协方差矩阵第 i 行、第 j 列的元素为：
$$
R_{ij}=E[x_i x_j^*]
$$

对角线元素为：

$$
R_{ii}=E[|x_i|^2]
$$

它表示第 i 个阵元的平均功率。

非对角线元素表示不同阵元之间的复相关关系，其中同时包含：

- 共同信号成分；
- 相对相位关系；
- 幅度相关程度。

阵列的方向信息主要体现在这些跨阵元相关项中。

---

## 23. 协方差矩阵保存了什么

协方差矩阵不会保存每个原始快拍的具体值，也不会保存快拍的时间顺序。

它保存的是大量快拍共同表现出的二阶统计结构：

- 各阵元平均功率；
- 阵元之间的相关性；
- 数据主要分布方向；
- 各主要方向上的能量。

因此协方差矩阵是一种统计压缩，而不是无损数据存储。

# 第六章 向量投影

## 24. 单位方向向量

二维空间中，一个角度为 theta 的单位方向向量可以写为：

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

单位长度约束保证 `u` 只表示方向，不通过人为放大自身长度来放大投影结果。

---

## 25. 投影坐标

设二维点为：

$$
\mathbf{x}
=
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
$$

它在单位方向 `u` 上的投影坐标为：

$$
z=\mathbf{u}^T\mathbf{x}
$$

展开得到：

$$
z=x_1\cos\theta+x_2\sin\theta
$$

`z` 是一个标量，表示点在这条方向轴上的一维坐标。

> 这里是内积，而“向量长度乘以它与投影轴夹角的余弦”，正是几何上的投影长度。



---

## 26. 投影向量

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

所有点投影后都会落在方向 `u` 所在的同一条直线上。

---

# 第七章 投影方差与二次型

## 27. 投影后的方差

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
$$

因为：

$$
(\mathbf{u}^T\mathbf{x})^2
=
\mathbf{u}^T\mathbf{x}\mathbf{x}^T\mathbf{u}
$$

所以：

$$
\operatorname{Var}(z)
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

对于复数阵列信号，若输出为：

$$
z=\mathbf{u}^H\mathbf{x}
$$

则平均输出功率为：

$$
E[|z|^2]
=
\mathbf{u}^H\mathbf{R}\mathbf{u}
$$

---

## 28. 二次型的含义

表达式：

$$
\mathbf{u}^H\mathbf{R}\mathbf{u}
$$

称为二次型。

在协方差矩阵语境中，它表示数据沿方向 `u` 投影后的平均能量或方差。

可以把协方差矩阵理解为一个方向测试器：

1. 输入一个单位方向 `u`；
2. 计算二次型；
3. 得到数据沿该方向有多分散或有多少能量。

---

# 第八章 特征值和特征向量

## 29. 特征值方程

如果存在非零向量 `v`，满足：

$$
\mathbf{A}\mathbf{v}
=
\lambda\mathbf{v}
$$

则 `v` 是矩阵 A 的特征向量，lambda 是对应的特征值。

其几何含义是：矩阵作用在该向量上以后，向量方向不变，只发生比例缩放；当特征值为负数时还会发生方向翻转。

---

## 30. 具体数值例子

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

因此：

$$
\det
\begin{bmatrix}
2-\lambda & 1 \\
1 & 2-\lambda
\end{bmatrix}
=0
$$

计算得到：

$$
(2-\lambda)^2-1=0
$$

特征值为：

$$
\lambda_1=3
$$

$$
\lambda_2=1
$$

当特征值为 3 时，可以取特征向量：

$$
\mathbf{v}_1
=
\begin{bmatrix}
1 \\
1
\end{bmatrix}
$$

验证：

$$
\mathbf{A}\mathbf{v}_1
=
\begin{bmatrix}
3 \\
3
\end{bmatrix}
=
3\mathbf{v}_1
$$

当特征值为 1 时，可以取特征向量：

$$
\mathbf{v}_2
=
\begin{bmatrix}
1 \\
-1
\end{bmatrix}
$$

验证：

$$
\mathbf{A}\mathbf{v}_2
=
\begin{bmatrix}
1 \\
-1
\end{bmatrix}
=
\mathbf{v}_2
$$

---

## 31. 普通矩阵与协方差矩阵中的含义区别

对于一般线性变换矩阵：

- 特征向量表示作用后方向保持不变的特殊方向；
- 特征值表示沿该方向的缩放倍数。

对于协方差矩阵：

- 特征向量表示数据点云的主轴方向；
- 特征值表示数据沿该主轴方向的方差或能量。

这是因为协方差矩阵本身编码的是数据在各方向上的分散程度。

---

# 第九章 用具体协方差矩阵理解特征分解

## 32. 示例协方差矩阵

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
$$

展开得到：

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
\cos^2\theta+\sin^2\theta=1
$$

$$
2\sin\theta\cos\theta=\sin 2\theta
$$

得到：

$$
\mathbf{u}^T\mathbf{R}\mathbf{u}
=
5+4\sin 2\theta
$$

---

## 33. 不同方向上的方差

当方向为 0 度时：

$$
\mathbf{u}^T\mathbf{R}\mathbf{u}=5
$$

当方向为 45 度时：

$$
\mathbf{u}^T\mathbf{R}\mathbf{u}=9
$$

当方向为 90 度时：

$$
\mathbf{u}^T\mathbf{R}\mathbf{u}=5
$$

当方向为 135 度时：

$$
\mathbf{u}^T\mathbf{R}\mathbf{u}=1
$$

因此点云沿 45 度方向分布最宽，沿 135 度方向分布最窄。

对应的单位特征向量为：

$$
\mathbf{v}_1
=
\frac{1}{\sqrt{2}}
\begin{bmatrix}
1 \\
1
\end{bmatrix}
$$

$$
\mathbf{v}_2
=
\frac{1}{\sqrt{2}}
\begin{bmatrix}
1 \\
-1
\end{bmatrix}
$$

对应特征值为：

$$
\lambda_1=9
$$

$$
\lambda_2=1
$$

所以，协方差矩阵的特征向量正是点云的主轴，而特征值正是对应主轴上的方差。

# 第十章 拉格朗日乘子与特征值方程

## 34. 为什么需要约束

如果直接最大化二次型，而不限制向量长度，只要不断放大 `u`，目标函数就会无限增大。

因此要求 `u` 为单位向量：

$$
\mathbf{u}^T\mathbf{u}=1
$$

优化问题为：

$$
\max_{\mathbf{u}}
\mathbf{u}^T\mathbf{R}\mathbf{u}
$$

约束为：

$$
\mathbf{u}^T\mathbf{u}=1
$$

---

## 35. 拉格朗日乘子的几何含义

带约束极值点处，目标函数的等值面和约束曲面相切。

相切意味着两者的法向方向平行，而梯度就是等值面的法向量。因此：

$$
\nabla f
=
\lambda\nabla g
$$

lambda 用于补偿两个梯度向量长度的差异。

---

## 36. 构造拉格朗日函数

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

## 37. 为什么特征值等于投影方差

从特征值方程出发：

$$
\mathbf{R}\mathbf{u}
=
\lambda\mathbf{u}
$$

左乘转置向量：

$$
\mathbf{u}^T\mathbf{R}\mathbf{u}
=
\lambda\mathbf{u}^T\mathbf{u}
$$

因为 `u` 是单位向量：

$$
\mathbf{u}^T\mathbf{u}=1
$$

因此：

$$
\mathbf{u}^T\mathbf{R}\mathbf{u}
=
\lambda
$$

左边是投影方差，所以特征值就是对应特征方向上的方差。

最大特征值对应最大方差方向，最小特征值对应最小方差方向。

---

# 第十一章 协方差矩阵为什么保留原始向量之间的关系

## 38. 单个向量的外积

设：

$$
\mathbf{x}
=
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
$$

外积为：

$$
\mathbf{x}\mathbf{x}^T
=
\begin{bmatrix}
x_1^2 & x_1x_2 \\
x_2x_1 & x_2^2
\end{bmatrix}
$$

其中：

- 对角元素记录各维度的瞬时平方能量；
- 非对角元素记录两个维度在该样本中的共同变化关系。

对大量样本的外积求平均，就能保留反复出现的稳定关系。

---

## 39. 同方向数据的例子

假设所有样本都沿同一方向：

$$
\mathbf{x}[k]
=
\alpha_k
\begin{bmatrix}
1 \\
1
\end{bmatrix}
$$

则：

$$
\mathbf{x}[k]\mathbf{x}^T[k]
=
\alpha_k^2
\begin{bmatrix}
1 & 1 \\
1 & 1
\end{bmatrix}
$$

求平均得到：

$$
\mathbf{R}
=
E[\alpha_k^2]
\begin{bmatrix}
1 & 1 \\
1 & 1
\end{bmatrix}
$$

虽然单个样本的长度系数被平均掉了，但共同方向仍然保存在矩阵结构中。

最大特征向量正是：

$$
\begin{bmatrix}
1 \\
1
\end{bmatrix}
$$

---

## 40. 协方差矩阵不是恢复原始样本

特征值分解不会恢复每一个快拍，也无法知道每个样本出现的先后顺序。

它提取的是：

- 哪些空间模式在大量数据中持续存在；
- 每个模式的平均能量有多大；
- 模式之间是否正交或相关。

因此 EVD 的作用是提取统计结构，而不是重建原始数据。

---

## 41. 与奇异值分解的关系

对数据矩阵做奇异值分解：

$$
\mathbf{X}
=
\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^H
$$

则：

$$
\mathbf{X}\mathbf{X}^H
=
\mathbf{U}\boldsymbol{\Sigma}^2\mathbf{U}^H
$$

因此：

- 协方差矩阵的特征向量对应数据矩阵的左奇异向量；
- 协方差矩阵的特征值与奇异值的平方成正比；
- EVD 与 SVD 对数据主要空间方向的描述是一致的。

---

# 第十二章 回到阵列信号模型

## 42. 单信号、无噪声情况

单信号模型为：

$$
\mathbf{x}[k]
=
\mathbf{a}s[k]
$$

协方差矩阵为：

$$
\mathbf{R}
=
E[\mathbf{x}\mathbf{x}^H]
$$

代入接收模型：

$$
\mathbf{R}
=
E[\mathbf{a}s s^*\mathbf{a}^H]
$$

导向矢量不随快拍变化，因此可以移到期望之外：

$$
\mathbf{R}
=
\mathbf{a}E[|s|^2]\mathbf{a}^H
$$

定义信号功率：

$$
P_s=E[|s|^2]
$$

得到：

$$
\mathbf{R}
=
P_s\mathbf{a}\mathbf{a}^H
$$

---

## 43. 为什么导向矢量是协方差矩阵的特征向量

用协方差矩阵乘导向矢量：

$$
\mathbf{R}\mathbf{a}
=
P_s\mathbf{a}\mathbf{a}^H\mathbf{a}
$$

内积为：

$$
\mathbf{a}^H\mathbf{a}
=
\|\mathbf{a}\|^2
$$

因此：

$$
\mathbf{R}\mathbf{a}
=
P_s\|\mathbf{a}\|^2\mathbf{a}
$$

这与特征值方程形式完全一致：

$$
\mathbf{R}\mathbf{a}
=
\lambda_s\mathbf{a}
$$

对应特征值为：

$$
\lambda_s
=
P_s\|\mathbf{a}\|^2
$$

所以在单信号、无噪声情况下，导向矢量本身就是协方差矩阵的非零特征向量。

---

## 44. 加入空间白噪声

设噪声满足：

$$
E[\mathbf{n}\mathbf{n}^H]
=
\sigma_n^2\mathbf{I}
$$

并且信号与噪声不相关，则：

$$
\mathbf{R}
=
P_s\mathbf{a}\mathbf{a}^H
+
\sigma_n^2\mathbf{I}
$$

对导向矢量方向：

$$
\mathbf{R}\mathbf{a}
=
\left(
P_s\|\mathbf{a}\|^2
+
\sigma_n^2
\right)
\mathbf{a}
$$

因此信号方向的特征值为：

$$
\lambda_s
=
P_s\|\mathbf{a}\|^2
+
\sigma_n^2
$$

对于任意与导向矢量正交的向量 `v`：

$$
\mathbf{a}^H\mathbf{v}=0
$$

有：

$$
\mathbf{R}\mathbf{v}
=
\sigma_n^2\mathbf{v}
$$

因此其余方向的特征值均为噪声功率：

$$
\lambda_n=\sigma_n^2
$$

理想情况下，单信号阵列的特征值结构是：一个较大的信号特征值，以及 N-1 个相等的噪声特征值。

---

## 45. 多信号情况

多信号模型为：

$$
\mathbf{x}[k]
=
\mathbf{A}\mathbf{s}[k]
+
\mathbf{n}[k]
$$

信号协方差矩阵为：

$$
\mathbf{R}_s
=
E[\mathbf{s}\mathbf{s}^H]
$$

阵列协方差矩阵为：

$$
\mathbf{R}
=
\mathbf{A}\mathbf{R}_s\mathbf{A}^H
+
\sigma_n^2\mathbf{I}
$$

如果有 M 个线性独立信号，且 M 小于 N，则：

- 最大的 M 个特征值对应信号成分；
- 对应特征向量张成信号子空间；
- 剩余 N-M 个特征向量张成噪声子空间。

需要注意：多信号情况下，单个信号特征向量不一定等于某一根导向矢量。更准确地说，所有真实导向矢量都位于信号子空间中。

---

# 第十三章 信号子空间与噪声子空间

## 46. 特征值分解

协方差矩阵是厄米矩阵，可以进行特征值分解：

$$
\mathbf{R}
=
\mathbf{E}\boldsymbol{\Lambda}\mathbf{E}^H
$$

按照特征值从大到小排列：

$$
\lambda_1\geq\lambda_2\geq\cdots\geq\lambda_N
$$

将特征向量划分为信号子空间和噪声子空间：

$$
\mathbf{E}
=
\begin{bmatrix}
\mathbf{E}_s & \mathbf{E}_n
\end{bmatrix}
$$

---

## 47. 信号子空间

如果有 M 个独立信号，信号子空间矩阵为：

$$
\mathbf{E}_s
=
\begin{bmatrix}
\mathbf{e}_1 &
\mathbf{e}_2 &
\cdots &
\mathbf{e}_M
\end{bmatrix}
$$

所有真实信号导向矢量满足：

$$
\mathbf{a}(\theta_m)
\in
\operatorname{span}(\mathbf{E}_s)
$$

也就是说，真实信号的空间模式可以由信号特征向量线性表示。

---

## 48. 噪声子空间

噪声子空间矩阵为：

$$
\mathbf{E}_n
=
\begin{bmatrix}
\mathbf{e}_{M+1} &
\cdots &
\mathbf{e}_N
\end{bmatrix}
$$

信号子空间与噪声子空间互相正交，因此真实导向矢量满足：

$$
\mathbf{E}_n^H\mathbf{a}(\theta_m)=0
$$

这条正交关系是 MUSIC 算法的核心依据。

---

# 第十四章 本阶段概念关系总结

## 49. 从原始数据到子空间的完整链条

```text
电磁波从某个方向到达
        ↓
阵元间产生路径差和相位差
        ↓
形成该方向的导向矢量
        ↓
每个采样时刻得到一个快拍
        ↓
大量快拍组成数据矩阵
        ↓
计算样本协方差矩阵
        ↓
协方差矩阵编码稳定的跨阵元关系
        ↓
特征值分解寻找主要空间模式
        ↓
大特征值方向构成信号子空间
        ↓
小特征值方向构成噪声子空间
        ↓
利用子空间正交性进行 DOA 估计
```

---

## 50. 核心公式

阵列接收模型：

$$
\mathbf{x}[k]
=
\mathbf{A}\mathbf{s}[k]
+
\mathbf{n}[k]
$$

样本协方差矩阵：

$$
\hat{\mathbf{R}}
=
\frac{1}{K}\mathbf{X}\mathbf{X}^H
$$

方向 `u` 上的输出功率：

$$
P_{\mathrm{out}}
=
\mathbf{u}^H\mathbf{R}\mathbf{u}
$$

特征值方程：

$$
\mathbf{R}\mathbf{u}
=
\lambda\mathbf{u}
$$

特征值分解：

$$
\mathbf{R}
=
\mathbf{E}\boldsymbol{\Lambda}\mathbf{E}^H
$$

信号与噪声子空间正交：

$$
\mathbf{E}_n^H\mathbf{a}(\theta_m)=0
$$

---

## 51. 最重要的概念区别

| 概念           | 主要含义                           |
| -------------- | ---------------------------------- |
| 快拍 `x[k]`    | 同一时刻所有阵元的复数采样         |
| 数据矩阵 `X`   | 多个快拍的完整集合                 |
| 协方差矩阵 `R` | 快拍中稳定的二阶空间统计关系       |
| 外积           | 记录一个快拍各分量之间的共同关系   |
| 投影           | 将高维数据映射到指定的一维方向     |
| 二次型         | 数据沿指定方向的平均能量或方差     |
| 特征向量       | 协方差矩阵对应的数据主轴或稳定模式 |
| 特征值         | 对应主轴方向的能量或方差           |
| 信号子空间     | 大特征值对应向量张成的空间         |
| 噪声子空间     | 小特征值对应向量张成的正交补空间   |

---

# 第十五章 后续学习路线

接下来可以依次学习：

1. MUSIC 算法：利用导向矢量与噪声子空间正交估计入射方向；
2. MUSIC 谱峰形成原因与分辨能力；
3. 信号数量估计：特征值阈值、AIC 和 MDL；
4. 相干信号与多径导致的协方差矩阵秩亏；
5. 空间平滑；
6. ESPRIT 算法；
7. MVDR 或 Capon 波束形成；
8. LCMV 多约束波束形成；
9. GNSS CRPA 抗干扰；
10. 空时自适应处理与宽带阵列模型。



---

---

# 附录 A：二维旋转矩阵的推导

## A.1 问题描述

假设二维平面中有一个向量：

$$
\mathbf{x}
=
\begin{bmatrix}
x \\
y
\end{bmatrix}
$$

将这个向量绕坐标原点逆时针旋转角度 theta，旋转后的向量记为：

$$
\mathbf{x}'
=
\begin{bmatrix}
x' \\
y'
\end{bmatrix}
$$

目标是推导旋转前后的坐标关系，并将其写成矩阵形式。

---

## A.2 使用极坐标表示原向量

设原向量长度为 r，与 x 轴正方向的夹角为 alpha。

原向量的横坐标为：

$$
x = r \cos(\alpha)
$$

原向量的纵坐标为：

$$
y = r \sin(\alpha)
$$

因此：

$$
\mathbf{x}
=
\begin{bmatrix}
r \cos(\alpha) \\
r \sin(\alpha)
\end{bmatrix}
$$

---

## A.3 旋转后的坐标

逆时针旋转角度 theta 后，向量长度不变，方向角由 alpha 变为：

$$
\alpha + \theta
$$

因此，旋转后的横坐标为：

$$
x'
=
r \cos(\alpha + \theta)
$$

旋转后的纵坐标为：

$$
y'
=
r \sin(\alpha + \theta)
$$

---

## A.4 横坐标的推导

根据余弦和角公式：

$$
\cos(\alpha + \theta)
=
\cos(\alpha)\cos(\theta)
-
\sin(\alpha)\sin(\theta)
$$

因此：

$$
x'
=
r\cos(\alpha)\cos(\theta)
-
r\sin(\alpha)\sin(\theta)
$$

由于：

$$
r\cos(\alpha)=x
$$

以及：

$$
r\sin(\alpha)=y
$$

所以：

$$
x'
=
x\cos(\theta)
-
y\sin(\theta)
$$

---

## A.5 纵坐标的推导

根据正弦和角公式：

$$
\sin(\alpha + \theta)
=
\sin(\alpha)\cos(\theta)
+
\cos(\alpha)\sin(\theta)
$$

因此：

$$
y'
=
r\sin(\alpha)\cos(\theta)
+
r\cos(\alpha)\sin(\theta)
$$

代入：

$$
r\sin(\alpha)=y
$$

以及：

$$
r\cos(\alpha)=x
$$

得到：

$$
y'
=
x\sin(\theta)
+
y\cos(\theta)
$$

---

## A.6 写成矩阵形式

现在已经得到：

$$
x'
=
x\cos(\theta)
-
y\sin(\theta)
$$

以及：

$$
y'
=
x\sin(\theta)
+
y\cos(\theta)
$$

组合成矩阵形式：

$$
\begin{bmatrix}
x' \\
y'
\end{bmatrix}
=
\begin{bmatrix}
\cos(\theta) & -\sin(\theta) \\
\sin(\theta) & \cos(\theta)
\end{bmatrix}
\begin{bmatrix}
x \\
y
\end{bmatrix}
$$

因此，二维逆时针旋转矩阵为：

$$
\mathbf{R}(\theta)
=
\begin{bmatrix}
\cos(\theta) & -\sin(\theta) \\
\sin(\theta) & \cos(\theta)
\end{bmatrix}
$$

旋转关系为：

$$
\mathbf{x}'
=
\mathbf{R}(\theta)\mathbf{x}
$$

---

## A.7 从标准基向量理解旋转矩阵

二维空间的两个标准基向量为：

$$
\mathbf{e}_1
=
\begin{bmatrix}
1 \\
0
\end{bmatrix}
$$

以及：

$$
\mathbf{e}_2
=
\begin{bmatrix}
0 \\
1
\end{bmatrix}
$$

任意二维向量都可以表示为：

$$
\mathbf{x}
=
x\mathbf{e}_1
+
y\mathbf{e}_2
$$

旋转是线性变换，因此：

$$
\mathbf{R}(\theta)\mathbf{x}
=
x\mathbf{R}(\theta)\mathbf{e}_1
+
y\mathbf{R}(\theta)\mathbf{e}_2
$$

第一个标准基向量旋转后的结果为：

$$
\mathbf{R}(\theta)\mathbf{e}_1
=
\begin{bmatrix}
\cos(\theta) \\
\sin(\theta)
\end{bmatrix}
$$

第二个标准基向量旋转后的结果为：

$$
\mathbf{R}(\theta)\mathbf{e}_2
=
\begin{bmatrix}
-\sin(\theta) \\
\cos(\theta)
\end{bmatrix}
$$

因此，旋转矩阵的两列分别是两个标准基向量旋转后的坐标：

$$
\mathbf{R}(\theta)
=
\begin{bmatrix}
\cos(\theta) & -\sin(\theta) \\
\sin(\theta) & \cos(\theta)
\end{bmatrix}
$$

---

## A.8 旋转 90 度的例子

当旋转角度为：

$$
\theta
=
\frac{\pi}{2}
$$

有：

$$
\cos\left(\frac{\pi}{2}\right)=0
$$

以及：

$$
\sin\left(\frac{\pi}{2}\right)=1
$$

旋转矩阵为：

$$
\mathbf{R}\left(\frac{\pi}{2}\right)
=
\begin{bmatrix}
0 & -1 \\
1 & 0
\end{bmatrix}
$$

取向量：

$$
\mathbf{x}
=
\begin{bmatrix}
1 \\
0
\end{bmatrix}
$$

旋转后：

$$
\mathbf{x}'
=
\begin{bmatrix}
0 & -1 \\
1 & 0
\end{bmatrix}
\begin{bmatrix}
1 \\
0
\end{bmatrix}
$$

得到：

$$
\mathbf{x}'
=
\begin{bmatrix}
0 \\
1
\end{bmatrix}
$$

这表示向量从 x 轴正方向逆时针旋转到了 y 轴正方向。

---

## A.9 旋转矩阵为什么保持长度不变

旋转前的长度平方为：

$$
\|\mathbf{x}\|^2
=
\mathbf{x}^{T}\mathbf{x}
$$

旋转后：

$$
\mathbf{x}'
=
\mathbf{R}\mathbf{x}
$$

所以：

$$
\|\mathbf{x}'\|^2
=
\mathbf{x}'^{T}\mathbf{x}'
$$

代入旋转关系：

$$
\|\mathbf{x}'\|^2
=
(\mathbf{R}\mathbf{x})^{T}
(\mathbf{R}\mathbf{x})
$$

根据矩阵转置规则：

$$
(\mathbf{R}\mathbf{x})^{T}
=
\mathbf{x}^{T}\mathbf{R}^{T}
$$

因此：

$$
\|\mathbf{x}'\|^2
=
\mathbf{x}^{T}
\mathbf{R}^{T}
\mathbf{R}
\mathbf{x}
$$

旋转矩阵满足：

$$
\mathbf{R}^{T}\mathbf{R}
=
\mathbf{I}
$$

因此：

$$
\|\mathbf{x}'\|^2
=
\mathbf{x}^{T}\mathbf{x}
$$

也就是：

$$
\|\mathbf{x}'\|
=
\|\mathbf{x}\|
$$

所以旋转只改变向量方向，不改变向量长度。

---

## A.10 顺时针旋转矩阵

顺时针旋转角度 theta，等价于逆时针旋转负 theta：

$$
\mathbf{R}(-\theta)
=
\begin{bmatrix}
\cos(\theta) & \sin(\theta) \\
-\sin(\theta) & \cos(\theta)
\end{bmatrix}
$$

---

## A.11 复数相位旋转

复数可以写成：

$$
z
=
x + jy
$$

也可以写成：

$$
z
=
r e^{j\alpha}
$$

乘以复指数：

$$
e^{j\theta}
$$

得到：

$$
z'
=
z e^{j\theta}
=
r e^{j(\alpha+\theta)}
$$

这表示复数幅度不变，相位增加 theta。

将复数乘法展开：

$$
z'
=
(x+jy)
\left(
\cos(\theta)
+
j\sin(\theta)
\right)
$$

整理实部和虚部：

$$
z'
=
\left(
x\cos(\theta)
-
y\sin(\theta)
\right)
+
j
\left(
x\sin(\theta)
+
y\cos(\theta)
\right)
$$

因此：

$$
x'
=
x\cos(\theta)
-
y\sin(\theta)
$$

以及：

$$
y'
=
x\sin(\theta)
+
y\cos(\theta)
$$

这与二维旋转矩阵完全一致。

因此：

$$
e^{j\theta}
\quad
\text{对应二维平面中的角度旋转}
$$

---

## A.12 核心结论

二维逆时针旋转矩阵为：

$$
\mathbf{R}(\theta)
=
\begin{bmatrix}
\cos(\theta) & -\sin(\theta) \\
\sin(\theta) & \cos(\theta)
\end{bmatrix}
$$

旋转后的向量为：

$$
\mathbf{x}'
=
\mathbf{R}(\theta)\mathbf{x}
$$

旋转矩阵满足：

$$
\mathbf{R}^{T}\mathbf{R}
=
\mathbf{I}
$$

因此旋转矩阵只改变向量方向，不改变向量长度。
