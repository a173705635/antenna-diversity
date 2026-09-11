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
# 第二十五章：LCMV 多约束波束形成

LCMV 的英文全称为：

**Linearly Constrained Minimum Variance**

中文通常称为：

**线性约束最小方差波束形成**

LCMV 可以看作 MVDR 的多约束扩展。

两者的主要区别是：

- MVDR 只设置一个无失真约束；
- LCMV 可以同时设置多个线性约束。

在 GNSS 阵列处理中，LCMV 适合处理以下问题：

- 同时保护多个卫星方向；
- 对已知干扰方向形成零陷；
- 在目标方向附近形成宽保护区；
- 在有限阵元条件下分配阵列自由度。

---

## 1. 回顾 MVDR

MVDR 的目标是：

> 在保证目标方向信号无失真通过的前提下，使阵列输出总功率最小。

目标函数为：

$$
\min_{\mathbf{w}}
\mathbf{w}^{H}
\mathbf{R}
\mathbf{w}
$$

约束条件为：

$$
\mathbf{w}^{H}
\mathbf{a}_{0}
=
1
$$

其中：

- `w` 是阵列复权向量；
- `R` 是接收信号协方差矩阵；
- `a0` 是目标方向的导向矢量；
- 上标 `H` 表示共轭转置。

MVDR 的最优权值为：

$$
\mathbf{w}_{\mathrm{MVDR}}
=
\frac{
\mathbf{R}^{-1}
\mathbf{a}_{0}
}{
\mathbf{a}_{0}^{H}
\mathbf{R}^{-1}
\mathbf{a}_{0}
}
$$

这个权值完成两件事：

1. 保证目标方向增益为 1；
2. 根据协方差矩阵抑制干扰和噪声。

---

## 2. 为什么需要 LCMV

MVDR 只有一个约束。

但 GNSS 中往往存在多颗不同方向的卫星，因此可能需要同时满足多个条件，例如：

- 卫星 1 方向保持单位增益；
- 卫星 2 方向保持单位增益；
- 干扰方向形成零陷；
- 某个方向保持指定增益。

这时就需要把单约束扩展为多约束。

---

## 3. 约束矩阵

假设希望控制 `Q` 个方向。

每个方向对应一个导向矢量：

$$
\mathbf{a}_{1},
\mathbf{a}_{2},
\ldots,
\mathbf{a}_{Q}
$$

将这些导向矢量按列排列，组成约束矩阵：

$$
\mathbf{C}
=
\begin{bmatrix}
\mathbf{a}_{1} &
\mathbf{a}_{2} &
\cdots &
\mathbf{a}_{Q}
\end{bmatrix}
$$

如果阵列有 `N` 个阵元，则约束矩阵的尺寸为：

$$
\mathbf{C}
\in
\mathbb{C}^{N \times Q}
$$

其中：

- 每一列对应一个受约束方向；
- 每一行对应一个阵元。

---

## 4. 期望响应向量

对每个约束方向，都要指定一个期望响应。

将这些期望响应组成向量：

$$
\mathbf{f}
=
\begin{bmatrix}
f_{1} \\
f_{2} \\
\vdots \\
f_{Q}
\end{bmatrix}
$$

多方向约束可以统一写成：

$$
\mathbf{C}^{H}
\mathbf{w}
=
\mathbf{f}
$$

这里：

- `C` 决定约束哪些方向；
- `f` 决定这些方向分别得到什么响应。

---

## 5. 多颗卫星单位增益约束

假设希望同时保护三颗卫星，并让三个方向都保持单位增益。

此时：

$$
\mathbf{C}
=
\begin{bmatrix}
\mathbf{a}_{1} &
\mathbf{a}_{2} &
\mathbf{a}_{3}
\end{bmatrix}
$$

期望响应向量为：

$$
\mathbf{f}
=
\begin{bmatrix}
1 \\
1 \\
1
\end{bmatrix}
$$

对应约束为：

$$
\mathbf{w}^{H}
\mathbf{a}_{1}
=
1
$$

$$
\mathbf{w}^{H}
\mathbf{a}_{2}
=
1
$$

$$
\mathbf{w}^{H}
\mathbf{a}_{3}
=
1
$$

---

## 6. LCMV 优化问题

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

它的含义是：

> 在满足所有指定方向响应的前提下，使阵列输出功率最小。

因为受保护方向不能被压低，所以算法只能通过调整其他空间方向的响应来降低：

- 干扰功率；
- 噪声功率；
- 非目标方向的输出功率。

---

## 7. LCMV 的最优权值

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

为了理解这个公式，可以分成三步。

第一步：

$$
\mathbf{U}
=
\mathbf{R}^{-1}
\mathbf{C}
$$

第二步：

$$
\mathbf{G}
=
\mathbf{C}^{H}
\mathbf{U}
$$

也就是：

$$
\mathbf{G}
=
\mathbf{C}^{H}
\mathbf{R}^{-1}
\mathbf{C}
$$

第三步：

$$
\mathbf{w}_{\mathrm{LCMV}}
=
\mathbf{U}
\mathbf{G}^{-1}
\mathbf{f}
$$

---

## 8. 公式各部分的含义

### 8.1 协方差加权

$$
\mathbf{R}^{-1}
\mathbf{C}
$$

这一步根据当前接收环境中的干扰、噪声和空间相关性，对约束方向进行自适应调整。

### 8.2 约束之间的耦合

$$
\mathbf{C}^{H}
\mathbf{R}^{-1}
\mathbf{C}
$$

这个矩阵描述不同约束方向之间的耦合关系。

如果两个方向非常接近，那么对应的两个导向矢量也会很接近，此时该矩阵可能接近奇异，导致：

- 权值变得很大；
- 算法对误差很敏感；
- 数值稳定性下降。

### 8.3 期望响应

$$
\mathbf{f}
$$

期望响应向量决定每个受约束方向的输出。

例如：

$$
\mathbf{f}
=
\begin{bmatrix}
1 \\
1 \\
1
\end{bmatrix}
$$

表示三个方向都保持单位增益。

例如：

$$
\mathbf{f}
=
\begin{bmatrix}
1 \\
0 \\
0
\end{bmatrix}
$$

表示：

- 第一个方向保持单位增益；
- 第二个方向形成零陷；
- 第三个方向形成零陷。

---

## 9. MVDR 是 LCMV 的特殊情况

如果只有一个约束，则：

$$
\mathbf{C}
=
\mathbf{a}_{0}
$$

并且：

$$
\mathbf{f}
=
1
$$

代入 LCMV 权值：

$$
\mathbf{w}
=
\mathbf{R}^{-1}
\mathbf{a}_{0}
\left(
\mathbf{a}_{0}^{H}
\mathbf{R}^{-1}
\mathbf{a}_{0}
\right)^{-1}
$$

可以写成：

$$
\mathbf{w}
=
\frac{
\mathbf{R}^{-1}
\mathbf{a}_{0}
}{
\mathbf{a}_{0}^{H}
\mathbf{R}^{-1}
\mathbf{a}_{0}
}
$$

这就是 MVDR 权值。

因此，MVDR 可以看作单约束情况下的 LCMV。

---

## 10. 使用 LCMV 显式形成零陷

假设：

- 目标方向为 `0°`；
- 第一个干扰方向为 `-30°`；
- 第二个干扰方向为 `40°`。

构造约束矩阵：

$$
\mathbf{C}
=
\begin{bmatrix}
\mathbf{a}(0^\circ) &
\mathbf{a}(-30^\circ) &
\mathbf{a}(40^\circ)
\end{bmatrix}
$$

设置期望响应：

$$
\mathbf{f}
=
\begin{bmatrix}
1 \\
0 \\
0
\end{bmatrix}
$$

对应约束为：

$$
\mathbf{w}^{H}
\mathbf{a}(0^\circ)
=
1
$$

$$
\mathbf{w}^{H}
\mathbf{a}(-30^\circ)
=
0
$$

$$
\mathbf{w}^{H}
\mathbf{a}(40^\circ)
=
0
$$

含义是：

- 目标方向保持单位增益；
- `-30°` 方向形成零陷；
- `40°` 方向形成零陷。

---

## 11. MVDR 自动零陷与 LCMV 指定零陷

| 方法            | 是否需要提前知道干扰方向 | 零陷产生方式             |
| --------------- | -----------------------: | ------------------------ |
| MVDR            |                   不需要 | 根据协方差矩阵自动形成   |
| LCMV 多目标保护 |               不一定需要 | 利用剩余自由度自适应形成 |
| LCMV 显式零陷   |                     需要 | 通过约束强制形成         |

实际中可以组合使用：

1. 对卫星方向设置单位增益；
2. 对已知干扰方向设置零响应；
3. 对其他未知干扰依靠最小方差准则自动抑制。

---

## 12. 阵列自由度

假设阵列有 `N` 个阵元。

权向量为：

$$
\mathbf{w}
\in
\mathbb{C}^{N}
$$

如果设置了 `Q` 个独立约束，剩余自适应自由度大约为：

$$
D_{\mathrm{remain}}
\approx
N-Q
$$

其中：

- `N` 是阵元数；
- `Q` 是独立约束数；
- `Dremain` 是剩余自适应自由度。

通常需要：

$$
Q<N
$$

这样才能保留一定能力，用于抑制其他未知干扰。

---

## 13. 当约束数量等于阵元数

如果：

$$
Q=N
$$

则：

$$
D_{\mathrm{remain}}
=
0
$$

此时所有自由度都被约束占用。

虽然可能仍然存在满足约束的权值，但基本没有剩余自由度继续自适应抑制其他干扰。

---

## 14. 当约束数量大于阵元数

如果：

$$
Q>N
$$

并且这些约束彼此独立，通常无法找到满足全部约束的精确解。

因此，独立约束数量通常不能超过阵元数。

实际系统为了保留抗干扰能力，约束数量一般还要明显小于阵元数。

---

## 15. 四阵元阵列的限制

假设：

$$
N=4
$$

如果希望同时严格保护四颗卫星，则：

$$
Q=4
$$

剩余自由度为：

$$
D_{\mathrm{remain}}
=
4-4
=
0
$$

此时无法继续对未知干扰进行自适应抑制。

如果还要对一个干扰方向设置零陷，则需要：

$$
Q=5
$$

但此时：

$$
Q>N
$$

通常无法实现。

因此，四阵元阵列不能使用一套公共权值，同时严格保护很多颗卫星并形成多个零陷。

---

## 16. 每颗卫星单独形成权值

GNSS 中更合理的方法之一是：

> 每颗卫星单独生成一套权值。

对第 `i` 颗卫星，可以使用：

$$
\mathbf{w}_{i}
=
\frac{
\mathbf{R}^{-1}
\mathbf{a}_{i}
}{
\mathbf{a}_{i}^{H}
\mathbf{R}^{-1}
\mathbf{a}_{i}
}
$$

对应约束为：

$$
\mathbf{w}_{i}^{H}
\mathbf{a}_{i}
=
1
$$

如果有 `N` 个阵元，那么每颗卫星单独处理时，剩余自由度大约为：

$$
N-1
$$

同一份阵列数据可以生成多路输出：

$$
y_{1}[k]
=
\mathbf{w}_{1}^{H}
\mathbf{x}[k]
$$

$$
y_{2}[k]
=
\mathbf{w}_{2}^{H}
\mathbf{x}[k]
$$

$$
y_{i}[k]
=
\mathbf{w}_{i}^{H}
\mathbf{x}[k]
$$

这样每颗卫星都有自己的空间滤波结果。

---

## 17. 每颗卫星独立权值的代价

这种方法的代价是计算量增加。

需要：

- 为每颗卫星计算独立权值；
- 进行多路复数乘加；
- 维护多套导向矢量；
- 更新卫星方向；
- 更新设备姿态；
- 维持多路捕获和跟踪通道。

因此，它更适合：

- 后相关处理；
- 通道化接收机；
- 软件接收机；
- FPGA、DSP 或 GPU 并行实现。

---

## 18. 前相关公共权值

如果在相关处理之前，只输出一路信号：

$$
y[k]
=
\mathbf{w}^{H}
\mathbf{x}[k]
$$

那么所有卫星必须共用同一套权值。

此时通常无法严格保护所有卫星方向。

常见折中方法包括：

- 只保护少量关键卫星；
- 优先保护高仰角卫星；
- 优先保护几何分布较好的卫星；
- 对天空某个区域保持较高增益；
- 优先抑制最强干扰；
- 接受部分卫星增益下降。

---

## 19. 多点约束形成宽保护区

如果目标方向存在误差，可以在目标附近设置多个约束点。

例如：

$$
\mathbf{w}^{H}
\mathbf{a}(\theta_{0}-\Delta)
=
1
$$

$$
\mathbf{w}^{H}
\mathbf{a}(\theta_{0})
=
1
$$

$$
\mathbf{w}^{H}
\mathbf{a}(\theta_{0}+\Delta)
=
1
$$

这样可以在目标方向附近形成较宽的保护区域。

它能够提高对以下误差的鲁棒性：

- 卫星方向误差；
- 阵列姿态误差；
- 阵元位置误差；
- 通道相位误差；
- 阵列流形误差。

代价是每增加一个约束点，就会消耗一个额外自由度。

---

## 20. 导数约束

除了多个离散角度点，还可以对导向矢量的一阶导数设置约束。

第一个约束为：

$$
\mathbf{w}^{H}
\mathbf{a}(\theta_{0})
=
1
$$

第二个约束为：

$$
\mathbf{w}^{H}
\left.
\frac{
\partial
\mathbf{a}(\theta)
}{
\partial
\theta
}
\right|_{\theta=\theta_{0}}
=
0
$$

第二个约束表示目标方向附近的波束响应变化率为零。

这样可以让主瓣顶部更加平坦，提高对小范围方向失配的稳健性。

---

## 21. LCMV 的推导

目标函数为：

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

构造拉格朗日函数：

$$
\mathcal{L}
=
\mathbf{w}^{H}
\mathbf{R}
\mathbf{w}
+
\boldsymbol{\lambda}^{H}
\left(
\mathbf{C}^{H}
\mathbf{w}
-
\mathbf{f}
\right)
+
\left(
\mathbf{w}^{H}
\mathbf{C}
-
\mathbf{f}^{H}
\right)
\boldsymbol{\lambda}
$$

对权向量求驻点，得到：

$$
\mathbf{R}
\mathbf{w}
+
\mathbf{C}
\boldsymbol{\lambda}
=
\mathbf{0}
$$

因此：

$$
\mathbf{w}
=
-
\mathbf{R}^{-1}
\mathbf{C}
\boldsymbol{\lambda}
$$

代入约束条件：

$$
\mathbf{C}^{H}
\mathbf{w}
=
\mathbf{f}
$$

得到：

$$
-
\mathbf{C}^{H}
\mathbf{R}^{-1}
\mathbf{C}
\boldsymbol{\lambda}
=
\mathbf{f}
$$

因此：

$$
\boldsymbol{\lambda}
=
-
\left(
\mathbf{C}^{H}
\mathbf{R}^{-1}
\mathbf{C}
\right)^{-1}
\mathbf{f}
$$

再代回权向量：

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

---

## 22. Python 中的稳定实现

理论公式中包含：

$$
\mathbf{R}^{-1}
\mathbf{C}
$$

代码中不建议直接计算矩阵逆：

```python
R_inv = np.linalg.inv(R)
```

更稳定的方法是求解线性方程：

```python
U = np.linalg.solve(R_loaded, C)
```

这里求解的是：

$$
\mathbf{R}_{\mathrm{loaded}}
\mathbf{U}
=
\mathbf{C}
$$

因此：

$$
\mathbf{U}
=
\mathbf{R}_{\mathrm{loaded}}^{-1}
\mathbf{C}
$$

接着计算：

```python
middle = C.conj().T @ U
```

最后计算权值：

```python
w_lcmv = U @ np.linalg.solve(middle, f)
```

完整写法如下：

```python
U = np.linalg.solve(
    R_loaded,
    C
)

middle = C.conj().T @ U

w_lcmv = U @ np.linalg.solve(
    middle,
    f
)
```

---

## 23. 目标保护与两个零陷的代码示例

假设：

- 目标方向为 `0°`；
- 第一个干扰方向为 `-30°`；
- 第二个干扰方向为 `40°`。

构造导向矢量：

```python
a_target = steering_vector(0.0)

a_jammer_1 = steering_vector(-30.0)

a_jammer_2 = steering_vector(40.0)
```

构造约束矩阵：

```python
C = np.hstack([
    a_target,
    a_jammer_1,
    a_jammer_2
])
```

设置期望响应：

```python
f = np.array([
    [1.0],
    [0.0],
    [0.0]
], dtype=complex)
```

计算 LCMV 权值：

```python
U = np.linalg.solve(
    R_loaded,
    C
)

middle = C.conj().T @ U

w_lcmv = U @ np.linalg.solve(
    middle,
    f
)
```

检查约束：

```python
responses = C.conj().T @ w_lcmv

print(responses)
```

理论输出应接近：

```text
[[1 + 0j]
 [0 + 0j]
 [0 + 0j]]
```

实际计算中可能出现很小的数值误差，例如：

```text
[[ 1.000000 + 0.000000j]
 [ 0.000001 - 0.000002j]
 [-0.000001 + 0.000001j]]
```

这仍然可以认为约束已经满足。

---

## 24. LCMV 与对角加载

实际中通常使用样本协方差矩阵：

$$
\hat{\mathbf{R}}
=
\frac{1}{K}
\mathbf{X}
\mathbf{X}^{H}
$$

当 Snapshot 数较少或协方差矩阵病态时，可以进行对角加载：

$$
\mathbf{R}_{\mathrm{DL}}
=
\hat{\mathbf{R}}
+
\delta
\mathbf{I}
$$

然后使用：

$$
\mathbf{w}_{\mathrm{LCMV}}
=
\mathbf{R}_{\mathrm{DL}}^{-1}
\mathbf{C}
\left(
\mathbf{C}^{H}
\mathbf{R}_{\mathrm{DL}}^{-1}
\mathbf{C}
\right)^{-1}
\mathbf{f}
$$

Python 示例：

```python
loading = (
    1e-3
    * np.trace(R_hat).real
    / N
)

R_loaded = (
    R_hat
    + loading * np.eye(N)
)
```

对角加载的作用包括：

- 改善矩阵条件数；
- 防止矩阵接近奇异；
- 降低有限样本误差；
- 提高导向矢量失配下的稳定性。

---

## 25. 对角加载过大的影响

如果加载量很大：

$$
\mathbf{R}_{\mathrm{DL}}
\approx
\delta
\mathbf{I}
$$

此时协方差矩阵中的干扰结构会被削弱。

算法会逐渐接近普通约束波束形成。

因此：

- 加载较小，零陷更深，但可能不稳定；
- 加载较大，更稳健，但零陷可能变浅；
- 加载参数需要折中选择。

---

## 26. 约束矩阵病态问题

如果两个约束方向非常接近，例如：

$$
\theta_{1}
=
10^\circ
$$

$$
\theta_{2}
=
10.2^\circ
$$

那么：

$$
\mathbf{a}(\theta_{1})
\approx
\mathbf{a}(\theta_{2})
$$

约束矩阵的两列会近似线性相关。

此时：

$$
\mathbf{C}^{H}
\mathbf{R}^{-1}
\mathbf{C}
$$

可能接近奇异。

可能出现：

- 权值幅度异常大；
- 白噪声增益升高；
- 波束图对误差非常敏感；
- 数值计算不稳定；
- 小幅阵列误差导致明显失真。

因此，约束点不能无限密集。

---

## 27. GNSS 四阵元自由度分析

假设：

$$
N=4
$$

希望保护三颗重要卫星：

$$
Q=3
$$

剩余自由度为：

$$
D_{\mathrm{remain}}
=
4-3
=
1
$$

理论上只剩一个独立自适应维度。

如果再显式加入一个干扰零陷：

$$
Q=4
$$

则：

$$
D_{\mathrm{remain}}
=
4-4
=
0
$$

此时所有自由度都被约束占用。

---

## 28. 为什么 CRPA 常见四阵元或七阵元

CRPA 的英文全称为：

**Controlled Reception Pattern Antenna**

中文可以译为：

**受控接收方向图天线**

对于 `N` 阵元阵列，在保留一个主约束后，理想剩余自由度约为：

$$
N-1
$$

因此：

| 阵元数 | 保留一个主约束后的理论剩余自由度 |
| -----: | -------------------------------: |
|      2 |                                1 |
|      3 |                                2 |
|      4 |                                3 |
|      7 |                                6 |

四阵元阵列理论上可提供约三个额外空间自由度。

七阵元阵列理论上可提供约六个额外空间自由度。

但这只是理想窄带条件下的上限。

---

## 29. 一个干扰源不一定只占一个自由度

如果干扰是：

- 单一方向；
- 窄带；
- 稳定平面波；
- 无明显多径；

那么它的空间协方差通常接近秩 1。

但如果干扰具有以下特征：

- 宽带；
- 多径；
- 多个散射方向；
- 多发射天线；
- 快速移动；
- 极化变化；
- 快速时变；

它可能占据多个空间或时空维度。

因此，干扰源数量不一定等于干扰子空间维数。

阵列真正需要处理的是独立干扰模式的数量。

---

## 30. 前相关 LCMV

前相关处理中，先对原始阵列数据进行空间合并：

$$
y[k]
=
\mathbf{w}^{H}
\mathbf{x}[k]
$$

然后再进入 GNSS 捕获和跟踪模块。

优点：

- 后端结构简单；
- 只输出一路数据；
- 容易接入传统 GNSS 接收机。

缺点：

- 所有卫星共用同一套权值；
- 很难严格保护大量卫星方向；
- 部分卫星可能受到衰减。

---

## 31. 后相关 LCMV

后相关处理中，每个阵元分别完成码相关。

对于第 `i` 颗卫星，形成多阵元相关输出：

$$
\mathbf{z}_{i}[k]
=
\begin{bmatrix}
z_{1,i}[k] \\
z_{2,i}[k] \\
\vdots \\
z_{N,i}[k]
\end{bmatrix}
$$

然后针对第 `i` 颗卫星使用独立权值：

$$
y_{i}[k]
=
\mathbf{w}_{i}^{H}
\mathbf{z}_{i}[k]
$$

优点：

- 每颗卫星可以使用独立空间权值；
- 更适合多卫星、多方向场景；
- 卫星保护更灵活。

缺点：

- 运算量更大；
- 接收机结构更复杂；
- 每个阵元需要保留独立相关通道。

---

## 32. LCMV 的核心价值

MVDR 主要解决：

> 保持一个目标方向，同时根据协方差矩阵抑制其他方向。

LCMV 可以进一步控制：

- 哪些方向保持单位增益；
- 哪些方向形成零陷；
- 哪些方向保持指定幅度；
- 哪些区域形成宽保护区；
- 哪些方向附近提高鲁棒性。

LCMV 是从单目标自适应波束形成，走向多方向可控波束形成的重要方法。

---

## 33. 本节核心公式

约束矩阵：

$$
\mathbf{C}
=
\begin{bmatrix}
\mathbf{a}_{1} &
\mathbf{a}_{2} &
\cdots &
\mathbf{a}_{Q}
\end{bmatrix}
$$

期望响应向量：

$$
\mathbf{f}
=
\begin{bmatrix}
f_{1} \\
f_{2} \\
\vdots \\
f_{Q}
\end{bmatrix}
$$

目标函数：

$$
\min_{\mathbf{w}}
\mathbf{w}^{H}
\mathbf{R}
\mathbf{w}
$$

约束条件：

$$
\mathbf{C}^{H}
\mathbf{w}
=
\mathbf{f}
$$

最优权值：

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

剩余自由度：

$$
D_{\mathrm{remain}}
\approx
N-Q
$$

---

## 34. 与前面内容的关系

```text
阵列接收数据
    │
    ▼
协方差矩阵
    │
    ├── MUSIC
    │     利用噪声子空间估计方向
    │
    ├── ESPRIT
    │     利用信号子空间旋转关系估计方向
    │
    ├── 空间平滑
    │     处理相干信号导致的秩亏
    │
    ├── MVDR
    │     单约束最小方差波束形成
    │
    └── LCMV
          多约束最小方差波束形成
```

MUSIC 和 ESPRIT 主要用于：

> 估计信号方向。

空间平滑主要用于：

> 处理相干信号导致的秩亏。

MVDR 和 LCMV 主要用于：

> 根据目标方向和协方差矩阵进行抗干扰波束形成。

---

## 35. 下一阶段

下一步学习：

**GSC：广义旁瓣相消器**

GSC 不会改变 LCMV 的目标，而是把 LCMV 分解为两个部分：

```text
固定保护支路
    +
自适应干扰相消支路
```

GSC 可以更直观地解释：

- 目标信号为什么不会被自适应支路消除；
- 阻塞矩阵如何去除目标成分；
- 自适应滤波器如何估计干扰；
- LCMV 如何转化为无约束自适应问题。
