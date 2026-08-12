---
tags:
  - 阵列信号处理
  - 波束形成
  - LCMV
  - 多约束
  - 抗干扰
aliases:
  - LCMV波束形成
  - 线性约束最小方差
created: 2026-07-19
---

# 第09章：LCMV 与 GNSS 多星自由度

> LCMV（Linearly Constrained Minimum Variance）是 MVDR 的多约束扩展。它可以同时保护多个方向、显式设置零陷、形成宽保护区，是 GNSS 多卫星抗干扰场景中的关键方法。

前置阅读：[[07_DOA功率与SINR估计]]、[[08_MVDR波束形成]]

---

## 1. 回顾 MVDR

MVDR 的目标是：在保证目标方向信号无失真通过的前提下，使阵列输出总功率最小。

$$
\min_{\mathbf{w}} \mathbf{w}^{H}\mathbf{R}\mathbf{w}
\quad \text{s.t.} \quad
\mathbf{w}^{H}\mathbf{a}_{0} = 1
$$

MVDR 的最优权值为：

$$
\mathbf{w}_{\mathrm{MVDR}}
=
\frac{\mathbf{R}^{-1}\mathbf{a}_{0}}{\mathbf{a}_{0}^{H}\mathbf{R}^{-1}\mathbf{a}_{0}}
$$

---

## 2. 为什么需要 LCMV

MVDR 只有一个约束。但 GNSS 中往往存在多颗不同方向的卫星，因此可能需要同时满足多个条件：

- 卫星 1 方向保持单位增益
- 卫星 2 方向保持单位增益
- 干扰方向形成零陷
- 某个方向保持指定增益

这就需要用 LCMV 把单约束扩展为多约束。

---

## 3. 约束矩阵与期望响应

假设希望控制 $Q$ 个方向，每个方向对应一个导向矢量 $\mathbf{a}_1, \mathbf{a}_2, \ldots, \mathbf{a}_Q$。

**约束矩阵：**

$$
\mathbf{C}
=
\begin{bmatrix}
\mathbf{a}_{1} & \mathbf{a}_{2} & \cdots & \mathbf{a}_{Q}
\end{bmatrix}
\in \mathbb{C}^{N \times Q}
$$

**期望响应向量：**

$$
\mathbf{f}
=
\begin{bmatrix}
f_{1} \\ f_{2} \\ \vdots \\ f_{Q}
\end{bmatrix}
$$

多方向约束统一写成：

$$
\boxed{
\mathbf{C}^{H}\mathbf{w} = \mathbf{f}
}
$$

- $\mathbf{C}$ 决定约束哪些方向
- $\mathbf{f}$ 决定这些方向分别得到什么响应

---

## 4. 多颗卫星单位增益约束

假设同时保护三颗卫星，让三个方向都保持单位增益：

$$
\mathbf{C} = \begin{bmatrix} \mathbf{a}_{1} & \mathbf{a}_{2} & \mathbf{a}_{3} \end{bmatrix},\quad
\mathbf{f} = \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}
$$

对应约束为：

$$
\mathbf{w}^{H}\mathbf{a}_{1} = 1,\quad
\mathbf{w}^{H}\mathbf{a}_{2} = 1,\quad
\mathbf{w}^{H}\mathbf{a}_{3} = 1
$$

---

## 5. LCMV 优化问题

$$
\boxed{
\min_{\mathbf{w}} \mathbf{w}^{H}\mathbf{R}\mathbf{w}
\quad \text{s.t.} \quad
\mathbf{C}^{H}\mathbf{w} = \mathbf{f}
}
$$

> 在满足所有指定方向响应的前提下，使阵列输出功率最小。受保护方向不能被压低，算法只能通过调整其他空间方向的响应来降低干扰和噪声功率。

MVDR 是 LCMV 的单约束特例：$\mathbf{C} = \mathbf{a}_0,\ \mathbf{f} = 1$。

---

## 6. LCMV 的最优权值

$$
\boxed{
\mathbf{w}_{\mathrm{LCMV}}
=
\mathbf{R}^{-1}\mathbf{C}
\left(
\mathbf{C}^{H}\mathbf{R}^{-1}\mathbf{C}
\right)^{-1}
\mathbf{f}
}
$$

分为三步理解：

**第一步：** $\mathbf{U} = \mathbf{R}^{-1}\mathbf{C}$ — 根据接收环境中的干扰、噪声和空间相关性对约束方向进行自适应调整

**第二步：** $\mathbf{G} = \mathbf{C}^{H}\mathbf{R}^{-1}\mathbf{C}$ — 描述不同约束方向之间的耦合关系。两个方向很接近时该矩阵可能接近奇异

**第三步：** $\mathbf{w}_{\mathrm{LCMV}} = \mathbf{U}\mathbf{G}^{-1}\mathbf{f}$ — 施加期望响应

---

## 7. 使用 LCMV 显式形成零陷

假设目标方向为 $0^\circ$，干扰方向为 $-30^\circ$ 和 $40^\circ$：

$$
\mathbf{C}
=
\begin{bmatrix}
\mathbf{a}(0^\circ) & \mathbf{a}(-30^\circ) & \mathbf{a}(40^\circ)
\end{bmatrix}
,\quad
\mathbf{f}
=
\begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}
$$

对应约束：

$$
\mathbf{w}^{H}\mathbf{a}(0^\circ) = 1,\quad
\mathbf{w}^{H}\mathbf{a}(-30^\circ) = 0,\quad
\mathbf{w}^{H}\mathbf{a}(40^\circ) = 0
$$

含义：目标方向单位增益，两个干扰方向零陷。

---

## 8. MVDR 自动零陷与 LCMV 指定零陷对比

| 方法 | 是否需要提前知道干扰方向 | 零陷产生方式 |
|------|:----------------------:|-------------|
| MVDR | 不需要 | 根据协方差矩阵自动形成 |
| LCMV 多目标保护 | 不一定需要 | 利用剩余自由度自适应形成 |
| LCMV 显式零陷 | 需要 | 通过约束强制形成 |

实际中可以组合使用：对卫星方向设置单位增益，对已知干扰方向设置零响应，对其他未知干扰依靠最小方差准则自动抑制。

---

## 9. 阵列自由度

$N$ 阵元复权向量具有 $N$ 个复数维度。严格决定自由度消耗的不是约束列数 $Q$，而是约束矩阵的秩：

$$
\boxed{
D_{\mathrm{remain}}
=
N-r,
\qquad
r=\operatorname{rank}(\mathbf C)}
$$

只有 $Q$ 个约束线性独立时，才有 $r=Q$ 和 $D_{\mathrm{remain}}=N-Q$。

- 约束越多，可用于自适应抑制干扰的自由度越少
- 约束数量接近阵元数时，权值几乎被完全确定
- 独立约束数量超过阵元数时，通常无法同时满足全部约束

| 阵元数 | 保留一个主约束后的理论剩余自由度 |
|:------:|:-------------------------------:|
| 2 | 1 |
| 3 | 2 |
| 4 | 3 |
| 7 | 6 |

---

## 10. 四阵元阵列的限制

$N=4$ 时，若同时严格保护四颗卫星（$Q=4$），$D_{\mathrm{remain}} = 0$，无法继续对未知干扰进行自适应抑制。若还要设置一个干扰零陷（$Q=5$），则 $Q > N$，通常无法实现。

> 因此，四阵元阵列**不能**使用一套公共权值同时严格保护很多颗卫星并形成多个零陷。

---

## 11. 每颗卫星单独形成权值

GNSS 中更合理的方法：**每颗卫星单独生成一套权值。**

对第 $i$ 颗卫星使用 MVDR：

$$
\mathbf{w}_{i}
=
\frac{\mathbf{R}^{-1}\mathbf{a}_{i}}{\mathbf{a}_{i}^{H}\mathbf{R}^{-1}\mathbf{a}_{i}}
$$

同一份阵列数据生成多路输出：

$$
y_{i}[k] = \mathbf{w}_{i}^{H}\mathbf{x}[k]
$$

每颗卫星剩余自由度约为 $N-1$。代价是计算量增加，需要为每颗卫星计算独立权值、进行多路复数乘加。

---

## 12. 前相关公共权值的折中

如果只输出一路信号 $y[k] = \mathbf{w}^{H}\mathbf{x}[k]$，所有卫星共用同一套权值。常见折中方法：

- 只保护少量关键卫星
- 优先保护高仰角卫星
- 优先保护几何分布较好的卫星
- 对天空某个区域保持较高增益
- 优先抑制最强干扰
- 接受部分卫星增益下降

---

## 13. 多点约束形成宽保护区

如果目标方向存在误差，可以在目标附近设置多个约束点：

$$
\mathbf{w}^{H}\mathbf{a}(\theta_{0}-\Delta) = 1,\quad
\mathbf{w}^{H}\mathbf{a}(\theta_{0}) = 1,\quad
\mathbf{w}^{H}\mathbf{a}(\theta_{0}+\Delta) = 1
$$

可以提高对卫星方向误差、阵列姿态误差、阵元位置误差的鲁棒性。代价是每增加一个约束点消耗一个额外自由度。

---

## 14. 导数约束

对导向矢量的一阶导数设置约束：

$$
\mathbf{w}^{H}\mathbf{a}(\theta_{0}) = 1,\quad
\mathbf{w}^{H}\left.\frac{\partial\mathbf{a}(\theta)}{\partial\theta}\right|_{\theta=\theta_{0}} = 0
$$

第二个约束表示目标方向附近的波束响应变化率为零，让主瓣顶部更加平坦，提高对小范围方向失配的稳健性。

---

## 15. LCMV 的推导

构造拉格朗日函数：

$$
\mathcal{L}
=
\mathbf{w}^{H}\mathbf{R}\mathbf{w}
+
\boldsymbol{\lambda}^{H}(\mathbf{C}^{H}\mathbf{w} - \mathbf{f})
+
(\mathbf{w}^{H}\mathbf{C} - \mathbf{f}^{H})\boldsymbol{\lambda}
$$

对权向量求驻点，代入约束条件，得到：

$$
\mathbf{w}_{\mathrm{LCMV}}
=
\mathbf{R}^{-1}\mathbf{C}
\left(\mathbf{C}^{H}\mathbf{R}^{-1}\mathbf{C}\right)^{-1}
\mathbf{f}
$$

---

## 16. Python 稳定实现

不建议直接计算矩阵逆，应求解线性方程：

```python
# 第一步：求解 R · U = C
U = np.linalg.solve(R_loaded, C)

# 第二步：计算中间矩阵
middle = C.conj().T @ U

# 第三步：求解 middle · q = f 并得到最终权值
w_lcmv = U @ np.linalg.solve(middle, f)

# 检查约束
responses = C.conj().T @ w_lcmv
print(responses)  # 应接近 f
```

---

## 17. 对角加载

当 Snapshot 数较少或协方差矩阵病态时：

$$
\mathbf{R}_{\mathrm{DL}} = \hat{\mathbf{R}} + \delta\mathbf{I}
$$

```python
loading = 1e-3 * np.trace(R_hat).real / N
R_loaded = R_hat + loading * np.eye(N)
```

- 加载较小：零陷更深，但可能不稳定
- 加载较大：更稳健，但零陷可能变浅
- 加载很大：趋近普通约束波束形成

---

## 18. 约束矩阵病态问题

如果两个约束方向非常接近（如 $10^\circ$ 和 $10.2^\circ$），对应导向矢量近似线性相关，$\mathbf{C}^{H}\mathbf{R}^{-1}\mathbf{C}$ 可能接近奇异，导致权值幅度异常大、输出白噪声增加、WNG 下降，并且对小误差非常敏感。

> 因此，**约束点不能无限密集。**

### 18.1 用 SVD 判断有效约束秩

对约束矩阵做：

$$
\mathbf C=\mathbf U\boldsymbol\Sigma\mathbf V^H
$$

根据奇异值相对阈值确定数值秩 $r$。这能够识别重复、近重复或无法由当前阵列分开的约束，而不是机械地把每一列都当成一个独立自由度消耗。

如果约束病态，应优先考虑：

- 合并物理意义重复的约束；
- 减少过密的多点保护采样；
- 用导数约束代替大量近邻点；
- 使用伪逆和明确的数值秩阈值；
- 增加阵元，而不是只通过更强正则化掩盖维数不足。

---

## 19. 一个干扰源不一定只占一个自由度

若干扰是单方向、窄带、稳定平面波、无明显多径，其空间协方差通常接近秩 1。但如果干扰具有宽带、多径、多个散射方向、极化变化等特征，它可能占据多个空间或时空维度。

> 阵列真正需要处理的是**独立干扰模式的数量**，而不只是干扰源数量。

---

## 20. 前相关与后相关 LCMV

| | 前相关 | 后相关 |
|------|--------|--------|
| **方式** | 先空间合并，再进入捕获跟踪 | 每个阵元分别完成码相关，再对各卫星用独立权值 |
| **优点** | 后端结构简单；只输出一路数据 | 每颗卫星独立空间权值；保护更灵活 |
| **缺点** | 所有卫星共用一套权值；部分卫星可能衰减 | 运算量大；接收机结构更复杂 |

### 20.1 卫星方向必须转换到阵列坐标系

卫星方向通常由星历和接收机位置在导航坐标系中给出，而导向矢量需要阵列坐标系中的来波方向。若姿态旋转矩阵为 $\mathbf R_{\mathrm{att}}$：

$$
\mathbf u_{\mathrm{array}}
=
\mathbf R_{\mathrm{att}}^T\mathbf u_{\mathrm{nav}}
$$

姿态误差、安装矩阵误差和时间不同步都会转化为导向矢量失配。多星 LCMV 不是只需要卫星方位角列表，它依赖星历、接收机位置、姿态、安装关系和实测阵列流形共同正确。

---

## 21. LCMV 的核心价值

MVDR 主要解决：保持一个目标方向，根据协方差矩阵抑制其他方向。

LCMV 进一步控制：

- 哪些方向保持单位增益
- 哪些方向形成零陷
- 哪些方向保持指定幅度
- 哪些区域形成宽保护区
- 哪些方向附近提高鲁棒性

> LCMV 是从单目标自适应波束形成走向**多方向可控波束形成**的重要方法。

### 21.1 GNSS 阵列规模的直观解释

在只保留一个主约束的理想窄带场景中：

| 阵元数 | 理论剩余自由度 | 直观上限 |
|---:|---:|---|
| 2 | 1 | 只能进行一次独立空间调整 |
| 4 | 3 | 最多约三个理想秩 1 干扰模式 |
| 7 | 6 | 能处理更多独立模式或保留更多保护约束 |

这只是维数上限，不是性能承诺。多径、角度扩展、校准误差和约束病态都会降低有效能力。

### 21.2 实施检查项

设计完成后至少检查：

$$
\epsilon_c
=
\|\mathbf C^H\mathbf w-\mathbf f\|
$$

并记录：

- $\operatorname{rank}(\mathbf C)$ 与条件数；
- 剩余自由度 $N-r$；
- $\|\mathbf w\|$ 和 WNG；
- 各卫星方向及邻域增益；
- 已知干扰方向的响应；
- 姿态、通道幅相和角度误差敏感性。

---

## 22. 核心公式汇总

约束矩阵与期望响应：

$$
\mathbf{C}
=
\begin{bmatrix}
\mathbf{a}_{1} & \mathbf{a}_{2} & \cdots & \mathbf{a}_{Q}
\end{bmatrix},\quad
\mathbf{f}
=
\begin{bmatrix}
f_{1} \\ f_{2} \\ \vdots \\ f_{Q}
\end{bmatrix}
$$

优化问题：

$$
\min_{\mathbf{w}} \mathbf{w}^{H}\mathbf{R}\mathbf{w}
\quad \text{s.t.} \quad
\mathbf{C}^{H}\mathbf{w} = \mathbf{f}
$$

最优权值：

$$
\mathbf{w}_{\mathrm{LCMV}}
=
\mathbf{R}^{-1}\mathbf{C}
\left(\mathbf{C}^{H}\mathbf{R}^{-1}\mathbf{C}\right)^{-1}
\mathbf{f}
$$

剩余自由度（$r=\operatorname{rank}(\mathbf C)$）：

$$
D_{\mathrm{remain}} = N-r
$$

仅当 $Q$ 个约束线性独立时，才可写成 $N-Q$。

---

## 23. 从测向到抗干扰的逻辑链

```text
阵列接收数据
    │
    ▼
样本协方差矩阵
    │
    ├──────────────────────────┐
    │                          │
    ▼                          ▼
特征分解与子空间            最小输出功率优化
    │                          │
    ▼                          ▼
MUSIC / ESPRIT              MVDR / LCMV
    │                          │
    ▼                          ▼
估计来波方向                形成波束和零陷
```

- MUSIC 和 ESPRIT 回答：**信号从什么方向来**
- MVDR 和 LCMV 回答：**已知需要保护的方向后，怎样组合阵元信号以抑制干扰**

---

## 24. 与前面内容的关系

```text
阵列接收数据
    │
    ▼
协方差矩阵
    │
    ├── MUSIC → 利用噪声子空间估计方向
    ├── ESPRIT → 利用信号子空间旋转关系估计方向
    ├── 空间平滑 → 处理相干信号导致的秩亏
    ├── MVDR → 单约束最小方差波束形成
    └── LCMV → 多约束最小方差波束形成
```

---

## 常见疑惑

| 疑惑 | 解答 |
|------|------|
| LCMV 是否必须知道干扰方向？ | 不一定。只给出保护约束时，LCMV 利用协方差自动抑制高能量方向 |
| LCMV 与 MVDR 的区别？ | MVDR 一个保护约束；LCMV 多个线性约束。MVDR 是 LCMV 的特例 |
| 约束越多是否越安全？ | 不一定。消耗更多自由度，可能使干扰抑制能力下降 |
| 四阵元能否保护四颗卫星并抑制干扰？ | 若 $Q=4=N$，剩余自由度为 0，无法自适应抑制未知干扰 |
| 多个保护方向很接近会怎样？ | 约束矩阵条件数变差，需要使用加载、约束合并或宽主瓣设计 |

---

## 下一步

LCMV 直接求解带约束优化问题。[[10_GSC与鲁棒阻塞矩阵|GSC 与鲁棒阻塞矩阵]] 将它分解为固定保护支路和自适应相消支路，更适合在线迭代实现。
