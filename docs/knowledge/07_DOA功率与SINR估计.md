# 第07章：DOA、功率与 SINR 估计

> 找到谱峰只是第一步。工程系统还需要知道干扰有多强、噪声底在哪里、波束形成后改善了多少。本章把方向、功率分解和性能指标连接起来。

前置阅读：[[05_信号源数量与干扰维数估计]]、[[06_DOA估计_MUSIC_ESPRIT与空间平滑]]

---

## 1. 多源协方差模型

$$
\mathbf x[k]
=
\sum_{p=1}^{P}\mathbf a(\theta_p)s_p[k]
+\mathbf n[k]
$$

一般协方差为：

$$
\mathbf R_x
=
\mathbf A\mathbf R_s\mathbf A^H
+\mathbf R_n
$$

若各源互不相关且噪声为空间白噪声：

$$
\boxed{
\mathbf R_x
=
\sum_{p=1}^{P}P_p\mathbf a_p\mathbf a_p^H
+\sigma_n^2\mathbf I}
$$

后续功率分解依赖的并不只是 DOA，还依赖“源互不相关”和“噪声模型正确”等假设。

---

## 2. 三种常用空间谱

### 2.1 MUSIC 伪谱

$$
\boxed{
P_{\mathrm{MUSIC}}(\theta)
=
\frac1{
\mathbf a^H(\theta)
\widehat{\mathbf U}_n
\widehat{\mathbf U}_n^H
\mathbf a(\theta)}}
$$

谱峰位置用于估计 DOA。峰值高度不等于源的真实物理功率，因此 MUSIC 适合“找方向”，不应直接拿谱峰高度计算干扰功率。

### 2.2 Capon 空间谱

$$
\boxed{
P_{\mathrm{Capon}}(\theta)
=
\frac1{
\mathbf a^H(\theta)
\widehat{\mathbf R}^{-1}
\mathbf a(\theta)}}
$$

它表示在方向 $\theta$ 保持单位响应时的最小输出功率。Capon 不需要显式划分信号/噪声子空间，但非常依赖协方差逆的稳定性，应与 [[03_有限快拍协方差估计与正则化]] 联合使用。

### 2.3 常规波束扫描

$$
\mathbf w_{\mathrm{CBF}}(\theta)
=
\frac{\mathbf a(\theta)}
{\mathbf a^H(\theta)\mathbf a(\theta)}
$$

$$
P_{\mathrm{CBF}}(\theta)
=
\mathbf w_{\mathrm{CBF}}^H(\theta)
\widehat{\mathbf R}
\mathbf w_{\mathrm{CBF}}(\theta)
$$

它分辨率较低，但模型简单、解释直观，适合作为高分辨算法的基线和异常回退方法。

---

## 3. 已知 DOA 后的多源功率估计

根据估计方向构造：

$$
\widehat{\mathbf A}
=
[\mathbf a(\widehat\theta_1),\ldots,
\mathbf a(\widehat\theta_P)]
$$

若源互不相关：

$$
\mathbf R_s=\operatorname{diag}(P_1,\ldots,P_P)
$$

则：

$$
\widehat{\mathbf R}
\approx
\sum_{p=1}^{P}P_p\mathbf a_p\mathbf a_p^H
+\sigma_n^2\mathbf I
$$

向量化后：

$$
\operatorname{vec}(\widehat{\mathbf R})
\approx
\sum_{p=1}^{P}P_p
(\mathbf a_p^*\otimes\mathbf a_p)
+\sigma_n^2\operatorname{vec}(\mathbf I)
$$

这成为关于 $P_1,\ldots,P_P,\sigma_n^2$ 的线性最小二乘问题。物理功率应满足非负约束；若普通最小二乘得到明显负功率，通常说明 DOA、源数或协方差模型存在失配。

### 3.1 单源特例

若：

$$
\mathbf R
=
P_s\mathbf a\mathbf a^H
+\sigma_n^2\mathbf I
$$

且 $\|\mathbf a\|^2=N$：

$$
\mathbf a^H\mathbf R\mathbf a
=
P_sN^2+\sigma_n^2N
$$

已知噪声功率时：

$$
\boxed{
P_s=
\frac{\mathbf a^H\mathbf R\mathbf a-\sigma_n^2N}{N^2}}
$$

---

## 4. 噪声功率估计

在白噪声假设下，若有效源数为 $P$，可用最小 $N-P$ 个特征值平均：

$$
\boxed{
\widehat\sigma_n^2
=
\frac1{N-P}
\sum_{i=P+1}^{N}\widehat\lambda_i}
$$

以下情况会使估计产生偏差：

- 源数低估，信号特征值混入噪声组；
- 有色噪声使噪声特征值本来就不相等；
- 有限快拍使噪声特征值散开；
- 相干源改变信号协方差的有效秩；
- 加载后的特征值不能直接当作原始噪声功率。

---

## 5. 输入与输出 SINR

### 5.1 输入 SINR

若参考阵元的目标、干扰和噪声功率分别为 $P_s$、$P_j$、$P_n$：

$$
\boxed{
\mathrm{SINR}_{\mathrm{in}}
=
\frac{P_s}{P_j+P_n}}
$$

$$
\mathrm{SINR}_{\mathrm{in,dB}}
=
10\log_{10}\mathrm{SINR}_{\mathrm{in}}
$$

### 5.2 输出 SINR

令目标、干扰、噪声协方差分别为 $\mathbf R_s$、$\mathbf R_j$、$\mathbf R_n$。给定权值 $\mathbf w$：

$$
P_{s,\mathrm{out}}
=
\mathbf w^H\mathbf R_s\mathbf w
$$

$$
P_{j+n,\mathrm{out}}
=
\mathbf w^H(\mathbf R_j+\mathbf R_n)\mathbf w
$$

$$
\boxed{
\mathrm{SINR}_{\mathrm{out}}
=
\frac{\mathbf w^H\mathbf R_s\mathbf w}
{\mathbf w^H(\mathbf R_j+\mathbf R_n)\mathbf w}}
$$

若 $\mathbf R_s=P_s\mathbf a_0\mathbf a_0^H$，则：

$$
P_{s,\mathrm{out}}
=
P_s|\mathbf w^H\mathbf a_0|^2
$$

只有当真实导向矢量满足单位响应时，输出目标功率才等于参考功率 $P_s$。

---

## 6. 干扰抑制度与零陷深度

输出干扰功率：

$$
P_{j,\mathrm{out}}
=
\mathbf w^H\mathbf R_j\mathbf w
$$

可定义抑制度：

$$
\mathrm{Supp}_j
=
10\log_{10}
\frac{P_{j,\mathrm{in}}}{P_{j,\mathrm{out}}}
$$

单个方向的阵列增益为：

$$
G_j
=
20\log_{10}|\mathbf w^H\mathbf a(\theta_j)|
$$

零陷深度只是辅助指标，不能代替实际输出干扰功率，因为真实干扰可能：

- 具有角度扩展；
- 通过多径从多个方向到达；
- 与模型导向矢量失配；
- 占据不止一个空间模式。

---

## 7. DOA 误差、精度与分辨率

单次误差：

$$
\epsilon_\theta
=
\widehat\theta-\theta_{\mathrm{true}}
$$

多次试验的均方根误差：

$$
\mathrm{RMSE}_\theta
=
\sqrt{E[(\widehat\theta-\theta)^2]}
$$

应区分：

- **系统偏差：** 校准、坐标或模型错误造成稳定偏移；
- **随机方差：** 噪声和有限快拍造成估计抖动；
- **离群错误：** 谱峰配对错误或弱源完全漏检；
- **分辨率：** 区分两个近邻源的能力；
- **精度：** 单个源角度估计接近真值的程度。

高精度不代表一定能分辨两个近邻源。

---

## 8. 推荐参数估计流程

```text
估计协方差并检查秩、条件数
    ↓
加载或收缩稳定协方差
    ↓
估计有效模式数
    ↓
MUSIC / Capon / ESPRIT 估计 DOA
    ↓
根据 DOA 拟合源功率与噪声功率
    ↓
构造目标、干扰和噪声协方差
    ↓
设计波束权值
    ↓
计算输出 SINR、抑制度和目标增益
```

DOA、功率和 SINR 不是彼此独立的问题。DOA 误差会进入导向矩阵，继续影响功率分解、保护约束和最终输出性能。

下一步学习 [[08_MVDR波束形成]]，利用协方差和目标导向矢量设计自适应空间权值。

