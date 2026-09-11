# 第39章：窄带干扰 DOA、功率与 SINR 估计

## 39.1 本章目标

检测到干扰后，工程系统通常还需要估计：

- 干扰方向；
- 干扰功率；
- 噪声功率；
- 输入与输出 SINR；
- 干扰抑制度。

本章把这些参数与协方差模型联系起来。

## 39.2 窄带多源模型

$$
\mathbf x[k]=
\sum_{p=1}^{P}\mathbf a(\theta_p)s_p[k]+\mathbf n[k]
$$

协方差：

$$
\mathbf R_x=
\mathbf A\mathbf R_s\mathbf A^H+\mathbf R_n
$$

若各源互不相关且噪声为空间白噪声：

$$
\mathbf R_x=
\sum_{p=1}^{P}P_p\mathbf a_p\mathbf a_p^H
+\sigma_n^2\mathbf I
$$

## 39.3 MUSIC DOA 估计回顾

特征分解：

$$
\widehat{\mathbf R}=
\widehat{\mathbf U}_s\widehat{\boldsymbol\Lambda}_s\widehat{\mathbf U}_s^H
+
\widehat{\mathbf U}_n\widehat{\boldsymbol\Lambda}_n\widehat{\mathbf U}_n^H
$$

理想方向满足：

$$
\mathbf U_n^H\mathbf a(\theta_p)=0
$$

MUSIC 谱：

$$
\boxed{
P_{\mathrm{MUSIC}}(\theta)=
\frac1{
\mathbf a^H(\theta)
\widehat{\mathbf U}_n
\widehat{\mathbf U}_n^H
\mathbf a(\theta)}}
$$

谱峰位置给出 DOA。

## 39.4 Capon 空间谱

Capon 谱：

$$
\boxed{
P_{\mathrm{Capon}}(\theta)=
\frac1{
\mathbf a^H(\theta)
\widehat{\mathbf R}^{-1}
\mathbf a(\theta)}}
$$

它表示在方向 $\theta$ 保持单位响应时的最小输出功率。

与 MUSIC 相比，Capon 不需要事先明确划分信号和噪声子空间，但依赖稳定协方差逆。

## 39.5 波束扫描功率

传统扫描波束：

$$
\mathbf w_{\mathrm{CBF}}(\theta)
=
\frac{\mathbf a(\theta)}{\mathbf a^H(\theta)\mathbf a(\theta)}
$$

输出功率：

$$
P_{\mathrm{CBF}}(\theta)
=
\mathbf w_{\mathrm{CBF}}^H(\theta)
\widehat{\mathbf R}
\mathbf w_{\mathrm{CBF}}(\theta)
$$

分辨率低于 MUSIC 和 Capon，但鲁棒、直观。

## 39.6 已知 DOA 后的功率估计

假设 DOA 已估计，构造：

$$
\widehat{\mathbf A}=[\mathbf a(\widehat\theta_1),\ldots,\mathbf a(\widehat\theta_P)]
$$

模型：

$$
\widehat{\mathbf R}
\approx
\widehat{\mathbf A}\mathbf R_s\widehat{\mathbf A}^H+\sigma_n^2\mathbf I
$$

若源互不相关：

$$
\mathbf R_s=\operatorname{diag}(P_1,\ldots,P_P)
$$

可以把各个矩阵向量化：

$$
\operatorname{vec}(\widehat{\mathbf R})
\approx
\sum_{p=1}^{P}P_p
(\mathbf a_p^*\otimes\mathbf a_p)
+
\sigma_n^2\operatorname{vec}(\mathbf I)
$$

这变成关于 $P_p$ 和 $\sigma_n^2$ 的线性最小二乘问题。

## 39.7 单源功率估计

若只有一个源：

$$
\mathbf R=P_s\mathbf a\mathbf a^H+\sigma_n^2\mathbf I
$$

若 $\|\mathbf a\|^2=N$，沿导向方向投影：

$$
\mathbf a^H\mathbf R\mathbf a
=P_sN^2+\sigma_n^2N
$$

若已知噪声功率：

$$
\boxed{
P_s=
\frac{
\mathbf a^H\mathbf R\mathbf a-\sigma_n^2N
}{N^2}}
$$

## 39.8 噪声功率估计

在白噪声假设下，若源数为 $P$，可用最小 $N-P$ 个特征值平均：

$$
\boxed{
\widehat\sigma_n^2=
\frac1{N-P}
\sum_{i=P+1}^{N}\widehat\lambda_i}
$$

有限快拍和有色噪声下，这只是近似。

## 39.9 输入 SINR

若参考阵元目标功率、干扰功率和噪声功率分别为：

$$
P_s,\quad P_j,\quad P_n
$$

输入 SINR：

$$
\boxed{
\mathrm{SINR}_{\mathrm{in}}=
\frac{P_s}{P_j+P_n}}
$$

dB 表示：

$$
\mathrm{SINR}_{\mathrm{in,dB}}=
10\log_{10}\mathrm{SINR}_{\mathrm{in}}
$$

## 39.10 输出 SINR

目标协方差、干扰协方差和噪声协方差分别为：

$$
\mathbf R_s,\quad\mathbf R_j,\quad\mathbf R_n
$$

给定权值 $\mathbf w$：

$$
P_{s,\mathrm{out}}=\mathbf w^H\mathbf R_s\mathbf w
$$

$$
P_{j+n,\mathrm{out}}=
\mathbf w^H(\mathbf R_j+\mathbf R_n)\mathbf w
$$

所以：

$$
\boxed{
\mathrm{SINR}_{\mathrm{out}}=
\frac{
\mathbf w^H\mathbf R_s\mathbf w
}{
\mathbf w^H(\mathbf R_j+\mathbf R_n)\mathbf w
}}
$$

## 39.11 单目标输出功率

若：

$$
\mathbf R_s=P_s\mathbf a_0\mathbf a_0^H
$$

则：

$$
P_{s,\mathrm{out}}
=P_s|\mathbf w^H\mathbf a_0|^2
$$

若单位响应约束成立：

$$
\mathbf w^H\mathbf a_0=1
$$

则：

$$
P_{s,\mathrm{out}}=P_s
$$

## 39.12 干扰抑制度

可定义：

$$
\mathrm{JSR}_{\mathrm{supp}}
=
10\log_{10}
\frac{P_{j,\mathrm{in}}}{P_{j,\mathrm{out}}}
$$

也可直接看方向增益：

$$
G_j=20\log_{10}|\mathbf w^H\mathbf a(\theta_j)|
$$

方向零陷深度不能完全替代实际输出干扰功率，因为干扰可能有方向扩展和模型失配。

## 39.13 DOA 估计误差

$$
\epsilon_\theta=\widehat\theta-\theta_{\mathrm{true}}
$$

多次试验可统计：

$$
\mathrm{RMSE}_\theta=
\sqrt{E[(\widehat\theta-\theta)^2]}
$$

应区分：

- 系统性偏差；
- 随机方差；
- 离群错误。

## 39.14 分辨率与精度

分辨率是区分两个近邻源的能力，精度是单个源估计误差大小。高精度不等于能分辨两个非常接近的源。

## 39.15 参数估计流程

建议：

1. 估计并正则化协方差；
2. 估计源数；
3. MUSIC/Capon/ESPRIT 估计 DOA；
4. 根据 DOA 拟合功率和噪声；
5. 设计波束权值；
6. 分别计算目标、干扰和噪声输出功率；
7. 评估 SINR 与抑制度。

## 39.16 核心结论

DOA、功率和 SINR 不是彼此独立的三个问题。DOA 误差会进入导向矩阵，进一步影响功率分解、约束设计和最终输出 SINR。
