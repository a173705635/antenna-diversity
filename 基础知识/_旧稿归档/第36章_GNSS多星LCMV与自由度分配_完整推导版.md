# 第36章：GNSS 多星 LCMV 与自由度分配

## 36.1 多星约束模型

保护 $Q$ 颗卫星，构造：

$$
\mathbf C=[\mathbf a_1,\mathbf a_2,\ldots,\mathbf a_Q]
$$

期望响应：

$$
\mathbf f=[f_1,\ldots,f_Q]^T
$$

约束：

$$
\boxed{\mathbf C^H\mathbf w=\mathbf f}
$$

## 36.2 LCMV 问题

$$
\min_{\mathbf w}\mathbf w^H\mathbf R_x\mathbf w
$$

满足：

$$
\mathbf C^H\mathbf w=\mathbf f
$$

## 36.3 拉格朗日推导

构造：

$$
\mathcal L=
\mathbf w^H\mathbf R_x\mathbf w
+\boldsymbol\lambda^H(\mathbf C^H\mathbf w-\mathbf f)
+(\mathbf w^H\mathbf C-\mathbf f^H)\boldsymbol\lambda
$$

对 $\mathbf w^*$ 求导：

$$
\mathbf R_x\mathbf w+\mathbf C\boldsymbol\lambda=0
$$

所以：

$$
\mathbf w=-\mathbf R_x^{-1}\mathbf C\boldsymbol\lambda
$$

代入约束：

$$
-\mathbf C^H\mathbf R_x^{-1}\mathbf C\boldsymbol\lambda=\mathbf f
$$

因此：

$$
\boldsymbol\lambda
=-
(\mathbf C^H\mathbf R_x^{-1}\mathbf C)^{-1}\mathbf f
$$

最终：

$$
\boxed{
\mathbf w_{\mathrm{LCMV}}
=\mathbf R_x^{-1}\mathbf C
(\mathbf C^H\mathbf R_x^{-1}\mathbf C)^{-1}\mathbf f}
$$

## 36.4 最小范数固定权值

GSC 固定支路常取：

$$
\boxed{
\mathbf w_q=
\mathbf C(\mathbf C^H\mathbf C)^{-1}\mathbf f}
$$

验证：

$$
\mathbf C^H\mathbf w_q=\mathbf f
$$

## 36.5 自由度

若：

$$
r=\operatorname{rank}(\mathbf C)
$$

则满足约束后剩余自由度：

$$
\boxed{N-r}
$$

若 $Q$ 个约束独立，则为 $N-Q$。这些自由度用于形成自适应零陷。

## 36.6 为什么不能无限增加保护星数

当 $Q$ 接近 $N$ 时，剩余干扰抑制自由度很少。当 $Q=N$ 且约束独立时，权值几乎被完全确定；当 $Q>N$ 时一般可能无解。

## 36.7 多星共用权值的局限

一个公共权值只形成一个空间响应：

$$
B(\theta)=\mathbf w^H\mathbf a(\theta)
$$

多个卫星方向同时保持单位响应，会使波束图在多个方向被固定，从而减少干扰零陷可调空间。

## 36.8 每星独立权值

相关后处理可为每颗卫星设计：

$$
\mathbf w_q=
\frac{\mathbf R^{-1}\mathbf a_q}
{\mathbf a_q^H\mathbf R^{-1}\mathbf a_q}
$$

这样避免一个权值同时保护所有星，但计算和通道结构更复杂，且相关前强干扰仍可能造成前端问题。

## 36.9 约束病态

若两颗卫星方向很近：

$$
\mathbf a_1\approx\mathbf a_2
$$

则 $\mathbf C$ 列近似相关，矩阵：

$$
\mathbf C^H\mathbf R^{-1}\mathbf C
$$

可能病态，导致权值范数大和 WNG 下降。

## 36.10 数值秩处理

可对 $\mathbf C$ 做 SVD：

$$
\mathbf C=\mathbf U\boldsymbol\Sigma\mathbf V^H
$$

根据奇异值阈值确定有效秩，只在独立约束子空间内求解，避免直接把重复约束当成新增自由度消耗。

## 36.11 显式干扰零陷

若干扰方向已知，可构造：

$$
\mathbf C=[\mathbf A_s,\mathbf A_j]
$$

$$
\mathbf f=[\mathbf1^T,\mathbf0^T]^T
$$

但每个显式零陷同样消耗约束自由度，通常只在干扰方向可靠且自由度足够时采用。

## 36.12 鲁棒多点保护

对卫星 $q$ 可在邻近角度采样：

$$
\theta_q-\Delta\theta,\quad\theta_q,\quad\theta_q+\Delta\theta
$$

扩大保护主瓣，但约束数快速增加。必须权衡保护宽度和干扰抑制能力。

## 36.13 姿态与坐标变换

卫星方向通常在导航坐标系中表示。阵列坐标方向：

$$
\mathbf u_{\mathrm{array}}
=\mathbf R_{\mathrm{att}}^T\mathbf u_{\mathrm{nav}}
$$

姿态误差会直接转化为导向矢量误差，因此多星约束依赖星历、接收机位置、姿态和安装矩阵。

## 36.14 对角加载多星 LCMV

$$
\mathbf R_{\mathrm{DL}}=\widehat{\mathbf R}+\delta\mathbf I
$$

$$
\boxed{
\mathbf w=
\mathbf R_{\mathrm{DL}}^{-1}\mathbf C
(\mathbf C^H\mathbf R_{\mathrm{DL}}^{-1}\mathbf C)^{-1}\mathbf f}
$$

加载可缓解快拍不足、约束病态和模型失配。

## 36.15 设计检查

至少检查：

$$
\epsilon_c=\|\mathbf C^H\mathbf w-\mathbf f\|
$$

以及：

- $\|\mathbf w\|$；
- WNG；
- 约束矩阵条件数；
- 干扰方向增益；
- 卫星邻域增益；
- 姿态误差敏感性。

## 36.16 核心结论

多星 LCMV 的核心矛盾是：

$$
\boxed{\text{卫星保护数量}\quad\leftrightarrow\quad\text{剩余抗干扰自由度}}
$$

阵元数不只是决定角分辨率，也决定能同时保护多少独立方向并保留多少零陷能力。
