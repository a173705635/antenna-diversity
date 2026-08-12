# 第37章：多约束 GSC 与阻塞矩阵鲁棒性

## 37.1 GSC 分解

LCMV 约束：

$$
\mathbf C^H\mathbf w=\mathbf f
$$

GSC 写成：

$$
\boxed{\mathbf w=\mathbf w_q-\mathbf B\mathbf g}
$$

其中：

$$
\mathbf C^H\mathbf w_q=\mathbf f
$$

$$
\mathbf C^H\mathbf B=0
$$

因此：

$$
\mathbf C^H\mathbf w=\mathbf f
$$

对任意 $\mathbf g$ 都成立。

## 37.2 固定权值

最小范数解：

$$
\boxed{\mathbf w_q=\mathbf C(\mathbf C^H\mathbf C)^{-1}\mathbf f}
$$

若约束矩阵病态，应使用数值秩和伪逆，而不是机械求逆。

## 37.3 阻塞矩阵维数

若：

$$
r=\operatorname{rank}(\mathbf C)
$$

则：

$$
\dim\operatorname{null}(\mathbf C^H)=N-r
$$

因此：

$$
\mathbf B\in\mathbb C^{N\times(N-r)}
$$

## 37.4 SVD 构造

完整 SVD：

$$
\mathbf C^H=\mathbf U\boldsymbol\Sigma\mathbf V^H
$$

写成：

$$
\mathbf V=[\mathbf V_r,\mathbf V_0]
$$

其中 $\mathbf V_0$ 对应零奇异值，于是：

$$
\boxed{\mathbf B=\mathbf V_0}
$$

并满足：

$$
\mathbf B^H\mathbf B=\mathbf I
$$

## 37.5 主支路和辅助支路

$$
d[k]=\mathbf w_q^H\mathbf x[k]
$$

$$
\mathbf u[k]=\mathbf B^H\mathbf x[k]
$$

$$
e[k]=d[k]-\mathbf g^H[k]\mathbf u[k]
$$

## 37.6 维纳解

定义：

$$
\mathbf R_u=E[\mathbf u\mathbf u^H]
=\mathbf B^H\mathbf R_x\mathbf B
$$

$$
\mathbf p_{ud}=E[\mathbf u d^*]
=\mathbf B^H\mathbf R_x\mathbf w_q
$$

则：

$$
\boxed{\mathbf g_{\mathrm{opt}}=\mathbf R_u^{-1}\mathbf p_{ud}}
$$

总权值：

$$
\boxed{
\mathbf w_{\mathrm{GSC}}=
\mathbf w_q-
\mathbf B(\mathbf B^H\mathbf R_x\mathbf B)^{-1}
\mathbf B^H\mathbf R_x\mathbf w_q}
$$

理想条件下与 LCMV 解等价。

## 37.7 真实目标泄漏

设计保护子空间为 $\operatorname{col}(\mathbf C)$，真实目标子空间为 $\operatorname{col}(\mathbf C_{\mathrm{true}})$。若二者不一致：

$$
\boxed{\mathbf B^H\mathbf C_{\mathrm{true}}\neq0}
$$

定义泄漏矩阵：

$$
\mathbf L=\mathbf B^H\mathbf C_{\mathrm{true}}
$$

泄漏量：

$$
\epsilon_{\mathrm{leak}}=\|\mathbf L\|_F
$$

## 37.8 为什么会自消除

真实目标辅助分量：

$$
\mathbf u_s=\mathbf B^H\mathbf C_{\mathrm{true}}\mathbf s
$$

固定支路目标：

$$
d_s=\mathbf w_q^H\mathbf C_{\mathrm{true}}\mathbf s
$$

若两者相关，维纳或 LMS/RLS 会把目标泄漏当作可预测成分并从主支路中减去。

## 37.9 多点保护阻塞

构造扩展保护矩阵：

$$
\mathbf C_{\mathrm{rob}}=
[\mathbf a(\theta_0-\Delta\theta),
\mathbf a(\theta_0),
\mathbf a(\theta_0+\Delta\theta)]
$$

再令：

$$
\mathbf B_{\mathrm{rob}}\subset\operatorname{null}(\mathbf C_{\mathrm{rob}}^H)
$$

可扩大阻塞保护角域，但减少辅助自由度。

## 37.10 导数保护

小角度误差下：

$$
\mathbf a(\theta_0+\Delta\theta)
\approx
\mathbf a(\theta_0)+
\Delta\theta\frac{\partial\mathbf a}{\partial\theta}\bigg|_{\theta_0}
$$

若同时满足：

$$
\mathbf B^H\mathbf a(\theta_0)=0
$$

$$
\mathbf B^H\frac{\partial\mathbf a}{\partial\theta}\bigg|_{\theta_0}=0
$$

则一阶角度误差泄漏受到抑制。

## 37.11 辅助正则化

$$
\mathbf g_\delta=
(\mathbf R_u+\delta\mathbf I)^{-1}\mathbf p_{ud}
$$

限制 $\|\mathbf g\|$，降低病态协方差和目标泄漏的放大。

## 37.12 泄漏 LMS

$$
\boxed{
\mathbf g[k+1]
=(1-\mu\gamma)\mathbf g[k]
+\mu\mathbf u[k]e^*[k]}
$$

等价于在均方误差上增加 $\gamma\|\mathbf g\|^2$ 惩罚。

## 37.13 时变约束

卫星方向和平台姿态变化导致：

$$
\mathbf C=\mathbf C[k]
$$

所以 $\mathbf w_q[k]$ 和 $\mathbf B[k]$ 也变化。阻塞基变化会改变辅助坐标系，实际系统需要平滑更新、权值映射或适当重置。

## 37.14 阻塞基不唯一

若 $\mathbf T$ 为可逆矩阵：

$$
\mathbf B'=\mathbf B\mathbf T
$$

仍满足：

$$
\mathbf C^H\mathbf B'=0
$$

理论可行子空间相同。SVD 给出的正交基通常数值性质更好。

## 37.15 核心结论

阻塞矩阵阻塞的是设计模型中的保护子空间，不是自动识别真实目标。鲁棒 GSC 的核心权衡是：

$$
\boxed{\text{保护子空间越宽}\quad\leftrightarrow\quad\text{辅助抗干扰自由度越少}}
$$
