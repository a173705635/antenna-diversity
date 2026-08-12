# 第42章：窄带 GNSS 阵列抗干扰系统综合设计与评估

## 42.1 本章目的

本章把前面知识串成完整窄带系统流程，不再进入 STAP。核心模块包括：

- 阵列和校准；
- 窄带模型检查；
- 协方差估计与正则化；
- 干扰数和 DOA 估计；
- MVDR、LCMV、GSC 或 PI；
- 在线自适应；
- 性能和鲁棒性评估。

## 42.2 系统信号模型

$$
\mathbf x[k]=
\mathbf A_s\mathbf s[k]
+\mathbf A_j\mathbf j[k]
+\mathbf n[k]
$$

协方差：

$$
\mathbf R_x=
\mathbf A_s\mathbf R_s\mathbf A_s^H
+\mathbf A_j\mathbf R_j\mathbf A_j^H
+\mathbf R_n
$$

窄带条件：

$$
B\tau_{\max}\ll1
$$

## 42.3 前端与同步要求

阵列处理前应保证：

- 通道采样时钟同步；
- 本振相干或相位关系可校准；
- 增益不饱和；
- 通道延迟已知；
- ADC 动态范围足以容纳强干扰。

数字算法无法恢复已经被前端饱和丢失的信息。

## 42.4 阵列校准模块

应维护：

$$
\widehat{\mathbf a}_{\mathrm{true}}(\theta)
$$

或校准矩阵。校准验证指标包括：

$$
\epsilon_a=
\frac{\|\widehat{\mathbf a}-\mathbf a_{\mathrm{ref}}\|}
{\|\mathbf a_{\mathrm{ref}}\|}
$$

$$
\rho=
\frac{|\widehat{\mathbf a}^H\mathbf a_{\mathrm{ref}}|}
{\|\widehat{\mathbf a}\|\|\mathbf a_{\mathrm{ref}}\|}
$$

## 42.5 协方差模块

样本协方差：

$$
\widehat{\mathbf R}=
\frac1K\sum_{k=1}^{K}\mathbf x[k]\mathbf x^H[k]
$$

加载：

$$
\mathbf R_{\mathrm{DL}}=
\widehat{\mathbf R}+\delta\mathbf I
$$

应在线监测：

- 有效秩；
- 条件数；
- 最大和最小特征值；
- 窗口平稳性；
- 异常快拍比例。

## 42.6 干扰检测与参数估计

可用：

- 特征值间隙；
- AIC/MDL；
- MUSIC/Capon；
- 输出功率变化；
- 时间连续性。

估计结果包括：

$$
\widehat P_j,\quad
\widehat\theta_j,\quad
\widehat P_{j,p}
$$

它们用于决定约束、自适应模式和自由度是否足够。

## 42.7 波束形成模式选择

### 单目标已知方向

MVDR：

$$
\mathbf w=
\frac{\mathbf R^{-1}\mathbf a_0}
{\mathbf a_0^H\mathbf R^{-1}\mathbf a_0}
$$

### 多星公共权值

LCMV：

$$
\mathbf w=
\mathbf R^{-1}\mathbf C
(\mathbf C^H\mathbf R^{-1}\mathbf C)^{-1}\mathbf f
$$

### 在线自适应

GSC：

$$
\mathbf w=\mathbf w_q-\mathbf B\mathbf g
$$

### 目标方向未知

功率反演可作为简单盲抗干扰方法，但目标保护能力较弱。

## 42.8 自由度预算

若约束秩为 $r$：

$$
D_{\mathrm{free}}=N-r
$$

若独立干扰模式数为 $P_j$，需要检查：

$$
P_j\le D_{\mathrm{free}}
$$

若不满足，应减少约束、增加阵元、接受部分抑制或使用每星独立权值结构。

## 42.9 目标保护指标

目标方向增益：

$$
G_s=|\mathbf w^H\mathbf a_{\mathrm{true}}|
$$

增益误差：

$$
\epsilon_G=|G_s-1|
$$

多星约束残差：

$$
\epsilon_c=\|\mathbf C^H\mathbf w-\mathbf f\|
$$

GSC 泄漏：

$$
\epsilon_B=\|\mathbf B^H\mathbf C_{\mathrm{true}}\|_F
$$

## 42.10 输出 SINR

$$
\mathrm{SINR}_{\mathrm{out}}=
\frac{\mathbf w^H\mathbf R_s\mathbf w}
{\mathbf w^H(\mathbf R_j+\mathbf R_n)\mathbf w}
$$

SINR 提升：

$$
\Delta\mathrm{SINR}_{\mathrm{dB}}
=
\mathrm{SINR}_{\mathrm{out,dB}}
-
\mathrm{SINR}_{\mathrm{in,dB}}
$$

## 42.11 白噪声增益

单位目标响应下：

$$
\mathrm{WNG}=
\frac1{\|\mathbf w\|^2}
$$

低 WNG 表示对通道白噪声和模型误差敏感。不能只追求最深零陷而忽略 WNG。

## 42.12 干扰抑制度

输出干扰功率：

$$
P_{j,\mathrm{out}}=
\mathbf w^H\mathbf R_j\mathbf w
$$

抑制度：

$$
\mathrm{Supp}_{j}=10\log_{10}
\frac{P_{j,\mathrm{in}}}{P_{j,\mathrm{out}}}
$$

波束图零陷深度只是辅助指标，真实输出功率更重要。

## 42.13 收敛和跟踪指标

在线算法应统计：

- 达到目标抑制度所需快拍数；
- 稳态输出功率；
- 权值抖动；
- 环境突变后的恢复时间；
- 目标增益最低值。

## 42.14 鲁棒性试验

应系统扫描：

- 导向角误差；
- 通道增益误差；
- 通道相位误差；
- 快拍数；
- 干扰功率；
- 干扰角间隔；
- 姿态误差；
- 协方差窗口长度；
- 加载量。

性能不能只在一个理想场景下验证。

## 42.15 算法复杂度

空间协方差维数为 $N\times N$。

- LMS/NLMS 每快拍约 $O(N)$；
- RLS 每快拍约 $O(N^2)$；
- 批量求解和特征分解约 $O(N^3)$。

对于少阵元 GNSS 终端，批量加载 MVDR/LCMV 通常可行；在线高速更新时需结合硬件资源。

## 42.16 故障和降级模式

系统应设计：

- 协方差病态时增大加载；
- 约束无解时减少保护约束；
- 校准失效时退化为 CBF；
- 自适应异常时冻结权值；
- 前端饱和时报告不可恢复状态。

可靠系统必须允许降级，而不是在异常条件下继续输出看似正常的权值。

## 42.17 推荐的窄带系统流程

```text
验证窄带条件
    ↓
通道同步与阵列校准
    ↓
样本协方差估计
    ↓
秩、条件数和异常检查
    ↓
加载或收缩正则化
    ↓
干扰数量与 DOA 估计
    ↓
自由度预算
    ↓
选择 MVDR / LCMV / GSC / PI
    ↓
在线更新与保护监测
    ↓
SINR、WNG、目标增益和抑制度评估
```

## 42.18 后续可选方向

完成本窄带路线后，可按需求进入：

- 每星独立波束形成与码域联合；
- GNSS 欺骗空间一致性检测；
- 宽带干扰和 STAP；
- 大规模阵列降秩处理；
- 实测阵列流形和硬件闭环测试。

## 42.19 核心结论

窄带阵列抗干扰系统的性能不是由某一个算法决定，而是由以下链条共同决定：

$$
\boxed{
\text{前端动态范围}
\rightarrow
\text{校准}
\rightarrow
\text{协方差}
\rightarrow
\text{约束与自由度}
\rightarrow
\text{鲁棒权值}
\rightarrow
\text{在线监测}}
$$
