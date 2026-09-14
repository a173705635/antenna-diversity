# 多天线抗欺骗与测向：工程误差统一数学模型

日期：2026-09-14。对象：当前 GPS L1、2–4 天线、75.9 mm × 160.9 mm 平面阵列；不引入 IMU。本文继承[原 49 项偏差清单](2026-09-13-sdcp-error-inventory.md)，逐项给出模型，并新增 16 项系统问题。它是工程建模基线，不是“所有物理故障已穷尽”的保证，也不表示这些模型均已加入仿真。

**结论：应建立“复数信号链 + 有效相位弧内误差方程 + 相关协方差 + 离散失效状态”四层模型。** 当前没有本机多天线实测标定数据，不能给出可信的总误差毫米数或硬件检测率。本文数值仅为解析换算；既有识别/测向结果仍以[原实验报告](2026-09-14-doa-double-difference.md)为准。

## 1. 符号、符号约定与建模层次

天线索引 $a=0,\ldots,M-1$，卫星/PRN 索引 $s$，路径或发射源索引 $q$，历元 $k$。$\mathbf p_a$ 为天线位置，$\mathbf b_a=\mathbf p_a-\mathbf p_0$；$\mathbf u_s$ 是接收机指向来波源的单位向量；$\lambda=c/f_{L1}=0.190293672798\,\mathrm m$。$L$ 为载波距离（m），$\phi$ 为相位（rad），$f$ 为 Hz，时间为 s。方差必须同时注明单位、带宽、积分时间及有效状态。

采用传播复相位 $\exp(-j2\pi\rho/\lambda)$，因此额外复增益相位 $\delta\phi$ 对载波距离的贡献为

$$\mathcal M(\delta\phi)=-\frac{\lambda}{2\pi}\delta\phi.\tag{1}$$

它与“传播延迟增加导致距离增加”一致：纯 RF 延迟 $H(f)=e^{-j2\pi f\tau}$ 给出 $\delta L=c\tau$。ADC 时间提前采样 $x(t+\delta t)$ 则给出 $\delta L=-\lambda f_{\rm ADC}\delta t$，这里的 $f_{\rm ADC}$ 是 ADC 输入实际模拟频率，不是随意选取的载频或数字 NCO 频率。

### 1.1 复数链路层：用于混合、非线性及错误锁定

$$v_a(t)=\sum_{s,q} A_{asq}(t)c_s(t-\tau_{asq})d_s(t-\tau_{asq})e^{j\phi_{asq}(t)}+n_a(t),$$
$$\mathbf v_{\rm port}=\mathbf C(f,T,\text{安装})\mathbf v,\qquad y_a[n]=Q_a\!\left\{\mathcal F_a[\mathbf v_{\rm port}](t_{a,n})\right\},$$
$$Z_{as,k}=\operatorname{Corr}_{s,k}\{\operatorname{DFE}_a(y_a)\},\qquad \widehat L_{as,k}=\operatorname{TrackPhase}(Z_{as,0:k}).\tag{2}$$

$\mathcal F$ 包含模拟滤波、增益、I/Q 非理想及非线性；$Q$ 包含量化；$t_{a,n}$ 包含时钟误差；跟踪有内部状态。多径、真假混合、压缩、错误锁定必须在这一层或经过验证的等效相关器层建模，不能先把各路径“相位距离”作线性平均。

### 1.2 有效相位弧层：可加误差模型

仅在分量归属稳定、未饱和、跟踪有效的同一相位弧内使用

$$L_{as}=\rho_{as}+c(\delta t_a-\delta t_s)+T_{as}-I_{as}+B_a+B_{as}^{\rm dir}+\beta_{as}^{\rm arc}+m_{as}+\epsilon_{as},\tag{3}$$
$$P_{as}=\rho_{as}+c(\delta t_a-\delta t_s)+T_{as}+I_{as}+B^P_{as}+m^P_{as}+\epsilon^P_{as}.\tag{4}$$

$B_a$ 是在本频率、历元对各 PRN 共同的通道距离偏置，$B_{as}^{\rm dir}$ 是方向/频率/状态相关项；$m$ 是复数链路在适用条件下的等效误差。码与载波的电离层符号相反，码群时延不能直接当作载波相位偏置。基本观测方程依据 [ESA GNSS Basic Observables](https://gssc.esa.int/navipedia/index.php/GNSS_Basic_Observables)，这里的分组和工程扩展为本项目建模。

当前 NCO 载波重建为 $L=-\lambda[\phi_{\rm NCO}(t_i)+2\pi f_{\rm NCO}(t_i)(t-t_i)]/(2\pi)$。其每弧初相未必已绑定标准整周起点，所以先用一般实数常量 $\beta^{\rm arc}$；只有初相/极性条件成立后，才细分为 $\lambda N+\lambda h/2$，$N\in\mathbb Z,h\in\{0,1\}$。**不能同时把一般弧偏置、整周和半周当作三个独立误差重复相加。**

### 1.3 参数、随机量与状态层

$$\mathbf e_k=\mathbf H_k\boldsymbol\theta+\mathbf G_k\mathbf q_k+\boldsymbol\epsilon_k,\quad
\mathbf q_{k+1}=\mathbf F\mathbf q_k+\mathbf w_k,\quad
\boldsymbol\Sigma_e(k,l)=\mathbf G_k\boldsymbol\Sigma_q(k,l)\mathbf G_l^T+\boldsymbol\Sigma_\epsilon(k,l).\tag{5}$$

$\boldsymbol\theta$ 表示待标定固定参数；$\mathbf q$ 表示具有跨通道、跨星、跨时间相关性的漂移；$\epsilon$ 为剩余随机项。固定未知量没有测量或先验时不能擅自赋零均值高斯分布。另设状态

$$z_k\in\{\text{有效弧、捕获、失锁、重捕、周跳、饱和、标定失效、缺测、错误分量}\}.\tag{6}$$

状态决定式 (3) 是否适用及输出是否必须为 unknown。质量指标恢复不等于旧相位弧或真实信号归属恢复。

## 2. 天线差、星间双差、时间差：消掉什么，留下什么

定义 $d_{as}=L_{as}-L_{0s}$。远场一阶几何下

$$d_{as}=-\mathbf b_a^T\mathbf u_s+\Delta_a B+\Delta_a B_s^{\rm dir}+\Delta_a\beta_s^{\rm arc}+\Delta_a m_s+\Delta_a\epsilon_s+\Delta_a c\delta t+\Delta_a(T_s-I_s).\tag{7}$$

卫星钟等真正公共项消失；真实短基线大气差较小，不是严格为零。对参考星 $r$ 再差：

$$D_{as,r}=d_{as}-d_{ar}=-\mathbf b_a^T(\mathbf u_s-\mathbf u_r)+\Delta_s\Delta_a(B^{\rm dir}+\beta^{\rm arc}+m+\epsilon+T-I).\tag{8}$$

公共 $\Delta_a B$ 及公共接收机钟差在同步、同频条件下消失；逐星初相、逐星方向误差、非同步误差和混合偏差不保证消失。若 $s,r$ 都由同一远场源发射，$\mathbf u_s=\mathbf u_r$，**源方向本身也消失**。若 $r$ 的真实性与方向可信，则

$$D_{as,r}-\mathbf b_a^T\mathbf u_r=-\mathbf b_a^T\mathbf u_s+\text{剩余误差}.\tag{9}$$

这就是参考星测向的前提；仅凭接收质量良好不能确认前提成立。时间差 $\delta_t d_k=d_k-d_{k-1}$ 消掉不变偏置，同时消掉静态几何；留下偏置漂移、动态几何、跳变，并改变噪声协方差。去均值或去趋势同样可能删去攻击特征。

### 2.1 协方差不能按基线数盲目缩小

将原观测写为 $\mathbf Y\in\mathbb R^{M\times S}$，天线差矩阵 $\mathbf D_a$，星间差矩阵 $\mathbf D_s$：

$$\mathbf Y_{DD}=\mathbf D_a\mathbf Y\mathbf D_s^T,\quad
\boldsymbol\Sigma_{DD}=(\mathbf D_s\otimes\mathbf D_a)\boldsymbol\Sigma_Y(\mathbf D_s\otimes\mathbf D_a)^T.\tag{10}$$

独立原观测方差均为 $\sigma^2$ 时：一条 SD 方差 $2\sigma^2$；共用参考天线的两条 SD 协方差 $\sigma^2$；一条 DD 方差 $4\sigma^2$；固定天线对且共用参考星的两条 DD 协方差 $2\sigma^2$。全部 $M(M-1)/2$ 基线只有 $M-1$ 个独立空间差分维度；双差最多 $(M-1)(S-1)$ 个。相关矩阵退化时使用独立差分基或广义逆。

对于方差 $\sigma^2$、相关系数 $\rho_h$ 的平稳时间序列，

$$\operatorname{Var}(\bar e)=\frac{\sigma^2}{N}\left[1+2\sum_{h=1}^{N-1}(1-h/N)\rho_h\right].\tag{11}$$

AR(1) 的 $N_{\rm eff}\approx N(1-r)/(1+r)$ 只是长窗近似；短窗必须用有限和。不能把当前 10 个相关历元当作 10 次独立测量。

## 3. 继承的 49 项：逐项方程与工程边界

表中“部分”表示存在等效或相关模块，不表示当前正式 RF 实验启用了完整模型；“待测/待建”表示没有本机闭环证据。$\mathcal M$ 见式 (1)，所有相位函数先定义在复数链路，再按有效条件映射为距离。各行存在因果重叠，预算时应选择物理参数层，不能机械求和。

### 3.1 天线、安装与线缆

| ID | 问题及数学模型 | 差分/标定边界 | 当前覆盖 |
|---|---|---|---|
| A01 | PCO：$e_{as}=-\delta\mathbf p_a^T\mathbf u_s$ | SD 留下差分 PCO；DD 随星方向改变；须测安装后相位中心 | 有实际/假定几何接口，未实测 |
| A02 | PCV：$e_{as}=g_a(\mathbf u_s,f,T)$，可用球谐/角度网格展开 | 方向相关，不能靠常数标定全部消除 | 有方向响应能力，未实测标定 |
| A03 | 极化/绕转：$e_{as}=\mathcal M[\arg(\mathbf h_a^H\mathbf e_s)]$ | 姿态、极化及 LOS 相关；与完整复方向图去重 | 有 Jones/极化模型，当前未验证安装后响应 |
| A04 | 遮挡增益：$A_{as}=A_sG_a(\mathbf u_s)O_{as}$，$C/N_0\propto A_{as}^2/N_0$ | 本身先影响 SNR/路径权重，再影响跟踪相位和缺测 | 部分，可配置方向增益 |
| A05 | 天线互耦：$\mathbf v_{\rm port}=\mathbf C(f,\mathbf Z_L)\mathbf v$ | 有效响应逐方向变化；不能仅设独立相位噪声 | 有耦合矩阵接口，无实测矩阵 |
| A06 | 罩体/人体/水冰：$H_a=H_{a0}+\Delta H_a(\mathbf u,f,T,\text{环境})$ | 环境改变旧标定；其相位计入 A02 或本项一次 | 待测，未全面环境建模 |
| A07 | 变形：$\mathbf b_a(t)=\mathbf b_{a0}+\mathbf K_a\Delta T+\mathbf v_a(t)$ | $e=-\delta\mathbf b^T\mathbf u$；刚体转动也改变投影 | 支持几何设置，无实测热/振动模型 |
| C01 | 固定线缆：$H_a=e^{-j2\pi f\tau_a}$，$B_a=c\tau_a$ | 同频公共通道差可标定/DD 消除；载波只给模糊相位 | 固定相位/时延接口，未测线缆 |
| C02 | 温弯/滞回：$\tau_a=\tau_{a0}+k_{Ta}\Delta T+k_{ba}b(t)+h_a[\text{历史}]$ | 常数校准不能消漂移；温度回程不一定可逆 | 待测/待建 |

PCO/PCV 的计量定义参见 [NOAA 天线标定](https://www.ngs.noaa.gov/research/gnss-satellite/gnss-antenna-calibration.shtml)，极化绕转参见 [ESA phase wind-up](https://gssc.esa.int/navipedia/index.php/Carrier_Phase_Wind-up_Effect)。若使用安装后的完整复方向图，已包含的 PCO、PCV、极化、互耦效应不可再次独立添加。

### 3.2 模拟前端、LO 与 ADC

| ID | 问题及数学模型 | 差分/标定边界 | 当前覆盖 |
|---|---|---|---|
| R01 | 固定幅相：$H_a=g_ae^{j\phi_a}$，$B_a=\mathcal M(\phi_a)$ | 同频常相位在 DD 消失；幅度改变 SNR 不随之消失 | 已有人为固定幅相 RF 对照 |
| R02 | 频响：$H_a(f)=G_a(f)e^{j\phi_a(f)}$，$\tau_{g,a}=-\partial\phi_a/\partial(2\pi f)$ | PRN Doppler 不同，残差约 $\lambda\Delta\tau_g\Delta f$；群延迟不确定中心相位 | 无完整模拟带内频响模型 |
| R03 | AGC/AM–PM：$H_a(t)=g_a(u_t)e^{j[\phi_{a0}+k_{AP}u_t+h_a(t-t_{sw})]}$ | 增益切换有瞬态和状态依赖；只测稳态不足 | 未完整建模 |
| R04 | 热噪声：$n\sim\mathcal{CN}(0,N_0B)$，$F_{tot}=F_1+(F_2-1)/G_1+\cdots$ | 经带宽及跟踪传递后进入相位；不再叠加另一份同源 PLL 噪声 | 有等效级联噪声 |
| R05 | I/Q 不平衡：$y=\mu x+\nu x^*+d$ | 镜像项与输入频率/相位有关，非固定偏置 | 待建 |
| R06 | DC/杂散：$y=x+d+\sum_i a_ie^{j(2\pi f_it+\psi_i)}$ | 经相关器泄漏取决于频率与码；不一定被 SD/DD 消除 | 未系统覆盖 |
| R07 | 非线性：$y=a_1x+a_3x\lVert x\rVert^2+\cdots$，或限幅算子 | 阻塞、互调、压缩不可用高斯小误差替代 | 有简化 tanh/削顶，无实测互调/AM–PM |
| R08 | 电子串扰：$y_a=x_a+\sum_{b\ne a}\kappa_{ab}(f)x_b$ | 可能人为制造跨通道相干；区别于端口互耦 A05 | 未测电子串扰 |
| L01 | 共 LO 不同路径：$\phi_a(t)=\phi_c(t-\tau_a)+\phi_{a0}$ | 路径差和滤波差使共模不能完全抵消 | 正式实验理想共享参考 |
| L02 | 上电分支：$\phi_{a0}=2\pi k_a/K_a+\eta_a$ | 每次启动/重置重新辨识，不能默认永久常数 | 未建真实分频器启动状态 |
| L03 | LO 残差：$S_{\Delta\phi}=\lVert H_ae^{-j\omega\tau_a}-H_be^{-j\omega\tau_b}\rVert^2S_c+S_a+S_b$ | 同晶振不等于同 LO；共享相位也受不同路径/环路影响 | 待测谱与相关性 |
| AD01 | 孔径偏斜：$e_a=-\lambda f_{\rm ADC}\delta t_a$（单音近似） | 同采样时钟仍可有固定孔径差；宽带须作用于原波形 | 无实际 skew 模型 |
| AD02 | 抖动：$\delta\phi\simeq2\pi f_{\rm ADC}\delta t$，$\sigma_{\Delta t}^2=\sigma_a^2+\sigma_b^2-2\operatorname{Cov}(t_a,t_b)$ | 区分公共时钟抖动与独立孔径抖动 | 待建 |
| AD03 | 量化/INL：$y=Q(x+o)+\epsilon_{INL}(x)$ | $\Delta^2/12$ 仅为适当输入/抖动条件下近似；削顶另算状态 | 有量化，无实测 INL/DNL |
| AD04 | 采样频差：$t_a[n]=t_0+n/f_s+\delta t_{a0}+\varepsilon_an/f_s+w_a[n]$ | $\varepsilon_a$ 为时间尺度相对误差，累积跨通道失同步 | 正式链理想采样率，无真实漂移 |

ADC 参数的测量条件可参见 [Analog Devices AN-835](https://www.analog.com/en/resources/app-notes/an-835.html)；SDR 相位标定与 LO 问题参见 [Analog Devices 天线标定说明](https://www.analog.com/en/resources/technical-articles/simplify-antenna-calibrations-using-sdr.html)。本文没有引用其中器件指标作为本机规格。

### 3.3 数字链、跟踪、观测量与判决

| ID | 问题及数学模型 | 差分/标定边界 | 当前覆盖 |
|---|---|---|---|
| D01 | FIR/CIC：$y_a[n]=\sum_l h_a[l]x_a[n-l]$，线性相位延迟 $(N_h-1)/(2f_s)$ | 相同响应/时间对齐才抵消；PRN 频差耦合残差 | 有 129 tap FIR 与流式状态；理想公共响应 |
| D02 | DDC NCO：$y=x e^{-j(\phi_{a0}+2\pi\delta f_at)}$ | 重置产生新初相；频差变成漂移 | 有 NCO/连续处理，无硬件重置误差模型 |
| D03 | 定点：$y=Q_B(h*x)$，溢出 $y=\operatorname{wrap}_B(v)$ 或饱和 | 舍入可小误差，回绕溢出须故障注入 | 有部分量化，无完整 FPGA 位真模型 |
| D04 | 抽取状态：$y_a[k]=(h*x)[kD+r_a]$ | $r_a$ 不同即不同采样时刻；启动滤波不稳态 | 有流式一致性检查，未覆盖所有硬件重配置 |
| D05 | 混叠：$Y(f)=\sum_m H(f+mf_s)X(f+mf_s)$ | 带外能量折入后无法由后端简单去除 | 有理想 DFE，未覆盖实机抗混叠不足 |
| D06 | 传输/标签：$y_a[n]=s_a x_{\pi(a)}[g_a(n)]$，可另有 $x^*$ | $g$ 表示丢/重/乱序，$\pi$ 表示通道错配，I/Q 交换需复数变换 | 有单天线录波异常证据，无完整多通道故障注入 |
| T01 | 初始过渡：$\delta\mathbf x_{k+1}=\mathbf A\delta\mathbf x_k+\mathbf w_k$ | 稳态公式不适用捕获阶段；需准入等待 | 已有捕获/跟踪及生命周期原型 |
| T02 | PLL 热噪声：$\sigma_L=\frac\lambda{2\pi}\sqrt{\frac{B_n}{C/N_0}(1+\frac1{2T(C/N_0)})}$ | 常见小误差模型，$C/N_0$ 用线性 Hz；实际环路传递/相关性须识别 | 有 RF 跟踪噪声，无实机有效带宽标定 |
| T03 | 动态应力：$E(s)=[1-H_{PLL}(s)]\Phi_{in}(s)$ | 加速度/jerk 与环路阶数相关；不同通道响应破坏共模 | 跟踪实现存在，动态压力覆盖有限 |
| T04 | 码载波耦合：$Z\propto R(\delta\tau)e^{j\delta\phi}$，$e_{DLL}\simeq e_0/D'(0)$ | 非对称频响/多径令 $R$ 复数；接近零斜率不能线性化 | 有码跟踪，未全面失配验证 |
| T05 | 比特/半周：$Z'=(-1)^{h_k}Z$，$\Delta L=\lambda h_k/2$ | 二倍相位隐藏半周，不证明弧连续 | Costas 与真实电文已有验证 |
| T06 | 周跳/重捕：$\beta_k=\beta_{k-1}+\lambda\Delta N_k+\lambda\Delta h_k/2$，重捕可另置初相 | 取余可隐藏跳变；需新弧及独立准入 | 已有 RF 中断/重捕证据，自动认证未实现 |
| T07 | 错分量：$q_k\in\{\text{真、假、反射、旁峰}\}$，$L=L^{(q_k)}+e$ | 高 C/N0、小残差均不证明 $q$ 为真 | 已有混合/捕获失败场景，未实现真实性认证 |
| O01 | 观测定义：$L=-\lambda[\phi_i+2\pi f_i(t-t_i)]/(2\pi)$ | 符号、相位单位、时间插值必须一致 | 已审计 NCO 重建约定 |
| O02 | 弧起点：$L=\widetilde L+\beta_{as}^{arc}$ | 仅满足初相条件才令 $\beta=\lambda N+\lambda h/2$；与 T05/T06 去重 | 已明确未知初相，未普遍固定整周 |
| O03 | 输出精度/映射：$\widehat L=\alpha L_{\pi(a),\pi(s)}+o+\epsilon_Q$ | 单位倍数/错星号是结构性错误；16 位相位舍入界 $\lambda/2^{17}$ | 软件接口可查，无全面设备协议验证 |
| O04 | 时间轴：$t_{obs}=t_{sample}+\delta t_{tag}$，$e\simeq\dot L\delta t_{tag}$ | 时标误标不同于物理 RF 延迟；正确时间戳下 CPU 延迟不改变相位 | 有共同历元约定，真实多设备同步待验证 |
| S01 | 几何补偿：$e\simeq-\delta\mathbf b^T\mathbf u-\mathbf b^T\delta\mathbf u$ | 阵列姿态/星历/PVT 方向误差需分别传播 | 已有几何误差实验，无实机方向标定 |
| S02 | 标定残差：$e=B(t)-\widehat B(t_0)$ | 温度、重启、安装改变及不可信参考可使标定失效 | 有可信 LOS 假设下标定，未实现参考认证 |
| S03 | 窗口/阈值：$v=\mathbf W\mathbf d$，$\Sigma_v=\mathbf W\Sigma_d\mathbf W^T$，$\gamma=F_0^{-1}(\alpha)$ | 去趋势改变信号；阈值依赖统计量及场景分布 | 有模型校准和窗口判决，硬件阈值未标定 |
| S04 | 非唯一归因：$p(y)=\sum_zp(y\mid z)p(z)$ | 多径/故障/欺骗可同样拟合，不能把异常自动标注为欺骗 | 有三态门控，无完整故障归因 |
| S05 | 选择/PVT 反馈：$\widehat x=\arg\min_x\sum_iw_i(y)r_i(x)^2$ | 权重与误差相关；用被污染 PVT 校准可吸收攻击 | 已有选择消融，当前检测不使用真值几何 |

PLL 小误差模型与跟踪结构依据 [ESA PLL](https://gssc.esa.int/navipedia/index.php/Phase_Lock_Loop_(PLL))。公式不意味着当前代码的等效 $B_n$ 已经测得，也不覆盖失锁/错误锁定。

### 3.4 传播环境

| ID | 问题及数学模型 | 差分/标定边界 | 当前覆盖 |
|---|---|---|---|
| E01 | 多径/NLOS：$Z=Z_0+\sum_qZ_q$，$m=\mathcal M[\arg(Z/Z_0)]$ | $Z_0$ 消失时该 LOS 参照模型不适用；不能引用弱多径界 | 有多径与混合实验，非全部环境 |
| E02 | 星钟/大气/星端相位：$e_{as}=-c\delta t_s+T_{as}-I_{as}+w_{as}$ | 真正公共部分 SD 消失；梯度项 $\Delta e\simeq\nabla e^T\mathbf b$ 仍在 | 简化传播，未做全环境模型 |
| E03 | 干扰/欺骗：$v=v_{GNSS}+j(t)+\sum_qv_{fake,q}(t)$ | 压制、同步替代、分布式源产生不同假设；同源判据不覆盖所有攻击 | 已有全星/部分/混合场景，无完整威胁覆盖 |

## 4. 新增 16 项：当前识别与测向必须补充的约束

| ID | 问题及数学模型 | 工程含义与当前状态 |
|---|---|---|
| X01 | 近场：$\rho_a-\rho_0=\lVert R\mathbf u-\mathbf b_a\rVert-R$ | 球面波不能总按平面波拟合；当前 DOA 采用远场，未做近场距离联合估计 |
| X02 | 相位多解：$z_a=e^{j2\pi h d_a/\lambda}$，$d_a\equiv d_a+n_a\lambda/h$ | 当前 $h=2$ 容忍半周但增加模糊；扫描候选不等于唯一方向 |
| X03 | 阵列秩/镜像：$\operatorname{rank}(\mathbf B)\le2$，平面上 $\mathbf B\mathbf u(a,e)=\mathbf B\mathbf u(a,-e)$ | 两天线约束一个投影；平面三/四天线仍有上下镜像，地平线/天顶局部病态；已报告 |
| X04 | 坐标接口：$\mathbf u_{nav}=[\cos e\sin a,\cos e\cos a,\sin e]^T$，$a_{math}=\pi/2-a_{nav}$ | 天线通用模块与 RF/DOA 的角度约定不同，需显式转换；不能据此断言正式实验已有错误 |
| X05 | 未知通道偏置与 DOA 混淆：$\mathbf y=-\mathbf B\mathbf u+\boldsymbol\beta$ | 任取 $\mathbf u'$，令 $\beta'=\beta+\mathbf B(\mathbf u'-\mathbf u)$ 得相同数据；已验证反例 |
| X06 | 参考星污染：$D_{s,r}=-\mathbf B(\mathbf u_s-\mathbf u_r)+e$ | 若仍代入真实星历 $\mathbf u_r^{true}$，方向项误差为 $\mathbf B(\mathbf u_r-\mathbf u_r^{true})$；已有假参考失败证据 |
| X07 | 真假相干混合：$Z=Z_T(1+r e^{j\psi})$，$e=\mathcal M[\arg(1+r e^{j\psi})]$ | 小拟合残差可能对应错误方向；已有 6 dB 混合反例 |
| X08 | 多发射源/分布式欺骗：$Z_{as}=\sum_qA_{asq}e^{j2\pi\mathbf p_a^T\mathbf u_q/\lambda+j\phi_{sq}}$ | 每星来源不同或合成空间场可逃离单同源假设；尚无完备对抗验证 |
| X09 | 子集扫描/多重判决：$T_{min}=\min_{C\in\mathcal C}T_C$，$P_{FA}=P_0(T_{min}<\gamma)$ | 必须校准整个扫描流程；不可给每簇独立 1% 后仍称总体 1%；当前有整体簇校准 |
| X10 | 分布迁移：$F_0(t\mid\xi_{deploy})\ne F_0(t\mid\xi_{cal})$ | $\xi$ 包括天线、天空、噪声、姿态、干扰；1 m 实验不能沿用原阵列阈值；实机待标定 |
| X11 | 非随机缺测：$P(V=1\mid e,z)\ne P(V=1)$ | 只统计有效窗会隐藏失锁；需同时报告有效覆盖率、unknown、检出与保护率；已有部分门控统计 |
| X12 | 时间相关/重叠窗：$\operatorname{Cov}(T_k,T_{k+1})\ne0$ | 8 个同记录窗口不是 8 次独立攻击；用独立种子/场次做置信区间；现有 RF 样本有限 |
| X13 | 告警延迟与输出暴露：$t_{alarm}=t_{onset}+t_{wait}+t_{confirm}+t_{compute}$ | 保护指标应含攻击后继续输出的时长与误差；正确时间戳不消告警延迟；当前门控不恢复导航 |
| X14 | 码/载波硬件差异：$B^P=c\tau_g$，$B^L=\mathcal M[\phi(f_c)]$ | 群延迟与中心相位是不同量，多频还需各频校准；当前不能以码定位正常认证载波 |
| X15 | 标定污染/真值泄漏：$\widehat\theta=\arg\min_\theta\lVert y-H\theta\rVert_W^2$ | 攻击分量在 $\operatorname{col}(H)$ 内会被标定吸收；训练、阈值、测试及真值评价必须分离 |
| X16 | 真实性不可观：$p(y\mid H_{true})=p(y\mid H_{fake})$ 时无仅依赖 $y$ 的可分检验 | 空间一致、PVT 自洽和质量恢复都不是真实性证明；本系统输出“异常/无证据/未知”，不是认证 |

X 项是补充的系统约束，与原条目有交叉引用关系，不是再增加 16 份独立噪声。

## 5. 必须展开的非线性、可观测性与判据

### 5.1 混合信号必须经过相关器再计算相位

矩形相干积分的简化模型为

$$Z_{as}=\sum_q A_{asq}R_{sq}(\delta\tau_{asq})\operatorname{sinc}(\pi\delta f_{asq}T)e^{j\pi\delta f_{asq}T}e^{j\phi_{asq}}+\eta_{as},\tag{12}$$

这里 $\operatorname{sinc}(x)=\sin x/x$，$R$ 可为受滤波影响的复相关函数；跨 PRN 泄漏需要在求和中增加其他码分量。对于一真一假，相关后幅度比是 $r$，并不自动等于发射功率比的平方根。仅在 $r\ll1$ 时才可近似 $e\simeq-\lambda r\sin\psi/(2\pi)$。$r\approx1,\psi\approx\pi$ 时相消，幅度近零，相位不稳定，不能继续套小高斯相位误差。

当前 RF 的 6 dB 混合案例中，三/四天线能生成低残差候选，但候选集合最近真方向误差仍约 30°/26°。这是已有仿真反例，不是实机误差规格；说明“拟合好”并不验证单来波模型或真实性。

### 5.2 DOA 局部误差传播不解决全局模糊

导航方位角约定下

$$\mathbf u_a=[\cos e\cos a,-\cos e\sin a,0]^T,\quad
\mathbf u_e=[-\sin e\sin a,-\sin e\cos a,\cos e]^T,$$
$$\mathbf J=-\mathbf B[\mathbf u_a\ \mathbf u_e],\quad
\delta\boldsymbol\vartheta\simeq(\mathbf J^T\Sigma_d^{-1}\mathbf J)^{-1}\mathbf J^T\Sigma_d^{-1}\delta\mathbf d.\tag{13}$$

此式只在正确模糊分支、局部可辨识、小误差条件下成立。若信息矩阵奇异，不能给出有限唯一方向协方差；若存在其他相位分支，应报告全部候选/覆盖率，而非只报告用真值挑出的最近候选精度。当前搜索是数值候选搜索，未证明穷尽所有连续空间解。

共同未知通道偏置的不可辨识反例见 X05。星间 DD 去掉该偏置，但对同源全星也去掉 $\mathbf B\mathbf u$；因此没有“既完全免校准又仅靠同源 DD 得到绝对方向”的免费收益。

### 5.3 近场与几何误差

当 $\lVert\mathbf b\rVert/R\ll1$，

$$\rho_a-\rho_0=-\mathbf b_a^T\mathbf u+\frac{\lVert\mathbf b_a\rVert^2-(\mathbf b_a^T\mathbf u)^2}{2R}+O(\lVert\mathbf b_a\rVert^3/R^2).\tag{14}$$

本手机对角线约 177.9 mm，$R=10$ m 时二阶项最大约 1.58 mm；1 m 基线在 10 m 时约 50 mm。后者仍应使用精确球面式评估，而不能把二阶展开视作精确值。这是指定距离的几何示例，不代表目前实验已包含近场。

基线方向误差 $\delta\theta$ 对长度 $b$ 的最坏投影误差界为 $2b\sin(\delta\theta/2)$。固定偏置可在一个方向校准为零，却在另一方向留下误差：$e(\mathbf u)-e(\mathbf u_0)=-\delta\mathbf b^T(\mathbf u-\mathbf u_0)$。

### 5.4 现有检测统计量的误差入口

对每基线定义 $z_s=\exp(j4\pi d_s/\lambda)$。原集中度分数 $T=1-\lVert S^{-1}\sum_sz_s\rVert$ 越小越疑似同源；对任意公共偏置 $B$，$z_s\mapsto e^{j4\pi B/\lambda}z_s$，故 $T$ 原本已经不变。全星 DD 分数为

$$T_{DD}=\frac1{\binom S2}\sum_{s<r}\left[1-\cos\frac{4\pi(d_s-d_r)}\lambda\right]
=\frac S{S-1}\left(1-\left\lVert\frac1S\sum_sz_s\right\rVert^2\right).\tag{15}$$

单基线固定星数时二者是单调变换；多基线平均/子集选择后不必完全等价。新增 DD 并不会创造独立空间信息；必须重标定其量纲范围与阈值。当前实验检出改善很小，不能把“抵消偏置”解读为必然大幅提升检出率。

## 6. 如何形成可用的误差预算与验证闭环

先拟合物理参数，再传播到目标量。对 $g(\theta)$，小误差下 $\Sigma_g=J_g\Sigma_\theta J_g^T$；对检测概率、模糊分支及失锁状态，用复数/跟踪仿真与状态统计，不能只传播 RMS。预算至少分为：未消确定性偏差界、随机协方差、状态失效概率与持续时间、模型不确定性。没有可识别的概率分布时保留范围或待测，不伪造总 RMS。

| 测试阶段 | 激励与可识别量 | 必须保存/防止混淆 | 通过后才能宣称 |
|---|---|---|---|
| 公共源分路注入 | 同源扫频、功率、温度、重启；拟合 $H_a(f,T,g)$、相对噪声谱 | 分配器/线缆自身响应；不同启动的离散分支；原始 IQ 与时间戳 | 在该频段/状态下的通道校准残差 |
| 数字时间轴故障注入 | skew、采样频差、丢重样、NCO 重置、抽取相位 | 区分物理采样时刻与标签；检查通道/PRN 映射 | 时间与弧状态故障能被发现或安全停判 |
| 安装后方向扫描 | 已知方向/极化，足够覆盖角度；测复方向图与实际基线 | 远场条件或球面模型；参考仪器误差；留出独立角度 | 标定角域内的方向相关残差 |
| 环境/动态循环 | 温变、弯折、人体/罩体、振动、遮挡、AGC/阻塞 | 参数历史与滞回；重复场次；不要在线把未知攻击吸收入标定 | 标定有效范围与失效触发条件 |
| 跟踪闭环 | 扫 C/N0、动态、路径延迟/相位、真假相关后幅度比 | 失锁、错峰、弧 ID、unknown；噪声相关时间 | 条件跟踪误差与恢复边界 |
| 端到端留出验证 | 多天空、姿态、2–4 天线、纯真/纯假/部分/混合/多源 | 标定与测试独立种子/场次；按场次自举或独立二项统计 | 该测试分布内的虚警、检出、延迟、覆盖率 |
| 测向验证 | 已知源位置，覆盖距离/角度/模糊分支 | 输出候选数量、镜像、置信域、错误接受率，不用真值选解部署 | 候选覆盖与在已声明先验下的唯一解可靠性 |

公共注入无法独立辨识天线方向图；单一来波方向不能同时辨识任意通道偏置和方向；单个温度点不能识别温度系数。标定设计矩阵必须达到所估参数所需秩，且保留未参与拟合的数据验证。

建议仿真参数接口统一保存 `parameter/value/unit/distribution/correlation_group/state_dependency/calibration_id/provenance`。`provenance` 区分“人工压力值、公开示例、硬件实测”；本轮只建立文档和解析核验，没有修改 RF、检测或测向默认行为。

## 7. 当前代码覆盖与尚缺证据

- [rf.py](../../python_sim_simplified/rf.py)：级联等效噪声、固定幅相、简化限幅、ADC 量化；缺真实带内频响、I/Q、时钟抖动与互调标定。
- [antenna.py](../../python_sim_simplified/antenna.py)：可配置几何、复响应/极化及耦合；模块能力不等于当前理想天线正式 RF 场景已验证这些误差。
- [dfe.py](../../python_sim_simplified/dfe.py)、[positioning.py](../../python_sim_simplified/positioning.py)：数字处理和载波定义；真实多设备时间误差尚未闭环。
- [sdcp_arc.py](../../python_sim_simplified/sdcp_arc.py)：生命周期原型依赖外部质量与相位状态证据，不能自行证明重捕后的参考有效。
- [spatial_spoofing.py](../../python_sim_simplified/spatial_spoofing.py)：数据集中度/簇检测、模型阈值、三态门控；不是完整故障分类或真实性认证。
- [spatial_direction.py](../../python_sim_simplified/spatial_direction.py)、[spatial_reference_direction.py](../../python_sim_simplified/spatial_reference_direction.py)：DD、偏置校准假设、方向候选与参考星法；近场、可信参考来源及自动可靠选解未完成。

既有 RF 测向校准使用 LOS 数据；其中一个测试种子与标定种子相同，不能把全部结果称为独立标定留出验证。现有真实数据为单天线录波，不足以验证实机多天线相位一致性。本文不改变上述证据边界。

## 8. 本轮解析核验与交付

下表为脚本计算的量级示例，所有条件均为指定输入，不是硬件规格；“幅值”不包含约定相关的正负号。

| 指定条件 | 等效量或结果 | 使用限制 |
|---|---|---|
| L1 额外相位 1° | 距离幅值 0.528594 mm | 相位弧有效 |
| RF 传播时延 10 ps | 距离 +2.997925 mm | 纯延迟、按 RF 载频 |
| ADC 输入 4.092 MHz，采样偏斜 10 ps | 距离幅值 0.007787 mm | 单音近似，不能换用 L1 载频 |
| NCO 2000 Hz，时间标签偏差 1 μs | 距离幅值 0.380587 mm | 时间插值误差，不是 CPU 耗时 |
| 差分群延迟 1 ns，PRN 频差 5 kHz | 相对距离变化幅值 0.000951 mm | 中心频率固定相位另需标定 |
| PLL 15 Hz、1 ms、40 dB-Hz | 单路 1.201945 mm；独立双路 SD 1.699808 mm | 小误差近似，非代码已测等效带宽 |
| AR(1) 相关系数 0.7，10 个样本 | 均值方差为单样本的 0.415505；有效样本约 2.40671 | 有限和结果；长窗近似仅给 1.76471，不宜代替 |
| 手机对角线，源距 10 m，方向垂直该基线 | 球面精确曲率项 1.582356 mm | 平面波遗漏项；非已有实验注入 |

本轮 **33 项解析与文档一致性检查通过**，覆盖下面所列关系。未新增检测率结论。

解析脚本：[verify_models.py](data/2026-09-14-engineering-error-model/verify_models.py)；运行输出：[analytical_examples.json](data/2026-09-14-engineering-error-model/analytical_examples.json)。执行方式：在工程根目录运行 `python docs/reports/data/2026-09-14-engineering-error-model/verify_models.py`。

核验内容：49 项旧 ID 完整性、16 项新增 ID、局部文档链接、相位/时间单位换算、SD/DD 共参考协方差、有限窗时间相关方差、DD 集中度恒等式、公共偏置抵消与同源方向消失、未知偏置不可辨识、DOA Jacobian 有限差分、近场精确式与展开式、相干混合非线性、平面镜像以及相位分支不变性。它验证本文解析关系，不替代真实 RF 或硬件实验；本轮未重复执行未修改的接收机全套回归。

优先补模顺序：**通道频温漂与启动状态 → 采样/时间轴 → 安装后复方向图与几何 → 真假混合/多径相关器闭环 → 独立硬件阈值与候选覆盖验证**。这样才能判断哪类误差可以标定、哪类可由差分抵消、哪类会直接破坏当前检测或测向假设。
