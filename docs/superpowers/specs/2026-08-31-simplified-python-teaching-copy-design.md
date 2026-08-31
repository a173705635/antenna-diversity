# Python 端到端仿真精简教学副本设计

日期：2026-08-31

## 目标

在不修改现有 `python_sim/` 的前提下，新建独立的 `python_sim_simplified/` 教学副本。副本保留 GNSS 天线分集与空域抗干扰仿真的完整核心数据流和主要数学关系，删除面向异常输入、生产部署、断点恢复和大规模研究编排的工程保护代码，使读者能够顺序理解 RF → DFE → 空域合并 → 捕获 → 跟踪。

副本只面向受控、有效的研究输入，不代替原工程，也不承诺对错误配置、缺失文件、退化矩阵或极端数值安全运行。

## 目录结构

```text
python_sim_simplified/
├─ config.py
├─ antenna.py
├─ gnss.py
├─ rf.py
├─ dfe.py
├─ diversity.py
├─ acquisition.py
├─ tracking.py
├─ metrics.py
├─ plotting.py
├─ pipeline.py
├─ run_end_to_end.py
├─ requirements.txt
└─ README.md
```

各文件只承担一种职责：配置、天线、信号场景、模拟前端、数字前端、空域合并、捕获、跟踪、指标、绘图、流水线和运行入口相互分离。模块之间传递 NumPy 数组和简单字典，不复制原工程的大型结果对象、严格快照和诊断模型。

## 保留的核心能力

### 配置与场景

- GPS L1 C/A、复数等效 IF、码延迟、载波 Doppler 和码 Doppler；
- 手机 `TL/TR/BL/BR` 中任意不重复的 2–4 个角点；
- 一颗目标卫星和 0–3 个具有独立方向、功率、频偏、相位与时间门的静态 CW 干扰；
- 可直接编辑的仿真时长、采样率、RF、DFE、捕获和跟踪参数。

### 完整天线模型

- 手机三维相位中心和空间传播相位；
- `ideal`、`parametric`、`measured` 三种方向图来源；
- 复数双极化 Jones 响应、姿态、遮挡、电压增益方向图、复数互耦和逐通道幅相响应；
- true/assumed 两套状态，允许展示目标导向矢量和通道校准失配；
- measured CSV 继续使用 `element_name,frequency_hz,azimuth_deg,elevation_deg,h_az_real,h_az_imag,h_el_real,h_el_imag` 列格式，并保留规则网格插值。

measured CSV 使用直接文件读取和数组构造；文件缺失、列错误或网格错误由 Python、CSV 或 NumPy 原始异常直接暴露，不再包装、归类或生成哈希摘要。

### 接收机核心链路

- RF 级联增益与噪声系数、热噪声、I/Q 限幅和复数 ADC；
- 公共 NCO DDC、FIR、群延迟裁切、抽取和 DFE 输出量化；
- SINGLE、EGC、loaded MVDR 和捕获前 Blind Null；
- Doppler × 码相位 FFT 捕获；
- Early/Prompt/Late DLL、FLL-assisted PLL、浮点 Carrier/Code NCO 和 C/N0；
- RF/DFE/合并频谱、空间响应、捕获热力图和跟踪曲线。

## 删除和简化规则

### 删除工程保护

- 配置对象的类型、范围、有限值和 shape 全量校验；
- 包装底层异常的 `try/except`；
- 文件存在性、SHA-256、测量摘要、严格 schema 和重复行审计；
- 零输入功率、零分母、退化特征向量、奇异归一化和非有限输出的显式保护分支；
- Blind Null 的 `unsafe_reference_projection` 等安全旁路；
- 绘图清理、临时文件发布、原子写入和失败恢复；
- JSON checkpoint、manifest、断点续跑、排名、九表十二图报告和七阶段 studies；
- 顺序 Monte Carlo、批量扫描和 Carrier NCO 30/43-bit 专项对比。

算法所需的对角加载、FIR设计、相关运算、环路滤波和数值公式不视为保护代码，必须保留。

### 保留算法判决

- `ideal/parametric/measured` 天线模型选择；
- SINGLE/EGC/MVDR/Blind Null分支选择；
- Blind Null 的主特征模态检测门限；
- 捕获峰值比判决和捕获成功后才进入跟踪；
- DLL/FLL/PLL历元更新及干扰时间门。

原则是删除“输入不正常时如何安全退化”，保留“有效输入下算法如何工作”。

## Blind Null 教学实现

精简函数保留以下可顺序阅读的主线：

```text
DFE多通道样本
→ 样本协方差
→ 通道功率归一化相关矩阵
→ 最大特征值与主特征向量
→ 干扰检测分数
→ 正交投影参考通道
→ 复数权值
→ wᴴX合并输出
```

检测未达到门限时保留返回参考通道的算法判决；检测通过后直接投影，不再检查零功率、退化向量、参考投影功率、非有限权值或输出。

## 与原工程的关系

- `python_sim/` 是可信、带保护和完整测试的研究工程；
- `python_sim_simplified/` 是独立教学实现，不被原包导入，也不反向导入原包内部函数；
- 两边可以共享 NumPy、SciPy 和 Matplotlib 依赖，但不共享源码；
- 精简副本的默认有效场景应与原工程采用相同物理参数、随机种子和主要公式；
- 不要求错误输入行为、诊断字段或报告文件与原工程一致。

## 验证与完成标准

1. `python_sim_simplified/run_end_to_end.py` 可独立完成场景、RF、DFE、四种合并、捕获、跟踪、C/N0和绘图；
2. 默认有效场景下输出有限，至少一个抗干扰分支成功捕获并跟踪；
3. measured CSV最小样例能够加载并参与天线响应计算；
4. 两通道固定相位误差能够进入复数协方差和Blind Null权值；
5. 精简副本包含清晰 README，明确无保护假设、运行方法、模块阅读顺序和与原工程的差异；
6. 原工程完整测试仍通过；
7. Git差异中 `python_sim/` 下的原有文件没有修改，只新增精简目录、设计/计划和必要的精简版验证文件。
