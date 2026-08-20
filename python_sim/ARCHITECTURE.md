# 手机角点空域抗干扰仿真架构

![手机角点空域抗干扰仿真架构](docs/simulation_architecture.svg)

图片文件：[`docs/simulation_architecture.svg`](docs/simulation_architecture.svg) 是可编辑矢量主文件；[`docs/simulation_architecture.png`](docs/simulation_architecture.png) 用于报告和演示；实现细节见 [`DESIGN.md`](DESIGN.md)。

## 一句话定位

这是一个以 HUAWEI Mate 70 外形尺寸为几何边界、支持任意 2–4 个角点和分层窄带天线模型的 GPS L1 C/A 端到端研究链，用于比较 SINGLE、EGC 和 MVDR 在 0–3 个静态 CW 干扰下的空间响应、捕获与跟踪结果。默认仍是旧理想角点，参数化模型只用于合成敏感性研究。

## 中央数据路径

```text
GPS L1 C/A + 0–3 CW interferers
→ direction + normalized Jones polarization
→ 2–4 selected phone corners + per-element orientation
→ polarized pattern × blockage × 3-D phase
→ complex port coupling → channel amplitude / phase
→ RF / complex ADC
→ coherent DFE
→ SINGLE / EGC / loaded MVDR
→ Doppler × code-phase ACQ
→ gated DLL / FLL-assisted PLL TRK
→ branch metrics, spatial scans, sequential Monte Carlo
```

RF 到 DFE 始终保留 `(channels, samples)` 形状，因此同一处理链可接受 2、3 或 4 个激活角点。三种合并器输出单通道样本后共享同一捕获与跟踪配置；只有捕获成功的分支才进入跟踪。

## Studies 研究编排层

研究层位于现有解析路径和完整样本路径之上，不替代其中任何一个：

```text
JSON / preset / explicit CLI overrides
                 │
                 ▼
        resolved immutable configuration
                 │
        ┌────────┴─────────┐
        ▼                  ▼
analytical spatial path    selected full-chain path
baseline / direction /     RF → ADC → DFE → SINGLE/EGC/MVDR
power / error / jammer     → acquisition → tracking / C/N₀
        │                  │
        └────────┬─────────┘
                 ▼
      atomic stage checkpoints + manifest
                 │
                 ▼
 tables / figures / summary / Markdown report
```

`antenna_diversity.studies` 固定按 baseline、single-factor、combined-error、multi-jammer、selection、full-chain、report 七阶段执行。每个成功阶段先原子发布严格 JSON checkpoint，manifest 记录路径、SHA-256、状态和记录数；异常保留先前证据并把当前阶段标为失败。恢复运行只读取完成且哈希有效的前置 checkpoint，不能让早期阶段看到未来数据。

selection 只根据已保存的归一化解析记录选择代表场景；full-chain 只运行这些已选场景。report 再从七阶段 checkpoint 中装配固定 9 表、12 图、摘要和本地链接，其中天线证据来自 baseline 保存的 `antenna_diagnostics` 标量行。**报告阶段绝不重新运行天线模型、解析扫描或 RF/DFE/捕获/跟踪仿真**，因此报告内容与已保存证据保持一致，也不会因制图而消耗新的随机数。

解析 `output_sinr_db` 衡量当前已解析天线模型下的空间功率与协方差关系；完整链的 `acquisition_metric`、捕获/跟踪状态和可空 `final_cn0_db_hz` 衡量样本接收机表现。这些量共享配置和 case ID 以便追溯，但含义不同，架构不把解析 SINR 当作采样 C/N₀。

## 当前已建模

- Mate 70 外形尺寸 160.9 × 75.9 × 7.8 mm，以及中心原点的理想角点坐标；
- 任意不重复的 2–4 角点子集，共 11 种稳定顺序组合；
- 手机坐标中的方位角/仰角和位置点积构成的 3-D 阵列流形；
- `ideal`、`parametric`、`measured` 三种来源，以及每角点独立姿态和双极化 Jones 响应；
- true/assumed 分离的方向图、遮挡、互耦、相位中心、通道响应与目标极化；
- 多区域方向性电压衰减和按激活顺序抽取的固定频率复数互耦矩阵；
- 单星 GPS L1 C/A、码延迟、载波/码 Doppler、等效复数 IF；
- 0–3 个静态 CW 干扰，每个具有独立方向、功率、频偏、相位与起止时间门；
- RF 级联增益/NF、热噪声、限幅、复数 ADC，以及相干 DDC/FIR/抽取/重量化；
- SINGLE、EGC、loaded MVDR，FFT 捕获和 E/P/L DLL + FLL-assisted PLL；
- 功率分解、输出 SINR、协方差诊断、逐干扰增益的解析空间指标和扫描；
- 顺序、种子有序、严格 JSON 的 Monte Carlo 汇总；
- 9 张 CSV 表、12 幅 PNG 图及模型来源/标签/测量摘要证据。

## 当前未建模

- **真实天线位置**：角点是理想相位中心接口，不声称对应 Mate 70 内部天线；
- **真实测量输入**：接口可加载规则双极化 CSV，但仓库不附带 Mate 70 暗室数据或 S 参数；
- 全波机壳/人体散射，以及互耦造成的外部噪声相关性；
- 多径、NLOS、运动和动态 DOA；
- 宽带、扫频、脉冲或其他非 CW 干扰；
- 多颗卫星、多 PRN 并发、导航数据、位/帧同步和 PVT；
- 在线 DOA、鲁棒导向矢量估计和 MVDR 滑窗实时更新；
- 完整硬件非理想，例如相位噪声、I/Q 失衡、时钟抖动和逐级压缩。

## 对外表述边界

推荐：

> 已构建一套以 Mate 70 外形为几何边界的 GPS L1 C/A 空域抗干扰研究链，覆盖任意 2–4 角点、理想/参数化/测量表格天线接口、0–3 个静态 CW 干扰、RF/DFE、SINGLE/EGC/MVDR、捕获跟踪和可追溯七阶段报告。

只有用户提供可追溯测量表时才能称为“测量方向图驱动”；仍不得称为“Mate 70 产品级验证”，因为真实相位中心、平台全波响应、传播环境及硬件标定未被验证。

## 两条基线与一个参数化入口

`default_config()` 是手机理想四角默认；`legacy_ula_config()` 与 `run_legacy_ula.py` 保留原始双阵元半波长 ULA 结论；`parametric_phone_config()` 是明确标注 `synthetic_phone_corner_v1` 的非实测入口。三者并存，使真实性扩展不改变旧基线的含义。
