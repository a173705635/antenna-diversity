"""日常只改本文件顶部条件；运行器与接收算法不随扫描点复制。

方向/增益示例默认不选中。真实方向与名义方向是独立字段：改真实角度
不会让接收机自动知道它。DFE 相位仅旋转完整通道；估计值是外部给定量。
"""

from copy import deepcopy

from core.config import Direction
from core.presets import default_config
from experiments.paths import INTERFERENCE_RESULTS
from experiments.interference.validation_support import ValidationCase, ValidationSettings


# 常用编辑区：天线顺序第一根是 SINGLE 参考；可用任意两至四根物理天线。
ANTENNAS = ("TL", "BR")
MODES = ("single", "egc", "mvdr", "blind_null")
SEEDS = (42, 43, 44)
# 每个用例独立生成采样；配对条件保持相同 seed 和物理配置，以复现相同噪声。
DURATION_MS = 100
# 功率统计首尾各舍弃此时长，避开滤波瞬态；不裁掉送入捕获跟踪的数据。
POWER_TRIM_MS = 10
SHOW_PLOTS = True
OUTPUT_DIR = INTERFERENCE_RESULTS / "validation"
# 每次成功生成数据后刷新此目录；需要保留不同实验时修改目录名。

# 单因素扫描；PHASE_ANTENNA 按物理名字定位，不使用固定行号。
PHASE_ANTENNA = "BR"
PHASE_SCAN_DEG = (0, 30, 60, 90, 120, 180)
TARGET_AZIMUTH_SCAN_DEG = (30, 40, 50)
ELEMENT_GAIN_ANTENNA = "BR"
ELEMENT_GAIN_SCAN_DB = (0.0, 6.020599913279624)

# 默认复现旧相位实验。改为 ("direction",) 或 ("gain",) 即单独运行示例；
# 可同时选择多个具名系列，但不会把不同因素做隐式笛卡尔积。
SELECTED_SERIES = ("phase",)
SCENES = ("noise_only", "single_cw")
# 只跑具名条件时填名字，参考条件会自动加入；空元组使用上面的系列选择。
# 例：("single_cw/phase_90/calibrated",)
SELECTED_CASES = ()

# 基础物理方向与接收机名义假设可分别修改；扫描只覆盖下面标明的单因素。
TARGET_TRUE_DIRECTION = Direction(30.0, 60.0)
# Direction 的顺序为方位角、仰角，单位均为度；ASSUMED 是接收机知道的方向。
TARGET_ASSUMED_DIRECTION = Direction(30.0, 60.0)
JAMMER_DIRECTION = Direction(80.0, 10.0)


def _base_config(scene: str):
    """生成一个场景的基础接收配置；noise_only 去掉干扰，single_cw 保留首个 CW。"""
    cfg = default_config()
    cfg.phone.active_antennas = ANTENNAS
    cfg.diversity.modes = MODES
    cfg.interferers = [] if scene == "noise_only" else cfg.interferers[:1]
    cfg.signal.true_direction = deepcopy(TARGET_TRUE_DIRECTION)
    cfg.signal.assumed_direction = deepcopy(TARGET_ASSUMED_DIRECTION)
    for source in cfg.interferers:
        source.direction = deepcopy(JAMMER_DIRECTION)
    return cfg


def build_cases() -> list[ValidationCase]:
    """选择具名条件或系列，并按 seed 顺序生成独立的最终配置。"""
    cases = []
    known_names = set()
    selected_names = set(SELECTED_CASES)
    phase_series = not SELECTED_CASES and "phase" in SELECTED_SERIES
    direction_series = not SELECTED_CASES and "direction" in SELECTED_SERIES
    gain_series = not SELECTED_CASES and "gain" in SELECTED_SERIES
    for scene in SCENES:
        base = _base_config(scene)
        for phase in PHASE_SCAN_DEG:
            # 每个横轴点放三组：无误差基准、注入但不补偿、注入后准确逆补偿。
            # baseline 不注入 phase，只把相同基准画到各扫描点用于配对比较。
            prefix = f"{scene}/phase_{phase}"
            common = dict(scene_label=scene, scan_label="DFE channel phase",
                          scan_value=phase, scan_unit="deg")
            ideal_checks = ("ideal_noise",) if scene == "noise_only" else ()
            baseline = ValidationCase(
                f"{prefix}/baseline", deepcopy(base), series_label="baseline",
                checks=ideal_checks, **common,
            )
            uncalibrated = ValidationCase(
                f"{prefix}/uncalibrated", deepcopy(base), series_label="uncalibrated",
                dfe_phase_deg={PHASE_ANTENNA: phase}, **common,
            )
            calibrated = ValidationCase(
                f"{prefix}/calibrated", deepcopy(base), series_label="calibrated",
                # 这里故意令估计相位等于真实相位，验证补偿闭环，不是自动估相。
                dfe_phase_deg={PHASE_ANTENNA: phase},
                dfe_estimated_phase_deg={PHASE_ANTENNA: phase},
                compensate_dfe_phase=True, reference_case=f"{prefix}/baseline",
                checks=("phase_pair",), **common,
            )
            known_names.update((baseline.name, uncalibrated.name, calibrated.name))
            # 选准确补偿时先加入同扫描点的基准；重复点名不会重复生成条件。
            if phase_series or baseline.name in selected_names or calibrated.name in selected_names:
                cases.append(baseline)
            if phase_series or uncalibrated.name in selected_names:
                cases.append(uncalibrated)
            if phase_series or calibrated.name in selected_names:
                cases.append(calibrated)
        for azimuth in TARGET_AZIMUTH_SCAN_DEG:
            # 只改真实方向，不改名义导向，研究接收机方向假设不准的影响。
            cfg = deepcopy(base)
            cfg.signal.true_direction = Direction(azimuth, cfg.signal.true_direction.elevation_deg)
            direction = ValidationCase(
                f"{scene}/direction_{azimuth}", cfg, scene_label=scene,
                series_label="true direction", scan_label="target true azimuth",
                scan_value=azimuth, scan_unit="deg",
            )
            known_names.add(direction.name)
            if direction_series or direction.name in selected_names:
                cases.append(direction)
        for gain_db in ELEMENT_GAIN_SCAN_DB:
            # dB 增益作用于阵元响应，6.02 dB 约为电压两倍；名义增益保持不变。
            cfg = deepcopy(base)
            cfg.phone.antenna_model.true_state.amplitude_error_db[ELEMENT_GAIN_ANTENNA] = gain_db
            gain = ValidationCase(
                f"{scene}/gain_{gain_db}", cfg, scene_label=scene,
                series_label="true element gain", scan_label=f"{ELEMENT_GAIN_ANTENNA} gain",
                scan_value=gain_db, scan_unit="dB",
            )
            known_names.add(gain.name)
            if gain_series or gain.name in selected_names:
                cases.append(gain)

    # 具名选择中的拼写错误必须立即显式报出，不能变成空批次。
    for name in SELECTED_CASES:
        if name not in known_names:
            raise KeyError(name)

    prepared = []
    for seed in SEEDS:
        for candidate in cases:
            case = deepcopy(candidate)
            case.config.random_seed = seed
            case.config.signal.duration_ms = DURATION_MS
            case.config.tracking.max_epochs = DURATION_MS
            prepared.append(case)
    return prepared


def build_settings() -> ValidationSettings:
    """生成仅用于统计、保存与绘图的批次设置。"""
    return ValidationSettings(
        power_trim_ms=POWER_TRIM_MS, output_dir=OUTPUT_DIR, show_plots=SHOW_PLOTS,
    )
