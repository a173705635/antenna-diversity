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
MODES = ("single", "egc", "mvdr")
SEEDS = (42, 43, 44)
DURATION_MS = 100
POWER_TRIM_MS = 10
SHOW_PLOTS = True
OUTPUT_DIR = INTERFERENCE_RESULTS / "validation"

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
TARGET_ASSUMED_DIRECTION = Direction(30.0, 60.0)
JAMMER_DIRECTION = Direction(80.0, 10.0)


def _base_config(scene: str):
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
    cases = []
    for scene in SCENES:
        base = _base_config(scene)
        for phase in PHASE_SCAN_DEG:
            prefix = f"{scene}/phase_{phase}"
            common = dict(scene_label=scene, scan_label="DFE channel phase",
                          scan_value=phase, scan_unit="deg")
            ideal_checks = ("ideal_noise",) if scene == "noise_only" else ()
            cases.append(ValidationCase(
                f"{prefix}/baseline", deepcopy(base), series_label="baseline",
                checks=ideal_checks, **common,
            ))
            cases.append(ValidationCase(
                f"{prefix}/uncalibrated", deepcopy(base), series_label="uncalibrated",
                dfe_phase_deg={PHASE_ANTENNA: phase}, **common,
            ))
            cases.append(ValidationCase(
                f"{prefix}/calibrated", deepcopy(base), series_label="calibrated",
                dfe_phase_deg={PHASE_ANTENNA: phase},
                dfe_estimated_phase_deg={PHASE_ANTENNA: phase},
                compensate_dfe_phase=True, reference_case=f"{prefix}/baseline",
                checks=("phase_pair",), **common,
            ))
        for azimuth in TARGET_AZIMUTH_SCAN_DEG:
            cfg = deepcopy(base)
            cfg.signal.true_direction = Direction(azimuth, cfg.signal.true_direction.elevation_deg)
            cases.append(ValidationCase(
                f"{scene}/direction_{azimuth}", cfg, scene_label=scene,
                series_label="true direction", scan_label="target true azimuth",
                scan_value=azimuth, scan_unit="deg",
            ))
        for gain_db in ELEMENT_GAIN_SCAN_DB:
            cfg = deepcopy(base)
            cfg.phone.antenna_model.true_state.amplitude_error_db[ELEMENT_GAIN_ANTENNA] = gain_db
            cases.append(ValidationCase(
                f"{scene}/gain_{gain_db}", cfg, scene_label=scene,
                series_label="true element gain", scan_label=f"{ELEMENT_GAIN_ANTENNA} gain",
                scan_value=gain_db, scan_unit="dB",
            ))
    return cases


def build_settings() -> ValidationSettings:
    selected = []
    for scene in SCENES:
        if "phase" in SELECTED_SERIES:
            for phase in PHASE_SCAN_DEG:
                selected.extend(f"{scene}/phase_{phase}/{series}"
                                for series in ("baseline", "uncalibrated", "calibrated"))
        if "direction" in SELECTED_SERIES:
            selected.extend(f"{scene}/direction_{azimuth}"
                            for azimuth in TARGET_AZIMUTH_SCAN_DEG)
        if "gain" in SELECTED_SERIES:
            selected.extend(f"{scene}/gain_{gain_db}"
                            for gain_db in ELEMENT_GAIN_SCAN_DB)
    return ValidationSettings(
        seeds=SEEDS, duration_ms=DURATION_MS, power_trim_ms=POWER_TRIM_MS,
        selected_cases=SELECTED_CASES or tuple(selected),
        output_dir=OUTPUT_DIR, show_plots=SHOW_PLOTS,
    )
