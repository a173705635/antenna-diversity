"""日常只改本文件顶部条件；运行器与接收算法不随扫描点复制。

方向/增益示例默认不选中。真实方向与名义方向是独立字段：改真实角度
不会让接收机自动知道它。DFE 相位仅旋转完整通道；估计值是外部给定量。
"""

from copy import deepcopy

import numpy as np

from core.config import CouplingPath, Direction
from core.presets import default_config
from experiments.paths import INTERFERENCE_RESULTS_DIR
from experiments.interference.validation_support import ValidationCase, ValidationSettings


# 所有实验共用种子、时长、并行数和统计去边；绘图开关按实验区分。
# 选择实验类型后，输出固定写入 results/interference/<实验类型>/。
EXPERIMENT_TYPE = "phase_direction_gain"  # 可选 "antenna_gain"、"mutual_coupling"、"phase_center_offset"。
SEEDS = tuple(range(42, 43))  # 所有实验共用；单次运行设为 (42,)。
# 每个用例独立生成采样；配对条件保持相同 seed 和物理配置，以复现相同噪声。
# 功率统计首尾各舍弃此时长，避开滤波瞬态；不裁掉送入捕获跟踪的数据。
EVALUATION_EDGE_TRIM_MS = 10

# 每个用例独立生成 DURATION_MS 毫秒采样；WORKERS 控制并行进程数。
DURATION_MS = 100
WORKERS = 4
GENERATE_PLOTS_FLAG = True
SHOW_PLOTS_FLAG = True
# 所有条件共用激活天线和方法；第一根是 SINGLE 参考，第二根用于双阵元条件。
ACTIVE_ANTENNAS = ("TL", "BR")
MODES = ("single", "blind_null")  # 可显式加入 "egc"、"mvdr"；顺序即运行与结果顺序。

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

# 所有实验共用基础目标方向。
# 真实方向与接收机名义假设可分别修改；扫描只覆盖下面标明的单因素。
TARGET_TRUE_DIRECTION = Direction(30.0, 60.0)
# Direction 的顺序为方位角、仰角，单位均为度；ASSUMED 是接收机知道的方向。
TARGET_ASSUMED_DIRECTION = Direction(30.0, 60.0)
JAMMER_DIRECTION = Direction(80.0, 10.0)


def _base_config(scene: str):
    """生成指定天线/方法的基础场景；noise_only 无干扰，single_cw 保留首个 CW。"""
    base_cfg = default_config()
    base_cfg.phone.active_antennas = ACTIVE_ANTENNAS
    base_cfg.diversity.modes = MODES
    base_cfg.signal.duration_ms = DURATION_MS
    # max_epochs 是跟踪历元上限，duration_ms 是输入时长；同数值保持原充分大上限。
    base_cfg.tracking.max_epochs = DURATION_MS
    if scene == "noise_only":
        base_cfg.interferers = []
    elif scene == "single_cw":
        base_cfg.interferers = base_cfg.interferers[:1]
    base_cfg.signal.true_direction = deepcopy(TARGET_TRUE_DIRECTION)
    base_cfg.signal.assumed_direction = deepcopy(TARGET_ASSUMED_DIRECTION)
    for interferer in base_cfg.interferers:
        interferer.direction = deepcopy(JAMMER_DIRECTION)
    return base_cfg


def _pattern_gain_cases() -> list[ValidationCase]:
    """21 个唯一物理增益条件；所有方法共用同一条件的前端。"""
    case_definitions = [(gain_db, 0) for gain_db in range(0, -11, -1)]
    for gain_db in (-3, -6, -10, -20):
        case_definitions.append((gain_db, gain_db))
    for gain_db in (-3, -6, -10, -20, -40):
        case_definitions.append((0, gain_db))
    case_definitions.append((-20, 0))

    cases = []
    first, second = ACTIVE_ANTENNAS[:2]
    for case_number, (first_gain_db, second_gain_db) in enumerate(case_definitions, 1):
        case_cfg = _base_config("noise_only")
        # 额外激活阵元保留预设增益，仅改变扫描的前两根。
        for state in (case_cfg.phone.antenna_model.true_state,
                      case_cfg.phone.antenna_model.assumed_state):
            state.amplitude_error_db[first] = first_gain_db
            state.amplitude_error_db[second] = second_gain_db
        cases.append(ValidationCase(
            name=f"TC-{case_number:03d}", config=case_cfg, scene_label="noise_only",
            scan_label="gain condition",
        ))
    return cases


def _coupling_paths(magnitude_db: float | None, phase_deg: float | None) -> tuple[CouplingPath, ...]:
    """复数电压传递系数按两个有向路径写入；None 明确表示不互耦。"""
    if magnitude_db is None:
        return ()
    return (
        CouplingPath(ACTIVE_ANTENNAS[0], ACTIVE_ANTENNAS[1], magnitude_db, phase_deg),
        CouplingPath(ACTIVE_ANTENNAS[1], ACTIVE_ANTENNAS[0], magnitude_db, phase_deg),
    )


def _coupling_cases() -> list[ValidationCase]:
    """12 个物理/名义条件，各方法共用相同前端。"""
    conditions = (
        (None, None, None, None),
        (-30.0, 0.0, -30.0, 0.0),
        (-20.0, 0.0, -20.0, 0.0),
        (-10.0, 0.0, -10.0, 0.0),
        (-10.0, 90.0, -10.0, 90.0),
        (-10.0, 180.0, -10.0, 180.0),
        (-10.0, 270.0, -10.0, 270.0),
        (-10.0, 90.0, None, None),
        (-10.0, 90.0, -13.0, 90.0),
        (-10.0, 90.0, -7.0, 90.0),
        (-10.0, 90.0, -10.0, 60.0),
        (-10.0, 90.0, -10.0, 120.0),
    )
    cases = []
    for number, (true_db, true_phase, assumed_db, assumed_phase) in enumerate(conditions, 1):
        cfg = _base_config("noise_only")
        cfg.phone.antenna_model.true_state.coupling_paths = _coupling_paths(true_db, true_phase)
        cfg.phone.antenna_model.assumed_state.coupling_paths = _coupling_paths(assumed_db, assumed_phase)
        cases.append(ValidationCase(
            name=f"MC-{number:03d}", config=cfg,
            scene_label="noise_only", scan_label="coupling condition",
        ))
    return cases


def _phase_center_cases() -> list[ValidationCase]:
    """16个真实/名义位置条件；沿共用真实方向建立偏移。"""
    direction = TARGET_TRUE_DIRECTION
    azimuth = np.deg2rad(direction.azimuth_deg)
    elevation = np.deg2rad(direction.elevation_deg)
    line_of_sight = np.array([np.cos(elevation) * np.cos(azimuth),
                              np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
    transverse = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])
    along = lambda millimetres: tuple(millimetres * line_of_sight)
    across = lambda millimetres: tuple(millimetres * transverse)
    # 每个向量单位mm；未列出的阵元位置保持Mate70角点预设。
    first, second = ACTIVE_ANTENNAS[:2]
    conditions = (
        ("PC-001", {}, {}),
        ("PC-002", {first: along(10), second: along(10)}, {}),
        ("PC-003", {second: across(10)}, {}),
        ("PC-004", {second: along(1)}, {}),
        ("PC-005", {second: along(5)}, {}),
        ("PC-006", {second: along(10)}, {}),
        ("PC-007", {second: along(20)}, {}),
        ("PC-008", {second: along(-10)}, {}),
        ("PC-009", {second: along(10)}, {second: along(10)}),
        ("PC-010", {second: along(10)}, {second: along(5)}),
        ("PC-011", {second: along(10)}, {second: along(15)}),
        ("PC-012", {second: along(10)}, {second: along(-10)}),
        ("PC-013", {first: along(-5), second: along(5)}, {}),
        ("PC-014", {second: (10, 0, 0)}, {}),
        ("PC-015", {second: (0, 10, 0)}, {}),
        ("PC-016", {second: (0, 0, 10)}, {}),
    )
    cases = []
    for case_id, true_offsets_mm, assumed_offsets_mm in conditions:
        cfg = _base_config("noise_only")
        for antenna, offset_mm in true_offsets_mm.items():
            origin_m = np.asarray(cfg.phone.phase_centers_m[antenna])
            cfg.phone.phase_centers_m[antenna] = tuple(origin_m + np.asarray(offset_mm) * 1e-3)
        for antenna, offset_mm in assumed_offsets_mm.items():
            origin_m = np.asarray(cfg.phone.assumed_phase_centers_m[antenna])
            cfg.phone.assumed_phase_centers_m[antenna] = tuple(origin_m + np.asarray(offset_mm) * 1e-3)
        cases.append(ValidationCase(
            name=case_id, config=cfg, scene_label="noise_only",
            scan_label="phase center condition",
        ))
    return cases


def _phase_direction_gain_cases() -> list[ValidationCase]:
    """按所选系列构造相位、方向和增益模板。"""
    cases = []
    for scene in SCENES:
        base_cfg = _base_config(scene)
        if "phase" in SELECTED_SERIES:
            for phase_deg in PHASE_SCAN_DEG:
                # 每个横轴点固定三组；各组使用独立完整配置。
                phase_case_prefix = f"{scene}/phase_{phase_deg}"
                common_case_fields = dict(
                    scene_label=scene, scan_label="DFE channel phase",
                    scan_value=phase_deg, scan_unit="deg",
                )
                # ideal_checks = ("ideal_noise",) if scene == "noise_only" else ()
                baseline_case = ValidationCase(
                    f"{phase_case_prefix}/baseline", deepcopy(base_cfg), series_label="baseline",
                    # checks=ideal_checks,  # 分量 SNR 核验暂停。
                    **common_case_fields,
                )
                uncalibrated_case = ValidationCase(
                    f"{phase_case_prefix}/uncalibrated", deepcopy(base_cfg), series_label="uncalibrated",
                    dfe_phase_deg={PHASE_ANTENNA: phase_deg}, **common_case_fields,
                )
                calibrated_case = ValidationCase(
                    f"{phase_case_prefix}/calibrated", deepcopy(base_cfg), series_label="calibrated",
                    # 这里故意令估计相位等于真实相位，验证补偿闭环，不是自动估相。
                    dfe_phase_deg={PHASE_ANTENNA: phase_deg},
                    dfe_estimated_phase_deg={PHASE_ANTENNA: phase_deg},
                    compensate_dfe_phase_flag=True, **common_case_fields,
                )
                cases.extend((baseline_case, uncalibrated_case, calibrated_case))
        if "direction" in SELECTED_SERIES:
            for azimuth_deg in TARGET_AZIMUTH_SCAN_DEG:
                # 扫描真实入射方向；名义方向保持固定，仅供诊断与显式对照。
                case_cfg = deepcopy(base_cfg)
                case_cfg.signal.true_direction = Direction(azimuth_deg, case_cfg.signal.true_direction.elevation_deg)
                direction_case = ValidationCase(
                    f"{scene}/direction_{azimuth_deg}", case_cfg, scene_label=scene,
                    series_label="true direction", scan_label="target true azimuth",
                    scan_value=azimuth_deg, scan_unit="deg",
                )
                cases.append(direction_case)
        if "gain" in SELECTED_SERIES:
            for gain_db in ELEMENT_GAIN_SCAN_DB:
                # dB 增益作用于阵元响应，6.02 dB 约为电压两倍；名义增益保持不变。
                case_cfg = deepcopy(base_cfg)
                case_cfg.phone.antenna_model.true_state.amplitude_error_db[ELEMENT_GAIN_ANTENNA] = gain_db
                gain_case = ValidationCase(
                    f"{scene}/gain_{gain_db}", case_cfg, scene_label=scene,
                    series_label="true element gain", scan_label=f"{ELEMENT_GAIN_ANTENNA} gain",
                    scan_value=gain_db, scan_unit="dB",
                )
                cases.append(gain_case)

    return cases


def build_cases() -> list[ValidationCase]:
    """按实验类型选择对应的用例构造函数。"""
    unsupported_modes = set(MODES) - {"single", "blind_null", "egc", "mvdr"}
    if unsupported_modes:
        raise ValueError(f"Unsupported validation modes: {sorted(unsupported_modes)}")
    builders = {
        "antenna_gain": _pattern_gain_cases,
        "mutual_coupling": _coupling_cases,
        "phase_center_offset": _phase_center_cases,
        "phase_direction_gain": _phase_direction_gain_cases,
    }
    if EXPERIMENT_TYPE not in builders:
        raise ValueError(f"Unknown experiment type: {EXPERIMENT_TYPE}")
    templates = builders[EXPERIMENT_TYPE]()
    cases = []
    for seed in SEEDS:
        for template in templates:
            case = deepcopy(template)
            case.config.random_seed = seed
            cases.append(case)
    return cases


def build_settings() -> ValidationSettings:
    """共用统计窗口、输出推导和并行数；仅绘图选项随实验类型变化。"""
    if EXPERIMENT_TYPE in ("mutual_coupling", "phase_center_offset"):
        generate_plots_flag = False
        show_plots_flag = False
    else:
        generate_plots_flag = GENERATE_PLOTS_FLAG
        show_plots_flag = SHOW_PLOTS_FLAG
    return ValidationSettings(
        evaluation_edge_trim_ms=EVALUATION_EDGE_TRIM_MS,
        output_dir=INTERFERENCE_RESULTS_DIR / EXPERIMENT_TYPE,
        experiment_type=EXPERIMENT_TYPE,
        show_plots_flag=show_plots_flag,
        workers=WORKERS,
        generate_plots_flag=generate_plots_flag,
    )
