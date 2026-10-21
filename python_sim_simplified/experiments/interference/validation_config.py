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


# 所有实验共用种子、时长、并行数、统计去边和绘图开关。
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
MODES = ("single", "blind_null", "egc", "mvdr")  # 后两种为预设/已知响应对照。

# 单因素扫描；PHASE_ANTENNA 按物理名字定位，不使用固定行号。
PHASE_ANTENNA = "BR"
PHASE_SCAN_DEG = (0, 30, 60, 90, 120, 180)
TARGET_AZIMUTH_SCAN_DEG = (30, 40, 50)
ELEMENT_GAIN_ANTENNA = "BR"
ELEMENT_GAIN_SCAN_DB = (0.0, 6.020599913279624)

# 互耦系数是复电压比；0 表示无互耦，不对应有限 dB 值。
COUPLING_STRENGTH_SCAN_V_PER_V = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
COUPLING_STRENGTH_PHASE_DEG = 0
COUPLING_PHASE_STRENGTH_V_PER_V = 0.3
COUPLING_PHASE_SCAN_DEG = (0, 45, 90, 135, 180, 225, 270, 315)

# 两根天线各自沿目标来波视线移动；零点是两条曲线共用的物理基准。
PHASE_CENTER_LOS_SCAN_MM = (-40, -20, -10, -5, 0, 5, 10, 20, 40)

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
    """17 个唯一物理增益条件；所有方法共用同一条件的前端。"""
    case_definitions = [(gain_db, 0) for gain_db in range(0, -11, -1)]
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
    """固定名义无耦合，分别扫描真实耦合幅度和相位，共用交点。"""
    cases = []
    shared_point_flag = (COUPLING_PHASE_STRENGTH_V_PER_V in COUPLING_STRENGTH_SCAN_V_PER_V
                         and COUPLING_STRENGTH_PHASE_DEG in COUPLING_PHASE_SCAN_DEG)
    conditions = [
        ("strength_phase" if shared_point_flag and strength == COUPLING_PHASE_STRENGTH_V_PER_V
         else "strength",
         strength, COUPLING_STRENGTH_PHASE_DEG, "coupling voltage ratio", strength, "V/V")
        for strength in COUPLING_STRENGTH_SCAN_V_PER_V
    ]
    conditions.extend(
        ("phase", COUPLING_PHASE_STRENGTH_V_PER_V, phase,
         "coupling phase", phase, "deg")
        for phase in COUPLING_PHASE_SCAN_DEG
        if not (shared_point_flag and phase == COUPLING_STRENGTH_PHASE_DEG)
    )
    for number, (series, strength, phase, scan_label, scan_value, scan_unit) in enumerate(conditions, 1):
        case_cfg = _base_config("noise_only")
        # 真实双向路径使用相同复系数；接收机仍按无互耦名义模型工作。
        true_db = 20 * np.log10(strength) if strength else None
        case_cfg.phone.antenna_model.true_state.coupling_paths = _coupling_paths(true_db, phase)
        case_cfg.phone.antenna_model.assumed_state.coupling_paths = ()
        cases.append(ValidationCase(
            name=f"MC-{number:03d}", config=case_cfg, scene_label="noise_only",
            series_label=series, scan_label=scan_label,
            scan_value=scan_value, scan_unit=scan_unit,
        ))
    return cases


def _phase_center_cases() -> list[ValidationCase]:
    """前两根天线分别沿真实来波视线移动，名义坐标始终固定。"""
    direction = TARGET_TRUE_DIRECTION
    azimuth = np.deg2rad(direction.azimuth_deg)
    elevation = np.deg2rad(direction.elevation_deg)
    line_of_sight = np.array([np.cos(elevation) * np.cos(azimuth),
                              np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
    first, second = ACTIVE_ANTENNAS[:2]
    conditions = [("baseline", 0)]
    conditions.extend((first, offset) for offset in PHASE_CENTER_LOS_SCAN_MM if offset != 0)
    conditions.extend((second, offset) for offset in PHASE_CENTER_LOS_SCAN_MM if offset != 0)
    cases = []
    for number, (series, offset_mm) in enumerate(conditions, 1):
        case_cfg = _base_config("noise_only")
        if series != "baseline":
            origin_m = np.asarray(case_cfg.phone.phase_centers_m[series])
            case_cfg.phone.phase_centers_m[series] = tuple(origin_m + offset_mm * line_of_sight * 1e-3)
        cases.append(ValidationCase(
            name=f"PC-{number:03d}", config=case_cfg, scene_label="noise_only",
            series_label=series, scan_label="LOS offset", scan_value=offset_mm, scan_unit="mm",
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
    """共用统计窗口、输出推导、并行数和显式绘图开关。"""
    return ValidationSettings(
        evaluation_edge_trim_ms=EVALUATION_EDGE_TRIM_MS,
        output_dir=INTERFERENCE_RESULTS_DIR / EXPERIMENT_TYPE,
        experiment_type=EXPERIMENT_TYPE,
        show_plots_flag=SHOW_PLOTS_FLAG,
        workers=WORKERS,
        generate_plots_flag=GENERATE_PLOTS_FLAG,
    )
