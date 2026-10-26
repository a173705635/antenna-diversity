"""日常只改本文件顶部条件；运行器与接收算法不随扫描点复制。

DFE 相位旋转完整通道，接收机的名义目标响应保持固定。
"""

from copy import deepcopy

import numpy as np

from core.config import CouplingPath, Direction, RFStage, WidebandInterferer
from core.presets import default_config
from experiments.paths import INTERFERENCE_RESULTS_DIR
from experiments.interference.validation_support import ValidationCase, ValidationSettings


# 所有实验共用种子、时长、并行数、统计去边和绘图开关。
# 选择实验类型后，输出固定写入 results/interference/<实验类型>/。
EXPERIMENT_TYPE = "phase_offset"  # 另可选 "rf_agc"、"rf_group_delay"、"antenna_temperature"、"antenna_gain"、"mutual_coupling"、"phase_center_offset"。
SCENES = ("noise_only", "single_cw")  # 可加 "single_wideband"；所有实验共用此选择。
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
RF_GROUP_DELAY_ANTENNA = "BR"
RF_GROUP_DELAY_SCAN_NS = (0, 10, 20, 50, 100, 200)

# 示例固定 VGA 档位，并非真实芯片数据。每档 NF + Gain = 54 dB：以 290 K
# 标准输入计，VGA 输出噪声不变；级联后仍须计算每路真实 RF 输入等效噪温。
AGC_ANTENNA = "BR"
AGC_VGA_STATES = (
    RFStage("VGA", 20.0, 34.0),
    RFStage("VGA", 26.0, 28.0),
    RFStage("VGA", 32.0, 22.0),
    RFStage("VGA", 38.0, 16.0),
    RFStage("VGA", 44.0, 10.0),
)

# 温偏指固定温度相对参考温度的偏差；系数为示例假设，每个用例温度恒定。
# 两个系列分别只改指定天线真实增益或真实相位，名义状态始终固定于参考温度。
TEMPERATURE_OFFSET_ANTENNA = "BR"
TEMPERATURE_REFERENCE_C = 25.0
TEMPERATURE_SCAN_C = (-20.0, 0.0, 25.0, 40.0, 60.0, 80.0)
TEMPERATURE_GAIN_SLOPE_DB_PER_C = -0.02
TEMPERATURE_PHASE_SLOPE_DEG_PER_C = 1.0

# 互耦系数是复电压比；0 表示无互耦，不对应有限 dB 值。
COUPLING_STRENGTH_SCAN_V_PER_V = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
COUPLING_STRENGTH_PHASE_DEG = 0
COUPLING_PHASE_STRENGTH_V_PER_V = 0.3
COUPLING_PHASE_SCAN_DEG = (0, 45, 90, 135, 180, 225, 270, 315)

# 两根天线各自沿目标来波视线移动；零点是两条曲线共用的物理基准。
PHASE_CENTER_LOS_SCAN_MM = (-40, -20, -10, -5, 0, 5, 10, 20, 40)

# 所有实验共用基础目标方向。
# 真实方向与接收机名义假设可分别修改；扫描只覆盖下面标明的单因素。
TARGET_TRUE_DIRECTION = Direction(30.0, 60.0)
# Direction 的顺序为方位角、仰角，单位均为度；ASSUMED 是接收机知道的方向。
TARGET_ASSUMED_DIRECTION = Direction(30.0, 60.0)
JAMMER_DIRECTION = Direction(80.0, 10.0)


def _base_config(scene: str):
    """生成指定场景；宽带场景仅保留预设的 2 MHz、-90 dBm、零偏频干扰。"""
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
    elif scene == "single_wideband":
        base_cfg.interferers = [source for source in base_cfg.interferers
                                if isinstance(source, WidebandInterferer)]
    base_cfg.signal.true_direction = deepcopy(TARGET_TRUE_DIRECTION)
    base_cfg.signal.assumed_direction = deepcopy(TARGET_ASSUMED_DIRECTION)
    for interferer in base_cfg.interferers:
        interferer.direction = deepcopy(JAMMER_DIRECTION)
    return base_cfg


def _pattern_gain_cases(scene: str) -> list[ValidationCase]:
    """17 个唯一物理增益条件；所有方法共用同一条件的前端。"""
    case_definitions = [(gain_db, 0) for gain_db in range(0, -11, -1)]
    for gain_db in (-3, -6, -10, -20, -40):
        case_definitions.append((0, gain_db))
    case_definitions.append((-20, 0))

    cases = []
    first, second = ACTIVE_ANTENNAS[:2]
    for case_number, (first_gain_db, second_gain_db) in enumerate(case_definitions, 1):
        case_cfg = _base_config(scene)
        # 额外激活阵元保留预设增益，仅改变扫描的前两根。
        for state in (case_cfg.phone.antenna_model.true_state,
                      case_cfg.phone.antenna_model.assumed_state):
            state.amplitude_error_db[first] = first_gain_db
            state.amplitude_error_db[second] = second_gain_db
        cases.append(ValidationCase(
            name=f"{scene}/TC-{case_number:03d}", config=case_cfg, scene_label=scene,
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


def _coupling_cases(scene: str) -> list[ValidationCase]:
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
        case_cfg = _base_config(scene)
        # 真实双向路径使用相同复系数；接收机仍按无互耦名义模型工作。
        true_db = 20 * np.log10(strength) if strength else None
        case_cfg.phone.antenna_model.true_state.coupling_paths = _coupling_paths(true_db, phase)
        case_cfg.phone.antenna_model.assumed_state.coupling_paths = ()
        cases.append(ValidationCase(
            name=f"{scene}/MC-{number:03d}", config=case_cfg, scene_label=scene,
            series_label=series, scan_label=scan_label,
            scan_value=scan_value, scan_unit=scan_unit,
        ))
    return cases


def _phase_center_cases(scene: str) -> list[ValidationCase]:
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
        case_cfg = _base_config(scene)
        if series != "baseline":
            origin_m = np.asarray(case_cfg.phone.phase_centers_m[series])
            case_cfg.phone.phase_centers_m[series] = tuple(origin_m + offset_mm * line_of_sight * 1e-3)
        cases.append(ValidationCase(
            name=f"{scene}/PC-{number:03d}", config=case_cfg, scene_label=scene,
            series_label=series, scan_label="LOS offset", scan_value=offset_mm, scan_unit="mm",
        ))
    return cases


def _phase_offset_cases(scene: str) -> list[ValidationCase]:
    """每个场景和 DFE 相位点运行一个独立完整用例；零度是自然基准。"""
    cases = []
    for phase_deg in PHASE_SCAN_DEG:
        cases.append(ValidationCase(
            name=f"{scene}/phase_{phase_deg:g}", config=_base_config(scene),
            scene_label=scene, series_label="phase",
            scan_label="DFE channel phase", scan_value=phase_deg, scan_unit="deg",
            dfe_phase_deg={PHASE_ANTENNA: phase_deg},
        ))
    return cases


def _rf_group_delay_cases(scene: str) -> list[ValidationCase]:
    """每场景独立扫描包络与 RF 载波的同一物理时延；名义响应不补偿。"""
    cases = []
    for delay_ns in RF_GROUP_DELAY_SCAN_NS:
        case_cfg = _base_config(scene)
        case_cfg.rf.group_delay_offset_ns[RF_GROUP_DELAY_ANTENNA] = delay_ns
        cases.append(ValidationCase(
            name=f"{scene}/rf_delay_{delay_ns:g}ns", config=case_cfg,
            scene_label=scene, series_label="rf_group_delay",
            scan_label="RF delay", scan_value=delay_ns, scan_unit="ns",
        ))
    return cases


def _rf_agc_cases(scene: str) -> list[ValidationCase]:
    """逐个固定 VGA 档位运行完整用例；只改变指定物理链路。"""
    cases = []
    for vga in AGC_VGA_STATES:
        case_cfg = _base_config(scene)
        reference_stages = case_cfg.rf.stages
        selected_stages = tuple(vga if stage.name == "VGA" else stage
                                for stage in reference_stages)
        case_cfg.rf.stages_by_antenna[AGC_ANTENNA] = selected_stages
        gain_difference_db = sum(stage.gain_db for stage in selected_stages) - sum(
            stage.gain_db for stage in reference_stages)
        cases.append(ValidationCase(
            name=f"{scene}/agc_gain_{gain_difference_db:g}db", config=case_cfg,
            scene_label=scene, series_label="rf_agc", scan_label="AGC gain difference",
            scan_value=gain_difference_db, scan_unit="dB",
        ))
    return cases


def _antenna_temperature_cases(scene: str) -> list[ValidationCase]:
    """每场景分开验证温偏引起的真实增益、相位变化；参考温度各运行一次。"""
    cases = []
    for series in ("gain", "phase"):
        for temperature_c in TEMPERATURE_SCAN_C:
            case_cfg = _base_config(scene)
            gain_slope = TEMPERATURE_GAIN_SLOPE_DB_PER_C if series == "gain" else 0.0
            phase_slope = TEMPERATURE_PHASE_SLOPE_DEG_PER_C if series == "phase" else 0.0
            temperature_offset_c = temperature_c - TEMPERATURE_REFERENCE_C
            gain_delta_db = gain_slope * temperature_offset_c
            phase_delta_deg = phase_slope * temperature_offset_c
            true_state = case_cfg.phone.antenna_model.true_state
            # 在预设基准上叠加变化；另一幅相字段、全部名义配置和噪声链保持原样。
            if series == "gain":
                true_state.amplitude_error_db[TEMPERATURE_OFFSET_ANTENNA] = (
                    true_state.amplitude_error_db.get(TEMPERATURE_OFFSET_ANTENNA, 0.0)
                    + gain_delta_db
                )
            else:
                true_state.phase_error_deg[TEMPERATURE_OFFSET_ANTENNA] = (
                    true_state.phase_error_deg.get(TEMPERATURE_OFFSET_ANTENNA, 0.0)
                    + phase_delta_deg
                )
            cases.append(ValidationCase(
                name=f"{scene}/{series}_temperature_{temperature_c:g}c",
                config=case_cfg, scene_label=scene, series_label=series,
                scan_label="temperature", scan_value=temperature_c, scan_unit="°C",
                temperature_offset={
                    "antenna": TEMPERATURE_OFFSET_ANTENNA,
                    "reference_temperature_c": TEMPERATURE_REFERENCE_C,
                    "temperature_c": temperature_c,
                    "gain_slope_db_per_c": gain_slope,
                    "phase_slope_deg_per_c": phase_slope,
                    "gain_delta_db": gain_delta_db,
                    "phase_delta_deg": phase_delta_deg,
                },
            ))
    return cases


def build_cases() -> list[ValidationCase]:
    """按实验类型和场景生成模板，再按种子复制独立配置。"""
    unsupported_modes = set(MODES) - {"single", "blind_null", "egc", "mvdr"}
    if unsupported_modes:
        raise ValueError(f"Unsupported validation modes: {sorted(unsupported_modes)}")
    builders = {
        "antenna_gain": _pattern_gain_cases,
        "mutual_coupling": _coupling_cases,
        "phase_center_offset": _phase_center_cases,
        "phase_offset": _phase_offset_cases,
        "rf_group_delay": _rf_group_delay_cases,
        "rf_agc": _rf_agc_cases,
        "antenna_temperature": _antenna_temperature_cases,
    }
    if EXPERIMENT_TYPE not in builders:
        raise ValueError(f"Unknown experiment type: {EXPERIMENT_TYPE}")
    templates = []
    for scene in SCENES:
        templates.extend(builders[EXPERIMENT_TYPE](scene))
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
