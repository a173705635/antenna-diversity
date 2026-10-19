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


# 共用种子和统计去边；时长、绘图、并行数下方按实验分别说明。
# 选择实验类型后，输出固定写入 results/interference/<实验类型>/。
EXPERIMENT_TYPE = "phase_center_offset"  # 可选 "antenna_gain"、"mutual_coupling"、"phase_direction_gain"。
SEEDS = tuple(range(42, 72))  # 所有实验共用；单次运行设为 (42,)。
# 每个用例独立生成采样；配对条件保持相同 seed 和物理配置，以复现相同噪声。
# 功率统计首尾各舍弃此时长，避开滤波瞬态；不裁掉送入捕获跟踪的数据。
EVALUATION_EDGE_TRIM_MS = 10

# 增益与相位/方向/增益实验共用运行设置；首批 TC 可并行执行。
DURATION_MS = 100
GENERATE_PLOTS_FLAG = True
SHOW_PLOTS_FLAG = True
WORKERS = 4
# 相位/方向/增益扫描可选两至四根天线；第一根是 SINGLE 参考。
PHASE_DIRECTION_GAIN_ANTENNAS = ("TL", "BR")
PHASE_DIRECTION_GAIN_MODES = ("single", "mrc", "mvdr", "blind_null")

# 增益、互耦、相位中心条件表固定 TL/BR；不要通过扫描天线设置修改它们。
# 互耦实验独立设置时长和并行数。
COUPLING_DURATION_MS = 100
COUPLING_WORKERS = 4

# 固定相位中心偏移：单独控制时长和并行数，默认不弹图。
PHASE_CENTER_DURATION_MS = 100
PHASE_CENTER_WORKERS = 4

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

# 相位/方向/增益与增益实验共用基础目标方向；互耦/相位中心使用固定方向。
# 真实方向与接收机名义假设可分别修改；扫描只覆盖下面标明的单因素。
TARGET_TRUE_DIRECTION = Direction(30.0, 60.0)
# Direction 的顺序为方位角、仰角，单位均为度；ASSUMED 是接收机知道的方向。
TARGET_ASSUMED_DIRECTION = Direction(30.0, 60.0)
JAMMER_DIRECTION = Direction(80.0, 10.0)


def _base_config(scene: str, antennas: tuple[str, ...], modes: tuple[str, ...]):
    """生成指定天线/方法的基础场景；noise_only 无干扰，single_cw 保留首个 CW。"""
    base_cfg = default_config()
    base_cfg.phone.active_antennas = antennas
    base_cfg.diversity.modes = modes
    base_cfg.interferers = [] if scene == "noise_only" else base_cfg.interferers[:1]
    base_cfg.signal.true_direction = deepcopy(TARGET_TRUE_DIRECTION)
    base_cfg.signal.assumed_direction = deepcopy(TARGET_ASSUMED_DIRECTION)
    for interferer in base_cfg.interferers:
        interferer.direction = deepcopy(JAMMER_DIRECTION)
    return base_cfg


def _pattern_gain_cases() -> list[ValidationCase]:
    """建立增益用例；对照按参数与算法关联，TC编号仅表示本批次顺序。"""
    case_definitions = []
    case_names = {}

    def add_case(tl_gain_db, br_gain_db, mode, reference_key=None):
        """登记条件，并将物理对照键转换为本批次的用例名称。"""
        reference_name = (
            None if reference_key is None else case_names[reference_key]
        )
        case_definitions.append((tl_gain_db, br_gain_db, mode, reference_name))
        # 相同条件可属于不同分类，对照统一引用首次登记的条件。
        case_names.setdefault(
            (tl_gain_db, br_gain_db, mode), f"TC-{len(case_definitions):03d}"
        )

    for gain_db in range(0, -11, -1):
        reference = None if gain_db == 0 else (0, 0, "single")
        add_case(gain_db, 0, "single", reference)
    modes = ("mrc", "mvdr")
    for mode in modes:
        add_case(0, 0, mode)
    # 共同衰减与同算法的零增益基准比较。
    for gain_db in (-3, -6, -10, -20):
        for mode in modes:
            add_case(gain_db, gain_db, mode, (0, 0, mode))
    for gain_db in (0, -3, -6, -10, -20, -40):
        for mode in modes:
            add_case(0, gain_db, mode)
    # 强弱交换与相同增益、相同算法的交换前条件比较。
    for gain_db in (-6, -20):
        for mode in modes:
            add_case(gain_db, 0, mode, (0, gain_db, mode))

    cases = []
    for seed in SEEDS:
        for case_number, (tl_gain_db, br_gain_db, mode, reference_case_name) in enumerate(case_definitions, 1):
            case_cfg = _base_config("noise_only", ("TL", "BR"), (mode,))
            case_cfg.random_seed = seed
            case_cfg.signal.duration_ms = DURATION_MS
            case_cfg.tracking.max_epochs = DURATION_MS
            # I0：只有真实/名义阵元电压增益同步改变，几何相位和热噪声不动。
            antenna_gains_db = {"TL": tl_gain_db, "BR": br_gain_db}
            case_cfg.phone.antenna_model.true_state.amplitude_error_db = antenna_gains_db.copy()
            case_cfg.phone.antenna_model.assumed_state.amplitude_error_db = antenna_gains_db.copy()
            cases.append(ValidationCase(
                name=f"TC-{case_number:03d}", config=case_cfg, scene_label="noise_only",
                series_label=mode, scan_label="test case", scan_value=case_number,
                scan_unit="TC", reference_case=reference_case_name, checks=("pattern_gain",),
            ))
    return cases


def _coupling_paths(magnitude_db: float | None, phase_deg: float | None) -> tuple[CouplingPath, ...]:
    """复数电压传递系数按两个有向路径写入；None 明确表示不互耦。"""
    if magnitude_db is None:
        return ()
    return (
        CouplingPath("TL", "BR", magnitude_db, phase_deg),
        CouplingPath("BR", "TL", magnitude_db, phase_deg),
    )


def _coupling_config(
    seed: int,
    true_db: float | None, true_phase_deg: float | None,
    assumed_db: float | None, assumed_phase_deg: float | None,
):
    """从干净默认配置生成物理和名义互耦独立的双天线、无干扰条件。"""
    cfg = default_config()
    cfg.random_seed = seed
    cfg.signal.duration_ms = COUPLING_DURATION_MS
    cfg.tracking.max_epochs = COUPLING_DURATION_MS
    cfg.phone.active_antennas = ("TL", "BR")
    cfg.interferers = []
    cfg.signal.true_direction = Direction(30.0, 60.0)
    cfg.signal.assumed_direction = Direction(30.0, 60.0)
    cfg.phone.antenna_model.true_state.coupling_paths = _coupling_paths(true_db, true_phase_deg)
    cfg.phone.antenna_model.assumed_state.coupling_paths = _coupling_paths(assumed_db, assumed_phase_deg)
    return cfg


def _coupling_cases() -> list[ValidationCase]:
    """12 个物理/名义条件各运行三种合并方法，编号不随 seed 变化。"""
    conditions = (
        ("MC-008", None, None, None, None),
        ("MC-009", -30.0, 0.0, -30.0, 0.0),
        ("MC-010", -20.0, 0.0, -20.0, 0.0),
        ("MC-011", -10.0, 0.0, -10.0, 0.0),
        ("MC-012", -10.0, 90.0, -10.0, 90.0),
        ("MC-013", -10.0, 180.0, -10.0, 180.0),
        ("MC-014", -10.0, 270.0, -10.0, 270.0),
        ("MC-015", -10.0, 90.0, None, None),
        ("MC-016", -10.0, 90.0, -13.0, 90.0),
        ("MC-017", -10.0, 90.0, -7.0, 90.0),
        ("MC-018", -10.0, 90.0, -10.0, 60.0),
        ("MC-019", -10.0, 90.0, -10.0, 120.0),
    )
    modes = ("single", "mrc", "mvdr")
    cases = []
    for seed in SEEDS:
        for case_id, true_db, true_phase, assumed_db, assumed_phase in conditions:
            for mode in modes:
                cfg = _coupling_config(seed, true_db, true_phase, assumed_db, assumed_phase)
                cfg.diversity.modes = (mode,)
                reference = None if case_id == "MC-008" else f"MC-008-{mode}"
                cases.append(ValidationCase(
                    name=f"{case_id}-{mode}", config=cfg,
                    scene_label="noise_only", series_label=mode,
                    scan_label="coupling condition", scan_value=int(case_id[3:]),
                    scan_unit="MC", reference_case=reference, checks=("mutual_coupling",),
                ))
    return cases


def _phase_center_cases() -> list[ValidationCase]:
    """16个真实/名义位置条件，按同一seed展开SINGLE和MVDR。"""
    direction = Direction(30.0, 60.0)
    azimuth = np.deg2rad(direction.azimuth_deg)
    elevation = np.deg2rad(direction.elevation_deg)
    line_of_sight = np.array([np.cos(elevation) * np.cos(azimuth),
                              np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
    transverse = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])
    along = lambda millimetres: tuple(millimetres * line_of_sight)
    across = lambda millimetres: tuple(millimetres * transverse)
    # 每个向量单位mm；未列出的阵元位置保持Mate70角点预设。
    conditions = (
        ("PC-001", {}, {}),
        ("PC-002", {"TL": along(10), "BR": along(10)}, {}),
        ("PC-003", {"BR": across(10)}, {}),
        ("PC-004", {"BR": along(1)}, {}),
        ("PC-005", {"BR": along(5)}, {}),
        ("PC-006", {"BR": along(10)}, {}),
        ("PC-007", {"BR": along(20)}, {}),
        ("PC-008", {"BR": along(-10)}, {}),
        ("PC-009", {"BR": along(10)}, {"BR": along(10)}),
        ("PC-010", {"BR": along(10)}, {"BR": along(5)}),
        ("PC-011", {"BR": along(10)}, {"BR": along(15)}),
        ("PC-012", {"BR": along(10)}, {"BR": along(-10)}),
        ("PC-013", {"TL": along(-5), "BR": along(5)}, {}),
        ("PC-014", {"BR": (10, 0, 0)}, {}),
        ("PC-015", {"BR": (0, 10, 0)}, {}),
        ("PC-016", {"BR": (0, 0, 10)}, {}),
    )
    cases = []
    for seed in SEEDS:
        for case_id, true_offsets_mm, assumed_offsets_mm in conditions:
            for mode in ("single", "mvdr"):
                cfg = default_config()
                cfg.random_seed = seed
                cfg.signal.duration_ms = PHASE_CENTER_DURATION_MS
                cfg.tracking.max_epochs = PHASE_CENTER_DURATION_MS
                cfg.phone.active_antennas = ("TL", "BR")
                cfg.interferers = []
                cfg.signal.true_direction = deepcopy(direction)
                cfg.signal.assumed_direction = deepcopy(direction)
                cfg.diversity.modes = (mode,)
                for antenna, offset_mm in true_offsets_mm.items():
                    origin_m = np.asarray(cfg.phone.phase_centers_m[antenna])
                    cfg.phone.phase_centers_m[antenna] = tuple(origin_m + np.asarray(offset_mm) * 1e-3)
                for antenna, offset_mm in assumed_offsets_mm.items():
                    origin_m = np.asarray(cfg.phone.assumed_phase_centers_m[antenna])
                    cfg.phone.assumed_phase_centers_m[antenna] = tuple(origin_m + np.asarray(offset_mm) * 1e-3)
                cases.append(ValidationCase(
                    name=f"{case_id}-{mode}", config=cfg,
                    scene_label="noise_only", series_label=mode,
                    scan_label="phase center condition", scan_value=int(case_id[3:]),
                    scan_unit="PC", reference_case=(None if case_id == "PC-001"
                                                     else f"PC-001-{mode}"),
                    checks=("phase_center_noise_only",),
                ))
    return cases


def _phase_direction_gain_cases() -> list[ValidationCase]:
    """按具名条件或系列构造相位、方向和增益用例，并展开各 seed 的独立配置。"""
    cases = []
    known_case_names = set()
    selected_case_names = set(SELECTED_CASES)
    phase_series_flag = not SELECTED_CASES and "phase" in SELECTED_SERIES
    direction_series_flag = not SELECTED_CASES and "direction" in SELECTED_SERIES
    gain_series_flag = not SELECTED_CASES and "gain" in SELECTED_SERIES
    for scene in SCENES:
        base_cfg = _base_config(scene, PHASE_DIRECTION_GAIN_ANTENNAS,
                                PHASE_DIRECTION_GAIN_MODES)
        for phase_deg in PHASE_SCAN_DEG:
            # 每个横轴点放三组：无误差基准、注入但不补偿、注入后准确逆补偿。
            # 基准用例不注入相位误差，只把相同基准画到各扫描点用于配对比较。
            phase_case_prefix = f"{scene}/phase_{phase_deg}"
            common_case_fields = dict(
                scene_label=scene, scan_label="DFE channel phase",
                scan_value=phase_deg, scan_unit="deg",
            )
            ideal_checks = ("ideal_noise",) if scene == "noise_only" else ()
            baseline_case = ValidationCase(
                f"{phase_case_prefix}/baseline", deepcopy(base_cfg), series_label="baseline",
                checks=ideal_checks, **common_case_fields,
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
                compensate_dfe_phase_flag=True, reference_case=f"{phase_case_prefix}/baseline",
                checks=("phase_pair",), **common_case_fields,
            )
            known_case_names.update((baseline_case.name, uncalibrated_case.name, calibrated_case.name))
            # 选准确补偿时先加入同扫描点的基准；重复点名不会重复生成条件。
            if (phase_series_flag or baseline_case.name in selected_case_names
                    or calibrated_case.name in selected_case_names):
                cases.append(baseline_case)
            if phase_series_flag or uncalibrated_case.name in selected_case_names:
                cases.append(uncalibrated_case)
            if phase_series_flag or calibrated_case.name in selected_case_names:
                cases.append(calibrated_case)
        for azimuth_deg in TARGET_AZIMUTH_SCAN_DEG:
            # 只改真实方向，不改名义导向，研究接收机方向假设不准的影响。
            case_cfg = deepcopy(base_cfg)
            case_cfg.signal.true_direction = Direction(azimuth_deg, case_cfg.signal.true_direction.elevation_deg)
            direction_case = ValidationCase(
                f"{scene}/direction_{azimuth_deg}", case_cfg, scene_label=scene,
                series_label="true direction", scan_label="target true azimuth",
                scan_value=azimuth_deg, scan_unit="deg",
            )
            known_case_names.add(direction_case.name)
            if direction_series_flag or direction_case.name in selected_case_names:
                cases.append(direction_case)
        for gain_db in ELEMENT_GAIN_SCAN_DB:
            # dB 增益作用于阵元响应，6.02 dB 约为电压两倍；名义增益保持不变。
            case_cfg = deepcopy(base_cfg)
            case_cfg.phone.antenna_model.true_state.amplitude_error_db[ELEMENT_GAIN_ANTENNA] = gain_db
            gain_case = ValidationCase(
                f"{scene}/gain_{gain_db}", case_cfg, scene_label=scene,
                series_label="true element gain", scan_label=f"{ELEMENT_GAIN_ANTENNA} gain",
                scan_value=gain_db, scan_unit="dB",
            )
            known_case_names.add(gain_case.name)
            if gain_series_flag or gain_case.name in selected_case_names:
                cases.append(gain_case)

    # 具名选择中的拼写错误必须立即显式报出，不能变成空批次。
    for name in SELECTED_CASES:
        if name not in known_case_names:
            raise KeyError(name)

    seeded_cases = []
    for seed in SEEDS:
        for case_template in cases:
            case = deepcopy(case_template)
            case.config.random_seed = seed
            case.config.signal.duration_ms = DURATION_MS
            case.config.tracking.max_epochs = DURATION_MS
            seeded_cases.append(case)
    return seeded_cases


def build_cases() -> list[ValidationCase]:
    """按实验类型选择对应的用例构造函数。"""
    if EXPERIMENT_TYPE == "antenna_gain":
        return _pattern_gain_cases()
    if EXPERIMENT_TYPE == "mutual_coupling":
        return _coupling_cases()
    if EXPERIMENT_TYPE == "phase_center_offset":
        return _phase_center_cases()
    if EXPERIMENT_TYPE == "phase_direction_gain":
        return _phase_direction_gain_cases()
    raise ValueError(f"Unknown experiment type: {EXPERIMENT_TYPE}")


def build_settings() -> ValidationSettings:
    """共用统计窗口和输出推导；仅并行与绘图选项随实验类型变化。"""
    if EXPERIMENT_TYPE == "mutual_coupling":
        workers = COUPLING_WORKERS
        generate_plots_flag = False
        show_plots_flag = False
    elif EXPERIMENT_TYPE == "phase_center_offset":
        workers = PHASE_CENTER_WORKERS
        generate_plots_flag = False
        show_plots_flag = False
    else:
        workers = WORKERS if EXPERIMENT_TYPE == "antenna_gain" else 1
        generate_plots_flag = GENERATE_PLOTS_FLAG
        show_plots_flag = SHOW_PLOTS_FLAG
    return ValidationSettings(
        evaluation_edge_trim_ms=EVALUATION_EDGE_TRIM_MS,
        output_dir=INTERFERENCE_RESULTS_DIR / EXPERIMENT_TYPE,
        show_plots_flag=show_plots_flag,
        workers=workers,
        generate_plots_flag=generate_plots_flag,
    )
