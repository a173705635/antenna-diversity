"""日常只改本文件顶部条件；运行器与接收算法不随扫描点复制。

方向/增益示例默认不选中。真实方向与名义方向是独立字段：改真实角度
不会让接收机自动知道它。DFE 相位仅旋转完整通道；估计值是外部给定量。
"""

from copy import deepcopy

from core.config import Direction
from core.presets import default_config
from experiments.paths import INTERFERENCE_RESULTS_DIR
from experiments.interference.validation_support import ValidationCase, ValidationSettings


# 常用编辑区：天线顺序第一根是 SINGLE 参考；可用任意两至四根物理天线。
BATCH = "pattern_gain_first44"  # 改为 "legacy" 可运行原相位/方向/增益系列。
ANTENNAS = ("TL", "BR")
MODES = ("single", "egc", "mvdr", "blind_null")
SEEDS = tuple(range(42, 72))
# 每个用例独立生成采样；配对条件保持相同 seed 和物理配置，以复现相同噪声。
DURATION_MS = 100
# 功率统计首尾各舍弃此时长，避开滤波瞬态；不裁掉送入捕获跟踪的数据。
EVALUATION_EDGE_TRIM_MS = 10
SHOW_PLOTS_FLAG = True
OUTPUT_DIR = INTERFERENCE_RESULTS_DIR / "pattern_gain_first44"
WORKERS = 4  # 首批 TC 互相独立；Windows 上按进程并行、逐条记录进度。
# 每次成功生成数据后刷新此目录；需要保留不同实验时修改目录名。

# 单因素扫描；PHASE_ANTENNA 按物理名字定位，不使用固定行号。
PHASE_ANTENNA = "BR"
PHASE_SCAN_DEG = (0, 30, 60, 90, 120, 180)
# legacy 相位扫描：无误差基准、未补偿、准确补偿；选 calibrated 会自动带上基准。
PHASE_VARIANTS = ("baseline", "uncalibrated")
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
    base_cfg = default_config()
    base_cfg.phone.active_antennas = ANTENNAS
    base_cfg.diversity.modes = MODES
    base_cfg.interferers = [] if scene == "noise_only" else base_cfg.interferers[:1]
    base_cfg.signal.true_direction = deepcopy(TARGET_TRUE_DIRECTION)
    base_cfg.signal.assumed_direction = deepcopy(TARGET_ASSUMED_DIRECTION)
    for interferer in base_cfg.interferers:
        interferer.direction = deepcopy(JAMMER_DIRECTION)
    return base_cfg


def _pattern_gain_cases() -> list[ValidationCase]:
    """按文档 TC-001～044 建立单方法、单物理条件，不继承上一条的改动。"""
    case_definitions = []
    for gain_db in (0, -3, -6, -10, -20):
        case_definitions.append((gain_db, 0, "single", None))
    modes = ("egc", "egc_strict", "mvdr")
    case_definitions.extend((0, 0, mode, None) for mode in modes)
    for gain_db in (-3, -6, -10, -20):
        case_definitions.extend((gain_db, gain_db, mode, f"TC-{6 + modes.index(mode):03d}")
                           for mode in modes)
    for gain_db in (0, -3, -6, -10, -20, -40):
        case_definitions.extend((0, gain_db, mode, None) for mode in modes)
    for gain_db, swapped_reference_start in ((-6, 27), (-20, 33)):
        case_definitions.extend((gain_db, 0, mode, f"TC-{swapped_reference_start + modes.index(mode):03d}")
                           for mode in modes)

    cases = []
    for seed in SEEDS:
        for case_number, (tl_gain_db, br_gain_db, mode, reference_case_name) in enumerate(case_definitions, 1):
            case_cfg = _base_config("noise_only")
            case_cfg.random_seed = seed
            case_cfg.signal.duration_ms = DURATION_MS
            case_cfg.tracking.max_epochs = DURATION_MS
            case_cfg.diversity.modes = (mode,)
            # I0：只有真实/名义阵元电压增益同步改变，几何相位和热噪声不动。
            antenna_gains_db = {"TL": tl_gain_db, "BR": br_gain_db}
            case_cfg.phone.antenna_model.true_state.amplitude_error_db = antenna_gains_db.copy()
            case_cfg.phone.antenna_model.assumed_state.amplitude_error_db = antenna_gains_db.copy()
            if 2 <= case_number <= 5:
                reference_case_name = "TC-001"
            cases.append(ValidationCase(
                name=f"TC-{case_number:03d}", config=case_cfg, scene_label="noise_only",
                series_label=mode, scan_label="test case", scan_value=case_number,
                scan_unit="TC", reference_case=reference_case_name, checks=("pattern_gain",),
            ))
    return cases


def build_cases() -> list[ValidationCase]:
    """选择具名条件或系列，并按 seed 顺序生成独立的最终配置。"""
    if BATCH == "pattern_gain_first44":
        return _pattern_gain_cases()
    cases = []
    known_case_names = set()
    selected_case_names = set(SELECTED_CASES)
    phase_series_flag = not SELECTED_CASES and "phase" in SELECTED_SERIES
    direction_series_flag = not SELECTED_CASES and "direction" in SELECTED_SERIES
    gain_series_flag = not SELECTED_CASES and "gain" in SELECTED_SERIES
    for scene in SCENES:
        base_cfg = _base_config(scene)
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
            if ((phase_series_flag and ("baseline" in PHASE_VARIANTS or "calibrated" in PHASE_VARIANTS))
                    or baseline_case.name in selected_case_names or calibrated_case.name in selected_case_names):
                cases.append(baseline_case)
            if (phase_series_flag and "uncalibrated" in PHASE_VARIANTS) or uncalibrated_case.name in selected_case_names:
                cases.append(uncalibrated_case)
            if (phase_series_flag and "calibrated" in PHASE_VARIANTS) or calibrated_case.name in selected_case_names:
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


def build_settings() -> ValidationSettings:
    """生成仅用于统计、保存与绘图的批次设置。"""
    return ValidationSettings(
        evaluation_edge_trim_ms=EVALUATION_EDGE_TRIM_MS, output_dir=OUTPUT_DIR, show_plots_flag=SHOW_PLOTS_FLAG,
        workers=WORKERS if BATCH == "pattern_gain_first44" else 1,
    )
