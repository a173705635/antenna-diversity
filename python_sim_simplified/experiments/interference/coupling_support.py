"""互耦专用的独立几何复算、确定性模型检查与跨 seed 汇总。"""

import hashlib

import numpy as np

from core.antenna import antenna_response, build_antenna_model, coupling_matrix
from core.config import CouplingPath, SPEED_OF_LIGHT_MPS
from core.dfe import linear_dfe
from core.gnss import generate_scenario
from core.presets import default_config
from core.rf import filter_rf_input


def _transfer(paths: tuple[CouplingPath, ...], receiver: str, source: str) -> complex:
    """按有向物理名称单独求电压传递，不调用生产矩阵构造函数。"""
    return sum(10 ** (path.magnitude_db / 20) * np.exp(1j * np.deg2rad(path.phase_deg))
               for path in paths if path.receiver == receiver and path.source == source)


def independent_manifold(cfg, *, estimated_flag: bool,
                         include_coupling_flag: bool = True) -> np.ndarray:
    """由阵元位置、方向和显式交叉相加独立计算双通道理想阵列导向。"""
    names = cfg.phone.active_antennas
    direction = cfg.signal.assumed_direction if estimated_flag else cfg.signal.true_direction
    position_map = cfg.phone.assumed_phase_centers_m if estimated_flag else cfg.phone.phase_centers_m
    state = cfg.phone.antenna_model.assumed_state if estimated_flag else cfg.phone.antenna_model.true_state
    azimuth = np.deg2rad(direction.azimuth_deg)
    elevation = np.deg2rad(direction.elevation_deg)
    unit = np.array([np.cos(elevation) * np.cos(azimuth),
                     np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
    open_circuit = {
        name: np.exp(-2j * np.pi * cfg.signal.carrier_hz / SPEED_OF_LIGHT_MPS
                     * np.dot(position_map[name], unit))
        for name in names
    }
    coupled = [open_circuit[receiver] + sum(
        _transfer(state.coupling_paths, receiver, source) * open_circuit[source]
        for source in names if source != receiver
    ) for receiver in names] if include_coupling_flag else [open_circuit[name] for name in names]
    channel_amplitude = (cfg.phone.estimated_channel_amplitude_error_db if estimated_flag
                         else cfg.phone.channel_amplitude_error_db)
    channel_phase = (cfg.phone.estimated_channel_phase_error_deg if estimated_flag
                     else cfg.phone.channel_phase_error_deg)
    return np.array([
        value * 10 ** (channel_amplitude[name] / 20)
        * np.exp(1j * np.deg2rad(channel_phase[name]))
        for name, value in zip(names, coupled)
    ])


def _check(name: str, error: float, limit: float, reason: str) -> dict:
    passed_flag = bool(error <= limit)
    return {"case": name, "level": "model", "status": "passed" if passed_flag else "failed",
            "passed_flag": passed_flag, "reason": reason, "max_abs_error": error,
            "limit": limit}


def model_checks() -> dict[str, dict]:
    """七项确定性检查：真实矩阵及天线响应与独立、显式的电压相加比较。"""
    definitions = (
        (None, None, ("TL", "BR"), np.array([1 + 0j, 0j])),
        (-20.0, 0.0, ("TL", "BR"), np.array([1 + 0j, 0j])),
        (-20.0, 0.0, ("TL", "BR"), np.array([0j, 1 + 0j])),
        (-20.0, 90.0, ("TL", "BR"), np.array([1 + 0j, 0j])),
        (-20.0, 180.0, ("TL", "BR"), np.array([1 + 0j, 0j])),
        (-10.0, 45.0, ("TL", "BR"), np.array([1 + 2j, -2 + 0.5j])),
        (-20.0, 0.0, ("BR", "TL"), np.array([0j, 1 + 0j])),
    )
    checks = {}
    for number, (magnitude_db, phase_deg, names, input_voltage) in enumerate(definitions, 1):
        cfg = default_config()
        cfg.phone.active_antennas = names
        cfg.interferers = []
        paths = (() if magnitude_db is None else (
            CouplingPath("TL", "BR", magnitude_db, phase_deg),
            CouplingPath("BR", "TL", magnitude_db, phase_deg),
        ))
        cfg.phone.antenna_model.true_state.coupling_paths = paths
        cfg.signal.duration_ms = 1
        expected_matrix = np.eye(2, dtype=complex)
        coefficient = (0j if magnitude_db is None else
                       10 ** (magnitude_db / 20) * np.exp(1j * np.deg2rad(phase_deg)))
        expected_matrix[0, 1] = coefficient
        expected_matrix[1, 0] = coefficient
        actual_matrix = coupling_matrix(cfg.phone.antenna_model.true_state, names)
        expected_voltage = np.array([
            input_voltage[0] + coefficient * input_voltage[1],
            coefficient * input_voltage[0] + input_voltage[1],
        ])
        model = build_antenna_model(cfg.phone, cfg.signal.carrier_hz)
        actual_response, _ = antenna_response(model, cfg.signal.true_direction,
                                              cfg.signal.true_polarization)
        expected_response = independent_manifold(cfg, estimated_flag=False)
        error = max(float(np.max(np.abs(actual_matrix - expected_matrix))),
                    float(np.max(np.abs(actual_response - expected_response))),
                    float(np.max(np.abs(actual_matrix @ input_voltage - expected_voltage))))
        if number == 6:
            # 两路不同的复数时间序列检查完整交叉混合，而非共同卫星标量。
            waveforms = np.array([[1 + 2j, -0.5 + 0.3j, 2 - 1j],
                                  [0.2 - 1j, -3 + 2j, 1.5 + 0.7j]])
            expected_mixed = np.vstack((waveforms[0] + coefficient * waveforms[1],
                                        coefficient * waveforms[0] + waveforms[1]))
            error = max(error, float(np.max(np.abs(actual_matrix @ waveforms - expected_mixed))))
            # 额外核对真实卫星波形经生产天线响应的传播。
            _, truth = generate_scenario(cfg)
            waveform = truth["desired_waveform"]
            # 独立几何阵元电压从关闭互耦的配置求得。
            raw = independent_manifold(cfg, estimated_flag=False, include_coupling_flag=False)
            expected_waveform = np.vstack((raw[0] * waveform + coefficient * raw[1] * waveform,
                                           coefficient * raw[0] * waveform + raw[1] * waveform))
            error = max(error, float(np.max(np.abs(truth["desired_component"] - expected_waveform))))
        if number == 7:
            # 物理名称换序后，恢复到 TL/BR 顺序应得到相同电压。
            standard_input = input_voltage[::-1]
            standard_output = np.array([standard_input[0] + coefficient * standard_input[1],
                                        coefficient * standard_input[0] + standard_input[1]])
            error = max(error, float(np.max(np.abs((actual_matrix @ input_voltage)[::-1]
                                                   - standard_output))))
        name = f"MC-{number:03d}"
        checks[name] = _check(name, error, 1e-10, "independent directed voltage mixing and antenna mapping")
    return checks


def coupling_row_fields(case, frontend, channels, components, window, weights) -> dict:
    """以已知源波形和实测热噪声复算固定权值输出 SNR，不由输出功率反推源功率。"""
    cfg = case.config
    model = build_antenna_model(cfg.phone, cfg.signal.carrier_hz)
    true_production, _ = antenna_response(model, cfg.signal.true_direction,
                                          cfg.signal.true_polarization)
    assumed_production, _ = antenna_response(model, cfg.signal.assumed_direction,
                                             cfg.signal.assumed_polarization,
                                             estimated_flag=True)
    true = independent_manifold(cfg, estimated_flag=False)
    assumed = independent_manifold(cfg, estimated_flag=True)
    rf = frontend["rf_info"]
    dfe = frontend["dfe_info"]
    waveform = frontend["truth"]["desired_waveform"][None, :]
    # 本批次 RF 各路参数相同、通道时延为零；源标量只经过一次相同的线性链路。
    filtered = filter_rf_input(waveform, rf["filter_coefficients"],
                               rf["fs_hz"], rf["if_hz"])
    scale = (10 ** (rf["channel_total_gain_db"][0] / 20)
             * np.exp(1j * np.deg2rad(rf["phase_error_deg"][0])) / rf["adc_lsb_v"])
    source = linear_dfe(filtered * scale, dfe)[0] / dfe["output_scale"]
    source_power = float(np.mean(np.abs(source[window]) ** 2))
    target = components["target"][:, window]
    thermal = components["thermal"][:, window]
    noise_covariance = thermal @ thermal.conj().T / thermal.shape[1]
    theoretical_target = abs(np.vdot(weights, true)) ** 2 * source_power
    theoretical_noise = float(np.vdot(weights, noise_covariance @ weights).real)
    measured_target = float(np.mean(np.abs(weights.conj() @ target) ** 2))
    measured_noise = float(np.mean(np.abs(weights.conj() @ thermal) ** 2))
    sample_covariance = channels @ channels.conj().T / channels.shape[1]
    loading = (cfg.diversity.diagonal_loading_factor * np.trace(sample_covariance).real
               / channels.shape[0])
    if case.series_label == "mvdr":
        eigenvalues, eigenvectors = np.linalg.eigh(sample_covariance + loading * np.eye(2))
        solved = eigenvectors @ ((eigenvectors.conj().T @ assumed) / eigenvalues)
        independent_weights = solved / np.vdot(assumed, solved)
    elif case.series_label == "egc":
        independent_weights = assumed / np.vdot(assumed, assumed)
    elif case.series_label == "egc_strict":
        independent_weights = (assumed / np.abs(assumed)) / np.sum(np.abs(assumed))
    else:
        independent_weights = np.array([1 / np.conj(assumed[0]), 0j])
    true_paths = cfg.phone.antenna_model.true_state.coupling_paths
    assumed_paths = cfg.phone.antenna_model.assumed_state.coupling_paths
    true_pre = independent_manifold(cfg, estimated_flag=False, include_coupling_flag=False)
    assumed_pre = independent_manifold(cfg, estimated_flag=True, include_coupling_flag=False)
    snr_theory = 10 * np.log10(theoretical_target / theoretical_noise)
    snr_measured = 10 * np.log10(measured_target / measured_noise)
    true_response = np.vdot(weights, true)
    return {
        "physical_dfe_sha256": hashlib.sha256(np.ascontiguousarray(frontend["dfe_channels"]).tobytes()).hexdigest(),
        "true_coupling_db": true_paths[0].magnitude_db if true_paths else None,
        "true_coupling_phase_deg": true_paths[0].phase_deg if true_paths else None,
        "assumed_coupling_db": assumed_paths[0].magnitude_db if assumed_paths else None,
        "assumed_coupling_phase_deg": assumed_paths[0].phase_deg if assumed_paths else None,
        "true_pre_gain_db": (20 * np.log10(np.abs(true_pre))).tolist(),
        "true_pre_phase_deg": np.rad2deg(np.angle(true_pre)).tolist(),
        "true_post_gain_db": (20 * np.log10(np.abs(true))).tolist(),
        "true_post_phase_deg": np.rad2deg(np.angle(true)).tolist(),
        "assumed_pre_gain_db": (20 * np.log10(np.abs(assumed_pre))).tolist(),
        "assumed_pre_phase_deg": np.rad2deg(np.angle(assumed_pre)).tolist(),
        "assumed_post_gain_db": (20 * np.log10(np.abs(assumed))).tolist(),
        "assumed_post_phase_deg": np.rad2deg(np.angle(assumed)).tolist(),
        "input_target_power_by_antenna": np.mean(np.abs(target) ** 2, axis=1).tolist(),
        "input_thermal_power_by_antenna": np.mean(np.abs(thermal) ** 2, axis=1).tolist(),
        "noise_covariance_real_imag": [[[float(z.real), float(z.imag)] for z in row]
                                       for row in noise_covariance],
        "source_scalar_power": source_power,
        "theoretical_target_power": theoretical_target,
        "theoretical_thermal_power": theoretical_noise,
        "theoretical_thermal_snr_db": float(snr_theory),
        "measured_thermal_snr_db": float(snr_measured),
        "thermal_snr_theory_error_db": float(snr_measured - snr_theory),
        "manifold_max_abs_error": float(max(np.max(np.abs(true - true_production)),
                                             np.max(np.abs(assumed - assumed_production)))),
        "weight_relative_error": float(np.linalg.norm(weights - independent_weights)
                                       / np.linalg.norm(independent_weights)),
        "assumed_unit_constraint_error": float(abs(np.vdot(weights, assumed) - 1)),
        "true_response_gain": float(abs(true_response)),
        "true_response_gain_db": float(20 * np.log10(abs(true_response))),
        "true_response_phase_deg": float(np.rad2deg(np.angle(true_response))),
        "acquisition_code_tolerance_chips": 0.5,
        "acquisition_doppler_half_bin_hz": float(np.diff(cfg.acquisition.doppler_bins_hz)[0] / 2),
    }


def summarize_coupling(rows: list[dict]) -> dict:
    """按语义编号配对无互耦与已知响应条件，统计捕获、跟踪及独立数值判据。"""
    grouped = {}
    indexed = {(row["case"], row["seed"]): row for row in rows}
    for row in rows:
        grouped.setdefault(row["case"], []).append(row)
    cases = {}
    checks = {}
    for name, observations in grouped.items():
        first = observations[0]
        case_id = name[:6]
        mode = first["mode"]
        paired_reference = f"MC-008-{mode}" if case_id != "MC-008" else None
        known_reference = (f"MC-012-{mode}" if case_id in
                           {"MC-015", "MC-016", "MC-017", "MC-018", "MC-019"} else None)
        for row in observations:
            errors = (row["manifold_max_abs_error"], row["weight_relative_error"],
                      row["assumed_unit_constraint_error"],
                      abs(row["thermal_snr_theory_error_db"]))
            passed_flag = bool(max(errors[:3]) <= 1e-8 and errors[3] <= 1e-7)
            checks[f'coupling_math:{name}:{row["seed"]}'] = {
                "status": "passed" if passed_flag else "failed", "passed_flag": passed_flag,
                "reason": f"manifold/weight/constraint/SNR errors {errors}; limits 1e-8/1e-7",
            }
            if known_reference is not None:
                reference = indexed[(known_reference, row["seed"])]
                same_flag = row["physical_dfe_sha256"] == reference["physical_dfe_sha256"]
                checks[f'coupling_physical_pair:{name}:{row["seed"]}'] = {
                    "status": "passed" if same_flag else "failed", "passed_flag": same_flag,
                    "reason": "same-seed physical DFE SHA256 equals MC-012",
                }
        finite = [row["tracking_cn0_late_median_db_hz"] for row in observations
                  if row["tracking_cn0_late_median_db_hz"] is not None]
        target_power_mean = np.mean([row["input_target_power_by_antenna"]
                                     for row in observations], axis=0)
        thermal_power_mean = np.mean([row["input_thermal_power_by_antenna"]
                                      for row in observations], axis=0)
        acquisition_pass = sum(row["acquired_flag"] for row in observations)
        correct = sum(row["acquired_flag"] and abs(row["acquired_code_error_chips"])
                      <= row["acquisition_code_tolerance_chips"]
                      and abs(row["acquired_doppler_error_hz"])
                      <= row["acquisition_doppler_half_bin_hz"] for row in observations)
        cases[name] = {
            "case": name, "level": "link", "mode": mode, "seeds": len(observations),
            "true_coupling_db": first["true_coupling_db"],
            "true_coupling_phase_deg": first["true_coupling_phase_deg"],
            "assumed_coupling_db": first["assumed_coupling_db"],
            "assumed_coupling_phase_deg": first["assumed_coupling_phase_deg"],
            "no_coupling_reference": paired_reference,
            "known_response_reference": known_reference,
            "antennas": first["antennas"],
            "true_response_gain_db_mean": float(np.mean(
                [row["true_response_gain_db"] for row in observations])),
            "true_response_phase_deg_mean": float(np.rad2deg(np.angle(np.mean(
                np.exp(1j * np.deg2rad([row["true_response_phase_deg"] for row in observations])))))),
            "thermal_snr_mean_db": float(np.mean([row["measured_thermal_snr_db"] for row in observations])),
            "snr_delta_vs_no_coupling_mean_db": (
                float(np.mean([row["measured_thermal_snr_db"]
                               - indexed[(paired_reference, row["seed"])]["measured_thermal_snr_db"]
                               for row in observations])) if paired_reference else None),
            "snr_delta_vs_known_response_mean_db": (
                float(np.mean([row["measured_thermal_snr_db"]
                               - indexed[(known_reference, row["seed"])]["measured_thermal_snr_db"]
                               for row in observations])) if known_reference else None),
            "acquisition_peak_ratio_mean": float(np.mean([row["acquisition_metric"] for row in observations])),
            "acquisition_threshold_pass_count": acquisition_pass,
            "acquisition_correct_count": correct,
            "acquisition_code_tolerance_chips": first["acquisition_code_tolerance_chips"],
            "acquisition_doppler_half_bin_hz": first["acquisition_doppler_half_bin_hz"],
            "tracking_success_count": sum(row["tracking_success_flag"] for row in observations),
            "tracking_cn0_mean_db_hz": float(np.mean(finite)) if finite else None,
            "tracking_cn0_valid_seed_count": len(finite),
            "tracking_cn0_missing_count": len(observations) - len(finite),
            "manifold_max_abs_error": max(row["manifold_max_abs_error"] for row in observations),
            "weight_max_relative_error": max(row["weight_relative_error"] for row in observations),
            "thermal_snr_theory_max_error_db": max(abs(row["thermal_snr_theory_error_db"])
                                                   for row in observations),
        }
        for index, antenna in enumerate(first["antennas"].split(",")):
            for field in ("true_pre_gain_db", "true_pre_phase_deg", "true_post_gain_db",
                          "true_post_phase_deg", "assumed_pre_gain_db",
                          "assumed_pre_phase_deg", "assumed_post_gain_db",
                          "assumed_post_phase_deg"):
                cases[name][f"{antenna}_{field}"] = first[field][index]
            cases[name][f"{antenna}_input_target_power_mean"] = float(target_power_mean[index])
            cases[name][f"{antenna}_input_thermal_power_mean"] = float(thermal_power_mean[index])
        passed_flag = all(checks[f'coupling_math:{name}:{row["seed"]}']["passed_flag"]
                          for row in observations)
        if known_reference is not None:
            passed_flag = passed_flag and all(
                checks[f'coupling_physical_pair:{name}:{row["seed"]}']["passed_flag"]
                for row in observations)
        cases[name]["status"] = "passed" if passed_flag else "failed"
        checks[f"coupling_case:{name}"] = {
            "status": cases[name]["status"], "passed_flag": passed_flag,
            "reason": "all independent manifold, fixed-weight SNR and physical-pair checks",
        }
    return {"passed_flag": all(value["status"] == "passed" for value in cases.values()),
            "cases": cases, "checks": checks}
