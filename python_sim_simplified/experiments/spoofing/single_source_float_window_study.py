"""Causal 20-epoch code/carrier float fit on five saved three-antenna tracks.

Each epoch has an independent local XYZ and clock in metres. Carrier adds one
free metre-valued bias per satellite and tracking arc. There is no motion
prior, oracle position, receiver publication, or change to core navigation.
"""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from core.gps_simulation import receiver_reference
from core.provenance import source_hashes
from experiments.paths import SPOOFING_RESULTS as RESULTS


BASE = RESULTS / "single_source_suppression_study"
OUTPUT = BASE / "float_window"
HISTORY_OUTPUT = BASE / "float_history"
MODE = "history"
SELECTED = RESULTS / "mechanism-four" / "summary.json"
CASE_INPUTS = {
    "baseline_weak": (BASE / "tracking.npz", RESULTS / "weak-ten" / "inputs" / "116901_weak_30_24_2"),
    "pll_weak": (BASE / "pll" / "tracking.npz", RESULTS / "weak-ten" / "inputs" / "116901_weak_30_24_2"),
    "pll25_weak": (BASE / "pll25" / "tracking.npz", RESULTS / "weak-ten" / "inputs" / "116901_weak_30_24_2"),
    "control": (RESULTS / "mechanism-four" / "evaluations" /
                "f4ac3dd3b9c138d16e8afd15ab37a3cd442bfc1790515604d36f5fb80e488318" / "tracking.npz",
                RESULTS / "weak-ten" / "inputs" / "114101_control_30_24_2"),
    "genuine_absent": (RESULTS / "mechanism-four" / "evaluations" /
                       "b05c24e65c576a695e47c256816a46841a62fda70b3ffc5037d8b0a8823680b6" / "tracking.npz",
                       RESULTS / "weak-ten" / "inputs" / "116901_absent_30_24_2"),
}
WINDOW_EPOCHS = 20
CODE_SIGMA_M = 30.
CARRIER_SIGMA_M = .03
SVD_RTOL = 1e-10
TARGET_NULL_TOL = 1e-7
FORMAL_HORIZONTAL_STD_STRATUM_M = 10.


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def solve_window(design, code_residual, carrier_residual, valid, arc, target,
                 width=WINDOW_EPOCHS):
    """Fit current XYZ after marginalizing independent epoch clocks and arcs.

    `design` is the metre-per-metre local XYZ geometry derivative. Both
    residual arrays are measured range minus range at the nominal reference,
    in metres. All state coordinates are metres, so SVD and null tolerances
    are dimensionless relative matrix thresholds. Only rows at/before target
    are read. Formal covariance uses the fixed research noise weights.
    """
    first = max(0, target - width + 1)
    epoch_count = target - first + 1
    satellites = valid.shape[1]
    arc_keys = sorted({(satellite, int(arc[epoch, satellite]))
                       for epoch in range(first, target + 1)
                       for satellite in range(satellites)
                       if valid[epoch, satellite] and np.isfinite(carrier_residual[epoch, satellite])})
    arc_column = {key: 4 * epoch_count + index for index, key in enumerate(arc_keys)}
    state_count = 4 * epoch_count + len(arc_keys)
    rows = []
    values = []
    carrier_origin = {}
    code_count = carrier_count = 0
    for epoch in range(first, target + 1):
        epoch_column = 4 * (epoch - first)
        for satellite in range(satellites):
            if not valid[epoch, satellite]:
                continue
            spatial = design[epoch, satellite]
            if np.isfinite(code_residual[epoch, satellite]):
                row = np.zeros(state_count)
                row[epoch_column:epoch_column + 3] = spatial / CODE_SIGMA_M
                row[epoch_column + 3] = 1 / CODE_SIGMA_M
                rows.append(row)
                values.append(code_residual[epoch, satellite] / CODE_SIGMA_M)
                code_count += 1
            if np.isfinite(carrier_residual[epoch, satellite]):
                key = (satellite, int(arc[epoch, satellite]))
                origin = carrier_origin.setdefault(key, carrier_residual[epoch, satellite])
                row = np.zeros(state_count)
                row[epoch_column:epoch_column + 3] = spatial / CARRIER_SIGMA_M
                row[epoch_column + 3] = 1 / CARRIER_SIGMA_M
                row[arc_column[key]] = 1 / CARRIER_SIGMA_M
                rows.append(row)
                values.append((carrier_residual[epoch, satellite] - origin) / CARRIER_SIGMA_M)
                carrier_count += 1
    target_columns = 4 * (epoch_count - 1) + np.arange(3)
    if not rows:
        return dict(observable=False, position_m=np.full(3, np.nan),
                    formal_sigma_xyz_m=np.full(3, np.nan),
                    formal_horizontal_std_m=np.nan, condition=np.inf,
                    target_null_norm=1., rank=0, state_count=state_count,
                    code_count=0, carrier_count=0, arc_bias_count=0)
    matrix = np.stack(rows)
    observation = np.array(values)
    left, singular, right = np.linalg.svd(matrix, full_matrices=True)
    rank = int(np.count_nonzero(singular > singular[0] * SVD_RTOL))
    target_null_norm = float(np.max(np.linalg.norm(right[rank:, target_columns], axis=0)))
    observable = target_null_norm < TARGET_NULL_TOL
    condition = float(singular[0] / singular[rank - 1])
    if observable:
        estimate = right[:rank].T @ ((left[:, :rank].T @ observation) / singular[:rank])
        target_covariance_factor = right[:rank, target_columns].T / singular[:rank]
        sigma = np.sqrt(np.sum(target_covariance_factor**2, axis=1))
        position = estimate[target_columns]
        horizontal_std = float(np.linalg.norm(sigma[:2]))
    else:
        position = np.full(3, np.nan)
        sigma = np.full(3, np.nan)
        horizontal_std = np.nan
    return dict(observable=observable, position_m=position,
                formal_sigma_xyz_m=sigma, formal_horizontal_std_m=horizontal_std,
                condition=condition, target_null_norm=target_null_norm,
                rank=rank, state_count=state_count,
                code_count=code_count, carrier_count=carrier_count,
                arc_bias_count=len(arc_keys))


def solve_history(design, code_residual, carrier_residual, valid, arc):
    """Causal square-root compression of all past carrier-arc information.

    The current epoch has a free XYZ and clock. Past epoch states are removed
    by projecting their rows off the current A column space, then SVD-compressing
    only the remaining arc-bias equations. No position/clock prior is retained.
    """
    epoch_count, satellites = valid.shape
    positions = np.full((epoch_count, 3), np.nan)
    sigma_xyz = np.full((epoch_count, 3), np.nan)
    horizontal_std = np.full(epoch_count, np.nan)
    condition = np.full(epoch_count, np.nan)
    null_norm = np.full(epoch_count, np.nan)
    ranks = np.zeros(epoch_count, int)
    state_counts = np.zeros(epoch_count, int)
    code_counts = np.zeros(epoch_count, int)
    carrier_counts = np.zeros(epoch_count, int)
    arc_counts = np.zeros(epoch_count, int)
    arc_column = {}
    carrier_origin = {}
    prior_matrix = np.empty((0, 0))
    prior_observation = np.empty(0)
    for epoch in range(epoch_count):
        for satellite in range(satellites):
            if valid[epoch, satellite] and np.isfinite(carrier_residual[epoch, satellite]):
                key = (satellite, int(arc[epoch, satellite]))
                if key not in arc_column:
                    arc_column[key] = len(arc_column)
                    carrier_origin[key] = carrier_residual[epoch, satellite]
                    prior_matrix = np.pad(prior_matrix, ((0, 0), (0, 1)))
        bias_count = len(arc_column)
        current_rows = []
        bias_rows = []
        values = []
        for satellite in range(satellites):
            if not valid[epoch, satellite]:
                continue
            spatial = design[epoch, satellite]
            if np.isfinite(code_residual[epoch, satellite]):
                current_rows.append(np.r_[spatial, 1.] / CODE_SIGMA_M)
                bias_rows.append(np.zeros(bias_count))
                values.append(code_residual[epoch, satellite] / CODE_SIGMA_M)
                code_counts[epoch] += 1
            if np.isfinite(carrier_residual[epoch, satellite]):
                key = (satellite, int(arc[epoch, satellite]))
                bias_row = np.zeros(bias_count)
                bias_row[arc_column[key]] = 1 / CARRIER_SIGMA_M
                current_rows.append(np.r_[spatial, 1.] / CARRIER_SIGMA_M)
                bias_rows.append(bias_row)
                values.append((carrier_residual[epoch, satellite] - carrier_origin[key])
                              / CARRIER_SIGMA_M)
                carrier_counts[epoch] += 1
        current = np.array(current_rows).reshape(-1, 4)
        bias = np.array(bias_rows).reshape(len(bias_rows), bias_count)
        observation = np.array(values)
        joint = np.vstack((np.hstack((current, bias)),
                           np.hstack((np.zeros((len(prior_matrix), 4)), prior_matrix))))
        joint_observation = np.r_[observation, prior_observation]
        state_counts[epoch] = joint.shape[1]
        arc_counts[epoch] = bias_count
        if len(joint):
            left, singular, right = np.linalg.svd(joint, full_matrices=True)
            rank = int(np.count_nonzero(singular > singular[0] * SVD_RTOL))
            ranks[epoch] = rank
            null_norm[epoch] = float(np.max(np.linalg.norm(right[rank:, :3], axis=0)))
            condition[epoch] = float(singular[0] / singular[rank - 1])
            if null_norm[epoch] < TARGET_NULL_TOL:
                estimate = right[:rank].T @ ((left[:, :rank].T @ joint_observation)
                                               / singular[:rank])
                factor = right[:rank, :3].T / singular[:rank]
                sigma_xyz[epoch] = np.sqrt(np.sum(factor**2, axis=1))
                positions[epoch] = estimate[:3]
                horizontal_std[epoch] = np.linalg.norm(sigma_xyz[epoch, :2])
        else:
            null_norm[epoch] = 1.
            condition[epoch] = np.inf

        if len(current):
            basis, singular_current, _ = np.linalg.svd(current, full_matrices=False)
            current_rank = np.count_nonzero(singular_current > singular_current[0] * SVD_RTOL)
            basis = basis[:, :current_rank]
            projected_bias = bias - basis @ (basis.T @ bias)
            projected_observation = observation - basis @ (basis.T @ observation)
        else:
            projected_bias = bias
            projected_observation = observation
        if bias_count:
            stacked = np.vstack((prior_matrix, projected_bias))
            stacked_observation = np.r_[prior_observation, projected_observation]
            if len(stacked):
                left_bias, singular_bias, right_bias = np.linalg.svd(
                    stacked, full_matrices=False)
                bias_rank = int(np.count_nonzero(singular_bias
                                                 > singular_bias[0] * SVD_RTOL))
                prior_matrix = singular_bias[:bias_rank, None] * right_bias[:bias_rank]
                prior_observation = left_bias[:, :bias_rank].T @ stacked_observation
    return dict(position_m=positions, formal_sigma_xyz_m=sigma_xyz,
                formal_horizontal_std_m=horizontal_std, condition=condition,
                target_null_norm=null_norm, rank=ranks, state_count=state_counts,
                code_count=code_counts, carrier_count=carrier_counts,
                arc_bias_count=arc_counts, observable=np.isfinite(positions).all(axis=1))


def geometry_and_admission(observations):
    reference, rotation = receiver_reference()
    displacement = observations["satellites"] - reference
    reference_range = np.linalg.norm(displacement, axis=2)
    design = -(displacement / reference_range[:, :, None]) @ rotation.T
    code_residual = observations["ranges"] - reference_range
    carrier_residual = observations["carrier"] - reference_range
    valid = (np.isfinite(observations["ranges"])
             & np.isfinite(observations["carrier"])
             & (observations["lock"] >= .5)
             & (observations["cn0"] >= 25.)
             & (observations["gain"] >= .05)
             & (observations["support"] >= 3.)
             & (observations["arc"] >= 0))
    return design, code_residual, carrier_residual, valid


def replay_case(name, tracking_path, input_path):
    metadata = json.loads((input_path / "metadata.json").read_text(encoding="utf-8"))
    with np.load(tracking_path) as saved:
        observations = dict(saved)
    design, code, carrier, valid = geometry_and_admission(observations)
    times = observations["time"]
    available_time = observations["available_time"]
    positions = np.full((len(times), 3), np.nan)
    formal_sigma = np.full((len(times), 3), np.nan)
    horizontal_std = np.full(len(times), np.nan)
    condition = np.full(len(times), np.nan)
    null_norm = np.full(len(times), np.nan)
    rank = np.zeros(len(times), int)
    state_count = np.zeros(len(times), int)
    code_count = np.zeros(len(times), int)
    carrier_count = np.zeros(len(times), int)
    arc_count = np.zeros(len(times), int)
    for target in range(len(times)):
        result = solve_window(design, code, carrier, valid, observations["arc"], target)
        positions[target] = result["position_m"]
        formal_sigma[target] = result["formal_sigma_xyz_m"]
        horizontal_std[target] = result["formal_horizontal_std_m"]
        condition[target] = result["condition"]
        null_norm[target] = result["target_null_norm"]
        rank[target] = result["rank"]
        state_count[target] = result["state_count"]
        code_count[target] = result["code_count"]
        carrier_count[target] = result["carrier_count"]
        arc_count[target] = result["arc_bias_count"]
    truth_east = metadata["case"]["velocity_east"] * times
    error = np.linalg.norm(positions[:, :2] - np.column_stack((truth_east, np.zeros(len(times)))), axis=1)
    post = times >= metadata["case"]["attack_start"]
    observable = np.isfinite(error)
    stratum = observable & (horizontal_std <= FORMAL_HORIZONTAL_STD_STRATUM_M)
    output = OUTPUT / f"{name}.npz"
    np.savez_compressed(output, observation_time=times, available_time=available_time,
                        position=positions, formal_sigma_xyz_m=formal_sigma,
                        formal_horizontal_std_m=horizontal_std, condition=condition,
                        target_null_norm=null_norm, rank=rank, state_count=state_count,
                        code_count=code_count, carrier_count=carrier_count,
                        arc_bias_count=arc_count, admitted_count=valid.sum(axis=1),
                        horizontal_error_m=error, post=post, diagnostic_std_le10=stratum)
    def metrics(mask):
        values = error[mask]
        return dict(count=int(mask.sum()), good20=int((values <= 20).sum()),
                    error30=int((values >= 30).sum()), error50=int((values >= 50).sum()),
                    median_error_m=None if not len(values) else float(np.median(values)),
                    p95_error_m=None if not len(values) else float(np.percentile(values, 95)),
                    max_error_m=None if not len(values) else float(values.max()),
                    median_formal_horizontal_std_m=None if not len(values) else float(np.median(horizontal_std[mask])))
    return dict(name=name, tracking_sha256=digest(tracking_path),
                metadata_sha256=digest(input_path / "metadata.json"),
                result_sha256=digest(output), opportunities=int(post.sum()),
                all_observable=metrics(post & observable),
                formal_std_le10=metrics(post & stratum),
                maximum_condition_observable=None if not np.any(post & observable)
                else float(np.max(condition[post & observable])))


def plot_results(rows, output_dir=OUTPUT):
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    for row in rows:
        with np.load(output_dir / f"{row['name']}.npz") as saved:
            time = saved["observation_time"]
            axes[0].plot(time, saved["horizontal_error_m"], label=row["name"])
            axes[1].plot(time, saved["formal_horizontal_std_m"], label=row["name"])
    axes[0].set(ylabel="Horizontal error of observable fits (m)")
    axes[1].set(xlabel="Observation time (s)", ylabel="Formal horizontal std (m)")
    axes[1].axhline(FORMAL_HORIZONTAL_STD_STRATUM_M, color="black", linestyle=":")
    for axis in axes:
        axis.set_yscale("symlog", linthresh=2.)
        axis.grid(alpha=.3)
        axis.legend(fontsize=8)
    figure.savefig(output_dir / "float_history.png" if output_dir == HISTORY_OUTPUT
                   else output_dir / "float_window.png", dpi=160)
    plt.close(figure)


def window_main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sources = source_hashes()
    selected = json.loads(SELECTED.read_text(encoding="utf-8"))
    selected_rows = {row["after_id"]: row for row in selected["selected_rows"]}
    for label in ("control", "genuine_absent"):
        tracking_path, input_path = CASE_INPUTS[label]
        row = selected_rows[tracking_path.parent.name]
        metadata = json.loads((input_path / "metadata.json").read_text(encoding="utf-8"))
        assert row["input"].replace("\\", "/") == input_path.relative_to(RESULTS).as_posix()
        assert metadata["samples_sha256"] == row["after"]["input_sha256"]
    stage_one = json.loads((BASE / "summary.json").read_text(encoding="utf-8"))
    assert digest(CASE_INPUTS["baseline_weak"][0]) == stage_one["tracking_sha256"]
    for label in ("pll_weak", "pll25_weak"):
        parent = CASE_INPUTS[label][0].parent
        stage = json.loads((parent / "summary.json").read_text(encoding="utf-8"))
        assert digest(CASE_INPUTS[label][0]) == stage["tracking_sha256"]
    rows = []
    for label, (tracking_path, input_path) in CASE_INPUTS.items():
        print(label, "float window start", flush=True)
        rows.append(replay_case(label, tracking_path, input_path))
        print(label, rows[-1]["all_observable"], flush=True)
    plot_results(rows)
    summary = dict(scope="Offline causal 20-epoch float fitting; never receiver publication",
                   source_sha256=sources, selected_summary_sha256=digest(SELECTED),
                   window_epochs=WINDOW_EPOCHS, code_sigma_m=CODE_SIGMA_M,
                   carrier_sigma_m=CARRIER_SIGMA_M, svd_rtol=SVD_RTOL,
                   target_null_norm_tolerance=TARGET_NULL_TOL,
                   formal_horizontal_std_stratum_m=FORMAL_HORIZONTAL_STD_STRATUM_M,
                   model="independent XYZ and clock per epoch; carrier bias per satellite arc; all metre-valued",
                   observation_time_post_definition="observation_time >= case.attack_start",
                   caveat="formal covariance uses assumed weights, not an integrity bound",
                   cases=rows)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    assert source_hashes() == sources
    print("Saved", OUTPUT, flush=True)


def replay_history_case(name, tracking_path, input_path):
    metadata = json.loads((input_path / "metadata.json").read_text(encoding="utf-8"))
    with np.load(tracking_path) as saved:
        observations = dict(saved)
    design, code, carrier, valid = geometry_and_admission(observations)
    result = solve_history(design, code, carrier, valid, observations["arc"])
    time = observations["time"]
    truth_east = metadata["case"]["velocity_east"] * time
    error = np.linalg.norm(result["position_m"][:, :2]
                           - np.column_stack((truth_east, np.zeros(len(time)))), axis=1)
    post = time >= metadata["case"]["attack_start"]
    observable = result["observable"]
    stratum = observable & (result["formal_horizontal_std_m"]
                            <= FORMAL_HORIZONTAL_STD_STRATUM_M)
    output = HISTORY_OUTPUT / f"{name}.npz"
    np.savez_compressed(output, observation_time=time,
                        available_time=observations["available_time"],
                        **result, admitted_count=valid.sum(axis=1),
                        horizontal_error_m=error, post=post,
                        diagnostic_std_le10=stratum)

    def metrics(mask):
        values = error[mask]
        return dict(count=int(mask.sum()), good20=int((values <= 20).sum()),
                    error30=int((values >= 30).sum()), error50=int((values >= 50).sum()),
                    median_error_m=None if not len(values) else float(np.median(values)),
                    p95_error_m=None if not len(values) else float(np.percentile(values, 95)),
                    max_error_m=None if not len(values) else float(values.max()),
                    median_formal_horizontal_std_m=None if not len(values)
                    else float(np.median(result["formal_horizontal_std_m"][mask])))

    return dict(name=name, tracking_sha256=digest(tracking_path),
                metadata_sha256=digest(input_path / "metadata.json"),
                result_sha256=digest(output), opportunities=int(post.sum()),
                all_observable=metrics(post & observable),
                formal_std_le10=metrics(post & stratum),
                maximum_condition_observable=None if not np.any(post & observable)
                else float(np.max(result["condition"][post & observable])))


def history_main():
    HISTORY_OUTPUT.mkdir(parents=True, exist_ok=True)
    sources = source_hashes()
    selected = json.loads(SELECTED.read_text(encoding="utf-8"))
    selected_rows = {row["after_id"]: row for row in selected["selected_rows"]}
    for label in ("control", "genuine_absent"):
        tracking_path, input_path = CASE_INPUTS[label]
        row = selected_rows[tracking_path.parent.name]
        metadata = json.loads((input_path / "metadata.json").read_text(encoding="utf-8"))
        assert row["input"].replace("\\", "/") == input_path.relative_to(RESULTS).as_posix()
        assert metadata["samples_sha256"] == row["after"]["input_sha256"]
    stage_one = json.loads((BASE / "summary.json").read_text(encoding="utf-8"))
    assert digest(CASE_INPUTS["baseline_weak"][0]) == stage_one["tracking_sha256"]
    for label in ("pll_weak", "pll25_weak"):
        tracking_path = CASE_INPUTS[label][0]
        stage = json.loads((tracking_path.parent / "summary.json").read_text(encoding="utf-8"))
        assert digest(tracking_path) == stage["tracking_sha256"]
    rows = []
    for label, (tracking_path, input_path) in CASE_INPUTS.items():
        print(label, "square-root history start", flush=True)
        rows.append(replay_history_case(label, tracking_path, input_path))
        print(label, rows[-1]["all_observable"], flush=True)
    plot_results(rows, HISTORY_OUTPUT)
    summary = dict(scope="Offline causal all-history square-root float fit; never receiver publication",
                   source_sha256=sources, selected_summary_sha256=digest(SELECTED),
                   code_sigma_m=CODE_SIGMA_M, carrier_sigma_m=CARRIER_SIGMA_M,
                   svd_rtol=SVD_RTOL, target_null_norm_tolerance=TARGET_NULL_TOL,
                   formal_horizontal_std_stratum_m=FORMAL_HORIZONTAL_STD_STRATUM_M,
                   model="free current XYZ/clock; SVD-eliminated past XYZ/clocks; persistent per-satellite-arc biases",
                   observation_time_post_definition="observation_time >= case.attack_start",
                   caveat="formal covariance uses assumed weights, not an integrity bound",
                   cases=rows)
    (HISTORY_OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    assert source_hashes() == sources
    print("Saved", HISTORY_OUTPUT, flush=True)


if __name__ == "__main__":
    history_main() if MODE == "history" else window_main()
