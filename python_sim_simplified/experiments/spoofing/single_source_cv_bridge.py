"""Conditional offline constant-velocity bridge of saved float measurements.

This estimates motion only after enough recent accepted float positions. Its
predictions never become measurements or reset the age timer. It does not feed
the receiver's navigation output or claim GNSS continuity during prediction.
"""

import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np

from core.provenance import source_hashes
from experiments.paths import SPOOFING_RESULTS as RESULTS
from experiments.spoofing.single_source_float_window_study import BASE, CASE_INPUTS
from experiments.spoofing.single_source_float_validation import source_cases


OUTPUT = BASE / "cv_bridge"
WINDOW_S = .2
MIN_MEASUREMENTS = 3
MIN_SPAN_S = .05
MAX_PREDICTION_AGE_S = .2
FORMAL_HORIZONTAL_STD_M = 10.


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cv_bridge(time, measured_position, formal_horizontal_std):
    """Emit measured, predicted, or unavailable positions causally."""
    count = len(time)
    output = np.full((count, 3), np.nan)
    velocity_out = np.full((count, 3), np.nan)
    status = np.full(count, "unavailable", dtype="U11")
    last_measured = np.full(count, np.nan)
    last_predicted = np.full(count, np.nan)
    age = np.full(count, np.nan)
    fit_time = np.full(count, np.nan)
    fit_span = np.full(count, np.nan)
    fit_points = np.zeros(count, int)
    accepted = []
    velocity = None
    velocity_fit_time = np.nan
    velocity_fit_span = np.nan
    velocity_fit_points = 0
    last_measurement_index = None
    last_prediction_time = np.nan
    for index, current_time in enumerate(time):
        measured = (np.isfinite(measured_position[index]).all()
                    and formal_horizontal_std[index] <= FORMAL_HORIZONTAL_STD_M)
        if measured:
            accepted.append(index)
            recent = [item for item in accepted if time[item] >= current_time - WINDOW_S - 1e-12]
            if len(recent) >= MIN_MEASUREMENTS and time[recent[-1]] - time[recent[0]] >= MIN_SPAN_S - 1e-12:
                sample_time = time[recent] - np.mean(time[recent])
                sample_position = measured_position[recent]
                velocity = np.sum(sample_time[:, None] * sample_position, axis=0) / np.sum(sample_time**2)
                velocity_fit_time = current_time
                velocity_fit_span = time[recent[-1]] - time[recent[0]]
                velocity_fit_points = len(recent)
            else:
                velocity = None
                velocity_fit_time = np.nan
                velocity_fit_span = np.nan
                velocity_fit_points = 0
            last_measurement_index = index
            output[index] = measured_position[index]
            status[index] = "measured"
            age[index] = 0.
        elif last_measurement_index is not None:
            age[index] = current_time - time[last_measurement_index]
            # 1 ps only absorbs binary floating-point rounding at the 0.2 s boundary.
            if velocity is not None and age[index] <= MAX_PREDICTION_AGE_S + 1e-12:
                output[index] = (measured_position[last_measurement_index]
                                 + velocity * age[index])
                status[index] = "predicted"
                last_prediction_time = current_time
        if last_measurement_index is not None:
            last_measured[index] = time[last_measurement_index]
        last_predicted[index] = last_prediction_time
        if velocity is not None:
            velocity_out[index] = velocity
            fit_time[index] = velocity_fit_time
            fit_span[index] = velocity_fit_span
            fit_points[index] = velocity_fit_points
    return dict(position_m=output, velocity_m_s=velocity_out, status=status,
                last_measured_time_s=last_measured,
                last_predicted_time_s=last_predicted, age_s=age,
                velocity_fit_time_s=fit_time, velocity_fit_span_s=fit_span,
                velocity_fit_points=fit_points)


def input_cases():
    cases = []
    history = json.loads((BASE / "float_history" / "summary.json").read_text(encoding="utf-8"))
    for row in history["cases"]:
        path = BASE / "float_history" / f"{row['name']}.npz"
        assert digest(path) == row["result_sha256"]
        metadata = CASE_INPUTS[row["name"]][1] / "metadata.json"
        cases.append((row["name"], path, metadata))
    validation = json.loads((BASE / "float_validation" / "summary.json").read_text(encoding="utf-8"))
    source = {label: metadata for label, _, metadata in source_cases()}
    for row in validation["cases"]:
        path = BASE / "float_validation" / f"{row['label']}.npz"
        assert digest(path) == row["result_sha256"]
        cases.append((row["label"], path, source[row["label"]]))
    assert len(cases) == 16
    return cases


def score(time, result, metadata):
    post = time >= metadata["case"]["attack_start"]
    truth = np.column_stack((metadata["case"]["velocity_east"] * time,
                             np.zeros((len(time), 2))))
    error = np.linalg.norm(result["position_m"][:, :2] - truth[:, :2], axis=1)
    scores = {}
    for kind in ("measured", "predicted", "unavailable"):
        mask = post & (result["status"] == kind)
        values = error[mask]
        scores[kind] = dict(count=int(mask.sum()),
                            good20=int((values <= 20).sum()),
                            error30=int((values >= 30).sum()),
                            error50=int((values >= 50).sum()),
                            max_error_m=None if kind == "unavailable" or not len(values)
                            else float(values.max()),
                            p95_error_m=None if kind == "unavailable" or not len(values)
                            else float(np.percentile(values, 95)))
    tail = {kind: int((result["status"][-50:] == kind).sum())
            for kind in ("measured", "predicted", "unavailable")}
    return dict(opportunities=int(post.sum()), statuses=scores, tail_50=tail), error


def process_case(name, path, metadata_path):
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(path) as saved:
        time = saved["observation_time"]
        position = saved["position_m"]
        formal = saved["formal_horizontal_std_m"]
    result = cv_bridge(time, position, formal)
    metrics, error = score(time, result, metadata)
    target = OUTPUT / f"{name}.npz"
    np.savez_compressed(target, observation_time=time, **result,
                        horizontal_error_m=error,
                        formal_horizontal_std_m=formal)
    return dict(name=name, source_result_sha256=digest(path),
                metadata_sha256=digest(metadata_path), result_sha256=digest(target),
                **metrics)


def synthetic_counterexamples():
    time = np.arange(0., .401, .01)
    observed = np.isclose(time, 0.) | np.isclose(time, .05) | np.isclose(time, .1)
    examples = {}
    for name in ("acceleration_20mps2", "turn_and_brake", "false_support"):
        truth = np.zeros((len(time), 3))
        if name == "acceleration_20mps2":
            truth[:, 0] = 10 * time**2
        elif name == "turn_and_brake":
            truth[:, 0] = 10 * np.minimum(time, .1)
            truth[:, 1] = 10 * np.maximum(time - .1, 0.)
        measurement = np.full((len(time), 3), np.nan)
        formal = np.full(len(time), np.nan)
        if name == "false_support":
            observed_here = np.isclose(np.mod(np.arange(len(time)), 5), 0)
            measurement[observed_here, 0] = 100 * time[observed_here]
            measurement[observed_here, 1:] = 0.
        else:
            observed_here = observed
            measurement[observed_here] = truth[observed_here]
        formal[observed_here] = 1.
        result = cv_bridge(time, measurement, formal)
        error = np.linalg.norm(result["position_m"][:, :2] - truth[:, :2], axis=1)
        examples[name] = dict(truth=truth, measurement=measurement,
                              formal=formal, result=result, error=error)
    path = OUTPUT / "synthetic_counterexamples.npz"
    arrays = {"time": time}
    for name, example in examples.items():
        arrays.update({f"{name}_{key}": value for key, value in
                       (("truth", example["truth"]),
                        ("measurement", example["measurement"]),
                        ("formal", example["formal"]),
                        ("position", example["result"]["position_m"]),
                        ("status", example["result"]["status"]),
                        ("error", example["error"]))})
    np.savez_compressed(path, **arrays)
    report = {name: dict(max_available_error_m=float(np.nanmax(example["error"])),
                         measured=int((example["result"]["status"] == "measured").sum()),
                         predicted=int((example["result"]["status"] == "predicted").sum()),
                         unavailable=int((example["result"]["status"] == "unavailable").sum()))
              for name, example in examples.items()}
    return dict(result_sha256=digest(path), scenarios=report,
                note="Synthetic measurement-domain counterexamples, not generated RF")


def plot_cases(rows):
    figure, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    for kind, color in (("measured", "#3878a8"), ("predicted", "#dd9133"),
                        ("unavailable", "#9a9a9a")):
        height = np.array([row["statuses"][kind]["count"] / row["opportunities"] for row in rows])
        axis.bar(x, height, bottom=bottom, color=color, label=kind)
        bottom += height
    axis.set(ylabel="Fraction of observation-time opportunities", ylim=(0, 1))
    axis.set_xticks(x, [row["name"] for row in rows], rotation=55, ha="right", fontsize=8)
    axis.grid(alpha=.25)
    axis.legend()
    figure.savefig(OUTPUT / "cv_bridge.png", dpi=160)
    plt.close(figure)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sources = source_hashes()
    rows = []
    for name, path, metadata in input_cases():
        print(name, "CV bridge start", flush=True)
        rows.append(process_case(name, path, metadata))
        print(name, {kind: row["count"] for kind, row in rows[-1]["statuses"].items()}, flush=True)
    synthetic = synthetic_counterexamples()
    plot_cases(rows)
    summary = dict(scope="Conditional offline CV bridge of 16 saved float fits; no receiver publication",
                   source_sha256=sources, window_s=WINDOW_S,
                   min_measurements=MIN_MEASUREMENTS, min_span_s=MIN_SPAN_S,
                   max_prediction_age_s=MAX_PREDICTION_AGE_S,
                   measurement_formal_horizontal_std_limit_m=FORMAL_HORIZONTAL_STD_M,
                   note="Predictions do not update velocity history or reset measurement age",
                   cases=rows, synthetic_counterexamples=synthetic)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    assert source_hashes() == sources
    print("Saved", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
