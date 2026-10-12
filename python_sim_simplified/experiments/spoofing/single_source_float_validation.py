"""Saved-track validation and explicit faults for the offline history float fit."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from core.provenance import source_hashes
from core.spoofing.recovery_geometry import WAVELENGTH
from experiments.paths import SPOOFING_RESULTS as RESULTS
from experiments.spoofing.single_source_float_window_study import (
    BASE, geometry_and_admission, solve_history)


OUTPUT = BASE / "float_validation"
HISTORY = RESULTS / "single_source_history_study"
SELECTED = RESULTS / "mechanism-four" / "summary.json"
FAULT_TRACKING = BASE / "pll" / "tracking.npz"
FAULT_SATELLITE = 10
FAULT_ONSET_S = .5


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def coherent_displacement(design, code, carrier, time, onset=.5, final_east_m=100.):
    """Add exactly H_xyz*d(t) to both observables; not a generated RF attack."""
    east = final_east_m * np.maximum(time - onset, 0.) / (time[-1] - onset)
    displacement = np.column_stack((east, np.zeros((len(time), 2))))
    added_range = np.einsum("tsd,td->ts", design, displacement)
    return code + added_range, carrier + added_range, displacement


def carrier_fault(carrier, time, satellite, onset=.5, kind="slip"):
    """Undetected one-satellite same-arc carrier fault, with code unchanged."""
    changed = carrier.copy()
    if kind == "slip":
        changed[time >= onset, satellite] += WAVELENGTH
    else:
        changed[:, satellite] += .5 * np.maximum(time - onset, 0.)
    return changed


def source_cases():
    cases = []
    history = json.loads((HISTORY / "summary.json").read_text(encoding="utf-8"))
    for row in history["new_rf_results"]:
        root = HISTORY / row["label"]
        antennas = row["antennas"]
        tracking = root / f"tracking_{antennas}.npz"
        metadata = root / "metadata.json"
        assert digest(tracking) == row["tracking_sha256"]
        assert digest(metadata) == row["metadata_sha256"]
        case = json.loads(metadata.read_text(encoding="utf-8"))
        assert case["rf_sha256"] == row["rf_sha256"]
        cases.append((f"{row['label']}_{antennas}A", tracking, metadata))
    selected = json.loads(SELECTED.read_text(encoding="utf-8"))
    for row in selected["selected_rows"]:
        if ("116921_weak" not in row["input"] and "117901_weak" not in row["input"]):
            continue
        if row["subset"] not in ("three", "four"):
            continue
        tracking = RESULTS / row["after_root"] / "evaluations" / row["after_id"] / "tracking.npz"
        metadata = RESULTS / row["input"] / "metadata.json"
        case = json.loads(metadata.read_text(encoding="utf-8"))
        assert case["samples_sha256"] == row["after"]["input_sha256"]
        cases.append((f"{Path(row['input']).name}_{row['subset']}", tracking, metadata))
    assert len(cases) == 11
    return cases


def metrics(observations, metadata, estimate):
    time = observations["time"]
    post = time >= metadata["case"]["attack_start"]
    truth_east = metadata["case"]["velocity_east"] * time
    error = np.linalg.norm(estimate["position_m"][:, :2]
                           - np.column_stack((truth_east, np.zeros(len(time)))), axis=1)
    observable = estimate["observable"]
    qualified = post & observable
    restricted = qualified & (estimate["formal_horizontal_std_m"] <= 10.)
    missed = np.r_[False, ~observable[post], False].astype(int)
    starts = np.flatnonzero(np.diff(missed) == 1)
    ends = np.flatnonzero(np.diff(missed) == -1)

    def score(mask):
        values = error[mask]
        return dict(count=int(mask.sum()), good20=int((values <= 20).sum()),
                    error30=int((values >= 30).sum()), error50=int((values >= 50).sum()),
                    median_error_m=None if not len(values) else float(np.median(values)),
                    p95_error_m=None if not len(values) else float(np.percentile(values, 95)),
                    max_error_m=None if not len(values) else float(values.max()))

    return dict(opportunities=int(post.sum()), all_observable=score(qualified),
                formal_std_le10=score(restricted),
                longest_unobservable_epochs=int(max(ends - starts, default=0)),
                tail_50_observable=int(observable[-50:].sum())), error


def validate_case(label, tracking, metadata_path):
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(tracking) as saved:
        observations = dict(saved)
    design, code, carrier, valid = geometry_and_admission(observations)
    result = solve_history(design, code, carrier, valid, observations["arc"])
    score, error = metrics(observations, metadata, result)
    result_path = OUTPUT / f"{label}.npz"
    np.savez_compressed(result_path, observation_time=observations["time"],
                        available_time=observations["available_time"],
                        **result, horizontal_error_m=error,
                        admitted_count=valid.sum(axis=1))
    return dict(label=label, tracking_sha256=digest(tracking),
                metadata_sha256=digest(metadata_path), result_sha256=digest(result_path),
                **score)


def fault_study():
    stage = json.loads((BASE / "pll" / "summary.json").read_text(encoding="utf-8"))
    assert digest(FAULT_TRACKING) == stage["tracking_sha256"]
    with np.load(FAULT_TRACKING) as saved:
        observations = dict(saved)
    design, code, carrier, valid = geometry_and_admission(observations)
    arc = observations["arc"]
    time = observations["time"]
    assert np.array_equal(np.unique(arc[valid[:, FAULT_SATELLITE], FAULT_SATELLITE]), [0])
    qualified_before = int(valid[time < FAULT_ONSET_S, FAULT_SATELLITE].sum())
    qualified_after = int(valid[time >= FAULT_ONSET_S, FAULT_SATELLITE].sum())
    assert (qualified_before, qualified_after) == (23, 105)
    baseline = solve_history(design, code, carrier, valid, arc)
    with np.load(BASE / "float_history" / "pll_weak.npz") as saved:
        np.testing.assert_allclose(baseline["position_m"], saved["position_m"],
                                   atol=1e-7, equal_nan=True)
    changed_code, changed_carrier, displacement = coherent_displacement(
        design, code, carrier, time)
    assert np.array_equal(valid, geometry_and_admission(observations)[3])
    coherent = solve_history(design, changed_code, changed_carrier, valid, arc)
    before = time < FAULT_ONSET_S
    np.testing.assert_array_equal(coherent["position_m"][before], baseline["position_m"][before])
    np.testing.assert_allclose(coherent["position_m"],
                               baseline["position_m"] + displacement,
                               atol=1e-5, equal_nan=True)
    np.testing.assert_allclose(coherent["formal_sigma_xyz_m"],
                               baseline["formal_sigma_xyz_m"], atol=1e-8, equal_nan=True)
    slip_carrier = carrier_fault(carrier, time, FAULT_SATELLITE, kind="slip")
    drift_carrier = carrier_fault(carrier, time, FAULT_SATELLITE, kind="drift")
    np.testing.assert_array_equal(slip_carrier[:, :FAULT_SATELLITE],
                                  carrier[:, :FAULT_SATELLITE])
    np.testing.assert_array_equal(drift_carrier[:, :FAULT_SATELLITE],
                                  carrier[:, :FAULT_SATELLITE])
    slip = solve_history(design, code, slip_carrier, valid, arc)
    drift = solve_history(design, code, drift_carrier, valid, arc)
    for result in (slip, drift):
        np.testing.assert_array_equal(result["position_m"][before],
                                      baseline["position_m"][before])
    result_path = OUTPUT / "faults_pll_weak.npz"
    np.savez_compressed(result_path, observation_time=time,
                        displacement_east_m=displacement[:, 0],
                        baseline_position=baseline["position_m"],
                        coherent_position=coherent["position_m"],
                        slip_position=slip["position_m"],
                        drift_position=drift["position_m"],
                        baseline_formal_sigma=baseline["formal_sigma_xyz_m"],
                        coherent_formal_sigma=coherent["formal_sigma_xyz_m"],
                        slip_formal_sigma=slip["formal_sigma_xyz_m"],
                        drift_formal_sigma=drift["formal_sigma_xyz_m"],
                        admitted_count=valid.sum(axis=1), arc=arc)
    mask = baseline["observable"] & (time >= FAULT_ONSET_S)
    def delta(result):
        difference = np.linalg.norm(result["position_m"][:, :2]
                                    - baseline["position_m"][:, :2], axis=1)
        return dict(count=int(mask.sum()), max_horizontal_delta_m=float(np.max(difference[mask])),
                    p95_horizontal_delta_m=float(np.percentile(difference[mask], 95)))
    return dict(source_tracking_sha256=digest(FAULT_TRACKING),
                result_sha256=digest(result_path), satellite_zero_based=FAULT_SATELLITE,
                satellite_prn=FAULT_SATELLITE + 1, arc_label=0,
                qualified_before_onset=qualified_before,
                qualified_after_onset=qualified_after,
                satellite_selection="fixed PRN11 because qualified observations exist on both sides; not chosen by fault outcome",
                onset_observation_time_s=FAULT_ONSET_S,
                coherent_final_east_m=100., slip_metres=WAVELENGTH,
                drift_metres_per_s=.5,
                coherent=delta(coherent), slip=delta(slip), drift=delta(drift),
                note="Measurement-domain faults; unchanged admission/arc; not generated RF")


def plot_cases(rows):
    labels = [row["label"] for row in rows]
    positions = np.arange(len(rows))
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    axes[0].bar(positions, [row["all_observable"]["count"] / row["opportunities"]
                            for row in rows])
    axes[0].set(ylabel="Observable fraction", ylim=(0, 1))
    axes[1].bar(positions, [row["all_observable"]["max_error_m"] or 0 for row in rows])
    axes[1].set(ylabel="Maximum observable horizontal error (m)", yscale="symlog")
    axes[1].set_xticks(positions, labels, rotation=55, ha="right", fontsize=8)
    for axis in axes:
        axis.grid(alpha=.25)
    figure.savefig(OUTPUT / "validation.png", dpi=160)
    plt.close(figure)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sources = source_hashes()
    rows = []
    for label, tracking, metadata in source_cases():
        print(label, "start", flush=True)
        rows.append(validate_case(label, tracking, metadata))
        print(label, rows[-1]["all_observable"], flush=True)
    print("measurement faults start", flush=True)
    faults = fault_study()
    plot_cases(rows)
    summary = dict(scope="11 saved-track all-history fits plus three measurement-domain faults",
                   source_sha256=sources,
                   history_summary_sha256=digest(HISTORY / "summary.json"),
                   selected_summary_sha256=digest(SELECTED),
                   post_definition="observation_time >= case.attack_start",
                   note="Offline estimate only; no receiver publication or new RF",
                   cases=rows, faults=faults)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    assert source_hashes() == sources
    print("Saved", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
