"""Offline five-block common-source null history on cached beam responses.

The current block's weights are available only after that RF block has ended.
Same-block scores are retrospective; shifted scores apply both methods' previous
block weights to the current block. Generator truth is used only for scoring.
"""

import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np

from experiments.paths import RESULTS
from core.provenance import source_hashes
from core.recovery_geometry import FS


ROOT = RESULTS / "single_source_suppression_study"
INPUT = ROOT / "response_audit" / "block_response.npz"
NOMINAL = RESULTS / "weak-ten" / "inputs" / "116901_weak_30_24_2" / "observations.npz"
OUTPUT = ROOT / "subspace_history"
TRIGGER = 3.
WINDOW = 5
LOADING = .01


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def smooth_nulls(directions, triggered):
    """Average up to five consecutive observed null projectors, causally."""
    projectors = np.tile(np.eye(3, dtype=complex), (len(directions), 1, 1))
    count = np.zeros(len(directions), int)
    eigengap = np.full(len(directions), np.nan)
    history = []
    for block, direction in enumerate(directions):
        if not triggered[block]:
            history.clear()
            continue
        history.append(np.outer(direction, direction.conj()))
        history = history[-WINDOW:]
        count[block] = len(history)
        eigenvalues, vectors = np.linalg.eigh(np.mean(history, axis=0))
        eigengap[block] = eigenvalues[-1] - eigenvalues[-2]
        projectors[block] -= np.outer(vectors[:, -1], vectors[:, -1].conj())
    return projectors, count, eigengap


def candidate_weights(weights, nominal, ratio):
    """Estimate one antenna-space null from each triggered 3-by-star beam block.

    Return causal candidate weights and diagnostics. A nontriggered block clears
    the history and retains the original weights. The five-block window includes
    the current block, so these weights cannot be applied until its end.
    """
    result = weights.copy()
    singular_values = np.full((len(weights), 3), np.nan)
    directions = np.zeros((len(weights), 3), complex)
    triggered = ratio >= TRIGGER
    for block, matrix in enumerate(weights):
        if ratio[block] < TRIGGER:
            continue
        left, values, _ = np.linalg.svd(matrix, full_matrices=True)
        assert values[2] <= 1e-10 * values[0] < values[1], "Triggered beam must have rank two"
        singular_values[block] = values
        directions[block] = left[:, -1]
    projectors, count, eigengap = smooth_nulls(directions, triggered)
    assert np.all(eigengap[triggered] > 1e-10), "A unique average null is required"
    for block in np.flatnonzero(triggered):
        projected = projectors[block] @ nominal
        denominator = np.sum(nominal.conj() * projected, axis=0).real + LOADING
        result[block] = projected / denominator
    return result, singular_values, eigengap, count


def score(weights, nominal, fake, physical_gain, calibration):
    """Truth-only output fake/true power ratio and white-noise SNR gain."""
    channel = physical_gain / calibration
    true_response = np.einsum("bms,ms->bs", weights.conj(), channel[:, None] * nominal)
    fake_response = np.einsum("bms,m->bs", weights.conj(), channel * fake)
    noise_gain = np.sum(abs(weights / calibration[None, :, None]) ** 2, axis=1)
    js = 10 * abs(fake_response) ** 2 / abs(true_response) ** 2
    snr_gain = abs(true_response) ** 2 / noise_gain
    return true_response, fake_response, noise_gain, js, snr_gain


def step_counterexample():
    """Synthetic complex-space direction step; this is not a physical DOA run."""
    first = np.array([1., 0., 0.], complex)
    second = np.array([.5, np.sqrt(.75), 0.], complex)
    sources = np.array([first] * WINDOW + [second] * WINDOW)
    residual = np.empty(WINDOW)
    for offset in range(WINDOW):
        recent = sources[offset + 1:WINDOW + offset + 1]
        covariance = np.mean([np.outer(v, v.conj()) for v in recent], axis=0)
        _, vectors = np.linalg.eigh(covariance)
        residual[offset] = 1 - abs(np.vdot(vectors[:, -1], second)) ** 2
    shifted_first = 1 - abs(np.vdot(first, second)) ** 2
    return sources, residual, shifted_first


def plot(time, triggered, same, shifted, step_residual):
    figure, axes = plt.subplots(3, 1, figsize=(9, 9), constrained_layout=True)
    for label, values in same.items():
        axes[0].plot(time[triggered], np.median(10 * np.log10(values[triggered]), axis=1), label=label)
    axes[0].set(ylabel="Same-block median fake/true (dB)")
    for label, values in shifted.items():
        axes[1].plot(time[1:][triggered[1:]],
                     np.median(10 * np.log10(values[triggered[1:]]), axis=1), label=label)
    axes[1].set(xlabel="RF block midpoint (s)", ylabel="Previous-weight fake/true (dB)")
    axes[2].plot(np.arange(1, WINDOW + 1), step_residual, marker="o")
    axes[2].axhline(.75, color="gray", linestyle=":", label="Previous-block first step")
    axes[2].set(xlabel="Blocks after synthetic source step", ylabel="Residual source power")
    for axis in axes:
        axis.grid(alpha=.3)
        axis.legend(fontsize=8)
    figure.savefig(OUTPUT / "subspace_history.png", dpi=160)
    plt.close(figure)


def main():
    sources_hash = source_hashes()
    with np.load(INPUT) as saved:
        response = dict(saved)
    with np.load(NOMINAL) as saved:
        nominal = saved["nominal"][:3]
    weights = response["weights"]
    ratio = response["ratio"]
    assert np.all(response["calibration_rate"] == 0), "This score assumes static calibration"
    candidate, singular_values, eigengap, count = candidate_weights(weights, nominal, ratio)
    physical_gain = response["physical_gain"][:3]
    fake = response["fake_steering"][:3]
    calibration = response["calibration"]
    baseline_score = score(weights, nominal, fake, physical_gain, calibration)
    candidate_score = score(candidate, nominal, fake, physical_gain, calibration)
    shifted_baseline = score(weights[:-1], nominal, fake, physical_gain, calibration)
    shifted_candidate = score(candidate[:-1], nominal, fake, physical_gain, calibration)
    triggered = ratio >= TRIGGER
    full = count == WINDOW
    shifted_full = full[:-1] & triggered[1:]
    sources, step_residual, step_shifted_first = step_counterexample()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    arrays = OUTPUT / "subspace_history.npz"
    np.savez_compressed(
        arrays, baseline_weights=weights, candidate_weights=candidate,
        ratio=ratio, triggered=triggered, history_count=count,
        full_window_same=full, full_window_shifted=shifted_full,
        block_end_available_time_s=response["block_end_samples"] / FS,
        singular_values=singular_values, eigengap=eigengap,
        midpoint_time=response["midpoint_time"],
        baseline_true_response=baseline_score[0], baseline_fake_response=baseline_score[1],
        baseline_noise_gain=baseline_score[2], baseline_js=baseline_score[3],
        baseline_snr_gain=baseline_score[4],
        candidate_true_response=candidate_score[0], candidate_fake_response=candidate_score[1],
        candidate_noise_gain=candidate_score[2], candidate_js=candidate_score[3],
        candidate_snr_gain=candidate_score[4],
        shifted_baseline_js=shifted_baseline[3],
        shifted_baseline_snr_gain=shifted_baseline[4],
        shifted_candidate_js=shifted_candidate[3],
        shifted_candidate_snr_gain=shifted_candidate[4],
        synthetic_source_vectors=sources, synthetic_step_residual=step_residual,
        synthetic_shifted_first_residual=step_shifted_first)
    plot(response["midpoint_time"], triggered,
         {"Baseline": baseline_score[3], "Five-block": candidate_score[3]},
         {"Baseline": shifted_baseline[3], "Five-block": shifted_candidate[3]},
         step_residual)

    def median_db(values, selection):
        return float(np.median(10 * np.log10(values[selection])))

    summary = dict(
        scope="Cached three-antenna static-source offline response; no RF, tracking, or navigation",
        input_sha256={str(INPUT): digest(INPUT), str(NOMINAL): digest(NOMINAL)},
        source_sha256=sources_hash, arrays_sha256=digest(arrays),
        trigger_threshold=TRIGGER, history_window_blocks=WINDOW, loading=LOADING,
        triggered_blocks=int(triggered.sum()), full_window_blocks=int(full.sum()),
        shifted_full_window_blocks=int(shifted_full.sum()),
        triggered_rank_two=True, calibration_rate_rad_s=response["calibration_rate"].tolist(),
        full_window_same_block_baseline_js_median_db=median_db(baseline_score[3], full),
        full_window_same_block_candidate_js_median_db=median_db(candidate_score[3], full),
        full_window_same_block_baseline_snr_gain_median_db=median_db(baseline_score[4], full),
        full_window_same_block_candidate_snr_gain_median_db=median_db(candidate_score[4], full),
        full_window_shifted_baseline_js_median_db=median_db(shifted_baseline[3], shifted_full),
        full_window_shifted_candidate_js_median_db=median_db(shifted_candidate[3], shifted_full),
        full_window_shifted_baseline_snr_gain_median_db=median_db(shifted_baseline[4], shifted_full),
        full_window_shifted_candidate_snr_gain_median_db=median_db(shifted_candidate[4], shifted_full),
        warmup_same_block_baseline_js_median_db=median_db(baseline_score[3], triggered & ~full),
        warmup_same_block_candidate_js_median_db=median_db(candidate_score[3], triggered & ~full),
        synthetic_step_residual=step_residual.tolist(),
        synthetic_shifted_first_residual=float(step_shifted_first),
        limitation="Same-block scoring looks back after block completion; synthetic step is not a physical DOA model")
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    assert source_hashes() == sources_hash
    print("Saved", OUTPUT)


if __name__ == "__main__":
    main()
