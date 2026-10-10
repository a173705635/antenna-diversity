"""Fixed-basis toy study of directional source-evidence statistics.

This entry uses synthetic 1 ms complex prompts. It never runs RF or changes
the receiver. The F calibrations apply only to an independent fixed basis and
white complex Gaussian noise; mismatch examples deliberately violate that model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, f, ncf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "single_source_suppression_study" / "evidence_model"
SEED = 2026092501
WINDOW_MS = 20
GROUP_MS = 5
GROUPS = WINDOW_MS // GROUP_MS
BATCH = 5000
NOISE_TRIALS = 1_000_000
CASE_TRIALS = 20_000
MISMATCH_TRIALS = 100_000


def statistics(prompts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return old ratio, coherent on F statistic, coherent off F statistic.

    The last axis is an orthonormal residual basis with unit desired direction
    at index zero. Complex white noise has unit mean energy per coordinate.
    """
    trials, _, dimension = prompts.shape
    groups = prompts.reshape(trials, GROUPS, GROUP_MS, dimension)
    on = prompts[:, :, 0]
    off = prompts[:, :, 1:]
    old = np.mean(abs(on) ** 2, axis=1) / np.mean(abs(off) ** 2, axis=(1, 2))

    coherent = groups.sum(axis=2) / np.sqrt(GROUP_MS)
    on_coh = np.mean(abs(coherent[:, :, 0]) ** 2, axis=1)
    off_coh = np.mean(abs(coherent[:, :, 1:]) ** 2, axis=(1, 2))
    off_mean = groups[:, :, :, 1:].mean(axis=2, keepdims=True)
    noise_hat = np.sum(abs(groups[:, :, :, 1:] - off_mean) ** 2, axis=(1, 2, 3))
    noise_hat /= (WINDOW_MS - GROUPS) * (dimension - 1)
    return old, on_coh / noise_hat, off_coh / noise_hat


def gates(dimension: int) -> dict[str, float]:
    """Match Ton's ideal marginal Pfa to old ratio >=3 for this dimension."""
    off_dimensions = dimension - 1
    old_pfa = f.sf(3., 2 * WINDOW_MS, 2 * WINDOW_MS * off_dimensions)
    return {
        "old": 3.,
        "old_pfa": float(old_pfa),
        "ton": float(f.isf(old_pfa, 2 * GROUPS,
                             2 * (WINDOW_MS - GROUPS) * off_dimensions)),
        "toff": float(f.ppf(.99, 2 * GROUPS * off_dimensions,
                            2 * (WINDOW_MS - GROUPS) * off_dimensions)),
    }


def binomial_interval(successes: int, trials: int) -> tuple[float, float]:
    """Exact two-sided Clopper-Pearson 95% interval."""
    lower = 0. if successes == 0 else beta.ppf(.025, successes, trials - successes + 1)
    upper = 1. if successes == trials else beta.ppf(.975, successes + 1, trials - successes)
    return float(lower), float(upper)


def complex_noise(rng: np.random.Generator, trials: int, dimension: int) -> np.ndarray:
    return (rng.standard_normal((trials, WINDOW_MS, dimension))
            + 1j * rng.standard_normal((trials, WINDOW_MS, dimension))) / np.sqrt(2)


def same_data_basis_noise(rng: np.random.Generator, trials: int, dimension: int) -> np.ndarray:
    """Remove one sample-covariance eigenvector, then align a fixed reference.

    This is a data-dependence counterexample, not the receiver's projector.
    """
    raw = complex_noise(rng, trials, dimension + 1)
    covariance = np.einsum("ntm,ntp->nmp", raw, raw.conj()) / WINDOW_MS
    _, vectors = np.linalg.eigh(covariance)
    retained = vectors[:, :, :dimension]
    projected = np.einsum("ntm,nmq->ntq", raw, retained.conj())
    desired = retained[:, 0, :].conj()
    desired /= np.linalg.norm(desired, axis=1, keepdims=True)
    initial = np.broadcast_to(np.eye(dimension, dtype=complex)[None, :, 1:],
                              (trials, dimension, dimension - 1))
    aligned, _ = np.linalg.qr(np.concatenate((desired[:, :, None], initial), axis=2))
    return np.einsum("ntq,nqr->ntr", projected, aligned.conj())


def scenario_prompts(rng: np.random.Generator, trials: int, dimension: int,
                     scenario: dict) -> tuple[np.ndarray, float]:
    """Generate prompts and mean deterministic signal energy per 1 ms.

    Noise has E|n_j|²=1. For a signal along desired coordinate, the returned
    energy equals T*C/N0_out with T=1 ms. Angled/mixed signals are defined by
    their total residual-space energy instead of an antenna or sky DOA.
    """
    kind = scenario["kind"]
    if kind == "same_data_basis_noise":
        return same_data_basis_noise(rng, trials, dimension), 0.
    prompts = complex_noise(rng, trials, dimension)
    if kind == "temporal_ar_noise":
        rho = .4
        for tick in range(1, WINDOW_MS):
            prompts[:, tick] = rho * prompts[:, tick - 1] + np.sqrt(1 - rho**2) * prompts[:, tick]
        return prompts, 0.
    if kind == "spatial_corr_noise":
        correlation = np.full((dimension, dimension), .3)
        np.fill_diagonal(correlation, 1.)
        return prompts @ np.linalg.cholesky(correlation).T, 0.
    if kind == "noise":
        return prompts, 0.

    seconds = np.arange(WINDOW_MS) * .001
    angle = np.deg2rad(scenario.get("angle_deg", 0.))
    direction = np.zeros(dimension, complex)
    direction[0] = np.cos(angle)
    direction[1] = np.sin(angle)
    amplitude = np.sqrt(.001 * 10 ** (scenario["cno_dbhz"] / 10))
    signal = amplitude * direction[None, None, :]
    if kind == "genuine_bit_flip":
        signs = np.where(np.arange(WINDOW_MS) < 7, 1., -1.)
        signal = signal * signs[None, :, None]
    elif kind == "genuine_residual_25hz":
        signal = signal * np.exp(2j * np.pi * 25 * seconds)[None, :, None]
    elif kind == "genuine_phase_jitter":
        phase = rng.normal(0., .2, (trials, WINDOW_MS))
        signal = signal * np.exp(1j * phase)[:, :, None]
    elif kind == "true_fake_mixture":
        true_amplitude = np.sqrt(.001 * 10 ** (30 / 10))
        signal = signal + true_amplitude * np.eye(dimension, dtype=complex)[0][None, None, :]
    prompts += signal
    return prompts, float(np.mean(np.sum(abs(signal) ** 2, axis=-1)))


def scenarios() -> list[dict]:
    cases = [{"name": "noise", "kind": "noise", "trials": NOISE_TRIALS}]
    cases += [{"name": f"genuine_{cno}", "kind": "genuine", "cno_dbhz": cno,
               "trials": CASE_TRIALS} for cno in (30, 32, 35)]
    cases += [{"name": name, "kind": kind, "cno_dbhz": 32, "trials": CASE_TRIALS}
              for name, kind in (("bit_flip_at_7ms", "genuine_bit_flip"),
                                 ("residual_25hz", "genuine_residual_25hz"),
                                 ("phase_jitter_0p2rad", "genuine_phase_jitter"))]
    cases += [{"name": f"wrong_{cno}_{angle}deg", "kind": "wrong_direction",
               "cno_dbhz": cno, "angle_deg": angle, "trials": CASE_TRIALS}
              for cno in (32, 35) for angle in (10, 30, 60)]
    cases += [{"name": f"true30_fake40_{angle}deg", "kind": "true_fake_mixture",
               "cno_dbhz": 40, "angle_deg": angle, "trials": CASE_TRIALS}
              for angle in (30, 60)]
    cases += [{"name": name, "kind": kind, "trials": MISMATCH_TRIALS}
              for name, kind in (("temporal_ar_0p4_noise", "temporal_ar_noise"),
                                 ("spatial_corr_0p3_noise", "spatial_corr_noise"),
                                 ("same_data_basis_noise", "same_data_basis_noise"))]
    return cases


def ideal_probabilities(dimension: int, scenario: dict, threshold: dict) -> dict | None:
    """Ideal fixed-basis marginal rates; no joint-rate independence claim."""
    kind = scenario["kind"]
    if kind not in ("noise", "genuine"):
        return None
    snr = 0. if kind == "noise" else .001 * 10 ** (scenario["cno_dbhz"] / 10)
    off_df = 2 * WINDOW_MS * (dimension - 1)
    residual_df = 2 * (WINDOW_MS - GROUPS) * (dimension - 1)
    noncentrality = 2 * WINDOW_MS * snr
    survival = f.sf if noncentrality == 0 else lambda cut, numerator, denominator: ncf.sf(
        cut, numerator, denominator, noncentrality)
    return {
        "old": float(survival(threshold["old"], 2 * WINDOW_MS, off_df)),
        "ton": float(survival(threshold["ton"], 2 * GROUPS, residual_df)),
        "toff_reject": .01,
    }


def run_case(rng: np.random.Generator, dimension: int, scenario: dict,
             threshold: dict) -> dict:
    counts = {"old": 0, "ton": 0, "joint3": 0, "toff_reject": 0}
    signal_energy = 0.
    trials = scenario["trials"]
    for start in range(0, trials, BATCH):
        size = min(BATCH, trials - start)
        prompts, energy = scenario_prompts(rng, size, dimension, scenario)
        old, ton, toff = statistics(prompts)
        old_pass = old > threshold["old"]
        ton_pass = ton > threshold["ton"]
        off_pass = toff <= threshold["toff"]
        counts["old"] += int(old_pass.sum())
        counts["ton"] += int(ton_pass.sum())
        counts["joint3"] += int((ton_pass & off_pass).sum())
        counts["toff_reject"] += int((~off_pass).sum())
        signal_energy += size * energy
    metrics = {name: {"count": count, "rate": count / trials,
                      "ci95": binomial_interval(count, trials)}
               for name, count in counts.items()}
    predicted = ideal_probabilities(dimension, scenario, threshold)
    comparison = None if predicted is None else {
        name: {"mc_minus_ideal": metrics[name]["rate"] - predicted[name],
               "ideal_binomial_se": float(np.sqrt(predicted[name] * (1 - predicted[name]) / trials))}
        for name in ("old", "ton", "toff_reject")}
    return {"name": scenario["name"], "kind": scenario["kind"], "dimension": dimension,
            "trials": trials, "parameters": {key: value for key, value in scenario.items()
                                           if key not in ("name", "kind", "trials")},
            "residual_signal_snr_1ms": signal_energy / trials,
            "metrics": metrics,
            "ideal_fixed_basis": predicted, "mc_ideal_comparison": comparison}


def plot_results(rows: list[dict], thresholds: dict) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    colors = {"old": "#386cb0", "ton": "#e38b31", "joint3": "#458d6b"}
    for axis, dimension in zip(axes, (2, 3)):
        points = [row for row in rows if row["dimension"] == dimension]
        cno = np.linspace(22, 40, 200)
        for name in ("old", "ton"):
            predicted = [ideal_probabilities(dimension,
                         {"kind": "genuine", "cno_dbhz": float(value)},
                         thresholds[str(dimension)])[name] for value in cno]
            axis.plot(cno, predicted, color=colors[name], label=f"{name} ideal F/ncf")
        genuine = [row for row in points if row["kind"] == "genuine"]
        for name in ("old", "ton", "joint3"):
            axis.scatter([row["parameters"]["cno_dbhz"] for row in genuine],
                         [row["metrics"][name]["rate"] for row in genuine],
                         color=colors[name], marker="o" if name != "joint3" else "x",
                         label=f"{name} MC" if dimension == 2 else None)
        axis.set(title=f"Residual dimension q={dimension}", xlabel="Aligned C/N0_out (dB-Hz)",
                 ylabel="Acceptance probability", ylim=(-.02, 1.02))
        axis.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    figure.savefig(OUTPUT / "ideal_direction.png", dpi=170)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for axis, dimension in zip(axes, (2, 3)):
        points = [row for row in rows if row["dimension"] == dimension
                  and row["kind"] in ("wrong_direction", "true_fake_mixture")]
        labels = [row["name"] for row in points]
        positions = np.arange(len(points))
        for offset, name in zip((-.25, 0, .25), ("old", "ton", "joint3")):
            axis.bar(positions + offset, [row["metrics"][name]["rate"] for row in points],
                     width=.24, color=colors[name], label=name if dimension == 2 else None)
        axis.set(title=f"q={dimension}: wrong / mixed directions", ylabel="Acceptance rate",
                 xticks=positions, xticklabels=labels, ylim=(0, 1))
        axis.tick_params(axis="x", labelrotation=55, labelsize=7)
    axes[0].legend(fontsize=8)
    figure.savefig(OUTPUT / "direction_mismatch.png", dpi=170)
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_path = Path(__file__).resolve()
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    rng = np.random.default_rng(SEED)
    thresholds = {str(q): gates(q) for q in (2, 3)}
    rows = []
    for dimension in (2, 3):
        for scenario in scenarios():
            print(f"q={dimension} {scenario['name']}: {scenario['trials']} trials", flush=True)
            rows.append(run_case(rng, dimension, scenario, thresholds[str(dimension)]))
    plot_results(rows, thresholds)
    summary = {
        "scope": "Synthetic 1 ms prompts only; no receiver or RF changes",
        "seed": SEED, "window_ms": WINDOW_MS, "group_ms": GROUP_MS,
        "batch": BATCH, "source_sha256": source_hash,
        "noise_model": "CN(0,1) per complex coordinate unless mismatch case",
        "signal_snr_definition": "mean residual-space deterministic signal energy per 1 ms / unit coordinate noise energy",
        "gates": thresholds, "rows": rows,
        "limits": ["F/ncf probabilities assume independent fixed orthonormal basis and iid complex noise",
                   "Ton and Toff share a denominator; their joint acceptance has no independent-product calibration",
                   "Same-data eigenbasis toy is not the receiver projector",
                   "20 ms toy ignores RF tracking, navigation, bit synchronization and spatially varying beam weights"],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_hash
    print(f"Saved {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
