"""Offline complex-response audit of the saved 116901 weak three-antenna RF.

Replays startup calibration and 20 ms beam weights, but never runs receive
tracking or navigation. Generator-only gain and spoof steering are used after
the replay for diagnosis; neither is supplied to the receiver beam.
"""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from core.array_acquisition import acquire_array
from core.gps_simulation import receiver_reference
from core.presets import default_config, mate70_phase_centers
from core.provenance import source_hashes
from core.recovery_geometry import FS, sky, steering
from core.single_source import beam, calibrate
from core.single_source_scene import local_signal
from core.tracking import track
from experiments.paths import RESULTS


INPUT = RESULTS / "weak-ten" / "inputs" / "116901_weak_30_24_2"
BASELINE = RESULTS / "single_source_suppression_study"
OUTPUT = BASELINE / "response_audit"
ANTENNAS = (0, 1, 2)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def effective_response(weights, steering_vectors, physical_gain, calibration,
                       calibration_rate, time):
    """Return h=wᴴ diag(gtrue/gcal exp(-j rate t)) a and noise gain.

    `weights` is [block, antenna, source], `steering_vectors` is
    [antenna, source], and `time` is one timestamp per block in seconds.
    Physical gain does not multiply receiver thermal noise.
    """
    relative_gain = (physical_gain / calibration)[None, :] * np.exp(
        -1j * time[:, None] * calibration_rate[None, :])
    response = np.einsum("bms,bm,ms->bs", weights.conj(), relative_gain,
                         steering_vectors)
    noise_gain = np.sum(abs(weights / calibration[None, :, None]) ** 2, axis=1)
    return response, noise_gain


def phase_changes(weights, steering_vectors, physical_gain, calibration,
                  calibration_rate, midpoint_time, boundary_time):
    """Separate beam-change phase at one boundary from center-to-center drift."""
    midpoint, _ = effective_response(weights, steering_vectors, physical_gain,
                                     calibration, calibration_rate, midpoint_time)
    before, _ = effective_response(weights[:-1], steering_vectors, physical_gain,
                                   calibration, calibration_rate, boundary_time)
    after, _ = effective_response(weights[1:], steering_vectors, physical_gain,
                                  calibration, calibration_rate, boundary_time)
    boundary_jump = np.angle(after * before.conj())
    midpoint_change = np.angle(midpoint[1:] * midpoint[:-1].conj())
    return boundary_jump, midpoint_change


def generator_truth(metadata, nominal):
    """Recreate only the generator's geometry and two gain draws, not its RF."""
    case = metadata["case"]
    seed = metadata["seed"]
    directions = sky(seed, case["stars"])
    positions = np.array(list(mate70_phase_centers().values()))[:case["antennas"]]
    genuine = steering(directions, positions, 0.)
    np.testing.assert_array_equal(genuine, nominal)
    tangent = np.cross(directions[0], [0, 0, 1.])
    tangent /= np.linalg.norm(tangent)
    angle = np.radians(case["angle"])
    spoof_direction = directions[0] * np.cos(angle) + tangent * np.sin(angle)
    fake = steering(spoof_direction[None], positions, 0.)[:, 0]
    rng = np.random.default_rng(seed)
    gain = np.exp(1j * np.radians(case["phase_deg"] * rng.normal(size=case["antennas"])))
    gain *= 10 ** (case["amplitude_db"] * rng.normal(size=case["antennas"]) / 20)
    return gain, fake


def replay_beam(record):
    """Reproduce frontend through beam weights using 20 ms RF blocks only."""
    samples = record["samples"]
    nominal = record["nominal"]
    codes = record["codes"]
    cfg = default_config()
    cfg.acquisition.doppler_bins_hz = np.arange(-4000., 4001., 250.)
    cfg.acquisition.noncoherent_ms = 100
    cfg.tracking.cn0_window_epochs = 100
    cfg.tracking.max_epochs = round(samples.shape[1] / FS * 1000)
    cfg.tracking.code_replica_bandwidth_hz = .9e6
    cfg.tracking.refine_initial_phase = True
    cfg.tracking.carrier_discriminator = "costas"
    startup = round(.1 * FS)
    width = round(.02 * FS)
    templates = []
    prns = []
    for satellite in range(len(codes)):
        acquired = acquire_array(samples[:, :startup], FS, satellite + 1,
                                 1.023e6, cfg.acquisition)
        if acquired["success"]:
            history = track(samples[0, :startup], FS, satellite + 1, acquired,
                            cfg.tracking, 1575.42e6, 1.023e6)
            if history["success"]:
                prns.append(satellite)
                templates.append((history["epoch_start_samples"][-1] / FS,
                                  history["code_nco_phase_chips"][-1],
                                  history["code_nco_rate_hz"][-1],
                                  history["carrier_nco_frequency_hz"][-1]))
    startup_time = np.arange(startup) / FS
    startup_local = np.stack([local_signal(codes[i], template, startup_time)
                              for i, template in zip(prns, templates)])
    calibration, rate = calibrate(samples[:, :startup], startup_local, nominal[:, prns])

    def calibrated_block(start, end):
        scale = calibration[:, None] * np.exp(
            1j * rate[:, None] * np.arange(start, end)[None, :] / FS)
        return samples[:, start:end] / scale

    calibrated_startup = calibrated_block(0, startup)
    startup_covariance = calibrated_startup @ calibrated_startup.conj().T / startup
    starts = np.arange(0, samples.shape[1], width)
    weights = np.empty((len(starts), len(ANTENNAS), len(codes)), complex)
    gains = np.empty((len(starts), len(codes)))
    ratios = np.empty(len(starts))
    for block_index, start in enumerate(starts):
        end = min(start + width, samples.shape[1])
        block = calibrated_block(start, end)
        block_time = np.arange(start, end) / FS
        local = np.stack([local_signal(codes[i], template, block_time)
                          for i, template in zip(prns, templates)])
        if start < startup:
            weight = nominal / len(nominal)
            retained = np.ones(len(codes))
            ratio = 0.
        else:
            weight, retained, ratio = beam(block, nominal, local,
                                           startup_covariance, prns)
        weights[block_index] = weight
        gains[block_index] = retained
        ratios[block_index] = ratio
    return dict(startup_prns=np.array(prns), calibration=calibration,
                calibration_rate=rate, block_start_samples=starts,
                block_end_samples=np.minimum(starts + width, samples.shape[1]),
                weights=weights, gain=gains, ratio=ratios)


def plot_audit(block_time, replay, true_response, fake_response, boundary_jump):
    figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True,
                                constrained_layout=True)
    axes[0].plot(block_time, replay["ratio"], label="Beam residual trigger")
    axes[0].axhline(3., color="black", linestyle=":", linewidth=1)
    axes[0].set(ylabel="Trigger ratio")
    axes[1].plot(block_time, np.median(abs(true_response), axis=1),
                 label="Median true response amplitude")
    axes[1].plot(block_time, np.median(abs(fake_response), axis=1),
                 label="Median fake response amplitude")
    axes[1].set(ylabel="|Effective response|")
    axes[2].plot(block_time[1:], np.median(abs(boundary_jump), axis=1),
                 label="Median same-boundary true phase jump")
    axes[2].set(xlabel="RF block midpoint (s)", ylabel="|Phase jump| (rad)")
    for axis in axes:
        axis.grid(alpha=.3)
        axis.legend(fontsize=8)
    figure.savefig(OUTPUT / "response_audit.png", dpi=160)
    plt.close(figure)


def main():
    metadata = json.loads((INPUT / "metadata.json").read_text(encoding="utf-8"))
    with np.load(INPUT / "observations.npz") as saved:
        record = dict(saved)
    record["samples"] = np.load(INPUT / "samples.npy", mmap_mode="r")[list(ANTENNAS)]
    input_hashes = {name: digest(INPUT / name)
                    for name in ("samples.npy", "observations.npz", "metadata.json")}
    assert input_hashes["samples.npy"] == metadata["samples_sha256"]
    assert input_hashes["observations.npz"] == metadata["observations_sha256"]
    stage_one = json.loads((BASELINE / "summary.json").read_text(encoding="utf-8"))
    assert input_hashes == stage_one["input_sha256"]
    assert digest(BASELINE / "tracking.npz") == stage_one["tracking_sha256"]
    sources = source_hashes()
    print("Input hashes verified; startup and beam replay start", flush=True)
    replay = replay_beam({**record, "nominal": record["nominal"][list(ANTENNAS)]})
    with np.load(BASELINE / "tracking.npz") as cached:
        for key in ("startup_prns", "calibration", "calibration_rate"):
            np.testing.assert_array_equal(replay[key], cached[key], err_msg=key)
        indices = (cached["time"] / .02).astype(int)
        np.testing.assert_array_equal(replay["gain"][indices], cached["gain"], err_msg="gain")
        np.testing.assert_array_equal(replay["ratio"][indices], cached["ratio"], err_msg="ratio")
        observation_time = cached["time"]
        available_time = cached["available_time"]
        quality = ((cached["lock"] >= .5) & (cached["cn0"] >= 25.)
                   & (cached["gain"] >= .05) & (cached["support"] >= 3.))
    print("Cached startup/gain/ratio exactly matched; truth-only audit start", flush=True)

    physical_gain, fake_steering = generator_truth(metadata, record["nominal"])
    starts = replay["block_start_samples"]
    ends = replay["block_end_samples"]
    midpoint_time = (starts + ends) / (2 * FS)
    boundary_time = starts[1:] / FS
    genuine = record["nominal"][list(ANTENNAS)]
    fake = fake_steering[list(ANTENNAS), None]
    true_response, noise_gain = effective_response(
        replay["weights"], genuine, physical_gain[list(ANTENNAS)],
        replay["calibration"], replay["calibration_rate"], midpoint_time)
    fake_vectors = np.broadcast_to(fake, genuine.shape)
    fake_response, _ = effective_response(
        replay["weights"], fake_vectors, physical_gain[list(ANTENNAS)],
        replay["calibration"], replay["calibration_rate"], midpoint_time)
    true_boundary, true_midpoint = phase_changes(
        replay["weights"], genuine, physical_gain[list(ANTENNAS)],
        replay["calibration"], replay["calibration_rate"], midpoint_time, boundary_time)
    fake_boundary, fake_midpoint = phase_changes(
        replay["weights"], fake_vectors, physical_gain[list(ANTENNAS)],
        replay["calibration"], replay["calibration_rate"], midpoint_time, boundary_time)
    true_start, _ = effective_response(replay["weights"], genuine,
        physical_gain[list(ANTENNAS)], replay["calibration"],
        replay["calibration_rate"], starts / FS)
    true_end, _ = effective_response(replay["weights"], genuine,
        physical_gain[list(ANTENNAS)], replay["calibration"],
        replay["calibration_rate"], ends / FS)
    midpoint_error = np.maximum(abs(true_start - true_response),
                                abs(true_end - true_response))
    channel_half_block_span = (abs(replay["calibration_rate"])
                               * (ends - starts).max() / (2 * FS))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    audit_path = OUTPUT / "block_response.npz"
    np.savez_compressed(audit_path, **replay, midpoint_time=midpoint_time,
                        true_response=true_response, fake_response=fake_response,
                        noise_gain=noise_gain, true_boundary_phase_jump=true_boundary,
                        true_midpoint_phase_change=true_midpoint,
                        fake_boundary_phase_jump=fake_boundary,
                        fake_midpoint_phase_change=fake_midpoint,
                        true_midpoint_edge_abs_error=midpoint_error,
                        channel_half_block_phase_span_rad=channel_half_block_span,
                        physical_gain=physical_gain, fake_steering=fake_steering,
                        cached_observation_time=observation_time,
                        cached_available_time=available_time,
                        cached_quality=quality)
    plot_audit(midpoint_time, replay, true_response, fake_response, true_boundary)
    steady = (available_time >= .5) & quality.any(axis=1)
    score = abs(true_response[indices]) ** 2 / noise_gain[indices]
    quality_score = score[steady[:, None] & quality]
    boundary_by_block = np.zeros_like(true_response.real)
    boundary_by_block[1:] = abs(true_boundary)
    qualified_boundary = boundary_by_block[indices][steady[:, None] & quality]
    ideal_cn0_dbhz = (metadata["cn0_after_db_hz"]
                      + 10 * np.log10(quality_score))
    report = dict(
        scope="Offline truth-only complex-response audit; no receive tracking or navigation",
        source_sha256=sources, input_sha256=input_hashes,
        baseline_tracking_sha256=digest(BASELINE / "tracking.npz"),
        block_response_sha256=digest(audit_path),
        antennas=ANTENNAS, attack_start_s=metadata["case"]["attack_start"],
        block_count=len(starts), startup_prns=replay["startup_prns"].tolist(),
        calibration_rate_rad_s=replay["calibration_rate"].tolist(),
        channel_half_block_phase_span_rad=channel_half_block_span.tolist(),
        steady_qualified_count=int(len(quality_score)),
        steady_qualified_response_snr_gain_median=float(np.median(quality_score)),
        steady_qualified_response_snr_gain_p10=float(np.percentile(quality_score, 10)),
        steady_qualified_response_snr_gain_p90=float(np.percentile(quality_score, 90)),
        steady_qualified_ideal_cno_dbhz_median=float(np.median(ideal_cn0_dbhz)),
        steady_qualified_boundary_jump_abs_rad_median=float(np.median(qualified_boundary)),
        steady_qualified_boundary_jump_abs_rad_p99=float(np.percentile(qualified_boundary, 99)),
        steady_qualified_boundary_jump_abs_rad_max=float(qualified_boundary.max()),
        max_true_midpoint_edge_abs_error=float(midpoint_error.max()),
        beam_triggered_blocks=int((replay["ratio"] >= 3.).sum()),
        generator_truth_usage="truth gain/steering only after exact cached replay verification")
    (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    assert source_hashes() == sources
    print("Audit saved", OUTPUT, "qualified observations", len(quality_score), flush=True)


if __name__ == "__main__":
    main()
