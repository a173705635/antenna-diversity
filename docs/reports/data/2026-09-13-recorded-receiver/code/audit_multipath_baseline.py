"""Reproduce the LOS-only sampling baseline, without external interferers.

Diagnostic runner; uses the existing receiver without changing its algorithms.
The tail window is descriptive, not a declaration that tracking has settled.
"""

import json
import argparse
from pathlib import Path

import numpy as np

from acquisition import acquire
from config import SPEED_OF_LIGHT_MPS
from dfe import process_dfe
from gnss import generate_scenario
from metrics import tracking_code_bias_chips
from pipeline import run_end_to_end
from presets import default_config
from tracking import _sample_code, track


def baseline_config():
    cfg = default_config()
    cfg.interferers = []
    cfg.signal.multipath_paths = []
    cfg.diversity.modes = ("egc",)
    cfg.multipath_detection.enable = False
    cfg.multipath_suppression.enable = False
    return cfg


def summarize(name, tracking, truth, fs_hz, cfg):
    record = {"case": name, "samples_per_chip": fs_hz / cfg.signal.code_rate_hz}
    if not tracking["success"]:
        return {**record, "status": tracking["status"]}
    error_m = tracking_code_bias_chips(
        tracking, truth, fs_hz, cfg.tracking.coherent_ms
    ) * SPEED_OF_LIGHT_MPS / cfg.signal.code_rate_hz
    if error_m.size == 0:
        return {**record, "status": "no_tracking_epochs"}
    tail = error_m[-30:]
    return {
        **record,
        "status": "epochs_available_not_lock_verified",
        "epochs": int(error_m.size),
        "all_epochs_rms_m": float(np.sqrt(np.mean(error_m**2))),
        "tail_epochs": int(tail.size),
        "tail_mean_code_phase_error_m": float(np.mean(tail)),
        "tail_std_m": float(np.std(tail)),
        "tail_rms_m": float(np.sqrt(np.mean(tail**2))),
        "code_phase_error_m": error_m.tolist(),
    }


def run_audit(bandlimited=False):
    records = []
    for noise in (True, False):
        cfg = baseline_config()
        if bandlimited:
            cfg.tracking.code_replica_bandwidth_hz = cfg.dfe.cutoff_hz
        if not noise:
            cfg.rf.temperature_k = 0.0
        result = run_end_to_end(cfg)
        record = summarize(
            "full_chain_noise" if noise else "full_chain_no_noise",
            result["branches"]["egc"]["tracking"],
            result["truth"], result["dfe_info"]["fs_out_hz"], cfg,
        )
        record["dfe_nonzero_samples"] = int(np.count_nonzero(result["dfe_channels"]))
        record["adc_nonzero_samples"] = int(np.count_nonzero(result["adc_codes"]))
        records.append(record)

    for decimation in (1, 2, 4):
        cfg = baseline_config()
        if bandlimited:
            cfg.tracking.code_replica_bandwidth_hz = cfg.dfe.cutoff_hz
        cfg.dfe.decimation = decimation
        cfg.dfe.output_bits = 0
        channels, truth = generate_scenario(cfg)
        channels, info = process_dfe(
            channels, cfg.dfe, cfg.signal.fs_rf_hz, cfg.signal.if_hz
        )
        steering = truth["desired_steering_assumed"]
        combined = steering.conj() @ channels / np.vdot(steering, steering)
        fs_hz = info["fs_out_hz"]
        acquisition = acquire(
            combined, fs_hz, cfg.signal.prn, cfg.signal.code_rate_hz, cfg.acquisition
        )
        tracking = track(
            combined, fs_hz, cfg.signal.prn, acquisition, cfg.tracking,
            cfg.signal.carrier_hz, cfg.signal.code_rate_hz,
        )
        records.append(summarize(
            f"linear_no_noise_decimation_{decimation}", tracking, truth, fs_hz, cfg
        ))

    # An exact nominal-rate example exposes the local replica's staircase:
    # all offsets strictly inside one quarter-chip interval produce the same code.
    phase = np.arange(4092) / 4.0
    code = truth["code"]
    replica_a = _sample_code(code, phase + 0.01)
    replica_b = _sample_code(code, phase + 0.24)
    return {
        "tracking_replica": "ideal_lowpass" if bandlimited else "rectangular",
        "scope": "LOS only; no multipath; no external interference; seed 42; 100 ms",
        "error_sign": "local code phase minus LOS true code phase; not pseudorange sign",
        "tail_definition": "last 30 available epochs; stationarity not verified",
        "nominal_4spc_replica_changed_samples_0_01_to_0_24_chip": int(
            np.count_nonzero(replica_a != replica_b)
        ),
        "records": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bandlimited", action="store_true")
    args = parser.parse_args()
    result = run_audit(args.bandlimited)
    destination = Path(__file__).resolve().parent / "results" / "multipath_baseline_audit"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / ("baseline_bandlimited.json" if args.bandlimited else "baseline.json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    for record in result["records"]:
        print(json.dumps({k: v for k, v in record.items() if k != "code_phase_error_m"}))
    print(path)
