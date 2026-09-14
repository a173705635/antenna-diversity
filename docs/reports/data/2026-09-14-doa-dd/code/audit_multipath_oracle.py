"""Pure multipath ideal-reference audit; truth weights are NOT a detector.

Uses the shared signal generator, DFE, combiners and receiver. No RF noise or
quantization, so spatial constraints can be checked independently of detection.
"""

import json
import argparse
from pathlib import Path

import numpy as np

from acquisition import acquire
from antenna import build_antenna_model
from audit_multipath_baseline import baseline_config, summarize
from config import Direction, MultipathPath
from dfe import process_dfe
from diversity import combine_known, combine_lcmv
from gnss import generate_scenario
from tracking import track
from multipath import detect_multipath, refine_single_path_delays


def run_case(echo=False, phase=0., doppler=1500., delay=137.25, decimation=4, refine=False,
             compare_detector=False, echo_delay=.5, echo_direction=(110., 15.)):
    cfg = baseline_config()
    cfg.signal.code_bandwidth_hz = cfg.dfe.cutoff_hz
    cfg.tracking.code_replica_bandwidth_hz = cfg.dfe.cutoff_hz
    cfg.tracking.refine_initial_phase = refine
    cfg.signal.doppler_hz = doppler
    cfg.signal.code_delay_chips = delay
    cfg.dfe.output_bits = 0
    cfg.dfe.decimation = decimation
    if echo:
        cfg.signal.multipath_paths = [MultipathPath(
            Direction(*echo_direction), relative_power_db=-6.,
            excess_delay_chips=echo_delay, reflection_phase_rad=phase)]
    channels, truth = generate_scenario(cfg)
    channels, info = process_dfe(channels, cfg.dfe, cfg.signal.fs_rf_hz, cfg.signal.if_hz)
    los = truth["desired_steering_true"]
    detection = None
    delay_refinement = None
    if compare_detector:
        # Fixed EGC pilot avoids selecting the oracle branch or using truth to
        # choose a favourable acquisition result for the detector.
        pilot, _ = combine_known(channels, truth["desired_steering_assumed"], "egc", .01)
        pilot_acq = acquire(pilot, info["fs_out_hz"], cfg.signal.prn,
                            cfg.signal.code_rate_hz, cfg.acquisition)
        if not pilot_acq["success"]:
            raise ValueError("EGC pilot acquisition failed in detector audit")
        detection = detect_multipath(channels, info["fs_out_hz"], cfg.signal.prn,
            cfg.signal.code_rate_hz, pilot_acq, cfg.acquisition,
            build_antenna_model(cfg.phone, cfg.signal.carrier_hz),
            cfg.signal.assumed_direction, cfg.signal.assumed_polarization,
            cfg.multipath_detection, cfg.multipath_suppression.max_steering_correlation)
        candidates = detection["candidates"]
        if len(candidates) == 1 and candidates[0]["spatial_status"] == "resolvable":
            delay_refinement = refine_single_path_delays(
                channels, info["fs_out_hz"], cfg.signal.prn, cfg.signal.code_rate_hz,
                cfg.signal.carrier_hz, pilot_acq, truth["desired_steering_assumed"],
                candidates[0]["steering_estimated"], cfg.dfe.cutoff_hz,
                cfg.multipath_detection.max_excess_delay_chips)
    rows = []
    modes = ("single", "egc", "oracle_lcmv", "detected_lcmv") if compare_detector else ("single", "egc", "oracle_lcmv")
    for mode in modes:
        if mode in ("oracle_lcmv", "detected_lcmv"):
            steerings = ([p["steering_true"] for p in truth["multipath_paths"]]
                         if mode == "oracle_lcmv" else
                         [p["steering_estimated"] for p in detection["candidates"]
                          if p["spatial_status"] == "resolvable"])
            constraint_los = los if mode == "oracle_lcmv" else truth["desired_steering_assumed"]
            combined, combining = combine_lcmv(channels, constraint_los,
                steerings, .01, .95)
        else:
            combined, combining = combine_known(channels, los, mode, .01)
        acq = acquire(combined, info["fs_out_hz"], cfg.signal.prn,
                      cfg.signal.code_rate_hz, cfg.acquisition)
        tr = (track(combined, info["fs_out_hz"], cfg.signal.prn, acq,
                    cfg.tracking, cfg.signal.carrier_hz, cfg.signal.code_rate_hz)
              if acq["success"] else {"success": False, "status": "acquisition_failed"})
        row = summarize(mode, tr, truth, info["fs_out_hz"], cfg)
        w = combining["weights"]
        row.update({
            "los_gain": float(abs(np.vdot(w, los))),
            "echo_gain": [float(abs(np.vdot(w, p["steering_true"]))) for p in truth["multipath_paths"]],
            "white_noise_gain": float(np.vdot(w, w).real),
            "acquired": bool(acq["success"]),
            "oracle_uses_truth": mode == "oracle_lcmv",
            "initialization": tr.get("initialization"),
            "suppression_status": combining.get("suppression_status"),
        })
        rows.append(row)
    detected = None if detection is None else {
        "status": detection["status"],
        "single_path_delay_refinement": delay_refinement,
        "candidates": [{"delay_chips": p["excess_delay_chips"],
                         "azimuth_deg": p["direction"].azimuth_deg,
                         "elevation_deg": p["direction"].elevation_deg,
                         "spatial_status": p["spatial_status"]}
                        for p in detection["candidates"]]}
    return {"echo": echo, "echo_delay_chips": echo_delay, "echo_direction_deg": echo_direction,
            "detection": detected,
            "reflection_phase_rad": phase, "doppler_hz": doppler,
            "code_delay_chips": delay, "decimation": decimation, "refine": refine, "branches": rows}


def run_audit(refine=False, compare_detector=False):
    baseline = [run_case(doppler=d, delay=t, decimation=q, refine=refine)
                for d, t in [(1500., 137.25), (-1500., 137.25),
                             (1500., 137.37), (-1500., 137.37)]
                for q in (1, 4)]
    echoes = [run_case(echo=True, phase=p, refine=refine) for p in (0., np.pi/2, np.pi)]
    if compare_detector:
        echoes = [run_case(echo=True, phase=p, refine=refine, compare_detector=True, echo_delay=d)
                  for d in (.25, .5, .7) for p in (0., np.pi)]
        echoes.append(run_case(echo=True, phase=np.pi, refine=refine, compare_detector=True,
                               echo_direction=(113., 17.)))
    return {"scope": "no external interference, no RF noise/ADC/output quantization; linear DFE",
            "source": "ideal lowpass 1.8 MHz; normalized mean code power",
            "limitations": "shared source/local reference is consistency evidence, not independent physical validation; last 30 epochs not certified steady state",
            "baseline": baseline, "single_echo": echoes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--compare-detector", action="store_true")
    args = parser.parse_args()
    result = run_audit(args.refine, args.compare_detector)
    name = "oracle_refined.json" if args.refine else "oracle.json"
    if args.compare_detector:
        name = "detector_refined.json" if args.refine else "detector_coarse.json"
    path = Path(__file__).resolve().parent / "results" / "multipath_baseline_audit" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    for case in result["baseline"] + result["single_echo"]:
        print("case", {k:v for k,v in case.items() if k != "branches"}, flush=True)
        for row in case["branches"]:
            print({k:v for k,v in row.items() if k != "code_phase_error_m"}, flush=True)
    print(path)
