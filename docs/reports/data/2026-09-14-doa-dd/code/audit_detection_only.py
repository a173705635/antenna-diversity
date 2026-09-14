"""Detection-only paired-noise pilot study. No suppression or tracking metrics."""

import argparse
import json
from pathlib import Path

import numpy as np

from acquisition import acquire
from antenna import build_antenna_model, direction_vector
from audit_multipath_baseline import baseline_config
from config import Direction, MultipathPath
from dfe import process_dfe
from diversity import combine_known
from gnss import generate_scenario
from multipath import detect_multipath, refine_single_path_delays
from rf import process_rf


def trial(kind, seed):
    cfg = baseline_config()
    cfg.signal.duration_ms = 6
    cfg.signal.code_bandwidth_hz = cfg.dfe.cutoff_hz
    cfg.random_seed = seed
    direction = {"echo_grid": Direction(110.,15.), "echo_offgrid": Direction(113.,17.),
                 "echo_same_direction": cfg.signal.true_direction}.get(kind)
    if direction is not None:
        cfg.signal.multipath_paths = [MultipathPath(direction, relative_power_db=-6.,
            excess_delay_chips=.5, reflection_phase_rad=np.pi)]
    channels, truth = generate_scenario(cfg)
    adc, _ = process_rf(channels,cfg.rf,cfg.reference_impedance_ohm,seed)
    channels, info = process_dfe(adc,cfg.dfe,cfg.signal.fs_rf_hz,cfg.signal.if_hz)
    pilot, _ = combine_known(channels,truth["desired_steering_assumed"],"egc",.01)
    acq = acquire(pilot,info["fs_out_hz"],cfg.signal.prn,cfg.signal.code_rate_hz,cfg.acquisition)
    row = {"kind":kind,"seed":seed,"acquired":bool(acq["success"]),
           "has_candidate":False,"candidate_count":0,"status":"acquisition_failed",
           "delay_error_chips":None,"angle_error_deg":None,"refined_delay_error_chips":None}
    if not acq["success"]:
        return row
    detection = detect_multipath(channels,info["fs_out_hz"],cfg.signal.prn,
        cfg.signal.code_rate_hz,acq,cfg.acquisition,build_antenna_model(cfg.phone,cfg.signal.carrier_hz),
        cfg.signal.assumed_direction,cfg.signal.assumed_polarization,cfg.multipath_detection,.95)
    candidates = detection["candidates"]
    row.update(status=detection["status"],has_candidate=bool(candidates),candidate_count=len(candidates))
    if direction is not None and candidates:
        # Strongest candidate chosen from data, not the candidate closest to truth.
        best = candidates[0]
        row["delay_error_chips"] = abs(best["excess_delay_chips"]-.5)
        cosine = np.clip(direction_vector(best["direction"]) @ direction_vector(direction),-1,1)
        row["angle_error_deg"] = float(np.degrees(np.arccos(cosine)))
        if len(candidates)==1 and best["spatial_status"]=="resolvable":
            fit = refine_single_path_delays(channels,info["fs_out_hz"],cfg.signal.prn,
                cfg.signal.code_rate_hz,cfg.signal.carrier_hz,acq,truth["desired_steering_assumed"],
                best["steering_estimated"],cfg.dfe.cutoff_hz,3.)
            row["delay_fit_status"] = fit["status"]
            if fit["status"]=="refined":
                row["refined_delay_error_chips"] = abs(fit["excess_delay_chips"]-.5)
    return row


def summarize_trials(rows):
    summary = []
    for kind in dict.fromkeys(row["kind"] for row in rows):
        subset = [row for row in rows if row["kind"]==kind]
        acquired = sum(row["acquired"] for row in subset)
        candidates = sum(row["has_candidate"] for row in subset)
        entry = {"kind":kind,"trials":len(subset),"acquired":acquired,
                 "candidate_positive":candidates,"candidate_rate_unconditional":candidates/len(subset),
                 "candidate_rate_given_acquisition":candidates/acquired if acquired else None}
        for key in ("delay_error_chips","angle_error_deg","refined_delay_error_chips"):
            values=[row[key] for row in subset if row[key] is not None]
            entry[key+"_count"]=len(values)
            entry[key+"_median"]=float(np.median(values)) if values else None
        summary.append(entry)
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials",type=int,default=20)
    args=parser.parse_args()
    if args.trials<1: parser.error("trials must be positive")
    rows=[]
    for kind in ("los_only","echo_grid","echo_offgrid","echo_same_direction"):
        for seed in range(100,100+args.trials): rows.append(trial(kind,seed))
        print(summarize_trials(rows)[-1],flush=True)
    destination=Path(__file__).resolve().parent/"results"/"detection_only"
    destination.mkdir(parents=True,exist_ok=True)
    (destination/"baseline.json").write_text(json.dumps({
        "scope":"current detector; no external interferers; default RF thermal noise and quantization; 6 ms records; EGC acquisition; no tracking or LCMV",
        "interpretation":"candidate-positive rate is not confirmed path-identification probability; no threshold calibration; descriptive pilot study only",
        "summary":summarize_trials(rows),"trials":rows},indent=2,allow_nan=False),encoding="utf-8")
