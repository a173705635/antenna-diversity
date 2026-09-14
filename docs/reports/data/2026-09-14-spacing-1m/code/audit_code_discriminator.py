"""Noiseless LOS DLL curves; keep source and receiver samples identical."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from audit_multipath_baseline import baseline_config
from dfe import process_dfe
from gnss import _desired_waveform
from tracking import _TrackingState, _correlate_epoch, bandlimited_code_replica


def run_curves():
    offsets = np.linspace(-0.3, 0.3, 241)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    records = []
    for ax, (doppler, delay) in zip(axes.flat, [
        (1500., 137.25), (-1500., 137.25),
        (1500., 137.37), (-1500., 137.37),
    ]):
        cfg = baseline_config()
        cfg.signal.doppler_hz = doppler
        cfg.signal.code_delay_chips = delay
        cfg.dfe.output_bits = 0
        n = np.arange(round(cfg.signal.fs_rf_hz * .008))
        waveform, code, received_rate = _desired_waveform(cfg, n, n / cfg.signal.fs_rf_hz)
        data, info = process_dfe(waveform[None, :], cfg.dfe,
                                 cfg.signal.fs_rf_hz, cfg.signal.if_hz)
        fs = info["fs_out_hz"]
        start = round(.005 * fs)
        indices = np.arange(round(.001 * fs))
        block = data[0, start:start + indices.size]
        curves = {}
        for label, replica in [
            ("Rectangular", code),
            ("Ideal lowpass", bandlimited_code_replica(code, cfg.signal.code_rate_hz, cfg.dfe.cutoff_hz)),
        ]:
            values = []
            for offset in offsets:
                state = _TrackingState(doppler,
                    cfg.signal.initial_phase_rad + 2*np.pi*doppler*start/fs,
                    start*received_rate/fs-delay+offset, received_rate)
                early, _, late = _correlate_epoch(
                    block, indices, replica, state, fs, cfg.tracking.early_late_spacing_chips)
                values.append(float((abs(early)-abs(late))/(abs(early)+abs(late))))
            curves[label] = values
            ax.plot(offsets, values, label=label)
        ax.axhline(0, color="gray", linewidth=.7)
        ax.axvline(0, color="gray", linewidth=.7)
        ax.set(title=f"Doppler {doppler:g} Hz; delay {delay:g} chip",
               xlabel="Local phase minus LOS phase (chip)", ylabel="DLL discriminator")
        ax.legend()
        records.append({"doppler_hz": doppler, "delay_chips": delay, "curves": curves})
    destination = Path(__file__).resolve().parent / "results" / "multipath_baseline_audit"
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / "discriminator.png", dpi=160)
    plt.close(fig)
    (destination / "discriminator.json").write_text(json.dumps({
        "scope": "LOS, linear DFE, no noise, no interference; exact carrier/LOS phase for diagnostic only",
        "offset_chips": offsets.tolist(), "records": records,
    }, indent=2), encoding="utf-8")
    print(destination / "discriminator.png")


if __name__ == "__main__":
    run_curves()
