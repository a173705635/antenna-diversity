# Real single-antenna GNSS evidence

Narrative: [report](../../2026-09-13-real-single-antenna.md).

`source/download.json` fixes the ION HackRF live-sky URL, byte range and SHA-256 for the downloaded first 10 seconds. Raw IQ stays in the ignored local results directory. `download_recording.py` reproduces the exact segment and checks its hash.

## Retained runs

| Directory | File interval | Purpose |
|---|---|---|
| `legacy_1s` | 0–1 s | Original tracker; failure evidence |
| `costas_1s` | 0–1 s | Carrier-only change cannot cure input discontinuity |
| `costas_start20ms_1s` | 0.02–1.02 s | Still crosses the later discontinuity |
| `legacy_start1s_1s` | 1–2 s | Original tracker on a new arc |
| `costas_start1s_1s` | 1–2 s | Full PRN scan, short confirmation |
| `costas_start1s_4s` | 1–5 s | PRNs 7/30 duration extension |
| `legacy_start1s_9s` | 1–10 s | Main original-discriminator comparison |
| `costas_start1s_9s` | 1–10 s | Main final validation; all GPS PRNs scanned |
| `costas_quant8_start1s_9s` | 1–10 s | Extra DFE quantization, PRNs 7/13/29 |

While downloading, the file was named `first10s.int8.part`; it is the same prefix now verified as `first10s.int8`. Early manifests retain that execution path. Main nine-second manifests include input-window hashes. No navigation bit truth was supplied. Traces retain native 1 ms epochs.

Summaries retain all requested PRNs, including acquisition failures. NPZs retain actual NCO and E/P/L histories. Early keys `dll_rms_chips_normalized` and `legacy_lock_median` are historical names explained in the report; final runs use dimensionless DLL and configured-lock naming.

`navigation_acceptance.json` adds a postprocessing check for nonconstant bits, consecutive parity-valid words and plausible TLM/HOW. Original per-PRN navigation JSONs retain raw diagnostics. All-zero words pass parity, so parity count alone is insufficient. `ten_words_parity_ok=False` in final Costas runs reflects an incomplete tail after TLM, not a failure of the 13 consecutive words.

`check_continuity.py` independently searches PRNs 7/30 every 10 ms. It does not repair gaps or certify carrier continuity. Initial 3-second and complete 10-second scans are retained separately. `verify_and_plot.py` recomputes statistics/NCO increments, checks navigation acceptance and renders the comparison plot. `source/` holds original metadata, a reference-observation excerpt and test logs. `receiver_source/` preserves relevant final code. `sha256.json` covers evidence files except itself.

This is neither dual-antenna SDCP evidence nor absolute phase/pseudorange/position accuracy evidence. Reuse rights are not inferred from the short XML owner field.
