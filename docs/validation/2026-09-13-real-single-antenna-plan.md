# Real single-antenna GNSS validation plan

Date: 2026-09-13. Scope authorized: use publicly recorded single-antenna signals to validate the receiver. Continue inline in the existing research workspace; preserve prior uncommitted work.

## Design and acceptance

Use authentic live-sky GPS L1 C/A raw samples with original source metadata and a local SHA-256. Prefer a small continuous segment with public independent receiver output. The Tampere Zenodo page currently returns HTTP 429; the ION HackRF example has downloadable XML, continuous IQ, and MuSNAT reference observations. These are acceptable alternatives for the same goal.

Read signed interleaved IQ, use the documented sample rate and IF, then pass through the existing DFE, acquisition, and tracking. Do not synthesize an antenna, RF noise, navigation bits, ephemeris, or truth for the recorded input. Preserve the 33 MHz simulation default; adapt the audit to the recorded sample rate. Single-antenna data cannot validate SDCP. Without decoded navigation time/ephemeris and a justified comparison reference, do not claim pseudorange or positioning accuracy.

## Steps

- [x] Download a bounded continuous recording and original metadata; record byte range, URL, hashes, acquisition conditions and inconsistencies.
- [x] Add a small audit entry point with strict IQ framing and sample-window checks. First scan all GPS PRNs with the existing acquisition, save positive and negative results.
- [x] Run unchanged tracking as a baseline and retain its diagnostics. Identify genuine live-signal incompatibilities before changing receiver behavior.
- [x] If required, reproduce failures in deterministic tests, make the smallest opt-in receiver correction, and compare the same recorded samples before/after. Check data-bit transitions without concealing accumulated phase discontinuities.
- [x] Evaluate continuous output, discriminator/lock diagnostics, independently accumulated prompt statistics and observed navigation-bit structure. Compare independent receiver observations only with justified time/sign alignment.
- [x] Run relevant regression tests, independently recompute report values from retained traces, and record limits alongside all results. Update report and navigation indexes.

## Artifacts and reproducibility

Entry point: `python_sim_simplified/audit_real_gnss.py`. Original large recording: ignored `python_sim_simplified/results/real_gnss_hackrf/`. Evidence and machine-readable results: `docs/reports/data/2026-09-13-real-single-antenna/`. Narrative report: `docs/reports/2026-09-13-real-single-antenna.md`. Use existing NumPy/SciPy/pytest/matplotlib and Python standard library only.

Completed: ten-second source segment verified; main nine-second ten-PRN comparison, three-PRN DFE quantization, 335 scalar recomputations and 23 navigation acceptance checks. Full suite 162 passed before the final navigation-only negative test; final targeted suite 22 passed. See the linked report for limits.
