# Recorded GPS receiver integration plan

User-approved scope: move reusable reception, LNAV decoding, time synchronization and observation construction out of audit scripts; connect to the existing GPS position solver and validate on real data. Preserve existing experiments and numerical evidence. Execute inline, following the already applied brainstorming, writing-plans, ponytail and test-driven verification workflow.

## Design

`recorded_receiver.py`: sample-file reading, shared DFE/capture/tracking orchestration and an offline GPS receive-to-position API. `gps_lnav.py`: bit/word/subframe synchronization, parity, satellite time anchors, and coherent subframe 1/2/3 ephemeris decoding. `gps_broadcast.py`: broadcast orbit and clock propagation. `gps_observables.py`: common reception-time pseudoranges from decoded TOW and tracked code, carrier observations with unresolved ambiguities, and corrections upstream of the existing `positioning.solve_position`. Audit scripts retain experiments, reporting and truth comparison. No receiver function consumes the antenna reference coordinates or simulation coarse-range truth.

## Checks

- [x] Preserve prior IQ reading and navigation diagnostic imports as compatibility exports; reproduce the existing ten-second observations without changing their archived evidence.
- [x] Test GPS parity, field scaling/signs and IOD consistency; reject invalid or missing timing rather than inventing observations.
- [x] Test transmitter-time mapping at code-period boundaries and gaps; independently test orbit/clock geometry and clock-origin invariance.
- [x] Extend the actual recording to 40 seconds, with byte ranges and hashes. Decode complete subframes and ephemerides where the data supports them.
- [x] Run real static code positioning through the reusable pipeline, record positions, clocks, residuals, navigation status and exclusions. Antenna metadata is evaluation-only, not survey-grade truth.
- [x] Run regressions and data/architecture checks; document results, physical limitations and reproducible commands.

The initial delivery is offline single-frequency GPS code PVT. It does not claim real-time streaming, complete almanac/UTC/ionosphere page coverage, integer ambiguity fixing, calibrated absolute error or dynamic-trajectory validation.

Completed: [report](../reports/2026-09-13-recorded-receiver.md), 10 decoded ephemerides, 380 real code positions, 172 passing regressions. Vertical offset remains an explicitly documented follow-up, not silently corrected.
