# Per-satellite pseudorange diagnosis

Approved scope: diagnose existing GPS L1 SPP/float position bias; keep production defaults and estimator weights unchanged. Continue ponytail and systematic debugging. Reuse existing receiver, continuous code implementation and archived observations.

- [x] Summarize archived per-PRN errors against geometry plus receiver clock, including means, temporal scatter and cross-seed correlations. Truth enters evaluation only.
- [x] Add optional `code_bandwidth_hz=None` to `build_scene` for a diagnostic source control; default signal remains unchanged. Add one diagnostic runner and small numerical checks.
- [x] Generate the same rectangular 8-PRN source and RF/DFE once per seed. Compare rectangular local replica with existing continuous 1.8 MHz local replica on identical samples, without changing acquisition or DLL gains.
- [x] Use linear/no-noise/no-ADC/no-DFE-quantization controls to distinguish aggregate RF effects from replica effects. Compare mixed versus isolated PRNs under the same linear processing. Do not call these realistic noiseless RF precision results.
- [x] Compare rectangular versus continuous sources while keeping local continuous replica fixed; use isolated controls where needed to separate source-grid effects from PRN interaction. All contrasts preserve time, nominal code rate, geometry and original loop gains.
- [x] Recompute ordinary and float positioning for full-chain alternatives, archive observations/diagnostics, verify input/source hashes and result arithmetic, write conclusions without overclaiming an additive noise decomposition. Run focused and existing tests.
