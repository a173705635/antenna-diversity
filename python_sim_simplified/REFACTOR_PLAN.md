# Core and experiments implementation plan

> For agentic workers: use subagent-driven-development for independent implementation and review; preserve the approved numerical behavior.

**Goal:** one maintained core package and one independent entry per current experiment, with the original research tree frozen once.

**Architecture:** `core/` owns physical models, RF/DFE, acquisition/tracking, observations, detection and positioning. `experiments/` owns scenarios, CLI, evaluation and plots. Neither core nor current tests depend on frozen research drivers. `archive/pre_refactor_2026-09-16/` preserves the original top-level files and tests byte-for-byte; existing large results/validation directories stay in place.

**Tech stack:** existing Python, NumPy, SciPy, matplotlib and pytest only.

**Spec:** user-approved scope in this conversation: RF/DFE/acquisition/tracking, SDCP/multipath, latest single-source receiver, GPS positioning and recorded IQ. Keep algorithms and parameters unchanged. Daily runs do not duplicate source; explicit final freeze remains available. No additional defensive branches.

## Execution and verification

- [x] Freeze 584 original files including untracked document builder; verify SHA256 against every source.
- [x] Capture original `python -m pytest tests -q` baseline: 401 passed, 1 xfailed. Unqualified pytest originally had 139 historical-copy collection errors; fixed discovery in new tree.
- [x] Extract common numerical code with relative package imports; separate GPS simulation/receiver functions from command drivers.
- [x] Extract current single-source receiver and its actual dependencies; frontend/navigate and generated RF match frozen numerical outputs.
- [x] Migrate SDCP baseline/multipath/window-change experiments and independent verification; source copies are explicit opt-in.
- [x] Provide independent GPS SPP, L1/L5, recorded IQ, full-chain and single-source experiment CLIs; retain result arrays, summaries and plots.
- [x] Remove inactive original Python files from active root after snapshot verification. Preserve historical Markdown locations and original results/validation evidence.
- [x] Port applicable existing tests, add package-boundary and old/new numerical comparisons. Current discovery contains 158 tests.
- [x] Independent review completed; full-chain output overwrite risk corrected. Final main suite 149 passed plus 9 additional migrated tests passed. See REFACTOR_VALIDATION.md.

## Decisions / ledger

- Work in the authorized shared project rather than duplicate large local RF archives into a separate checkout. All implementation is confined to python_sim_simplified; the verified pre-change snapshot is the rollback reference. No commits or shared-branch operations are necessary.
- Follow-up authorization: commit and push the completed refactor; archive stays local and ignored. Default tests must not depend on local archive/results; explicit local_evidence tests retain the old/new comparisons.
- Preserve existing algorithm modules where coherent; do not combine unrelated methods just to achieve the illustrative eight-file diagram.
- Frozen legacy execution is explicit, separate from active pytest discovery. The snapshot includes historical tests and report builders for restoration; historical imports must never leak into the active core.
- Shared interfaces: core.gps_simulation.receiver_reference/build_scene/satellite_setup/run_record; core.single_source.frontend/navigate; core.single_source_scene.Case/generate; core.provenance.source_hashes/freeze_source; core.serialization.serializable.
- Parallel ownership: receiver task owns six receiver-specific core modules and receiver regression test; SDCP task owns its two experiment drivers, two verification tools and core.sdcp_observation; coordinator owns all other core modules, packaging, freeze, remaining experiments and integration.
