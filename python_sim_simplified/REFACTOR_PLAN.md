# Core and experiments implementation plan

## 2026-10-12: approved research-direction layout

**Goal:** share the common receiver code, separate interference/spoofing/multipath algorithms and experiments, and update their consumers and documentation together.

**Approved design:** keep common modules directly in `core/`; place dedicated modules in `core/interference/`, `core/spoofing/`, and `core/multipath/`. Mirror those directions under `experiments/`, `docs/`, and `results/`. General GPS/recorded-IQ experiments and `experiments/paths.py` remain shared. Keep filenames and numerical behavior; no forwarding wrappers or duplicate algorithms. `full_chain` belongs to interference as a documented mixed interference/multipath demonstration, with its existing configuration unchanged.

**Global constraints:** work in this existing authorized workspace; modify only the maintained simplified project. Preserve local archive and frozen result bytes, research parameters, source identity checks, and independent verification. Do not commit or push as part of this task. No new dependencies or defensive fallback paths. Preserve general navigation/recorded data locations when not owned by a direction.

### Follow-up ownership correction (2026-10-12)

The first directional pass left six dedicated modules in the shared layer. Re-audit actual consumers, not hypothetical future reuse. Move the SDCP quartet to `core/multipath/`, the measured common-emitter array acquisition/projection pair to `core/spoofing/`, and the five multipath-only plotting functions to `core/multipath/plotting.py`. Keep all function bodies and experiment parameters unchanged. Update imports, documentation, recursive freezes and historical-key file location mapping together; never rewrite old evidence keys or relax byte-identity checks.

`config.py` and `models.py` remain shared contracts; `gnss.py` and `pipeline.py` are joint orchestration; `metrics.py` and the remaining `plotting.py` are common receiver/branch comparisons. A shared orchestrator may call multiple research modules; that does not make those dedicated modules shared. `configs/motion_summary.json` is an old spoofing result summary, not a configuration: preserve its bytes under `docs/spoofing/motion_summary.json`, map the old evidence key, and exclude it from new source/config manifests. Preserve the three pre-existing plotting/test/document changes without reverting them.

Validation: first reproduce missing directional ownership with layout tests, then run affected numerical/plotting/provenance tests, the full default suite, module and link checks, definition-level AST comparison against the pre-correction working files, frozen archive verification, and independent read-only review. Record outcomes in REFACTOR_VALIDATION.md. The following completed ledger describes the initial pass, not this correction.

**Follow-up outcome:** completed without staging or pushing. Two historical report-key lookup errors found by review were reproduced and fixed with strict byte checks, not fallback behavior. Final verification: 556 default tests passed (25 local-evidence tests deselected), three independent old/new numerical comparisons passed, all 124 modules imported, 909 file links resolved, and all 584 frozen original files matched. The pre-existing plotting definitions/tests and interference documentation were preserved. Details are in REFACTOR_VALIDATION.md.

### Task 1: move dedicated code and update active references

- [x] Capture the current default test baseline before editing: 528 passed and 6 Windows-sandbox atomic-replace errors; those same 6 tests passed outside the sandbox, for 534 passing default tests total. 25 local-evidence tests were deselected.
- [x] Move dedicated core modules, directional experiment entries, and specialized diagnostic tools; add package initializers. Keep shared RF/DFE/tracking/GPS primitives shared where used across directions. The array pair was initially misclassified; see the follow-up ownership correction above.
- [x] Move dedicated echo/interferer scene helpers and their small result dataclasses out of shared `gnss.py`; move blind-null combining and multipath-specific LCMV combining out of shared `diversity.py`. Preserve function bodies and shared scenario/pipeline orchestration; do not add old-path re-export wrappers.
- [x] Update absolute/relative imports, dynamic imports, script bootstraps, test paths, monkeypatches, and provenance to recursively cover the maintained nested code. No old import compatibility shims.
- [x] Classify new outputs and known existing result trees by direction; update current consumers. Move existing results without editing frozen contents; leave unclassified and general positioning inputs in place.
- [x] Organize current and previously temporary research Markdown under matching `docs/` directories, repair active links and commands. Keep historical commands/source maps explicitly historical.
- [x] Update README, local project instructions and existing refactor documentation to describe the final paths.
- [x] Run targeted layout/import/path tests, then the default suite; run practical smoke checks in temporary outputs without overwriting saved research evidence.

### Task 2: independent review and final verification

- [x] Reviewer checks import boundaries, output reset paths, moved report links, current/legacy provenance separation, and source freeze completeness.
- [x] Correct substantive findings and rerun affected checks; Main runs final verification and reports any evidence-only limitations.

**Review focus:** nested source files omitted from freeze; wrong `__file__` parent after a move; reset deleting a direction directory instead of one experiment; historical manifests silently rewritten; replay input moved without consumer updates; existing source-byte gates accidentally weakened; default tests depending on ignored local evidence.

**Progress:** completed without staging, committing, or pushing. Initial workspace clean at `326d3f7`; two read-only audits and independent review completed. Main took over after the implementation agent became unavailable, corrected the review findings, and verified 539 default tests plus three independent old/new numerical comparisons. All 77 existing result trees retain their metadata fingerprints and selected root summary/manifest/plan/snapshot byte hashes; 37 trees moved. All maintained modules import and checked document links resolve. Deleted validation evidence is described explicitly as historical, with no substitute results. Exact commands and verification limits are in REFACTOR_VALIDATION.md. The earlier sections below describe the previous completed refactor and remain historical.

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
