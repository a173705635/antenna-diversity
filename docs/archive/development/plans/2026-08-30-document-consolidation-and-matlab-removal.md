# MATLAB Removal and Documentation Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the obsolete MATLAB implementation and consolidate the repository's living Markdown documentation around the Python project.

**Architecture:** Python remains the only active implementation. A new root README routes readers to status, usage, design, architecture, and theory; detailed documents retain separate responsibilities. MATLAB/E5 assets remain recoverable from Git commit `6f5e686`.

**Tech Stack:** Git, Markdown, Python 3.11+, pytest

**Spec:** `docs/superpowers/specs/2026-08-30-document-consolidation-and-matlab-removal-design.md`

## Global Constraints

- Do not modify Python algorithms, configuration, tests, or generated results.
- Do not add or remove the untracked Word file, `.idea/`, `requirements.txt`, or `CODEX_MEMORY_GNSS_ANTENNA_DIVERSITY.md`.
- Delete only the resolved workspace path `F:/antenna_diversity/matlab_sim` and the six explicitly named historical Markdown files.
- Preserve the recovery reference to Git commit `6f5e686`.

---

### Task 1: Remove obsolete MATLAB assets

**Files:**
- Delete: `matlab_sim/`
- Delete: `docs/superpowers/specs/2026-08-04-matlab-rf-dfe-trk-chain-design.md`
- Delete: `docs/superpowers/plans/2026-08-04-matlab-rf-dfe-trk-chain.md`
- Delete: `docs/superpowers/specs/2026-08-08-galileo-e5-fullband-joint-tracking-design.md`
- Delete: `docs/superpowers/plans/2026-08-08-galileo-e5-fullband-joint-tracking.md`
- Delete: `docs/superpowers/specs/2026-08-09-acquisition-search-visualization-design.md`
- Delete: `docs/superpowers/plans/2026-08-09-acquisition-search-visualization.md`

**Interfaces:**
- Consumes: Git-tracked legacy assets at commit `6f5e686`.
- Produces: A Python-only working tree while retaining recovery through Git history.

- [x] **Step 1: Verify exact deletion targets**

Run:

```powershell
Resolve-Path -LiteralPath F:\antenna_diversity\matlab_sim
git ls-files -- matlab_sim docs/superpowers
```

Expected: the resolved directory is exactly under `F:\antenna_diversity`, and all seven target groups are tracked.

- [x] **Step 2: Delete the approved tracked assets**

Run `git rm -r` with the exact paths listed in this task. Do not use wildcards or environment-variable-derived paths.

- [x] **Step 3: Verify deletion scope**

Run:

```powershell
Test-Path -LiteralPath F:\antenna_diversity\matlab_sim
git diff --name-status -- matlab_sim docs/superpowers
```

Expected: `False`; the diff lists only the MATLAB tree and the six named Markdown files, plus the already committed design and current plan.

### Task 2: Establish the living-document entry points

**Files:**
- Create: `README.md`
- Modify: `PROJECT_STATUS.md`
- Modify: `python_sim/ARCHITECTURE.md`
- Modify: `python_sim/DESIGN.md`
- Delete: `python_sim/docs/diagram-style.md`

**Interfaces:**
- Consumes: current Blind Null V1 state documented in `PROJECT_STATUS.md` and the three Python documents.
- Produces: one repository landing page and non-overlapping status, usage, design, architecture, and theory entry points.

- [x] **Step 1: Add the root landing page**

Create `README.md` with the project purpose, current Python-only status, a concise RF→DFE→spatial processing→ACQ→TRK flow, links to the five canonical documents, and the legacy recovery commit.

- [x] **Step 2: Update the project status boundary**

In `PROJECT_STATUS.md`, set the update date to `2026-08-30`, remove the root-memory file from key current entries, and replace the MATLAB-present wording with the Git-history recovery statement.

- [x] **Step 3: Merge the diagram style note**

Append a concise `制图与维护约定` section to `python_sim/ARCHITECTURE.md`, update `python_sim/DESIGN.md` to link to that section, then delete `python_sim/docs/diagram-style.md`.

- [x] **Step 4: Check Markdown references**

Run:

```powershell
rg -n "matlab_sim|diagram-style|2026-08-04-matlab|2026-08-08-galileo|2026-08-09-acquisition" -g "*.md" -g "!.worktrees/**"
```

Expected: only intentional Git-history discussion in the new spec/plan or root README remains; no live link points to a deleted path.

- [x] **Step 5: Validate repository links and scope**

Check every relative Markdown link in `README.md`, `PROJECT_STATUS.md`, `python_sim/README.md`, `python_sim/DESIGN.md`, and `python_sim/ARCHITECTURE.md`. Run `git diff --check` and inspect `git diff --stat` plus `git status --short`.

- [x] **Step 6: Run the Python regression suite**

Run:

```powershell
cd F:\antenna_diversity\python_sim
$env:MPLBACKEND='Agg'
python -W error -m pytest -q
```

Expected: exit code 0 with no failed tests.

- [x] **Step 7: Commit the approved cleanup**

Stage only the deletions, `README.md`, `PROJECT_STATUS.md`, `python_sim/ARCHITECTURE.md`, `python_sim/DESIGN.md`, and this plan. Commit with:

```text
chore: remove matlab simulation and consolidate docs
```
