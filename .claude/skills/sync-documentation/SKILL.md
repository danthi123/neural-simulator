---
name: sync-documentation
description: Use when code changes (especially under sim/, research/runners/, experiment/, or new findings) may have made docs stale. Compares actual file state (line counts, class line numbers, runner flags, file counts, module exports) against claims made in README.md, CLAUDE.md, CHANGELOG.md, CONTRIBUTING.md, QUICKSTART.md, docs/SCIENCE_ROADMAP.md, and research/findings/INDEX.md. Updates numerical drift directly; flags semantic changes (new "best" recipe, new flag added, new milestone) for human review.
---

# Sync Documentation

This skill verifies the project's documentation against the actual code state and surfaces drift. It runs a fixed set of checks, makes safe automatic updates (numerical drift), and reports semantic changes that need human review (new best recipe, new flag added, new milestone).

**Announce at start:** "Using sync-documentation to verify docs against code."

## What this skill does

The skill walks through eight check categories. For each, it reads the actual file state, compares to what the docs claim, and either fixes the drift or reports it.

### Check A: Line counts on sim/*.py

Every `sim/*.py` module is referenced in `CLAUDE.md` with a comment of the form `bridge.py # 5347 lines — ...`.

```bash
wc -l sim/bridge.py sim/config.py sim/connectivity.py sim/enums.py sim/kernels.py sim/profiles.py sim/regions.py sim/neuromodulators.py sim/data_bus.py
```

For each module, compare to `CLAUDE.md` (around line 64-74) and update if drifted by more than ~5 lines. Also verify the aggregate `# 9 modules, ~9.4K lines` line.

### Check B: Class line numbers in CLAUDE.md

CLAUDE.md cites specific line numbers for SimulationBridge and key methods. Verify:

```bash
grep -n "^class SimulationBridge\|^class CoreSimConfig\|^class VisualizationConfig\|^class RuntimeState\|^class GPUConfig" sim/bridge.py sim/config.py
grep -n "    def _run_one_simulation_step\|    def _initialize_simulation_data" sim/bridge.py
```

Compare to CLAUDE.md's "**SimulationBridge** (`sim/bridge.py:170`)" and "Simulation stepping (`_run_one_simulation_step` at line 3655)" lines. Update line numbers if drifted.

### Check C: Test file count

```bash
find tests/ -maxdepth 1 -name "test_*.py" | wc -l
```

Verify against `# 28 test files` claims in `CLAUDE.md`, `CONTRIBUTING.md`, and `README.md`. Update if drifted.

### Check D: Runner count

```bash
grep -l "def main\|^def run" research/runners/*.py | grep -v aggregate | wc -l
```

Verify against `# 12 headless runners` claims in `CLAUDE.md` and `CONTRIBUTING.md`.

### Check E: Findings count

```bash
ls research/findings/*.md | wc -l
```

Verify against `# 60+ files` claims in `CLAUDE.md` and `CONTRIBUTING.md`. Update if rounded number is off by ≥10.

### Check F: g11_bg_runner CLI flags

```bash
grep "add_argument" research/runners/g11_bg_runner.py | sed -E 's/.*add_argument..([^"]+)".*/\1/' | sort
```

Cross-reference with the flag lists in:
- `README.md` (around line 425, "Available capabilities" section)
- `CLAUDE.md` (around line 405, "Current flagship" recipe)

Each flag mentioned in the runner should appear in at least one doc. New flags need human review for inclusion. Removed flags need to be deleted from docs.

### Check G: sim/__init__.py exports vs Programmable API examples

```bash
cat sim/__init__.py
```

Verify the public API import in `README.md` (around line 450) and `CONTRIBUTING.md` (around line 405) matches the actual exports:

```python
from sim import (
    SimulationBridge, CoreSimConfig, VisualizationConfig,
    RuntimeState, GPUConfig, NeuronModel, NeuronType,
)
```

If `sim/__init__.py` has new exports not in the example, suggest adding them. If exports were removed but docs still use them, the example is broken.

### Check H: New milestone findings + INDEX

```bash
ls -t research/findings/*.md | head -5
git log --oneline -10 | grep -iE "milestone|best|🎉"
```

For each recent finding doc:
1. Is it in `research/findings/INDEX.md`'s top table?
2. Is its headline number in `README.md`'s status block?
3. Is its recipe in `CLAUDE.md`'s "Recommended configuration" section?
4. Is there a `CHANGELOG.md` entry for it?

Recent commits with "MILESTONE", "best", or 🎉 emoji are the strongest signal that doc updates are needed.

## What to fix automatically vs flag for human

### Fix automatically (numerical drift, no judgement needed)

- Line counts in CLAUDE.md (Check A)
- Class line numbers in CLAUDE.md (Check B)
- Test file count (Check C)
- Runner count (Check D)
- Findings count rounded numbers (Check E)
- Trivial typos / formatting fixes spotted in passing

### Flag for human review (semantic, needs judgement)

- **New flag added to runner** (Check F): show the human the new flag and ask whether/where to document it
- **New module export** (Check G): show the human and ask if it should be in the public-API example
- **New milestone finding without doc updates** (Check H): show the human the finding's headline number and ask whether to update README/CLAUDE/CHANGELOG
- **"Best recipe" appears to have changed**: any time a new finding shows a better number than the current `README.md` status block, flag it but do NOT update without confirmation
- **Removed flag still mentioned**: if a flag is in docs but not in the runner anymore, ask before deleting

## What this skill MUST NOT do

- Rewrite recommended/flagship configs (only flag for human)
- Delete or archive findings (read-only on findings)
- Change `CHANGELOG.md` entry tone or restructure existing entries
- Touch `USER_GUIDE.md` (mostly stable, GUI-focused — out of scope)
- Touch `LICENSE`, `requirements.txt`, code under `sim/`, `experiment/`, `research/runners/`, `tests/`
- Force-push, amend, or rewrite git history

## Output format

End the skill run with this report:

```
## Doc-sync report — <date>

### ✓ Verified clean
- [list of checks that passed]

### Auto-updated (numerical drift)
- [file:line: old → new value]

### Needs human review
- [item: what changed, where, suggested action]

### Skipped
- [things this skill doesn't touch]
```

Keep the report under ~30 lines. If everything is clean, a one-liner is fine: "All 8 checks pass, no drift found."

## Quick command reference

```bash
# Module line counts
wc -l sim/*.py | sort -k2

# Class/method line numbers
grep -n "^class\|    def _run_one_simulation_step\|    def _initialize_simulation_data" sim/bridge.py sim/config.py

# Counts
find tests/ -maxdepth 1 -name "test_*.py" | wc -l                # tests
grep -l "def main\|^def run" research/runners/*.py | grep -v aggregate | wc -l  # runners
ls research/findings/*.md | wc -l                                # findings
ls simulation_profiles/*.json | wc -l                            # profiles

# Recent milestones
git log --oneline -20 | grep -iE "milestone|best|🎉"

# Runner flags
grep "add_argument" research/runners/g11_bg_runner.py | wc -l    # total flag count
```

## Why this skill exists

Doc drift accumulates silently. A line number was off by ~100 lines in CLAUDE.md, a "41 test files" claim was actually 28, "16 runners" was 12, "28+ findings" was 60+ — all caught only when a full overhaul was run on 2026-04-28. The PostToolUse hook in `.claude/settings.json` nudges this skill when code under `sim/`, `research/runners/`, `experiment/`, or new findings change. The skill's role is to make drift cheap to detect, so the docs stay trustworthy.
