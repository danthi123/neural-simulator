---
name: sync-documentation
description: Use when code OR findings change may have made docs stale — and run it SAME-CYCLE when a committed finding changes a wall/gap STATUS, the CURRENT FRONTIER, or a next-action. Two layers. (1) MECHANICAL drift — line counts, class line numbers, runner/test/findings counts, g11 flags, sim/__init__ exports — auto-fixed. (2) SEMANTIC summary-doc sync — the roadmap wall-ledger + GAP_CLOSURE_MISSION CURRENT STATE + AUTONOMOUS_STATE + ROADMAP.md must reflect the LATEST findings/git (status/frontier/next-action synced, contradictions between docs reconciled, abandoned docs banner-ed, a plain-language header + project-shorthand glossary kept). Layer (2) is what drifted on 2026-07-24 (findings committed, board left stale) and a mechanical pass alone cannot catch it.
---

# Sync Documentation

This skill verifies the project's documentation against the actual code state and surfaces drift. It runs a fixed set of checks, makes safe automatic updates (numerical drift), and reports semantic changes that need human review (new best recipe, new flag added, new milestone).

**Announce at start:** "Using sync-documentation to verify docs against code."

## What this skill does

The skill walks through the check categories below. **Checks A–H are MECHANICAL** (numerical / line-number drift — auto-fixed). **Check I is the SEMANTIC summary-doc sync — the layer that actually drifts** and that a mechanical pass cannot catch (the 2026-07-24 failure). For each, it reads the actual state, compares to what the docs claim, and fixes or reports.

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

### Check I: SEMANTIC summary-doc sync (the layer that drifts — DO this, don't just flag)

The **findings** are the source of truth; the **summary docs** are pointers that go stale. When a committed finding
changes a wall/gap STATUS, the CURRENT FRONTIER, or a "next action", these must move WITH it. Ground-truth first:

```bash
git log --oneline -40
ls -t research/findings/*.md | head -40
```

Then reconcile each, and **UPDATE them** (this is the same-cycle sync, not a flag-for-human — the invoker has the session context):

1. **Freshness** — `GAP_CLOSURE_MISSION.md` CURRENT STATE, the master roadmap (`docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md`) §8 "next to queue" + §2 faculty tags + §7 walls-ledger, and `ROADMAP.md` status must reflect the latest commits/findings. A "next action / next bet" pointing at *already-done* work is the worst drift (it mis-aims the next session) — fix those first.
2. **Contradictions** — cross-check docs against each other (e.g. a legacy gap-table vs the CURRENT STATE); reconcile every contradictory status to the LATEST verdict, or retire the superseded table.
3. **Abandoned docs** — any summary doc >~7 days stale that is no longer the working board (e.g. `AUTONOMOUS_STATE.md`) gets a one-line **`⚠️ SUPERSEDED → see <live board>`** banner at the top (don't silently leave 9-day-old drift looking live).
4. **Accessibility** — verify: (a) `GAP_CLOSURE_MISSION.md` CURRENT STATE opens with a ≤5-line **STATE OF THE PROJECT** header (date · one-line north-star · current frontier · the single literal next command), everything below it explicitly "history"; (b) `ROADMAP.md` (the plain-language skim surface) is LINKED from the top of CLAUDE.md + GAP_CLOSURE_MISSION as "read this to skim"; (c) a **project-shorthand glossary** exists and covers the coinages (FHRR, BTSP, BDSP, GNW, PPMI, VSA, meta-d′, gap#N, RANK-N, DR-N, EMERGE, "the moat", "the composer", slot-binder) — inherent domain terms (NMDA/STDP/theta-gamma) need no gloss, project shorthand does. Add/fix if missing.
5. **Prune** — CURRENT STATE is append-at-top; move closed entries older than the live resume-anchor to `docs/project-history-archive.md` so stale doesn't sit interleaved with live.

Any genuine judgment-fork (which of two live directions is "the" frontier) is flagged for the human; the status/frontier/next-action *record-keeping* is done here.

### Check J: Document STRUCTURE rules (docs/WRITING.md) — RUN IT, do not eyeball it

```bash
.venv/bin/python tools/check_docs.py
```

Two rules, both mechanical, zero judgment:
- **W1** — a voided doc is registered in `docs/RETRACTED.md`, and no governed file cites it without `⛔` on the
  same line. Fix by adding the registry row, or marking the citation.
- **W2** — prose lines in governed files are <=800 chars (tables/code exempt). Fix with
  `.venv/bin/python tools/split_long_doc_lines.py --apply` (splits at sentence / `·` / `;` boundaries and REFUSES
  to write if content changes).

**Why it belongs in this skill.** W1 is drift #12 — the stale pointer — made checkable. On adoption it found three
live stale citations, one of them on the MASTER ROADMAP presenting a RETRACTED attribution as a current finding.
That is exactly the failure this skill exists to prevent and it had gone unnoticed for days.

**W2 is a PRECONDITION for W1, not a style preference:** a marker cannot sit next to the claim it kills when a
bullet is 14,222 characters long. Proven empirically at adoption — splitting that line exposed two further stale
citations that had been "marked" only by a `⛔` 13,000 characters away.

A PostToolUse hook also runs this automatically whenever a governed file is edited, so Check J is normally already
green by the time this skill runs. If it is not, fix it before doing anything else in this skill — a stale pointer
mis-aims the next session.

**It does NOT check truth.** Six of the nine 2026-07-28 retractions were instrument failures that pass both rules.
Truth is `.claude/skills/verify-go/SKILL.md`; term conditions are `docs/TERMS.md`.

### Check K: USER-FACING doc freshness — README especially (Checks A–J do NOT cover it)

**The gap this closes (owner-flagged 2026-08-19): Check I syncs the INTERNAL boards (roadmap / GAP_CLOSURE /
ROADMAP.md); Checks A–H fix COUNTS in README/CONTRIBUTING. NOTHING checks the README's PROSE — the project
tagline, the honest-status/capabilities table, the "what it can/can't do" narrative — so it drifts silently.** A
workflow refresh on adoption found the README's entire honest-status table stale (the conversational-integration /
language / emotion / memory rows all described pre-pivot state) while every mechanical check was green.

The user-facing docs are `README.md`, `USER_GUIDE.md`, `CONTRIBUTING.md`, `CHANGELOG.md`. Reconcile their PROSE
against ground truth (the same source Check I uses): `docs/PRODUCTION_INTEGRATION_LEDGER.yaml` (what is actually
wired + on-by-default), `ROADMAP.md` recent milestones, the latest findings. Look for: a stale project one-liner
(must be the pivoted north-star, not an old fact-recall/RAG/nav-demo framing); an honest-status/capabilities section
that omits shipped default-on faculties or lists retired scaffolds as current; stale runnable commands/flags in
USER_GUIDE (VERIFY every command against the runner's argparse — a wrong command is the worst user-facing defect);
a CHANGELOG missing recent milestones.

**Two hard rules for any user-facing edit** (they are user-visible, so the bar is higher):
- **Honesty boundary is load-bearing here too:** self-reports are FUNCTIONAL read-outs, never "felt"/"phenomenal"/
  "conscious"/"sentient" claims, and a header/tagline reads STANDALONE (a disclaimer two sentences later does not
  neutralise a "felt mood" heading). Do NOT overclaim: describe on-by-default behaviour as the product, research
  de-risks as research, and keep the honest caveats (co-residency ≠ one substrate; the Qwen mouth is a scaffold;
  scaffold_retired is 0).
- **Trust-but-verify any agent-written user-facing doc** before landing: confirm each runnable command + cited
  finding actually exists.

**For a deep semantic pass, dispatch the `refresh-user-facing-docs` workflow** (ground-truth → per-doc grounded
edit → an adversarial HONESTY-verify pass → you review the diff + flags, fix, gate, land). On adoption it rewrote
the README honestly and its honesty pass caught a "felt mood" header the edit had introduced. This is the mechanism
that stops the README from going stale between passes; run it when the README's prose is >~2 weeks behind the
frontier or the owner asks.

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
