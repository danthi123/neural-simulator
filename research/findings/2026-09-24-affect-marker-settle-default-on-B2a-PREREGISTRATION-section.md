---
type: finding
status: live
date: 2026-09-24
lane: load-bearing-fraction
mechanism: PRE-REGISTRATION of a SETTLE-specific section for the midnight B2a production-default battery. Prepares (does NOT enable) flipping BRAIN_AFFECT_MARKER_SETTLE default-ON on a PARKED branch, research/settle-default-on-prep, based on bd391aa31. No 6-seed shard has run under this document.
seeds: [42, 43, 44, 100, 101, 102]
verdict: PRE-REGISTRATION only. Filed in its own commit before any shard or capability-gate run it governs. No 6-seed result is claimed here. The dev-seed-7 mechanism check referenced below is NOT a result this document governs (see "What this document does NOT cover").
runner: research/runners/load_bearing_fraction.py (guarded by tools/assert_flipped_defaults.py, THIS branch's copy only -- see below)
artifacts:
  - research/findings/raw/_affect_marker_settle_default_flip/s7_equivalence.json
---

# Affect-marker SETTLE default-ON: B2a PREREGISTRATION section (parked branch, S09/AG-FLIP) (2026-09-24)

**Filed in its own commit, before any 6-seed shard governed by the hypothesis below exists.** This document is the
SETTLE-specific section of the midnight plan's B2a prereg (step S09 of the night's plan, held as an
orchestrator-side coordination file, not a versioned repo artifact); it is authored standalone because no combined
B2a document exists yet on `main` to append to. AG-REG/the orchestrator fold this section in by reference if/when
B2a is assembled.

## Status: PARKED, not shipped, not run

Per the night's plan step S00(a), the owner was asked whether to flip `BRAIN_AFFECT_MARKER_SETTLE`
default-ON as a new production default. **As of this commit, `GAP_CLOSURE_MISSION.md` CURRENT STATE still records
no owner answer**, and the plan's own default therefore applies: SETTLE ships OFF, measured only as an opt-in row
in `b2b-caps`. Nothing in this document, and nothing on the branch it describes, changes `main`'s shipped default.
The branch `research/settle-default-on-prep` (based on `bd391aa31`) is a PREPARATION only -- it lands in `M1`/`F`
only if a human merges it after an explicit yes, per the S09 fallback rule.

**⛔ MERGING THIS BRANCH IS NOT A NO-OP FOR PRODUCTION (AGFLIP review framing fix, same day).** This branch's own
review returned `"safe_to_merge": true`, and that field means only "this branch is internally correct and does not
break anything if merged" -- it does NOT mean "merging is inert." `settle_enabled()`'s only call site
(`webapp/affect_drives_chat.py`'s `get_reader(seed=seed)`) is production code: merging this branch to `main` would
flip `BRAIN_AFFECT_MARKER_SETTLE`'s shipped default to ON **immediately**, in the live chat pipeline, not merely
"prepare" it. This is a genuine default-ON flip staged on a branch, not the "default-off additive" shape most
branches in this repo have. Do not merge until ALL of: (1) the owner's S00(a) fork answer is recorded YES in
`GAP_CLOSURE_MISSION.md`, (2) a real 6-seed G-SETTLE1+G-SETTLE2 capability gate reads GO (not the dev-seed-7 module
check above, which is a mechanism sanity check, not a capability gate), (3) a SOUND review of that gate, and (4)
per the owner's explicit instruction carried into this review-fix session: **`research/settle-multiturn-contrast`
(a separate midnight-plan lane, live at this writing) must pass first** -- SETTLE ships ON only after that
multi-turn contrast lane clears, and until then this branch stays PARKED and CORRECT, not flipped.

## Standing result this section must not contradict or re-litigate

`2026-09-23-affect-marker-settle-fullbrain-contrast-PARTIAL-6seed.md`, verdict **NO-GO** (for the flag as
*generally necessary*): with `BRAIN_AFFECT_MARKER_SETTLE=1`, `affect-marker-spiking-wta` is lesion-verified
load-bearing on **6/6** full-brain seeds; with it off, **4/6** (s42, s43 not load-bearing). SETTLE is credited on
only **2/6** seeds (42, 43) -- the flag rescues two seeds and is inert on the other four, so it is not "generally
necessary," which is why the prior gate's own pre-registered contrast criterion read NO-GO and the flip stayed
owner-reserved. **This document does not claim that verdict is reversed.** It registers a DIFFERENT, narrower
hypothesis for a possible future default-ON gate, honestly distinguished from the retracted-style "SETTLE is
necessary" framing.

## New hypothesis (as instructed by the S09 plan step, verbatim)

> With SETTLE default-ON at production defaults, affect-marker-spiking-wta is 6/6 and no row falls below its
> flipdefaults-adequate count.

Made concrete against the standing adequate-probe reference
(`2026-09-23-allfixes-adequate-battery-6seed-robust-core-24.md`, tag `allfixes2`, run WITHOUT SETTLE):

- **G-SETTLE1 (the row this flag targets):** `affect-marker-spiking-wta` lesion-verified load-bearing 6/6 seeds
  (42, 43, 44, 100, 101, 102) under the adequate probe, at production defaults, WITH SETTLE default-ON and no other
  env override. (Baseline without SETTLE: 4/6, per `allfixes2`.)
- **G-SETTLE2 (no-regression floor):** every one of the other 25 `allfixes2` rows scores AT LEAST its `allfixes2`
  load-bearing count. 24 of 26 exercised rows were robust-core 6/6; the 2 that were not are
  `affect-marker-spiking-wta` itself (4/6, targeted above) and `da-gated-encoding` (0/6 in that battery, because
  today's merged v3 DA mechanism was not enabled there -- not this section's concern, but it must not be made
  WORSE by SETTLE being on).
- **UNDEFINED is never scored as 0** (`tools.lab.undefined_if_empty`): a row with zero exercised trials under
  SETTLE-on is UNDEFINED for this gate, not a failing 0/6.
- A GO on G-SETTLE1+G-SETTLE2 is necessary but not sufficient to flip the default; the S07 conditional-flag rule
  ("Each ends default-ON only if...") additionally requires a SOUND review and the owner's fork answer, both
  still open.

## Rows this section adds (thin probe, SETTLE's own code path)

Per the S09 instruction "it adds thin rows on SETTLE's code path": the row is `affect-marker-spiking-wta`
(existing `FACULTY_LESIONS`/`FACULTY_PROBES` entry in `research/runners/load_bearing_fraction.py`; no new row
literal is added there, per the AG-REG ownership boundary in the midnight plan -- this section runs the SAME row
under a different environment, not a new lesion/probe). The thin-probe count is exercised BESIDE the adequate
count above, matching the `flipdefaults-thin` / `flipdefaults-adequate` pairing already used for the three landed
fixes (`2026-09-23-flip-validated-fixes-production-default-validation-prereg.md`).

## The guard this section's future shards would use

`tools/assert_flipped_defaults.py`'s `FLIPPED` registry gained a fourth entry on `research/settle-default-on-prep`
ONLY: `"BRAIN_AFFECT_MARKER_SETTLE": ("research.runners._affect_marker_wta_derisk", "_SETTLE_DEFAULT_ON")`. This
entry does **not** exist on `main`/`bd391aa31` and is not proposed to land there by this document; it exists so
that a guard run AT a revision with this branch merged in refuses (a) any BRAIN_* override in the job environment,
including a leftover `BRAIN_AFFECT_MARKER_SETTLE=`, and (b) a revision that predates the flip (the constant absent
from source), exactly the "guard catches a pre-flip revision" property the three landed flags already have.
Verified directly (seed-7 dev check, this commit's companion): `problems({})` is empty on this branch;
`problems({"BRAIN_AFFECT_MARKER_SETTLE": "1"})` names the override; a `_source_constant` read against `main`
(which lacks `_SETTLE_DEFAULT_ON`) returns `_MISSING`, the same path already exercised by the 3-flag registry.

## The ready-to-run shard command, corrected (AGFLIP review fix, same day)

**No 6-seed shard is staged by this branch or this document** (S09's own text does not ask AG-FLIP to stage
`JOBS.txt` lines here, unlike S04, and none of the S07 conditional-flag preconditions -- owner yes, a real 6-seed
capability gate, a SOUND review -- exist yet). The build step's own report (workflow journal, not a repo file)
did however draft a "ready-to-run shard command" for whoever later stages it, and that draft had two defects
caught by review, both fixed HERE (not merely in a report) so the correct text is what survives:

1. `--faculty affect-marker-spiking-wta` is not a valid flag on `research/runners/load_bearing_fraction.py`
   (confirmed by `grep add_argument` in that file: no `--faculty`; the real flag is `--only`, a comma-separated
   faculty-key restrictor). `--faculty` belongs to the unrelated `research/runners/first_chat_console.py` --
   a copy-paste mix-up.
2. The draft also omitted the `LB_*_DRIVE_PROBE=1` environment variables that the ADEQUATE probe (the one
   G-SETTLE1/G-SETTLE2 above require) sets, and that every real line in
   `research/findings/raw/_load_bearing/_shards/flipdefaults-adequate/JOBS.txt` sets. Omitting them would not
   crash -- it would silently run the THIN probe instead, a wrong-but-quiet result.

The corrected command, verified against the real adequate-probe convention (diffed line-for-line against
`flipdefaults-adequate/JOBS.txt`'s own `affect-marker-spiking-wta` rows, seed substituted only):

```
SEED=<42|43|44|100|101|102>
OUT_DIR="research/findings/raw/_load_bearing/_shards/settle_default_on/s${SEED}/affect-marker-spiking-wta"
.venv/bin/python tools/assert_flipped_defaults.py && \
mkdir -p "$OUT_DIR" && \
env SIM_BACKEND=numpy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    LB_AFFECT_DRIVE_PROBE=1 LB_BG_SELECT_DRIVE_PROBE=1 LB_CG_DRIVE_PROBE=1 LB_DISCOURSE_REGISTER_DRIVE_PROBE=1 \
    LB_EPISODIC_DRIVE_PROBE=1 LB_NONCONTRADICTION_DRIVE_PROBE=1 LB_OPEN_ENDED_DISTRIB_PROBE=1 \
    LB_PMEM_DRIVE_PROBE=1 LB_SURPRISE_CONFIRM_PROBE=1 \
    .venv/bin/python -u -m research.runners.load_bearing_fraction --only affect-marker-spiking-wta \
    --seed "$SEED" --repeats 2 \
    --out "$OUT_DIR"/lb.json
```

run from a `pool_provision`'d checkout of this branch merged onto whatever `F` is at staging time. `mem_gb` for
this shard is still **TBD, not measured this session** (only the ~0.33 GB WTA-module-only equivalence check was
measured; a full-brain LBF shard is a materially heavier build and needs its own RSS probe before a real job is
queued) -- declared, not invented, per `tools.lab.undefined_if_empty`.

## The guard this section's future shards would use

`research/runners/_affect_marker_wta_derisk.py`, lines 171/178 (per the plan step's own citation) --
`settle_enabled()` changed from "env unset -> OFF" to "env unset -> `_SETTLE_DEFAULT_ON` (= `True` on this
branch); env set (any value) -> parsed as before." `BRAIN_AFFECT_MARKER_SETTLE=0` still forces the pre-existing
60&nbsp;ms/40&nbsp;ms read. `BRAIN_PMEM_OP_STABILIZER` is explicitly **NOT** touched by this branch (S06c/S09(4)):
it stays a seed-keyed lookup table, circular for a default-flip, and out of scope here.

## What this document does NOT cover (declared residual)

A full pinned multi-turn `/api/brain-chat` transcript equivalence check (the kind run for the three landed fixes
in `2026-09-23-flip-validated-fixes-production-default-validation-prereg.md`'s "flag-OFF chat identity check") was
**not** run for this branch. Two reasons, both declared rather than assumed: (1) no reusable pinned-transcript
tool exists in this repo as of this commit -- each prior lane's version was an ad hoc script whose output artifact
was committed but whose driver script was not; (2) `pool2` (named in the S09 plan step for this exact check) is
not reachable from this session (`ssh pool2`: name not resolved; no `research/queue/.aws_pool2` file), and the
local box's available RAM was measured tight (3-9&nbsp;GB free) for the whole session, so a full tiny-demo brain
build was judged disproportionate given `BRAIN_AFFECT_MARKER_SETTLE` has exactly ONE call site
(`webapp/affect_drives_chat.py`'s `get_reader(seed=seed)` -> `research.runners._affect_marker_wta_derisk`,
grep-verified, no other reader of `SETTLE_ENV`/`settle_enabled`/`_SETTLE_DEFAULT_ON` in the tree). What WAS run
instead, committed in the NEXT commit as
`research/findings/raw/_affect_marker_settle_default_flip/s7_equivalence.json` (per `gates/prereg_before_run`,
which is why it is not staged alongside this document): a seed-7 equivalence check at the module's own documented
API
(`select_valence`/`select_arousal`, the only surface SETTLE reaches), in three fresh subprocesses on this
revision -- env unset (new default), `=1` (explicit), `=0` (explicit) -- each over the mood/arousal sweep from
the module's own `__main__` smoke (including the +0.069 boundary case the standing NO-GO finding's op-level
diagnosis names). Result: `hash(unset) == hash(explicit "1")` (both `08aed7b1...`), and
`hash(explicit "0") != hash(explicit "1")` (confirms the OFF path is untouched, not merely unchanged by omission).
`prereg-same-commit: N/A -- the mechanism check's artifact commits AFTER this document, per gates/prereg_before_run`

## Honesty

Functional read-out only. This document registers a HYPOTHESIS about a lesion-verified load-bearing COUNT, not a
felt/phenomenal claim; nothing here asserts the flip is a good idea, only what a future gate would need to show
for it to become one. The S1 (host `np.argsort`-over-rates + `DEAD_MARGIN` readout) and S2 (host Gaussian-tuned
drive current) shortcuts named in the module's own docstring are unchanged by this preparation and remain named
residuals, not closed by this branch.
