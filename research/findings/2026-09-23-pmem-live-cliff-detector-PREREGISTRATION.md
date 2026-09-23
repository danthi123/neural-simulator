---
type: preregistration
status: locked-not-executed
date: 2026-09-23
mechanism: pmem-live-cliff-detector
lane: load-bearing
spec: research/runners/_pmem_live_cliff_detector_derisk.py
promotion_value: none
---

# Prospective-memory LIVE per-seed cliff detector: design locked before the run

**Status: implementation-ready, no 6-seed run executed yet.** This document freezes the mechanism, the
constants and the GO/NO-GO gate BEFORE `research/runners/_pmem_live_cliff_detector_derisk.py --derisk` is run,
per the task's own instruction ("commit the pre-registration in its own commit BEFORE any run it governs").

## Why (the named next controller, banked by the parent finding)

`research/findings/2026-09-23-live-homeostat-nogo-6seed.md` (merged @67b7fbc9c) is an honest NO-GO 5/6: ONE
global cliff-safety ceiling (`G_CEILING=9000`) forbids the gain seed 101 needs (the static table used `11000`
for that seed specifically). That finding names the exact next mechanism and explicitly declines to build it
itself (banking a miss, not tuning to force a pass): *"a LIVE CLIFF DETECTOR: extend the integral step with a
local-derivative safety check ... a homeostat that finds its OWN safe ceiling live"*.

**The companion-process question first (CLAUDE.md's "deepest lesson").** What does the real system run
alongside a slow integral set-point controller that we replaced with a single constant? Biology's answer is
**metaplasticity**: a companion process that itself tracks recent activity and adjusts *how far* plasticity is
allowed to go, rather than a single fixed bound applied identically to every cell. This is the Bienenstock-
Cooper-Munro (BCM) **sliding threshold** — According to PubMed, Lee & Kirkwood (2019), "Mechanisms of
Homeostatic Synaptic Plasticity," *Front Cell Neurosci* 13:520,
[DOI:10.3389/fncel.2019.00520](https://doi.org/10.3389/fncel.2019.00520) (PMID 31849610): <!--derived-->
*"sliding threshold ... operate[s] by adjusting the threshold for synaptic plasticity"* based on the cell's own recent activity
history, as opposed to synaptic scaling's fixed-gain rescaling. The parent build's `G_CEILING=9000` is a single
constant threshold; a sliding threshold is instead **measured live, per unit, from that unit's own response** —
exactly the generalization this build makes: each seed's ceiling is discovered from that seed's own live
coincidence trajectory, not shipped as one number for every seed. External source recorded:
`research/queue/.external_searches.jsonl` (lane `load-bearing`, this DR round).

## What the parent build's own committed grid already shows (read, not re-derived)

`research/findings/raw/_pmem_operating_point_stabilizer.json`, `per_seed[*].grid_evals` (FAC_G_GRID =
6000..11000 pA, step 1000):

| seed | 6000 | 7000 | 8000 | 9000 | 10000 | 11000 | shape |
|---|---|---|---|---|---|---|---|
| 42  | .3439 | .3439 | .3450 | .3378 | .3433 | .3483 | always above target, mild wobble |
| 43  | .3361 | .3406 | .3406 | .3372 | .3361 | .3356 | always above target |
| 44  | .2111 | .2078 | .2139 | .2183 | **.1506** | .1589 | rises to a peak at 9000, then a HARD CLIFF |
| 100 | .2839 | .2861 | .2933 | .2956 | .2911 | .2894 | rises to a peak at 9000, then a MILD, sustained decline (no cliff) |
| 101 | .2761 | .2861 | .2889 | .2889 | .2906 | .2922 | monotonically rising all the way to 11000 — **no cliff in the validated domain** |
| 102 | .3133 | .3100 | .3111 | .3128 | .3128 | .3117 | always above target, mild wobble |

The parent live-homeostat finding's own committed seed-44 trace (`research/findings/raw/_pmem_live_homeostat.json`,
low-init trajectory, MAX_STEP=500) additionally shows a **known, RECOVERABLE one-step dip**: iter1→iter2
(g 6500→7000) `rel` drops 0.2150→0.2078, a **−0.0072** step, then RECOVERS the very next step (iter3, g=7500,
`rel`=0.2089 > 0.2078) and keeps climbing to the true peak (0.2183 @ 9000). The same build's finer seed-44 pilot
scan (not a committed artifact, reproduced in that finding's "Files" section) shows the TRUE cliff edge is a
**single-step, −0.0233** collapse (9100→9200 pA: `rel` 0.2144→0.1911) that does **not** recover by 10000/11000. <!--derived-->
(quoted from `research/findings/2026-09-23-live-homeostat-nogo-6seed.md` prose — its own pilot scan, not a committed artifact.)

**These two already-committed magnitudes (−0.0072 recoverable dip vs. −0.0233 non-recoverable cliff) <!--derived-->
are the ONLY numbers this pre-registration uses to size the new detector's constants — frozen from PRIOR
evidence, not fit to this build's own outcome.**

## The mechanism (additive; NO sim/ edit; reuse-by-import of the committed live-homeostat + facilitation substrate)

For each seed, from an initial gain `g_0`, run the SAME Turrigiano-style rate-limited integral law as the
parent build (`K_I=20000`, `MAX_STEP=500` pA, reused unchanged — not re-tuned), but replace the parent's single
GLOBAL `G_CEILING=9000` with a **per-seed LIVE ceiling** that starts at a generous, uniform, already-VALIDATED
cap and can only shrink, live, when THIS seed's own measurement says to:

1. `G_CEILING_CAP = 11000.0` pA — the TOP of the parent build's own pre-registered, already-measured grid
   (`FAC_G_GRID`). This is a hard, uniform, global safety bound (never per-seed): the controller structurally
   cannot probe outside the domain the parent build already empirically characterized for every seed (no
   extrapolation into unmeasured territory). Every seed's per-seed LIVE ceiling starts here and is clamped
   DOWN, never up.
2. At each step, MEASURE `rel_i` at the candidate `g_i` (`_measure_rel`, reused by import from
   `_pmem_live_homeostat_derisk` — the identical fresh-build read).
3. Determine ONCE per seed, from the FIRST live read at `g_init`, whether this seed is a **CLIMBER**
   (`REL_TARGET − rel(g_init) > 0`, i.e. below target, needs to increase gain) or a **SETTLED** seed (already
   at/above target). A settled seed uses the PARENT build's identical logic UNCHANGED (descend/settle toward
   `G_FLOOR`, no cliff-detection applies — descending into already-validated low-gain territory carries no
   documented collapse risk). Only CLIMBING seeds run the new cliff-detection logic below (this is a LIVE,
   per-seed branch driven by that seed's OWN first measurement, not a hand-picked seed list).
4. For a CLIMBING seed, track the running best `(g_best, rel_best)` seen so far (initialized at `g_init`).
   At each subsequent climbing step (`g_i > g_{i-1}`):
   - `step_drop = rel_{i-1} − rel_i` (the LOCAL, one-step derivative).
   - **Abrupt-cliff check:** if `step_drop > DROP_THRESH (0.015)` — a single step already worse than the
     midpoint of the parent's own recoverable-dip (0.0072) and non-recoverable-cliff (0.0233) magnitudes <!--derived--> — this
     step is treated as having found the cliff edge immediately (no need to spend a second step confirming a
     collapse this large; the parent build's own pilot showed a comparable single-step collapse never recovers
     by higher gains).
   - **Sustained-decline check:** else, if `rel_i < rel_best` (any decrease from the running best), increment a
     `decline_streak`; if `decline_streak >= DECLINE_PATIENCE (2)` — i.e. TWO consecutive climbing steps that
     fail to beat the running best — this is treated as a genuine, non-recovering decline (the parent's own
     data shows its one known recoverable dip lasted exactly ONE step before recovering past the pre-dip value;
     patience=2 tolerates that dip while still catching a decline that persists).
   - On EITHER trigger: **revert** the candidate `(g, rel)` to `(g_best, rel_best)`, **clamp this seed's live
     ceiling to `g_best`** (below the drop, per the task's own wording), record the event, and declare this
     seed CONVERGED at `g_best` (a pinned, live-detected safety boundary — the direct generalization of the
     parent build's fixed `G_CEILING=9000`, now discovered per seed instead of shipped as one number).
   - If NEITHER triggers and `rel_i > rel_best`: update the running best and continue.
5. **The canonical converged answer for a CLIMBING seed is always `(g_best, rel_best)`** — the best point this
   seed's own trajectory ever measured, whether reached by a cliff-triggered reversion or (for a seed like 101,
   which never declines inside the validated domain) by simply climbing to `G_CEILING_CAP` with the running
   best AT that cap. A SETTLED seed's canonical answer is the parent build's identical converged/pinned value
   (unchanged logic).
6. Repeat until converged (a cliff/decline trigger, OR the parent's ordinary `G_TOL`/`CONVERGE_STREAK` settle
   condition) or `MAX_ITERS (24)` is exhausted (doubled from the parent's 16 — the live ceiling now starts
   further from the floor, 11000 vs 9000, so more climbing steps may be needed before either a trigger or a
   natural settle).

## Constants, frozen here, BEFORE the run (uniform across all 6 seeds — no per-seed constant anywhere)

| constant | value | source |
|---|---|---|
| `G_FLOOR` | 6000.0 | reused unchanged (`FAC_G_DEFAULT`) |
| `G_CEILING_CAP` | 11000.0 | the top of the PARENT build's own already-validated grid (never extrapolated beyond) |
| `K_I` | 20000.0 | reused unchanged from the parent live homeostat (not re-tuned) |
| `MAX_STEP` | 500.0 pA | reused unchanged |
| `G_TOL` / `CONVERGE_STREAK` | 25.0 / 3 | reused unchanged |
| `DROP_THRESH` | 0.015 `rel` | midpoint of the parent's own committed dip (0.0072) and cliff (0.0233) magnitudes <!--derived--> |
| `DECLINE_PATIENCE` | 2 consecutive climbing steps | the parent's one known recoverable dip lasted exactly 1 step |
| `MAX_ITERS` | 24 | doubled from the parent's 16 (wider live-ceiling range to cross) |
| `G_INIT_LOW` / `G_INIT_MID` | 6000.0 / 8000.0 | reused unchanged |

## The pre-registered GATE (fixed here, before running)

GO iff, on **all 6 seeds**: (a) the live-converged margin (`converged_rel − FIRE_THR`) is strictly positive;
(b) it meets-or-beats the STATIC table's own committed margin (`_pmem_operating_point_stabilizer.json`, read,
never re-derived); (c) load-bearing is preserved; (d) the frozen N=5 silence gate holds; (e) two different
inits converge to the SAME neighborhood; (f) no probe, on any seed/init, ever visits a `fac_g` outside
`[G_FLOOR, G_CEILING_CAP]` (the hard, uniform, already-validated domain bound — this is this build's own
definition of "never enters the collapse zone," an explicit, pre-registered REDEFINITION from the parent
build's "never visits above the ceiling at all," because this design's entire mechanism is to TRANSIENTLY probe
above a seed's own eventual ceiling in order to find it; the safety invariant that survives is that the
probing stays inside the domain the parent build already fully characterized, and the FINAL selected gain for
every seed is always a directly measured, non-collapsed point by construction — see docs/TERMS.md discipline on
checking a term's CODE CONDITION before use); (g) determinism (an identical re-run reproduces an identical
trajectory, including cliff events); (h) default-off byte-identical (unset env ⇒ the existing code path,
verified by an exact compare against the parent's independently committed grid-point-0 read).

**Falsifiability check pre-registered on the mechanism itself** (not just the outcome): the detector must
actually fire a cliff/decline event on seed 44 (the KNOWN-cliff seed) on BOTH inits, and must NOT fire one on
seed 101 (the KNOWN-no-cliff-in-domain seed) on either init — a detector that fires on everyone, or on no one,
would not be a genuine live detector and the run reports this directly (`prove_it_discriminates`).

## Honest anticipated residual (named before running, not after)

Even this design cannot exceed `G_CEILING_CAP=11000` — if some future seed's true optimum lay beyond the
parent's own validated grid, this build would not find it (by design: never extrapolate beyond measured
domain). That would be a genuinely NEW, differently-shaped residual from this run's own measurement, not a
foreseen failure of the 6 seeds in scope here (whose full behavior up to 11000 is already characterized above).

## Files (to be created by the run this document governs)

- `research/runners/_pmem_live_cliff_detector_derisk.py` (new): the controller, `--selftest` (pure-function,
  no brain build), `--derisk` (6-seed x 2-init), `--smoke` (1 seed).
- `research/runners/prospective_memory_production_organ.py`: a new `pmem_live_cliff_detector_enabled()`
  (env `BRAIN_PMEM_LIVE_CLIFF_DETECTOR`, default-OFF), checked BEFORE the existing live-homeostat / op-stabilizer
  branches in `_ensure_pm` (so this new controller takes priority when its flag is set — byte-identical to
  today's shipped default when it is not).
- Artifact (does not exist yet — will be written by the run this document governs, not cited as evidence here):
  `_pmem_live_cliff_detector.json` under `research/findings/raw/`.
