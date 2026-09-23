---
type: preregistration
status: locked-not-executed
date: 2026-09-23
mechanism: pmem-live-cliff-detector
lane: load-bearing
spec: research/runners/_pmem_live_cliff_detector_derisk.py
supersedes: research/findings/2026-09-23-pmem-live-cliff-detector-PREREGISTRATION.md
promotion_value: none
---

# Prospective-memory live cliff detector, v2: design locked before any v2 measurement

**Status: locked, nothing under this design has been measured.** This document freezes the v2 read-out, the
seed roles, the detector procedure, the null and the gate BEFORE any calibration scan, held-out evaluation or
organ compare runs. It supersedes the v1 pre-registration, which was executed and returned UNDEFINED.

## AMENDMENT LOG (what I had seen before writing this)

1. **v1 pre-registration** (`2026-09-23-pmem-live-cliff-detector-PREREGISTRATION.md`, commit a7b34f92c) and the
   v1 build (commit f30ab4666), including the post-lock edit to v1's decline rule that had no amendment log.
2. **The full v1 6-seed run on the canonical seeds** (42/43/44/100/101/102), every trajectory, committed verbatim as
   `research/findings/raw/_pmem_live_cliff_detector.json` (run_id 1790180553-1157771). Its own verdict:
   UNDEFINED, two-init agreement on 4 of 6 seeds, and the discrimination anti-cheat FAILED because the detector
   fired on seed 101 as well as seed 44.
3. **The adversarial review** of v1: the read-out was an arg-max, the constants were fit on the evaluation seeds,
   there was no held-out set and no null, and byte-identity was inferred rather than asserted.
4. The parent artifacts already on main: the static stabilizer grid (`_pmem_operating_point_stabilizer.json`)
   and the live-homeostat run (`_pmem_live_homeostat.json`).

**Consequence: the canonical six are IN-SAMPLE for v2.** Everything about this design was shaped while their
data were visible, so passing on them is necessary but carries no weight as evidence. Primary evidence comes
only from seeds nobody has measured.

**Nothing about seeds 7-12 or 200-205 has been measured or looked at.** None of them appears in any prospective-memory artifact.
**The v1 trajectories are NOT re-scored under the v2 rule.** They were produced by a different closed-loop law.
The trajectory a v2 controller would visit is different, so re-scoring them would be evidence about neither design.

## What changes, issue by issue

| review issue | v1 | v2 |
|---|---|---|
| circular read-out | reported the running best `(g_best, rel_best)` = arg-max of the evaluation metric; gated "meets-or-beats" the static arg-max table | reports the controller's SETTLED state: the last iterate and the `rel` measured there. The static comparison is kept as a descriptive number and does not gate |
| constants fit in-sample | DROP_THRESH / DECLINE_PATIENCE sized from seed 44's dip and cliff and seed 100's decline | a one-sided CUSUM (Page 1954). Its only data-dependent quantity, the jitter scale sigma, is estimated from calibration seeds 7-12. The multipliers are the conventional control-chart defaults |
| no held-out seeds | evaluated on the canonical six only | primary evaluation on held-out seeds 200-205; the canonical six are reported as in-sample |
| no null | none | the same detector on within-seed shuffles of each seed's open-loop scan, with the seed as the unit |
| byte-identical inferred | "returns False unset" read from the code | the production organ is built with the flag unset, in this tree and in a `git archive` of a pinned pre-change SHA, then exact-compared, with a negative control that must differ |

## Seed roles (frozen)

- **Calibration seeds `SEEDS_CALIB` = 7, 8, 9, 10, 11, 12.** Used only to estimate sigma, and never evaluated.
- **Held-out evaluation seeds `SEEDS_HELDOUT` = 200, 201, 202, 203, 204, 205.** The primary evidence.
- **Canonical seeds 42, 43, 44, 100, 101, 102.** In-sample: they are required to pass, but they are not evidence.

## The detector procedure (frozen; the numbers come out of the calibration seeds and nothing else)

1. For each calibration seed, measure `rel` on the open-loop lattice `fac_g` = 6000, 6500, ..., 11000 (11 points,
   the controller's own rate-limited step). Each measurement uses the parent's fresh-build `_measure_rel`.
2. Pool the one-step differences of all calibration scans. Then `sigma = 1.4826 x MAD(diffs)`. <!--derived--> The MAD is used
   because a few genuine cliffs cannot inflate it.
3. CUSUM slack **k = 0.5** x sigma and decision interval **h = 5.0** x sigma. These are the conventional one-sided
   CUSUM design values, fixed here before sigma is known and never tuned afterwards.
4. `--freeze` writes `frozen_constants.json` (not yet created) in the v2 raw directory, with the sha256
   of every calibration file. That file is committed **in its own commit before any evaluation job is queued**.
   Every evaluation file records the sha256 of the frozen constants it ran under. The aggregate refuses (UNDEFINED)
   if any evaluation file was produced under different constants.

The plant is deterministic: a repeated read at one gain gives the same `rel`. "Jitter" here therefore means how
irregular the response is from one gain to the next, not measurement noise. That is the variation a detector on
this plant has to tolerate.

## The controller (the law is identical for every seed)

`g_{i+1} = clip(g_i + clip(K_I * (REL_TARGET - rel_i), -MAX_STEP, MAX_STEP), G_FLOOR, ceiling_i)`. The constants
K_I = 20000, MAX_STEP = 500, G_FLOOR = 6000, G_TOL = 25, CONVERGE_STREAK = 3 and SETPOINT_TOL = 750 are the
parent live homeostat's, reused unchanged. MAX_ITERS = 24. Inits are 6000 and 8000.

- The CUSUM is updated on climbing moves only: `S_i = max(0, S_{i-1} + (rel_{i-1} - rel_i) - k)`.
- On `S_i > h` (an ALARM), the ceiling drops to the last gain at which S was 0 (Page's change-point estimate) and S
  is reset. The same integral law then continues under the lower ceiling.
- The loop has converged once `|g_{i+1} - g_i| < G_TOL` has held for 3 consecutive iterations.
- **The reported quantity is `(g_last, rel measured at g_last)`**: the settled state a running homeostat would
  actually sit at.

`G_CEILING_CAP = 11000` is DECLARED. It is inherited as the top of the stabilizer's grid, and it happens to equal
seed 101's static pick, so it is in-sample-adjacent. The held-out seeds judge it.

## The null (frozen)

Each seed's open-loop lattice scan is one sample, and seeds are independent substrates. The steps within one scan
are serially dependent, so they are never treated as exchangeable. The scoring:

- For each scan, the detector's first alarm scores **+1** if no later lattice point recovers to the onset level,
  and **-1** if one does. No alarm, or an alarm at the last point, scores 0. The statistic `T` is the sum over seeds.
- The null applies the same detector to 1000 within-seed shuffles of the same scans (rng seed 20260923).
- `p = (1 + #{T_null >= T_real}) / 1001`.
- **Primary: the held-out scans.** The canonical-six scans are reported descriptively.
- If the detector raises no alarm on any held-out scan, the null is **UNDEFINED**. That is not a pass.

## Default-off byte-identity (frozen method)

**Pinned pre-change SHA: `4c141b8e8413e39913b2ef72b13f18c3db656dc0`.** It is origin/main when this branch was
merged up for the fix round, and its organ file has no cliff-detector branch. Method:

- `--default-off-compare` extracts that SHA with `git archive`.
- It runs one scripted production-organ session (form an intention, two distractor turns, a cue turn) at seed 44
  in a fresh process for each tree, with every `BRAIN_PMEM_*` variable stripped.
- It exact-compares the serialized outputs for two configs: the shipped default and `BRAIN_PMEM_FACILITATION=1`.
  The second config goes through the elif chain this build edited.
- **Negative control:** this tree with `BRAIN_PMEM_OP_STABILIZER=1` must differ from the pinned facilitation run.
  If it does not, the compare cannot fail, and the result is UNDEFINED.

## The GATE (frozen)

**Per seed, every criterion must hold.**

- Converged from both inits.
- The two inits settle within SETPOINT_TOL of each other.
- The settled margin (intact N=3 `rel` at the settled gain minus FIRE_THR) is strictly positive.
- The settled margin is at least the shipped-constant margin (the same read at `fac_g = 6000`).
- The seed is load-bearing at the settled gain.
- The frozen N=5 silence clauses all hold.
- Every probe stays inside [6000, 11000].
- A re-run of the low-init trajectory is identical, including its alarms.

**GO iff all of the following hold:**

- All 6 held-out seeds pass.
- At least one held-out seed is a climber. If none is, the controller was never exercised, and the result is
  UNDEFINED.
- All 6 canonical seeds pass. This is necessary, but it is in-sample.
- The held-out null is defined and has p < 0.05.
- The default-off compare shows `exact_equal_all` in the data, and its negative control differs.

Missing inputs, stale constants or an undefined null give **UNDEFINED, never a pass**. The static-table margin
comparison on the canonical six is printed as descriptive only.

## Host-shortcut declaration

The controller and the CUSUM detector are **host arithmetic** that reads the task's own coincidence read-out and
sets a scalar synaptic gain. That is a host set-point controller, and nothing in this run is credited to the brain.
Slow homeostatic synaptic scaling (Turrigiano 2011) and a sliding modification threshold (Lee & Kirkwood 2019,
DOI:10.3389/fncel.2019.00520) are ANALOGIES <!--derived--> for the target, not what this code implements. Those processes are
per-neuron and driven by the neuron's own activity. They are not driven by a task-level output metric.

## If this does not pass

A NO-GO on held-out seeds means that set-point regulation on the task read-out, with a change-point ceiling, does
not generalize. The next method then moves the regulator onto the substrate: a per-neuron activity-driven scaling
of the facilitation gain, read from the maintained assembly's own firing rate rather than from `rel`. That method
removes the host controller instead of retuning it.

## Job order (each step committed before the next)

1. This pre-registration, in its own commit.
2. Calibration scans (`--calibrate --seed s`, s = 7..12), run as pool jobs.
3. `--freeze`, and the frozen constants committed in their own commit.
4. Evaluation (`--eval --seed s`, held-out and canonical), run as pool jobs; `--default-off-compare`, run locally
   under a memory cap.
5. `--aggregate` writes `verdict.json` (not yet created) in the v2 raw directory `_pmem_live_cliff_detector_v2`.
