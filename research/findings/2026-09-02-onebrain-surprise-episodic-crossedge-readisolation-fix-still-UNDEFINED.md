---
type: finding
status: live
date: 2026-09-02
mechanism: onebrain-integration-surprise-episodic-crossedge-read-isolation-fix
lane: onebrain-integration
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_onebrain_integration_surprise_episodic_crossedge_readfix_numpy6seed.json
  - research/findings/raw/_onebrain_integration_surprise_episodic_crossedge_6seed.json
runner: research/runners/_onebrain_integration_surprise_episodic_crossedge.py
builds_on:
  - research/findings/2026-08-27-onebrain-surprise-episodic-crossedge-UNDEFINED.md
  - research/findings/2026-09-02-c2-metacog-read-isolation-fix-GO.md
  - research/findings/2026-09-02-read-isolation-audit-C2-bug-class-across-14-runners.md
---

# Read-isolation fix ported to surprise→source_provenance (audit row FW-2): REAL and NECESSARY, but INSUFFICIENT — F2's own lesion-control pass rate improves 1/6→3/6, the crux stays UNDEFINED, not the predicted flip to GO or a clean NO-GO

**One-line:** Porting the C2 read-isolation fix (`_EXTRA_RESET_ARRAYS`: `cp_refractory_timers`,
`cp_prev_firing_states`, `cp_neuron_activity_ema`, `cp_neuron_firing_thresholds`, restored to a true-rest
snapshot on every `_hard_reset`) into this runner's TWO bespoke reset sites demonstrably changes every seed's
F2 measurement (confirming the leak is real, not a false alarm) and **improves** the crux's own internal
validity check (`f2_lesion_removes_shift`, requiring the lesion-ratio test to hold on ALL 6 seeds) from
**1/6 to 3/6 seeds passing** — but the audit's own predicted outcome for this row ("recoverable to GO (or a
clean NO-GO)") does **not** materialize on either branch: `delta_intact` still never clears
`F2_INTACT_FLOOR=0.010` on any seed, the precondition still fails overall (now on 3 seeds instead of 5), and
the verdict is **UNDEFINED again** — same status, same precondition name, materially different numbers. This
is itself the honest finding the task asked for: the read-isolation leak was a genuine, partial contributor to
this crux's failure, but it was not the wall's entire explanation — a separate, real residual remains.

## What was ported (Port A) — the SAME fix, applied to BOTH bespoke resets in this file

`research/runners/_crossedge_surprise_metacog_derisk.py`'s `_EXTRA_RESET_ARRAYS` / `_rest_extra` /
`_hard_reset` pattern was ported verbatim (module-level tuple + a per-instance snapshot taken once at true
rest, restored on every reset) into `_onebrain_integration_surprise_episodic_crossedge.py`, at **two** sites —
this runner has the bespoke reset inlined twice, and both needed the fix for the comparison inside
`_migration_invariant` to stay internally consistent:

1. `SurpriseEpisodicPool.__init__`/`_hard_reset` — the primary reset every F1-F4/train/`amb_read` call uses.
2. `_migration_invariant`'s `read_surprise0` closure — a SEPARATE bridge (`b0`, the plain no-cross-edge pool)
   with its own inlined copy of the same pre-fix reset shape, used only for the migration-invariant's own
   `confirm0`/`contradict0` reads. Left unfixed, it would have compared a now-leak-free `sep`-side lesioned
   read against a still-leaky `b0`-side read — an avoidable, newly-introduced inconsistency; both are fixed
   identically.

This is "Port A" as scoped by the task: reusing the SAME 4-array list the framework's own
`onebrain_merge_framework.MergedPool._PER_NEURON_STATE` already carries (a superset that also includes
membrane/recovery/conductances/firing/external-input, which this runner's `_hard_reset` already handled via
its own `_CONDUCT` tuple) — not wrapping calls in `read_isolation()`/`sequence_isolation()`, which are a
different mechanism (multi-organ co-residence isolation across a persistent shared bridge) this runner's
bespoke-reset design does not use.

## The selftest — a fails-in-failing-direction guard, not just a claim

<!--derived-->

`_selftest_read_isolation()` (new, `--selftest` CLI flag) builds a pool, lesions the cross-edge (zeroes it —
so nothing legitimate could differ between reads), and asserts two back-to-back `amb_read(hold_surprise=False)`
calls are bitwise identical. Verified in both directions during this port (ad-hoc runs, not JSON artifacts —
same convention as the C2 finding's own instrumentation table):

| check | fix DISABLED (restore loop neutered) | fix ENABLED |
|---|---|---|
| two identical consecutive `amb_read(False)` calls, seed 42, lesioned | `AssertionError: READ-ISOLATION REGRESSION` (raised — the assertion actually fires) | `margin_1=-0.0025000000000000022 margin_2=-0.0025000000000000022 diff=0.0 PASS` |

The disabled-fix run was produced by temporarily adding `and False` to the restore loop's guard, running
`--selftest`, confirming the `AssertionError` fires, then reverting (`git diff` against the committed file is
empty after the revert). This is the "shown to fail on a case it should catch" standard this project's own
gate-selftest discipline requires, applied to a runner-level regression guard rather than a commit-time gate.

## numpy 6-seed vary/lesion — BEFORE (banked) vs AFTER (this fix)

`research/findings/raw/_onebrain_integration_surprise_episodic_crossedge_readfix_numpy6seed.json` (AFTER, this
session) vs `research/findings/raw/_onebrain_integration_surprise_episodic_crossedge_6seed.json` (BEFORE,
banked 2026-08-27, `CROSS_EDGE_LR=0.15`/`N_EPISODES=150` unchanged — an apples-to-apples config match).
Both runs are `SIM_BACKEND=numpy`; peak RSS observed during the AFTER run was ~420MB (`ps -o rss`), well under
the 4GB budget.

<!--derived-->

| seed | block | `delta_intact` BEFORE→AFTER | `delta_lesion` BEFORE→AFTER | `frac_attributable` BEFORE→AFTER | lesion-ratio (<0.34) BEFORE→AFTER |
|---|---|---|---|---|---|
| 42  | (1,5) | 0.004336 → 0.005312 | 0.003047 → 0.004062 | 0.297297 → 0.235294 | fail (0.703) → **fail, worse** (0.765) |
| 43  | (6,7) | 0.006719 → 0.006875 | 0.002969 → 0.002187 | 0.558140 → 0.681818 | fail (0.442) → **PASS** (0.318) |
| 44  | (0,4) | 0.008711 → 0.008750 | 0.003047 → 0.003750 | 0.650224 → 0.571429 | fail (0.350) → **fail, worse** (0.429) |
| 100 | (3,0) | 0.003672 → 0.002187 | 0.002383 → 0.000625 | 0.351064 → 0.714286 | fail (0.649) → **PASS** (0.286) |
| 101 | (4,0) | 0.006445 → 0.006250 | 0.001758 → 0.001875 | 0.727273 → 0.700000 | PASS (0.273) → PASS (0.300) |
| 102 | (0,4) | 0.008477 → 0.007500 | 0.003516 → 0.003750 | 0.585253 → 0.500000 | fail (0.415) → **fail, worse** (0.500) |

(BEFORE values re-derived here as `round(x, 6)` of the cited BEFORE artifact's `runs[*].F2.{delta_intact,
delta_lesion, frac_attributable}`; AFTER values likewise from the cited AFTER artifact. The lesion-ratio
column — `delta_lesion / delta_intact` against the runner's own `F2_LESION_RATIO=0.34` — is not a stored JSON
field and is computed here from the two cited columns, hence marked derived.)

**Every other arm is unchanged: F1 6/6, F3 6/6, F4 6/6, emergence 6/6, lesion-recovers-migration 6/6, on both
BEFORE and AFTER** (`payload.per_arm`, both artifacts). Only F2 — the crux this fix targets — moves.

**Overall verdict, AFTER:** `GO: false`, `n_go: 0/6` (`payload.GO`, `payload.n_go`,
`research/findings/raw/_onebrain_integration_surprise_episodic_crossedge_readfix_numpy6seed.json`). The
`preconditions` block carries exactly one failed entry, `f2_lesion_removes_shift`
(`ok: false`, `"the F2 shift must VANISH under lesion or it is a confound, not the cross-edge (the crux
control)"`) — the same precondition, same name, that failed BEFORE (there it failed alongside a second,
smoke-test-only artifact `anti_cheat_random_assignment` that is not a real 6-seed finding). Per
`tools/verdict.Verdict`, one failed precondition forces `UNDEFINED`, exactly as it did before the fix.

## Reading the table honestly: the leak is real, the correction is not uniform, and the wall was partly real

<!--derived-->

The fix changes **every seed's** `delta_intact` AND `delta_lesion` (not just the lesion arm) — direct
confirmation that the leaked refractory/homeostatic state contaminates the intact-condition reads too, not
only the lesioned control, since both conditions run through the same `_hard_reset` between every one of the
4 `amb_read` blocks `_f2()` calls (`base_i`, `held_i`, then an in-place lesion, then `base_l`, `held_l`). The
magnitude of each seed's change (`|BEFORE − AFTER|`, roughly 0.0001–0.0018 across the six deltas above) is the
same order as `delta_lesion` itself (0.0006–0.0041) — matching the audit's own diagnosis that the leak's
magnitude is comparable to what it corrupts.

**But the correction is not one-directional.** Three seeds move TOWARD passing the lesion-ratio test (43 and
100 newly clear it; 101 already cleared it and stays close), and three move AWAY from it (42, 44, 102 all get
further from the 0.34 bound than they were before the fix). A plausible mechanism, not independently verified
beyond what the table shows: `_f2()`'s four `amb_read` blocks are asymmetric in drive (the `held_*` blocks run
a 60-step CONTRADICT pre-phase plus the 100-step recall at `CUE_PA=2000`pA on TWO regions, `base_*` runs only
the 100-step recall at `EPISODE_DRIVE_PA=2500`pA on one), so the pre-fix leak from `held_i` into `base_l`
(they are adjacent in the call sequence, separated only by the in-place lesion) was not a fixed bias — it
depended on which of the 12 concept blocks this seed's `_assign_blocks` RNG happened to land the CONTRADICT
role on, which overlaps differently with the `prov_generated`/`prov_perceived` read population each time. An
order-dependent leak with a seed-dependent sign is consistent with (not proof of) the C2 finding's own
characterization of this exact bug class as order-dependent rather than a fixed offset.

**`delta_intact` still never reaches `F2_INTACT_FLOOR=0.010` on any seed** — closest is seed 44 at 0.008750,
about 87.5% of the floor <!--derived-->, materially unchanged from BEFORE's closest (44 at 0.008711, ~87%
<!--derived-->). This part of the 2026-08-27 finding's own analysis — that the readout construction may be
compressing the signal via rate-response sublinearity at high recurrent drive (Sanzeni, Histed & Brunel 2020)
— is **not addressed by this fix at all**, and the numbers here are consistent with that being a genuine,
separate contributor that the read-isolation confound was sitting on top of, not substituting for.

## Verdict: does this match the audit's prediction?

**No, honestly.** The audit finding's FW-2 row predicted: *"the leak magnitude ~= the delta_lesion it
corrupts → recoverable to GO (or a clean NO-GO)"*. Neither branch of that disjunction happened. The precondition
`f2_lesion_removes_shift` genuinely tightened (1/6 → 3/6 seeds now pass the lesion-ratio sub-check it is built
from — a real, measured improvement, not noise: three seeds crossed the 0.34 boundary in the predicted
direction) but did not reach 6/6, so the crux is not "recovered to GO". It is also not a "clean NO-GO" in the
sense the audit meant (a well-behaved negative where every seed's own control validates cleanly) — three
seeds' lesion controls got MEASURABLY WORSE, not just "still short". The honest read, per this task's own
instruction, is that **the wall was partly real**: read-isolation was A genuine, partial contributor to F2's
crux failing (proven by the selftest and by the fact that the precondition's pass count moved at all), but it
is not the WHOLE explanation, and a second, still-uncharacterized source of the lesion-control's per-seed
instability remains open. This finding does not claim to have identified that second source; it only
establishes, honestly, that removing the first one was insufficient.

### Correction to the audit finding's FW-2 prediction

<!--derived-->

`research/findings/2026-09-02-read-isolation-audit-C2-bug-class-across-14-runners.md`'s FW-2 row states this
runner's outcome is "recoverable" and offers "GO (or a clean NO-GO)" as the two live possibilities. This
finding's measured result is neither, so a scoped partial-correction row is added to `docs/RETRACTED.md`
against that one predictive bullet (the audit's diagnosis of the bug's presence, magnitude, and mechanism in
this runner is otherwise confirmed correct by the table above, and its other 13 rows are untouched by this
finding).

## Files

- `research/runners/_onebrain_integration_surprise_episodic_crossedge.py` — the fix (`_EXTRA_RESET_ARRAYS` +
  `_rest_extra` snapshot/restore in `SurpriseEpisodicPool.__init__`/`_hard_reset`, and the matching fix inside
  `_migration_invariant`'s `read_surprise0` closure on the separate `b0` bridge) + the new
  `_selftest_read_isolation()` guard and `--selftest` CLI flag. No `sim/` edit; additive; no production wiring;
  no default flip (this remains a standalone `research/runners/*` de-risk).
- `research/findings/raw/_onebrain_integration_surprise_episodic_crossedge_readfix_numpy6seed.json` (+
  `.prov.json` sidecar) — the AFTER artifact this finding cites (`SIM_BACKEND=numpy`,
  `--seeds 42,43,44,100,101,102`).
- `research/queue/_onebrain_surprise_episodic_readfix_cupy_verify.sh` — guarded cupy 6-seed re-verify, SKIP-
  guarded on `_EXTRA_RESET_ARRAYS` being present in `research/runners/_onebrain_integration_surprise_episodic_
  crossedge.py` on the MAIN checkout (`/home/dant123/Projects/sim`), so it will not run against pre-merge or
  reverted code. Queued via `bash tools/gpu_queue.sh add 'bash research/queue/_onebrain_surprise_episodic_
  readfix_cupy_verify.sh'`.

Functional read-outs only; no phenomenal-experience claim.
