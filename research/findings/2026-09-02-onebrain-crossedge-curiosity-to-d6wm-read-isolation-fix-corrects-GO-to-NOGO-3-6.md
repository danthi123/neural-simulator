---
type: finding
status: live
date: 2026-09-02
mechanism: onebrain-crossedge-curiosity-d6wm-read-isolation-fix
board: one-brain integration / measurement-integrity (C2 bug class, audit item IG-1)
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_onebrain_crossedge_curiosity_to_d6wm_readfix_6seed.json
  - research/findings/raw/_onebrain_xedge_curiosity_d6_production_frozen_readfix_6seed.json
runner: research/runners/_onebrain_crossedge_curiosity_to_d6wm.py
builds_on:
  - research/findings/2026-09-02-read-isolation-audit-C2-bug-class-across-14-runners.md
  - research/findings/2026-09-02-c2-metacog-read-isolation-fix-GO.md
---

# curiosity.ask -> d6.w0 cross-edge, read-isolation fixed: the banked "GO 6/6" was inflated by an unrestored reset — corrected verdict is NO-GO 3/6, live in production, default-ON decision flagged for owner review

**One-line:** the read-isolation audit's IG-1 item (`research/findings/2026-09-02-read-isolation-audit-C2-bug-class-across-14-runners.md`)
flagged this runner as the ONE **inflated GO live in production**, predicting a corrected re-verify would flip
seed 43 to NO-GO (5/6). The REAL correction is larger than that prediction: `AskToW0Pool._hard_reset()` was
missing not only the audited 4-array C2 leak but a second, runner-specific leak in the NMDA-recurrent
conductance and synapse-pulse buffers this pair's own mechanism rides on. Restoring BOTH closes the leak to
BITWISE read identity (a new `--selftest`), and the honestly re-measured 6-seed verdict is **NO-GO 3/6**, not
5/6 — confirmed on both the runner-level pool and the REAL production wrapper's own self-test (identical
numbers to full precision). This is a live default-ON production faculty (`/api/brain-chat`'s D6 hold-query
qualifier); the default-ON decision is **flagged for owner review**, not changed here.

## 1. Background — what the audit predicted, and why the real fix goes further

The audit (§2, IG-1) ran a partial re-verify restoring only the 4 arrays the C2 template fix names
(`cp_refractory_timers`, `cp_prev_firing_states`, `cp_neuron_activity_ema`, `cp_neuron_firing_thresholds`) and
predicted seed 43 alone would flip, landing at NO-GO 5/6. Porting exactly that 4-array fix here (Port A,
`onebrain_merge_framework.MergedPool._PER_NEURON_STATE`, which already lists all 4) left the fix's own
fails-in-failing-direction selftest FAILING: two back-to-back, identically-conditioned reads on a
mechanism-zeroed (lesioned) pool were still non-identical <!--derived--> (`w0` rate 0.067875 vs 0.060750 on seed 42). Direct
instrumentation (the same before/after diff class the audit used) isolated the residual to
`cp_conductance_g_nmda_rise` / `cp_conductance_g_nmda_recurrent` / `cp_conductance_g_nmda_recurrent_rise` and
`cp_synapse_pulse_timers` / `cp_synapse_pulse_progress` — the framework's own `_SEQ_EXTRA_STATE` tuple, the set
`sequence_isolation()` (not `read_isolation()`) restores. This pair's own module docstring already names NMDA-
recurrent dynamics as load-bearing for the read's operating point ("riding NMDA-mediated recurrent dynamics ...
pushes the target population past its own recurrent excitation's effective operating point"), which is exactly
why this runner leaks somewhere the C2 template runner does not: `read_w0()` is a genuine MULTI-TURN stateful
read (a condition-blind LOAD phase, then the scored ask-driven phase, inside one `_hard_reset()`), the shape
`sequence_isolation()`'s own docstring says needs the wider tuple.

## 2. The fix — Port A, extended to both framework tuples

`research/runners/_onebrain_crossedge_curiosity_to_d6wm.py`, `AskToW0Pool.__init__`/`_hard_reset`: snapshot
`list(MergedPool._PER_NEURON_STATE) + list(MergedPool._SEQ_EXTRA_STATE)` at TRUE REST (immediately after the
existing 40-step zero-input settle), once; restore the whole set on every `_hard_reset()`, in addition to the
pre-existing piecemeal v/u/conductance/firing resets (redundant for anything the tuples also name, additive for
what they alone name). This pool has no co-resident organ to protect, so a plain snapshot/restore inside
`_hard_reset` is the direct equivalent of wrapping every read in `sequence_isolation()` — reusing the
framework's own already-tested primitive, not a hand-rolled list.

**Selftest (fails-in-failing-direction guard), `--selftest`:** on a mechanism-zeroed pool (the `ask_to_w0`
cross-edge itself lesioned to 0.0 via `lesion_cross_edges`, genuinely inert, not merely untrained), two
back-to-back `read_w0("familiar")` calls:

<!--derived--> (the table's three data columns are ad-hoc instrumentation against the pre-fix, partial-fix, and
final code on this same pool/seed — not a JSON artifact; the third column reproduces the exact numbers
`--selftest`'s own stdout line prints for the shipped code.)

| | pre-fix `_hard_reset()` | `_PER_NEURON_STATE`-only (Port A as audited) | `_PER_NEURON_STATE` + `_SEQ_EXTRA_STATE` (this fix) |
|---|---|---|---|
| repeat-read `w0` rate <!--derived--> | 0.067875 then 0.060750 (differ) | 0.067875 then 0.060750 (differ, unchanged) | 0.092000 then 0.092000 (bitwise identical) |

**A second, incidental fix landed with this one.** The runner's own `main()` wrapped the OUTCOME
(`n_go == len(runs)`, named `all_seeds_go`) as a THIRD `Vd.require(...)` precondition alongside two genuine
validity checks (`lesion_removes_bias`, `byte_identical_off`). This is the exact anti-pattern
`tools/gates/verdict_preconditions.py` exists to catch (`docs/FAILURE_GATE_MATRIX.md`'s "the OUTCOME is not a
precondition" rule) — it was latent and harmless while `n_go` was always 6/6 (`all_seeds_go` never failed), but
the moment this fix produced a genuine 3/6, `Vd.decide()` would have collapsed the run to UNDEFINED instead of
the honestly-measured NO-GO. Fixed: `all_seeds_go` removed from `Vd.require(...)`; the outcome is now passed
directly as `Vd.decide(go=all_go)`'s own argument, matching every other runner's `Verdict` usage. `preconditions`
in the cited artifact now correctly carries only the 2 genuine validity checks, both `ok: true`.

## 3. The result — NO-GO 3/6, confirmed on the runner AND the real production wrapper

<!--derived--> BEFORE (banked, `research/findings/raw/_onebrain_crossedge_curiosity_to_d6wm_6seed.json`, the
original 2026-09-01 finding's own table) vs AFTER (`research/findings/raw/_onebrain_crossedge_curiosity_to_d6wm_readfix_6seed.json`,
this finding):

| seed | grown BEFORE | grown AFTER | Δintact BEFORE | Δintact AFTER | Δlesion BEFORE | Δlesion AFTER | GO BEFORE | GO AFTER |
|---|---|---|---|---|---|---|---|---|
| 42  | 2.020219 | 2.578103 | -0.011375 | -0.012500 | -0.000750 | +0.000000 | GO | GO |
| 43  | 1.889884 | 1.589106 | -0.010500 | -0.003500 | +0.000250 | +0.000000 | GO | **NO-GO** |
| 44  | 1.981809 | 2.888025 | -0.013000 | -0.015500 | +0.000250 | +0.000000 | GO | GO |
| 100 | 1.970807 | 2.186628 | -0.010750 | -0.009000 | -0.000125 | +0.000000 | GO | GO |
| 101 | 2.117586 | 1.622379 | -0.014250 | -0.005500 | -0.000375 | +0.000000 | GO | **NO-GO** |
| 102 | 1.739121 | 1.661438 | -0.010000 | -0.002000 | -0.001125 | +0.000000 | GO | **NO-GO** |

`INTACT_FLOOR=0.008`, `LESION_RATIO=0.34` (unchanged). `n_go` **6/6 -> 3/6**; `payload.GO` **true -> false**.
`delta_lesion` is now exactly `0.0` and `frac_attributable` exactly `1.0` on every one of the 6 seeds (the
lesioned control is now a genuine, noise-free zero, same signature the C2 fix produced) — the READ is now
trustworthy; the OUTCOME it reports is a real 3/6, not an instrument artifact.

**This is a bigger correction than the audit's own prediction (NO-GO 5/6, one seed).** Removing the leak did not
uniformly shrink the effect — on seeds 42/44 the true (isolated) `delta_intact` is LARGER in magnitude than the
leaky reading (the leak was working against the true effect on those seeds), while on 43/101/102 it collapsed
toward near-zero (the leak was inflating the observed suppression there). Both directions are consistent with an
ORDER-dependent residual, not a directional bias — exactly what the audit's own mechanism diagnosis predicted,
just larger in magnitude here than the partial 4-array fix alone revealed. **Training is also contaminated**, as
flagged: the grown weights differ substantially (BEFORE 1.7-2.1 across seeds; AFTER 1.59-2.89), because every
training episode's own `_hard_reset()` was leaking too, not only the scored reads.

**Confirmed on the REAL production pipeline, not a toy probe.** `research/runners/onebrain_xedge_curiosity_d6_production.py`'s
`XedgeCuriosityD6ProductionPool.read_w0` delegates verbatim to `AskToW0Pool.read_w0` (no reimplementation), so
this fix applies there automatically with no separate patch. Its own 6-seed self-test
(`--grow --seeds 42,43,44,100,101,102`, `research/findings/raw/_onebrain_xedge_curiosity_d6_production_frozen_readfix_6seed.json`)
reproduces the SAME shift values to full precision (seed 42 `shift_intact=-0.012500`, seed 43
`shift_intact=-0.003500`, etc.) and the SAME 3/6 outcome (`n_go: 3`, seeds 43/101/102 fail
`clears_registered_floor`) — this is not a runner-level artifact that happens to differ from what production
actually runs; it IS what production runs.

## 4. What survives, and what does not

**Dies:** the "6-seed GO (6/6)" headline of `research/findings/2026-09-01-onebrain-crossedge-curiosity-to-d6wm-GO.md`
and the "6-seed GO on the production wrapper's own self-test" claim of
`research/findings/2026-09-01-onebrain-crossedge-curiosity-to-d6wm-production-wire-GO.md`. Both are marked
`status: superseded` with an in-place `⛔ CORRECTION` section pointing here; `docs/RETRACTED.md` carries both
rows (PARTIAL).

**Survives:** the mechanism itself. The cross-edge genuinely GROWS from the substrate's own Hebbian rule on
every seed (no seed's `no_corruption` check nor `emergence.PASS` changed). The suppression sign is unchanged and
still cleanly lesion-attributable (now MORE cleanly — `frac_attributable` is exactly 1.0, not 0.89-1.02) on the
3 seeds that clear the floor. `byte_identical_off` is unaffected (this fix touches only the harness reset, never
the pool's wiring/construction). The biological framing (§1 of the original finding — the honest correction from
a predicted DA-gating BOOST to a measured attentional-capture SUPPRESSION) is untouched; if anything the cleaner
read makes that correction's own measurement more trustworthy, not less.

**NOT re-verified here (honest scope limit):** `research/findings/2026-09-01-onebrain-crossedge-curiosity-d6wm-semantic-drop-GO.md`
builds directly on `AskToW0Pool`/`pool.cross_weight` (the SAME frozen edge, SAME `read_w0`) for its own 6-seed
GO — it almost certainly inherits an analogous correction (its own `train()`/read machinery is the identical
`AskToW0Pool`), but this finding did not re-run it. Flagged, not fixed, here.

## 5. Default-ON decision — flagged for owner review, NOT changed by this correction

`docs/PRODUCTION_INTEGRATION_LEDGER.yaml`'s `onebrain-xedge-curiosity-d6` row still reads `on_by_default: YES`
(`_XEDGE_CD6_DEFAULT_ON = True` in `onebrain_xedge_curiosity_d6_production.py`) — **this finding does not flip
it.** The 2026-09-01 AUTO-FLIP decision explicitly rested on "validated-GO + load-bearing + moat-safe +
byte-identical-off + no-regression" per that finding's own §7; the "validated-GO" premise is now false (NO-GO
3/6, not 6/6). Whether a 3/6 cross-edge should remain the live text an actual `/api/brain-chat` hold-query turn
returns (vs. reverting to `_XEDGE_CD6_DEFAULT_ON = False` pending re-tuning, or accepting a probabilistic
per-seed-like faculty) is an owner UX call, not a mechanical read-isolation fix — flagged here, decided
elsewhere. No `webapp/server.py` edit, no ledger edit, in this finding.

## 6. Honest residuals

- **Why 3 specific seeds (43/101/102) fail and 3 (42/44/100) pass is not diagnosed here** — this finding fixes
  the INSTRUMENT and reports the corrected OUTCOME; it does not re-tune `N_EPISODES`/`ASK_DRIVE_PA`/`HMAX` to
  chase 6/6 again. Per the task's own framing, the mechanism may need re-tuning against the now-trustworthy read,
  not just a re-measurement — that re-tuning is not attempted here.
- **The semantic-drop finding is un-reverified** (§4).
- **Cupy is not run here** (numpy CPU only, per this lane's scope); a guarded cupy 6-seed re-verify is queued
  (see the accompanying report) — numpy and cupy have diverged before on adjacent edges, so this NO-GO 3/6 is
  NOT assumed to reproduce identically on cupy without that confirmation.
- **RSS/backend:** `SIM_BACKEND=numpy`, watched under 4GB RSS throughout (small ~994-neuron merged pool).

## 7. Files

`research/runners/_onebrain_crossedge_curiosity_to_d6wm.py` (MODIFIED — `_hard_reset`/`__init__` read-isolation
fix, `_selftest_read_isolation`, `--selftest` CLI, the `Verdict` outcome-as-precondition fix) ·
`research/findings/raw/_onebrain_crossedge_curiosity_to_d6wm_readfix_6seed.json` ·
`research/findings/raw/_onebrain_xedge_curiosity_d6_production_frozen_readfix_6seed.json`. Reused, unmodified:
`research/runners/onebrain_merge_framework.py` (`MergedPool._PER_NEURON_STATE`, `_SEQ_EXTRA_STATE`) ·
`research/runners/onebrain_crossedge_gate.py` (`lesion_cross_edges`, reused by the new selftest) ·
`research/runners/onebrain_xedge_curiosity_d6_production.py` (unmodified — inherits the fix by reuse-by-import).
No `sim/` file touched; no `webapp/server.py` edit; no `docs/PRODUCTION_INTEGRATION_LEDGER.yaml` edit; no
production default changed.

Functional read-outs only; no phenomenal-experience claim.
