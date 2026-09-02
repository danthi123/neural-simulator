---
type: finding
status: no-go
date: 2026-08-27
mechanism: onebrain-integration-r3v2-noncorrupting-dopamine-credit
lane: one-brain / integration / emergence-bar
artifacts:
  - research/findings/raw/_onebrain_integration_r3v2_noncorrupting_6seed.json
runner: research/runners/_onebrain_integration_r3v2_noncorrupting_dopamine_credit.py
supersedes_diagnosis_of: research/findings/2026-08-27-onebrain-integration-R3-spiking-dopamine-credit-PARTIAL.md
---

**ADDENDUM (2026-09-02, no verdict change)** — the 2026-09-02 read-isolation audit flagged this runner's
`_hard_reset()` for the same C2 bug class the metacog fix closed (`cp_refractory_timers`/`cp_prev_firing_states`
never restored) and asked whether fixing it flips this NO-GO. It does not: a fresh 6-seed re-verify with the
fix applied reproduces NO-GO 0/6, best seed (100) still missing `F2_INTACT_FLOOR=0.008` (now by 0.00170, a
small non-monotonic shift from the pre-fix 0.00198). Full detail, per-seed table, and the sibling `R3v3`
re-verify (which DOES flip — GO 6/6 → NO-GO 3/6) are in
`research/findings/2026-09-02-r3v2-r3v3-read-isolation-refix-r3v3-GO-flips-to-NOGO.md`. Artifact:
`research/findings/raw/_onebrain_integration_r3v2_noncorrupting_readfix_numpy6seed.json`.

# One-brain INTEGRATION R3-v2 — the migration-byte-identity precondition now HOLDS (R3's `da_credit` corruption diagnosis was WRONG; the real cause was two freeze-ordering bugs, now fixed); the full functional gate is DEFINED — NO-GO on F2, an honest small-effect negative, not an UNDEFINED precondition failure

**One-line:** R3's PARTIAL finding blamed `da_credit`'s fixed coincidence synapses for corrupting the shared
merge pool's connectivity. Direct instrumented measurement shows that diagnosis is **WRONG**: those synapses
never move a single bit (0.0 diff, every seed). The real corruption was **two independent freeze/snapshot-
ORDERING bugs in the runner**, both exposed only because `da_credit` registers a **permanently-present**
"dopamine" neuromodulator (R2 never hit either, because R2 never registers one). Both are fixed here with **NO
mechanism change and NO `sim/` edit**. Result: `migration_byte_identity` now **HOLDS 6/6**, `no_corruption_intact`
now **HOLDS 6/6**, and the DOPAMINE-LESION control **stays 6/6** (the mechanism R3 proved load-bearing is
untouched). With the precondition clean, the full F1-F4 gate is now **DEFINED** — and it is a clean **NO-GO**:
**F2 fails 0/6**, not because the cross-edges are inert (they are >=70% lesion-attributable on every seed) but
because the shift's absolute magnitude sits under a floor (`F2_INTACT_FLOOR=0.008` <!--derived--> — the
constant R2/R3-v2 pre-register in `_onebrain_integration_r2_threefactor_selforganized.py`, not a per-run
measurement) that was pre-registered for R2's larger, non-DA-mediated host-scalar pathway, not for R3's
DA-mediated one.

## The root-cause correction (why R3's own diagnosis was wrong)

R3's PARTIAL finding read the evidence (`no_corruption_intact 0/6`, `lesion_recovers_migration 0/6`, both at a
uniform `frozen_maxdrift=16.0` on every seed) and concluded the newly-added `da_credit` organ's fixed
`sel/teach -> snc` synapses were perturbing the shared pool. Instrumented, per-mask before/after diffing
(building `R3Pool` step by step and diffing `cp_connections.data` at each checkpoint) shows those 4 synapse
groups carry **exactly 0.0 diff** from build through training, on every seed tested — the DOPAMINE-LESION
control's own 6/6 result already implied this (a corrupted coincidence circuit could not reliably gate
learning), but the PARTIAL finding did not connect the two results.

The actual corruption has two DISTINCT causes, both freeze/snapshot-**ordering** bugs, neither touching
`da_credit`'s circuit:

1. **R3Pool's own baseline was snapshotted too early.** `comp_organ.ensure_built()`'s first action
   (`SpikingRoleCompetition.set_cue_weight`, called from `_build_comp`) **directly overwrites** comprehension's
   `cue_* -> sel_agent/sel_patient` pathway weights with the calibrated `INSTALLED_CUE_WEIGHTS` — a deterministic,
   gain-independent raw array write, not an STDP-driven change. R2 captures its migration baseline **after**
   `comp_organ.ensure_built()` runs (so this install is already "baked in"); R3's WHITELIST-FREEZE-FIRST
   reordering (done for a real, separate reason — protecting GATE-tagged candidate edges from calibration-time
   reward-driven drift) accidentally moved the baseline snapshot earlier too, so R3 counted this legitimate
   one-time install as "corruption."
2. **`_migration_invariant`'s baseline pool (`pool0`) was never frozen at all.** In R2 this is harmless: with no
   "dopamine" modulator registered and `current_reward_signal` never set on `pool0`, `effective_signal` in
   `sim/bridge.py`'s C2 block is exactly 0.0, so the whole reward-modulated block (including its weight-CLIP,
   ~line 10725-10741) never activates. R3's `da_credit` organ registers a "dopamine" `NeuromodulatorConfig`
   **unconditionally** (permanent infrastructure), so `effective_signal = DA_concentration - DA_baseline` is read
   every step, even at rest — measured baseline concentration ~6.7e-5, safely above the `1e-6` activation floor.
   On `pool0` (gain=1.0 everywhere, never frozen), that nonzero-at-rest reading is enough to run the reward
   block's weight-CLIP every calibration step. `cfg.stdp_w_max` at calibration time is still its **unset default,
   2.0** (R3Pool only raises it to `HMAX=20.0` after `comp_organ.ensure_built()` returns) — so D6's own un-gated
   internal recurrent "hold" weights (design value ~25, e.g. `w1->w1`) get clipped down to exactly 2.0. Measured
   directly: `pool0`'s `w1->w1` sample reads `[24.89, 25.81, 25.93, ...]` before `ensure_built()` and uniformly
   `[2.0, 2.0, 2.0, ...]` after, on an unfrozen pool; the identical call on a frozen pool (gain=0 for `w1->w1`,
   as `r3.pool` already gets) leaves it byte-unchanged. **`_migration_invariant` was comparing an intact `r3.pool`
   to a self-corrupted `pool0`** — that mismatch, not anything `da_credit` did, is what failed the invariant.

## The fix (R3-v2, `_onebrain_integration_r3v2_noncorrupting_dopamine_credit.py`)

Two ordering corrections, zero mechanism change: (1) `R3v2Pool` captures `_frozen_w0` **after**
`comp_organ.ensure_built()` (and after `da_lesioned` mode's own wiring edit), matching R1/R2's convention
exactly; the plasticity-gate freeze itself stays where R3 put it (early — still needed to protect the candidate
edges). (2) `_migration_invariant` applies the identical `cp_plasticity_rate_gain[:] = 0.0` flat freeze to
`pool0` before its own `comp_organ.ensure_built()` call (no candidate GATE exists on a `with_cross=False` pool,
so nothing needs re-opening). The coincidence-detector circuit, the dopamine `ProductionRule`, the
DOPAMINE-LESION control, and every F1-F4/R3-a arm are reused byte-identical from R2/R3.

## Per-arm, 6-seed (42/43/44/100/101/102), numpy CPU

Artifact: `research/findings/raw/_onebrain_integration_r3v2_noncorrupting_6seed.json`.

| arm | R3 (broken precondition) | R3-v2 (fixed) | reading |
|---|---|---|---|
| `no_corruption_intact` | 0/6 (drift=16.0 every seed) | **6/6 (drift=0.0 every seed)** | fix #1: baseline snapshot ordering |
| `lesion_recovers_migration` (connectivity byte-identical) | 0/6 | **6/6** | fix #2: `pool0` freeze |
| F1 faculty-still-works | 6/6 | 6/6 | unaffected (never touched `pool0`) |
| F2 vary-then-lesion | 0/6 | 0/6 (unchanged) | real, >=70% lesion-attributable, but under-floor — see below |
| F3 no-runaway | 6/6 | 6/6 | unaffected |
| F4 moat | 6/6 | 6/6 | unaffected |
| R3-a three-factor (intact selective / removed inert / shuffled degraded) | 5/6 | 5/6 (unchanged) | unaffected |
| R3 DOPAMINE-LESION control | **6/6** | **6/6** | mechanism confirmed load-bearing, untouched by the fix |

`F2`, `F3`, `F4`, `F1`, and R3-a never touch `pool0` (they read/lesion only `r3.pool`, which was never
corrupted), so their pass rates are numerically identical before and after the fix — the fix's whole effect is
concentrated exactly where the diagnosis predicted (the two migration-invariant arms), which is itself a check
that the correction targets the right mechanism.

## F2: a real, attributable, but under-floor effect — an honest NO-GO, not tuned away

F2's held-vs-none margin shift is measurable and >=70% attributable to the candidate cross-edges on every seed
(lesioning them collapses the shift toward zero, sometimes past it): `frac_attributable_agent` ranges
0.70-1.21, `frac_attributable_patient` ranges 0.95-2.13 across the 6 seeds. But `delta_agent_intact` tops out at
0.0060 (seed 100) against the pre-registered `F2_INTACT_FLOOR=0.008` <!--derived--> — every seed falls short. That floor was
calibrated for R2's host-scalar three-factor pathway (a raw `current_reward_signal=1.0` pulse,
`N_EPISODE_PAIRS=100`); R3/R3-v2's DA-population-mediated credit necessarily delivers a smaller `effective_signal`
per step (a spiking population's concentration deviation from baseline, scaled by
`compute_plasticity_rate_multiplier()`, is not a unit-magnitude pulse) even when a coincidence event unambiguously
fires. **This is not re-tuned here** — per `docs/TERMS.md`'s GO discipline, loosening a pre-registered floor to
flip a verdict is exactly the overclaim pattern the terms file exists to block. The honest reading: R3-v2's
mechanism is genuinely load-bearing and genuinely selective, but under-powered relative to R2's floor at the
current protocol scale (`N_EPISODE_PAIRS`, `DA_SENSITIVITY`, `REWARD_LR`) — a legitimate next rung (re-calibrate
a DA-mediated-pathway-specific floor, or scale the protocol), not a broken mechanism.

## What this means (honest)

**Closed:** the migration-byte-identity PRECONDITION R3 failed to establish. `da_credit` is confirmed
**non-corrupting** — the full functional gate is now validly measurable on this pool, and the DOPAMINE-LESION
control (R3's crux, the mechanism-load-bearing proof) is unaffected and still 6/6.

**Open, correctly characterized (not a wall):** the full F1-F4 gate is a **defined NO-GO** driven by F2 alone —
a real, attributable, small effect below a floor calibrated for a different pathway. The next rung is a
DA-pathway-specific floor calibration or a protocol-scale sweep, not further integration-corruption debugging.

**A standing note for the next organ:** any research runner that registers a **permanently-present**
neuromodulator on a shared `MergedPool` must freeze `cp_plasticity_rate_gain` on **every** pool instance used in
a byte-identity/no-corruption comparison (including baseline and lesion-control pools), not just the "main"
pool under test — the reward-modulated block's weight-CLIP is gain-gated, not update-magnitude-gated, so an
unfrozen pool with ANY nonzero-at-rest neuromodulator reading can silently clamp an untouched high-value weight
down to whatever `stdp_w_max` happens to default to at that moment in the build sequence.

Functional read-outs only; no phenomenal-experience claim.
