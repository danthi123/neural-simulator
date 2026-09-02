---
type: finding
status: live
date: 2026-09-02
mechanism: onebrain-r4-selfschema-production-wirein-read-isolation-reverify
board: one-brain integration / measurement-integrity (C2 bug class, record follow-through)
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_onebrain_xedge_selfschema_production_frozen_readfix_6seed.json
runner: research/runners/onebrain_xedge_selfschema_production.py
builds_on:
  - research/findings/2026-09-02-onebrain-r4-selfschema-provenance-read-isolation-fix-flips-GO-to-NOGO.md
  - research/findings/2026-08-27-onebrain-r4-selfschema-provenance-production-GO.md
---

# R4's production wire-in (default-OFF) re-verified against the fixed R4Pool: the wiring's OWN GO (lesion-attributability, 6/6) SURVIVES, but the disclosed "5/6 clear R4's floor" residual is now 2/6 — a further, third instance of the same correction pattern

**One-line:** `research/findings/2026-08-27-onebrain-r4-selfschema-provenance-production-GO.md` builds an
INDEPENDENT `R4Pool(seed)` instance (`research/runners/_onebrain_integration_r4_selfschema_provenance.py`,
"reused by import ... not reimplemented") and calls its own `.train()` — the exact same leaky-then-fixed
training path `fa4e10271` corrected at the runner level. Re-running this wire-in's own 6-seed self-test against
the current (fixed) `R4Pool`: the wiring's OWN verdict (`n_go`, lesion-attributability through the live
`crossedge_provenance_shift` hook — the crux this finding was actually GO on) **stays 6/6, unchanged**. But the
DISCLOSED RESIDUAL number ("5/6 clear R4's own pre-registered floor") **drops to 2/6** — the same magnitude of
degradation the parent r4 runner-level fix found (6/6 → 2/6), propagated here because this wire-in trains its
own `R4Pool` fresh rather than reading a saved weight. `BRAIN_ONEBRAIN_XEDGE_SELFSCHEMA` remains default-OFF
(`_XEDGE_SS_DEFAULT_ON = False`, unchanged) — not a live production over-claim.

## The re-verification

Re-ran the finding's own command (`SIM_BACKEND=numpy python -m research.runners.onebrain_xedge_selfschema_production
--seeds 42,43,44,100,101,102 --out ...`) on a worktree branched from post-`fa4e10271` `main` (so `R4Pool._hard_reset`
already restores `MergedPool._PER_NEURON_STATE`, verified via that fix's own `--selftest`).

| seed | intact shift BEFORE fix (from the original finding's own table) | intact shift AFTER fix | clears R4 floor (0.010) BEFORE | clears R4 floor AFTER | n_go contribution |
|---|---|---|---|---|---|
| 42  | not itemized in the original finding's prose (only the aggregate 5/6 was reported) | +0.008125 | — | NO | GO (lesion-attributable) |
| 43  | — | +0.009688 | — | NO | GO |
| 44  | — | +0.012188 | — | YES | GO |
| 100 | — | +0.008750 | NO (the original finding's own disclosed exception) | NO | GO |
| 101 | — | +0.008125 | — | NO | GO |
| 102 | — | +0.013438 | — | YES | GO |

`n_go` (lesion-attributable through the live `crossedge_provenance_shift` hook, `frac_attributable_to_cross_edge=1.0`
on every seed, `no_signal_no_bias_ok=True` on every seed) stays **6/6 — the wire-in's own headline GO does not
change**. `n_clears_r4_registered_floor` (the honestly-disclosed secondary residual) drops from the original
**5/6** to **2/6** (only seeds 44 and 102 still clear `F2_INTACT_FLOOR=0.010`).

## Why the wiring's own GO survives while the floor residual worsens

The wire-in's own pass/fail bar (`GO = lesion_attributable and no_signal_ok`, per the runner's own
`_selftest_loadbearing`) never depended on the absolute magnitude clearing R4's floor — that check was always a
SEPARATE, secondary, disclosed residual ("HONEST RESIDUAL... `clears_r4_registered_floor` reports how many seeds
still clear R4's own pre-registered floor... may be < n_go; GO is graded on lesion-attributability... not
silently re-using R4's floor as if unchanged", per the original finding's own note text). The read-isolation fix
changes the TRAINED weight's magnitude (via the same training-leak mechanism `fa4e10271` documents for the
runner-level R4Pool) without touching the sign or the lesion's completeness — so the binary lesion-attributable
GO survives structurally, while the continuous, floor-relative residual moves with the corrected (generally
smaller) shift magnitudes.

## Scope and honesty

This is the THIRD instance in this record-follow-through lane of the same shape: a wiring/rung's own headline
verdict SURVIVES a read-isolation correction, while a specific disclosed number changes materially (here, a
residual getting WORSE, unlike the other two cases where the primary metric itself narrowed). `docs/RETRACTED.md`
carries a PARTIAL row for the original finding's "5/6 clear R4's own... floor" figure. No `docs/PRODUCTION_
INTEGRATION_LEDGER.yaml` row exists for this wire-in (it has never been flipped default-ON), and no summary doc
(`ROADMAP.md`, `GAP_CLOSURE_MISSION.md`, the master roadmap plan) cites it, so no further doc reconciliation is
needed beyond the RETRACTED.md entry and this write-up.

## Files

No code changed — this is a re-verification only (the fix that changed the measured numbers,
`research/runners/_onebrain_integration_r4_selfschema_provenance.py`'s `R4Pool._hard_reset`, was already applied
and verified by `fa4e10271`). New: `research/findings/raw/_onebrain_xedge_selfschema_production_frozen_readfix_6seed.json`.

Functional read-outs only; no phenomenal-experience claim.
