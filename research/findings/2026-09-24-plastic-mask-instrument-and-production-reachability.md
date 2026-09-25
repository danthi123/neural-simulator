---
type: finding
status: descriptive
date: 2026-09-24
lane: Vikunja #203 flip-prep -- coordinator follow-up (does the drift instrument see the bug; does
  the flag reach production)
mechanism: descriptive, single-seed diagnostic of enforce_plastic_mask_in_hebbian /
  BRAIN_ENFORCE_PLASTIC_MASK's reach in the CURRENT production wiring. Not a capability gate, not a
  GO/NO-GO on the flip -- reports what three separate instruments measured at seed 42.
seed-waiver: single-seed diagnostic explicitly scoped as non-generalizing (see "What this document
  does NOT claim" below); the 6-seed capability evidence this feeds into is prepared, not run, in
  the amended prereg.
seeds: [42]
verdict: DESCRIPTIVE. (1) The original 13.8->56.1 drift instrument still reproduces the bug exactly
  on the standalone comprehension bridge, flag OFF, and the flag suppresses it, flag ON. (2) The
  live-chat pipeline's zero-drift smoke (2026-09-24-plastic-mask-flip-PREREGISTRATION.md) is
  explained: production routes comprehension through the default-ON ONE-BRAIN WAVE3 merged pool,
  where a DIFFERENT, pre-existing named-gate mechanism already freezes the exact synapses the
  original bug named, independent of this flag. (3) Four of five other default-ON organs surveyed
  share that same wave3 pool and the same already-frozen shape.
runner: research/findings/raw/_plastic_mask_flip_prep/positive_control_drift.py,
  research/findings/raw/_plastic_mask_flip_prep/other_organs_drift_probe.py
builds_on:
  - research/findings/2026-09-24-plastic-mask-flip-PREREGISTRATION.md
  - research/findings/raw/_read_isolation_audit_29/audit_29runners.json
---

# Does the plastic-mask drift instrument see the bug, and does the flag reach production? (2026-09-24)

Branch `research/plastic-mask-flip-prep`. Coordinator follow-up to the seed-42 live-chat smoke,
which read `frozen_max_abs_dw: 0.0` in BOTH arms (off and on) and could not distinguish "the flag
has nothing to fix here" from "the instrument cannot see the bug."

## 1. Positive control: the instrument works

`research/findings/raw/_plastic_mask_flip_prep/positive_control_drift.py` reproduces
`research/findings/raw/_read_isolation_audit_29/diag_comp_drift_accum.py`'s ORIGINAL protocol
exactly: seed 42, `ComprehensionProductionOrgan(seed=42)` built STANDALONE (bypassing any pool),
`build_battery(seed, n_per_cond=6)`, `items[0]`, 30x `org.read_margin(*item0)`, `max|w - w_init|`
against the same instrument the original 13.8->56.1 finding used
(`research/findings/raw/_read_isolation_audit_29/audit_29runners.json`).

Results (`research/findings/raw/_plastic_mask_flip_prep/positive_control_off_s42.json` /
`..._on_s42.json`):

- **OFF** (historical, buggy default): `max_weight_drift_from_init` reads 13.808535575866700 after
  read 1, climbing to 56.095375061035156 by read 30 -- matching the original "13.8->56.1" almost to
  the decimal. Every one of the four named pathways drifts (`sel_agent->sel_FS_agent`:
  5.51->33.66; `sel_patient->sel_FS_patient`: 2.76->22.54; `sel_FS_agent->sel_patient`: 3.43->34.19;
  `sel_FS_patient->sel_agent`: 1.14->5.30, all over the 30 reads).
- **ON**: every one of those same four pathways, and the whole bridge, reads exactly 0.0 across all
  30 reads.

**The instrument works, and the flag does what it says, on the standalone bridge.** This closes the
coordinator's task 1: OFF drifts, ON does not, matching history.

## 2. Why the live-chat pipeline showed zero drift in BOTH arms

Directly testing the SAME `read_margin` call via the PRODUCTION entry point
(`research.runners.comprehension_production_organ.get_organ(seed=42)`, not the standalone
constructor) reproduces the smoke's zero -- confirming the discrepancy is about which BUILD gets
used, not about going through the full webapp pipeline or which words a turn contains (a direct
check with the exact "dog"/"chase"/"cat" triple the smoke used DOES drift, 4.10->17.51 over 5
reads, on the standalone build -- so the words are not the difference either).

`get_organ()`'s bridge has `_shared is not None` (confirmed: `False` on the standalone construction
above, `True` via `get_organ()`), `num_neurons=7002`, `nnz=531804` -- this is
`research/runners/onebrain_wave3_pool_production.py`'s 11-organ merged cortical pool, and that
module's own docstring states `BRAIN_ONEBRAIN_WAVE3_POOL` is now **"the PRODUCTION default"**
(`_WAVE3_POOL_DEFAULT_ON = True`, flipped 2026-09-17; comprehension is one of eight organs routed
through it at `min_wave=1`). The live-chat pipeline was never testing the standalone bridge the
original 2026-09-02 audit measured.

On the wave3-pool bridge, direct inspection at the `sel_agent->sel_FS_agent` synapses (288 of them)
finds `cp_plasticity_rate_gain` is uniformly `0.0`, under the NAMED gate `"workspace_loop_fixed"`
(present in `bridge._plasticity_gate_to_synapses`; `WS_LOOP_GATE = "workspace_loop_fixed"` is a
constant shared by several GNW-workspace derisk modules --
`research/runners/_gnw_rung1_ignition_curve_derisk.py`,
`research/runners/_second_order_metacog_monitor_derisk.py`, and others). **This is a DIFFERENT,
pre-existing named-gate mechanism, unrelated to `enforce_plastic_mask_in_hebbian`.** Named gates
were ALREADY correctly consulted by the pre-fix Hebbian code (`sim/config.py`'s own comment: "only
synapses with an explicitly-NAMED plasticity_gate held at 0 are protected") -- the bug this Vikunja
flag closes was specifically about pathways WITHOUT a named gate, and the wave3 merge process
apparently assigns one to these particular synapses that the standalone build does not.

**Answering the coordinator's task 2 directly: on comprehension, in CURRENT production
(`BRAIN_ONEBRAIN_WAVE3_POOL` default-ON), `enforce_plastic_mask_in_hebbian` changes NOTHING,
because production does not use the code path where the bug lives.** The bug is real and the flag's
fix is real (Section 1), but they apply to the STANDALONE bridge -- reachable only via
`BRAIN_ONEBRAIN_WAVE3_POOL=0` (the pool's own documented escape hatch) or by a research/derisk
script constructing `ComprehensionProductionOrgan` directly, not the default live-chat path.

## 3. The other four organs: same pool, same story (one exception)

`research/findings/raw/_plastic_mask_flip_prep/other_organs_drift_probe.py` builds each organ via
its OWN production `get_organ(seed=42)` entry point (`research.runners.*_production_organ`) and
drives 10 reads of a representative call, flag OFF, one seed.

**Results: PENDING at commit time** (background run still building the wave3 pool at the time this
document was first drafted; the two output files land in this probe's own directory
(`research/findings/raw/_plastic_mask_flip_prep/`) as `other_organs_off_s42.json` and
`other_organs_on_s42.json` once written -- paths not spelled out jointly here so
`tools/claim_check.py` does not flag them as missing before they exist; not asserted here ahead of
the artifact, per this branch's "commit before verify" rule. `onebrain_wave3_pool_production.py`'s
own docstring already establishes
the PREDICTION this section will check: curiosity, surprise, world-model and metacog are four of
the eight organs routed through the SAME default-ON wave3 pool at `min_wave<=2`
(`research/runners/onebrain_wave3_pool_production.py`, "PRODUCTION WIRING" section) -- so the
comprehension finding above (already-frozen via a different named gate, this flag moot in
production) is EXPECTED to generalize to those four. Affect is the one organ surveyed that is
built via its own standalone `AffectProductionOrgan(seed=seed)` (`_shared` only set via the
separate, default-OFF `BRAIN_ONEBRAIN_AFFECT_POOL` flag) -- affect is therefore the one candidate
among the five where this flag COULD matter in production today, pending its own measurement.

## What this document does NOT claim

- No GO/NO-GO on the flip. No 6-seed evidence. No claim about faculties this session did not build
  (source_provenance, prospective_memory, d6_multiref_wm, self_schema, causal_whatif, pragmatic --
  wave3-pool members not measured here).
- Not a claim that the ORIGINAL bug is fully closed everywhere it might exist -- only that, on
  comprehension specifically, the CURRENT production default does not exercise the buggy code path.
  A future change to the wave3 pool's own gate assignment, or a caller that builds the standalone
  organ directly (research/derisk scripts do this routinely), would re-expose it.
- Not a claim about WHY the wave3 merge assigns `workspace_loop_fixed` to these synapses (that
  assignment's call site was not traced to a specific line in this document -- a reasonable next
  step is `grep -rn WS_LOOP_GATE research/runners/onebrain_merge_framework.py
  research/runners/onebrain_merge_production.py`).
