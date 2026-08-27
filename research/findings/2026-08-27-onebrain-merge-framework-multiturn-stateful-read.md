---
status: live
type: finding
lane: integration
date: 2026-08-27
---

# One-brain merge framework — a MULTI-TURN STATEFUL READ harness + prospective_memory ORGAN-READ closed in co-residence

Status: GO (pmem organ-read byte-identity, in co-residence with the frozen-forward organs) + an honest, precisely
characterized SCALE residual for the full 7-organ strict batch. This builds the reusable multi-turn stateful read
infrastructure the framework lacked (all prior organ-reads were single drive->settle->read passes) and uses it to
close `prospective_memory` — the ONE remaining Group-A organ whose read is a SEQUENCE of turns whose STATE (a
self-sustaining cortex<->dlpfc attractor + per-neuron SFA trace) must persist UNRESET across the turns. NO `sim/`
edit. All runs are tiny numpy nets on the CPU.

Artifacts: `research/findings/raw/_onebrain_merge_multiturn_pmem_6seed.json` (the byte-identity batched verify —
pmem + self_schema + comprehension, 6 seeds; carries a `tools.verdict.Verdict` `preconditions` block) +
`research/findings/raw/_onebrain_merge_multiturn_pmem_nondegen_6seed.json` (the per-seed non-degeneracy read:
fire/held/same-pool-silent per seed).

## The multi-turn stateful read HARNESS (the reusable infra)

The framework's `read_isolation(key)` is PER-CALL: snapshot at enter, restore every OTHER organ's slice at exit,
let the active slice evolve. That already lets a MULTI-turn read hold its slice across the turns inside ONE guard;
the two things it does NOT do — and that a deep, stateful organ needs — are added here, GENERAL (any stateful
organ declares its read this way, not pmem-specific):

- `MergedPool.sequence_isolation()` — a SEQUENCE-scoped guard. It snapshots EVERY mutable array (per-neuron + the
  per-synapse pulse/rise buffers), the runtime timing counters, AND the global RNG state (np.random + Python
  `random`) at enter, lets the active organ's slice evolve UNRESET across every turn (form -> hold through
  arbitrary intervening turns -> cue), and at exit restores the FULL snapshot so no co-resident organ is perturbed
  — even by the whole-bridge `_reset_dynamics` the read calls between its own sub-sequences. The RNG restore was
  load-bearing: pmem's calibration draws a Python `random.random()`, which — un-restored — silently perturbed a
  downstream organ's read on the shared bridge (an order-dependent leak invisible to the per-array restore).
- The per-organ OPERATING-POINT reconciliation on the shared substrate, applied inside the guard + restored at
  exit (mirrors causal_whatif's save/restore of its train config): pmem's attractor/SFA are tuned at `dt=0.5`
  while the pool runs at `dt=1.0` (the other organs' point), so the read sets `cc.dt_ms` + the delay horizon +
  RESCALES the cached conductance decays (`decay**(new/old)=exp(-new/tau)`, exact, no tau needed) to 0.5 for the
  pmem read only, restoring at exit — the other organs stay byte-identical at `dt=1.0`.

## Threading `shared=` through the 3-class prospective_memory hierarchy (byte-identical when None)

`base ProspectiveMemory -> HomeostaticProspectiveMemory -> SFANmdaProspectiveMemory`, each of which builds/steps
its own bridge in `__init__` (a multi-stage homeostat + NMDA-plateau CALIBRATION). Threaded with an additive
`shared=None`:

- base `__init__`: `if shared is None:` wraps the ENTIRE original build verbatim; the shared branch ADOPTS
  `pool.bridge`, discovers the cortex/dlpfc/rel slices, and computes the same perm-based assembly index maps +
  edge references — it NEVER builds a bridge, edits `sim/`, or steps the substrate. The attractor + cue-monitor
  outer-product edges move to the descriptor's `explicit_wiring_fn` (build-time, both arms identical, like
  self_schema's GNW loops); the rel-internal recurrence comes from `build_wiring_plan(per_region_seed=True)`.
- Homeostatic + SFANmda: `shared` flows through `**kw`; their per-seed bias/theta MODULE caches are BYPASSED when
  shared, so the merged and coresident arms each calibrate INDEPENDENTLY on their own slice — the byte-identity of
  the calibrated bias/theta is a GENUINE result, not a cache hit.

BYTE-IDENTICAL-WHEN-`shared=None` — VERIFIED by git-stash/rerun: the standalone SFANmda de-risk produces the exact
same output (every per-seed fire/held rate, the calibrated bias, the plateau separation-margin diagnostic) with vs
without this arc's edits — the whole `shared=None` path is skipped verbatim, so the change is purely additive.

## What closes — pmem organ-READ byte-identity in co-residence (6 seeds)

Co-resident with `self_schema` + `comprehension` on ONE wired pool (N=3442), pmem's REAL multi-turn form -> hold
(N=5 intervening distractor turns) -> cue read is byte-identical merged-vs-coresident:

- `read_maxerr == 0.0` AND `answer_same == True` on EVERY seed (6/6); substrate-init byte-identity 6/6; legacy
  discriminator diverges 6/6; POOL ALL-GO True; the batched Verdict decides GO.
- NON-DEGENERATE (the coincidence-gated release is real, not a vacuous all-silent read), 6 seeds, from
  `research/findings/raw/_onebrain_merge_multiturn_pmem_nondegen_6seed.json`:
  <!--derived-->
  - The MULTI-TURN HOLD is robust on ALL 6 seeds — the self-sustaining attractor holds the deferred intention
    across the N intervening turns: every per-seed `held_min` in the cited nondegen artifact is >= 0.32 (min 0.3287,
    max 0.3438).
  - The RELEASE fires on the HELD item and NOT on a same-pool control: `fire_A_on_cueA` vs rel_A's OWN silence
    (before-cue hold ramp, wrong-cue, no-intention) clears by `fire_over_samepool` = 3.2..8.5 every seed. Seed 44 is
    the weakest (3.2x, fire 0.025) — the exact seed the de-risk itself found hardest (its plateau-rescued one),
    now on the harder noise-free co-residence substrate.

## The two seams the pool substrate FORCED (all byte-identical merged-vs-coresident; NO sim/ edit)

The pool substrate differs from pmem's tuned standalone in ways that co-residence byte-identity REQUIRES, so the
mechanism was re-homed onto them (both arms use the SAME re-tuned values -> byte-identity preserved; the standalone
de-risk is untouched):

1. `num_traits=1` (required: with >1 the per-neuron trait draw is a global-RNG index-order draw =>
   co-residence-DEPENDENT) delivers ~6x LESS effective synaptic current per unit weight than the de-risk's
   `num_traits=5`. Reconciled by a single POOL GAIN of 6x on EVERY pmem synaptic weight (attractor 50->300,
   cue-monitor, rel-recurrent) — the delivered currents match the standalone's, so the whole tuned attractor +
   cue-monitor + homeostat + plateau balance transfers, and the homeostat's EXTERNAL-current (pA) bias keeps its
   scale.
2. Conductance noise is OFF (required for determinism) => the rel accumulator sits BELOW rheobase at bias 0.
   Reconciled by allowing the homeostat bias to go POSITIVE (BIDIRECTIONAL homeostasis — Turrigiano's set-point
   control lifts a hypo-excitable pool too, not only hyperpolarizes) + a stronger SFA/plateau so the sustained
   hold-ramp adapts away and the transient coincidence is supralinearly amplified.

## Honest boundary — the FULL 7-organ strict batch: a total-N SpMV-DETERMINISM residual (sim/-edit)

pmem's OWN read is byte-identical in co-residence (above). But adding pmem — the LARGEST organ (1720 neurons) — to
the FULL 7-organ pool pushes total-N (=4968) past the point where the OTHER LONG-INTEGRATION reads
(`source_provenance`, `causal_whatif`, `d6_multiref_wm`) stay byte-identical, so the full `--keys all` batch is
NO-GO. The mechanism is a NON-synaptic, layout-mediated coupling, precisely characterized:

- The default synaptic-input path is `connections.T @ fired_2col`, a TRANSPOSE sparse matmul whose FP summation
  ORDER varies with the matrix layout (total-N / edge interleaving). For a FROZEN-forward read (single settle)
  this stays below the answer margin; a SPIKING-DYNAMICS read integrated over hundreds of steps (pmem's attractor
  hold; d6's slow-NMDA reverberation) AMPLIFIES a single-ULP per-step delta into an EXACTLY-1-spike read
  divergence (measured: d6's `hold_alive_min` shifts by 1/360 with an OTHERWISE byte-identical slice — every
  per-neuron array AND every internal edge weight identical, answer preserved).
- FIXED (merge seam #2): `deterministic_transpose_matvec=True` pins the byte-identical CSR path for this matvec —
  it recovers pmem's byte-identity and does NOT regress the frozen organs (the original 5 read-organs stay 5/5 GO
  with the flag, without pmem). But the flag covers only the MAIN synaptic matvec; the SLOW-NMDA-RECURRENT
  increment (`sim/bridge.py:8973`, `_nr_mat.T @ prev_firing`, used ONLY by d6) and other config-enabled
  conductance matvecs have NO deterministic option, so d6 (and, at N=4968, source_provenance/causal_whatif)
  still tip. Closing the full 7-organ strict batch requires deterministic variants of ALL matvec paths — a
  `sim/` edit, out of this arc's NO-`sim/`-edit scope.

This does NOT invalidate the migration gate for the 5/7 already-closed organs (byte-identical among themselves, 5/5
GO) NOR pmem's own close (byte-identical in co-residence, 6/6 GO). It is a SCALE property of the STRICT gate: the
larger the pool, the more the residual non-determinism in the un-flagged conductance-matvec paths matters for
long-integration spiking reads. The finding is itself a deliverable — it maps where the strict byte-identity gate
meets the engine's FP-determinism floor, and names the exact engine edit that closes it.

## Files changed

- `research/runners/onebrain_merge_framework.py` — `MergedPool.sequence_isolation()` + `_SEQ_EXTRA_STATE`;
  `_base_config` merge seam #2 (`deterministic_transpose_matvec = not legacy`); pmem read plumbing (`_PMEM_CONFIG`,
  `_PMEM_POOL_GAIN`/`_PMEM_READ_PARAMS` + the gained wiring weights, `_pmem_wiring`, `_pmem_organ`/`_PMemReadOrgan`
  with the dt-local reconciliation + the multi-turn form/hold/cue read, `_pmem_reads`/`_pmem_answer`); the pmem
  descriptor gains config/explicit_wiring_fn/organ_cls/read_fn/answer_fn/supports_shared=True; spec_fn switched to
  the lighter base `ProspectiveMemory` with the gained rel recurrence; `GROUP_A_ORGANREAD_DEFERRED` pmem entry
  refined to the scale residual.
- `research/runners/_pmem_intention_latch_derisk.py` — additive `shared=None` on base `ProspectiveMemory` (adopt
  the pool bridge slice; gate the edge installs on `shared is None`; byte-identical when None).
- `research/runners/_pmem_perpool_homeostat_derisk.py`,
  `research/runners/_pmem_sfa_nmda_amplifier_derisk.py` — thread `shared=` through `**kw`; BYPASS the per-seed
  bias/theta module caches when shared (each arm calibrates independently -> genuine byte-identity).

NO `sim/` edit. All runs are tiny numpy nets (N<=4968) on the CPU.
