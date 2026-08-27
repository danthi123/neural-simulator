---
status: live
type: finding
lane: integration
date: 2026-08-27
---

# Deterministic conductance-matvec hardening (nmda_recurrent + gabab + graded_plateau) — and the REAL one-brain full-7 `--keys all` wall is co-residence-dependent nmda_slow ROUTING, not matvec summation-order

> **⚠️ ATTRIBUTION REFINED 2026-08-27 by [the dedup finding](2026-08-27-dedup-synapse-masks-closes-onebrain-full7-d6-nmda-slow.md).**
> This finding correctly localized the wall to `nmda_slow` ROUTING (not matvec summation-order — that part stands), but its
> DEEPER attribution — a co-residence-dependent `nmda_slow` density-0.9 RNG SUBSET DRAW — is REFUTED by direct reproduction:
> d6 built ALONE has a perfectly aligned `nmda_slow` mask (0/10819 misaligned), so the DRAW is fine.
> The real mechanism is a duplicate-edge per-synapse MASK MISALIGNMENT (`keyed` pre-dedup vs `cp_connections`
> post-`sum_duplicates`), closed by `cfg.dedup_synapse_masks` (d6 NO-GO -> GO 6/6, merged `b2228616`).
> Read the dedup finding for the corrected mechanism; `prospective_memory` remains a SEPARATE FP-accumulation seam.

Status: the sim/ hardening is VERIFIED CORRECT on its own bar — additive, guarded, byte-identical when off
(hash-verified in the data). The full-7 strict `onebrain_merge_verify --keys all` gate remains NO-GO, and the
hardening does NOT change that: this finding REFUTES the hypothesis that the remaining conductance-matvec paths
cause the NO-GO. On the numpy backend the batch runs on, those matvecs are ALREADY co-residence-invariant, so
hardening them changes the read divergence by exactly zero. The actual wall is a co-residence-DEPENDENT effective
connectivity (the `nmda_slow` receptor routing), localized precisely below.

## What shipped (the sim/ edit) — completes the flag's determinism contract on cupy

`cfg.deterministic_transpose_matvec` (default OFF) already pinned the MAIN synaptic transpose matvec and the
COINCIDENCE matvec to the segmented `add.reduceat` reduction (`sim.bridge._deterministic_csr_matvec`, no atomics ->
bit-identical every call; the 2026-08-25 fix). But three other conductance transpose matvecs were NOT covered:

- the SLOW-NMDA-RECURRENT increment (`_nr_mat.T @ prev_firing`, used by `enable_nmda_recurrent`) had NO
  deterministic branch at all;
- the GABA_B (`_gb_mat.T @ prev`) and GRADED-DENDRITIC-PLATEAU (`_gp_mat.T @ prev`) branches used only a bare
  `.tocsr()` before the library `@` — which the 2026-08-25 finding proved is STILL a cuSPARSE csrmv and
  bit-non-reproducible run-to-run on cupy (the flag was silently LYING about determinism for these paths).

All three now route through `_deterministic_csr_matvec(mat.T.tocsr(), prev)` under the SAME flag, matching the
main/coincidence paths. This is a genuine, correct completion of the flag's cupy-determinism contract (its VALUE is
on cupy, where cuSPARSE uses atomics; on numpy the library reductions are already deterministic). It is NOT a fix
for the batch (see below).

The GPU-only read-only step-megakernel matvec was deliberately LEFT unchanged: its dispatch is `is_gpu_backend()`-gated
(`sim/bridge.py:7958`), so it never runs on numpy and its flag-off byte-identity cannot be RUN-verified while the GPU
is busy. Shipping an unverifiable core-matvec change would violate the rigor bar; it is a scoped cupy-verified
follow-up (its bare `.tocsr() @ fired_2col` two-column multiply is the same latent cupy non-determinism).

## Flag-OFF byte-identity (the critical safety property) — AIRTIGHT

The edits are pure extract-to-variable refactors when the flag is OFF (the flag-off expression is the unchanged
`mat.T @ prev`). Verified three ways on numpy, EDITED (worktree) vs ORIGINAL (main):

- A 150-step net with EVERY edited path live (nmda_recurrent + gabab + coincidence + graded_dendritic_plateau +
  main; all synapse masks present), flag OFF, full conductance-state SHA256: `db03fc14…c5e0` IDENTICAL edited vs
  original.
- A 200-step nmda_recurrent + main net, flag OFF, full-state SHA256: `3c4f18b9…be38` IDENTICAL edited vs original.
- `tests/test_determinism.py` (numpy): 9 passed, 2 skipped (cupy-only), unchanged.

Artifacts: `research/findings/raw/2026-08-27-deterministic-conductance-matvec-hardening-evidence.json` +
`research/findings/raw/2026-08-27-onebrain-merge-verify-6seed-after-matvec-hardening.json`.

## Why the batch does NOT close — the real wall, localized

Baseline and post-edit `--keys all` (numpy) read the SAME: 5/7 GO; `prospective_memory` (read_maxerr 1/15, answer
FLIPS) and `d6_multiref_wm` (read_maxerr 1/360, answer preserved) NO-GO. read_maxerr is byte-for-byte UNCHANGED by
the hardening — the first proof the matvec paths are not the cause on numpy. Six seeds (42,43,44,100,101,102)
confirm the same NO-GO (pmem fails every seed).

Rigorous localization (probes in the evidence JSON):

1. The reads are REPRODUCIBLE run-to-run on a rebuilt merged pool (maxdiff 0.0) — the divergence is deterministic
   CO-RESIDENCE (merged-vs-coresident), not an unseeded host RNG.
2. On numpy the transpose matvecs are co-residence-INVARIANT: pmem's and d6's RAW main matvec, and pmem's
   AMPA-suppressed matvec, all read max|merged-core| = 0.0 for identical firing. Neither organ has cross-edges.
3. The divergence enters at the FIRST densely-firing step (d6 step 8, pmem step 7): with prev_firing and external
   current byte-identical, `cp_conductance_g_e` (and g_i, g_nmda, g_nmda_recurrent) diverge — i.e. the
   conductance ASSEMBLY differs, not the neuron dynamics.
4. For d6 the source is exact: its within-slice edge SET is identical merged-vs-core (27935 pairs), but the
   `nmda_slow` receptor TAG differs — 10819 edges tagged in each arm, of which only 5671 AGREE (5148 tagged in
   merged-only, 5148 in core-only). So the AMPA-suppressed matvec diverges by 225 while the raw matvec is 0.0. The
   effective connectivity (which synapses are slow-NMDA vs AMPA) is co-residence-dependent.
5. The trigger is `enable_nmda_recurrent`, unioned into the pool config by d6. Without it (the earlier N=3442
   pmem+self_schema+comprehension pool) pmem is byte-identical; adding d6 unions the flag, which activates the
   pool-wide AMPA-suppression rebuild whose `nmda_slow` mask is co-residence-dependent.

Mechanism: d6's slow-NMDA self-excitation is wired at `density=0.9` (`build_persistent_slot` in
`_d3_persistent_slot_derisk.py`) — a RANDOM subset draw. In the merged build the connectivity RNG is at a different
point (other organs drawn first), so a DIFFERENT subset of `w_k -> w_k` synapses is created/tagged `nmda_slow`.
This is exactly the co-residence-dependent global-RNG class the framework already documented and reconciled for
NOISE (onebrain_merge_framework.py, "enable_conductance_noise + enable_ou_process draw from a SINGLE global RNG
stream in neuron-index order … co-residence-DEPENDENT"), but which was MISSED for the connectivity/receptor draw.

HONESTY on pmem: d6 is nailed exactly (the `nmda_slow` tag divergence IS the effective-connectivity difference).
pmem's TRIGGER is nailed (it is byte-identical without `enable_nmda_recurrent`; it tips only once d6 unions the
flag), but its exact in-step coupling is NOT fully isolated: pmem carries NO `nmda_slow` synapses, its raw and
AMPA-suppressed matvecs read 0.0 merged-vs-core in isolation, and it has no cross-edges — yet its `cp_conductance_g_e`
diverges at step 7 in the full step. So pmem rides the SAME `enable_nmda_recurrent`-triggered wall, but the precise
numerical path for the `nmda_slow`-free pmem slice remains a residual puzzle to close alongside d6's routing.

## Correction of the prior finding

`2026-08-27-onebrain-merge-framework-multiturn-stateful-read.md` attributed the full-7 NO-GO to "the un-flagged
conductance-matvec paths" and a transpose-matvec "FP summation ORDER [that] varies with the matrix layout",
recommending deterministic variants of ALL matvec paths as the fix. That mechanism is a cupy-only cuSPARSE-atomic
effect; on numpy (where the strict batch runs) the matvecs are already deterministic, so the recommended fix is a
no-op there — confirmed: hardening them leaves read_maxerr unchanged. The prior finding checked per-neuron arrays
and edge WEIGHTS (identical) but not the receptor TAGS, which is where the real co-residence lives.

## The real fix (out of this arc's matvec scope; named for the next arc)

Make d6's `nmda_slow` effective connectivity co-residence-invariant — per-region-seed the `density=0.9`
self-excitation draw / receptor assignment (the same seam pattern already used for per-region threshold
heterogeneity and for reconciling noise OFF), so the SAME edges are tagged `nmda_slow` regardless of pool offset.
pmem is byte-identical once `enable_nmda_recurrent` is not unioned in, so closing d6's routing is expected to close
pmem's trigger too; that is the gate to re-run. This is a wiring/framework fix, not a conductance-matvec edit.

## Files changed

- `sim/bridge.py` — nmda_recurrent (~9086), gabab (~9326), graded_dendritic_plateau (~9276) conductance transpose
  matvecs gated to `_deterministic_csr_matvec` under `cfg.deterministic_transpose_matvec` (default OFF ->
  byte-identical). No other behavior touched; the megakernel matvec is unchanged.
