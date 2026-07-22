# gap#5 RANK 2 — the verbatim `_prepare` reuse RULES OUT the encode as the divergence: byte-faithful encode (w_within EXACTLY 27.4 == RANK 1) yet STILL 0 reactivation ⇒ a deeply-elusive rest-phase boundary. Reverted; gap#5's solid results stand.

**2026-07-22, CPU/numpy.** Final honest verdict on RANK 2 (ordered sequence replay) within-reactivation, after
exhausting the tractable diagnostic + fix paths.

## What was tried (both robust fixes) — BOTH FAIL
1. **settle=3→0** (the assembly-boundary silence-steps that starved the bistable latch): raised w_within 5.0→30.3.
   Necessary, not sufficient — still 0 reactivation.
2. **VERBATIM `_prepare` reuse** (refactor RANK 2's `_prepare_sequence` to CALL RANK 1's proven `_prepare` for the
   within-encode + recall setup, via an `assembly_fn`/`chain_callback` generalization, instead of reimplementing it):
   made the encode BYTE-FAITHFUL — **w_within = EXACTLY 27.4 == RANK 1's 27.4** — yet the n_mem=1 assembly STILL does
   not reactivate: **events=0, asm_active=[0]**, rest population rate 0.048 (RANK 1: ~0.06 with 3-6 events, memb~0.31).

## What this RULES OUT (the value of the negative)
The divergence is **NOT** in: (a) the within-encode (now byte-faithful, w_within EXACTLY RANK 1's); (b) the assembly
cells (same rng seed*17+3, same 240 cells for n_mem=1); (c) the post-encode handling (`_prepare_sequence` after
`_prepare` is READ-ONLY weight-extraction diagnostics — no bridge modification); (d) the rest-phase noise
(`_rest_and_detect` re-seeds its poisson noise deterministically from `seed`, `default_rng(seed*100003+11)` — IDENTICAL
for RANK 1 and RANK 2); (e) the config (an earlier bridge-diff: 0 scalar config diffs, same recall threshold/gates);
(f) the substrate seed (`_build` sets `cfg.seed`). ⇒ A genuine CONTRADICTION: byte-identical bridge + deterministic
identical noise, yet different rest dynamics (RANK 2 less active → no reactivation). This cannot be resolved by reading;
it points to a subtle rest-phase substrate-interaction / global-state effect that the next step (a per-step rest-phase
state trace comparing RANK 1 vs RANK 2 step-by-step) would isolate — a substantial deep diagnostic.

## Verdict + honest scope (per THE LAW: a wall is a METHOD verdict, not a capability abandonment)
RANK 2 within-reactivation is a **deeply-narrowed, characterized boundary** — the tractable fix paths (settle, verbatim
reuse, config/cells/noise-rng) are exhausted; the residual is a per-step-trace-only diagnostic. The verbatim-reuse
refactor (which modified RANK 1's PROVEN driver without fixing RANK 2) was **REVERTED** — both drivers restored to the
committed proven state (RANK 1's 6-seed-GO driver unmodified; RANK 2's committed parked state). **gap#5's solid results
STAND:** the COMPLETION mechanism is CLOSED (intrinsic dendritic bistability, 6-seed) and RANK 1 spontaneous
single-assembly reactivation is 6-seed GO — the imagination line's first rung is solid. RANK 2 (ordered SEQUENCE replay)
is the harder second rung, parked at this deeply-narrowed rest-phase boundary. NEXT (if revisited): the per-step
rest-phase state trace to isolate the global-state divergence. Evidence: `raw/gap5_r4/rank2_nmem1_verbatim.log`
(w_within 27.4, asm_active=[0]).
