# gap#5 RANK 2 — the within-reactivation blocker is NARROWED to an elusive encode divergence (bridges config-identical, same cells, R2 within-weight HIGHER, yet no reactivation); PARKED for a verbatim `_prepare` reuse

**2026-07-22.** Continuing RANK 2 (sequence replay). The forward-chain mechanism WORKS (asym +2.53, chain_rev=0 —
prior finding). The blocker is the within-reactivation: RANK 2's assemblies don't spontaneously reactivate while
RANK 1's do. Progress + honest park:

## What was fixed (real progress)
The subagent's diagnosis was correct + necessary: RANK 2's within-encode ran `_silence_soma_apical(settle=3)` at the
assembly boundary — those 3 settling SIM STEPS starved the following assembly's bistable within-latch. **Fix `settle=0`
raised w_within 5.0 → 30.3 (matching RANK 1's real 27.4).** (I'd wrongly assumed both were ~5.0; RANK 1's is ~27.)

## Why it's still blocked — narrowed but elusive
Even with the within weight matched, n_mem=1 still gives 0 reactivation (asm_active=[0]). A direct bridge-state DIFF
(`scratchpad/rank2_bridgediff.py`, both built n_mem=1 seed 42) shows the two bridges are **essentially identical**:
- **0 scalar config differences** (core_config identical);
- same `coincidence_k_threshold` (40), same `cp_plasticity_rate_gain` (mean 0.834), same `cp_transmission_gain` (None);
- same within-edge count (2842), same assembly size (240);
- **same assembly cells** — both draw `default_rng(seed*17+3).choice(ca3_idx, 240)` (for n_mem=1 the disjoint-pool
  draw is byte-identical to RANK 1's loop draw);
- substrate IS seeded (`_build` sets `cfg.seed=seed`, line 172 — NOT the 2026-07-17 unseeded gotcha).
- **R2's within weight is HIGHER** (30.3 vs 27.4) — so it is NOT a within-strength deficit.
The only measured residue: w_within differs slightly (27.4 vs 30.3) despite same cells + `enable_ou=False`, which means
the ENCODE DYNAMICS differ subtly (RANK 2 does an extra full `_silence_soma_apical` clear at the assembly boundary that
RANK 1's `_prepare` does not) — a small difference that shifts the weight distribution but should not, on its face,
abolish reactivation. The divergence that actually kills reactivation is NOT captured by the config/weight-mean/cells
comparison.

## Honest verdict + next step
The reimplemented `_prepare_sequence` diverges from RANK 1's proven `_prepare` in a way that survives matching cells,
config, substrate-seed, and within-weight. Rather than keep chasing a sub-threshold difference, the robust fix is a
**verbatim reuse of RANK 1's `_prepare` within/recall path** (build the bistable self-completing assembly with RANK 1's
exact code, then layer ONLY the disjoint multi-assembly draw + chain phase on top) — a moderate refactor.
**PARKED as a characterized follow-on.** gap#5-COMPLETION is CLOSED (intrinsic dendritic bistability) and RANK 1
single-assembly spontaneous reactivation is **6-seed GO** — the imagination line's first rung is solid; RANK 2 ordered
sequence-replay is the hard second rung, parked at this narrowed divergence. The `settle=0` fix + `--ca3-density`/
`--structural-sep` flags are committed (real improvements). NO `sim/` edit. Diagnostic: `scratchpad/rank2_bridgediff.py`.
