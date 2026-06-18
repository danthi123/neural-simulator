# Reconsolidation-in-the-loop — emergent feature #2, cheap-first de-risk (2026-06-18, CYCLE 194)

## The emergent claim

Reconsolidation (`OneBrainComposer.update_on_mismatch`: a corrective utterance reactivates a cued
fact and rewrites its patient IN PLACE, prediction-error-gated) is already built + validated as a
single op. The EMERGENT property the integrated PERSISTENT loop now makes demonstrable: when a
correction labilizes + rewrites ONE fact in a SHARED persistent store of many facts, is the update
**ISOLATED** (the other stored facts untouched — no collateral corruption), and does it hold as the
store **fills** (set-size K)? Biological reconsolidation is famously update-specific (Nader 2000;
catalog J.27) — the labilized trace is re-stabilized without erasing neighbours; a naive
shared-memory update would smear. This is the brain-like signature: correct one thing, leave the
rest intact.

This is brain-based: the correction is the composer's neural reactivation + PE-gated in-place
rewrite (reads the cleanup argmax off the spiking substrate); the host only utters the correction.
NO `sim/` edit; reuse-by-import (`update_on_mismatch` / `_calibrate_pe_labile` / `count_facts` /
`query_patient`). The store is **block-major** (each fact in its own (1+D) trigger→readout block in
`store_conns`), so isolation is STRUCTURAL — this de-risk confirms it holds end-to-end in the live
loop (including the BATCHED parallel read that fires all triggers at once) and across set-size.

## GO bar + anti-cheat controls

Store K facts; then correct EACH fact in turn (a genuine patient mismatch) in the live persistent
store, measuring:
- **REWRITE**: the corrected fact recalls its NEW patient (the in-place rewrite worked).
- **NO DUPLICATE**: `count_facts(agent, action) == 1` after each correction (not a contradictory append).
- **ISOLATION** (the load-bearing emergent claim): after correcting fact i, EVERY OTHER fact still
  recalls its current patient — collateral-damage rate ~0.
- **Control — restabilize**: a same-patient "correction" RE-STABILIZES (PE below the calibrated
  labilization gate → no spurious rewrite, count stays 1).
- **Control — moat**: a never-stored correction ABSTAINS (the no-confab moat — a missing trace is not
  fabricated).
- The PE gate is calibrated from the data (`_calibrate_pe_labile`, the same-vs-different midpoint over
  the current facts), NOT tuned to a downstream probe.
- **SET-SIZE**: all of the above hold as K grows (8 → 16 → 24).

An honest NEGATIVE (a correction smears neighbours, or reconsolidation breaks as K grows) maps a real
boundary of the shared persistent store + motivates the fix — itself the deliverable.

## Results

Full multi-seed × set-size matrix (seeds 42/43/44 × K=8/16/24) — **9/9 (seed × K) GO**, every
metric perfect at every cell:

| set-size K | seeds GO | rewrite | no-duplicate | ISOLATION | restabilize | moat |
|------------|----------|---------|--------------|-----------|-------------|------|
| 8          | 3/3      | 8/8     | 8/8          | **8/8**   | ✓           | ✓    |
| 16         | 3/3      | 16/16   | 16/16        | **16/16** | ✓           | ✓    |
| 24         | 3/3      | 24/24   | 24/24        | **24/24** | ✓           | ✓    |

Across all 9 cells: every correction rewrote the cued fact to its new patient in place, never
created a contradictory duplicate, and left EVERY other stored fact's recall untouched (isolation
perfect); the same-patient control re-stabilized (PE below the calibrated gate → no spurious
rewrite) and the never-stored control abstained (the moat held). The result is clean and scales —
the confab/leak nuance of feature #1 does not appear here because reconsolidation does not push the
substrate to its noise floor; it operates on intact, well-separated traces.

### Verdict — GO (a clean emergent property)

Update-specific reconsolidation — correct one fact in a shared persistent store, leave the rest
intact, no duplicates, holding as the store fills — is demonstrated end-to-end as an emergent
property of the live integrated one-brain loop (through the BATCHED parallel read + the PE-gated
neural rewrite + across set-size to K=24). The isolation is structurally grounded (the block-major
complex-synapse store) and confirmed to hold in vivo. This is the brain-like update-specificity of
biological reconsolidation (Nader 2000; catalog J.27), on the spiking substrate, no `sim/` edit.

## Reproduce

```bash
SIM_BACKEND=cupy python -m research.runners._emergent_reconsolidation_in_loop_derisk \
    --seeds 42 43 44 --K 8 16 24 --out research/findings/raw/_emergent_reconsolidation_in_loop.json
```
Runner: `research/runners/_emergent_reconsolidation_in_loop_derisk.py`. Scoping:
`2026-06-18-emergent-one-brain-features-research.md`.
