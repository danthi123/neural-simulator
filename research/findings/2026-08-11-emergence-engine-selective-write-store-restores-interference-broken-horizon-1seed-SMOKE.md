---
type: finding
status: contributing
date: 2026-08-11
mechanism: emergence-engine-selective-write-content-addressable-store-over-htm-allocation-keys
lane: emergence engine (recurrent spiking sequence/language cortex; roadmap L130 "scale spiking HTM Temporal-Memory generator")
instrument: research/runners/_emerge_selective_write_store_derisk.py — a content-addressable heteroassociative store keyed on the on-bridge HTM-TM's ALLOCATION SDRs (EMERGE-14, reuse-by-import), with SELECTIVE (mismatch/novelty-gated) vs ALWAYS write, a store-read LESION, a PERMUTE-KEYS attribution control, SWAP-FOLLOWS-CONTEXT, the n-gram floor + oracle; swept across allocation CAPACITY at the dist-17 interference break; 1-seed SMOKE (decisive 6-seed command below)
artifacts:
  - research/findings/raw/_emerge_selective_write_store/smoke_nseq3_L16_cells10_selectivity_seed42.json
  - research/findings/raw/_emerge_selective_write_store/smoke_nseq3_L16-24_fair_seed42.json
  - research/findings/raw/_emerge_selective_write_store/smoke_nseq3_L16_seed42.json
  - research/findings/raw/_emerge_selective_write_store/smoke_nseq3_L16_cells8_capwall_seed42.json
  - research/runners/_emerge_selective_write_store_derisk.py
---

# Emergence engine — a SELECTIVE-WRITE content-addressable store over the on-bridge HTM-TM's own (clean) ALLOCATION KEYS RESTORES the interference-broken horizon: at dist 17 (n_seq=3) the bare HTM chain breaks to 0.667 but the store recovers to 1.000, and the SELECTIVE-WRITE GATE is LOAD-BEARING exactly in the partial-allocation-merge regime (store 1.000 vs always-write 0.333). 1-seed SMOKE + the exact 6-seed command. <!--derived--> (0.667=2/3 and 0.333=1/3 are rounded displays of the full-precision values in the cited artifacts)

## Why this (our-own-record first — the named surpass of the horizon residual)
<!--derived-->
`2026-08-11-emergence-engine-htm-horizon-nonfading-but-finite-...SMOKE` measured the on-bridge HTM-TM horizon (EMERGE-14):
it learns high-order structure (clean HOLD at dist 9, htm 1.000, dAP-lesion 0.000, recurrence load-bearing), is NON-FADING
but FINITE — it carries a distal cue to dist ~17 at LOW interference, but under interference (n_seq>=3) ONE context's
priming CHAIN BREAKS by dist 17 (htm 0.667/0.750; swap tracks htm, so the failure is a MERGED chain genuinely following
the wrong cue, not a readout artifact). The residual it named — verbatim — was a **SELECTIVE-WRITE content-addressable
store over the HTM-TM's own (clean) ALLOCATION KEYS, so a broken/ambiguous priming chain is recoverable by
content-addressed completion**. This unifies the two banked threads (the delta/STP content store, which extended a horizon
ONLY given clean keys; the reservoir's keys were diffuse -> NEGATIVE) on the substrate the roadmap wants to scale: the
HTM-TM's allocation cells ARE clean/allocated, the exact key-quality the reservoir lacked.

## Mechanism (reuse-by-import of EMERGE-14; NO `sim/` edit)
KEYS = the HTM-TM's ALLOCATION SDRs at the SHARED-MIDDLE positions (t in 1..L) of the priming chain — the primed,
context-specific winner-cell subsets the bridge's coincidence recurrence produces. NOT the raw cue input at t=0 (keying on
that is a trivial cue->branch input lookup that bypasses the HTM; the keys are the HTM's OWN high-order allocation cells).
SELECTIVE WRITE (novelty/mismatch-gated): on each CONFIDENT step (chain intact -> `active` is a primed subset, not the
whole-column burst) WRITE (allocation-SDR -> branch); if a key already maps to a DIFFERENT branch it is a COLLISION (a
merged/ambiguous key) -> POISON it (remove + never re-add). READ/completion: content-address the store with the test
traversal's confident allocation SDRs, prefer the highest-overlap match (ties -> the fresher, branch-proximal position);
UNAMBIGUOUS -> emit its branch (a broken relay recovered from an earlier clean allocation key); AMBIGUOUS -> veto -> fall
back to the bare HTM. `SIM_BACKEND=numpy` (the horizon finding established these sub-1k-neuron coincidence loops are
launch-bound: cupy is slower; CPU is correct + faster).

## Result (1-seed SMOKE, seed 42, n_seq=3, chance 0.333; store/bare = branch-prediction accuracy, higher = better) <!--derived-->
<!--derived-->
(All 0.667/0.333 below are rounded 2/3 and 1/3 displays of the full-precision values in the cited raw JSON; 1.000/0.000 and the integer key counts are literal. This section is block-scoped derived to the next heading.)

Raw (seed 42, numpy/CPU): regime B `research/findings/raw/_emerge_selective_write_store/smoke_nseq3_L16_cells10_selectivity_seed42.json`;
regime A `research/findings/raw/_emerge_selective_write_store/smoke_nseq3_L16-24_fair_seed42.json` (dist 17+25) and
`research/findings/raw/_emerge_selective_write_store/smoke_nseq3_L16_seed42.json` (dist 17); regime C
`research/findings/raw/_emerge_selective_write_store/smoke_nseq3_L16_cells8_capwall_seed42.json`.
The store restores the horizon wherever CLEAN allocation keys survive; the SELECTIVE gate is load-bearing exactly where the
allocation PARTIALLY merges. Three capacity regimes (fair capacity = `k_win*n_seq + slack`; k_win=4, n_seq=3 -> min 12):

| regime | n_cells | dist | bare | STORE | store-lesion | always-write | permute | swap | sel_keys (poisoned) | always ambiguous |
|---|---|---|---|---|---|---|---|---|---|---|
| A · fair (clean keys) | 20 | 17 | 0.667 | **1.000** | 0.667 | 1.000 | 0.000 | 1.000 | 48 (0) | 0 |
| A · fair, farther | 20 | 25 | 0.667 | **1.000** | 0.667 | 1.000 | 0.000 | 1.000 | 72 (0) | 0 |
| B · partial merge | 10 | 17 | 0.333 | **1.000** | 0.333 | **0.333** | 0.000 | 1.000 | 18 (15) | 15 |
| C · full starvation | 8 | 17 | 0.333 | 0.333 | 0.333 | 0.333 | 0.000 | 0.333 | 16 (16) | 16 |

**Regime A (fair capacity, the horizon-finding's GO-question point).** The bare HTM chain breaks to 0.667 at dist 17 AND
holds at 0.667 out to dist 25; the store RESTORES it to **1.000 at both distances** (non-distance-limited). LOAD-BEARING
(store-read lesion -> 0.667, back to bare), ATTRIBUTABLE (permute the keys' branch labels by a derangement -> 0.000 <=
chance), CONTEXT-DRIVEN (swap-follows-context 1.000 — the completion follows the INJECTED cue, no confabulation). BUT the
SELECTIVE gate is INERT here: always-write is also 1.000 with ZERO ambiguous keys. Honest reading (the task's anti-cheat
(b)): at fair capacity the interference break is a PREDICTION-chain break with the ALLOCATION KEYS STILL CLEAN/DISJOINT, so
a plain content store over them suffices — "a bigger memory", not the selective mechanism.

**Regime B (mild capacity pressure, n_cells=10 < 12 — where the SELECTIVE-WRITE gate EARNS its keep).** The allocation now
PARTIALLY merges: 15 keys collide (ambiguous) but 18 clean keys survive. bare 0.333, and the SELECTIVE store recovers to
**1.000** by POISONING the 15 merged keys and completing from the 18 clean ones — while ALWAYS-WRITE is trapped at 0.333
(= bare): its ambiguous merged keys capture the fresher/branch-proximal read and veto it. All four anti-cheats bite:
load-bearing (lesion 0.333), selectivity GATES (store 1.000 − always 0.333 = 0.667), attributable (permute 0.000),
context-driven (swap 1.000). Runner verdict: **SMOKE-GO** (1-seed indicator).

**Regime C (full starvation, n_cells=8 << 12 — the SEPARATE capacity wall the horizon finding named).** The allocation
FULLY merges: all 16 keys are ambiguous; the selective gate poisons every one -> the store is empty -> collapse to chance
(0.333), same as bare/always. When NO clean allocation key survives, a content store cannot recover — the surpass there is
the allocation/capacity mechanism (homeostatic/competitive allocation), NOT the store. This is a boundary, not a GO.

## Verdict — the store restores the horizon; the selective gate is load-bearing in the partial-merge regime (1-seed SMOKE)
<!--derived-->
The GO question — "does the store restore the horizon under interference, htm toward 1.0 at dist>=17 with n_seq>=3 where
bare drops to ~0.67" — is answered **yes** on 1 seed: bare 0.667 -> store 1.000 at dist 17 AND dist 25, load-bearing +
attributable + context-driven. The NAMED mechanism (SELECTIVE write) is genuinely load-bearing (store 1.000 vs
always-write 0.333) exactly in the biologically-realistic partial-interference regime (n_cells=10), where it must DISCARD
the merged/ambiguous allocation keys to complete from the clean survivors. Two honest boundaries frame it: at fully-clean
(fair) capacity a plain content store suffices (selectivity inert); at fully-starved capacity no store can help (the
separate allocation-capacity wall). 1-seed is a SMOKE indicator; the decisive run is the 6-seed sweep.

## The decisive 6-seed command (CPU/numpy, NOT cupy; do NOT run the sweep here)
Two invocations. (1) the NAMED-mechanism GO: selectivity load-bearing at partial-allocation-merge; (2) the
horizon-restoration + non-distance-limited point at fair capacity (documents that selectivity is inert there).

```bash
OUTDIR=research/findings/raw/_emerge_selective_write_store; EXT=.json

# (1) SELECTIVITY-LOAD-BEARING GO — partial allocation merge (n_cells = k_win*n_seq + slack = 12 + (-2) = 10)
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._emerge_selective_write_store_derisk \
  --seeds 42 43 44 100 101 102 --n-seq 3 --distances 16 --epochs 35 --slack=-2 \
  --out "$OUTDIR/sel_nseq3_L16_cells10_6seed$EXT"

# (2) HORIZON-RESTORATION at FAIR capacity (n_cells=20), non-distance-limited (dist 17 + 25)
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._emerge_selective_write_store_derisk \
  --seeds 42 43 44 100 101 102 --n-seq 3 --distances 16 24 --epochs 35 --slack 8 \
  --out "$OUTDIR/fair_nseq3_L16-24_6seed$EXT"
```

Each point is ~30-60 s of CPU per seed x 6 arms; the coincidence loop is launch-bound, so the coordinator may parallelise
across the two invocations (and across seeds) via the pool. GO (1) = store restores AND all four anti-cheats hold 6/6
(selective > always by >= 0.15). GO (2) = store restores at dist 17 AND 25 6/6 (selectivity is expected inert — the honest
"plain-store-suffices-at-clean-capacity" boundary).

## NEXT
1. Run the 6-seed sweep above -> the multi-seed selectivity-load-bearing GO + the fair-capacity restoration surface.
2. The remaining boundary is Regime C (full allocation starvation): a content store cannot rescue a fully-merged
   allocation. The surpass is the allocation/capacity mechanism itself — a homeostatic/competitive (heterosynaptic-LTD)
   allocation that keeps allocation keys disjoint under capacity pressure, feeding the selective store the clean keys it
   needs. That is the deeper wall the horizon finding flagged as "capacity is a SEPARATE, harder wall".

Reuse-by-import (EMERGE-14 machinery); NO `sim/` edit. 1-seed SMOKE; the 6-seed sweep is the decisive run.
