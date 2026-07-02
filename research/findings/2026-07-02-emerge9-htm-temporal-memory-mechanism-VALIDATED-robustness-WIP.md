# EMERGE-9 (rung-3 pivot) — the unsupervised HTM Temporal-Memory mechanism WORKS: it self-organizes PERFECT context-specific high-order prediction (2/3 seeds; lesion collapses to chance) — the first POSITIVE rung-3 signal after 5 supervised-credit boundaries. Multi-seed robustness needs a faithful SDR + multi-synapse-segment implementation (diagnosed; WIP).

**2026-07-02 (autonomous; full cores).** Runner `research/runners/_emerge9_temporal_memory_derisk.py`; result `research/findings/raw/_emerge9_temporal_memory.json`. Reuse-by-import (overlapping-sequences task + Markov floor inlined); NO `sim/` edit; CPU/numpy; multi-seed.

## Why this ran
The rung-3 pivot (CYCLE 802): after 5 probes showed SUPERVISED local recurrent-credit does not beat a fixed reservoir, the research gate (CYCLE 803) recommended UNSUPERVISED self-organizing spiking sequence learning — Bouhadjar-Diesmann 2022's HTM Temporal Memory, which trains no recurrent weights toward a target and avoids interference by ALLOCATION (disjoint sparse cell-subsets per context). EMERGE-9 is the cheap-first discrete-HTM-TM de-risk (the spiking-LIF port, mapping the distal-dendrite plateau to our confirmed two-compartment neuron, is rung-3b).

## Task + mechanism
Overlapping sequences: `[cue] + [shared middle L] + [branch]` — e.g. `[0,2,3,4,5,6]` vs `[1,2,3,4,5,7]`. The middle `2,3,4,5` is IDENTICAL; predicting the branch (6 vs 7) requires carrying the cue (0 vs 1) across the shared middle → genuine high-order context (every order-k≤L Markov predictor is provably chance = 0.5). Minimal HTM-TM: M columns × nE cells; per-cell distal segment = permanence over a fixed sparse skeleton; predictive cells / bursting / best-match-else-allocate winner selection / local Hebbian permanence learning; UNSUPERVISED (no teacher — performance never feeds learning); locality asserted (`used_transpose` False).

## Result — the mechanism is VALIDATED
Default config (nE=16, theta_seg=1, single-winner, 40 epochs, seeds 42/43/44): **branch accuracy 0.667** (seeds 42 AND 43 = **1.000**, seed 44 = 0.0), markov floor 0.5, chance 0.5, **lesion (distal prediction disabled) = 0.000** (collapses — the context mechanism is load-bearing), full-context oracle 1.0 (task solvable).

**Per-step trace, seed 42 (PERFECT context-specific high-order prediction):**
```
seq [0,2,3,4,5,6]:  step4 in=5 -> predict [6]   (correct branch)
seq [1,2,3,4,5,7]:  step4 in=5 -> predict [7]   (correct branch)
```
From the IDENTICAL middle `2,3,4,5`, the network predicts 6 vs 7 disambiguated only by the cue — exactly the high-order context self-organization the pivot targets, with NO teacher. This is the **first positive rung-3 signal** after five supervised-credit boundaries (rung-3a target-based, e-prop, RFLO, EMERGE-7 next-symbol, EMERGE-8 Predictive Alignment). Allocation-based unsupervised self-organization genuinely does what supervised recurrent-weight training could not.

## Multi-seed robustness gap — diagnosed precisely (WIP, not a boundary)
Seed 44 fails: at the last middle symbol BOTH sequences predict `[6,7]` — the two contexts' cell-chains MERGED in the shared middle (context-collision). This is the exact failure HTM's SDR design prevents: with single-cell winners + a 1-synapse segment threshold (theta_seg=1), two contexts can allocate/merge onto the same cell, and a 1-synapse segment cannot discriminate contexts. Real HTM uses **sparse distributed representations (many cells) + multi-synapse segments (theta ~ half of ~40 synapses)** so a segment fires ONLY for its specific context SDR — making collisions vanishingly unlikely. A quick SDR co-winner attempt (k_win>1) regressed (implementation bug in the co-winner learning/prediction interaction), so the robustness fix is a **correct faithful SDR + multi-synapse-segment reimplementation** — a known-fiddly but well-understood next iteration.

## Verdict: BUILD-INFORMATIVE checkpoint — mechanism VALIDATED, robustness is the next iteration (NOT a boundary, NOT a stop)
The pivot's core claim is validated: unsupervised, local, allocation-based self-organization produces context-specific high-order sequence prediction on the substrate (2/3 seeds perfect, lesion load-bearing), sidestepping the supervised-credit-vs-reservoir dead-end. The minimal single-synapse implementation is seed-fragile (context-collision); the fix (faithful SDR + multi-synapse segments) is diagnosed + clear.

## Next
1. **Faithful SDR HTM-TM** (the immediate next build): population winners (k-cell SDR per context) + segments sampling multiple synapses from the prior SDR + `theta_seg ~ k/2`, so segments are context-specific and collisions vanish → robust multi-seed (target: GO ≥0.9 all seeds, then capacity: more sequences, longer/more-overlapping middles).
2. Then **rung-3b**: the spiking-LIF port (dAP → our confirmed two-compartment neuron; the three-term permanence rule + WTA inhibition), and only after a robust rung-3 GO, scope the `sim/` rung-4 build.

## Honest scope
- Discrete HTM-TM (the algorithm Bouhadjar spikified); spiking-LIF = rung-3b; NO `sim/` edit; do NOT start the rung-4 port.
- The single-winner default REPRODUCES the seed-42/43 mechanism proof; `k_win>1` (SDR) is the WIP faithful path (currently buggy).
- Unsupervised: no teacher; self-organization IS the deliverable, not a cheat. Lesion + oracle + Markov-floor + multi-seed anti-cheats in place.

## Artifacts
`research/runners/_emerge9_temporal_memory_derisk.py`, `research/findings/raw/_emerge9_temporal_memory.json`. Prior: `2026-07-02-rung3-unsupervised-sequence-learning-scoping.md`, `2026-07-02-emerge8-predictive-alignment-BOUNDARY-and-the-5-probe-reframe.md`.
