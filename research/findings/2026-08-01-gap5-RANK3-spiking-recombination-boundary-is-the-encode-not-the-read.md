---
type: finding
status: contributing
date: 2026-08-01
mechanism: recombinative-replay
lane: H-gap5
---

# gap#5 RANK 3 (imagination) — the FULL phase-gated SPIKING read does NOT recover the shared-hub recombination the mean-matrix proxy lost: the per-cell potentiated subset carries NO learned-successor concentration (ratio ≤ 1), so the boundary is the ENCODE, not the READ

**One-line:** The prior RANK-3 finding
(`2026-08-01-gap5-RANK3-gamma-organized-recombination-extracted-matrix-proxy-sits-at-chance.md`) sat at the 2/3 geometric
chance and hypothesised the cause was the READ — the **extracted MEAN** transition matrix averaged away a **potentiated
synapse SUBSET** that a full spiking postsynaptic-threshold read would recover. This builds that named next method and
**REFUTES the hypothesis**: a per-cell instrument on the REAL `cp_connections` shows the hub's out-drive onto the LEARNED
successors (C, Y) is **no more concentrated** than onto the UNLEARNED out-edges (A, X) — mean / top-k / max ratio all
**≤ 1** — and a full cued phase-gated spiking walk that RELIABLY reaches the hub (reachB = 1.0) reads **learned-exit AT
chance** while **co-igniting both successors** (co_ignite ≈ 1.0). Per THE LAW this is a **method verdict**, and it
**RELOCATES the boundary from the READ to the ENCODE**: the BTSP coincidence encode SATURATES non-selectively at the
shared node, so NO read (mean OR spiking) has a learned-successor signal to ride.

## What was built (NO `sim/` edit; reuse-by-import; the ONLY change vs the matrix runner is the READ)
`research/runners/_gap5_spiking_gamma_recombination_derisk.py` — the SAME shared-hub encode the matrix runner used
(`_prepare_sequence(..., chain_edges=SHARED_EDGES)`: A→B→C + X→B→Y, B = assembly 1 shared), read two ways:
- **INSTRUMENT (weight read, the mandated diagnostic):** from the real `cp_connections`, the per-post-cell summed input
  each candidate successor's cells receive FROM the hub B's cells. Report the MEAN (== the mean-matrix proxy), the TOP-K
  mean, and the MAX per cell — then the LEARNED (C,Y) vs UNLEARNED (A,X) ratio at each read level. If mean_ratio ≈ 1 but
  top-k / max ratio ≫ 1, the potentiated SUBSET carries the signal the mean lost. If top-k / max ratio ≈ 1 too, the
  ENCODE is non-selective at the hub. (This is the finding's "the instrument is part of the emulation".)
- **FULL SPIKING CUED GAMMA WALK:** cue a predecessor (A / X) with a 1000 pA × 150-step completion cue; each theta cycle
  the cue ignites the predecessor, post-fire self-avoidance silences it (the gamma reset), the potentiated pre→B synapses
  ignite B, B is silenced, and B→{C,Y} + the substrate's feedback inhibition + weak background OU noise resolve the
  per-cycle winner. The B-EXIT = the first successor to fire after B, classified stored / recomb / other over many cycles.

The RANK-3 gate's anti-cheat suite: **NO-SHARED** (X→D→Y, B≠D → X never reaches B → recomb must vanish), **NO-ENCODE**
(init weights → learned_exit collapses), **SCRAMBLE** (shuffle the between-assembly edges → learned_exit collapses),
**NO-NOISE** (deterministic → no C-vs-Y sampling).

## Result — seed 42 (GPU cupy) shown; full 6-seed below
Geometric chance of the learned-exit metric = **2/3**: after A→B silences {A,B}, three candidates {C, X, Y} remain and two
(C, Y) are learned successors, so a signal-free read lands on "a learned successor" 2/3 of the time.

| read / arm | value | note |
|---|---:|---|
| INSTRUMENT mean ratio (learned/unlearned) | 0.78 | ≈ the finding's mean-matrix 1.14 — undifferentiated |
| INSTRUMENT **top-k ratio** | **0.72** | the concentration read — **≤ 1, no potentiated-subset signal** |
| INSTRUMENT **max ratio** | **0.79** | even the single strongest cell — ≤ 1 |
| MAIN reachB_frac | 1.000 | the walk RELIABLY reaches the hub (the read RAN) |
| MAIN learned_exit_frac | 0.642 | **≈ chance 0.667** — no discrimination |
| MAIN recomb_frac | 0.118 | mostly stored (stored=30 recomb=4 other=19) |
| MAIN co_ignite_frac | 0.983 | B drives BOTH successors — the co-ignition boundary persists on spikes |
| NO-SHARED recomb | 0.000 | clean (no branch → no recombination) |
| NO-ENCODE learned_exit | 0.000 | collapses (no learned successors) |
| SCRAMBLE learned_exit | 0.333 | collapses below chance (structure destroyed) |
| NO-NOISE recomb | 0.000 | degenerate (sampling requires noise) |

Raw hub magnitudes (per-cell summed input from B, seed 42): learned mean 198.0 vs unlearned 254.0; learned top-k 976.0 vs
unlearned 1356.5 — the LEARNED successors receive **LESS** concentrated hub drive than the unlearned out-edges. This is
reported WITH the SCRAMBLE control (learned_exit collapses to 0.333) and the raw magnitudes, per the selectivity term.

## 6-SEED CONFIRMATION (seeds {42 43 44 100 101 102}, GPU cupy, n_cycles=30)
<!-- SIX_SEED_FILL -->

## Cause — the DIAGNOSTIC read (an instrument check, not a tuning lever)
The finding's stated hope was that the mean averaged away a strong potentiated SUBSET a spiking read would threshold on.
The per-cell instrument tests this DIRECTLY and finds the opposite: the top-k and max per-cell input from B onto the
learned successors is **≤** that onto the unlearned out-edges (ratio 0.72 / 0.79 at seed 42). There is no concentrated
subset — the BTSP coincidence encode SATURATES at the shared hub and writes B→everything broadly (the hub's cells are
driven as post in A→B and X→B AND as pre in B→C and B→Y across the encode, so they potentiate onto all four partners).
The full spiking walk DID reach the hub every cycle (reachB = 1.0), so the learned-exit-at-chance is NOT a read weakness:
there is genuinely no learned-successor signal in the synapses for ANY read to recover. The walk-level co-ignition
(co_ignite ≈ 1.0) is the same fact seen dynamically — B fires both successors because it drives both about equally.

Note on determinism: the spiking encode is GPU-non-deterministic (the RANK 2 lesson), so the per-seed instrument scatters
run-to-run (seed 42 top-k ratio across repeats spanned ~0.70–1.44); this SUPPORTS the verdict — a real concentration
signal would sit robustly ≫ 1, not scatter around 1. The 6-seed spread below is read the same way (central tendency ≈ 1).

## Verdict — per THE LAW, a METHOD verdict; the capability is not abandoned; the boundary MOVES
- **Banked failing method:** full phase-gated SPIKING replay reading the potentiated per-synapse subset at the shared hub
  → learned_exit at the 2/3 geometric chance, because the per-cell instrument shows the learned successors are NOT more
  concentrated than the unlearned out-edges (top-k / max ratio ≤ 1). The controls are clean (NO-SHARED recomb ~0,
  NO-ENCODE / SCRAMBLE collapse, NO-NOISE degenerate) and the walk reached the hub every cycle, so this is a
  correctly-measured, instrument-valid negative — not a broken mechanism.
- **What this REFUTES:** the prior finding's stated cause ("the MEAN discarded a potentiated subset the spiking read
  would recover"). The read is not the boundary; there is no subset signal to recover.
- **Where the boundary actually is (relocated):** the ENCODE. The BTSP coincidence rule at a SHARED node saturates and
  writes non-selectively, so the "learned successors" are indistinguishable from any other B out-edge at every read level
  (mean, top-k, max, and the live spiking walk).
- **Named next method (sharpened):** make the shared-hub encode SELECTIVE — a competitive / homeostatic process during
  the encode that prevents BTSP saturation at the shared node (so B→{C,Y} exceeds B→{X,A}), or a
  pattern-separated-then-completed hub so A and X converge on ONE B pattern whose learned successors dominate. Only then
  does any read have a signal to ride; the read (spiking vs mean) is not the lever.
- **gap#5 core unaffected:** completion CLOSED; replay-boundary SURPASSED 6-seed GO; RANK 1 reactivation 6-seed GO; RANK 2
  forward order gamma-WTA 3/3. RANK 3 imagination remains the open rung — now characterized from a THIRD angle (the full
  spiking read confirms the ENCODE, not the read, is the boundary), with the encode-selectivity method named.

Artifacts: `research/findings/raw/gap5_r4/spk_gamma_recomb_6seed.json` (per-seed instrument + walk + controls),
`spk_gamma_recomb_smoke.json`. Runner: `research/runners/_gap5_spiking_gamma_recombination_derisk.py` (no `sim/` edit).

FILENAME NOTE: the prior finding named the vehicle `_gap5_spiking_gamma_replay_derisk.py`, but that file already exists
(the RANK 2 forward-ORDER spiking runner). This work is in `_gap5_spiking_gamma_recombination_derisk.py` (RANK 3
shared-hub recombination on spikes) so the RANK 2 runner was not clobbered.
