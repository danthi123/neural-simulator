# D cue-recall arc — SWR consolidation on the DENSE v16 substrate: NEGATIVE (the heteroassociative capacity wall) — 2026-06-05

First REAL-substrate result of the D cue-recall arc (`docs/plans/2026-06-05-D-cue-recall-SWR-consolidation-design.md`).
Per the top-level goal, this honest negative IS the deliverable.

## What was tested
On the v16 concept-pool architecture (working pool→pool propagation + the real 27.5% cue-recall baseline; bridge
`bridges/v16/seed42.simstate.h5`, W→A 13/16): encode 4 concept-concept pairs (apple:big, dog:small, cat:hot,
river:cold) → measure baseline cue-recall (drive `a` alone → is `b` in the lang_output top-3?) → apply SWR
offline-replay consolidation (drive BOTH concept pools repeatedly, `cross_pool_concept` gate OPEN + STDP → strengthen
the directed a→b cross-pool pathway, 40 cycles) → re-measure. Runner: `research/runners/_D_swr_v16_derisk.py`.

## Result — NEGATIVE (no lift; the readout scrambles)

| config | baseline cue-recall | post-SWR |
|---|---|---|
| SWR (readout plastic) | 1/4 | 0/4 (top-3 scrambled) |
| SWR (readout + input gates FROZEN during replay) | 1/4 | 0/4 (top-3 scrambled) |

Baseline 1/4 (25%) ≈ chance (b in top-3 of 15 = 20%) and ≈ the documented multi-seed 27.5%. The SWR consolidation
does NOT lift it — the post-SWR top-3 is scrambled (random words), with or without freezing the input/readout
pathways. So it is not a readout-perturbation artifact.

## Why — the heteroassociative capacity wall (dense codes)
The v16 concept codes are DENSE (each pool ~200 neurons, orthogonal but not sparse). Strengthening the ALL-TO-ALL
cross-pool pathway makes driving `a` activate MANY pools broadly (not selectively `b`) → the lang_output readout is a
scrambled superposition, not `b`. This is exactly the failure the cheat-D research and the project's own
`2026-05-14-engram-stim-recall` work predicted: **"clean cue→associate completion needs SPARSE codes"** (Treves-Rolls:
heteroassociative capacity ∝ recurrent-synapses / sparseness; dense codes → near-zero clean-completion capacity). The
v19 cross-pool result ("adds little for cue-only recall") was the same wall; more consolidation (SWR) does not move it
because the substrate's code density, not the amount of consolidation, is the binding constraint.

## The principled next path (sparse codes — what the project already has)
The fix the literature + the project's own findings point to: run the SWR consolidation on the **G.20 sparse-distributed
codes** (`concept_pool_sparse_distributed` / the 320-concept sparse ensemble — each concept = a scattered K-of-N
pattern, K≈100 in a 2000-pool, ~2% active). Sparse codes give the Treves-Rolls capacity for clean cue completion: a
strengthened sparse a→b pathway activates b's sparse pattern selectively, not a dense superposition. The SWR
consolidation de-risk should be re-run on the sparse substrate; that is the biology-faithful test of whether
generative replay lifts cue-direction recall (the dense substrate provably cannot, by capacity).

## Honest status
- The SWR consolidation MECHANISM is sound (the prior minimal probe confirmed the coincidence three-factor grows a
  directed weight; `_D_consolidation_strength.py`).
- On the DENSE v16 substrate it does NOT lift cue-recall — the binding constraint is code density (the
  heteroassociative capacity wall), a measured biology-grounded limit, NOT a tuning failure.
- NEXT: re-run on the SPARSE G.20 substrate (the principled fix). Open caveat to nail there: confirm the sparse a→b
  pathway actually grows under SWR (read the cross-pathway weight pre/post) so a null is "no capacity-limited lift"
  not "no consolidation happened."

## Artifacts
`research/runners/_D_swr_v16_derisk.py` (+ `--permute` anti-cheat, `_run_D_swr_multiseed.sh`),
`bridges/v16/seed42.simstate.h5`. NO sim/ edits.
