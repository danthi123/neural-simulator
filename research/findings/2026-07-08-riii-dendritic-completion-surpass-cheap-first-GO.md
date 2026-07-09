# R-iii surpass, cheap-first rung (GO, 6-seed): a supra-linear DENDRITIC integration (NMDA-plateau + synaptic clustering) COMPLETES a partial cue where the point-neuron LINEAR summation fails, at the SAME recurrent connectivity — dendritic held-out completion 0.89 vs linear 0.26 (gap 0.63), specific (non-stored 0.00), and shuffled-weights collapse it (0.01). This de-risks the dendritic mechanism as SUFFICIENT to surpass the CYCLE-1064 CA3 completion boundary, before the substantial spiking two-compartment CA3 build. NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_riii_dendritic_completion_derisk.py` (minimal numpy autoassociator, linear vs dendritic read-out). NO `sim/` edit.
**Verdict:** GO (6-seed) — the dendritic non-linearity + clustering is sufficient (in the minimal model) to surpass the point-neuron completion boundary.

## Why this ran
CYCLE 1064/1064b established (rigorously, on the real spiking substrate) that a partial cue's recurrent drive to a held-out CA3 neuron is sub-threshold across weight/density/drive/disinhibition — the point-neuron LINEAR-summation limit. The biological fix (Kandel Ch 13, read in depth this session): the dendritic NMDA-plateau — a CLUSTER of coincident recurrent inputs on a dendritic branch triggers a regenerative plateau (a supra-linear boost) that fires the soma, where the linear sum was sub-threshold. This cheap-first rung tests whether that mechanism is SUFFICIENT, before the substantial spiking two-compartment CA3 build.

## The result — 6-seed (N=400 cells, ensemble M=16, density 0.25, 4 memories)
```
                    held-out completion   non-stored (specificity)
LINEAR (point soma)        0.26                  0.00              <- fails to complete (matches the substrate finding)
DENDRITIC (plateau+cluster) 0.89                 0.00              <- COMPLETES, specifically
shuffled-weights dendritic  0.01                  --               <- collapses (rides the learned attractor)
```
The dendritic read-out completes the held-out neurons at 0.89 vs the linear read-out's 0.26 — a 0.63 gap — at the IDENTICAL recurrent connectivity. The only difference is the non-linearity.

## The mechanism (two ingredients, both biological)
1. **Dendritic PLATEAU non-linearity** (Kandel Ch 13, Larkum BAC / NMDA spike): recurrent inputs partition into dendritic branches; a branch fires a supra-linear plateau iff its summed input exceeds a branch threshold; the soma fires iff ANY branch plateaus. This is the non-linear thresholding a point neuron lacks.
2. **Synaptic CLUSTERING** (Kastellakis-Poirazi): the plateau ALONE was insufficient (an intermediate NEGATIVE this session: with RANDOM branch assignment, dendritic ≈ linear ≈ 0.23) — the co-active same-ensemble recurrent inputs must CLUSTER on ONE branch so a partial cue concentrates → plateau. A held-out MEMBER clusters its same-ensemble inputs (they co-fired during learning); a non-member's inputs scatter (never co-active) → no plateau → the specificity (non-stored = 0.00).

## Anti-cheats (all pass)
- **LINEAR fails at the SAME connectivity (0.26 << 0.89):** the non-linearity is load-bearing, not more inputs.
- **SPECIFICITY (non-stored 0.00):** the clustering completes only ensemble members; non-members (scattered inputs) don't fire.
- **SHUFFLED weights (0.01):** the completion rides the learned attractor structure, not the non-linearity alone.
- Converged honestly: the plateau-alone NEGATIVE (0.23) was the intermediate; adding the biologically-correct clustering gave the GO — the two ingredients are BOTH required (Kandel plateau + Kastellakis clustering).

## Honest scope
A MINIMAL numpy model of the mechanism (plateau = a branch-threshold non-linearity; clustering = same-ensemble inputs assigned to one branch). It demonstrates SUFFICIENCY in principle — the dendritic mechanism CAN complete where the point neuron can't. It does NOT yet realize this on the spiking substrate. The next arc (substantial): a spiking two-compartment CA3 neuron (reuse the project's EMERGE dAP / `sim/dendritic_neuron.py` Larkum-BAC machinery) where recurrent inputs cluster on a dendritic compartment → plateau → soma completion, then re-run the (rigorous, adversarially-instrumented) partial-cue completion probe on the real bridge. The clustering also needs a developmental mechanism (co-active-synapse clustering), a further rung.

## What this establishes
The R-iii CA3 completion boundary (rigorously characterized in CYCLE 1064: point-neuron linear summation can't complete a partial cue) is surpassable by a dendritic supra-linear plateau + synaptic clustering — de-risked cheap-first at 6-seed with proper controls. The mechanism is now motivated + validated in principle; the spiking two-compartment CA3 realization is the next arc, which then unblocks the SWR generative-replay loop (R-iii's original goal).

## Files
`research/runners/_riii_dendritic_completion_derisk.py`; `tests/test_riii_dendritic_completion.py`. Prior: `2026-07-08-riii-DEFINITIVE-ca3-partial-cue-completion-fails-across-param-space.md` (CYCLE 1064/1064b, the boundary); Kandel Ch 13 (NMDA plateau, read in depth); `sim/dendritic_neuron.py` (Larkum BAC).
