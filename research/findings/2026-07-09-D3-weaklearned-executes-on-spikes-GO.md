# D3 integration — the WEAK-SUPERVISION-LEARNED transition δ EXECUTES ON SPIKES (learning + spiking execution both on the substrate), length-generalizing

**Date:** 2026-07-09
**Runner:** `research/runners/_d3_weaklearned_spiking_derisk.py` (reuse-by-import: RANK-1 `train_endstate` + rung-1 `build_fswta_score_bridge`/`fswta_drive`/`spiking_rollout_eval`; numpy; NO `sim/` edit).
**Verdict:** GO (S3, 6-seed) — closes the learning→spiking-execution loop.

## What this composes
Two D3 pieces were validated separately: **RANK-1** learned δ from END-STATE-only supervision (no per-step teaching, rate); the **one-loop** executed a *teacher-forced* δ on the spiking FS-WTA re-discretization. This composes them: **train δ by WEAK (end-state-only) supervision → roll it out with the SPIKING FS-WTA re-discretization** (a real `SimulationBridge` — the winner attractor pool FIRES, the next state is decoded from spikes, iterated). So the whole story is on the substrate: δ **learned** from weak supervision + δ **executed** on spikes, held-out-deeper.

## The result (S3, 6-seed; NO `sim/` edit)
| S3, 6-seed (42/43/44/100/101/102; chance 0.167) | held-out-DEEPER state-track |
|---|---|
| **weakly-learned δ, SPIKING FS-WTA re-discretization** | **1.000** every seed (host-agree **1.000**) |
| SHUFFLE-learned δ (memorization floor), same spiking rollout | 0.171 (≈ chance — collapses) |

GO all 6 seeds (SPIKING 1.000 ≫ SHUFFLE 0.171; host-agree 1.000).

## What the anti-cheats establish
- The δ **LEARNED from end-state-only supervision** (RANK-1, no per-step teaching) EXECUTES on the spiking FS-WTA re-discretization and **length-generalizes** to held-out-deeper (~1.0) — the weakly-learned transition table rolls out on spikes to arbitrary depth.
- **host-agree ~1.0**: the spiking winner == the host argmax over the weakly-learned scores (the FS-WTA faithfully re-discretizes the learned transition).
- **SHUFFLE-learned → collapse** (~0.24): a δ trained on shuffled endpoint labels executes at chance on spikes → the rollout is running the genuinely-learned transition, not noise.

## ⇒ the D3 recurrent-composition substrate is complete on the substrate
Combining the arc: the discrete-attractor recurrent multi-hop composition is **found** (mechanism = CA3), **learned from weak (end-state-only) supervision** (RANK-1, 6-seed), and **executed on spikes** (transition LIF + re-discretization FS-WTA in one loop, and now the *weakly-learned* δ on the spiking FS-WTA) — length-generalizing throughout. The simulated recurrent sequence/language cortex step is learned-from-weak-supervision AND spiking, end-to-end.

## Honest scope + next
- Validated on S3 (K=6). A5 (the structureless 60-way non-solvable worst case) needs the coverage/epochs lever for the weak-supervision learning to fully converge δ (in progress) — a scale-hardening residual, not a mechanism wall.
- The curriculum LEARNING itself is still a host loop; the spiking-substrate realization of the *learning* (rung-2's surrogate-gradient spiking LIF transition trained on RANK-1's detached-rollout-curriculum triples) is the deepest "fully-spiking" rung.
- RANK-3 (fully self-supervised δ from observation prediction, HAE/TEM) is the emergent follow-on; apply to real LANGUAGE sequences is the mission payoff.

## Files
`research/runners/_d3_weaklearned_spiking_derisk.py`; the D3 arc `2026-07-09-D3-*.md`.
