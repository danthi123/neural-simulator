# R-iii SWR generative-replay loop, Rung 1 — the completed CA3 assembly does not yet drive ca1 robustly at the small completion-validated scale; root-caused (read-substrate-first) to Schaffer CONVERGENCE (a network-scale lever), not the completion mechanism. Honest interpretable checkpoint on top of the committed CYCLE-1076 capstone.

**Date:** 2026-07-09 (CYCLE 1076 follow-on)
**Runners:** `research/runners/_riii_swr_generative_replay_derisk.py` (Rung 1), `_riii_ca1_transmission_probe.py` (the isolation probe). GPU. NO `sim/` edit.

## The goal (the capstone's payoff)
The CYCLE-1076 capstone (emergent CA3 pattern completion, 6-seed GO) enables the SWR generative-replay loop: during NREM, a partial/spontaneous SWR trigger reactivates the FULL CA3 assembly (completion), which drives ca1 -> cortex via the Schaffer collaterals, and STDP consolidates the pattern OFFLINE (systems consolidation; Phase 1.3 validated this with FULL-tag-drive replay -- the NEW capability is GENERATIVE replay from a DEGRADED cue). Rung 1 tests the prerequisite: does the partial-cue completion drive a CORRECT + assembly-SPECIFIC ca1 pattern?

## The result: ca1 does not fire from the completed assembly at this scale (a well-isolated NEGATIVE)
```
config (n_ca3=500, n_assembly=12, n_ca1=120, Schaffer density 0.30 weight 4.0)   partial->ca1 MATCH   ca1_raw_fire
gentle sustained cue, Schaffer boost x1 / x6 / x15                                0.000                5.00 (identical!)
synchronous gamma-burst reactivation (SWR ripple), boost x1 / x6                  0.000-0.035          6.67-7.00
```
The ca1 read is stuck at the ~5-spike noise floor regardless of the Schaffer weight or the drive mode. Diagnosis, each step read from the substrate (NOT guessed) -- the (a0) discipline:
1. **The boost APPLIES + propagates:** `_scale_pathway` scaled 17937 ca3->ca1 edges x15, and `base_synaptic_weights = self.cp_connections.data` (bridge.py:5945) confirms the per-step synaptic current is recomputed from `cp_connections.data` each step -> the x15 reaches the ca1 current. Ruled out: a cached/ignored weight.
2. **Drive mode is not it:** the synchronous gamma-burst (a real SWR ripple = a strong synchronous population burst, not a gentle cue) also leaves ca1 at the floor. Ruled out: the sustained-vs-synchronous drive.
3. **The ISOLATION probe (`_riii_ca1_transmission_probe`, no formation): TRANSMISSION WORKS but is WEAK.** Driving 40 CA3 cells DIRECTLY + hard (3000 pA -> 1194 CA3 spikes/60 steps) with the Schaffer x15 fires ca1 to **14 spikes vs a 2-spike baseline** (7x above baseline -> ca1 is NOT blocked by rheobase / feedback inhibition / a closed gate). But 14 ca1 spikes from a MASSIVE 40-cell / 1194-spike CA3 drive is weak absolute transmission.

## Root cause: Schaffer CONVERGENCE at this scale (a network-scale lever, NOT the completion mechanism)
ca1 firing scales with the NUMBER of converging CA3 inputs, not the Schaffer weight: 12 firing assembly cells x 0.30 density = ~3.6 inputs/ca1 (the completed assembly -> ca1 at the noise floor); 40 cells = ~12 inputs/ca1 (the probe -> ca1 fires, weakly). The completion was validated at a SMALL SPARSE assembly (12 cells in 500, the CYCLE-1076 GO), but robust ca1 drive needs MORE CA3->ca1 convergence -- i.e. the Kopsick regime (a larger CA3 + a larger assembly + denser/more Schaffer). This is a SCALE requirement, the same family as Kopsick's 275-cell assembly in 75,000 cells being simultaneously ROBUST (enough cells to drive downstream) AND sparse.

The completion mechanism is NOT the blocker (it is 6-seed GO and committed); the blocker is that the small network that makes the completion cheap to validate does not give ca1 enough Schaffer convergence.

## UPDATE (same cycle) — the SCALE fix is CONFIRMED for ca1-drive, and the Rung-1 metric is reframed: CA1 is not a pattern-separator, so specificity is a LEARNED (consolidation) job, not a fixed-Schaffer one
Scaling to n_ca3=1500, n_assembly=40 (~2.7%, ~12 Schaffer inputs/ca1, matching the isolation probe that fired ca1): **ca1 now FIRES strongly — ca1_raw_fire 135 -> 201 -> 266 -> 333** across the k-sweep (vs the ~5 noise floor at n_ca3=500). The scale hypothesis is CONFIRMED — a Kopsick-regime assembly drives ca1. BUT the ca1 pattern is NON-SPECIFIC at every k:
```
n_ca3=1500 n_assembly=40, Schaffer x15    partial->ca1 MATCH   cross   LINEAR-match   ca1_fire
k=60                                       0.763               0.763   0.788          266   (MATCH == cross exactly)
k=100                                      0.791               0.793   0.788          333   (MATCH < cross)
k=140                                      0.716               0.758   0.788          201
```
Every ca1 pattern is ~0.76-0.79 similar to every other (same-assembly AND cross-assembly), and the LINEAR (no-completion) control is the HIGHEST — i.e. the completion does NOT add assembly-specificity to ca1; the fixed Schaffer gives a BROAD ca1 activation shared across assemblies. **This is biologically CORRECT: CA1 is a relay/comparator, NOT a pattern-separator (that is DG/CA3, catalog D.12).** A fixed feedforward Schaffer cannot produce assembly-specific ca1 codes; the assembly-specific ca3->ca1->cortex mapping is exactly what SWR-replay STDP LEARNS during consolidation. So the Rung-1 "does the fixed Schaffer carry specificity to ca1" test was the WRONG metric (and it is a clean NEGATIVE for the right reason). The prerequisite that DID matter — ca1 must FIRE so STDP has post-synaptic spikes — is now met at scale.

## NEXT (specified, the continuing lever) — Rung 2, with the CORRECT (Phase-1.3) metric
The real payoff metric is **consolidation-to-cortex**, not ca1-code-specificity: after generative partial-cue REPLAY (partial cue -> completion -> ca3->ca1->cortex with the `ca3_to_ca1` + ca1->cortex STDP gates OPEN), can a DOWNSTREAM region RECALL the pattern WITHOUT the hippocampus (Phase 1.3's hippo-OFF retention test), and does the GENERATIVE (partial-cue-triggered) replay consolidate as well as full-tag replay? Controls: no-replay (no consolidation) + shuffled-assembly. Reuse the Phase-1.3 consolidation infra (`consolidation_trainer.run_concept_replay_phase` + the sleep gates). Build at the scale-confirmed n_ca3=1500 / n_assembly=40 (first re-confirm the CYCLE-1076 COMPLETION itself is specific at that scale via the completion runner + a re-tuned k — the CA3-side metric, separate from the ca1 read). The scale lever for ca1-drive is DONE; Rung 2 is the learned-consolidation build on top.

## (original) NEXT
Scale toward the Kopsick regime so BOTH the completion AND the ca1 drive work: a larger CA3 (n_ca3 ~1500-3000), a larger assembly (~1% -> 15-40 cells, still sparse, still completable -- re-validate the CYCLE-1076 completion at the new scale), and if needed a denser/stronger Schaffer (ca3->ca1 density). Then Rung 1 (completion -> specific ca1) -> Rung 2 (open the `ca3_to_ca1` STDP gate during replay -> the ca1 projection strengthens -> ca1 reactivates the pattern WITHOUT the ca3-recurrent completion, vs a no-replay control) -> the offline generative-replay consolidation loop. The capstone (the hard part) is solved + committed; this is a scale/wiring lever on top of it.

## Files
`research/runners/_riii_swr_generative_replay_derisk.py`, `_riii_ca1_transmission_probe.py`. Builds on `2026-07-09-riii-emergent-ca3-completion-kopsick-formation.md` (the CYCLE-1076 capstone). Consolidation infra: `research/runners/consolidation_trainer.py` (run_swr_replay_phase / run_concept_replay_phase), the ca3->ca1 Schaffer + ca1->cortex pathways (`enable_hippocampus_consolidation`). Kandel Ch 54 (SWR, systems consolidation); Kopsick et al. 2024 (PMC10996657).
