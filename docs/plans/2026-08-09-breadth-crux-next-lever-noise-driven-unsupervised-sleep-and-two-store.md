---
title: "Breadth crux NEXT LEVER — external research grounds two untested mechanisms (noise-driven unsupervised sleep replay; episode-gated two-store)"
date: 2026-08-09
type: plan
lane: memory-continual-learning
---

> **This is a forward-looking PLAN** (the next lever, grounded in an external-research round). The prior negatives
> it references are backward-looking measurements banked as their own findings; the numbers quoted from external
> papers are literature citations, not results asserted here.

# Breadth crux: the literature says MECHANISM GAP, and three of our negatives were the WRONG VARIANT

## Why this doc exists (the DR discipline, applied)

The sequential-teaching retention crux had **seven cheap levers — sleep-replay, engram-store, budget,
sparse, SHY, interleaving, weight-protection — ALL REFUTED or within-noise (0.55–0.70)** before any
external-literature round. That is the exact "≥2 levers on one defect without research" pattern the new
`gates/deep-research-at-wall` (DR) exists to stop. This is the research round, done properly (sources READ,
not skimmed), recorded via `tools/record_external_search.sh` (three sources, cited below). It grounds the
NEXT lever; it makes no experimental claim of its own.

## The decisive calibration: 0.55–0.70 is a mechanism gap, NOT a ceiling

- **SNN latent replay reaches ~92% class-incremental retention** (Compressed Latent Replays for SNNs, 2024;
  SHD, ≤4% drop, 100× less memory than naive rehearsal). A spiking substrate CAN retain ~10 classes at
  >0.9. Our 0.55–0.70 is **far below an achievable ceiling ⇒ a mechanism gap.** The wall is a method verdict,
  not the capability. [Do NOT declare it.]
- **BUT read the regime carefully.** Bazhenov's SNN sleep study (Golden et al., PLOS Comput Biol 2022,
  e1010628) reports, for a 2-task OVERLAPPING-readout problem: single-task ceiling **0.70**, catastrophic-
  forgetting floor **0.52 (chance)**, sleep-recovered **0.70 / 0.68**. Our replay/protect numbers
  (0.55–0.70) sit *inside this exact band* — so for an OVERLAPPING single readout, 0.70 is ~full recovery,
  not underperformance. The gap to 0.92 is unlocked by a DIFFERENT readout architecture (separated stores),
  not by pushing a single overlapping readout past its own ceiling.

## The three negatives were the WRONG VARIANT of a mechanism that provably works

| Our refuted lever | What we did | What the working literature does | The untested variant |
|---|---|---|---|
| "sleep-replay" (0.55) | **Supervised** replay of **stored** referent-cue patterns | **Noise-driven spontaneous reactivation** (input silenced, hidden units driven by Poisson noise; task-specificity of the noise NOT critical) + plasticity **switched to UNSUPERVISED STDP** → weights move to the task-manifold **intersection** ("joint weight representation") | noise-driven + unsupervised-STDP sleep phase — **never tested** |
| "weight-protection" (+0.017) <!--derived--> | EWC / SI / Phase-1.4 gate-freeze **alone** | CH-HNN (Nat Commun 2025, s41467-025-56405-9): metaplasticity `e^(-mW)` works **only combined** with episode-gated neuron masking | metaplasticity is the wrong HALF in isolation |
| "pattern-separation" (running, PS-SNN fixed-orthogonal) | **Fixed random orthogonal** targets | CH-HNN: a **slow store emits binary episode-masks** partitioning which fast-store neurons fire per episode (episode inference from a learned similarity signal) — NOT raw replay, NOT fixed-random | LEARNED context/similarity gating, not fixed-random |

**Read:** every refuted lever maps onto a mechanism that DOES prevent forgetting in the SNN literature — we
tested the crude/wrong-half/wrong-drive variant each time. That is a method verdict, exactly as the law says.

## The mechanism the biology runs that we replaced with a constant

Per the wall-reframe ("what else does the real system run alongside this that we proxied with a constant?"):
we implemented **ONE store** (a readout map) and forced it to learn N facts sequentially. CLS theory
(McClelland 1995; CH-HNN 2025) says the biology runs **TWO interacting stores** — a fast, pattern-separated
hippocampal store (DG-CA3, one episode → its own neuron subset) and a slow cortical store consolidated by
**interleaved offline replay**. The "constant" we substituted for the fast store is: *nothing* — new facts
overwrite the single map. No single-store lever (replay KIND, protection, separation-of-targets) can fully
fix a missing store; the architecture is the gap.

## Next lever (properly researched), cheapest-first

1. **Noise-driven, unsupervised-STDP sleep phase** (Bazhenov variant) — cheapest, genuinely untested on our
   substrate: after sequential acquisition, SILENCE the cue input, drive readout-upstream units with Poisson
   noise, switch the readout plasticity from supervised to UNSUPERVISED STDP for a sleep phase, measure
   retention. This is a different mechanism from every replay we ran (all supervised, all stored-pattern).
   Anti-cheat: the sleep phase must be neural (noise → spikes → STDP), not a host weight-averaging step.
2. **Two-store: fast pattern-separated store (per-fact neuron subset via a learned/context gate) + slow
   consolidated readout** (CH-HNN skeleton, biologized) — the architectural fix; bigger build. The running
   PS-SNN arm is the fixed-orthogonal first cut of the fast store; if it lifts retention, the learned-gate
   version is the follow-on. If it does NOT, the two-store split (not just separated targets) is the lever.

## Sources (READ in depth, not skimmed — DR discipline)

- Golden, Delanois, Sanda, Bazhenov (2022). *Sleep prevents catastrophic forgetting in spiking neural
  networks by forming a joint synaptic weight representation.* PLOS Comput Biol 18(11):e1010628.
- CH-HNN (2025). *Hybrid neural networks for continual learning inspired by corticohippocampal circuits.*
  Nature Communications, s41467-025-56405-9. (PMC11788432)
- *Compressed Latent Replays for Lightweight Continual Learning on SNNs* (2024). ResearchGate 381960454.
- McClelland, McNaughton, O'Reilly (1995). *Why there are complementary learning systems.* Psychol Rev.

NO-EXTERNAL-NEEDED: n/a — this doc IS the external round.
