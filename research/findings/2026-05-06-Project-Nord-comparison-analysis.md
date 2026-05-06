# Project Nord SNN Language Model — comparison + 2 ideas to adopt

**Date:** 2026-05-06 ~02:30 EDT
**Source:** https://github.com/gtausa197-svg/-Project-Nord-Spiking-Neural-Network-Language-Model
**Status:** Comparison analysis. Two ideas worth stealing; rest doesn't
apply due to fundamentally different goals.

---

## What Nord is

- **1.088B-parameter pure SNN language model**, trained from scratch
  via surrogate-gradient backprop on FineWeb-Edu (~9.67M samples,
  27K steps, ~$400 cost)
- 93% sparsity (only 7% of neurons fire per token)
- 3-zone architecture: Sensory → Association → Memory → Executive,
  plus Genesis Triple Memory (Structural+Personal+Auxiliary banks)
- Spike-driven MoE (8 experts, top-2 routing, 128 clusters)
- Reward-modulated STDP that **synergizes with backprop** (not standalone)
- Multi-scale temporal: T_fast=8 + T_slow=2 timesteps
- AssociativeLIF neurons with learnable per-cluster cascade gain
- LLaMA-3 tokenizer (128K vocab)
- Loss 4.4 at training end; generates coherent multilingual text

## Key contrasts with our project

| | Nord | Our project |
|---|---|---|
| **Goal** | Text generation (LM) | Biology-faithful embodied agent |
| **Scale** | 1.088B params, 27K training steps | ~6K neurons, ~3M synapses, 800 trials |
| **Primary training rule** | Surrogate-grad backprop | Pure local plasticity (Tier 1 embodied Hebbian) |
| **Neuron model** | Learnable LIF with per-cluster cascade | Izhikevich 2007 |
| **Architecture** | Transformer-inspired zones | Brain-region cascade (motor cortex, BG, hippocampus, etc.) |
| **Encoding** | Multi-scale temporal (T_fast + T_slow) | Rate-coded over fixed windows (stim_steps=50) |
| **Plasticity** | STDP fine-tunes after backprop | STDP IS the primary learning signal |
| **Tokenizer** | LLaMA-3 (128K vocab) | Hash-based 30-word vocab |
| **Sparsity** | 93% emergent | ~10% by design (sparse codes per token) |
| **Hardware** | RTX 5070 (8GB) | RTX 3090 (24GB) |

## What this confirms about our findings

**Nord's success with surrogate-grad backprop reinforces our W→A
verdict from 2026-05-05.** Both projects independently demonstrated:

- **Gradient-based learning scales to SNN at large** (Nord 1B scale,
  loss 4.4) and at our biological canon scale (B3 supervised: 3/3
  NESW aligned).
- **Pure scalar reward fails** for SNN at scale (Nord's docs cite
  prior work like SpikeBERT noting "self-accumulating dynamics"
  failure); our 3-factor verdict found the same thing at biological
  canon (1/6 alignment).
- **STDP is best as a complement, not the primary signal** (Nord uses
  it WITH backprop; our embodied Hebbian uses STDP+co-firing teacher
  signals — both succeed, the failure mode is STDP+scalar reward
  alone).

This is consistent across two completely independent projects with
very different goals. Worth noting in our verdict docs.

## What's directly adoptable

### Idea 1: EMA-smoothed `read_language_output` (small, low risk)

Nord uses per-channel learned exponential decay (α≈0.80) to bridge
binary spikes to continuous logits. The most recent timestep
contributes ~7× more than the earliest.

Our `read_language_output` does cosine match on raw cumulative spike
counts — coarser temporal aggregation.

**Adoption**: replace cumulative spike count with EMA over the eval
window. Per-neuron α ≈ 0.8 (or learnable in future). This should give
more stable A→W decoding by weighting recent spikes more heavily.

**Cost**: ~2-4 hours implementation + smoke retest.
**Expected gain**: A→W accuracy 45% → 50-60%.
**Risk**: low. Backward-compatible (default current behavior, opt-in
EMA).

### Idea 2: Cluster-structured motor pools (medium, addresses Tier 2.1)

Nord's AssociativeLIF has 64-128 clusters with learnable per-cluster
gain and soft neighbor weights. Our motor pools are flat 500-neuron
populations.

**Adoption**: subdivide each motor pool into N sub-clusters (e.g.,
2-4 clusters of 125-250 neurons within each motor_X). During Tier
2.1 training, different synonyms can preferentially activate
different sub-clusters → no STDP winner-take-all competition.

**Why this might fix Tier 2.1**: today's v1/v2 showed STDP locked in
to ONE synonym per motor pool because all neurons compete for the
same input pattern. Sub-clusters could let "north" claim cluster A
while "up" claims cluster B — both within motor_N action.

**Cost**: ~1 week implementation (cluster topology in BrainRegion +
sub-cluster routing).
**Expected gain**: Tier 2.1 W→A might align 4-6/6 (currently 0/6).
**Risk**: moderate. Requires architectural change to brain-region
framework.

**This is the right v4 fix if v3 (cofire 0.3) fails.** Combines well
with the scale-up the user mentioned (24GB VRAM headroom): bigger
motor pools (e.g. 1000 neurons / 4 clusters of 250 each) gives even
more capacity per synonym.

## What's NOT directly applicable

### Surrogate-gradient backprop (their core method)

Violates our biology-grounded constraint. Their backprop is per-weight
gradient — not biology-plausible. We already proved gradient works at
our scale via B3 (3/3 aligned at biology canon). Adopting their
training method would just confirm that result with extra steps and
abandon biology fidelity.

### Transformer-flavor zones

Their Sensory → Association → Memory → Executive zonal hierarchy
isn't isomorphic to actual brain regions. We have biology-plausible
regions (motor cortex with E/I balance, BG cascade with D1/D2
asymmetry, hippocampal trisynaptic loop, etc). Their zones are
*inspired* by brain organization but functionally are layered
transformer blocks.

### 1B-scale text generation

Different problem entirely. We have 30-word vocab; they have 128K.
We have 4 actions; they have full LLaMA-3 vocabulary. We optimize for
embodied bidirectional binding; they optimize for next-token
prediction loss.

### Genesis Triple Memory

Three parallel memory banks (Structural + Personal + Auxiliary) with
learned routing is an interesting design pattern but invented for
sequence-learning. Not directly biological. Could inspire Tier 3
architecture (verb bank + noun bank + composition bank) but not
critical path.

### Spike-driven MoE

8 experts with routing emerging from spike rates is interesting but
solves a problem we don't have. Our network doesn't need experts —
we have specialized brain regions instead.

## Net assessment

Nord is a real engineering accomplishment in a fundamentally different
research line. Their core contribution is **scale + practical LM via
surrogate-grad backprop on SNN.** We're working on **biology-faithful
embodied learning at small scale.**

Their result that backprop-trained SNNs converge at billion-scale is
**consistent with our finding** that gradient (B3) succeeds where
3-factor scalar feedback fails. Just at vastly different scales.

The 2 ideas worth stealing (EMA readout + cluster-structured motor
pools) are **minor enhancements**, not architectural pivots. They
fit naturally into our existing line.

## Action items

1. **Pre-stage EMA-smoothed readout** — implement now while v3
   cofire is in flight. Low risk, ~2-4 hours. Test against Tier 1
   baseline once shipped.
2. **Pre-stage cluster motor pools** — implement as v4 ready-to-launch
   if v3 fails. ~1 week, but skeleton can be ready in this session.
3. **Document this comparison** — done (this file).
4. **Cite Nord in the W→A verdict doc** as independent confirmation
   of "gradient works, scalar reward doesn't" finding.

## Files

- This finding
- Nord repo: github.com/gtausa197-svg/-Project-Nord-...
- Tier 1 breakthrough (the parallel result on biology side):
  `research/findings/2026-05-06-Tier1-BREAKTHROUGH-bidirectional-binding.md`
- W→A verdict (which Nord's surrogate-grad confirms):
  `research/findings/2026-05-05-W-to-A-VERDICT-global-scalar-feedback-fails.md`
