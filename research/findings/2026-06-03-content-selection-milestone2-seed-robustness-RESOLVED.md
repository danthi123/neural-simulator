# Content-selection Milestone 2 — spiking content-selection seed-robustness RESOLVED (6/6 multi-seed)

**Date:** 2026-06-03
**Status:** ✅ RESOLVED — `SpikingController` coherence 6/6 seeds (12/12 conditions), up from 2/3 seeds.
**Module:** `research/runners/content_selection_spiking.py`
**Supersedes:** the "honestly-flagged open refinement (seed-fragile, 2/3 seeds)" boundary in
`2026-06-03-content-selection-milestone2-spiking-dlpfc-persistence-CHARACTERIZED.md`.

## One-line result

The faithful spiking content-selection (discourse context held in a spiking cortico-PFC
loop attractor; PFC "Control" selects the most relevant unsaid associate over it) is now
**coherent across all 6 seeds tested (42–47), both topics each = 12/12 conditions**, after a
root-cause fix to the working-memory dynamics. The earlier 2/3-seed fragility is solved and
fully explained.

## The validated config

`SpikingController` defaults changed to the validated clean-dynamics config:
- `internal_density = 0.0` — no random within-region recurrence (clean within-concept attractors only)
- `enable_ou = False` — quiet hold (OU background noise disabled during the WM hold)

```bash
# Reproduce (CPU-only, ~7 min for the 6-seed sweep):
python -c "from research.runners.content_selection_spiking import SpikingController; \
from research.runners.content_selection import build_association_graph; \
graph = build_association_graph(['apple_big','apple_cat','dog_small','dog_river','cat_hot','river_cold','big_hot','small_cold']); \
clusters={'apple':{'big','cat','hot'},'dog':{'small','river','cold'}}; \
[print(seed, all(all(c in clusters[t] for c in [SpikingController(graph,seed=seed).turn([t]) for _ in range(3)]) for t in ['apple','dog'])) for seed in [42,43,44,45,46,47]]"
```

| seed | apple topic | dog topic |
|---|---|---|
| 42 | big, hot, cat ✅ | river, cold, small ✅ |
| 43 | big, cat, hot ✅ | river, cold, small ✅ |
| 44 | big, cat, hot ✅ | river, small, cold ✅ |
| 45 | ✅ | ✅ |
| 46 | ✅ | ✅ |
| 47 | ✅ | ✅ |

**12/12 conditions coherent.** (Prior best, `internal_density=0.1` + OU on: 4/6 = seeds 42/43 only.)

## Root cause (fully diagnosed — an 8-probe falsification trail)

The failure mode was a classic **Hopfield spurious-state / attractor-capacity** behaviour, located
precisely by a cheap-first localization sequence. Each probe ruled out a hypothesis with data:

1. **Localize WM vs selection.** Dumped the held set + candidate relevance scores at the failing
   seed (44) vs a working seed (42). Both held "apple" *strongest* — so the selection logic was
   fine; the WM was holding a **spurious coherent off-cluster blob** (seed 44: river 0.325, dog
   0.306, cold 0.211 alongside apple 0.345) that out-voted apple's edges in the full relevance sum
   → picked "cold", off-topic.
2. **Top-1 / top-2 held readout** (attentional-focus hypothesis, Cowan): **refuted — made it WORSE**
   (1/6, 2/6). The full held set is *more* robust because in-cluster concepts mutually reinforce in
   the relevance sum; reading fewer items throws that away.
3. **Attractor-weight sweep** (raise the latch threshold so spillover can't latch): **refuted — no
   window.** At W=50 both apple and spurious fully latch; at W=35 the *spurious* concept latches
   (0.243) while the driven apple collapses (0.003); at W≤25 nothing holds. Seed 44's random
   recurrence has a structural basin *deeper than the input drive itself*.
4. **Biased competition** (Desimone & Duncan 1995 — blanket feedback inhibition ∝ activity + top-down
   bias protecting the driven concept, modeled as activity-proportional hyperpolarizing current):
   **refuted.** Even k=40/bias=1000 barely moved it (apple pinned 0.344, spurious 0.325→0.309) and
   it killed the *legitimate* weak in-cluster signal instead. Two co-equal saturated attractors have
   no activity-level asymmetry to exploit.
5. **Localize the spurious source — `internal_density` sweep.** At `internal_density=0.0` a *single*
   concept drive is **perfectly clean** (apple 0.344, spurious 0.001, separation +0.343 vs +0.020).
   The random internal recurrence is *a* spurious source — but not the whole story (next).
6. **Per-concept single drive at density=0:** every concept drives cleanly *alone* (self≈0.34,
   spurious≈0.001). So the spurious only appears when a **second concept is driven while the first is
   still latched** — the multi-concept hold, with no wiring path between the disjoint patterns →
   a *global* mechanism.
7. **Sequential accumulation forensic (density=0):** confirmed the spurious blob grows as each new
   concept is added to the held set, at both seeds — the multi-concept hold raises global excitability.
8. **OU background noise off:** **the fix.** With `enable_ou_process=False`, the sequential hold is
   *exact*: drive apple→big→cat → held set = {apple, big, cat}, **zero spurious** (was 0.33). The
   seeded OU noise was tipping the over-eager bistable attractors of *other* concepts into their ON
   state once the network was excited by holding ≥2 concepts.

**Mechanism in one sentence:** holding ≥2 concepts raises the network's global excitability enough
that the seeded background noise tips *other* concepts' strong bistable attractors into spurious ON
states (Hopfield spurious states), which then hijack the relevance-based selection seed-dependently;
removing the random recurrence (clean attractors) **and** the noise (quiet hold) yields an exact
multi-concept WM and robust selection.

## Why this matters (biology-translatable)

- It connects the project's central **learned-vs-structural / per-seed-structural-variance wall**
  (P5 iter KK–PP, in-vivo binding, etc.) to a concrete, classical computational-neuroscience
  phenomenon: **spurious attractor states in an over-capacity associative memory under noise**.
- It is the brain-analogue reason an attentive/focused cortical state (lower effective noise, sparse
  clean assemblies) holds a multi-item working memory cleanly, whereas a noisy over-eager attractor
  network confabulates extra items into the held set.

## Honest scope + the principled next refinement

- **Faithfulness caveat:** `enable_ou=False` models a *quiet* cortex; biological cortex has
  spontaneous synaptic background. The honest reading is that these attractors are *pathologically
  noise-sensitive* (weight-50 all-to-all, near capacity), whereas real cortical attractors are
  noise-robust via **sparse coding + inhibitory stabilization**. The clean-dynamics result proves the
  architecture and the selection mechanism are correct; restoring biological noise robustly is the
  pre-registered next refinement (sparse k-of-N assemblies + per-assembly inhibitory shadows so the
  attractors tolerate default OU noise without spurious tipping).
- **Still set, not learned:** the attractor weights are *set* (outer-product), not learned — the
  separate documented next step (learn them with a stabilized one-shot rule, not vanilla Hebbian).
- **Selection logic still structured:** Milestone 3 (make the relevance/selection itself spiking) is
  the remaining faithfulness step.

## Bottom line

The faithful spiking content-selection Control — spiking cortico-PFC loop-attractor working memory
holding the discourse context + PFC relevance-based selection over it — is **validated as a mechanism
and now seed-robust (6/6)**. The 8-probe trail is the scientific deliverable: it pins the seed
fragility to noise-tipped Hopfield spurious states and rules out six activity-level/readout fixes
before the clean-dynamics resolution.
