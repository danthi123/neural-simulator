# Spiking similarity-matching learned cortex — BUILD PROPOSAL

> **⚠️ PROPOSAL PENDING OWNER APPROVAL.** The build is owner-gated (a weeks-scale commit). This document is
> the design proposal that accompanies the comprehensive 4-axis de-risk
> (`research/findings/2026-06-14-L1-learned-cortex-fair-test-GO.md`). It is **planning, not building** — no
> bridge build is started autonomously. On approval, the next step is a TDD implementation plan
> (writing-plans skill) per the project's standing build discipline.

## Goal

Replace the conversational cortex's idealized exact-inverse vector-symbolic binding (and the curated flat
2,048-concept similarity) with a **learned, generalizing, brain-based cortex** that acquires its concept
similarity **from real experience** (co-occurrence in text) — so "cat" behaves like "dog" because the
cortex *learned* they share contexts, not because a human curated it. This is the artificial-life /
biology-translatable goal (memory: `project_actual_goal_artificial_life_brain_analogue`), distinct from the
already-shipped flat curated cortex (the conversational *product*).

## Why now — the de-risk is comprehensively positive

The owner-directed "better-resourced de-risk" resolved **GO** across all four cheap axes (numpy/smoke,
multi-seed, full anti-cheat battery):

| axis | result |
|---|---|
| **Rule** — does a brain-based online learner reach the host ceiling on real data? | GO — exact Pehlevan-Chklovskii similarity-matching **+0.515** ≈ 98% of the offline PPMI+PCA optimum (+0.523), beats the project's own host method (+0.323); learning load-bearing (+0.312 over random projection); generalizes (held-out 0.72–0.88) |
| **Input-spiking** — survive Poisson-spike input? | GO — 78–89% of the rate ceiling at ~2–6 spikes/hub/concept |
| **Learning non-negativity + spiking** — rectified firing + spike-driven Hebbian? | GO — 90% of the signed rule |
| **Scale-capacity** — 64→256 concepts? | GO — extraction fraction holds 88–100% |

**The enabling operation** is common-mode removal (centering = a subtractive-inhibition EMA) — the
project's recurring whitening/decorrelation theme, here the single fix that lets the *local online* rule
extract the structure the *global* SVD recovers.

## Architecture (the brain-based pipeline) — SIMPLIFIED by the Phase-A capstone

All cognitive computation is neural; host code is limited to the environment (the corpus = the world's
co-occurrence statistics, the "sensory" input) per the BRAIN-BASED-ONLY standard.

> **Phase-A finding (`_l1_phaseA_end_to_end_spiking.py`, GO):** the end-to-end spiking pipeline recovers the
> structure at +0.545 (≈ the offline ceiling), and the **recurrent anti-Hebbian lateral — the original
> highest-risk piece — HURTS under spike noise** (+0.386 vs +0.545 without it). Common-mode removal done
> **explicitly** via subtractive inhibition is more spike-robust than the implicit recurrent lateral. The
> architecture below drops the recurrent lateral accordingly.

```
  context co-occurrence (the "sensory" experience)
        │
        ▼
  [INPUT LAYER: H context-hub neurons]  ── PPMI-shaped drive, realized brain-plausibly:
        │   • log(count)            = Weber-Fechner / dendritic compression
        │   • / hub-marginal        = the Phase-1 dendritic divisive gain  g = σ/(σ+EMA_j)   (ALREADY BUILT)
        │   • − population-mean     = subtractive-inhibition EMA (common-mode removal)   ← LOAD-BEARING, explicit
        │   • max(·,0)              = the spike threshold (rheobase)
        ▼
  [W_ff: hub→output synapses, PLASTIC]  ── bounded Hebbian feedforward (homeostatic weight clip),
        │                                   y_spk · x_spk  with a soft/hard bound = the firing-rate ceiling
        ▼
  [OUTPUT POOL: k cortical neurons]  ── y = relu(W_ff·x_in) over the integration window (non-negative spikes)
        │                                NO recurrent lateral (Phase-A: it hurts under spike noise)
        ▼
  concept code = output spike-count vector integrated over the readout window  → cosine = learned similarity
```

This is a **simplified non-negative Hebbian similarity learner**: the heavy lifting (common-mode removal) is
an explicit subtractive-inhibition front-end (feedforward inhibition), and the feedforward is a
homeostatically-bounded Hebbian — both map onto existing bridge machinery (see protected edits). The full
Pehlevan-Chklovskii recurrent lateral is **not** required (Phase-A capstone).

## Phased plan (each phase ends with a GO/NEGATIVE gate; a NEGATIVE is the deliverable)

**Phase A — full numpy spiking smoke (no sim/ edits) — ✅ DONE, GO.** The complete spiking pipeline
(spiking input + spiking output + spike-driven learning + readout integration) recovers the structure
end-to-end at +0.545 (≈ offline ceiling), multi-seed, learning load-bearing (vs random projection +0.121),
permuted clean. It also returned the build-simplifying finding above (drop the recurrent lateral). The last
all-numpy step before the bridge is complete. (`_l1_phaseA_end_to_end_spiking.py`.)

**Phase B — the bridge, 64 concepts (~1–2 weeks; FEWER protected edits than first thought).** Build the
simplified cortex on the `SimulationBridge`:
- output pool = a `BrainRegion` of k neurons; feedforward = hub→output plastic synapses; **no recurrent
  lateral** (Phase-A);
- PPMI-shaped input drive = the Phase-1 dendritic divisive gain (reuse) + log + threshold;
- **common-mode removal = an explicit subtractive-inhibition front-end** (a global inhibitory interneuron
  pool subtracting the slow population-mean drive — standard feedforward inhibition);
- bounded Hebbian feedforward = the bridge's existing **soft-bound STDP + homeostatic scaling** (the firing-
  rate ceiling is the bound) — likely **no new plasticity rule**;
- the integration window per concept presentation; spike-count readout over the window.
- **Anticipated protected edits (byte-review each):** likely **none new** — the bounded Hebbian maps onto
  existing soft-bound STDP + homeostasis, the subtractive inhibition onto existing inhibitory regions/
  pathways, the /marginal onto the already-shipped Phase-1 dendritic gain. Any edit that *is* needed gets a
  default-off guard (byte-identical when disabled) + a byte-review. **(Dropping the anti-Hebbian −M lateral
  removes the originally-highest-risk edit entirely.)**
- **Gate:** recover the structure on the real 64-concept corpus, multi-seed, full anti-cheat battery
  (permuted-similarity ~0, learning load-bearing vs random projection, host ceiling carries, S_true
  a-priori). NEGATIVE here = the bridge-dynamics rate→spike gap is the wall (the recurring theme) → maps it
  precisely.

**Phase C — scale (~1 week).** Scale 64 → 320 → 2,048 concepts (the documented tiers) with real
co-occurrence. **Gate:** extraction fraction holds; per-concept codes remain discriminable. Watch
real-data-noise-at-scale (the one axis the cheap de-risk could only test synthetically).

**Phase D — conversational integration (~1 week).** Feed the *learned* cortex codes into the existing
dual/CLS pipeline (attractor cleanup + the RF/VSA composer) in place of the curated flat codes. **Gate:**
the full conversational matrix passes (who/what QA, negation/yes-no, dialogue) AND **generalization emerges**
(similar concepts behave similarly — the whole point) AND the **no-confab abstention moat is NOT weakened**
(the Bogacz-Brown familiarity gate, validated at V=320). A regression in the moat is a hard stop.

## Reuse-by-import surface (minimize new protected code)

- `enable_dendritic_divisive_gain` (Phase-1 protected edit) — the /marginal divisive step.
- `BrainRegion` / `RegionPathway` / plastic-gate + transmission-gate infrastructure — the pool + synapses.
- the dual/CLS attractor cleanup + the RF composer + the familiarity-gate moat — Phase D integration.
- the Option-C corpus + PPMI host + S_true + the anti-cheat battery (this cycle's runners) — the gates.

## Honest risk register

1. **Real-data-noise at 2,048** — the rule axis was de-risked on 64 real concepts; scale was synthetic.
   The real 2,048-concept co-occurrence may be noisier than the 64-concept slice. (Phase C gate.)
2. **The bridge rate→spike learning gap** — the numpy smokes are faithful, but the `SimulationBridge`'s own
   LIF/Izhikevich dynamics + STDP timing may not realize the Oja/anti-Hebbian rule cleanly. This is *the*
   recurring project wall; Phase B is where it is honestly tested. (Phase B gate.)
3. ~~**The anti-Hebbian −M fixed-point protected edit.**~~ **RETIRED by Phase A** — the recurrent lateral is
   dropped (it hurts under spike noise); explicit subtractive inhibition + bounded Hebbian replace it,
   mapping onto existing bridge machinery. The originally-highest-risk edit is gone.
4. **Moderate ceiling** — real category structure is +0.52, not +0.9 → moderate ("cat ~ dog"), not perfect,
   generalization. Real, but set expectations.
5. **Moat preservation** — the learned codes must not let confabulation through. Phase D hard-stops on any
   moat regression.

## What this is NOT

Not a replacement for the shipped flat curated cortex (which stays the conversational *product*). Not a
claim that spikes are already done — Phases A–B are exactly where the rate→spike realization is earned.
Not started without owner approval.

## Decision the owner is asked to make

- **(A) Approve the build** → proceed to a TDD implementation plan (writing-plans) starting at Phase A.
- **(B) Approve with scope change** (e.g. stop after Phase B's 64-concept bridge proof, defer scale).
- **(C) Park the build** → bank the L1 GO as the validated path-forward; ship the flat cortex; revisit later.

The comprehensive de-risk supports (A). The build is owner-gated; this proposal awaits steer.
