# RUNG 1 GO (6-seed, adversarially verified) — a fully-emergent, on-bridge, NO-BPTT next-token LANGUAGE MODEL: a fixed spiking reservoir + a shallow one-step-local-delta read-out beats the bigram, the whole-prefix BAG, the memoryless NON-RECURRENT projection, AND a 4-gram on held-out next-token cross-entropy — the recurrent DYNAMICS are load-bearing.

**Date:** 2026-07-10
**Runner:** `research/runners/_emerge_reservoir_lm_derisk.py` (reuse-by-import: EMERGE-82 `OnBridgeLSM` fixed spiking reservoir + EMERGE-61 wash-out + EMERGE-62 `build_stream` corpus; NO `sim/` edit, NO edit to any existing runner).
**Verdict:** GO — the FIRST rung of the primary emergence-generation direction (the dendrite-free, learned-from-experience path that answers the whack-a-mole concern). Adversarial-verify `w3eviw8bv` = SURVIVES-WITH-SCOPE-FIXES → the decisive confound controls were run → dynamics-earned 6/6.

## The mechanism (emergent, on-bridge, no BPTT, no weight transport)
A FIXED-random recurrent Izhikevich `BrainRegion` on a real `SimulationBridge` (the EMERGE-82 reservoir; `internal_density` = the fixed recurrent conductance synapses) is driven per token; the per-token running-cumulative population spike-rate is the state. A SHALLOW linear-softmax read-out `W_out` is trained ONLINE by the ONE-STEP next-token delta rule `W += lr·(onehot(next) − softmax(W·state))⊗state − wd·W` — the next token IS the clean local target, so there is no deep credit, no BPTT, no weight transport (the ESN/LSM discipline: fixed recurrent cortex + a locally-trained output projection). Autoregressive rollout feeds the read-out's token back in.

## The result (6-seed: dev 42/43/44 + blind 100/101/102; V=24 controlled template grammar; held-out next-token cross-entropy, nats)
| | reservoir | bigram | bag-of-prefix | non-recurrent | 4-gram | vanilla-readout |
|---|---|---|---|---|---|---|
| **mean CE** | **0.830** | 1.075 | 0.970 | 1.090 | 0.974 | 0.912 |
| per-seed range | 0.792–0.866 | 1.051–1.098 | 0.946–0.983 | 1.064–1.121 | 0.945–0.993 | 0.796–1.056 |

**All four adversarial verdicts 6/6:** reservoir beats the bag **6/6**, beats the non-recurrent projection **6/6**, beats the 4-gram **6/6**, vanilla (no Polyak/weight-decay/label-smoothing) still beats the bigram **6/6**. Every input-destruction anti-cheat collapses every seed (shuffled-state ~3.4, permuted-corpus ~1.4, frozen == chance, silenced ~1.9, all ≥ bigram); the region is genuinely active (~0.031 spikes/neuron/step).

## Why the dynamics claim is EARNED (the adversarial-verify's confound resolved)
The verify (`w3eviw8bv`, SURVIVES-WITH-SCOPE-FIXES) flagged that the running-cumulative state is a bag-of-counts by construction, so "beats the bigram" alone could be a trivial whole-prefix bag with no recurrence. The controls settle it:
- **BAG-OF-PREFIX (0.970):** DOES beat the bigram (1.075) — seeing the whole prefix helps — but the reservoir (0.830) beats the bag by **0.14 nats, 6/6**. ⇒ the recurrent dynamics carry MORE than an unordered prefix count.
- **NON-RECURRENT projection (1.090, internal_density=0):** barely differs from the bigram (actually slightly worse) — a memoryless fixed spiking projection of the current token carries no prefix context. The reservoir beats it by **0.26 nats, 6/6**. ⇒ the RECURRENCE (temporal memory), not just the spiking projection, is load-bearing.
- **4-GRAM (0.974):** the reservoir beats a well-smoothed order-3 n-gram **6/6** ⇒ genuinely beyond fixed-order context, not just the bigram.
- **VANILLA read-out (0.912):** still beats the bigram **6/6** ⇒ the calibration (Polyak/wd/label-smoothing) is a small symmetric eval-honest add-on, NOT the source of the win.

## Honest scope (the correctly-bounded claim — per the adversarial-verify)
This is a TOKEN-level LM over a BOUNDED controlled template grammar (V=24, a function-word scaffold with `<unk>` content; held-out = IID resampling of the same templates, NOT out-of-distribution). It licenses: "a fixed on-bridge spiking reservoir + a local one-step-delta read-out learns sequential structure beyond a bigram/bag/non-recurrent/4-gram, on the project's own spiking substrate, with the recurrent dynamics load-bearing." It does NOT license open prose / open-vocabulary fluency / OOD generalization — the honest ceiling is the reservoir's fading memory (~depth-3, EMERGE-84/85). Rollouts are grammatical scaffolds with `<unk>` content, not prose. This bounded-but-emergent result IS the deliverable (it maps exactly what the fixed-reservoir path reaches without deep credit).

## ⇒ significance
The FIRST validated rung of the emergent-generation ladder: a genuinely learned-from-experience, on-bridge, no-BPTT generative language mechanism, reusing the already-emergent reservoir — the answer to the whack-a-mole concern (learn to generate, don't hand-build). NEXT: RUNG 2 — condition the generation on the theta-gamma WM buffer (EMERGE-85) so a distal referent shapes the continuation (buffer-slot-scramble must collapse it).

## Files
`_emerge_reservoir_lm_derisk.py` (the `--controls` flag runs the bag/non-recurrent/vanilla/4-gram controls, additive/default-off); adversarial-verify `w3eviw8bv`; scoping `2026-07-10-emergent-language-cortex-scoping-the-generation-gap.md`.
