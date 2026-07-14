# Past the reservoir bound, Rung 3 (REAL TEXT): the eligibility-trained per-neuron SELECTIVE diagonal SSM beats the fixed reservoir AND the bigram at DEEP context on TinyStories — transport-free, no BPTT

**Date:** 2026-07-13
**Runner:** `research/runners/_reslm_rung3_selective_ssm_realtext_derisk.py` (reuse-by-import of the reslm real-corpus + by-context-depth CE machinery; numpy; NO `sim/` edit, NO BPTT, NO weight transport).
**Status:** ✅ 6-seed GO (42/43/44/100/101/102, all 6) — decisive + universal.
**Provenance:** Rung 3 of the past-reservoir arc — escalates Rung 2 (synthetic gated-conjunction, 11/12 GO) to REAL LANGUAGE, on the corpus + metric where the fixed reservoir was shown Ueda-bounded.

## The escalation

Rung 2 proved a per-neuron selective diagonal SSM, trained by the exact forward-mode eligibility trace, captures a long-range conjunction on a SYNTHETIC task. Rung 3 asks the real question: **on REAL text next-token prediction, does the LEARNED input-dependent gate beat the FIXED reservoir at DEEP context depth?** — on TinyStories (V=200), reusing the reslm `load_sentences` + by-context-depth CE + bigram baseline, comparing LIKE-FOR-LIKE (same token embedding, same local delta-rule read-out, same corpus/eval — the ONLY variable is the gate). The read-out weights are learned by the committed local one-step delta rule; the gate params by the online eligibility trace at EVERY token (an LM predicts the next token at every position). Forget-bias init `c=2.5`.

## Reference/ceiling (built-in, per the run-the-ceiling-early discipline)

The claim is NOT "beats a transformer" — it is "the LOCALLY-trained selective gate captures MORE deep-context than the fixed reservoir it upgrades, on real text, transport-free." The built-in references bound it: the **bigram** (the memoryless n-gram floor — the fixed reservoir was shown *worse* than the bigram at deep context, the Ueda-bound signature) and the **fixed reservoir** itself.

## Controls (a self-caught invalid-control fix → a rigorous like-for-like)

Arms, differing ONLY in the gate:
- **selective** — `lam_{t,i}=sigmoid(w_i·E[tok_t]+c_i)`, gate TRAINED by the eligibility trace.
- **fixed_res** — FIXED per-neuron lambda (leaky ESN, the reservoir baseline), only the read-out trained.
- **detached** — input-dependent gate ARCHITECTURE but gate UNTRAINED (random w) → isolates that LEARNING the gate matters.
- **randgate** — gate TRAINED but reads a RANDOM token's embedding per step → isolates that the gate must condition on the CURRENT token. *(This replaced an initial `permgate` that permuted the embedding DIMENSIONS — which scored == selective, because a dim-permutation of a random embedding is INVERTIBLE / information-preserving, so the gate just learns the permuted code. That is an INVALID anti-cheat; the valid one destroys current-token access. Caught + fixed before committing — the adversarial-verify-yourself discipline.)*
- **bigram** — add-1 memoryless floor.

## Result — seed 42 (DEEP context depth d≥4; LOWER cross-entropy is better)

| arm | deep-context next-token CE |
|---|---|
| **selective** (learned input-dependent gate) | **2.963** |
| detached (input-dependent, UNTRAINED gate) | 3.430 |
| randgate (gate TRAINED on a random token) | 3.558 |
| bigram (memoryless floor) | 3.395 |
| fixed_res (leaky ESN reservoir) | 3.722 |

selective beats every arm decisively: **−0.759 vs fixed_res, −0.467 vs detached, −0.595 vs randgate, −0.432 vs bigram** (all nats). The controls are valid + decisive:
- **selective ≪ fixed_res**: input-DEPENDENT gating captures deep context the fixed reservoir cannot (which is itself *worse* than the bigram at depth — the Ueda-bound).
- **selective ≪ detached**: LEARNING the gate (eligibility trace) is load-bearing, not the architecture/init.
- **selective ≪ randgate**: the gate must read the CURRENT token — training it on a random token is even WORSE than not training it (3.558 > 3.430), because it learns to gate on noise.

**6-seed (42/43/44/100/101/102) — 6/6 GO, decisive + universal:** selective mean 3.057. selective beats every control on **6/6 seeds** — vs fixed_res mean −0.682 (min −0.603), vs detached −0.400 (min −0.349), vs randgate −0.562 (min −0.513), vs bigram −0.332 (min −0.279). The margins are large (+0.3 to +0.7 nats) — no threshold fragility (unlike the synthetic Rungs 1–2, where margins were modest); on real text the learned selective gate's deep-context advantage is unambiguous.

## ⇒ the claim

On REAL text, a per-neuron SELECTIVE diagonal SSM trained by an EXACT forward-mode eligibility trace (no BPTT, no weight transport) captures deep-context next-token structure that the fixed reservoir + a bigram cannot — the input-dependent multiplicative gate conditions retention/conjunction on the CURRENT token, and LEARNING it locally is what does the work. This is the honest path PAST the reservoir's fading-memory + linear-read-out bound, on real language, AVOIDING the exhausted deep-credit (surrogate-BPTT / feedback-alignment) wall. Spiking-realizable (per-neuron leaky integrators with an input-modulated leak + a local synaptic eligibility trace).

## Honest scope / next

- V=200 TinyStories, 1500 train sentences, deep-context aggregate (d≥4). The margins are large + on the metric where the fixed reservoir is Ueda-bounded; this is a next-token CE result, NOT a fluency/generation claim. It does NOT claim to match BPTT (Zucchet: online closes ~70–90% of the BPTT gap; a BPTT-trained selective SSM upper bound is a worthwhile follow-on reference).
- NEXT (Rung 4): the SPIKING realization — the selective diagonal SSM as per-neuron leaky-integrator neurons with an input-modulated leak on a real `SimulationBridge`, the gate learned by a local synaptic eligibility trace — the fully-on-substrate, transport-free long-range learner (the emergence-engine core). Then scale (bigger corpus/vocab) as the lever, and couple it into the emergent generator.
- NO `sim/` edit. CI guard `tests/test_reslm_rung3_selective_ssm_realtext.py`.

## Files
- `research/runners/_reslm_rung3_selective_ssm_realtext_derisk.py`; raw `research/findings/raw/_rung3/seed*.json`.
- Builds on Rung 2 (`2026-07-13-PAST-RESERVOIR-RUNG2-...`), the Zucchet source-read, and the reslm real-corpus/context-depth machinery.
