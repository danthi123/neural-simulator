# Pre-compute review: the spiking-vs-tiny-LLM gap is ALREADY measured — do NOT re-run it — 2026-06-03

**One line:** Before investing GPU time to measure how far a biology-faithful spiking generative model gets
versus a tiny LLM, a thorough review (3 parallel code/findings/baseline audits) found the project has
**already run this measurement multiple times, at larger scale, with rigorous controls** — and the answer is
a multiply-confirmed NEGATIVE. Re-running it would waste the compute the review was meant to protect.

## What the review found (the measurement is done)

The spiking generative LM was scaled and benchmarked against a same-corpus, same-hardware tiny Transformer,
with held-out perplexity + shuffled-text control + an absolute-competence floor:

| Model | Params | Held-out perplexity | Outcome |
|---|---|---|---|
| Spiking surrogate-grad BPTT LM (subword) | **25M** | **~203,753** (200× worse than random) | token-soup |
| Spiking LM (50M confirmation) | ~50M | token-soup (confirmed) | NEGATIVE |
| Spiking LM **distilled** from a competent teacher (Generator-D) | ~0.3M | 804 (still worse than random), 0/3 seeds | NEGATIVE |
| **Tiny Transformer (Generator-F), same corpus + 3090** | **~3.45–6M** | **~6.1** (84× better than random) | **coherent simple English** |

**The gap is ~33,000× in held-out perplexity, and it is ARCHITECTURAL, not a scale problem:** scaling the
spiking model 100× (0.3M→25M) and 8× data made train loss fit but held-out *worse* (overfit). A **4×-smaller
transformer generalizes where the 25M spiking net does not.** Phase 2 independently refuted the scale thesis
(50M spiking → direction-word cosine 0.85, *worse* than the 134K model's 0.72).

## This is the field's wall, not ours (confidence signal)

- SpikeGPT (45M/216M, direct surrogate-grad BPTT) reports our **exact** overfit-at-scale phenomenon and
  makes no coherence claims.
- Every *capable* spiking LM is an **ANN→SNN conversion/distillation** of a pretrained transformer
  (SpikeLLM, SpikingBERT, SpikeLM, FAS) — "capability is borrowed from the ANN."
- Competent general conversation empirically needs **~360M–500M params + trillions of tokens** — beyond
  from-scratch biology-faithful single-GPU training.

Sources (existing findings, all pre-dating this review): `2026-06-02-generative-ceiling-spiking-LM-NEGATIVE-overfit-not-size.md`,
`2026-05-17-generator-D-distillation-NEGATIVE.md`, `2026-05-17-generator-F-small-transformer-LM-PASS.md`,
`2026-05-17-conversational-capability-program-META-TERMINUS.md`,
`2026-05-09-Phase-2.3b-50M-cosine-REFUTED.md`,
`2026-06-03-deep-research-how-the-field-gets-past-our-generative-conversation-wall.md`.

## Decision: do NOT spend the compute re-measuring a settled result

"Scale up, from scratch + biology-faithful + single-GPU, to match a tiny LLM's *fluent generation*" is a
**documented dead-end** — and scaling makes it worse (overfit), so there is no scaling fix. Re-running the
gap measurement (even a cleaner param-matched char-Shakespeare version) would only re-confirm the documented
answer. That is not the best use of the compute the owner asked me to protect.

## Where compute SHOULD go (with the project goals in mind)

The project's top goal is "artificial life with proper brain analogue, biology-translatable insights; honest
negatives under strict biology are the deliverable; capabilities are instrumental." Under that goal, the
fluent-generation wall is an *honest negative already banked*. The distinctive, defensible, and genuinely-open
biology-faithful contribution is **composition + memory + no-confabulation** — exactly the arc advanced this
session (the phasor resonator decoder, nesting, learned-code memory, abstention). The two genuinely-untried
biology-faithful levers from the deep-research doc (resonator decoder — RESOLVED this session — and
thalamocortical dynamical gating) target *composition/decode*, not generation.

**Three honest forks for the owner to steer (the fluent-generation wall is real; pick with eyes open):**
1. **Advance the biology-faithful composition frontier** (recommended, goal-aligned): scale + deepen the
   phasor compositional substrate (the genuine "proper brain analogue" contribution); thalamocortical gating
   as the next untried lever. Structured, grounded, no-confab conversation — not free-form fluency.
2. **Hybrid** (if fluent prose is a product requirement): our validated VSA retrieval + no-confab abstention
   (the hard, distinctive half) as the memory for a small transformer generator (Generator-F exists). This
   relaxes biology *for the generator only*; the findings already identify it as the pragmatic path.
3. **Cloud-scale generative** (~360M+ params, trillions of tokens): the only documented route to real-LM-class
   fluency; real cost, abandons single-GPU from-scratch biology.

**Recommendation:** do not burn GPU on the settled measurement. Point compute at fork 1 (or fork 2 if the
owner wants prose fluency as a product). The review converted "we need to scale to a tiny LLM" from an
assumption into the documented fact that, biology-faithfully and from scratch, **we can't scale to it — and
that honest negative is already the deliverable.** The strength to build on is composition + memory + trust.
