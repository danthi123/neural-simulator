# Deep-credit real-task, part 6 (transitive inference) + the ARC PATTERN — transitive inference is NOT depth-required (any monotone scalar score is transitive → shallow); this completes a clear pattern: supervised deep-credit's depth-benefit lives in NONLINEAR CONJUNCTION / BINDING, and the "natural" perceptual/semantic/relational tasks are all shortcut-able (convolution / linear-embedding / scalar-score). The language-relevant depth capability where deep credit matters is ROLE-FILLER BINDING / systematic recombination.

**Date:** 2026-07-08
**Runner:** `research/runners/_transitive_inference_deep_credit_derisk.py` (reuse-by-import of the part-2 harness; NO `sim/` edit). Self-correcting Stage-0 gate.
**Verdict:** honest boundary (Stage-0 NOT depth-required, correctly did not run Stage-1) + the arc-completing pattern.

## Stage-0 — transitive inference is NOT depth-required (robust across ~7 configs)
Arbitrary-code entities in a linear order; train ADJACENT pairs, test held-out NON-ADJACENT. DEPTH-SEPARATING False everywhere: either the linear probe solves held-out non-adjacent (≈1.0, depth-gap ≤0 — deep nets OVERFIT the arbitrary codes and underperform), or noise/dim destroys the signal for all (≈chance). No window where 1-layer underfits + deep succeeds.
**Root cause:** the "greater-than" relation over a linear order is realizable by a MONOTONE SCALAR SCORE, and ANY scalar score is automatically transitive — so generalizing to held-out non-adjacent pairs needs only a shallow rank-score + comparison; the net never CHAINS. (This is why EMERGE-28 needed an explicit autoregressive HTM chain-rollout to do transitive inference UNSUPERVISED; a supervised MLP shortcuts to a score-comparator.)

## The ARC PATTERN (the load-bearing map — where supervised deep-credit depth does + does NOT live)
| real task | outcome | the shortcut that makes it shallow |
|---|---|---|
| CIFAR (perception) | wrong instrument | depth is CONVOLUTIONAL, not FC |
| raw-PPMI (category-inheritance) | linear | word embeddings make categories LINEARLY decodable (Levy-Goldberg) |
| transitive inference (order) | scalar-shortcut | any MONOTONE SCALAR SCORE is transitive |
| **part-2 XOR-over-pool (conjunction)** | **DEPTH-REQUIRED, rule learns it (0.69, 5/6)** | **none — nonlinear conjunction, per-item linearly uninformative** |

⇒ **supervised deep-credit's depth-benefit is real but NARROW: it requires a target that is a NONLINEAR CONJUNCTION / BINDING** — one that resists both the per-item-scalar-score shortcut and the linear-decode shortcut. The "natural" perceptual/semantic/ordinal tasks all have such shortcuts, so they do NOT exercise deep-credit depth. The genuine depth-required LANGUAGE capability where deep credit would matter is **ROLE-FILLER BINDING / systematic recombination** — where the answer depends on the CONJUNCTION of role and filler (not a per-item score, not linear). This is exactly the VSA-composer / EMERGE binding territory the project already develops.

## What this establishes for the deep lever (comprehensive)
The deep-credit mechanism is validated (ports to spikes; works at rate on the part-2 nonlinear-conjunctive composition). The real-task arc has comprehensively mapped WHERE its depth-benefit lives (nonlinear conjunction/binding) and where it does NOT (the shortcut-able natural tasks). ⇒ the deep-lever's genuine value for LANGUAGE is as a LEARNED binder/composer for role-filler conjunction — the exact capability the fixed VSA-algebra composer does by hand, and which the mission's lead orientation targets (replace the VSA scaffold with learned circuitry). NEXT: does supervised deep credit LEARN role-filler binding (the culminating, most-language-relevant, genuinely-depth-required test)?

## Files
`research/runners/_transitive_inference_deep_credit_derisk.py`; `research/findings/raw/_transitive_inference_seed42_smoke.json`. Prior arc: `2026-07-07-deep-credit-real-task-{cifar-fc-vision-wrong-instrument,compositional-semantics-GO}.md`, `2026-07-08-{onbridge-population-coding-Ksweep-*,deep-credit-real-word-inheritance-is-linear-*}.md`.
