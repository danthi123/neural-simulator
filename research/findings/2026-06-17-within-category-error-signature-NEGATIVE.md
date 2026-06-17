# Within-category-error "generalization signature" in the conversational binder — NEGATIVE (and it sharpens the generalization claim)

**Date:** 2026-06-17
**Status:** **NEGATIVE, 3 seeds — an honest negative that PREVENTS an overclaim and localizes where generalization
actually lives.** Follow-on to the CYCLE-120 consolidation GO.

## The question

The consolidation GO showed the production conversational agent recalls perfectly on the semantically-structured
320-concept stream-learned codes. Natural hypothesis: because those codes are semantically organized (the
structure that powers the cortex's generalization), the binder — when its read-out is noisy and recall drops —
should err SEMANTICALLY SENSIBLY (confuse dog with cat, a within-category error), not randomly (dog with
airplane). A brain-like generalizing memory should make within-category mistakes. Does it?

This is moat-respecting: the errors are on STORED facts under a degraded read-out, never confabulation on
unstored queries (the no-confab moat is untouched).

## The test (`research/runners/_genfrontier_within_category_conversation_signature.py`, CPU, numpy HRR)

Bind a fact `F = R_agent ⊗ c_i + R_action ⊗ c_j + R_object ⊗ c_k`, add Gaussian read-out noise of scale σ,
unbind a role, clean up by nearest of all 320 codes, classify the answer correct / within-category /
cross-category. Sweep σ to walk recall from 1.0 down to ~0.16 (so errors are plentiful and the within-category
fraction is well-estimated). Chance within-category among errors = (8−1)/(320−1) = **2.2%**. Four arms:

- **structured** — the real stream-learned codes, full bind/unbind pipeline.
- **raw_struct** — the structured codes but NO binding (noise added directly to the filler code, then cleanup).
  Isolates the CODE GEOMETRY from any wash-out the role-binding introduces.
- **random** — decorrelated codes (control: should be ≈ chance).
- **deranged** — structured codes with the category labels shuffled (control: should collapse to chance).

## Result (regime-mean within-category fraction among errors, recall 0.2–0.95, 3 seeds)

| arm | within-category among errors |
|---|---|
| structured (bound) | **5.2%** |
| raw_struct (no binding) | **5.0%** |
| random (control) | 2.3% |
| deranged (control) | ~2.3% |
| chance | 2.2% |

The structured codes carry a WEAK residual within-category bias (~5% vs 2.2% chance, ~2.3× chance) — real but
far below the "semantically sensible confusion" hypothesis (which would predict tens of percent). It does not
clear the pre-registered GATE (≥20% absolute and ≥2.5× the controls). **NEGATIVE.**

## Why it matters — the negative, with the CORRECTED mechanism (a small-margin structure overwhelmed by noise)

The decisive sub-result is **raw_struct ≈ structured (5.0% ≈ 5.2%)**: removing the role-binding entirely does NOT
recover a strong within-category bias, so the binding is not the cause.

**Correction (substantiated `2026-06-17`, `_genfrontier_learned_vs_raw_category_readout.py`).** My first
explanation here — "the structure was never in the raw nearest-neighbor geometry" — was WRONG, and a direct check
on the same 320 codes falsified it. Leave-one-out nearest-neighbour category accuracy is **~21% (8.4× the 2.5%
chance)**: the category structure IS present in the raw geometry. A learned linear read-out recovers only modestly
more (**~26%**), and the deranged-label control sits at chance (~2.3%). So the correct mechanism is:

> The 320 stream codes carry REAL category structure in raw proximity (kNN 21%), but it is **SMALL-MARGIN**: a
> same-category code sits at phase-cosine ~0.13 from the true code, versus 1.0 for the true code itself. The
> read-out noise required to produce a binder recall error (recall pushed below ~0.6) is large enough to
> overwhelm that thin margin, so the wrong pick lands near-random (~5% within-category, ~2.3× chance). The
> structure is there; the noise that causes errors simply dwarfs it.

This still prevents the tempting overclaim ("the agent confuses dog↔cat like a person would") — the conversational
binder's recall errors are NOT meaningfully semantic — but for the right reason: a real-but-thin category margin
swamped by error-inducing noise, NOT an absence of structure. It also tempers the "learned read-out" framing: a
linear read-out (26%) barely beats raw (21%) on these codes, so generalization is not uniquely a "learned
direction" here. The strong perception→concept generalization (held-out cat-acc **0.92**,
`2026-06-16-generalization-*`) came from a richer pathway — NMDA temporal integration + a block-diagonal category
read-out over **similarity-structured perception input** — not a linear read-out on these stream codes.

## Honest scope / what this is not

This does not weaken anything: the consolidation GO stands (perfect recall, moat intact), and the generalization
arc stands (cat-acc 0.92 via the learned read-out). It is a clean boundary on a *specific* further hypothesis
(within-category conversational recall errors), with the mechanism localized by the raw-vs-bound control. Honest
negatives are the deliverable.

## Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners._genfrontier_within_category_conversation_signature --seeds 42 43 44 --readout host
```

No `sim/` edit. Reuse-by-import: `hrr_bind/unbind/_cos`, `TAXONOMY_40x8`, the cached 320 stream codes.
