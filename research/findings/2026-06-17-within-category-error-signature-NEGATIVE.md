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

## Why it matters — the negative SHARPENS the generalization claim

The decisive sub-result is **raw_struct ≈ structured (5.0% ≈ 5.2%)**: removing the role-binding entirely does NOT
recover a strong within-category bias. So the binding is NOT washing out the structure — **the structure was
never in the raw nearest-neighbor geometry of the codes.** An isotropic perturbation of a code lands nearest a
*random* other code ~98% of the time, not a category neighbor.

This reconciles cleanly with the generalization arc (`2026-06-16-generalization-*`), where category generalization
was strong (held-out category accuracy **0.92**): there, the category was extracted by a LEARNED block-diagonal
category read-out + NMDA temporal integration — a **learned projection onto a category-read-out direction**, not
raw cosine distance over the full 300-dim code. So:

> The cortex codes' generalization lives in a LEARNED read-out subspace (a category direction a downstream region
> learns to read), NOT in raw code proximity. The conversational binder's cleanup is raw-nearest-neighbor, so its
> recall errors are near-random — NOT semantically biased.

This prevents a tempting overclaim ("the agent confuses dog↔cat like a person would") that the data do not
support, and it precisely localizes generalization: it is a property of a *learned read-out*, demonstrated in the
perception→concept pathway, and it does NOT automatically express itself as semantic structure in the
conversational binder's error pattern.

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
