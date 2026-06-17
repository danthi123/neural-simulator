# Multi-referent disambiguation — NEGATIVE on the plain WM loop (it needs an attention/salience pointer)

**Date:** 2026-06-17
**Status:** **NEGATIVE, 3 seeds — and it cleanly bounds the multi-turn capability.** The single-referent case is
GO (shipped: the `MultiTurnAgent`). When the working memory holds *several* referents, a bare pronoun ("it")
cannot be disambiguated by recency on the plain spiking loop — which referent dominates is seed-dependent
attractor competition, not recency. Multi-referent disambiguation needs an added salience/attention mechanism.

## The question

A bare pronoun usually binds the **most recent** salient referent ("the dog saw the cat. it ran." → it = the
cat). The multi-turn de-risk (`2026-06-17-multiturn-anaphora-derisk-GO.md`) flagged this as the honest next
stress. Does the spiking WM loop carry a **recency gradient** (the most-recently-written referent dominates the
read), so a bare pronoun binds it — or does it hold all referents as an equal **set** (the validated ≥3-concept
hold), leaving a bare pronoun genuinely ambiguous?

## Result (`_phaseB_multireferent_disambiguation_derisk.py`, 3 seeds, CPU)

| seed | NATURAL (write cat, then bird) recent=bird | ORDER-CTRL (write bird, then cat) recent=cat | REFRESH bird again |
|---|---|---|---|
| 42 | bird 0.33 ≫ cat 0.03 → recent wins | cat 0.00 vs bird 0.33 → **older wins** | bird 0.32 > cat 0.07 → recent |
| 43 | cat 0.42 > bird 0.22 → **older wins** | cat 0.11 < bird 0.28 → older wins | bird 0.275 ≈ cat 0.264 → tie |
| 44 | cat 0.315 ≈ bird 0.335 → tie | cat 0.32 ≈ bird 0.335 → tie | tie |

**NEGATIVE (0/3 natural, 1/3 refresh).** There is no reliable recency gradient: in NATURAL the recent referent
wins only on seed 42; on seed 43 the *older* one wins, on seed 44 it's a tie. The order-control does not flip the
winner (it should, if recency were the driver). So **which referent dominates is seed-dependent attractor
competition, not recency** — the loop holds both as a set (consistent with the validated multi-concept hold), and
re-driving the recent one once (REFRESH) creates the gradient on only 1/3.

## Reading it honestly

- **The single-referent multi-turn case is and stays GO** (the `MultiTurnAgent` resolves "it" when one referent
  is held, 3 seeds). This negative does not touch that.
- **Multi-referent disambiguation is the genuine wall for multi-turn dialogue on the plain loop.** A bare pronoun
  among several held referents cannot be resolved by the loop's own dynamics — the WM stores the *set*, not a
  ranked salience. This is mechanistically sensible: a content-addressable attractor memory holds *what* is in
  mind, not *which* is currently most salient.
- **The next mechanism is precise and biologically grounded (a buildable follow-on, not a dead end):** an
  explicit **salience / attention pointer** that tags the most-recent (or grammatically-foregrounded) referent —
  e.g. a transient gain boost on the salient attractor, or a separate one-hot "attentional spotlight" population
  that gates the read. Biology: attentional selection of a discourse referent (the brain does not bind a pronoun
  by raw memory strength; it uses attention/saliency). That is the right next de-risk if multi-referent dialogue
  is pursued.

## Where this leaves multi-turn dialogue

- **GO:** single-referent anaphora across turns (production `MultiTurnAgent`).
- **Mapped boundary:** multi-referent disambiguation needs an added attention/salience pointer (this finding).

An honest boundary that names the exact missing mechanism is the deliverable.

## Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners._phaseB_multireferent_disambiguation_derisk --seeds 42 43 44
```

No `sim/` edit. Reuse-by-import: `SpikingLoopContextBuffer`.
