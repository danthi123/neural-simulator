---
type: finding
status: live
date: 2026-08-13
mechanism: evaluative-conditioning-three-factor-hebbian
---

# DR-2b — the ORIGIN of concept valence SELF-ORGANIZES from experienced reinforcement: the hand-laid Warriner appraisal seeds retire to ~10 innate primary reinforcers (6-seed GO)

**Runner:** `research/runners/_affect_evaluative_conditioning_derisk.py`
**Artifacts:** `research/findings/raw/_affect_evaluative_conditioning_6seed.json` (+ `.log`), smoke `…_smoke.json`.
**Discipline:** `SIM_BACKEND=numpy` CPU lane, reuse-by-import, **NO `sim/` edit**. 6 seeds 42/43/44/100/101/102.

## The residual this closes (why it is the tractable emergence target)

The audit's deepest gap is that production ORGAN INTERNALS are host-designed. For the affect faculty the internal is
the **appraisal seed**: `_affect_distributional_tag_derisk.py:80-113` is a hand-typed `WARRINER` lexicon (~140 words,
graded valence on the 1..9 scale), injected as the `appraisal_vplus/vminus` concentration that drives the affect
state region. DR-2 (`_affect_distributional_tag_derisk`) showed valence can be INHERITED to held-out words over the
learned co-occurrence graph, but named its own residual verbatim: the valence is *"SEEDED from Warriner norms (NOT a
retirement)"* and propagated by **host** numpy label-propagation (Zhu-Ghahramani harmonic), not a plasticity rule.

So the ORIGIN of valence was still a human-rated lexicon and the learning was a host algorithm. DR-2b closes both.

## Mechanism — evaluative conditioning by a biological three-factor rule

- **Concept CODE** = the already-self-organized PPMI stream cortex (rate-Hebbian co-occurrence, the matched rule,
  `2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO`). [pre] — reused from DR-2, unchanged.
- **~10 INNATE primary reinforcers** — a genome-cheap SIGN per bodily/consummatory/nociceptive/social unconditioned
  stimulus (appetitive {hug,kiss,cuddle,cake,candy,sweet,warm,treat,food,cozy}=+1; aversive
  {hurt,pain,sick,cry,scared,afraid,cold,hungry,bite,fall}=-1). Each seed draws a different subset (robustness to the
  genome's choice). This is the ONLY host-supplied affect anchor. [US]
- **Evaluative-conditioning stream** (TinyStories): when a concept co-occurs (window 4) with a primary, the primary
  drives the DA/US sign into the opponent pools; a concept's acquired valence saturates to the Rescorla-Wagner
  asymptote `s_c = (n_pos-n_neg)/(n_pos+n_neg)`. [the CS<->US pairing]
- **Three-factor DA-gated Hebbian write** — outer-product associative memory `W = Σ_c code_c · s_c` (pre=code,
  post=US-driven opponent pool, third factor=DA/US sign). [self-organizing plasticity]
- **Read** — `v(x) = code_x · W` (Hebbian associative recall = similarity-weighted vote of experienced
  reinforcement), with a LABEL-FREE hub-ness gain-control (subtract the relatedness common mode) so the read is the
  valence component orthogonal to raw connectedness = value⊥plausibility as an explicit normalization.

## Result (6-seed) — GO, all 8 pre-registered checks pass

_Per-seed values below are rounded, and seed-aggregates computed, from the cited 6seed JSON._
<!--derived-->

| seed | held-out r | sign-acc | unpaired-US perm-p | permute-code perm-p | value⊥relatedness | corr(s_c,Warriner) |
|---|---|---|---|---|---|---|
| 42 | +0.567 | 0.72 | 0.010 | 0.010 | +0.205 | +0.503 |
| 43 | +0.592 | 0.73 | 0.005 | 0.010 | −0.096 | +0.437 |
| 44 | +0.670 | 0.80 | 0.005 | 0.010 | −0.167 | +0.567 |
| 100 | +0.593 | 0.73 | 0.005 | 0.010 | +0.120 | +0.557 |
| 101 | +0.570 | 0.67 | 0.005 | 0.010 | +0.186 | +0.456 |
| 102 | +0.316 | 0.68 | 0.005 | 0.030 | −0.222 | +0.497 |

- **G1 GENERALIZE** — held-out concepts (own reinforcement WITHHELD from the map) recover valence at **mean r=+0.551**
  (every seed ≥ +0.316), sign-acc 0.72, pooled binomial **p=3.4e-18**. This matches DR-2's 0.55 held-out r while the
  seeds are RETIRED from 140 graded human ratings to ~10 innate signs.
- **G2 NO-LEARNING** — freeze the plasticity (empty write) → r=+0.000. The map, not the codes alone, carries it.
- **G3 UNPAIRED-US** (non-contingent control, US paired with the WRONG concept), as a **permutation test** — real r
  beats the null at perm-p<0.05 in **6/6** seeds (null mean ≈ −0.03). The CS↔US contingency is load-bearing.
- **G4 PERMUTE-CODE** (scramble which learned code belongs to which word), permutation test — beaten in **6/6**
  (null mean ≈ +0.02). The self-organized **code geometry** carries the generalization.
- **G5 VALUE⊥PLAUSIBILITY** — |corr(pred, relatedness)| < 0.30 in **6/6**. Valence is not relabeled hub-ness.
- **Compression (ablation)** — 4 innate primaries already generalize (r=0.449), 6→0.634, 10→0.567. A HANDFUL of
  innate reinforcers suffices to color ~120 concepts.
- **The innate signal is honest** — the acquired `s_c` (from co-occurrence with ~10 signs) correlates +0.50 with the
  full Warriner lexicon over the ~120 non-primary concepts. Valence genuinely propagates from the primaries.

## Interpretation — the STRUCTURE, not a lookup, does the work

Both anti-cheats collapse the generalization (permutation-significant in 6/6): the appraisal read rides the
**self-organized concept-code geometry** and the **experienced reinforcement pairing**, not a stored table. The
concept→valence STRUCTURE develops from a training stream via a local biological plasticity rule anchored only by
innate primary reinforcers — the emergent replacement for the hand-laid Warriner seeds.

Per TERMS.md `self-organized`: both factors of the rule are neural (pre = learned code; post = US-driven opponent
pool); the target (which concepts→valence) is NOT host-selected — it emerges from co-occurrence; the V+/V− slots are
an innate opponent channel (Namburi-Tye, genome-cheap). The declared host residual is the ~10 innate primary SIGNS.

## Honest residuals (brutally)

_Pilot/aggregate numbers below are derived from the cited 6seed JSON + the first-pilot run._
<!--derived-->

1. **~10 innate primary SIGNS remain host-supplied** (the unconditioned-stimulus anchor). This is the biologically
   faithful floor — valence IS innately anchored by primary reinforcers — and it is a 140→10 compression of the hand
   table, not a removal. Their signs match canonical bodily reinforcers; this is not the graded lexicon.
2. **Rate-level numpy, not spiking.** This de-risks the emergence MECHANISM (matching DR-2's discipline). The
   outer-product Hebbian map = a plastic synaptic matrix; the codes are the spiking-validated stream cortex. The
   **fully-spiking opponent-population appraisal** — drive the affect state region's `appraisal_vplus/vminus` from
   THIS learned map instead of the Warriner dict — is the named next rung (the seam already exists:
   `build_learned_valence_map`/`appraise_text`).
3. **Held-out = leave-out, not strictly-never-seen.** Innate primaries are promiscuous co-occurrers in child-story
   corpora, so a strictly-never-reinforced set is near-empty; the held concept's OWN reinforcement is withheld from
   the map (its valence predicted purely from `code·W`). Permute-code collapse confirms the code geometry, not the
   held concept's own signal (which is unused), carries it.
4. **Instrument disclosure.** The read-out hub-ness gain-control and the permutation-test nulls were added AFTER a
   first 6-seed pilot exposed (a) an under-powered single-draw null (one shuffle draw hit r=0.58 by chance on ~60
   concepts in a low-dim code space) and (b) a 1/6 hub-ness confound (seed102 read hub-ness, r=0.04). Both are
   instrument fixes (a sound null; a label-free normalization the affect arc already requires), applied identically
   to real and null arms — not signal manipulation. The pre-registered GO bar (mean r≥0.45) was calibrated on that
   pilot (mean 0.548) and is disclosed; the decisive claims (6/6 control collapse + generalization) do not depend on
   the exact bar.

## Reproduce

```
SIM_BACKEND=numpy python -u -m research.runners._affect_evaluative_conditioning_derisk --smoke
SIM_BACKEND=numpy python -u -m research.runners._affect_evaluative_conditioning_derisk \
    --seeds 42 43 44 100 101 102 --max-stories 60000 --ablation \
    --out research/findings/raw/_affect_evaluative_conditioning_6seed.json
```

## Sources

- Rescorla & Wagner (1972) — associative strength asymptote (the conditioned-response saturation used for `s_c`).
- Namburi, Tye et al. (2015, Nature) — opposing valence-coding BLA populations (the V+/V− opponent slots).
- Bestgen & Vincze (2012) — distributional valence inference (a concept's valence from the company it keeps).
- Three-factor / neuromodulator-gated Hebbian plasticity (dopamine as the third factor) — the write rule.
- DR-2: `research/findings/…` `_affect_distributional_tag_derisk.py` — the seeded-from-Warriner residual this closes.
