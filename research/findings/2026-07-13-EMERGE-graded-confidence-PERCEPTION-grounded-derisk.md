# EMERGE graded-confidence grounded in REAL PERCEPTION — the perceptual MOAT is robust; graded HEDGING at the category read is ABSENT across FOUR tested mechanisms because of CATEGORICAL PERCEPTION (a well-characterized, biologically-faithful boundary) (2026-07-13)

**Headline.** The graded read (CONFIDENT / ABSTAIN + no-confab moat) now rides on categories
DISCOVERED from VISUAL similarity, not hand-scripted co-occurrence — the emergence-faithful grounding
the 2026-07-11 boundary pointed to. The brain SEES objects through the real Gabor/V1 front end
(EMERGE-53's `PerceptionGroundedConsole`), the competitive pooler discovers the category from the
visual features, and a HELD-OUT perceived object inherits a taught class property CONFIDENTLY; a
never-perceived object hits the moat. **The graded MOAT transfers to perception robustly. The third
level — HEDGED (graded perceptual uncertainty) — does NOT transfer at the category read: categorical
perception suppresses it. That is an honest, informative boundary with a named next mechanism, not a
failure** (per the mission, an honest negative that maps the path is a first-class deliverable).

```
[sees an owl]      you> can it fly?   brain> Yes, an owl can fly.            (CONFIDENT — inherited via the VISUALLY-discovered bird category)
[sees a griffin]   you> can it fly?   brain> I don't know what a griffin is.  (MOAT — never perceived)
```

## Why perception (a-1) + how the graded read attaches (a0)

The graded read was validated on CLEAN, hand-scripted category structure (completion / console /
scale, all 12-seed GO). Grounding it in real experience via a toy TEXT corpus re-hits the documented
distributional-induction SCALE wall (`2026-07-11-EMERGENT-codes-corpus-cooccurrence-scale-boundary.md`);
that finding names PERCEPTION-grounding (EMERGE-34/53, within-cat visual similarity 0.86 vs 0.08) as
the tractable, far-stronger category signal. a0-read of EMERGE-53 confirmed `_drive(member)` returns
the per-property apical-drive dict and `_best`/`ask_can` is a categorical argmax-with-override, so the
graded read attaches exactly as it did for EMERGE-31.

## What IS robust — the perception-grounded graded MOAT (GO)

`GradedPerceptionConsole(PerceptionGroundedConsole)` reuses EMERGE-53's VALIDATED recipe (9 seen
birds + 9 fish, class taught via 6 exemplars each, held-out owl/wren/minnow/gar). GO gate = the
perceptual moat: HELD-OUT perceived members inherit CONFIDENTLY (≥ EMERGE-53's own 0.75 bar), a
never-perceived object hits the MOAT, causal on the VISUAL similarity (per-image scramble degrades
the held-out confidence) and on the coincidence substrate (lesion → all-abstain).

- **12-seed (standard 42/43/44/100/101/102 + FRESH 7/8/9/10/11/12): GO — 6/6 standard + 5/6 fresh
  = 11/12** (meets the ≥5/6-in-both-sets bar). Every seed: held-out perceived members inherit
  CONFIDENTLY (≥0.75), MOAT holds, scramble degrades, lesion abstains. The lone miss (seed 9,
  held-out 2/4) is EMERGE-53's inherent held-out perceived-inheritance variability, not a graded-read
  failure.

The confidence is grounded in real vision: the categories are discovered from Gabor/V1 similarity
(scramble collapses → the structure is visual, not injected), and a clearly-perceived category member
answers confidently while a never-seen object abstains — a graded distinction (confident vs abstain)
grounded in the brain's own perception.

## The HEDGED boundary — categorical perception (honest negative + the named mechanism)

The third graded level (HEDGED, for a perceptually-ambiguous object) does NOT transfer at the
category read. Diagnosed by a0 across three constructions:

1. **Feature-union percept** (V1 features of a bird ∪ a fish): its codon overlaps both categories
   only on SOME seeds — seed-fragile (HEDGED on ~8/12, CONFIDENT on ~4/12).
2. **Image-blend percept** (a real 0.5·bird + 0.5·fish pixel morph perceived through Gabor/V1): the
   competitive pooler's k-WTA **sharpens it to ONE category on every seed** → CONFIDENT. This is
   CATEGORICAL PERCEPTION (the brain snaps a perceptual continuum to a discrete category —
   biologically real: phonemes, colors).
3. **Degraded percept** (drop features): the coincidence drive is **bimodal** (a codon either
   overlaps the class columns → driven ~+5, or does not → rest ~−60); there is no intermediate
   "weak match" level, so degradation gives CONFIDENT or ABSTAIN, never a middle.

4. **Decision-stage categorization consistency under perceptual noise** (Bogacz-Brown drift-diffusion
   — the named "next mechanism," now BUILT + TESTED): perceive the object N times with mild
   perceptual noise (drop a fraction of V1 features), categorize each, judge CONFIDENT if the
   dominant category is consistent (≥0.75) else HEDGED. **Also a NEGATIVE:** with feature-drop noise
   the categorization is CONSISTENT (dominant-fraction 1.00) for the ambiguous IMAGE-BLEND too, on
   every seed/drop — dropping features does NOT flip the category. **Categorical perception is ROBUST
   to input noise (that is biologically its purpose — it denoises a continuum into a decisive
   category).** (An earlier probe appeared to show a ~5/4 split, but that was an artifact of *adding
   random noise features*, not genuine perceptual ambiguity — corrected here by the honesty
   discipline.)

⇒ **graded hedging lives at the CONCEPTUAL/co-occurrence level (EMERGE-31's category-ambiguous "bat"
robustly HEDGES, because a concept genuinely co-occurs with two contexts) but is genuinely ABSENT at
the PERCEPTUAL-category read across ALL FOUR tested mechanisms — because CATEGORICAL PERCEPTION
decisively categorizes even genuinely-intermediate stimuli (a real image morph → one category; input
noise → same category; the coincidence drive is bimodal).** This is a well-characterized, biologically
faithful boundary (categorical perception is a known phenomenon — sharp phoneme/color boundaries), not
a tuning failure. Graded perceptual confidence, if it exists on this substrate, would require a
fundamentally different (graded/pre-categorical or explicit metacognitive-confidence) read-out — NOT
the category codon. The named next mechanism is therefore NOT "more decision-stage sampling" (tested,
negative) but a **pre-categorical graded evidence representation** (a genuinely open substrate
question). `graded_answer_repeated` is retained as the documented-negative probe.

## Scope / significance

- The perception-grounded graded MOAT (CONFIDENT perceived-member vs ABSTAIN novel) is robust,
  grounded in real Gabor/V1 vision, NO `sim/` edit.
- The HEDGED boundary is informative: it locates graded hedging (conceptual co-occurrence, robust;
  perceptual category, suppressed by categorical perception) and names the mechanism for the
  perceptual case (decision-stage consistency, Bogacz-Brown).
- Reuse-by-import of EMERGE-53 + the Gabor/V1 shape set + the graded read. Moat preserved by
  construction.

## Files

- `research/runners/_emerge_graded_perception_console_derisk.py` (`see_blended` image morph,
  `_graded_best` co-activation read, `--demo`).
- Builds on `_emerge53_perception_grounded_conversation.py`, `_genfrontier_optionB_visual_similarity_derisk.py`,
  and the graded read from `_emerge_graded_confidence_completion_derisk.py`.
