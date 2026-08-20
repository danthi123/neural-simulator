---
type: finding
status: live
date: 2026-08-19
mechanism: affect-appraisal-lexicon-coverage
lane: D-affect
integration_faculty: affect-coloring + affect-drives-response
seeds: [42]
instrument: internal-signal map (appraised valence -> neural differential -> tone_level -> content_plan) across an affect-strength ladder, + an end-to-end lesion A/B through the real /api/brain-chat handler
artifacts:
  - research/findings/raw/_affect_lexicon_fix/evidence.json
  - research/findings/raw/_affect_lexicon_fix/sensitivity_before.txt
  - research/findings/raw/_affect_lexicon_fix/sensitivity_after.txt
  - research/findings/raw/_affect_lexicon_fix/e2e_lesion.txt
---
# The affect appraisal missed strong-emotion words, so BOTH affect faculties went dormant exactly when emotion was strongest

**One line.** The affect faculties (#13 coloring + #84 tone lead, which read the SAME mood) fired on *moderate*
everyday words ("sad", "happy") but read **valence 0.0 for the most strongly-worded emotion** ("I am furious,
devastated, heartbroken") — because the appraisal salience lexicon was a 140-word set curated for the TinyStories
CHILD corpus and missed the common adult emotion words. So the brain's mood stayed neutral, and its emotional
coloring went **dormant precisely when a person is most emotional.** Fixed by expanding the declared seed lexicon
with ~40 calibrated adult strong-emotion words; strong emotion now moves the mood while neutral queries stay neutral.

## How it was found (the instrument is part of the emulation)
Owner steered the next arc to "make faculties drive (depth)". Re-checking the #13 affect-coloring faculty, an
earlier audit probe had read `intact == lesion` and I had flagged #13 as "not shown to drive." Rather than trust that
output, I **instrumented the internal signal** — mapped appraised valence → neural differential → `tone_level` →
`content_plan` across an affect-strength ladder — and found the opposite of a dead coupling: the coupling is fine,
but its INPUT was being zeroed. Moderate words ("really wonderful, I'm happy") crossed threshold (valence +0.81,
level 3), while *stronger* messages ("thrilled and delighted", "furious, devastated") read **valence +0.00** — the
words simply were not in the appraisal lexicon. The prior audit's "strong negative" probe used exactly those
missing words, which is why it read no change. The faculty was never hollow; the instrument (the lexicon) was
under-covered.

## Root cause (`research/runners/_affect_distributional_tag_derisk.py::WARRINER`)
`appraise_text` gates words through `WARRINER`, a 140-entry hand-curated set. It is TinyStories vocabulary (puppy,
dragon, witch, monster) with only a handful of emotion words (happy/sad/angry) — and it MISSED thrilled, delighted,
excited, ecstatic, overjoyed, elated, wonderful, joyful, devastated, furious, miserable, heartbroken, terrified,
anxious, depressed, frustrated, disappointed, upset, and more. A word not in `WARRINER` is skipped, and
`valence = mean(gated words)`, so a message of all-missing emotion words yields 0.0. The DR-2 learned map is only
2 words, so the seed norm IS the value for ~138/140 — the lexicon gap is the whole gap.

## The fix
Added ~40 common adult strong-emotion words to the seed lexicon with Warriner-approximate `(v9, a9)` norms,
sign-correct and `|v-5| >= _STRONG_MARGIN (2.0)`, calibrated to the existing entries (e.g. happy (8.5,6.1),
sad (2.1,4.6)). All added words are unambiguously affective, so they cannot color a neutral factual query. This is
a declared-seed-scaffold completeness fix, not a new mechanism — the appraisal gate is already documented as a
Warriner-seed host scaffold (the learned-gate drop-in was measured to break neutral-default, so the seed stays).

## Verification
Artifacts: research/findings/raw/_affect_lexicon_fix/evidence.json (+ sensitivity_before.txt, sensitivity_after.txt, e2e_lesion.txt in the same dir)

- **Internal-signal map (before → after):** strong/extreme positive 0.00 → +0.79 (level 3, 3 warm elaborations);
  strong/extreme negative 0.00 → −0.79 (level −2, 1 terse sentence). Artifacts
  [`sensitivity_before.txt`](research/findings/raw/_affect_lexicon_fix/sensitivity_before.txt) /
  [`sensitivity_after.txt`](research/findings/raw/_affect_lexicon_fix/sensitivity_after.txt).
- **No neutral regression:** 5 neutral factual/greeting queries (incl. "what is the content of the book?") stay at
  valence 0.0, level 0, default forthcomingness.
- **End-to-end lesion A/B through the real handler** (numpy, forthcomingness surface):
  [`e2e_lesion.txt`](research/findings/raw/_affect_lexicon_fix/e2e_lesion.txt) — INTACT, a strong-POS induction yields *'Gladly! The dog chases cat. The cat eats fish.'* (2 sentences, warm) while a strong-NEG induction yields *'Honestly — The dog chases cat.'* (1 sentence, curt) — mood now changes both forthcomingness (#13) and the tone lead (#84); before the fix both read valence 0 and were identical. Under `BRAIN_AFFECT_LESION` the NEG forthcomingness collapses (1 -> 2 sentences), proving #13 is load-bearing.
- **No substrate regression:** `tests/test_determinism.py` 9/9; the lexicon change is additive and byte-identical
  when the added words are absent (BRAIN_AFFECT=0 remains the oracle).

## Scope + residual
This makes the appraisal (and therefore both affect faculties) fire on strongly-worded emotion. Honest residuals
(unchanged by this fix, already declared scaffolds): the salience gate is still a curated seed lexicon (not a
learned/spiking appraisal population — the next rung is a fully-spiking opponent V+/V− appraisal); the appraisal
injection is host; the affect organ is a co-resident bridge not yet merged with the recall composer. The lexicon is
now broad enough for common conversational emotion, not exhaustive.
