---
type: finding
status: boundary
lane: affect-learned-gate-retirement
date: 2026-09-05
mechanism: learned-affect-gate-attempt-2
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_affect_learned_gate_derisk.py
artifacts:
  - research/findings/raw/_affect_learned_gate_derisk.json
builds_on:
  - research/findings/2026-08-12-D1-affect-appraisal-value-learned-DR2-not-hardcoded-lexicon.md
---

# Retiring the affect SALIENCE GATE's fixed `|v-5|>=2` threshold, attempt 2: FOUR genuinely-different learned mechanisms tested, ALL BOUNDARY — best achieves 29.4% recall at zero false positives, and the reason is a REGISTER confound, not a threshold-tuning gap.

**Scaffold-retirement backlog rank 7** (`research/coordination/scaffold_retirement_backlog.md`, "Warriner affect
lexicon+fixed threshold — HIGH · partial: value-half GO, gate-half failed"). The per-word appraisal VALUE is
already retired to the DR-2 learned distributional map (`build_learned_valence_map`, 6-seed GO, held-out
r=+0.811 <!--derived--> quoted from the cited 2026-08-12 D1 finding, not this run's own artifact). The
SALIENCE GATE — which words are allowed to move the mood at all — is still the host's fixed
`abs(warriner_valence - 5.0) >= 2.0` threshold (`_STRONG_MARGIN` in
`research/runners/affect_production_organ.py`). A first attempt to retire the gate too (thresholding the SAME
learned-value magnitude the VALUE half already computes) was abandoned per the 2026-08-12 D1 finding: "a full
drop-in (learned value AND learned gate) would color a plain 'what does the cat eat' positive" because
"distributional valence genuinely bleeds affect onto high-frequency action words (sit/run/jump/play/cat ...) —
UNSEPARABLE from genuine affect by any gain or threshold." This is the retry, using four mechanistically
DIFFERENT candidate signals instead of re-thresholding the same scalar. **Result: still BOUNDARY.** The
diagnosis this time is sharper than "no threshold works" — it is that every candidate reads the SAME
co-occurrence graph the confound lives in, so no statistic derived from it can separate the two classes.

## What was tried, and why each is genuinely different from the refuted lever

Reuses the de-risked DR-2 primitives by import (`_affect_distributional_tag_derisk.build_cooccurrence` ->
`codes_from_cooccurrence` -> `affinity_knn` -> leave-one-out `propagate`) — no reimplementation, no `sim/` edit,
numpy-CPU. On top of the SAME learned co-occurrence graph, four independent discriminating signals:

- **A. Arousal co-gate.** Valence AND learned AROUSAL both elevated (the circumplex model treats valence and
  arousal as orthogonal axes, so an incidentally-warm word need not be independently high-arousal).
- **B. Habituation / frequency exclusion.** Valence elevated AND the word is NOT among the most frequent
  (percentile-ranked within its own corpus build) — a word encountered very often across many contexts is
  neurally habituated, reducing its capacity to carry salience regardless of co-occurrence-derived valence
  (Kandel *Principles of Neural Science* 6e, Ch. 14 — habituation, the simplest form of non-associative
  learning).
- **C. Cross-resample stability.** Bootstrap-resample the training corpus at the 6 canonical seeds (80% of
  stories, without replacement); gate only if the learned valence's magnitude, sign, and low variance are a
  STABLE property of the word across resamples, not an artifact of which stories happened to be sampled.
- **D. Neighbor affect-purity.** Instead of the propagated scalar VALUE, read the COMPOSITION of the word's
  learned-graph neighborhood: what fraction of its propagation-neighbor weight mass is ITSELF a raw-gated
  affect word (a connectivity/concept-cell read, not a diffused magnitude).

Each was calibrated on the full 60,000-story corpus, then VALIDATED — not just re-checked — jointly against 6
independent 80%-bootstrap resamples (the canonical project seeds 42/43/44/100/101/102): a configuration only
counts if it achieves **zero false positives on the full corpus AND all 6 resamples simultaneously**. This
caught a real overfit during development: a purity+magnitude configuration read FP=0 on the full corpus and
5/6 resamples, then FP=1 at seed 42 — exactly the single-corpus-fit failure the joint check exists to catch.

## Result (6-seed, `research/findings/raw/_affect_learned_gate_derisk.json`)

<!--derived-->
The table below rounds this run's own `research/findings/raw/_affect_learned_gate_derisk.json` fields
(`worst_case_recall`, `full_corpus_recall`, per-candidate `params`) to 3 significant figures for display; the
GO-bar paragraph after it restates those same rounded figures in prose. Block-derived to the next heading.

On the common vocabulary present in the full corpus and all 6 resamples (n=164; 102 raw-gated "true affect"
words, 62 raw-excluded "neutral" words — the exact partition the fixed `_STRONG_MARGIN=2.0` threshold makes):

| candidate | joint FP=0 achievable? | worst-case recall (6-seed) | full-corpus recall | calibrated params |
|---|---|---|---|---|
| NAIVE (learned-magnitude only — the refuted 2026-08-12 lever, reproduced as a negative control) | **NO** | n/a | n/a | no `Tv` clears FP=0 jointly across the full corpus + all 6 resamples at all |
| A. arousal co-gate | yes | 0.010 | 0.020 | Tv=3.25, Ta=1.25 |
| B. habituation/frequency exclusion | yes | 0.147 | 0.167 | Tv=2.90, Fpct<=50 |
| C. cross-resample stability | yes | 0.020 | 0.020 | Tv=3.90, Sceil=0.20, Afloor=0.6 |
| D. neighbor affect-purity | yes | 0.147 | 0.167 | Tv=1.50, Pfloor>=0.86 |
| **D+B combined (best found)** | **yes** | **0.294** | 0.324 | Tv=2.70, Pfloor>=0.65, Fpct<=80 |

**GO bar (pre-registered in the runner):** worst-case recall >= 0.5 at joint FP=0. **None of the six
configurations clears it** — the best (D+B combined) recovers under a third of the words the fixed threshold
correctly flags as strongly affective; ~71% of genuinely affect-bearing words would be silently dropped from
the mood signal if this replaced the host gate. The NAIVE negative control is, if anything, a harder failure
than the 2026-08-12 finding characterized: under the joint full-corpus+6-resample validation, **no** magnitude
threshold on the learned value achieves zero false positives at all, even giving up almost all recall (at
single-corpus scale a Tv=4.0 cutoff had squeaked to FP=0 at 1% recall; that does not survive being checked
against all 6 resamples).

## Diagnosis: a REGISTER confound, not a threshold-tuning gap

The four candidates are not interchangeable in principle (orthogonal circumplex axis, non-associative
habituation, statistical robustness, and graph-connectivity composition are four different constructs), but
**empirically they fail for the same reason and on the same words**: `new`, `day`, `old`, `sit`, `look`,
`cat`, `night`, `moon`, `garden`, `wonder` — all raw-excluded (correctly neutral) — score high on learned
valence magnitude, high on neighbor affect-purity (0.30-0.87 in the full-corpus read), high on cross-resample
stability (std 0.10-0.30, 100% sign agreement across all 6 resamples — the OPPOSITE of the "unstable/noisy"
hypothesis candidate C was built to test), and are not reliably lower-frequency than genuine affect words
(several sit in the 40-90th frequency percentile, well inside the same range true affect words occupy). Every
candidate is a **downstream statistic of the same co-occurrence/propagation graph**, and TinyStories' own
narrative register frames ordinary actions and settings ("a new day", "she sat and looked", "the cat", "that
night", "the moon", "the garden") inside emotionally-resolved children's-story scenes about as consistently as
it uses real emotion words. "Co-occurs with warmth" and "is itself an affect word" are not separable from ANY
statistic read off that one graph, because the graph encodes the former by construction and the latter is
being asked of it as a proxy question.

This matches a documented limitation of the technique family this project's DR-2 mechanism belongs to:
label-propagation sentiment/valence lexicon induction (SentProp — Hamilton, W. L., Clark, K., Leskovec, J., &
Jurafsky, D., "Inducing Domain-Specific Sentiment Lexicons from Unlabeled Corpora," EMNLP 2016,
https://nlp.stanford.edu/pubs/hamilton2016inducing.pdf) evaluates positive/negative propagation separately
from a neutral class precisely because the propagated score alone does not cleanly carry a neutral/non-neutral
decision — ternary handling needs machinery beyond the propagated scalar itself. That is exactly what this
probe confirms empirically for THIS corpus and lexicon: no derived read of the propagated structure (magnitude,
arousal, frequency, stability, or neighborhood composition) supplies the missing neutral-class machinery.

## What this means for the backlog item

The gate-half of rank 7 remains **failed, now on a second, mechanistically-independent attempt**, with a
sharper diagnosis than "unseparable by any gain or threshold" (the 2026-08-12 framing): it is specifically that
**every statistic derivable from text co-occurrence inherits the same register confound**, so the surpass is
not "find a better statistic of this graph" but a different information channel. This reinforces, rather than
merely repeats, the D1 finding's own named next rung: a fully-spiking on-bridge opponent V+/V- appraisal
population whose valence is bound to the SIMULATION'S OWN experienced affective response during a pairing
(the amygdala/BLA route — Namburi, P. & Tye, K.M. et al., *Nature* 2015, opposing valence-coding populations)
rather than to which words a text corpus happens to place nearby. That channel is not confounded by narrative
register because it does not read affect off lexical company at all.

## Honest residuals

1. **Not exhaustive.** Four candidate families (plus one 2-way combination) were tested; a fifth mechanism
   might yet separate the classes. But all four fail via the SAME diagnosed mechanism (shared dependence on
   the co-occurrence graph), which is evidence against "a fifth statistic of the same graph" being the fix,
   not merely an absence of a positive result.
2. **The vocabulary stays closed.** This probe only re-scores words already in the ~180-word hand-curated
   Warriner-approximate seed lexicon (`WARRINER` in `_affect_distributional_tag_derisk.py`); it does not test
   whether the mechanism generalizes gate-worthiness to words OUTSIDE that set (a further, separate claim the
   D1 finding's own "does not retire the lexicon" caveat already flags for the VALUE half).
3. **Recall was optimized jointly for zero false positives; a looser bar was not separately explored.** A
   product owner willing to tolerate a SMALL, bounded false-positive rate (not zero) might find one of these
   candidates crosses 0.5 recall — not measured here because the task's bar was preserving the neutral default
   exactly, matching the named prior failure mode.
4. **Warriner arousal is the KNOWN weaker channel** (DR-2's own 6-seed report, quoted not re-measured here
   <!--derived-->: held-out arousal r=+0.694 vs valence r=+0.811) — candidate A's poor showing (1%
   worst-case recall) is plausibly floored partly by arousal-channel noise on top of the register confound;
   this is not separated out here.

## Production wiring: NONE

Per this being a genuine negative across all four candidates, **nothing was wired into
`research/runners/affect_production_organ.py` or `webapp/wkv_mouth_generator.py`.** Both are byte-unchanged —
`git diff --stat` against this commit's parent shows zero lines changed in either file. The fixed
`_STRONG_MARGIN = 2.0` Warriner-norm threshold remains the production salience gate. This file
(`_affect_learned_gate_derisk.py`) is additive-only: a standalone research probe that imports from, and does
not modify, the production organ.

## Anti-cheats

- **The refuted mechanism is reproduced as an explicit negative control** (candidate NAIVE), not just cited —
  this run's own artifact shows it failing even harder under joint validation than the single-corpus framing
  the 2026-08-12 finding used.
- **Joint full-corpus + 6-resample validation, not per-seed cherry-picking.** A configuration must clear FP=0
  on every one of 7 independent builds to count; the seed-42 counter-example found during development (5/6
  resamples + full corpus at FP=0, then FP=1 at seed 42 for purity+magnitude alone) is reported in the runner
  docstring as the concrete case this discipline caught.
- **`tools.lab.void_if`/`undefined_if_empty`** guard the vocabulary-intersection and raw-gate-split sizes so a
  degenerate build (e.g. an empty resample vocabulary) would abort loudly rather than silently reporting a
  hollow 0/0 recall.
- **Percentile-ranked frequency** (not an absolute count) so candidate B's threshold is computed independently,
  and comparably, within each corpus build rather than depending on absolute corpus size.
- **Byte-diff confirmed** on both production call sites (`affect_production_organ.py`,
  `webapp/wkv_mouth_generator.py`) showing zero changes.

## Sources

- Hamilton, W.L., Clark, K., Leskovec, J., & Jurafsky, D. (EMNLP 2016), "Inducing Domain-Specific Sentiment
  Lexicons from Unlabeled Corpora" (SentProp) — https://nlp.stanford.edu/pubs/hamilton2016inducing.pdf. Label
  propagation over a corpus-derived graph evaluates a neutral class separately from the propagated pos/neg
  score precisely because the score alone does not cleanly carry that decision.
- Kandel, *Principles of Neural Science* 6e, Ch. 14 — habituation as the simplest form of non-associative
  learning (the biological grounding for candidate B's frequency/habituation exclusion).
- Russell, J.A. (1980), *J. Pers. Soc. Psychol.* 39(6):1161 — the circumplex model (valence/arousal
  orthogonality grounding candidate A).
- Namburi, P., Tye, K.M. et al. (2015, *Nature*) — opposing BLA valence-coding populations; the mechanism class
  (experience-bound, not co-occurrence-bound) named as the surpass.
- `research/findings/2026-08-12-D1-affect-appraisal-value-learned-DR2-not-hardcoded-lexicon.md` — the first
  gate-retirement attempt and its "unseparable by any gain or threshold" finding, reproduced here as the NAIVE
  negative control.

## Reproduce

```
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._affect_learned_gate_derisk --smoke   # ~3s sanity
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._affect_learned_gate_derisk            # 6-seed, ~12s
```
