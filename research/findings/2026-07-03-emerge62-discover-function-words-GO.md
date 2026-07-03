# EMERGE-62 — DISCOVER the closed-class FUNCTION-WORD set from distributional statistics — GO (controlled stream), with an honest real-corpus precision boundary named

**Date:** 2026-07-03
**Verdict:** **GO** (6-seed, controlled EMERGE-domain stream) — the closed-class function-word inventory + the open/closed distinction SELF-ORGANIZE from distributional frequency + context-coverage; the discovered set feeds the EMERGE-59 spiking-Broca frames and renders correctly, moat intact. Honest nuance: on a NOISY real corpus, recall stays 1.00 but precision drops (the 3rd Yang-Getz cue is the named next signal). This is the RANK-1 cheap-first de-risk the research gate prescribed (`2026-07-03-self-organizing-grammatical-structure-research-gate.md`).
**Runner:** `research/runners/_emerge62_discover_function_words_derisk.py` (`--demo` / `--derisk`)
**CI:** `tests/test_emerge62_discover_function_words.py` (7 tests, CPU/numpy, offline) — all pass.
**Raw:** `research/findings/raw/_emerge62_discover_function_words.json`
**Reuse-by-import; NO `sim/` edit; the gate-first moat untouched (0 productions on abstains).**

---

## What this closes (the residual S2)

The just-built spiking-Broca producer (EMERGE-59/60/61, all GO) rendered EMERGE's reply frames on spikes but had TWO
host-designed residuals. **S2** = the hand-written function-word SET + the open/closed LABEL:
- `research/runners/argstructure_composer.py:99` — `FUNCTION_WORDS = {"the","a","an","to","on","in","of","with",...}`
- `_emerge59_spiking_broca_frame_slots_derisk.py:98-105` — the `FRAMES` FUNC/DET payloads `{the, can, does, not}`.

EMERGE-62 makes that inventory EMERGE from the language stream. The hand list becomes the **validation ground-truth**,
not the input. (S1 — the per-frame slot ORDER + slot INVENTORY — are the ranked follow-ons EMERGE-63/64.)

## The mechanism (self-organized, cheapest-first)

Over a language stream (a controlled SVO + function-word stream in the EMERGE frame domain — content words AND
function words; the same "generate a controlled stream" pattern EMERGE-30/33 use), compute per-word, **reusing the
project's statistics** (`corpus_stream.load_token_stream` for the stream; the running-frequency + windowed
co-occurrence the stream cortex computes; PPMI content read via the `learned_graded_cortex_fair_test.ppmi_matrix`
log-marginal-ratio algebra):
- **running FREQUENCY** — the Shi/Gervain "frequency-first" arm (infants segregate the closed class by frequency
  *before* meaning).
- **contextual COVERAGE** — the fraction of the vocabulary a word neighbours in a ±2 window (the Yang-Getz Goldilocks
  "diverse enough" arm: a function word co-occurs with MANY different content words; a content word — even a frequent
  one — occurs in FEW contexts, sharpened by selectional restriction). The PPMI mean is reported alongside as the
  content read.

The closed class EMERGES as the words HIGH on BOTH (`freq-pct >= 0.90 AND cover-pct >= 0.60`, **FIXED / pre-registered**,
chosen once on seed 42 then frozen). No hand-list as input. The DISCOVERED set then populates the EMERGE-59 frames'
FUNC/DET slots; the spiking-Broca producer renders "the owl can fly" using the SELF-DISCOVERED function words.

**Field grounding:** Yang & Getz (2026, arXiv 2601.21191, 186-language universality of the frequency + diversity +
phrase-boundary cues); Shi/Gervain (frequency-first segregation); Redington/Cartwright-Brent (distributional POS
induction by context-vector clustering); Dominey & Hinaut (open/closed split, thematic roles read from closed-class
position, self-organized grammar in spiking nets); catalog **G.12** (Kandel 6e Ch 55 pp 1382-1384: Broca agrammatism =
retained noun selection, LOST function-word use — the neurolinguistic double-dissociation making the closed class a
separable statistical population; the EMERGE-59 `b3` function-word-ablation control reproduces it behaviorally).

## Results — 6-seed (42/43/44/100/101/102), CPU/numpy

| metric | value | notes |
|---|---|---|
| discovery **F1** | **0.863** (0.846–0.880) | mean over 6 seeds |
| precision | 0.760 | FPs are the borderline adjectives (see below) |
| **recall** | **1.000** | every ground-truth closed-class word recovered, every seed |
| **frame-recall** | **1.00** | ALL frame function words `{the, can, does, not}` recovered, every seed → the frames CAN be fed |
| **render-ok on discovered set** | **1.00** | held-out facts render correctly ("the owl can fly", …) using the discovered function words |
| **moat calls on abstain** | **0** | the gate-first no-confab moat holds by construction |

**Anti-cheats — all COLLAPSE (project control-validity methodology: input-destruction + hold-out, not a fixed-random control):**
- **FREQUENCY-SHUFFLE** (permute the statistic↔word mapping → destroy the signal): F1 **0.079** (0.000–0.240) vs main
  0.863 — an **~11× margin**, well past the pre-registered ≥ 0.30.
- **NO-STREAM** (empty stream → no statistics): discovered set is **empty**, every seed.
- **HELD-OUT word** (withhold `does` [fw] and `trout` [cw] from the threshold-fitting slice; classify by their OWN
  stats vs frozen thresholds): `does` → CLOSED, `trout` → OPEN, **every seed** — generalization, not memorization.

**The false positives are the linguistically-borderline ADJECTIVES** (`big`, `fast`, `tall`, `grey`, `red`, `cold`,
`slow`, `wet`) plus, at seed 42 only, the single highest-frequency noun `cat`. Adjectives are a genuine open/closed
gray zone (a small, moderately-frequent set) — an honest, expected precision cost, not a mechanism failure.

**Sample (self-discovered function words drive the render):**
```
you> can an owl fly?          broca> the owl can fly            [INHERIT; producer INVOKED]
you> can a penguin fly?       broca> the penguin walks         [CANCEL;  producer INVOKED]
you> can a penguin fly? [deny] broca> the penguin does not fly [DENY;    producer INVOKED]
you> can a zzz fly?           broca> I don't know.             [MOAT;    producer NOT invoked]
```

## Honest real-corpus boundary (reported, not a GO gate) — names the exact next signal

The SAME rule was run on the project's REAL fact/QA corpus (`data/corpus/ra_finetune_corpus.txt`, 645K tokens, via
`corpus_stream.load_token_stream`). Result: **recall 1.00 + frame-recall 1.00** (ALL ground-truth closed-class words
AND all frame function words `{the, can, does, not}` are still discovered — `does`/`not`/`can` recovered cleanly) —
but **precision 0.078 / F1 0.145**: the mixed real corpus (fact/QA frames INTERLEAVED with TinyStories for
anti-forgetting) has hundreds of high-frequency, high-coverage narrative content words (character names, common verbs)
that also pass the frequency+coverage threshold.

This is exactly the residual the research gate anticipated: **frequency + coverage RELIABLY FIND the closed class
(perfect recall) but OVER-INCLUDE on noisy real text (low precision).** The precise next single-variable signal is
**Yang-Getz's 3rd cue — PHRASE-BOUNDARY / syntactic-POSITION ALIGNMENT** (a function word aligns to phrase edges; a
content word does not) — added as a third distributional statistic. Still not a wall; the mechanism (frequency +
diversity) is validated, the residual is precisely named.

## Scope + why this is BRAIN-BASED-ONLY compliant

- Discovers the closed-class INVENTORY (S2 — the function-word set + the open/closed distinction) from distributional
  experience for the BOUNDED EMERGE frame domain. It does NOT make the domain open-ended (open arbitrary generation,
  R4, is the separate deferred wall: the from-scratch spiking LM is ~4 orders too small, `2026-05-07-Phase-2.3a-NEGATIVE`).
- The discovery is offline lexicon/syllabus prep — like rendering a retinal image the neural retina then reads
  (`feedback_brain_based_only_standard`: host is legitimate for the environment/syllabus; the brain renders through
  spikes). The order production is on real spikes (EMERGE-59); the discovered set FEEDS those spiking frames.
- The gate-first no-confab moat is untouched (0 producer invocations on abstains, by construction).

## Method note (control validity)

`render_on_discovered` scores THIS de-risk's contribution — that the DISCOVERED function words correctly fill the
frame's closed-class slots (multiset match + every required function word present) — DECOUPLED from EMERGE-59's own,
already-GO (6-seed) slot-ORDER production. The rate-ranking read-out has a known validated tie-break at the 5-slot
negated-modal frame (a single equidistant-neighbour swap; EMERGE-59 `exact` = 1.0, `order` can dip ~0.95 at one seed).
That is EMERGE-59's concern (S1b / EMERGE-63), not the function-word discovery this de-risk gates; a MISSED function
word (the discovery's job) DOES fail render (verified — `test_missing_function_word_breaks_render`).

## Next (ranked, per the research gate)

- **EMERGE-63** (S1b): swap `FrameSlotCQ._teach_order`'s host-template order-teacher for a corpus n-gram / positional
  order-teacher (FrameCQ already produces frame-conditioned order on spikes, 6/6 GO — only the teacher SOURCE changes).
- **EMERGE-64** (S1a): extend `_bucketB_corpus_mined_frames` frame-mining to the FUNC-slot inventory.
- **EMERGE-65**: compose 62 + 63 + 64 into the fully-self-organized spiking-Broca producer (FRAMES built from
  statistics end-to-end).
- **The precision residual** (real-corpus over-inclusion): add the phrase-boundary-alignment cue as a 3rd statistic.

## Sources

Yang & Getz (2026) arXiv 2601.21191; Shi/Gervain "Word frequency as a cue to identify function words in infancy";
Redington, Chater & Finch (1998); Cartwright & Brent (1997); Dominey & Hinaut (PLoS ONE 2013; "Self-Organized
Artificial Grammar Learning in Spiking Neural Networks"); Kandel 6e Ch 55 pp 1382-1384 (feature-catalog G.12).
Project precedents: `_emerge59_spiking_broca_frame_slots_derisk.py`, `argstructure_composer.py:99`, `corpus_stream.py`,
`learned_graded_cortex_fair_test.ppmi_matrix`, `_emerge19_real_ppmi_generalization_derisk.py`,
`_bucketB_corpus_mined_frames_derisk.py`, `song_g1_core.py`. Research gate:
`research/findings/2026-07-03-self-organizing-grammatical-structure-research-gate.md`.
