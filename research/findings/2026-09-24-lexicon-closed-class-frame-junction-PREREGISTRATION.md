---
type: finding
status: live
date: 2026-09-24
lane: E · Language (learned referent lexicon, ready-to-switch-on)
mechanism: PRE-REGISTRATION of a frame-junction referent lexicon (new flag BRAIN_LEARNED_REFERENT_JUNCTION, default
  OFF). Inside the existing learned referent lexicon, the heard context reaches the noun/non-noun category pools only
  through spiking coincidence units, one per (word before, word after) pair, each firing only when BOTH neighbours are
  heard together (a Mintz 2003 frequent frame, detected by a two-input threshold AND). The direct single-offset
  frame->category edge is absent in this variant. Nothing names the closed class to the circuit.
seeds: [42, 43, 44, 100, 101, 102]
verdict: PRE-REGISTRATION only; no evaluation run has happened. This lane runs the dev check at seed 7 only (not an
  evaluation seed). The six-seed evaluation is left for a later lane.
runner: research/runners/_lexicon_closed_class_parse_diag.py
artifacts:
  - research/findings/raw/_lexicon_closed_class/diag_frame_s7_gt3.json
  - research/findings/raw/_lexicon_closed_class/diag_frame_s7.json
  - research/findings/raw/_lexicon_closed_class/frame_proxy_s7.json
external:
  - "Mintz 2003, Frequent frames as a cue for grammatical categories in child directed speech, Cognition 90:91-117,
    doi:10.1016/s0010-0277(03)00140-9 (PMID 14597271). <!--derived--> A frame is two jointly occurring words with one word
    between them; words sharing a frequent frame fall in one category with high accuracy. This fixes the unit:
    the (-1,+1) pair, detected jointly, not each side added separately."
  - "Chemla, Mintz, Bernal & Christophe 2009, Dev Sci 12:396-406, doi:10.1111/j.1467-7687.2009.00825.x (PMID <!--derived-->
    19371362). The discontinuity of the frame (context on both sides of the target) is what makes it efficient, and
    item-specific context words beat category-level ones. This is why the junction keeps word identity on each side."
  - "Polsky, Mel & Schiller 2004, Computational subunits in thin dendrites of pyramidal cells, Nat Neurosci
    7:621-627, doi:10.1038/nn1253 (PMID 15156147). <!--derived--> Nearby inputs on one thin branch sum sigmoidally (a
    conjunction subunit). This is the biological precedent for a unit that responds to two inputs together and not
    to either alone; the build uses a point-neuron threshold AND as its stand-in (declared residual)."
  - "Hochmann, Endress & Mehler 2010, Word frequency as a cue for identifying function words in infancy, Cognition
    115:444-457, doi:10.1016/j.cognition.2010.03.006 (PMID 20338552). <!--derived--> 17-month-olds map a new object to the
    infrequent noun, not to the frequent determiner. Behavioural target: function words are not taken as labels."
builds_on:
  - research/findings/2026-09-24-language-learned-referent-production-route-GO-6seed.md
  - research/findings/2026-09-24-d6-multiref-wm-learned-referent-env-flag-route-PREREGISTERED.md
  - research/findings/2026-07-03-emerge62-discover-function-words-GO.md
---

# Keeping closed-class words out of the learned referent set: frame-junction lexicon, PRE-REGISTRATION

Branch `research/lexicon-closed-class`, from `origin/main` at `dcc2c9a49`. The measurement instrument
(`research/runners/_lexicon_closed_class_parse_diag.py`) and its adjudication rule were committed first; this file
is committed on its own, before the mechanism is built and before any run it governs.

## The defect, measured (dev seed 7, committed artifacts)

`BRAIN_LEARNED_REFERENT_LEXICON` (default OFF) passed its own 6-seed route test, but with the flag on, the D6 referent
parse changes on many battery probe turns. The B2b review measured 60 of 112 turns with the seed-42 production
lexicon, and saw 'what', 'who', 'before', 'most' and 'today' admitted and 'anne' pushed out of the cap on `tom_fb`.
It wrote no artifact. `research/findings/raw/_lexicon_closed_class/diag_frame_s7_gt3.json` is the same measurement,
taken with the lexicon trained at dev seed 7:

<!--derived-->
- 42 of 112 turns change; 7 do not match under the committed rule (below).
- Non-noun admissions: 'when' (pmem_form, pmem_form2), 'most' (datc_t5, datci_t5), 'wonderful' (emo), 'crazy'
  (datc_t2, datci_t2). 'east' (lbf_cau_teach1, lbf_cau_whatif, "the sun rises in the east") is admitted and
  classed UNKNOWN: listed, not counted.
- 26 nouns the hand table lacks are recovered (owl, apple, marble, basket, room, sky, ...).
- `tom_fb`: 'anne', 'box' and 'where' leave the 5-referent cap, displaced by marble, basket, sally, leaves and room.
  None of those is a non-noun by the ground truth ('leaves' is a verb in this sentence but a noun form by dominant
  POS), so this turn matches under the rule. The cap keeps first mentions, a pre-existing host design choice.
- Peak RSS 394 MB; build + training 52 s on one core pair.

The offending words differ between seed 7 and seed 42; the class does not (closed-class and question words, plus
attributive modifiers).

**Instrument revision, before any mechanism run.** The first committed rule (`ebabbe1a6`, artifact
`diag_frame_s7.json`) was two-valued: a word not tagged NOUN in either spaCy map was a non-noun. That baseline
counted 'east' as a non-noun only because both maps omit it; they cover open-class words only, so absence is not
evidence of being closed-class. The rule is now three-valued. NON means the word is in the NLTK English stopword
list (`research/fixtures/closed_class_inventory_nltk_english.json`, 198 words, copied verbatim) or is tagged
VERB/ADJ. NOUN means it is tagged NOUN. UNKNOWN covers the rest (names, rare words, 'east'); those admissions are
listed, not counted. Re-running seed 7 reproduced the same parse (`parse_sha256` 81737f97... in both files). The
only change is 'east' moving from counted to listed. Nothing about the junction mechanism had been run when this
change was made.

## Why the current lexicon admits them (design evidence, dev seed 7)

The detector (`lexicon_spiking_frame_category`) presents a word through single-offset frame afferents (one per
(offset, context word)) and learns frame->category synapses from a teacher curriculum of 38 nouns and 37 content
non-nouns (verbs, adjectives, adverbs). No closed-class word is in the curriculum.

`research/findings/raw/_lexicon_closed_class/frame_proxy_s7.json` (`_lexicon_closed_class_frame_proxy.py`, host
analysis, design evidence only) splits the learned noun-minus-non-noun drive into the left offsets (-2,-1) and the
right offsets (+1,+2):

<!--derived-->
- held-out fixture nouns: left +85.7, right -3.6 (mean); adjectives: left +20.6, right -4.7; closed-class words:
  left -7.9, right -5.6.

So nearly all of the margin is the LEFT side: the learned weight of "the word before is *the* / *a*". A word that
follows a determiner without being a noun ('the MOST beautiful', 'a WONDERFUL day') gets noun drive from that one
side. A word whose frames the curriculum never covered ('when', 'what', 'who') sits near the decision boundary,
where training-seed noise decides it.

The companion process the single-offset code replaced: an infant's categorization runs on the FRAME, the joint
(before, after) context (Mintz 2003; Chemla et al. 2009). One side added to the other is not the same signal. The
same artifact scores curriculum-only evidence tables (the teacher's own seed words) per joint (-1,+1) frame and per
single offset:

<!--derived-->
| word group (held out of every hand list) | n | positive, joint frame | positive, single offsets |
|---|---|---|---|
| NOUN | 903 | 0.905 | 0.930 |
| ADJ | 297 | 0.071 | 0.643 |
| VERB | 501 | 0.026 | 0.058 |
| closed class (animacy `_STOP` minus D6 `_STOP`) | 82 | 0.122 | 0.220 |

This is a count over corpus statistics, not the spiking circuit; it predicts, it does not test.

## The mechanism (built default-OFF after this commit)

`BRAIN_LEARNED_REFERENT_JUNCTION=1` makes `lexicon_spiking_frame_category.get_lexicon()` build and train a
`FrameJunctionLexicon` instead of the single-offset one. The singleton records its variant; with the flag unset,
`get_lexicon()` is unchanged. The flag has no effect unless `BRAIN_LEARNED_REFERENT_LEXICON=1` routes the lexicon.

- ENVIRONMENT (unchanged): the same heard corpus, the same 400 frame afferents FR (offsets -2,-1,+1,+2 over the 100
  most-heard words), the same K occurrences per presentation, each driving its <=4 afferents for T_ON steps.
- JUNCTION POOL FJ (new): 100 x 100 excitatory neurons, J(a,b) receiving exactly two synapses, from FR(-1,a) and
  FR(+1,b), at a fixed weight W_J at which one afferent alone does not fire J and the two together do (a two-input
  threshold AND). The -2/+2 afferents do not project in this variant.
- LEARNED EDGE: FJ -> CN/CX all-to-all, uniform-with-jitter start, the same synapse-local Oja rule, the same teacher
  curriculum and epochs. There is NO FR -> CN/CX edge: every route from heard context to the category pools passes
  through a junction that needs both neighbours.
- COMPETITION AND READ-OUT: unchanged (reciprocal FSI inhibition; the host reads the winning pool with the existing
  MIN_RATE / DEAD_MARGIN abstain rule).
- LESIONS: `learned_edge` (FJ -> CN/CX restored to the start weights) serves the route runner's R4. New
  `coincidence`: every FR -> FJ weight is set to 2 x W_J, so one afferent delivers what two did and each junction
  fires on either neighbour alone (the AND becomes an OR). The learned FJ -> CN/CX weights are untouched. The
  existing weight-hash check in `decide()` verifies each lesion still holds at measurement.

Constants: W_J, the FJ -> CN/CX start weight and the Oja rate/normalisation are calibrated at dev seed 7 against
two targets fixed here: (i) the AND holds on every junction of a test sample (lone afferent: 0 spikes; pair: >= 1
spike per occurrence window); (ii) the category pools fire above MIN_RATE on the curriculum words during teaching.
They are NOT chosen by the parse-match or route gates below. They are frozen before any evaluation seed runs; a
change after the seed-7 dev check is recorded as an AMENDMENT, committed before the runs it governs.

## Evaluation gates (six seeds; each states the realistic outcome that fails it)

**G1 — the original referent GO still holds.** `_d6_learned_referent_env_flag_derisk` with
`BRAIN_LEARNED_REFERENT_JUNCTION=1` in the process env, seeds 42/43/44/100/101/102, own output directory, scored by
its own unmodified `score()`. PASS iff verdict GO: R1 6/6, R2 6/6, R3 min >= 0.50, R4 max <= 0.20 with every lever
moved, one recorded input. FAILS if requiring both neighbours costs too much noun recall (few noun occurrences have
a most-heard word on both sides), or if the learned-edge lesion leaves recovery above 0.20. As in the original,
these six runs share ONE lexicon trained at seed 42 (the production singleton); G1 replicates the route, not
the detector.

**G2 — the parse matches off except for genuinely new nouns.** `_lexicon_closed_class_parse_diag` with
`BRAIN_LEARNED_REFERENT_JUNCTION=1`, one process per seed, the lexicon TRAINED at each of the six seeds (six
independently trained detectors), all 112 probe turns, arms `--lesions none,coincidence`, adjudicated by
`adjudicate()` and scored by `score()` exactly as committed at `59cc404aa`:
- a turn matches iff no word the flag adds is NON (NLTK English stopword list, or spaCy dominant POS VERB/ADJ in
  `lexicon_referent_pos_gt.json`, then `_corpus_pos_map.json`); an added UNKNOWN word is listed, not counted;
- every word the flag drops was displaced by the full 5-referent cap while no added word was NON;
- shared words keep their order.

Per-seed pass: zero mismatching turns. G2 PASSES iff seed 42 (the seed production trains at) passes AND at least
5 of the 6 seeds pass. FAILS if any closed-class, question, verb or adjective form is admitted on any turn at seed
42 or on two or more seeds, or a flag-off referent is displaced while a non-noun holds a slot.

**G3 — the conjunction owns the fix (lesion).** The same six trained lexicons, parsed again under the
`coincidence` lesion (in the same process, after the intact parse). PASS iff on at least 5 of 6 seeds the lesioned
mismatch count exceeds the intact one (the lever moves; `tools.lab.lever`, required=False, recorded per seed).
FAILS if removing the AND leaves the parse as clean as intact. That would mean the clean parse comes from something
else the variant changed (fewer active inputs, weaker drive, different start weights), not from the conjunction.

**Cannot be passed by abstaining.** A lexicon that refuses every word passes G2 trivially, but then fails G1's R1
('owl' on "the wolf watches the owl") and R3 (held-out noun pairs). G1 and G2 must hold together.

`pooled GO` = G1 AND G2 AND G3. Anything else is NO-GO; a seed without its artifact, or artifacts carrying different
corpus sha256 between G1 and G2, is INCOMPLETE / MIXED-INPUT, never GO.

Report-only (not gated): the `tom_fb` parse and whether 'anne' is kept, per seed; the count of genuine new nouns
recovered on battery turns next to the single-offset lexicon at the same seed; lexicon build+train seconds and
peak RSS.

## Integrity smokes (must hold, not evidence)

- **Default OFF, asserted in data.** With `BRAIN_LEARNED_REFERENT_JUNCTION` unset, the post-change code at seed 7
  reproduces the committed pre-change artifact exactly. It must give `parse_sha256`
  81737f9706d815e56244e7e1886aa622617fe72cc22a5cc780626b67a2bc0d29 and `decisions_sha256`
  94a29a45b55112beab7383cb7ef6b1749593a2e3b2ee602aa5d1d7d89dd1871d, as in `diag_frame_s7_gt3.json`. That file was
  produced with the lexicon code unchanged from the pinned `dcc2c9a49`.
- **The AND holds** on a sample of junctions at the frozen W_J, and the `coincidence` lesion turns it into an OR.

## Dev check in this lane (seed 7 only; carries no evaluation weight)

Seed 7 is not an evaluation seed. This lane runs the integrity smokes, `_lexicon_closed_class_parse_diag` intact and
under the `coincidence` lesion, and the route runner's R1-R4 logic with a seed-7 junction lexicon. The route
runner trains its singleton at seed 42, so the dev harness patches that default to 7 and passes the corpus
explicitly; this is declared and used only in dev. No six-seed run and no default flip happen in this lane.

## Honest residuals

- The junction layer's WIRING is host-designed: one unit per (-1,+1) pair over the 100 most-heard words,
  exhaustive and fixed, not selected by development. A developmental version would grow or prune junctions by
  co-activity.
- The AND is a somatic threshold on a point neuron with two AMPA inputs, a stand-in for a thin-branch subunit
  (Polsky et al. 2004). The engine's dendritic plateau subunit is not used: its ~80 ms plateau outlasts one 10-step
  compressed occurrence and would carry one frame into the next.
- Only the immediate frame (-1,+1); a frame word outside the 100 most-heard words is silent.
- Unchanged from the lexicon: teacher-driven curriculum, host read-out of the winning pool, noun-hood not
  referent-hood, and the pre-existing host `_STOP` / `_PRONOUNS` / `_HOLD_QUERY_WORDS` filters in
  `extract_referents`. This build adds no word list; the closed class is never named to the circuit.
- The ground truth is type-level (dominant POS, a fixed stopword inventory). 'leaves' counts as a noun in "Sally
  leaves the room"; 'east' in "in the east" is UNKNOWN and not counted either way. A closed-class word that is in
  neither the inventory nor a map would also be UNKNOWN: every UNKNOWN admission is listed per seed for review.
- Functional read-outs only.
