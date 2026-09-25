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
verdict: PRE-REGISTRATION only; no evaluation run has happened. AMENDMENT 2 (2026-09-24) fixes 7 independent-review
  issues and adds 3 mechanism changes (short-term depression on FR->FJ, a sentence-boundary pause token, and a
  runner-side homeostatic settle for the learned_edge lesion), and runs the dev check at seed 7 AND seed 42 (neither
  an evaluation seed). The six-seed evaluation is left for a later lane.
runner: research/runners/_lexicon_closed_class_parse_diag.py
artifacts:
  - research/findings/raw/_lexicon_closed_class/diag_frame_s7_gt3.json
  - research/findings/raw/_lexicon_closed_class/diag_frame_s7.json
  - research/findings/raw/_lexicon_closed_class/frame_proxy_s7.json
  - research/findings/raw/_lexicon_closed_class/and_calibration_s7.json
  - research/findings/raw/_lexicon_closed_class/drive_ratio_s7.json
  - research/fixtures/lexicon_referent_pos_gt_tokenlevel.json
  - research/findings/raw/_lexicon_closed_class/and_population_stp_grid_s7.json
  - research/findings/raw/_lexicon_closed_class/and_population_stp_grid_s7_fine.json
  - research/findings/raw/_lexicon_closed_class/and_population_stp_grid_s7_finer.json
  - research/findings/raw/_lexicon_closed_class/and_population_stp_grid_s7_width.json
  - research/findings/raw/_lexicon_closed_class/drive_ratio_s7_amendment2.json
  - research/findings/raw/_lexicon_closed_class/or_match_factor_s7.json
  - research/findings/raw/_lexicon_closed_class/diag_frame_s42_v2_amendment2.json
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
  - "Abbott, Varela, Sen & Nelson 1997, Synaptic depression and cortical gain control, Science 275:220-224,
    doi:10.1126/science.275.5297.220 (PMID 8985017). <!--derived--> Short-term synaptic depression renders a
    postsynaptic neuron's steady-state response nearly independent of presynaptic firing rate. AMENDMENT 2's basis
    for using Tsodyks-Markram depression on FR->FJ synapses to stop a single fast-firing afferent from firing a
    junction alone."
  - "Turrigiano & Nelson 2004, Homeostatic plasticity in the developing nervous system, Nat Rev Neurosci 5:97-107,
    doi:10.1038/nrn1327 (PMID 14735113). <!--derived--> Neurons scale their synaptic weights multiplicatively to
    maintain a target firing rate after a perturbation. AMENDMENT 2's basis for the R4 learned_edge-lesion settle."
builds_on:
  - research/findings/2026-09-24-language-learned-referent-production-route-GO-6seed.md
  - research/findings/2026-09-24-d6-multiref-wm-learned-referent-env-flag-route-PREREGISTERED.md
  - research/findings/2026-07-03-emerge62-discover-function-words-GO.md
  - research/findings/2026-07-03-emerge62b-position-cue-GO.md
  - research/findings/2026-09-24-lexicon-closed-class-frame-junction-dev-s7-not-ready.md
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

## AMENDMENT 1 (2026-09-24, before any trained junction lexicon was scored; gates G1-G3 unchanged)

Calibration at dev seed 7 showed that the mechanism as written cannot run. Three changes follow. Each is fixed
by a rule stated here, and none was chosen by a gate.

**What had been seen when this was written.** The AND calibration grid (`and_calibration_s7.json`), the drive
ratio (`drive_ratio_s7.json`), and two 2-epoch training probes at seed 7 (scratch, not committed). The first
probe, at v2's weight scale, left every curriculum word abstaining. The second, drive-matched, decided 73 of 75
curriculum words correctly and printed the decisions for owl, apple, when, most, wonderful, crazy, what, who,
before and today. No constant below was chosen from those decisions.

1. **Occurrence window T_ON_J = 50 steps for the junction variant** (v2 keeps 10). An unbiased sweep (weights 2 to
   128, not committed) never fired a junction within 10 steps of onset, even to the pair. In the committed grid the
   slowest sampled junction's first spike to a pair came 19 to 43 steps after onset. <!--derived--> The frame
   afferents first fire about 4 steps after onset, and the RS unit integrates. So no AND can complete inside a
   10-step occurrence. 50 ms per heard occurrence is still about 5 times shorter than a spoken word. Presentations
   get 5 times longer, which is accepted (speed is secondary).
2. **Junction threshold: a constant hyperpolarizing current I_TONIC_J on every junction, plus W_J.** It stands in for
   tonic inhibition; without it, a lone afferent held for a few occurrences fires its junctions at any weight
   that lets a pair fire in time. Target (i) is restated so it covers sustained input. On every sampled junction
   (64, seed 7), a lone left or right afferent held for 150 steps gives 0 spikes, and the pair fires within one
   T_ON_J window. Selection rule: take the weight whose feasible bias range is widest (ties go to the smaller
   weight) and the middle of that range. Result: W_J = 300, I_TONIC_J = -650 pA (feasible -600..-700 at W_J 300;
   -800..-900 at 400; none at 500).
3. **Drive matching of the learned edge.** Junctions fire about 27.5 times less often than v2's frame afferents
   (`drive_ratio_s7.json`: 0.0191 vs 0.526 spikes per step, curriculum, untrained). <!--derived--> At v2's scale the pools stay
   silent after training. With S = 27.53, the start weight is S x W_INIT, the Oja rate S^2 x ETA and the
   normalisation OJA_BETA / S^2. This rescaling maps the junction Oja update onto v2's at matched aggregate drive
   (dW_J = S x dW). S is a measured ratio, not a tuned value.

Target (ii) as first written ("the pools fire above MIN_RATE during teaching") cannot fail, because the teacher
current drives them. It is dropped as a calibration target. Curriculum training accuracy is reported in the dev
check, not used to set anything.

The integrity smoke's AND check uses the restated target (i) on 256 sampled junctions at the frozen constants.
The `coincidence` lesion is unchanged (every FR -> FJ weight x 2, so a lone afferent delivers what the pair did).
Evaluation cost rises: one junction lexicon build + train takes about 15 minutes on numpy per seed.

## AMENDMENT 2 (2026-09-24, independent review of the seed-7 dev check; 7 issues + 3 mechanism changes; committed
before any run it governs)

An independent review of the round-1 dev-seed-7 check
(research/findings/2026-09-24-lexicon-closed-class-frame-junction-dev-s7-not-ready.md) found the round SOUND but
raised 7 instrument/gate issues and named 3 mechanism next-steps. Each is fixed here, before the dev check is
re-run at seed 7 AND seed 42 (review issue 5: the round-1 check never exercised the PRODUCTION seed).

### Instrument / gate fixes (review issues 1-2, 3-4, 6-7)

**1. G3 per-word margins + a drive-matched OR control (HONEST NEGATIVE: the two properties do not coexist)**
(`research/runners/_lexicon_closed_class_parse_diag.py::_parse_arm`, `lexicon_frame_junction.py::measure_or_match_factor`).
The fixed `OR_LESION_FACTOR=2x` "coincidence" lesion raises BOTH "a lone afferent now fires the junction" AND the
total population drive together, so a G3 mismatch-count increase under it does not separate "the conjunction
mattered" from "there is simply more drive now" -- round 1's own numbers show why: the intact-to-lesion change was
2->6 mismatching turns, and inspecting the new admission's MARGIN shows 'who' cleared DEAD_MARGIN by 0.003 against
the 0.002 bar, a single borderline word, not a clean separation. `_parse_arm` now reports, for every ADMITTED word
on every turn, its CN-CX rate margin and a `near_boundary` flag (margin < 1.5x DEAD_MARGIN). A new lesion kind
`"coincidence_matched"` (`OR_MATCH_FACTOR`) was meant to run ALONGSIDE `"coincidence"` as a drive-matched control.
`measure_or_match_factor` (dev seed 7, `or_match_factor_s7.json`) found it CANNOT be built as a single FR->FJ weight
scale: mean FJ population rate over real curriculum presentations is FLAT (0.0053-0.0065 spikes/step) for factor
1.02-1.1, then rises steeply -- 0.015 at 1.2, 0.256 at 1.4, >1.4 at 1.6+ -- crossing the intact arm's own rate
(0.0053) only at 1.02/1.05 (tied, gap 0.000167), a factor that sits WELL INSIDE the zero-violation `and_population`
feasible band (W_J 2950-3000 at this bias) and so barely perturbs the AND at all. "Genuinely OR-like" and
"drive-matched to intact" are not jointly reachable this way: by the time the factor breaks the AND, drive is
already many times the intact rate. `OR_MATCH_FACTOR = 1.02` (ties -> the smaller value). **Read
`coincidence_matched`'s result as a check that a small, AND-preserving weight change does not spuriously move the
parse -- NOT as a drive-matched OR** (the reviewer's ask, honestly not deliverable with this parameterization). G3's
own verdict is unchanged (`coincidence` moving the mismatch count); this control is report-only, alongside it.

**2. No-pass-by-abstaining (silent-NON gate)**
(`_parse_arm`'s `silent_non_words`/`silent_non_fraction`). Round 1 found 33 of 55 heard NON-ground-truth words at
seed 7 leaving BOTH category pools silent (rate < `L.MIN_RATE` in both CN and CX -- a failure to decide, not a
margin abstain) against v2's 5; this was never gated. `_parse_arm` now reports, per arm, every heard NON word whose
decision is silent-in-both-pools, and the fraction of heard NON words this covers. **G4 (new):** at each checked
seed, the junction's silent-NON fraction must not exceed 0.30 (a stated bar: roughly midway between v2's typical
single-digit-percent reading and round 1's ~60%, chosen because a functioning circuit should show SOME graded
signal on most closed-class words even where the net margin abstains -- not derived from these seeds' own data).
v2's own silent-NON fraction at the SAME seed is always printed alongside (comparison, per the review, is same-seed
only, never cross-seed or an absolute floor pulled from a different run).

**3. G2 ground truth: type-level POS tags miscredit / undercount whole word classes**
(`_lexicon_closed_class_token_pos_fixture.py`, new; `_lexicon_closed_class_parse_diag.py::load_token_gt`,
`make_turn_gt_class`). Confirmed exactly as the review states: `lexicon_referent_pos_gt.json` tags 'today' NOUN
(its dominant reading; the fixture is type-level, not sentence-scoped) and `_corpus_pos_map.json` covers only
NOUN/VERB/ADJ, so 'most' (in-map, ADJ -- already correctly NON before this fix), and every modal ('might'),
wh-word ('who'/'what'), indefinite pronoun ('something'/'everyone'/'anybody') and adverb
('never'/'ever'/'honestly'/'absolutely') NOT in the 198-word NLTK stopword list falls to UNKNOWN (listed, not
counted) -- exactly the class an admission mistake would hide in. `leaves` is credited NOUN even as the verb in
"Sally leaves the room" (tom_fb). FIX: a new fixture tags each of the 112 battery turns' OWN words IN THEIR OWN
SENTENCE (nltk `pos_tag`, averaged-perceptron, Penn Treebank tagset; built once, committed, no runtime nltk
dependency -- the same build-time-only convention `closed_class_inventory_nltk_english.json` already uses).
Verified on the actual battery: 'leaves' -> VBZ (NON) in tom_fb; 'who'/'what' -> WP, 'before' -> IN, 'when' -> WRB,
'might' -> MD, 'never'/'ever'/'honestly'/'absolutely' -> RB, 'most' -> RBS, all -> NON directly, no longer via two
list lookups landing on UNKNOWN. HONEST RESIDUALS, both verified on the battery, both kept as declared limitations
rather than hand-patched: (a) the Penn Treebank tagset has no indefinite-pronoun tag, so its OWN guidelines --
reproduced by nltk's tagger -- tag 'something'/'everyone'/'anybody' NN; a small explicit override
(`INDEFINITE_PRONOUNS`, 12 words) reclassifies these to NON in BOTH the type-level and token-level classifiers,
the same instrument-word-list status the NLTK stopword inventory already has; (b) the tagger is itself imperfect
-- 'east' tags RB (adverb) in "the sun rises in the east", a noun use it gets wrong -- a second, independently
imperfect instrument, not a superseding one. Per-turn adjudication now uses the token-level tag when available,
falling back to the type-level map only for a word the fixture does not cover.

**4. G2 is not parse parity (stated explicitly)**
On `tom_fb`, both v2 and the junction lexicon drop 'anne' and 'box' (the false-belief location) to the same
replacement set, and G2 scores this MATCH under rule (b) (a drop is explained iff the cap is full and no admitted
word is NON) -- correctly, by that rule's own definition, but that rule adjudicates "no closed-class word took a
referent slot", not "the parse is the same as any other lexicon's". **Stated explicitly: G2 is NOT a parse-parity
gate** and never has been; it is a closed-class-admission gate. The flip bar for the eventual six-seed evaluation
adds a new, separate **G5 (battery no-regression, evaluation-stage only, NOT run here):** the combined
`onebrain_regression_battery` behavioural outputs (not just D6's referent list) with the flag ON must show no NEW
regression against the flag-OFF production battery beyond what G1-G4 already accept. G5 is defined here so the
flip decision has it pre-registered; it needs the full battery + six seeds and is explicitly NOT run in this lane.

**5. The dev check must include seed 42 (the production seed), not only seed 7**
`_lexicon_closed_class_junction_dev.py` now takes `--seed` (default 7) and is run at BOTH 7 and 42 in this lane
(seed 42's D0/D4 have no pre-change hash pin -- see D0's `note` field -- and D4 there rebuilds a SECOND fresh
junction lexicon through the real, unpatched `get_lexicon()`, since `_d6_learned_referent_env_flag_derisk.run_seed`
always forces a fresh singleton: this is CORRECT and intended for seed 42 specifically, since 42 IS a real
production/evaluation seed, at roughly double the compute cost of seed 7's dev check). Re-measured the CURRENT
DEFECT precisely at seed 42 with v2 (no junction flag) under the FIXED instrument
(`diag_frame_s42_v2_amendment2.json`): 64/112 turns changed, 35 mismatches (offending: amazing, before, crazy,
east, leaves, most, what, who). The 64-changed figure matches the review's own re-measurement exactly; the
mismatch count (35, not the review-quoted 33) reflects this amendment's own token-level ground-truth fix landing
on top of the review's count -- read 35 as the current, artifact-backed number, not a further discrepancy to chase.

**6. `run()`/`score()` provenance + MIXED-INPUT**
(`_lexicon_closed_class_parse_diag.py::run`, `_git_sha`, `score`). `run()` now records `git_sha` and, for the
junction variant, `constants` (W_J, I_TONIC_J, T_ON_J, DRIVE_MATCH_S, OR_LESION_FACTOR, OR_MATCH_FACTOR,
stp_enabled) on every artifact. `score()`'s input-identity tuple now also covers the token-fixture hash, the
constants blob and the git SHA, so two seeds run under different constants (an amendment landing between them) or
different code are `MIXED-INPUT`, not silently pooled as homogeneous.

**7. Doc accuracy** (corrections recorded here; the ORIGINAL committed prose in the pre-registration/AMENDMENT 1
and the dev-s7-not-ready finding is left as the historical record, not rewritten):
  - the module docstring's T_ON_J comment said the first spike to a pair comes "12-26 steps after onset"; the
    committed `and_calibration_s7.json` grid's actual range is 19-43 steps (matching AMENDMENT 1's OWN prose, which
    already said 19-43 -- the module docstring alone had the wrong numbers). Fixed in the module docstring.
  - the dev finding's 'most' tally said "24 of the 32 [occurrences] are 'the most' + a word outside the 100 frame
    words". Re-measured directly (`FrameEnvironment.occurrences('most', 32, 7)`): 26 are left-frame-only, 6 are
    COMPLETE (-1,+1) frames, and 2 of those 6 are 'the most fun' -- a genuine mid-sentence frame with a real
    right-neighbour, NOT a punctuation-stripping artifact as the finding's prose implied for all 6. This is a
    materially different diagnosis (some of the noun-leaning evidence for 'most' is real, not entirely an
    instrument artifact), which is exactly why mechanism C below could not be expected to fully clear 'most' by
    itself (see its result).
  - the dev finding's new-noun-recovery paragraph said "'circus' (heard 3 times) is lost", omitting that 'step' is
    ALSO lost (v2 recovers {circus, step, ...} = 26 words; the junction intact arm recovers 25, missing BOTH).
    Verified by diffing `diag_frame_s7_gt3.json` and `junction_s7.json`'s own `new_gt_nouns_recovered` lists.

### Mechanism changes (review's "next steps"; brain-based, no word lists in the MECHANISM -- the ground-truth
fixes above are instrument word lists, never read by the circuit)

**A. AND robustness: short-term depression (Tsodyks-Markram; Abbott, Varela, Sen & Nelson 1997, Science 275:220)**
on the FR->FJ synapses only. `cfg.enable_short_term_plasticity=True`, per-type E->E defaults (U=0.5, tau_d=200ms,
tau_f=20ms; `sim/config.py`); `stp_disabled=True` on the `"built"` explicit-wiring plan group (FJ->CN/CX + the
inhibitory pathways stay STP-free), left False (default) on `"fr_fj"`. A single fast-firing afferent (the 'day'
column that drove 100 of round 1's 102 `and_population` violations) now depresses with repeated firing and can no
longer deliver full-strength drive alone, while a FRESH coincident pair still can (Abbott et al.'s own point:
depressing synapses render steady-state response nearly rate-independent, which is exactly the property the
'day'-column violation needed). RE-CALIBRATED at dev seed 7 via `and_population` over ALL 10,000 junctions (not
the 64/256-sample smoke, which round 1's own finding named as the lapse that missed the column): W_J 300 -> 2950,
I_TONIC_J -650 -> -762.5 (STD lowers steady-state efficacy substantially, so the nominal weight must rise; selection
rule unchanged from AMENDMENT 1 -- widest feasible zero-violation bias range, ties to the smaller weight, middle of
the range: W_J=2950 and 3000 tie at feasible width 15 pA (2950: -770..-755; 3000: -785..-770), 2950 wins).
**Result: 0/10,000 and_population violations at the frozen point** (`and_population_stp_grid_s7_width.json`), down
from round 1's best of 47/10,000 (the pre-STP grid could not reach zero at any sampled point). DRIVE_MATCH_S
re-measured at the new operating point (`drive_ratio_s7_amendment2.json`): 27.53 -> 165.1 -- STP suppresses the
leaky firing the old ratio partly reflected, so junctions are genuinely sparser now, not just differently scaled;
W_INIT_J/ETA_J/OJA_BETA_J follow via the same S-rescaling AMENDMENT 1 defined.

**B. R4 (untrained circuit must abstain): homeostatic synaptic SCALING, not a threshold.**
(`FrameJunctionLexicon._r4_homeostatic_settle`, engaged once by `set_lesion("learned_edge")` before any `decide()`
reads it.) Round 1's R4 lesion (uniform+jittered start weights) read 0.333 recovered-both-rate against the 0.20 bar
-- the jitter does not average out over a single active junction per occurrence, so it decides some words for CN at
random. FIX: a RUNNER-SIDE Turrigiano-style scale update, `scale = 1 + rate*(target_rate - actual_rate)` per
postsynaptic CN0/CX0 neuron -- the IDENTICAL formula `sim/config.py`'s engine-level `enable_synaptic_scaling`
implements (`sim/bridge.py`'s fused synaptic-scaling block) -- applied to the FJ->CN/CX weight matrix over one
epoch of the curriculum's own words (no teacher), the same reason the Oja rule above is already runner-side rather
than the engine's generic path: the engine's OWN synaptic-scaling clip bound is `hebbian_max_weight` if Hebbian
learning is on, else a hardcoded 5.0 -- and this circuit's weight scale (~S x 40, S=165.1) is orders of magnitude
above that, so setting the engine flag would clip EVERY synapse in the bridge to <= 5.0 on the first step it runs
(exactly the BOUND TRAP `tools.lab.bound_check` exists to catch). Biologically: this models the compensatory
re-equilibration of population activity a lesioned circuit runs, not a per-decision threshold -- the companion
process the uniform-jittered start weights alone do not supply.

**C. 'most': a sentence-boundary PAUSE token in the heard stream, environment-only.**
(`lexicon_frame_junction.load_tokens_with_pause`, wired into `lexicon_spiking_frame_category.get_lexicon()`'s
junction branch only -- v2 untouched.) The shared tokenizer (`_comprehension_learned_animacy_cue_derisk.load_tokens`,
`[a-z']+`) strips ALL punctuation, so "the most. Tom ..." silently splices Tom into 'most''s right frame as if the
sentences ran together -- the IDENTICAL defect EMERGE-62b already named and fixed for the position cue
(research/findings/2026-07-03-emerge62b-position-cue-GO.md: "the corpus tokeniser strips ALL punctuation, so it has
no sentence boundaries"), by the same "host is legitimate for the syllabus" boundary that finding already used.
FIX: `load_tokens_with_pause` inserts one PAUSE_TOKEN (a sentinel no real corpus word can match) into the flat
token stream at each `[.?!]`, so a sentence-final word's right-frame afferent registers "a heard pause", not the
next sentence's first word. RAG-checked before building (`.venv-rag/bin/python tools/rag/rag_search.py "prosodic
pause boundary cue infant speech segmentation sentence boundary" 5 --corpus all`): the strongest hit was
EMERGE-62b's own prior fix of the identical defect (no independent literature named a stronger operationalisation
for THIS specific gap than "restore the boundary the tokenizer already discards"), so the consistency-of-frames
decision rule the dev finding also floated is NOT built here -- see the result below for why the pause token alone
was, honestly, not expected to fully clear 'most' once the recount above showed 2 of 6 complete frames ('the most
fun') are genuine, not punctuation artifacts.

### AMENDMENT 2 ADDENDUM (2026-09-24, independent adversarial code review, before the dev-check results below were
read): two issues found in the mechanism code above, both fixed before any dev-check number was trusted.

**Mechanism C, declared side effect (was undeclared).** `FrameEnvironment.ctx` (the C=100 context words) is built
from RAW token counts with no exclusion list, and PAUSE_TOKEN is the single MOST FREQUENT token in the corpus
(176,822 occurrences on the full 19,971,040-byte tinystories.txt -- a sentence boundary is more common than any one
word; seed-independent, since `ctx` never depends on seed). It therefore wins a context-word slot on the SAME
frequency basis every other context word does, displacing exactly one word from the prior top-100: 'make' (a
common verb, not a curriculum or battery-critical word). This is the intended mechanism operating as designed
(PAUSE_TOKEN must occupy a real slot to be usable as a frame neighbour at all), not a bug -- but AMENDMENT 2's
original text did not say so, and a reviewer had to derive it by reading `FrameEnvironment.__init__`. Declared here
and in the module docstring. Not measured: whether losing 'make' as a neighbour-context measurably changes any
OTHER word's frame evidence (plausible, not expected to be large -- one slot in 100). The dev-check runs already in
flight when this was found use the corpus/environment exactly as measured here, so no re-run was needed for this
item.

**Mechanism B, a latent (never-triggered) state-machine gap.** `_r4_homeostatic_settle` writes `data[self.S]`
directly and calls `_install()` only afterward, so it never re-writes `data[self.S_inh]` / `data[self.S_j]`. Had
`set_lesion("learned_edge")` ever been reached while some OTHER lesion ("coincidence", "competition") was already
installed, the settle would have run its curriculum presentations against those lesioned inhibition/junction
weights instead of the intact circuit. No call site in this lane does that (`set_lesion` always reaches
`learned_edge` from an otherwise-intact lexicon), so this never fired -- confirmed by the reviewer reading every
call site, not assumed. Fixed defensively: `_r4_homeostatic_settle` now asserts `self.lesion is None` on entry, so
a future caller cannot introduce this silently.
