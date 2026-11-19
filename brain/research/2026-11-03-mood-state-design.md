# Mood / arousal state that shapes replies: design + record brief (2026-11-03)

Milestone (brain/ARCHITECTURE.md §4, 2026-11-30): "mood/arousal state shapes replies (lesion-proven)".
Capability: `brain/PREREQUISITES.md` "Affect shaping speech" (was NOT-READY: no state existed).
Owner rules that bind this design: feelings shape speech, never marker words; affect is multimodal and grounded,
not text-only; self-reports are functional read-outs, never claims of experience; brain-based only (host code only
for world / body / teacher or a declared scaffold).

## 1. What the biology says (the parts we need, and only those)

- **Core affect is two-dimensional:** valence (good/bad) x arousal (calm/activated) (Russell 1980; Barrett). It is
  grounded in the body (interoception: Craig 2002; Barrett & Simmons 2015), so the anchors of "good" and "bad" are
  bodily and social events, not words.
- **Emotion is fast and event-bound; mood is slow and diffuse.** An emotion is an appraisal of an event (novelty,
  intrinsic pleasantness, goal conduciveness, coping; Scherer's component process model). Mood is the slow residue:
  Eldar & Niv (2015, 2016) model mood as a running integral of recent outcomes / prediction errors, which in turn
  biases how the next outcomes are perceived. That is a leaky integrator with a long time constant.
- **Amygdala appraisal:** basolateral amygdala holds opponent positive- and negative-valence populations (Namburi,
  Tye et al. 2015; catalog O.58, O.12, O.13) that read cortical input and attach value via innate unconditioned
  stimuli and learned associations, then project back to cortex and to the neuromodulatory nuclei.
- **Neuromodulatory tone carries the state to cortex, diffusely:** noradrenaline (locus coeruleus) = arousal and
  adaptive GAIN on cortical input-output functions (Servan-Schreiber, Printz & Cohen 1990; Aston-Jones & Cohen 2005;
  BIO/locus-coeruleus-ne-adaptive-gain-swap.md); dopamine tonic level ~ average reward rate / vigour (Niv 2007);
  serotonin ~ aversive prediction, patience, negative-mood tone. None of these is a word; each is a low-dimensional
  signal broadcast through a fixed receptor pattern.
- **Mood-congruent processing** (Bower 1981; meta-analyses since): a mood biases what is retrieved and expressed
  toward congruent content. That is the observable we test: the same question gets a differently coloured answer.

## 2. The old track: what to port, what not (record, read 11-03)

| Old-track result | Lesson | Port? |
|---|---|---|
| `F/2026-07-24-P0.3-affect-state-region` (QUALIFIED-GO/BOUNDARY) | a slow opponent attractor holds a persistent mood that causally biases cognition, but as a bistable good/bad latch, not graded | PORT the opponent (pos - neg) structure and persistence; NOT the spiking latch (graded value needed) |
| `F/2026-08-08-graded-affect-staggered-bistable-ladder` (GO) | graded mood needed a quantized ladder of latches to be drift-robust in spiking | NOT now: a float leaky integrator is graded for free here; revisit when the core moves to spiking (2027 Q1) |
| `F/2026-08-13-...appraisal-origin-self-organizes-from-reinforcement` (DR-2b, 6-seed GO) | concept valence can come from ~10 innate primary reinforcers by evaluative conditioning; no hand lexicon | PORT the idea: innate bodily + social primaries (seed-drawn subset) define the valence axis in the cortex's own code |
| `F/2026-08-12-D1-...not-hardcoded-lexicon` | the Warriner word table was a host shortcut and had to be retired | DO NOT use a word list inside the brain; the only word list is the declared measuring instrument, outside the brain |
| `F/2026-08-13-affect-arousal-channel-BOUNDARY` | arousal as the SUM channel of the same opponent pools predicts arousal, modestly (r=+0.27) | PORT: arousal = fast leaky magnitude of appraisal (read out now; acting on cortex is the next arm) |
| `F/2026-09-04-gnw-lc-ne-adaptive-gain-swap` (DE-RISK GO) | NE as adaptive gain, lesion leaves a floor | PORT as the arousal design (gain, below) |
| `F/2026-08-12-GateB-affect-colors-production-chat` + Stage-A step 2 | affect must colour content and prose, never a tone tag; affect sits BELOW honesty (moat < honesty < affect: it never manufactures an answer or overrides an abstain) | PORT both rules. NOT the Qwen `MoodConditionedRenderer`: here the cortex itself speaks |
| affect ratchet / active-clear findings (07-31, 08-01) | a mood that cannot be evicted is a defect | the leaky integrator relaxes by construction; test recovery in the multi-week life (12-14) |

## 3. How the state is produced from the conversation

1. **Birth (scaffold, the body):** the cortex reads a seed-drawn subset of innate primaries through a `Body:` channel
   (warmth, food, a gentle hug, a friend smiling vs pain, cold, sickness, being shouted at, separation). The valence
   axis = mean positive trace - mean negative trace in the cortex's fast memory (all 12 layers). This is the
   amygdala's innate hook onto cortical codes; nothing is hand-labelled after birth.
2. **Appraisal (fast emotion):** each event the brain hears is read; its fast-memory trace (per-layer normalized) is
   projected on the axis, centred on neutral statements: `v = tanh(proj / scale)` in (-1, 1). Praise, gain and reunion
   should land positive; threat, loss, insult negative. Novelty (the cortex's own surprise, mean NLL of the turn)
   is the planned second input to arousal; not built in the prototype.
3. **Mood (slow):** `mood += (v - mood) / TAU_MOOD` (TAU = 4 events); `arousal += (|v| - arousal) / 1.5`.
   Restart/sleep should relax mood toward 0 (wired with sleep later, not in this prototype).

## 4. How it acts on the cortex (options weighed)

| Option | Biology | Verdict |
|---|---|---|
| A. Additive signed bias on the matrix memory: `state += GAIN * mood * axis` before the turn is read | diffuse valence projection; mood-congruent priming | **BUILT** (prototype). One scalar through one fixed pattern. Risk: at high gain the primaries' content intrudes (seen at GAIN 32), so GAIN is capped and primary echo is measured |
| B. Arousal as multiplicative gain on the output distribution (inverse temperature) | LC-NE gain on input-output functions (Servan-Schreiber 1990) | NEXT arm, for AROUSAL only, never for valence; justified as neuromodulatory gain, lesion = gain pinned at 1 |
| C. Arousal on per-head decay rates (how fast the cortex forgets context) | NE/ACh shift integration vs reset | later; needs the cortex forward to be opened (plastic_cortex has the hook) |
| D. Logit bias toward affect words | none | REJECTED: marker words, host templating |
| E. Text in the prompt ("You feel sad") | none | REJECTED: text injection, owner rule |

Composition law (from Stage-A): affect never decides WHETHER to answer or WHAT fact is recalled; the hippocampus
and the familiarity bar do. Affect only colours the cortex's own speech.

## 5. Lesion test (how we prove it changes the reply)

The probe (`brain/experiments/mood_probe.py`) cuts every other path: the induction events are appraised, but the
question is read from the same birth state in every arm, so the ONLY path from the events to the reply is the mood
scalar through the modulation. Common random numbers: identical sampling seed per (question, sample) in every arm.
Arms: pos / neg mood with modulation on; the same moods with modulation OFF (lesion); the same moods along a RANDOM
direction of the same per-layer norm (specificity: is it the valence axis, or any perturbation?).
In the live brain the stronger test is the weekly transcript with affect on vs off after the same conversation.

**Pre-registered read-outs (written before the six-seed run):** primary instrument = lexical sentiment per 100
words (declared, outside the brain, disjoint from every primary word); secondary = MiniLM embedding sentiment.
PASS for the prototype on 0.1B if: (1) mood diff (pos - neg) > 0 with the paired bootstrap 95% CI above 0 on the
lexical instrument on >= 5/6 seeds, and same sign on the embedding instrument on >= 5/6; (2) lesion: replies under
pos and neg byte-identical (fraction 1.0) on 6/6; (3) random direction: |lex diff| below the mood diff on >= 5/6;
(4) echo of primary words under mood not more than 0.10 above the lesion arm (mean); fluency (NLL under the
unmodulated cortex) reported, not gated. Appraisal sign accuracy reported per seed (on the same event pool the induction drew from; NOT held out).
GAIN was set on dev questions (separate from the test questions), seed 42 only: 8 weak, 16 clear, 32 intrudes
primary content ("pain", "exploded"); 16 frozen before the six-seed run.

## 6. Honest read-out

`Affect.readout()` returns numbers (mood, arousal, last appraisal, modulation on/off). Any verbal self-report must
come from the cortex itself, and is scored by how well it tracks these numbers (a functional read-out), never by a
host template, and never phrased by us as experience. Not built in the prototype; it belongs to the 12-14
"honest self-reports" milestone.

## 7. Scale and prerequisites

Tone control is partly an LM property (PREREQUISITES: check a small and a large size before attributing). 0.1B
replies are short and generic; the appraisal (a linear read of the fast-memory trace) is lexical like the
hippocampus key. Expected weak point: positive events sit close to neutral on the axis (a negativity asymmetry
seen in calibration); a learned appraisal (conditioning on outcomes, DR-2b style) is the method to try next.

## 8. Result (11-03)

Six seeds at 0.1B: `brain/experiments/results/2026-11-03-mood-0.1b.md`. CORRECTED after the skeptic: NOT a
pre-registered PASS. Criterion 4 (echo) fails on s101/s102 (+0.111 over the lesion arm, bar 0.10) under the echo
instrument registered here; the instrument was revised mid-run (invalid for the verdict). Criterion 2 holds by
construction (the switch returns the state untouched). The appraisal events were not held out. The on-vs-off contrast
is one-sided: positive mood ~0 on s100/s101. Positive appraisal weak on 2/6 seeds; not wired into the live brain.

## 9. Iteration 2 (11-03): PRE-REGISTERED before any iteration-2 run; instruments frozen

Why positive events read ~0 (dev diagnosis on the induction pool only, s42-s102, before this section was final):
on the current axis (mean positive primary trace - mean negative, all `Body:`), negative events land at -0.03..-0.14
while positive events land at -0.06..+0.09, inside the spread of neutral statements (-0.03..+0.04). Negative events
sit closer to the negative primaries (cos 0.25-0.30) than positive events sit to the positive ones (0.21-0.24, about
as close to the negative pole). The primaries are unbalanced: 7 bodily + 6 social positive vs 10 bodily + 3 social
negative, and the seed draws 8 of 13 at random, so a seed can get almost no social-positive anchor while the events
we appraise (praise, gain, reunion) are social. The axis therefore mostly encodes "bodily-harm vs not".

**(a) Fix, chosen on DEV events only by a rule fixed here.** Candidates: draw {orig = 8 of 13 at random,
bal = balanced lists (8 bodily + 8 social per sign; 1 bodily-positive, 2 social-positive and 5 social-negative primaries added, none sharing a word with
the lexical instrument lists, asserted by test), stratified draw 4 bodily + 4 social per sign} x read-out {lin =
current single axis, centred on neutral; twopole = cosine to each pole (pos and neg primary centroids, each centred on
the neutral reference), each pole z-scored on the neutral reference, valence = z_pos - z_neg (opponent BLA
populations, Namburi 2015); orth = pole-specific parts (each pole minus its projection on the other), each scaled so
its own primaries read 1}. Selection rule: on the dev pool (the 8 + 8 + 4 induction-pool events of `mood_probe.py`),
maximise the minimum over the six seeds of positive sign accuracy, subject to negative sign accuracy >= 0.875 on every
seed; tie -> larger mean positive d' vs neutral; tie -> simpler (lin < orth < twopole; orig < bal). The winner and its
scale are frozen in section 10 before the six-seed run.

**(b) Held-out appraisal.** A NEW disjoint event set (8 positive, 8 negative, 4 neutral; different scenarios, written
now, never used for induction, method selection or GAIN) is appraised per seed with the frozen method. Induction
draws 6 of the 8 dev-pool events per valence (as before).

**(c) Real lesion on both valences.** Arms: `pos`, `neg` (modulation on), `off` (modulation off = the birth state;
one arm because off is identical for any mood by construction, which is NOT counted as evidence), `rand_pos`,
`rand_neg`. Same 12 test questions x 3 samples, common random numbers per (question, sample) across arms, GAIN 16
unchanged (no retuning), TAU 4, max 60 tokens, 0.1B, seeds 42/43/44/100/101/102.

**(d) Random direction, matched norm.** One random direction per seed with each layer's Frobenius norm equal to the
frozen modulation pattern's, driven by the same mood scalar.

**Instruments (frozen):** lexical sentiment per 100 words (primary; word lists unchanged), MiniLM embedding sentiment
(secondary), echo = the REVISED content-word instrument (`primary_echo`, ECHO_STOP unchanged) over the primary lists
in force in iteration 2, applied to every seed; the registered-original echo is reported, not gated. Paired bootstrap
95% CI over the 36 (question, sample) pairs, 2000 resamples, numpy rng seeded with the seed.

**PASS criteria (each on >= 5/6 seeds unless stated):**
1. Held-out appraisal: positive sign accuracy >= 0.75 AND negative >= 0.75 (per seed).
2. Induced mood signs: m_pos > 0 and m_neg < 0 on 6/6.
3. Lesion, positive valence: lex(pos) - lex(off) CI above 0.
4. Lesion, negative valence: lex(neg) - lex(off) CI below 0.
5. Embedding agrees: emb(pos) - emb(off) > 0 and emb(neg) - emb(off) < 0 (means).
6. Specificity: paired (lex(pos) - lex(neg)) - (lex(rand_pos) - lex(rand_neg)) CI above 0.
7. Echo: mean echo of {pos, neg} minus echo of off <= 0.10 on 6/6.
Reported, not gated: fluency NLL under the unmodulated cortex, fraction of replies changed vs off, held-out neutral
appraisal, random arms vs off. Verdict words: all 7 met -> "prototype PASS (lesion on both valences)"; 3 or 4 failing
-> positive still one-sided, said plainly. A PASS is a lead toward the 11-30 milestone, not the milestone (not wired
into `Brain`, 0.1B only).

## 10. Iteration 2: dev selection record and FROZEN method (written before the six-seed run and before any held-out read)

Dev round 1 (the 6 candidates of section 9, dev pool only; min / mean positive sign accuracy over six seeds, negative
accuracy 1.00 everywhere): orig+lin 0.12 (the iteration-1 method), orig+twopole 0.00, orig+orth 0.12, bal+lin 0.25,
bal+twopole 0.00, bal+orth 0.25. NONE met the rule; balancing alone does not fix it on the all-layer read-out.
Dev round 2 (added after round 1, declared here as a second fork): the read-out layers {each single layer 0-11, all,
shallow half 0-5, deep half 6-11, last 3} x {axis = pos - neg, or two poles each centred on 6 neutral BODY states
(`BODY_NEUTRAL`) and unit-normalised before subtraction} x {orig, bal}: 64 cells. Best structured cell: **bal + body-
centred two poles + deep half (layers 6-11)**: positive 0.88-1.00, negative 1.00 on all six seeds (mean positive
0.92). Single layers 7 and 9 reach similar minima but are not chosen (one layer is a sharper fork; layer 9 drops
negative to 0.38 on s42). Biology fit: the amygdala's valence populations read association cortex, not early sensory
code (deep layers here); opponent positive/negative populations each have their own baseline (Namburi 2015).
With 70 cells tried on 20 dev events, the dev accuracy is optimistic by construction; the HELD-OUT set (section 9b)
is the test.

FROZEN for the six-seed run: `Affect(readout="twopole_deep")` (the new default), `draw_primaries` (4 bodily + 4
social per sign), modulation pattern unchanged in form ((mean pos - mean neg primary trace) / 2, all layers, now
from the balanced draw), GAIN 16, TAU 4, `mood_probe.py` defaults (arms pos,neg,off,rand_pos,rand_neg; 12 test
questions x 3 samples; 60 tokens), criteria and table: `brain/experiments/mood_table_v2.py`.

## 11. Iteration 2 result (11-03)

`brain/experiments/results/2026-11-03-mood2-0.1b.md`: NOT PASS, 6/7 criteria. Held-out appraisal 5/6, mood signs
6/6, negative lesion 6/6, embedding 6/6, specificity 5/6, echo 6/6; positive lesion (pos vs off, lexical CI above 0)
3/6. Appraisal is fixed; the remaining defect is an action asymmetry (one additive pattern darkens more easily than it
brightens; friendly birth register near ceiling; same-norm random perturbation lowers sentiment).

## 12. Iteration 3 (11-03): PRE-REGISTERED before any iteration-3 run; instruments frozen

Target: iteration 2 failed only its criterion 3 (positive mood vs off, 3/6). Two untested readings: (i) ceiling, the
friendly test register's off replies already read +2.5..+6.1 lexical; (ii) a generic perturbation cost, a matched-norm
random direction lowers sentiment for both signs (mean -1.29 / -2.22), so "vs off" mixes the valence effect with the
cost of any push. Iteration 3 changes the TEST REGISTER and the REFERENCE, nothing in the brain: affect.py, the
primaries, the two-pole deep-half appraisal (at 0.4B: layers 12-23, the same rule), TAU 4, the switch (default OFF),
the instruments (lexical lists hash-frozen, MiniLM anchors, revised echo) are unchanged. No words are added to the
cortex input (each question is read from the birth state, as before).

**(a) Register, selected on DEV questions only.** Two candidate registers written before any run
(`mood_probe.QUESTION_SETS`): `plain` (affect-free objects and places: "Describe a staircase.") and `mixed`
(ambivalent everyday situations: "What happens when you miss the bus?"), each 6 dev + 12 test questions, disjoint,
no instrument word in any question (test). Iteration 2's friendly dev questions are run as a reference only.
Dev data: off arm only (no modulation, so no mood effect is seen), seeds 42 and 43 (sampling seeds), 6 dev questions
x 2 samples, 40 tokens, at 0.1B and 0.4B. Rule (`mood_v3.select_register`): among candidates whose off replies contain
at least one instrument word in >= 25% of replies at both sizes, choose the one minimising, over the two sizes, the
larger |mean off lexical| (mid-scale = 0, as many positive as negative affect words); if none qualifies, the one with
the higher minimum coverage. The test questions of the chosen register are the test set; they are never used to
select anything.

**(b) Random-direction NULL for both signs.** Per seed, 2 random directions (each layer's Frobenius norm matched to
the frozen modulation pattern; direction 0 is iteration 2's, direction 1 the next draw of the same generator), each
driven by m_pos and by m_neg. null_pos = mean over the 2 directions of the rand_pos replies' score, per (question,
sample); likewise null_neg. Mood effect = pos - null_pos and neg - null_neg, paired, bootstrap 95% CI (2000
resamples, numpy rng seeded with the seed). The off arm stays (lesion) and pos - off / neg - off are reported.

**(c) Sizes and GAIN.** 0.1B: GAIN 16 (dev-chosen in iteration 1; not reselected). 0.4B: GAIN chosen on DEV only by
`mood_v3.select_gain`: seed 42, the chosen register's 6 dev questions x 1 sample, arms pos/neg/off, 40 tokens,
candidates {8, 16, 32}: maximise lex(pos) - lex(neg) subject to echo(mood) - echo(off) <= 0.10 and mean NLL(mood) /
NLL(off) <= 1.25 under the unmodulated cortex; tie -> smaller; none -> 8. Frozen before the 0.4B six-seed run.

**Test run (both sizes, six seeds 42/43/44/100/101/102, CPU, <= 3 concurrent jobs, 2 threads each):** arms pos, neg,
off, rand_pos, rand_neg, rand1_pos, rand1_neg; the chosen register's 12 test questions x 2 samples (24 paired rows),
max 40 tokens, common random numbers across arms. Smaller than iteration 2 (36 rows, 60 tokens) for compute only:
the workstation runs at load ~65 on 20 cores tonight; a 0.4B reply costs ~20 s.

**PASS criteria, per size (each on >= 5/6 seeds unless stated):**
- C1 held-out appraisal: positive and negative sign accuracy >= 0.75 (unchanged).
- C2 mood signs m_pos > 0 > m_neg on 6/6.
- C3 positive: lex(pos) - null_pos, CI above 0.
- C4 negative: lex(neg) - null_neg, CI below 0.
- C5 embedding agrees: emb(pos) - null_pos > 0 > emb(neg) - null_neg (means).
- C6 echo: mean echo of {pos, neg} - echo(off) <= 0.10 on 6/6.
Reported, not gated: pos - off and neg - off (iteration 2's criteria 3/4, on the new register) with CIs; null - off
per sign (the perturbation cost); the off arm's mean lexical on the TEST questions (is it mid-scale?); NLL; held-out
neutral appraisal. Verdict words: all six met at a size -> "prototype PASS at <size> against a random-direction null";
the writeup must then say whether positive mood also beats OFF (absolute brightening) and must not call a
null-relative PASS an absolute one. A PASS is a lead toward the 11-30 milestone, not the milestone (not wired into
`Brain`). Table: `python -m brain.experiments.mood_v3 table <dir> <size>`. JSONs record commit, torch version and
thread count (`provenance`; reproducibility note: a rerun must use the same --threads).

## 13. Iteration 3: dev selection record and FROZEN settings (written before any test-question result was read)

Register (`python -m brain.experiments.mood_v3 register results/2026-11-03-mood3/dev`; off arm, dev questions, seeds
42/43 x 2 samples, 24 replies per cell): plain 0.1B +0.25 (coverage 0.08), 0.4B +0.00 (0.00): fails the coverage
floor (its replies carry almost no affect words); mixed 0.1B +1.79 (0.38), 0.4B +0.73 (0.33); friendly (reference)
0.1B +4.41 (0.50), 0.4B +2.63 (0.58). SELECTED: `mixed`.
0.4B GAIN (`mood_v3 gain`, seed 42, mixed dev x 1 sample): 8 -> pos-neg +0.57 (ok); 16 -> +2.37 (echo -0.08, NLL
ratio 1.06, ok); 32 -> +10.11 but NLL ratio 1.66 (excluded). SELECTED: 16 (same as 0.1B).
FROZEN for the six-seed runs at both sizes: `--questions mixed_test --samples 2 --max-tokens 40 --arms
pos,neg,off,rand_pos,rand_neg --n-rand 2 --gain 16 --threads 2`.

## 14. Iteration 3 result (11-03)

`brain/experiments/results/2026-11-03-mood3.md`: NOT PASS at both sizes. 0.1B 5/6 criteria: positive mood brightens
vs the random null 6/6 and vs OFF 6/6 on the mid-scale `mixed` register (the iteration-2 defect was the friendly
register's ceiling); fails C6 echo on s43/s101 (+0.104/+0.125). 0.4B 2/6: negative vs null 6/6, positive vs null
1/6 (a random push raises sentiment at 0.4B, +1.7), held-out positive appraisal 4/6. Caveat: a quarter to a third of
negative-mood replies are apology/refusals ("I'm sorry, I can't help"); tone excluding "sorry" stays negative at about
half size (exploratory).

## 15. Iteration 4 (11-18): the frozen iteration-3 method at 1.5B. PRE-REGISTERED before any 1.5B run

Record brief:
- Iteration 3 left a lead at 0.1B (5/6 criteria, only echo failed) and a negative at 0.4B. At 0.4B positive mood was
  no different from a random push (1/6). `results/2026-11-03-mood3.md`.
- 1.5B is now the working size (open conversation, weekly). The mood method has never run there.
- PREREQUISITES: "tone control is partly an LM property; check a small and a large size". 0.1B and 0.4B disagree,
  so the 0.4B negative is a verdict on that size, not yet on the method.
- Prerequisites met: cortex fast memory, innate primaries (body scaffold), appraisal. Host decision: none new.
- Expected: negative mood darkens (it did at both sizes). Positive vs the null is the open question. Risk: a 1.5B
  random push may brighten replies as it did at 0.4B.

Method:
- Unchanged from sections 12 and 13: affect.py, primaries, two-pole deep-half appraisal (1.5B: the deeper half of
  its layers, same rule), TAU 4, `mixed` register, instruments hash-frozen, null of 2 random directions.
- GAIN chosen on DEV only by `mood_v3 gain <dir> 1.5b`. Dev data: seed 42, `mixed_dev` x 1 sample, arms
  pos/neg/off, 40 tokens, candidates {8, 16, 32}. Same rule as section 12(c). Recorded in section 16 before any
  test run.
- Test: six seeds 42/43/44/100/101/102 on pool CPU, `--questions mixed_test --samples 2 --max-tokens 40 --arms
  pos,neg,off,rand_pos,rand_neg --n-rand 2 --gain <dev> --threads <node>`.

Criteria: C1-C6 of section 12, unchanged, at 1.5B.

Decision:
- PASS at 1.5B: wire `Affect` into `Brain` behind its switch (default OFF). Then judge it on open conversation with
  the mood lesion, read whole, before any flip.
- C3 fails (positive no different from a random push) with C4 passing: pole-specific patterns, section 14 next item 2.
- C4 fails too: the additive matrix-memory bias does not shape tone at 1.5B. NO-GO on the method; next is a
  gain-type (arousal) modulation in place of an additive push.

## 16. Iteration 4: dev selection record and FROZEN settings (written before any test run)

- GAIN (`mood_v3 gain results/2026-11-18-mood4-1.5b/dev 1.5b`, seed 42, mixed dev x 1): 8 -> pos-neg +1.84 (ok);
  16 -> +4.52 (echo +0.000, NLL ratio 1.160, ok); 32 -> +3.12, NLL ratio 1.376 (excluded). SELECTED: 16.
- Embedding: the pool nodes have no `transformers`. The test runs `--no-embedding`; the frozen MiniLM score is added
  on the workstation by `mood_probe --add-embedding`, which recomputes summary and contrasts with the same rng order.
  Checked: on `mood3_0.1b_s42.json` it reproduces every contrast exactly (`tests/test_mood_add_embedding.py`).
  The JSON records `embedding_added_on`.
- FROZEN: `--model 1.5b --questions mixed_test --samples 2 --max-tokens 40 --arms pos,neg,off,rand_pos,rand_neg
  --n-rand 2 --gain 16 --no-embedding --threads <nproc-1>`; seeds pool40 42/43, pool41 44/100, pool42 101/102.

## 17. Iteration 5 (11-18): matched-pair axis, mid band, norm-scaled, push after the question is read. PRE-REGISTERED before any run

**Method.** Same as §15, except:
- the axis is the mean of 32 matched sign-flipped body pairs;
- the appraisal reads layers 8-15, zeroed at the pair-pole midpoint read as User;
- the push is alpha × mood × ‖M_l‖ × û_l on layers 8-15 only, added after the question and before the first reply token;
- mood is induced by the §15 integrator (TAU_MOOD=4).

**Dev rule, frozen before test.** Only dev events and `mixed_dev` are used, on seeds 42/43/44.
- **Stage A gate.** On every dev seed, pair-axis sign accuracy is at least 0.75 on both poles and beats the topic axis by at least 0.25. Mean |v| of neutral dev events is at most 0.15. If the gate fails, stop: the appraisal fault is not fixed by pairs.
- **Alpha.** The largest alpha with mean NLL ratio (mood arms / off) ≤ 1.16 on all three seeds. The lexical score is never consulted.
- **Discriminator.** Pooled over 72 rows per arm, the paired-bootstrap 95% CI of [(pos−neg) − (topic_pos−topic_neg)] and of [(pos−neg) − (shuf_pos−shuf_neg)] must both exclude 0. Otherwise stop before six seeds and go to branch 3.
- Band, placement and pair count are fixed in advance and not tuned.
- Before the test, a dev-selection record goes in a new section 18.

**Test.** 6 seeds (42/43/44/100/101/102), `mixed_test`, 3 samples, 40 tokens, 7 arms. About 40 min per seed; 2 seeds per node is about 80 min wall.
- Seeds 42/43/44 also ran dev, but on dev questions only. This is stated as a limit.

Each criterion holds on at least 5/6 seeds unless stated:
- **C1:** sign accuracy on the V5_TEST events ≥ 0.75 on both poles, and neutral mean |v| ≤ 0.15.
- **C2:** m_pos > 0 > m_neg on 6/6.
- **C3:** lex(pos) − mean(topic_pos, shuf_pos), CI above 0.
- **C4:** lex(neg) − mean(topic_neg, shuf_neg), CI below 0.
- **C5:** the embedding score agrees in sign (means).
- **C6:** echo of {pos, neg} − echo(off) ≤ 0.10 on 6/6, with pair words included.
- **C7:** NLL ratio ≤ 1.20, and mean reply length within ±25% of off.
- **C8:** neg-arm refusals ≤ off + 3 of 36.
- **C9:** the reply changes: under the lesion (off), replies are byte-identical in a repeat, and pos differs from off in at least 50% of replies.

Reported, not gated: pos−off and neg−off, the dose slope compared with iteration 4, and the per-seed alpha × m.

**Decision branches.**
1. **PASS** is a lead only. Next: wire it into `Brain` (OFF by default), run a lesion on whole open-conversation transcripts, and apply `land-result`.
2. **C3 fails but C4 passes, or the other way round.** Report a one-sided result. Next: arousal as gain, either on the readout's inverse temperature (declared scaffold, lesion = gain 1) or on per-head decay (that needs the forward opened).
3. **The Stage A or discriminator gate fails.** Verdict: the additive state push, as a method, fails at 1.5B. Judge it against scale and prerequisites, then move to gain-type modulation or mood-congruent recall with the margin and confabulation fixes.

Record brief (from the research workflow wf_07a11b91-c66: record, steering literature, biology, our 1.5B data; 3 designs, 6 skeptics):
- Record check rerun after `w_follow.sh` (11-18, recorded): the old track's affect-on-language wall was the same signature (a fixed push swallowed by a sharper LM; a margin-scaled push registered, 09-04).
- Old track: a fixed push was swallowed by a sharper LM (09-04 linattn: byte-identical output). A push scaled to the model's own margin was load-bearing (09-04 sharpness-aware GO). The 1.5B result is the same signature.
- No mood or arousal result that `brain/` uses has been retracted.
- iteration-4 JSONs:
  - The axis ranks events correctly (pos vs neg AUROC 1.00 on 6/6 seeds). Its zero is off: neutral events sit near -0.11, and m_pos is between -0.04 and +0.10 on 4/6 seeds. So C3 tested almost no push.
  - The effect grows with push size (neg vs null slope +0.66 per push unit, r=0.94).
  - The true axis costs less fluency than a random one (NLL 0.94-1.10 vs 1.22-1.59).
- Literature (CAA, ActAdd, RNN state steering): many matched pairs, a mid band, size as a fraction of the layer norm, applied after the prompt. We do the opposite on all four.
- The old held-out appraisal events have now been looked at post hoc. They are burned for C1, so fresh events are needed.

Build plan and full synthesis: `brain/research/2026-11-18-mood5-plan.md`.

## 19. Iteration 6 (11-18): arousal as encoding gain. PRE-REGISTERED before any run

(Section 18, the iteration-5 dev-selection record, was never written: Stage A failed and stopped the iteration.)
Plan, build list and judges: `brain/research/2026-11-18-arousal-gain-plan.md`.

**Method**
- **Drive:** the mean of the two highest per-token NLLs of the heard turn, from the current state.
- **Calibration:** MU and SD come from V5_DEV neutral events plus 12 ordinary sentences read from birth. They are disjoint from the stage-0 items and from both banks.
- **Gain:** `g = min(1 + SPAN*a, 2)` multiplies the heard turn's matrix-memory write, both for this reply and in the committed state. No direction and no words are added. The gain never goes below 1.
- **Conversations:** 4 turns each.
  - T1: an event, either surprising (a coined item or an unexpected event) or mundane (the same frame with a common noun).
  - T2 and T3: fixed mundane fillers.
  - T4: the open probe "How's your day going?".
- **Banks:**
  - Dev: 12 conversations (6 surprising, 6 mundane).
  - Test: 24 conversations (12 matched pairs), with coined words new to every earlier run.
- **Arms** (same sampling seed per conversation):
  - OFF.
  - ON.
  - LESION: g pinned at 1, arousal still computed and logged.
  - SHUFFLED: the g sequence of another conversation at the same seed.
  - CONSTANT: the mean ON g on every turn.
  - CURRENT: TAU = 1, no carry-over. This arm decides only the claim's name.
- **Seeds:** 42/43/44/100/101/102, 2 samples each. Seeds only resample the replies, so **the unit is the conversation**, and the paired bootstrap resamples conversations.
- **Readers:** Haiku and Sonnet, blind to arm and class, with order shuffled. Each sees the T1 text and one reply and gives three labels:
  - (a) takes up the T1 event;
  - (b) echoes or copies the user;
  - (c) invents a meaning.
- **Reader agreement:** Cohen's kappa on (a) must be at least 0.6 over a 40-item overlap. If it is lower, the verdict is INCONCLUSIVE.

**Dev rule** (frozen and committed as 19.1 before test)
- **Pilot (seed 42, SPAN 1; OFF, ON, CONSTANT; about 100 replies, 20 min):**
  - Gate: mechanical carry at T2 and T4 must be ON − CONSTANT ≥ +0.10, with no rise in echo.
  - If it fails, stop with a NO-GO for the method and go to branch (iii).
- **Dev (seeds 42/43/44, SPAN in {0.5, 1, 2}):**
  - Choose the SPAN with the highest mechanical carry of ON minus CONSTANT, subject to fluency ratio ≤ 1.15 and echo rise ≤ 0.05.
  - Futility: if that SPAN gives less than +0.10 pooled, stop.
  - Floor: if LESION's T2 carry on surprising conversations is below 0.15, the bank has failed. Reframe it once and do not tune SPAN.
  - Readers are never used on dev.

**Criteria** (test bank; all must hold except the claim name)

| | Measure | Bar |
|---|---|---|
| C0 | Arousal after T1, surprising vs mundane | AUROC ≥ 0.90; filler gain ≤ 1.1 on ≥ 90% of filler turns |
| C1 | Blind (a) rate at T2+T4, surprising conversations | ON − LESION ≥ +0.15, 95% CI > 0, positive on ≥ 5/6 seeds |
| C2 | Shuffled and constant nulls | ON − SHUFFLED ≥ +0.10 and ON − CONSTANT ≥ +0.10, each with CI > 0 |
| C3 | Selectivity, mundane conversations | \|ON − LESION\| ≤ 0.05 |
| C4 | Lesion changes the REPLY | ≥ 50% of ON replies differ from LESION on turns with g ≥ 1.2; ≤ 10% differ on turns with g < 1.05 |
| C5 | Harms (ON − LESION) | echo (`_is_copy` or reader b) ≤ +0.05; invents (c) ≤ +0.05; questions-back ≤ +0.05; no rise in "User:" leaks; fluency ≤ 1.15 |
| Name | ON vs CURRENT | ON − CURRENT ≥ +0.05: call it "arousal state sets encoding gain". Otherwise "surprise-gated encoding". |

**Decision branches**
- (i) All of C0-C5 hold: GO for "arousal shapes replies (lesion-proven)", under the earned name. Then run land-result with a fresh skeptic. The switch stays OFF.
- (ii) C1 holds but C2 fails: plain heard-turn gain works, but arousal timing is not shown. NO-GO for the arousal claim.
- (iii) C5 fails, or the pilot or dev futility gate fails: NO-GO for flat turn-level gain. Next is per-token gain on the matrix write of the high-NLL tokens only.
- (iv) C0 fails: the drive unit is wrong. Re-calibrate on dev only.

### 19.0 Amendment (11-19, after the 0.1B smoke, before any 1.5B run)
- The C0 filler bar ("gain <= 1.1 on >= 90% of filler turns") contradicts the registered carry-over: TAU 1.5 keeps
  arousal raised on the turn after a surprising T1 by design (0.1B smoke: T2 after surprising T1 mean gain 1.18,
  mundane fillers and T3 all <= 1.1). Pooled over both classes it cannot pass whenever half the bank is surprising.
- Amended C0 filler bar: gain <= 1.1 on >= 90% of filler turns in MUNDANE conversations, and on >= 90% of T3 turns in
  surprising conversations (the carry has decayed by then). T2 after a surprising T1 is the carry the claim is about and
  is reported, not bounded. Nothing else changes. Code: the harness reports both forms; the table uses the amended one.
- Smoke S1: split vs joint read differ by up to 6.3e-5 in logits (greedy replies identical), above the 1e-5 bar, so the
  registered fallback applies: every arm, OFF included, uses the split read in the harness (`--off-split auto`).
