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
