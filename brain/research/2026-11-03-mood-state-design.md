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
unmodulated cortex) reported, not gated. Appraisal sign accuracy on held-out events reported per seed.
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

Six seeds at 0.1B: `brain/experiments/results/2026-11-03-mood-0.1b.md`. Pre-registered criteria met 6/6; positive
appraisal weak on 2/6 seeds; not yet wired into the live brain.
