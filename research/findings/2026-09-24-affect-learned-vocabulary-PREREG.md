---
type: preregistration
status: preregistered
date: 2026-09-24
lane: A · Affect — appraisal vocabulary (rung (a) of the tone-selection PREREG's AMENDMENT 2)
seeds: [42, 43, 44, 100, 101, 102]
mechanism: research/runners/affect_learned_vocabulary.py behind BRAIN_AFFECT_LEARNED_VOCAB=1 (default-OFF). Word ->
  valence synapses onto a spiking V+/V- opponent pair, learned by a local instar rule from a heard text stream with the
  innate WARRINER seeds as the unconditioned stimulus; read by presenting the word alone to the same pair.
runner: research/runners/_affect_learned_vocabulary_derisk.py
verdict: PREREGISTERED; no evaluation seed has run. The seed-7 DEV operating point passes the calibration rule
  narrowly (see the DEV section), after three logged amendments of the DEV grid, so the
  evaluation is expected to be marginal on G1 and G3.
---

# Learned affect vocabulary: held-out negative words read negative through Hebbian word -> valence synapses (preregistration)

## Why

The content-locked tone-selection probe (`research/findings/2026-09-24-affect-tone-selection-content-locked-PREREG.md`,
AMENDMENT 2) found that the brain's appraisal hears only the 180-word WARRINER list. "sadness", "sorrow",
"melancholy", "loss" and "alone" read 0.0. The organ cannot perceive most negative wording, in the held mood or in a
candidate reply. This builds the missing perception as learned synapses, not as a longer host list.

## What the record already says (read before building)

`bash tools/before_you_build.sh` surfaced five prior attempts at a LEARNED affect GATE, all BOUNDARY or PARTIAL:
2026-08-12 D1 (a full learned gate coloured "what does the cat eat"); 2026-09-05 register-confound BOUNDARY (four
statistics of the TinyStories co-occurrence graph, best 29.4% recall at FP=0 on the 62 weak WARRINER words);
2026-09-05 experienced-opponent BOUNDARY; 2026-09-05 embodied-US BOUNDARY; 2026-09-17 grounded-stream PARTIAL. All
of them tried to REPLACE the norm gate on the WARRINER words at zero false positives, on TinyStories. This build
differs in three ways, each chosen against a named failure:
1. Scope. The norm gate still governs every WARRINER word. The learned vocabulary is consulted only for words the
   norm list lacks, so "what does the cat eat" is handled exactly as before.
2. Register. The heard stream is adult expository text (fineweb-edu), and the learning signal is measured against a
   SLIDING threshold that tracks the post-synaptic pool's recent increment, which subtracts the passage register.
3. Stability and balance. The learning signal is the US-evoked increment of a pool's rate over its CS-alone rate, so
   a word cannot potentiate itself through its own learned drive; a per-pool running RMS (synaptic scaling) puts the
   quiet V- pool and the busy V+ pool on the same footing.

## The method (the module docstring has the full rule)

Per brain seed, SEED_FRAC = 0.8 of the strong WARRINER words (|v-5| >= 2) are innate US afferents; the other 20% are
held out. The heard stream is the first 1e9 characters of `data/corpus/fineweb_edu.txt` (~160M tokens), cut into
13-token presentations. Each presentation is two 12-step trials from rest: the heard content words alone, then the
same words plus the innate US afferents (short-term depressed, U_DEP = 0.5, TAU_REC = 100 presentations). The
learning signal is the US-evoked increment, minus a sliding threshold (TAU_THETA = 80), in per-pool RMS units
(TAU_SCALE = 2000); the instar rate is max(1/(n + 50), 1e-4); training gain G = 500. A presentation with no innate US
has two identical deterministic trials (increment exactly 0, asserted in a test) and is not simulated. Replica 0 has
the true seed valences; replicas 1..8 have them permuted across the innate words; all nine hear the same stream.

The read goes through the PRODUCTION reader (`load_reader`, the path `appraise_text` uses): a fresh R = 1 circuit,
the learned synapses at READ_GAIN = 8 x their training weight, the word presented alone for 30 steps. A word reads
0 unless a pool reaches MIN_RATE = 0.002; valence = clip((r+ - r-) / 0.1364, -1, 1); |valence| < V_MIN = 0.4 reads 0. <!--derived-->

Brain-based boundary (declared). Brain: the learned synapses, the spiking opponent competition, the pool rates. Host:
the heard corpus and its chunking (environment); the WARRINER seed valences (the innate US, a declared scaffold); the
trial resets; the runner-side application of the local learning rule from spike rates; the slot execution of the
lexical layer; READ_GAIN (a stand-in for an attentional gain); the read of the two rates, R_REF and V_MIN (read-out).
The learning is "innate-US-driven", not "self-organized". appraise_text's averaging of word valences is the existing
host appraisal, unchanged.

## DEV calibration (seed 7, not an evaluation seed) — everything seen before this was written

All DEV reads use the even crc32 half of the independent lexicon, the seed-7 held-out seed words and the 20 FACT_DEV
sentences. The odd half and FACT_EVAL were never read. The rule is in
`research/runners/_affect_learned_vocabulary_calibrate.py` (committed before the dev weights were read, then amended three times; each amendment is logged in
its docstring with what had been seen). Artifacts: `research/findings/raw/_affect_learned_vocab/dev_s7/`.

- Host design diagnostics (not the mechanism): base-rate-corrected seed co-occurrence separated held-out negatives
  from neutral words poorly on TinyStories + wiki (the register confound again) and in adult wiki text alone
  (recall <= 0.18). On fineweb-edu it reached recall 0.48-0.50 at a 7-9% false-flag rate on fact words. A
  frame-feature (paradigmatic) classifier learned part of speech, not valence.
- A first spiking trainer on a continuous stream failed. A seed-7 instrument probe then measured the reason: the
  previous chunk's US response leaked into the next chunk's CS trial, and the increment carried the chunk's seed
  valence at r = 0.38. From rest it carries it at r = 0.95 / 0.96. Two full dev runs of that trainer were also lost
  unread to a full /tmp tmpfs.
- Fixed trainer, three variants at G = 500 (A: no depression; B: U_DEP 0.5; C: U_DEP 0.5, N0 300): best dev-negative
  recall 0.126 (variant B, `research/findings/raw/_affect_learned_vocab/dev_s7/calibration.json`). <!--derived--> Re-read over a G grid (amendment 1): recall 0.42-0.55 at G >= 2000, but 3-6 of 20 FACT_DEV
  sentences read above 0.25 through frequent topical words ('does', 'planet', 'solar', 'students'). Valence scale
  anchored to the held-out seeds (amendment 2): still no admissible cell. Strong-affect margin V_MIN (amendment 3):
  the chosen cell is variant B, G_read 4000, MIN_RATE 0.002, V_MIN 0.4. DEV: negative recall 0.295, wrong-sign 0.084, <!--derived-->
  positive recall 0.200, contrast D 0.241, FACT_DEV within 0.95 (19 of 20), held-out seeds decided 8 with sign <!--derived-->
  accuracy 0.50.
- A fourth variant trained on the whole 4.3 GB file read WORSE (dev-negative recall 0.04 at G 4000; held-out seed sign
  accuracy 0.25): more heard text did not help this rule, and why is not understood.
- The confirmation run TRAINED variant B at G = 4000 and read dev-negative recall 0.03: strong synapses saturate the
  pools in both trials and erase the increment. So the mechanism trains at G = 500 and reads at READ_GAIN = 8; the
  production reader reproduces the chosen cell exactly (`research/findings/raw/_affect_learned_vocab/dev_s7/confirmation.json`, passes the rule).
- Post-hoc, on the stored seed-7 tone-selection probe texts (`research/findings/raw/_affect_learned_vocab/dev_s7/tone_probe_rescore_s7.json`): with this
  vocabulary the share of prompts with an admissible candidate appraised <= -0.30 rises from 0.00-0.13 to 0.27-0.40
  per generator variant. With the new lock term, no variant reaches the tone-selection probe's 0.5 coverage floor.

What this predicts: the chosen cell was picked from ~240 cells per variant on 95 DEV negatives and 20 DEV sentences,
so the evaluation half will probably read lower; the held-out seed sign accuracy of 0.50 says the sign of strongly
affective words is not reliably learned. G1 and G3 are expected to be marginal. This is pre-registered as it stands;
a NO-GO is a verdict on this rule, and the named next rung is below.

## The gate, and how each part can fail

Evaluation items (fixed now): EVAL negatives = the independent lexicon's negative words whose crc32 is odd (the DEV
half, crc32 even, was used on seed 7); EVAL positives likewise; FACT_EVAL = the 40 neutral fact sentences in the
runner (none contains a strong WARRINER word; asserted as a precondition). A word never heard >= MIN_HEARD times has
no synapses and reads 0; it stays in the denominator.

| gate | passes iff (per seed; seed count below) | FAILS if (a realistic outcome) |
|---|---|---|
| G1 learning | share of EVAL negatives reading negative >= 0.20 AND share reading positive <= 0.15; on >= 5/6 seeds | the DEV cell was a lucky pick among many and the EVAL half reads below 0.20; or frequent negatives such as "loss" co-occur with positive seeds ("win/loss") and read positive |
| G2 shuffled-label control | true contrast D = P(neg \| EVAL neg) - P(neg \| EVAL pos) >= 0.15, AND D exceeds every one of the 8 shuffled replicas' D, AND the shuffled replicas' mean D <= 0.05; on >= 5/6 seeds | a negative word reads negative because of something other than the seed labels (frequency, part of speech, register), so permuted labels also produce D > 0; or a permutation happens to keep the sign structure |
| G3 neutral facts | >= 95% of the 40 FACT_EVAL sentences appraise with \|valence\| <= 0.25 (the affect ladder's dead zone) with the flag on; on 6/6 seeds | topical words (country names, history and science terms) learn a strong valence, as 'does' / 'planet' / 'solar' / 'students' did on DEV before the strong-affect margin |
| G4 lesion contrast | with BRAIN_AFFECT_LEARNED_VOCAB_LESION=1, 0 EVAL-negative probe sentences ("It was <word>.") read negative, while intact >= 0.20 read negative; on 6/6 seeds | the lesion does not reach the path appraise_text uses (a second copy of the synapses, a cache), so the lesioned arm still reads negative; or the intact arm does not reach 0.20 through appraise_text |

GO iff G1, G2, G3 and G4 pass and every precondition holds. A failed precondition makes the verdict UNDEFINED,
never a negative.

Preconditions: all 6 seeds present; the production reader loaded its weights on every seed; every FACT_EVAL sentence
reads exactly 0 with the flag off; the heard vocabulary hash is identical on every seed and replica block (same
stream); >= 20 EVAL negatives heard; INTEGRITY (passes by construction if the code is right, not evidence): under the
lesion, appraise_text's (valence, arousal, n_hits, words) equal the flag-off output on every probe sentence, and
every EVAL negative reads exactly 0.0 through the reader. Byte-identical-off is asserted in data by
`tests/test_affect_learned_vocabulary.py::test_off_byte_identical_to_pinned` against pinned commit 1d5766620.

Reported, not gated: the named words (sadness, saddened, unhappy, sorrow, melancholy, loss, loneliness, decay,
alone); the held-out seed words' sign accuracy; the 62 weak WARRINER words read through the learned path (the
register-confound set of the prior BOUNDARY findings; production never consults the learned path for them); EVAL
positives' recall; FACT_DEV; the attribution of the negative-probe share to the learned synapses (tools.lab).

Honest limits written before any evaluation data:
- G1's bar is a partial-coverage bar, not "every negative word". The organ will still miss most negative words.
- The EVAL lexicon is also the tone ruler of the tone-selection gate. Nothing is fit to it here, but a GO would make
  the organ's input and that ruler agree more; the tone-selection gate must keep scoring with the ruler's own words.
- Seeds vary the circuit heterogeneity, the innate/held split and the permutations. The heard stream is the same on
  every seed, so the six seeds are not six independent corpora.
- The reader circuit is a fresh R = 1 circuit, not the training circuit; the learned synapses carry over by value.

## Staging

- Seed 42 locally, under `tools/memcap.sh`:
  `SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._affect_learned_vocabulary_derisk --part all --seed 42 --replicas 0-8`
- Seeds 43, 44, 100, 101, 102 on the pool at this revision, one line per seed with `--part all --replicas 0-8
  --corpus-path ~/derisk-pool/corpus_extra/fineweb_edu.txt` (the same file, rsynced; the vocabulary-hash
  precondition checks it is the same stream). Each writes `research/findings/raw/_affect_learned_vocab/run1/`.
- Score: `.venv/bin/python -m research.runners._affect_learned_vocabulary_derisk --score research/findings/raw/_affect_learned_vocab/run1`

Next rung if this is a NO-GO (THE LAW): the sign of strong held-out seeds at 0.50 and the loss of signal with more
text say the teaching signal (co-occurrence with 80 innate words in 6% of chunks) is too sparse. The rung is
second-order conditioning, where learned words above V_MIN become teachers themselves, raising the teaching density.
A grounded US (interoceptive / prosodic) stays the longer-term surpass, as the 2026-09-05 boundary findings named.

Honesty boundary: this measures a functional perception of word valence. Nothing here claims felt emotion.

## AMENDMENT LOG

(none)
