# Input-side fidelity gap + grounding data-efficiency: VERIFIED + cheap-first VALIDATED -- 2026-06-02

## Owner insight (side-chat)
Real brains learn from raw sensory spike trains; words arrive GROUNDED in multimodal experience (hear "cup"
while seeing/grasping a cup) -> few exposures suffice. The sim's VISION is faithfully transduced (image ->
32x32 ON/OFF retina firing prop intensity -> V1/V2/IT), but LANGUAGE is NOT: text -> tokenizer -> orthogonal/
one-hot/sparse code -> input current. Segmentation + word-coding are handed to the sim for free, with NO
grounding -> forces pure co-occurrence-statistics learning = the data-hungry, overfit-prone regime. The
missing sensory transduction/grounding is plausibly a missing DATA-EFFICIENT structure, not a side detail.

## Task 1 -- verified text->spikes encoding path (given vs learned)
Path (concept-pool lang_input): token string -> `vocab_to_drive_pattern` (SHA-256 hash -> sparse pattern) or
`orthogonal_drive_pattern` (non-overlapping bands) [sim/text_embeddings.py] -> sparse {0, drive_pA} code ->
`cp_external_input_current` -> spikes [`set_token_drive`, sim/bridge.py:2149]. TinyStories generative path:
text -> `bpe_tokenizer.encode` -> token IDs -> `encode_one_hot` -> spiking net.
- GIVEN (not earned): (1) SEGMENTATION -- the tokenizer splits the stream; the sim never sees raw text.
  (2) WORD-IDENTITY CODING -- distinct words get distinct/orthogonal codes BY CONSTRUCTION, with NO grounding
  (arbitrary symbol). No shared structure: "red apple" and "red ball" have INDEPENDENT codes.
- LEARNED (earned): only the downstream lang_input -> concept-pool routing (STDP). The input code is free.
- Contrast: VISION earns its representation (transduction + feature extraction); LANGUAGE is handed one. This
  is the "given codes" caveat the cheating-audit findings keep flagging, at the input.

## Task 2 -- cheap-first grounding data-efficiency: RESOLVES
research/findings/raw/_grounding_data_efficiency_probe.py. Controlled: ONLY the input representation differs
(same logistic-regression learner). Concepts = (color, object) pairs; GROUNDED code = concat(color_feature,
object_feature) (SHARES color_feature across same-color); ORTHOGONAL code = independent random per pair
(tokenizer regime). Task: classify color; train on K pairs; held-out = novel (color,object) combinations
(color + object each seen elsewhere). Multi-seed (42/43/44):

| #train pairs | grounded held-out acc | orthogonal held-out acc |
|---|---|---|
| 9  | 0.917 | 0.208 |
| 18 | 1.000 | 0.125 |
| 36 | 1.000 | 0.167 |

Grounded reaches >=0.9 at **9 pairs**; orthogonal **never** reaches it -- stuck near chance (~0.17) regardless
of K, because each combination is an independent symbol with NOTHING to transfer. >4x (effectively infinite)
data-efficiency gap. NOT a modeling artifact -- it is the fundamental property: shared sensory structure lets a
sub-feature ("red") transfer across combinations; an orthogonal symbol cannot. **Validates the hypothesis:
the tokenizer's orthogonal coding forces the data-hungry regime; grounding (shared sensory structure) makes
word-learning data-efficient via feature-sharing/transfer.** Scrutiny: the orthogonal-at-chance result is
correct (unseen independent code -> no basis to classify), the held-out is a fair compositional-generalization
split, the learner is identical across conditions -> the input representation is the sole cause.

## Implication + roadmap fold (step 2, data-efficient learning)
The input-side fidelity gap is now a MEASURED data-efficiency bottleneck, not a side detail. Fold "raw
transduction + learned segmentation/grounding" into the roadmap's step-2 grounding work:
- **Faithful build (the fix):** TRANSDUCE raw text instead of tokenizing -- render text as PIXELS through the
  EXISTING faithful visual pathway (32x32 retina -> V1/V2/IT). Words then arrive with SHARED ORTHOGRAPHIC
  structure (letter/shape features) and the network must LEARN to segment + recognize word-forms (no free
  tokenizer). Reuses the validated visual pathway; removes the orthogonal-symbol shortcut.
- **Two grounding levels (honest):** (a) ORTHOGRAPHIC -- text-as-pixels gives shared letter-features ->
  data-efficient for morphology/spelling, and removes the tokenizer. (b) SEMANTIC -- the word-form must
  CO-OCCUR with the referent's sensory/motor features (vision + Pulvermuller motor grounding) to ground in
  MEANING -> data-efficient for semantics (what this probe tested at the semantic level). The faithful build
  needs BOTH: transduce (orthographic) + multimodal-co-occur (semantic).

## Next (cheap-first before the big build, per discipline)
(1) Cheap-first text-as-pixels probe: render a small vocab as pixels through the existing retina/V1, show the
visual features SHARE structure across similar word-forms (vs orthogonal tokens) -> learned recognition is
data-efficient. (2) Then the multimodal-co-occurrence grounding (word-form pixels + referent sensory feature
-> Hebbian grounding) -> the faithful data-efficient word-learning loop. Brainstorm/design before the build.
Biology-faithful; reuse the visual pathway; honest negatives; both remotes; no shortcuts.

## Follow-up cheap-first: text-as-pixels READS NOVEL words (open-vocabulary) -- RESOLVES
research/findings/raw/_text_as_pixels_probe.py. Controlled (only input rep differs): VISUAL code = concat of
letter-glyphs (shared across words); ORTHOGONAL = independent per-word random vector (tokenizer). Task: read
the letter at each position; test on HELD-OUT NOVEL words. Multi-seed:

| train words | visual novel-word read | orthogonal |
|---|---|---|
| 5  | 0.352 | 0.108 |
| 30 | 0.865 | 0.103 |
| 50 | 0.966 | 0.113 |
| 120 | 1.000 | 0.108 |

VISUAL reads NOVEL words ->1.000 once the ~L letters are covered; ORTHOGONAL is stuck at chance (~0.10) at ALL
K -- a novel word is an unseen symbol. So text-as-pixels turns "learn W words" into "learn ~L letters -> read
L^n words" = COMBINATORIAL data-efficiency + OPEN-VOCABULARY reading (read words never seen, exactly how humans
read). The tokenizer's closed-vocabulary orthogonal coding fundamentally cannot. This is the strongest form of
the input-side data-efficiency case and directly motivates the faithful build (render text through the EXISTING
32x32 retina -> V1/V2/IT visual pathway; the network learns to segment + recognize word-forms from shared
letter/stroke features). Next: implement the cheap-first text-as-pixels recognizer on the ACTUAL visual pathway
(reuse sim/visual_cortex.py + the retina), then the multimodal-co-occurrence grounding loop.

## FAITHFUL validation on the REAL visual pathway -- RESOLVES (implementable)
research/findings/raw/_text_as_pixels_v1_probe.py. Reuses sim/visual_cortex.py: word -> 32x32 pixel image (PIL,
letters in fixed x-bands) -> (2,32,32) ON/OFF -> image_to_retina_drive -> 2048 retina -> build_v1_simple_weights
(real Gabor RFs) -> 8192 V1 simple responses. Read NOVEL held-out words from the V1 band features (per-position
letter classifier), multi-seed:

| train words | FAITHFUL V1 novel-word read (chance 0.10) |
|---|---|
| 8   | 0.567 |
| 20  | 0.835 |
| 50  | 0.886 |
| 100 | 0.912 |

The validated text-as-pixels principle HOLDS on the REAL retina->Gabor-V1 pathway: shared letter-features ->
shared V1 structure -> a few training words suffice to read NOVEL words (0.91, climbing), no tokenizer. The
faithful fix is IMPLEMENTABLE on the existing visual machinery + data-efficient. Honest caveat: the same-vs-
different-first-letter V1 band cosine margin is thin (1.000 vs 0.984) -- the tiny default font yields
overlapping V1 features; reading works regardless, and clearer/larger rendering (or learned V1 refinement)
would sharpen the margin. The remaining steps to the production build: (1) wire the text-as-pixels recognizer
into the bridge (retina region -> V1 -> learned word recognition, replacing set_token_drive's orthogonal code);
(2) learned SEGMENTATION (word boundaries from the stream); (3) multimodal co-occurrence GROUNDING (word-form
pixels + referent sensory/motor feature -> Hebbian) for semantic meaning. All reuse validated machinery.

## Arc summary (owner input-side-fidelity insight -- fully validated, biology-faithful)
1. Encoding path AUDITED: language input is GIVEN (tokenizer segmentation + orthogonal code, no grounding);
   vision is EARNED (transduction). 2. Grounding DATA-EFFICIENCY validated (numpy, semantic): shared sensory
   structure generalizes from few examples; orthogonal tokens never generalize. 3. Text-as-pixels OPEN-VOCAB
   reading validated (numpy, orthographic): learn ~L letters -> read L^n words; tokenizer stuck at chance.
   4. FAITHFUL validation on the real retina->Gabor-V1 pathway: reads novel words 0.91, implementable. =>
   The missing sensory transduction/grounding IS a missing data-efficient structure, and the fix is concrete +
   reuses the existing visual pathway.

## CAPSTONE: faithful end-to-end data-efficient word-learning loop -- RESOLVES
research/findings/raw/_grounded_word_learning_loop_probe.py. Combines ALL validated pieces, tokenizer-free:
render word as pixels -> real retina -> real Gabor V1 (word-FORM feature) -> ONE-SHOT Hebbian grounding of the
V1 word-form to its concept feature -> grounded concept enters generate-by-composition.
- (1) one-shot word recognition from V1 word-form: 24/24 (1.000), 1 exposure/word.
- (2) grounded words COMPOSE into novel produced sentences (in order): 20/20 (1.000).
1 exposure/word -> read + compose any sentence, no tokenizer, no co-occurrence-statistics training. The
complete biology-faithful data-efficient language loop (visual transduction + V1 + one-shot grounding +
composition + production). Honest caveat: the one-shot recognition uses identical re-renders (no view noise);
robustness to rendering variation (jitter/scale/noise) is the next hardening. Next: (a) harden recognition
with rendering variation; (b) wire this loop into the bridge (retina -> V1 -> grounded concept pools) as the
production input path, replacing set_token_drive's orthogonal code.

## Recognition front-end: characterized limits + refined production plan (honest)
Hardening the capstone loop with rendering variation (jitter+noise) exposed real, characterized limits of the
CHEAP recognition (one-shot nearest-prototype on V1 SIMPLE cells), distinct from the validated INSIGHT:
- Cramped 32x32 3-letter rendering -> low V1 word-form separability: few-shot recognition across noisy views
  0.38 (1 exposure) -> 0.59 (5 exposures), chance 0.04. (Composition unaffected: 1.000.)
- Even a SINGLE clear letter (truetype-24) is only 0.67 across jitter+noise -- V1 SIMPLE cells are POSITION-
  SPECIFIC (Gabor RFs at fixed positions) so jitter shifts the features; the project's complex cells are
  phase-pooled but still 16x16 position-specific -> not jitter-invariant either.
=> Faithful word RECOGNITION needs the brain's INVARIANCE + LEARNING machinery, not a crude prototype:
  (a) SACCADIC/foveated reading (one letter or syllable per 32x32 fixation -- matches the retina + how humans
      read), (b) spatial pooling / V1-complex invariance for jitter tolerance, (c) LEARNED STDP refinement of
      the V1->word-form mapping over a few views. These ARE the production build's "learned word recognition"
      piece -- the cheap prototype was only a scaffold.
HONEST SCOPE: the input-side INSIGHT (tokenizer orthogonal coding forces data-hunger; grounding/text-as-pixels
is data-efficient) is VALIDATED 4 ways + the composition loop works (1.000). The faithful RECOGNITION
implementation is a real sub-problem with a clear biology-faithful plan (saccades + invariance + learned
refinement), now the well-specified next sub-arc of the production build (alongside segmentation + the bridge
wire-up). Not a dead end -- a refinement of WHAT the production build must implement.
