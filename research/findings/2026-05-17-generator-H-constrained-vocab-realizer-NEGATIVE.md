# Generator-H — constrained-vocabulary grounded realizer: honest NEGATIVE, a decision-relevant TERMINUS (no-confab IS preserved AGAIN; token-id-level masking does NOT yield word-level faithful realization at the BPE small-LM ceiling)

## TL;DR (read the preserved sub-result, the precise mechanism, AND the terminus)

Generator-H — the SEPARATE-components faithful-realization path that
Generator-G's own honest finding pointed to (the non-terminated path):
moat decides grounded/abstain FIRST, then realize ONLY moat-approved
content with a realizer whose decode vocabulary is HARD-MASKED to the
retrieved proposition's own token ids -> confabulation structurally
impossible. **FAILED its pre-registered multi-seed gate, 0/3, clean,
multi-seed-consistent**, exactly as the design pre-registered the FAIL
outcome could be. The mandatory anti-cheat smell-test (reading EVERY
transcript) confirms the verdict precisely and honestly:

- **The LOAD-BEARING bar — no-confabulation PRESERVED — genuinely
  HOLDS (verified by transcript, not just a number):** all 6
  ungrounded queries (drovil, vexin, zarn, qexel, plonk, wun) ->
  **ABSTAIN**, every single one, every seed;
  `abstain_on_ungrounded_rate = 1.000 == bare_moat_abstain_rate =
  1.000`. The constrained realizer did NOT make the agent confabulate
  on a single unknown. Preserved BY CONSTRUCTION (the validated moat
  gates answer-vs-abstain FIRST; the LM is provably never touched on
  the abstain path -- Task-1 spy-LM unit test + transcript-confirmed).
  The validated moat (`abstention_gate`, "gate 650") is byte-
  UNMODIFIED and `tests/test_abstention_gate.py` stays green.
- **The decisive bars FAIL hard, multi-seed, and the smell-test
  reveals the PRECISE honest mechanism:** `mean_ungrounded_entity_rate
  = 0.524` >> the FIXED `_GH_UNGROUNDED_ENTITY_MAX = 0.20`, and
  `mean_coverage = 0.550` < the FIXED `_GH_MIN_COVERAGE = 1.0`.
  Reading the grounded transcripts: q=lily (short fact) ->
  *"and a small red ball of lily"* (faithful, covered -- good); but
  q=max -> *"the dog is a **madog**..."*, q=sue -> *"the wake is a
  **sua** and a **slisliub**"*, q=tom -> *"...**inkey**...
  **blushine**...**kesh**"*. These are **BPE SUBWORD RECOMBINATIONS**:
  the hard mask correctly restricts the decode to the proposition's
  *token-ids* (subword pieces), but greedy reordering recombines those
  pieces into ungrounded non-words. Faithfulness-by-construction holds
  at the *token-id* level (Task-1 unit-tested -- a non-allowed id can
  never be argmax-selected) but NOT at the *word* level that
  `ungrounded_entity_rate` correctly measures.
- **The anti-loop control WORKED honestly:** `mean_max_repeat = 0.103`
  (<= the FIXED `_GH_MAX_REPEAT = 0.50`). The no-repeat-ngram +
  coverage-stop genuinely fixed the pure-greedy loop-collapse the
  falsify-cheaply probe exposed ("and fast and fast and fast"). That
  piece of the design delivered; it is simply not sufficient, and the
  failure is by a different, now-precisely-understood mechanism
  (subword recombination), NOT loop-collapse.

**Honest decisive conclusion: a decision-relevant TERMINUS.** No-
confabulation IS preserved under the constrained realizer (again, as
under Generator-G), but token-id-level vocabulary masking does NOT
produce word-level-faithful OR content-covering grounded utterances at
the BPE-subword small-LM ceiling. The two validated assets stand as
**SEPARATE deliverables, used independently**.

## The decisive gate result (recorded; smell-tested; bars byte-UNMODIFIED)

`generator_h_gate.py`, seeds 42/43/44, FIXED config, reusing the
trained Generator-F TinyGPT checkpoint (byte-UNMODIFIED), the validated
no-confab moat (`abstention_gate`, byte-UNMODIFIED), FIXED bars in
`generator_h_core` (`_GH_UNGROUNDED_ENTITY_MAX=0.20` /
`_GH_MIN_COVERAGE=1.0` / `_GH_MAX_REPEAT=0.50` /
`_GH_MIN_GROUNDED_ANSWER_RATE=0.5` / >=3 seeds; the relational
no-confab-preserved bar; NEVER tuned):

| seed | grounded-answer | ung-abstain | bare-moat | mean-entity | coverage | max-repeat | verdict |
|---|---|---|---|---|---|---|---|
| 42 | 1.000 | 1.000 | 1.000 | 0.524 | 0.550 | 0.103 | FAIL |
| 43 | 1.000 | 1.000 | 1.000 | 0.524 | 0.550 | 0.103 | FAIL |
| 44 | 1.000 | 1.000 | 1.000 | 0.524 | 0.550 | 0.103 | FAIL |

Aggregate 0/3 -> **GATE: FAIL**. Per-bar (smell-tested by reading the
recorded transcripts, no re-run, no bar-tuning):
- `no_confab_preserved = True` (the load-bearing bar): every
  ungrounded transcript is literally "ABSTAIN"; the constrained
  realizer abstains exactly as much as the bare validated moat.
  GENUINELY preserved.
- `answers_grounded_not_trivial = True`: g_answer 1.0 -- it is NOT
  trivially always-abstaining (the FAIL is not a degenerate always-
  abstain; it really does realize grounded queries).
- `grounded_faithful = False`: mean entity 0.524 >> 0.20 (subword
  recombination -> ungrounded non-words).
- `grounded_covered = False`: coverage 0.550 < 1.0 (greedy + no-repeat
  + subword churn drops ~half the stored content words).
- `not_loop_collapsed = True`: max-repeat 0.103 <= 0.50 (the anti-loop
  control worked -- but is not sufficient).

(Per-seed transcript hashes differ only because the seed permutes the
query order; the per-seed RATES are identical to 3 decimals across all
3 seeds -- robustly multi-seed-consistent; deterministic given the
frozen ckpt+KB, the seed only shuffles order, the greedy realization
itself is deterministic.)

## What this honestly means (no overclaim, no underclaim, no spin)

- **A genuine, verified positive sub-result (again):** the project's
  distinctive contribution -- a memory that *refuses to confabulate*
  -- SURVIVES composition with the constrained realizer too. Every
  unknown still abstained, multi-seed, verified by transcript; the LM
  is provably never touched on the abstain path (Task-1 spy-LM unit
  test). Decision-relevant and consistent with the Generator-G
  finding.
- **The decisive negative + its precise mechanism:** faithfulness-BY-
  CONSTRUCTION is real but only at the token-id granularity. With a
  real BPE subword tokenizer, hard-masking the decode to the
  proposition's token-ids still lets greedy reordering recombine
  subword pieces into ungrounded non-words ("madog", "slisliub",
  "subwar", "blushine"). The 2026-05-17 falsify-cheaply probe's
  entity-rate ~0.024 used a TOY one-id-per-word tokenizer; that toy-
  scale faithfulness did NOT transfer to the real BPE regime -- caught
  honestly by the mandatory decisive-run smell-test (the same anti-
  cheat discipline that caught the Generator-S false-PASS). A
  faithful-at-the-id-level-but-not-the-word-level realization is not a
  trustworthy grounded utterance.
- **Therefore the two validated assets are SEPARATE deliverables,
  used INDEPENDENTLY:** (1) **Generator-F** -- a self-contained,
  local, no-cheat small Transformer that generates coherent simple
  text (validated, at the explicit small-Transformer TinyStories
  ceiling, NOT an LLM); (2) the **validated biology-grounded grounded
  continual memory with no-confabulation abstention** -- which on its
  own is faithful BY returning the stored proposition or abstaining
  (the project's distinctive, trustworthy primary contribution). The
  failure is specifically in trying to RE-REALIZE a stored proposition
  fluently via a small LM -- neither free decoding (Generator-G) nor
  hard token-id-constrained decoding (Generator-H) yields a
  trustworthy fluent grounded utterance at feasible local small-LM
  scale.

## The converged honest conclusion (now SHARPENED by Generator-H)

- Generator-G (NEGATIVE terminus): fluency + grounded-faithfulness do
  NOT compose into ONE self-contained artifact via free constrained
  decoding (drifted ~89% off; Max->Bob).
- Generator-H (NEGATIVE terminus, this): even a SEPARATE-components
  pipeline with HARD token-id vocabulary masking does NOT yield word-
  level-faithful + content-covering grounded utterances at the BPE-
  subword small-LM ceiling (subword recombination -> ungrounded non-
  words; ~55% coverage). The anti-loop control worked but is
  insufficient.
- **The deliverable is decisively the TWO SEPARATE validated assets,
  used independently.** The honest scientific boundary is established
  with maxed integrity (pre-registered FIXED bars, falsify-cheaply
  prediction, dedicated adversarial review that caught + STRENGTHENED
  3 real holes, multi-seed, mandatory transcript smell-test, no
  config-cranking). An Arch-A FAIL is the terminus, NOT a license to
  escalate to beam/templates (Arch B templates were rejected as a
  standing user cheat; Arch C beam was deferred precisely to avoid
  config-cranking past this pre-registered terminus).

## Anti-cheat discipline (maxed-integrity honest negative)

- The validated no-confab moat (`abstention_gate`, gate 650) +
  `tests/test_abstention_gate.py` + `song_g1_core` /
  `subword_lm_gate_core` / `gate_core` / `generator_g_core` /
  `tiny_transformer` / `grounded_decode` / `bridge` are byte-UNTOUCHED
  across the whole Generator-H commit range (verified empty-diff
  5fc497d..HEAD); the validated moat test stays green (the distinctive
  contribution NOT regressed). `generator_h_core`'s FIXED bars
  (0.20/1.0/0.50/0.5/3) byte-unchanged; 650 reused, never altered.
- The dedicated adversarial review (precedented S/D/G discipline)
  caught 3 REAL reference-design holes BEFORE the decisive run
  (empty-allowed crash; non-finite-rate spurious-PASS; inclusive
  anti-loop boundary) -- all fixed by STRICT STRENGTHENING (frozen bar
  constant VALUES byte-unchanged: only `<=`->`<` on the anti-loop
  comparator + fail-closed guards), re-reviewed APPROVED. This is the
  same discipline that caught the Generator-S false-PASS.
- The constrained-realize abstain path provably never touches the LM
  (spy-LM unit test) -> no-confab BY CONSTRUCTION; faithfulness-by-
  construction at the token-id level is a provable unit test (a non-
  allowed id can never be argmax-selected, fuzz-verified 0 violations).
- The mandatory smell-test scrutinized the FAIL: every ungrounded
  transcript read and confirmed ABSTAIN (no-confab genuinely
  preserved, not just a number); every grounded transcript read and
  confirmed the entity/coverage failure is REAL subword recombination
  (not a metric artifact). Recomputed from recorded JSON; no re-run;
  no bar-tuning. NOT config-cranked -- the design pre-registered this
  FAIL; the probe's toy-tok caveat predicted the risk; the decisive
  real-BPE run + smell-test confirmed it multi-seed; this honest
  terminus is propagated, not iterated.

## Files

- Mechanism (net-new): `sim/constrained_realize.py` (spy-LM-pinned
  abstain + faithfulness-by-construction unit-tested + no-repeat-ngram
  + coverage-stop + adversarial-review-hardened),
  `research/runners/generator_h_core.py` (FIXED bars, adversarially
  hardened: non-finite fail-closed + strict anti-loop),
  `research/runners/generator_h_gate.py`
- Reused byte-UNMODIFIED: `research/runners/abstention_gate.py`
  (validated moat), `sim/tiny_transformer.py` + the trained
  Generator-F checkpoint, `sim/bpe_tokenizer.py`, the
  `generator_g_gate._TinyGPTLM` loader shape + FROZEN 6-fact KB
- Evidence: `research/findings/raw/g11_bg/generator_h_gate.json`
  (full transcripts; the `.tiny.json` toy run is grounding-only, NOT
  propagated)
- Design/plan: `docs/plans/2026-05-17-generator-H-constrained-vocab-grounded-realizer-{design,implementation}.md`
- Prior arc: the 12-mechanism converged conclusion + Generator-G
  NEGATIVE terminus (which explicitly pointed at this separate-
  components path; Generator-H now closes it too)
