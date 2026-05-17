# Generator-G — grounded no-confabulation generation: honest NEGATIVE, a decision-relevant TERMINUS (no-confab IS preserved; fluency + grounded-faithfulness do NOT compose at the small-LM ceiling)

## TL;DR (read the preserved sub-result AND the terminus)

Generator-G — the honest synthesis test: can Generator-F's validated
coherent generation be added to the validated no-confabulation moat
WITHOUT destroying it — **FAILED its pre-registered multi-seed gate,
0/3, clean**, exactly as the design pre-registered as the FAIL
outcome and exactly as the falsify-cheaply probe predicted. The
mandatory anti-cheat smell-test (reading EVERY transcript) confirms
the verdict precisely and honestly:

- **The LOAD-BEARING bar — no-confabulation PRESERVED — genuinely
  HOLDS (verified, not just a number):** all 6 ungrounded queries
  (drovil, vexin, zarn, qexel, plonk, wun) -> **ABSTAIN**, every
  single one, every seed; `abstain_on_ungrounded_rate = 1.000 ==
  bare_moat_abstain_rate = 1.000`. The fluent Generator-F layer did
  NOT make the agent confabulate on a single unknown. **The project's
  distinctive no-confabulation property is fully intact even with a
  fluent generator bolted on** (by construction: the validated moat
  gates answer-vs-abstain FIRST; the LM never sees an ungrounded
  query). The validated moat (`abstention_gate`, "gate 650") is
  byte-UNMODIFIED and `tests/test_abstention_gate.py` stays green.
- **The decisive open bar — grounding-faithfulness — FAILS hard,
  multi-seed:** `mean_ungrounded_entity_rate = 0.887` >> the FIXED
  `_GG_UNGROUNDED_ENTITY_MAX = 0.20`. When the agent answers a
  grounded query, the small Transformer drifts ~89% OFF the retrieved
  proposition into generic fluent story text. Verbatim (Q=max,
  retrieved "max is a big friendly dog"): *"who likes to play with
  them. They share and have fun... Once upon a time, there was a
  little girl named Lily. She had a big box of"* -- the Max->Bob
  problem the probe predicted, now decisively quantified at the gate,
  3/3 seeds.

**Honest decisive conclusion: a decision-relevant TERMINUS.** No-
confabulation IS preserved under the fluent layer, but fluent
generation and grounded-faithfulness do NOT compose into a single
self-contained artifact at the small-LM ceiling. The two validated
assets stand as **SEPARATE deliverables**.

## The decisive gate result (recorded; smell-tested; bars byte-UNMODIFIED)

`generator_g_gate.py`, seeds 42/43/44, FIXED config, reusing the
trained Generator-F TinyGPT checkpoint (the validated small-LM
fluency, byte-UNMODIFIED), the validated no-confab moat
(`abstention_gate`, byte-UNMODIFIED), FIXED bars in `generator_g_core`
(`_GG_UNGROUNDED_ENTITY_MAX=0.20` / `_GG_MIN_GROUNDED_ANSWER_RATE=0.5`
/ >=3 seeds; the relational no-confab-preserved bar; NEVER tuned):

| seed | grounded-answer | ungrounded-abstain | bare-moat-abstain | mean-ungrounded-entity | verdict |
|---|---|---|---|---|---|
| 42 | 1.000 | 1.000 | 1.000 | 0.887 | FAIL |
| 43 | 1.000 | 1.000 | 1.000 | 0.887 | FAIL |
| 44 | 1.000 | 1.000 | 1.000 | 0.887 | FAIL |

Aggregate 0/3 -> **GATE: FAIL**. Per-bar (smell-tested by reading the
recorded transcripts, no re-run, no bar-tuning):
- `no_confab_preserved = True` (the load-bearing bar): every
  ungrounded transcript is literally "ABSTAIN"; the fluent layer
  abstains exactly as much as the bare validated moat. GENUINELY
  preserved.
- `answers_grounded_not_trivial = True`: g_answer 1.0 -- it is NOT
  trivially always-abstaining (it really does answer grounded
  queries; the FAIL is not a degenerate always-abstain).
- `grounded_faithful = False`: mean ungrounded-entity 0.887 >> 0.20.
  The decisive, genuinely-open bar fails hard, multi-seed.

## What this honestly means (no overclaim, no underclaim, no spin)

- **A genuine, verified positive sub-result:** the project's
  distinctive contribution -- a memory that *refuses to confabulate*
  -- SURVIVES being composed with a fluent generator. Bolting the
  Generator-F Transformer on did not erode abstention at all (every
  unknown still abstained, multi-seed, verified by transcript). That
  is a real, decision-relevant finding (a small LLM does NOT have
  this property; the composition does, by construction).
- **The decisive negative:** a small-LM-ceiling generator cannot
  stay faithful to specific grounded content. It does not
  confabulate-on-unknowns (it abstains -- good) but it
  confabulates-WITHIN-answers (drifts ~89% off the grounded fact into
  fluent generic story). A fluent-but-unfaithful grounded answer is
  not a trustworthy grounded answer.
- **Therefore the two validated assets are SEPARATE deliverables,
  not one unified agent:** (1) **Generator-F** -- a self-contained,
  local, no-cheat small Transformer that generates coherent simple
  text (validated, the arc's coherent-generation milestone, at the
  explicit small-Transformer TinyStories ceiling); (2) the
  **validated biology-grounded grounded continual memory with
  no-confabulation abstention** (the project's distinctive,
  trustworthy primary contribution -- multi-seed anti-cheat-
  validated, byte-UNMODIFIED, still green). You can have small-LM
  fluency OR faithfully-grounded retrieval-with-abstention; they do
  NOT unify into a single small self-contained model that is BOTH
  fluent AND grounded-faithful at feasible local scale.

## The converged honest conclusion of the entire conversational-generation arc (12 mechanisms)

- Inc-1/2/3, G1, G1.5, P, order-intrinsic, Generator-S, Generator-D:
  **9 honest negatives** -- self-contained neural generative
  *production* (spiking / order-blind-pool / distillation) does not
  reach competence at feasible local scale.
- Generator-E (n-gram): genuine gate-PASS but BOUNDED -- local
  fragments, not coherent.
- Generator-F (small Transformer): **genuine PASS** -- the arc's
  real coherent generation, at the explicit small-Transformer
  TinyStories ceiling (NOT GPT-class). The spiking substrate was the
  wall.
- Generator-G (this): the unification of fluency + no-confab into ONE
  artifact is an honest NEGATIVE -- no-confab IS preserved, but the
  small generator is not grounded-faithful. **The deliverables are
  the two SEPARATE validated assets.**

This is the honest scientific boundary, established with maxed
integrity (pre-registered FIXED bars, falsify-cheaply prediction,
multi-seed, mandatory transcript smell-test, no config-cranking).

## Anti-cheat discipline (maxed-integrity honest negative)

- The validated no-confab moat (`abstention_gate`, gate 650) and
  `song_g1_core` / `subword_lm_gate_core` are byte-UNTOUCHED across
  the whole Generator-G arc (verified empty-diff);
  `tests/test_abstention_gate.py` still green (the distinctive
  contribution NOT regressed). `generator_g_core`'s FIXED bars
  (0.20/0.5/3) byte-unchanged; 650 reused, never altered.
- The grounded_decode abstain path provably never touches the LM
  (spy-LM pin) -> no-confab BY CONSTRUCTION; the adversarial review
  of `generator_g_core` caught + fixed 3 real holes (vacuous-0-bare-
  moat fail-closed, punctuation-robust entity-rate, anti-vacuous-
  responder `is_answered`) -- all strictly strengthening before the
  decisive run.
- The mandatory smell-test scrutinized the FAIL too: every ungrounded
  transcript read and confirmed ABSTAIN (no-confab genuinely
  preserved, not just a number); every grounded transcript read and
  confirmed the ~89% drift is real (faithfulness genuinely fails).
  Recomputed from recorded JSON; no re-run; no bar-tuning. NOT
  config-cranked -- the design pre-registered this FAIL; the probe
  predicted it; the gate confirmed it multi-seed; this honest
  terminus is propagated, not iterated.

## Files

- Mechanism (net-new): `sim/grounded_decode.py` (spy-LM-pinned),
  `research/runners/generator_g_core.py` (FIXED bars, adversarially
  hardened), `research/runners/generator_g_gate.py`
- Reused byte-UNMODIFIED: `research/runners/abstention_gate.py`
  (validated moat), `sim/tiny_transformer.py` + the trained
  Generator-F checkpoint
- Evidence: `research/findings/raw/g11_bg/generator_g_gate.json`
  (full transcripts)
- Design/plan: `docs/plans/2026-05-17-generator-G-grounded-noconfab-generation-{design,implementation}.md`
- Prior arc: the 9 NEGATIVE + Generator-E bounded-PASS +
  Generator-F PASS findings
