# Generator-H — Constrained-Vocabulary Grounded Realizer (separate-components; faithfulness BY CONSTRUCTION; the genuinely-open question is realization QUALITY) — Design (ACTIVE)

> **For Claude:** REQUIRED NEXT SKILL: superpowers:writing-plans (then
> superpowers:subagent-driven-development). Continuous autonomous arc
> (user 2026-05-17: a week autonomous, no stopping/asking, documented
> design calls, no config-cranking a terminated mechanism, self-
> contained at RUNTIME, local 3090, public corpus authorized, full
> architectural freedom). The pre-registered successor that takes the
> SEPARATE-deliverables path Generator-G's own finding pointed to.

## Why this is genuinely decision-relevant (NOT config-cranking a terminus)

The 12-mechanism conversational-GENERATION-PRODUCTION line is
terminally converged: 9 honest negatives + Generator-E (genuine but
BOUNDED, local fragments) + Generator-F (GENUINE validated PASS — a
self-contained small Transformer that generates coherent SIMPLE text
at the EXPLICIT small-Transformer TinyStories ceiling, NOT an LLM) +
Generator-G (honest NEGATIVE pre-registered TERMINUS — fluency +
grounded-faithfulness do NOT compose into ONE self-contained artifact
at feasible local small-LM scale; no-confab IS preserved by
construction; **the two validated assets are SEPARATE deliverables**).

Generator-G's own honest finding explicitly framed the non-terminated
path: a SEPARATE-components pipeline where the moat decides what is
true/known and an explicit content->surface realization step renders
ONLY moat-approved content. Generator-H is that path. It is NOT
"make the substrate generate" (terminated). It is NOT "unify
fluency+faithfulness in ONE small artifact via free/constrained
decoding" (Generator-G: the realizer there was FREE over the FULL
vocab and drifted ~89% off — Max->Bob). It is NOT a config-crank of
S/D/E/F/G (net-new realizer component + net-new pre-registered gate;
all validated assets reused byte-UNMODIFIED).

The genuinely-different mechanism: the realizer's decode vocabulary is
**dynamically HARD-RESTRICTED per query to the moat-approved retrieved
proposition's own token ids U a tiny closed function-word set**.
Confabulation (emitting an entity not in the retrieved content) is
**STRUCTURALLY IMPOSSIBLE**, not merely discouraged. "Bob" is not in
the allowed id set, so it physically cannot be emitted.

## Evidence grounding (falsify-cheaply, done BEFORE designing — 2026-05-17)

A cheap throwaway probe loaded the trained Generator-F TinyGPT + BPE +
the validated moat (all self-contained, CPU) and compared, on a 3-fact
toy KB + 3 ungrounded queries:

- **FREE decode (= exactly Generator-G):** mean ungrounded-entity-rate
  **0.791** ("max is a big friendly dog" -> "who likes to play with
  them. They share and have fun"). Reproduces Generator-G's terminus.
- **CONSTRAINED decode (hard-masked vocab — the new mechanism):** mean
  ungrounded-entity-rate **0.024** (~33x more faithful), 0/3 verbatim-
  echo, ungrounded 3/3 ABSTAIN (moat-first; LM never invoked).

**Honest smell-test of the actual transcripts (NOT rubber-stamping the
automated "POSITIVE"):** constrained GREEDY decode is faithful but
DEGENERATE in a new way — it loops ("and a small red ball and a small
red ball...", "and fast and fast and fast"). Conclusion that drives
this design: **faithfulness is solved BY CONSTRUCTION (decisive, real,
genuinely-different from Generator-G); realization QUALITY (non-loop,
covers the stored content) is the genuinely-open, pre-registrable
question.** Honest ceiling stated up front: a PASS is "faithful
STRUCTURED grounded utterances," explicitly NOT GPT-fluent prose,
NOT an LLM.

## Thesis

A self-contained pipeline of SEPARATE validated components:

1. **Validated no-confab moat** (`abstention_gate`, gate 650, byte-
   UNMODIFIED) decides answer-vs-abstain FIRST. On abstain the
   realizer is NEVER touched -> no-confab preserved BY CONSTRUCTION
   (already validated in Generator-G; re-pinned here by a spy-LM unit
   test).
2. On grounded: the **validated grounded memory** retrieval entry
   point yields the exact stored proposition (reused, NOT rebuilt).
3. **Net-new constrained realizer:** the validated Generator-F TinyGPT
   (byte-UNMODIFIED) decodes with per-step logits HARD-masked to
   {retrieved proposition token ids} U {tiny FIXED closed function-word
   set}, PLUS a no-repeat-ngram loop-block and a coverage-stop. Faithf-
   ulness is then a PROVABLE UNIT TEST (the mask provably excludes all
   non-allowed ids), not merely an empirical hope. The genuinely-open
   empirical question the gate measures is non-degeneracy.

This is the honest realization of the conversational goal within the
converged reality: faithful grounded utterances that refuse to make
things up — produced by validated components used as SEPARATE
cooperating parts, NOT fused into one small model (the Gen-G terminus).

## Pre-registered gate — LOAD-BEARING criteria (FIXED bars, never tuned)

Mirror the hardened anti-cheat discipline (FIXED module constants in
Generator-H's OWN core; do NOT import/modify gate_core / song_g1_core /
subword_lm_gate_core / generator_g_core / abstention_gate; >=3 seeds;
permuted/held-out controls; mandatory smell-test; never tuned).

1. **No-confabulation PRESERVED (load-bearing, relational):** on a
   held-out UNGROUNDED set (never stored) the pipeline ABSTAINS at
   >= the validated bare-moat abstain rate (fail-closed if no
   ungrounded control). One ungrounded query realized with content =
   FAIL. (By construction; verified.)
2. **Faithfulness BY CONSTRUCTION:** mean ungrounded-entity-rate (resp.
   content tokens not in retrieved U closed function set) <= a FIXED
   bar `_GH_UNGROUNDED_ENTITY_MAX = 0.20` (same value as Generator-G's
   `_GG` so it is DIRECTLY comparable; the probe shows ~0.02, so this
   bar is genuine and non-vacuous, not a moved goalpost). Additionally
   a pure UNIT TEST proves the masked-decode step can never select a
   non-allowed id (faithfulness is provable, not just measured).
3. **NON-DEGENERATE realization (THE genuinely-open question):** the
   realized utterance must (a) COVER the retrieved proposition's key
   content tokens (coverage >= `_GH_MIN_COVERAGE = 1.0` — every
   content word of the stored fact appears at least once) AND (b) NOT
   loop-collapse (max repeated-ngram fraction <= `_GH_MAX_REPEAT =
   0.50`, i.e. the "and fast and fast and fast" failure the probe
   exposed is a FAIL). This is the new, non-circular, pre-registered
   criterion that makes a PASS meaningful rather than rubber-stamping
   a faithful-but-degenerate loop.
4. **Anti-trivial:** grounded-answer-rate >= `_GH_MIN_GROUNDED_ANSWER
   _RATE = 0.5` (not trivially always-abstain).
5. **MANDATORY anti-cheat smell-test:** scrutinize a PASS HARDER than a
   FAIL (the probe already shows the automated metric could rubber-
   stamp a faithful-but-looping output). Read EVERY ungrounded
   transcript (must all ABSTAIN) and EVERY grounded transcript (must
   cover the fact AND not loop). Recompute from recorded JSON; no
   re-run; no bar-tuning.

PASS (scrutinized genuine) => the honest culmination of the converged
arc: a self-contained, local, no-cheat pipeline that produces FAITHFUL
(confabulation-structurally-impossible) grounded utterances covering
the stored content without looping, AND preserves the validated no-
confabulation property — at the EXPLICIT small-LM ceiling (structured
faithful utterances, NEVER spun as GPT-class / an LLM; the biology-
grounded no-confab grounded memory remains the distinctive primary
contribution). FAIL => the decision-relevant terminus: even
constrained-vocab realization cannot be made non-degenerate at the
small-LM ceiling; the deliverable is the two SEPARATE validated assets
used independently (retrieval + abstention), no faithful fluent
pipeline. Either way decision-relevant, pre-registered, NOT config-
cranked. An Arch-A FAIL is NOT a license to escalate to beam/templates.

## Architecture (net-new small; validated components reused UNMODIFIED)

Reuse byte-UNMODIFIED (DRY): `sim.tiny_transformer.TinyGPT` + the
trained Generator-F checkpoint
(`research/findings/raw/g11_bg/generator_f_gate.ckpt.s42.real.{pt,bpe.json}`,
NO retrain); `research.runners.abstention_gate` (moat, gate 650);
`sim.bpe_tokenizer`; the validated grounded-retrieval entry point
(reuse the SIMPLEST validated source of a `ranked` list, exactly as
Generator-G's runner did — chosen at plan time, NOT rebuilt); the
`_TinyGPTLM` loader pattern from `generator_g_gate` (DRY).

Net-new (small, pure-testable):
1. `sim/constrained_realize.py` — PURE policy. Given (retrieved_text,
   tok, lm, allowed-extra function set, no_repeat_ngram=int,
   max_new=int): build `allowed = set(tok.encode(retrieved)) U
   {function-word ids}`; greedy decode where each step (a) hard-masks
   logits to `allowed` (provably excludes all non-allowed ids — UNIT
   TEST), (b) blocks any next id that would complete a repeated
   n-gram of size `no_repeat_ngram`, (c) stops when all retrieved
   content-token ids have appeared >=1 (coverage-stop) or `max_new`.
   The moat-abstain short-circuit (reuse `abstention_gate.gate`
   FIRST; on None the lm is NEVER touched — spy-LM unit test) lives
   here, mirroring `grounded_decode`'s shape but NOT importing/
   modifying it. ASCII only.
2. `research/runners/generator_h_core.py` — PURE FIXED-bar verdict.
   OWN frozen constants `_GH_UNGROUNDED_ENTITY_MAX=0.20`,
   `_GH_MIN_COVERAGE=1.0`, `_GH_MAX_REPEAT=0.50`,
   `_GH_MIN_GROUNDED_ANSWER_RATE=0.5`, `_GH_MIN_SEEDS=3`. Functions:
   `coverage(resp, retrieved)`, `max_repeat_ngram_fraction(resp)`,
   `ungrounded_entity_rate(resp, retrieved, function_words)` (reuse
   Generator-G's exact definition for comparability — re-implement
   locally, do NOT import generator_g_core), `gh_verdict(...)`
   PASS iff no-confab-preserved AND faithful AND covered AND not-
   looped AND not-trivially-abstaining; `gh_aggregate_multiseed`.
   Adversarial CPU-TDD: always-abstain=>FAIL; missing control=>FAIL;
   looping output (probe's "and fast and fast")=>FAIL; verbatim-echo
   is faithful+covered+non-loop (the smell-test/copy read is the
   controller's job — recorded, not bar-gamed); <3 seeds=>FAIL;
   bars immutable to results.
3. `research/runners/generator_h_gate.py` — thin runner mirroring
   `generator_g_gate` SHAPE (moat-first; reuse `_TinyGPTLM`; FROZEN
   grounded/ungrounded query sets; per-seed transcripts recorded for
   the mandatory smell-test; >=3 seeds; <3 -> exit 2; kill-safe
   `.resume.json`; ASCII-only; banner states the HONEST CEILING —
   faithful STRUCTURED grounded utterances, NOT an LLM).

## Honest ceiling / risks (no overclaiming)

- The probe already shows constrained GREEDY decode loops. The honest
  expectation, stated up front: no-repeat-ngram + coverage-stop may
  NOT be enough to clear `_GH_MAX_REPEAT`/`_GH_MIN_COVERAGE` at the
  small-LM ceiling -> that honest FAIL is the decision-relevant
  terminus (two separate assets used independently). A PASS is
  reported STRICTLY as "faithful structured grounded utterances at the
  small-Transformer ceiling," with verbatim transcripts, NEVER spun
  as GPT-class / global-coherence / an LLM, NEVER as overturning the
  converged 12-mechanism conclusion.
- Faithfulness-by-construction is the genuinely-different,
  decisive contribution and is UNIT-TESTABLE (not a bar to tune).
- Self-contained at RUNTIME (trained TinyGPT weights + BPE JSON + the
  self-contained validated memory; no external dep/LLM/corpus/
  templates — templates are a standing user-rejected cheat, hence
  Arch B is REJECTED).
- Local 3090; reuse trained Generator-F ckpt (no LM retrain); kill-
  safe; ASCII-only. The cheap slice is fast (reuses the ckpt).

## Out of scope (YAGNI)

No external dependency at RUNTIME ever. No templates (rejected cheat).
No beam/extra knobs (Arch C deferred; an Arch-A FAIL is the honest
terminus, NOT a license to escalate — escalating would be config-
cranking past a pre-registered terminus). No new global bar; gate_core
/ song_g1_core / subword_lm_gate_core / generator_g_core /
abstention_gate / tiny_transformer byte-UNMODIFIED (Generator-H-core
has its OWN frozen `_GH_*` constants). No LM re-training. Larger scale
/ multi-turn dialogue are LATER increments noted but NOT detailed.

## Scientific basis (catalog + arc)

Retrieval-grounded constrained decoding (faithfulness-vs-fluency
literature); constrained-vocabulary / lexically-constrained decoding
and no-repeat-ngram anti-degeneracy (Paulus 2018; Holtzman 2020 — the
neural-text degeneration / repetition problem the probe reproduced);
the project's validated no-confabulation abstention moat (the
distinctive contribution); Generator-F (validated small-Transformer
generation, Eldan & Li small-LM ceiling); Generator-G's own finding
(separate-components content->surface is the non-terminated path). The
hardened anti-cheat discipline (held-out, multi-seed, FIXED bars,
mandatory smell-test scrutinizing a PASS harder than a FAIL) is the
adjudicator.
