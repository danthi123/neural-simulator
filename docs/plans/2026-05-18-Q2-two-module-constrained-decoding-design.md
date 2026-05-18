# Q2 — Two-module CONSTRAINED-DECODING faithful generation (validated generator proposes, validated no-confab grounded memory vetoes per-token) (design)

> Standing autonomy: documented design calls; brainstorm ->
> writing-plans -> subagent-driven-development -> pre-registered gate ->
> honest propagation EVERY outcome. No config-cranking, no overclaim,
> the no-confab moat byte-identical. PRE-STAGED while Q1's decisive
> in-sim run is in flight so a non-SCALE-CONFIDENT-PASS triggers an
> INSTANT, non-stop pivot (CORRECTED OPERATING MODE).

## Status

PRE-STAGED durable design (pivot queue Q2). Activated ONLY if Q1
(engram-bootstrap temporal-credit) is NOT a SCALE-CONFIDENT-PASS. If
Q1 is SCALE-CONFIDENT-PASS, Q2 stays dormant (the deliverable is met).

## Goal (one sentence)

Test whether wiring the project's two independently-validated assets --
(a) the trained coherent-simple generator (Generator-F TinyGPT, honest
non-LLM ceiling) and (b) the multi-seed-validated no-confabulation
grounded memory (`abstention_gate` "gate 650", byte-UNMODIFIED) -- as a
**per-token CONSTRAINED-DECODING** pair (the grounded memory VETOES
ungrounded next-tokens at EACH generation step) yields faithful grounded
generation that is **scale-confident** (faithfulness does NOT degrade as
the grounded KB scales across a pre-registered local ladder).

## Why this is genuinely distinct from Generator-G NEGATIVE (not a re-run)

Generator-G (`research/runners/generator_g_gate.py`) already tested the
OBVIOUS composition: the validated moat gates answer-vs-abstain FIRST,
then Generator-F generates greedily on grounded queries with the
retrieved proposition merely available in the prompt; faithfulness was
measured POST-HOC (`mean_ungrounded_entity_rate`). Result: honest
**NEGATIVE** -- the no-confab moat IS preserved by construction, but the
small generator **drifts ~89% off the grounded content** because NOTHING
constrains generation per-token (`gg_verdict` faithful bar 0.20; observed
~0.89). The two assets were declared SEPARATE deliverables.

Q2 changes the ARCHITECTURE, not the config: instead of "generate
freely-when-grounded, measure drift after," the grounded memory becomes
a **per-token decoding constraint**. At each generation step the
generator proposes a next-token distribution; the grounded memory
restricts the admissible set to tokens consistent with the retrieved
grounded proposition (a grounded-token allow-set / logit mask derived
from the retrieved text + a small closed function-word set), so the
generator **structurally cannot emit ungrounded content tokens** -- the
89% drift is prevented at generation time rather than measured
afterward. Constrained/guided decoding is a standard, well-understood
technique; the distinctive, validated piece is that the constraint
SOURCE is this project's no-confab grounded memory. This is a different
mechanism with the same goal, NOT a config-crank of Generator-G (which
is the forbidden move).

## Falsify-cheaply precursor

CHEAP and APPLICABLE here (unlike Q1): a throwaway pure-Python probe
(prefix `_`, deleted post-decision, recorded evidence to
`research/findings/raw/`) that, WITHOUT the heavy TinyGPT, simulates the
constrained-decoding contract on a tiny frozen KB: a mock proposer that
(i) emits grounded tokens, (ii) tries to emit ungrounded tokens -- and
verifies the per-token grounded veto reduces ungrounded-entity-rate from
~the Generator-G ~0.89 regime to <= the pre-registered faithful bar
while non-trivial answer rate stays >= the bar AND the no-confab
abstain-on-ungrounded is preserved bit-identically to the bare moat. If
the cheap probe shows the veto cannot even in principle hold faithfulness
without destroying answer rate (e.g. the allow-set is so tight every
answer becomes function-words-only -> `is_answered` False), that is an
honest cheap NEGATIVE -> propagate + pivot to Q3 (no in-LM build). If
GREEN, the heavy build is green-lit; the in-sim pre-registered
THREE-STATE gate decides honestly every outcome.

### Falsify-cheaply precursor — RESULT: cheap GREEN (recorded), with honest caveats + a pre-registration STRENGTHEN it surfaced

Ran (throwaway `_probe_q2_constrained_decode.py`, deleted; evidence
`research/findings/raw/q2_constrained_decode_probe_recorded.txt`), 5
seeds, reusing `abstention_gate` + `generator_g_core`
(`ungrounded_entity_rate`/`is_answered`/`FUNCTION_WORDS` + the frozen
`_GG_*` bars) byte-UNMODIFIED, NO torch/autograd. **CHEAP VERDICT =
GREEN**: V1 genuinely sees drift (unconstrained ungrounded-entity-rate
0.842 >> 0.20); the shuffled-grounding control genuinely FAILS
(0.90-1.0 >> 0.20 -- faithfulness tracks the TRUE grounding, not any
allow-set); no-confab preserved (moat abstains on ungrounded, LM never
touched); contract logic + reused validated metrics/moat compose
correctly.

**Mandatory smell-test (scrutinized HARDER than a FAIL), honest
caveats -- NOT spun:**
1. `con_uer = 0.000` is TAUTOLOGICAL: a per-token veto cannot emit an
   off-proposition content token, so ungrounded-entity-rate vs the
   true prop is 0 BY CONSTRUCTION. "Faithful" is therefore NOT the
   open question for Q2 (the mechanism guarantees it); the in-sim gate
   must NOT report `con_uer~0` as a discovered result -- it is
   mechanical.
2. The probe surfaced that bare `is_answered` (>= 1 CONTENT word in
   the whole response) is **too weak** a non-vacuity bar: a response
   of 1 grounded word + 11 function words passes it. The BINDING Q2
   risk -- does the veto collapse output into near-vacuity -- is NOT
   adequately measured by bare `is_answered`.
3. The cheap GREEN therefore de-risks ONLY the contract logic + metric
   soundness + control discrimination + no-confab preservation. It
   does NOT de-risk whether the REAL Generator-F under per-token
   constraint produces COHERENT non-vacuous text -- that remains the
   in-sim THREE-STATE gate's decisive job. Green-light, honestly
   bounded, never spun as "Q2 works."

**Pre-registration STRENGTHEN (decided NOW, before ANY in-sim run --
legitimate pre-registration, NOT tuning-to-a-result):** the Q2 in-sim
non-vacuity criterion is strengthened from bare `is_answered` to a
**grounded-CONTENT-word sufficiency** bar (a frozen minimum count/
fraction of distinct on-proposition CONTENT words per grounded
response, NOT merely >= 1), pre-registered in the implementation
plan's `constrained_decode_core`, NEVER tuned to a result. This is the
cheap probe doing its falsify-first job: it caught a too-weak bar
BEFORE the heavy build, so the bar is tightened pre-registration.

## Architecture (maximally DRY; net-new vs reused-UNMODIFIED)

**Reused UNMODIFIED (byte-empty in every commit-scoped diff):**
- `research/runners/abstention_gate.py` (`abstain`/`gate`/`650`) -- the
  validated no-confab moat. Gates answer-vs-abstain FIRST exactly as
  Generator-G does (no-confab preserved BY CONSTRUCTION on ungrounded).
- `sim/grounded_decode.py` `grounded_decode` -- the validated
  abstain/answer entry path (the abstain branch never touches the LM).
- The trained Generator-F artifact
  (`research/findings/raw/g11_bg/generator_f_gate.ckpt.s42.real` .pt /
  .bpe.json) + `sim.tiny_transformer.TinyGPT` + `sim.bpe_tokenizer`
  -- byte-UNMODIFIED (the validated coherent-simple generator).
- `research/runners/generator_g_core.py`
  `ungrounded_entity_rate` / `is_answered` / `FUNCTION_WORDS` -- the
  validated anti-vacuous faithfulness metrics, reused byte-UNMODIFIED
  (no new movable metric).

**Net-new (load-bearing) -- the per-token grounded VETO wiring ONLY:**
1. `research/runners/constrained_decode_gate.py` -- kill-safe runner.
   A `_GroundedConstrainedLM` wrapper around the byte-UNMODIFIED
   Generator-F TinyGPT: at each step, compute the model logits (reused
   unmodified), then apply a grounded **allow-mask** (token ids whose
   normalized surface form appears in the retrieved proposition UNION
   the closed `FUNCTION_WORDS` set UNION end/punct) -- argmax over the
   masked logits (still greedy = deterministic = faithful). On a
   grounded query: moat gates -> grounded_decode answer path ->
   constrained generation. On ungrounded: moat abstains FIRST (LM never
   touched) -- no-confab preserved bit-identically to the bare moat.
2. `research/runners/constrained_decode_core.py` -- its OWN pure
   FIXED-bar THREE-STATE verdict (mirrors `generator_g_core` DISCIPLINE
   exactly; does NOT import/mutate `generator_g_core` or any existing
   core). Frozen bars pre-registered in the implementation plan, NEVER
   tuned: `_CDC_FAITHFUL_MAX` (ungrounded-entity-rate ceiling),
   **`_CDC_MIN_GROUNDED_CONTENT`** (the cheap-probe-surfaced
   STRENGTHENED non-vacuity bar: min distinct on-proposition CONTENT
   words per grounded response -- NOT bare `is_answered>=1`),
   `_CDC_MIN_GROUNDED_ANSWER_RATE`, `_CDC_MIN_SEEDS=3`,
   no-confab-preserved relational bar (abstain_on_ungrounded >=
   bare_moat_abstain). Instrument-validity FIRST, fail-closed, VOID
   strictly distinct from FAIL, malformed/junk -> VOID-not-raise.
   `is_answered`/`ungrounded_entity_rate`/`FUNCTION_WORDS` reused from
   `generator_g_core` byte-UNMODIFIED; the strengthened
   grounded-CONTENT-word count is computed in `constrained_decode_core`
   from those reused primitives (no mutation of any existing core).

## Pre-registered in-sim THREE-STATE + SCALE LADDER (frozen, NEVER tuned)

- V1 (instrument soundness): the bare validated moat abstains on the
  ungrounded control (bare_moat_abstain_rate > 0) AND Generator-F
  UNconstrained reproduces the Generator-G drift regime
  (ungrounded-entity-rate well ABOVE `_CDC_FAITHFUL_MAX`) -- proving the
  instrument can SEE drift (so a faithful result is real signal, not a
  trivial/degenerate generator).
- Science: with the per-token grounded veto, mean ungrounded-entity-rate
  <= `_CDC_FAITHFUL_MAX` (NOTE: ~0 by construction -- mechanical, NOT
  the discriminating result; recorded as such) AND **the STRENGTHENED
  non-vacuity bar** holds: mean distinct on-proposition CONTENT words
  per grounded response >= `_CDC_MIN_GROUNDED_CONTENT` (a frozen
  count/fraction, NOT bare `is_answered>=1` -- the cheap probe proved
  bare `is_answered` too weak) AND grounded_answer_rate (under the
  strengthened bar) >= `_CDC_MIN_GROUNDED_ANSWER_RATE` AND no-confab
  preserved (abstain_on_ungrounded >= bare_moat). The DISCRIMINATING
  Q2 signature is "constrained stays NON-VACUOUS" (not "constrained is
  faithful" -- that is mechanical), vs the `unconstrained` control
  which is faithful-FALSE, and `shuffled_grounding` which is either
  faithful-FALSE-vs-true-prop or vacuity-collapsed.
- Controls (must fail): `unconstrained` (Generator-F greedy, no veto =
  the Generator-G regime; must FAIL faithfulness -> proves the veto is
  the discriminator, not the generator), `shuffled_grounding` (veto
  allow-set derived from a DIFFERENT proposition; must FAIL faithfulness
  or collapse answer rate -> proves faithfulness tracks the TRUE
  grounding, not any mask).
- THREE-STATE instrument-validity-FIRST fail-closed; **SCALE LADDER**
  `K in {6, 12, 24}` grounded propositions (pre-registered; frozen
  per-rung KB-construction rule in the plan); `_CDC_SCALE_TOL` frozen.
  SCALE-CONFIDENT iff every rung PASS AND faithfulness does NOT degrade
  beyond tol as K scales AND the science signature holds at the LARGEST
  rung. Works-small-but-faithfulness-degrades-with-KB-size =
  WORKS-SMALL-NO-SCALE-CONFIDENCE (honest non-success -> pivot Q3).
  Reuse the scale-confidence aggregator pattern from Q1
  (`scale_confidence`) -- pure, recomputed from the recorded JSON.

## Honest ceiling (stated up front, NEVER spun)

- **IS (only if SCALE-CONFIDENT PASS):** two validated assets compose
  via per-token grounded constrained decoding into a generator that
  stays faithful to retrieved grounded content BY CONSTRUCTION, and
  faithfulness does NOT degrade as the grounded KB scales -- i.e. the
  only thing between this local PoC and the desired functionality is
  QUANTITATIVE scale of the grounded memory + generator, not a
  qualitative architectural gap. Scale-confidence with a working local
  proof-of-concept at small capacity (the owner's stated deliverable).
- **IS NOT (never spun):** open-ended fluent composition. NOT an LLM.
  NOT GPT-class. NOT conversation-solved. The generator remains the
  Generator-F coherent-simple non-LLM ceiling; constrained decoding
  TRADES some fluency for faithfulness BY DESIGN. The claim is narrowly
  "faithful grounded generation is scale-confident," NOT "fluent
  open-ended generation locally." A faithfulness-vs-answer-rate
  collapse (veto so tight answers go vacuous) is an honest
  non-success, propagated, never spun, -> pivot Q3.

## Anti-cheat plan (non-negotiable)

Cheap falsify-first probe FIRST (honest GREEN/NEGATIVE, recorded);
pre-registered FIXED-bar THREE-STATE + scale ladder in
`constrained_decode_core` (own frozen bars, NEVER tuned, does NOT
import/mutate any existing core); the validated moat + `grounded_decode`
+ Generator-F artifact + `generator_g_core` metrics reused
byte-UNMODIFIED; dedicated ADVERSARIAL REVIEWER on the load-bearing
runner+core BEFORE Phase B (probe: is faithfulness genuinely caused by
the per-token veto and not a degenerate/vacuous generator; is
`unconstrained` a faithful reproduction of the Generator-G regime, not
a strawman; is no-confab bit-identical to the bare moat on ungrounded;
can a vacuous/V1-broken run be scored PASS not VOID; any movable bar;
any autograd beyond the byte-UNMODIFIED reused TinyGPT inference);
controller trust-but-verify each diff with the PROTECTED set byte-empty
(incl. `abstention_gate` + `tests/test_abstention_gate.py` 7/7,
`generator_g_core`, `grounded_decode`, every frozen `*_core`);
controller-only decisive multi-seed multi-rung run + MANDATORY
anti-cheat smell-test (scrutinize a nominal PASS HARDER than a FAIL;
recompute from recorded JSON; no re-run/no bar-tuning/no overclaim);
honest propagation EVERY outcome (findings + capability_status pillar +
schema green + push BOTH remotes) + on non-SCALE-CONFIDENT-PASS the
autonomous Q3 pivot (NO stop, NO owner-deferral). NOTE: Generator-F
inference uses `torch` (the reused, byte-UNMODIFIED validated artifact);
"NO new autograd" means the NET-NEW veto/gate code adds none and no
`.backward()`/training -- inference-only reuse of the already-validated
generator is permitted (it is the validated asset under composition).
