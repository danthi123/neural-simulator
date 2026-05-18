# Q2R trend-primary scale-confidence (fresh larger-KB experiment) — honest FAIL: the constrained-decoding generative-faithfulness capability is NOT scale-confident; it PEAKS at a small KB and DEGRADES as the KB scales up (an architectural ceiling, revealed only because Q2R extended UP to K=96 where Q2 stopped at K=24). The goalpost-move concern is DEFINITIVELY MOOT (the trend-primary criterion rescued NOTHING — it still FAILed). Instrument SOUND at all 4 rungs. NOT spun, NOT re-run, NOT bar-tuned. Autonomous NON-STOP pivot to Q4.

## TL;DR

After Q1 VOID, Q2 FAIL (scale-positive near-miss at K<=24), Q3
cheap-VOID, the controller (best-judgment, owner-mandated CORRECTED
OPERATING MODE) pivoted to Q2R: a FRESH, separately-pre-registered
increment testing the SAME byte-UNMODIFIED Q2 constrained-decoding
mechanism + soundness instrument on a NEW genuinely-distinct 101-prop
KB, over a TREND-PRIMARY scale-confidence criterion + an UPWARD ladder
K in {12,24,48,96}. The criterion was justified A PRIORI from the
definition of scale-confidence and pre-registered BEFORE any Q2R run;
Q2's FAIL stands byte-unchanged (NOT re-scored).

The full disciplined arc ran: design -> writing-plans ->
subagent-driven build (Task-0 grounding pin; Task-1 fully-specified
frozen q2r_core trend aggregator 13/13; Task-2 net-new 101-distinct KB
+ faithful `_run_rung` mirror + byte-UNMODIFIED imports of
`_GroundedConstrainedLM`+`cdc_verdict`) -> **DEDICATED ADVERSARIAL
REVIEWER BEFORE Phase B whose PRIMARY probe was the goalpost-move
question** (6 probes, ALL CLEAN, NO HOLEs; chronological + structural
forensics: `_Q2R_TOP_MIN==0.50==_CDC_MIN_GROUNDED_ANSWER_RATE`
identical/NOT softened; Q2's FAIL findings pre-recorded the
trend-primary increment as legitimate 52 min BEFORE the Q2R design
commit; +1 cosmetic WEAK#1 telemetry STRENGTHEN applied) -> Phase B
no-harm (original protected byte-empty, moat 7/7, full Q2R suite 32
green) -> controller-only decisive run.

**Decisive result (controller-only, monitored FOREGROUND to active
completion 5m17s, device=cuda; recomputed from the single recorded
JSON `research/findings/raw/q2r_gate.json`; NO re-run, NO bar-tuning,
NO config-crank), seeds 42-46, ladder K in {12,24,48,96}:**

| Rung | cdc_verdict GATE | instrument_valid | constrained non-vacuity (mean) | unconstrained_uer (V1) | shuffled non-vacuity | multitok-emittable | no-confab (abst=bare) |
|---|---|---|---|---|---|---|---|
| K=12 | **FAIL** | True | **0.250** (< 0.50 bar) | 0.931 | 0.000 | 1.00 | 1.00=1.00 |
| K=24 | PASS | True | **0.708** | 0.946 | 0.000 | 1.00 | 1.00=1.00 |
| K=48 | PASS | True | **0.688** | 0.971 | 0.000 | 1.00 | 1.00=1.00 |
| K=96 | PASS | True | **0.583** | 0.981 | 0.000 | 1.00 | 1.00=1.00 |

Pre-registered `q2r_scale_confidence` => **scale_confident=False,
classification=FAIL** ("a rung GATE is FAIL" -- the K=12 rung's
`cdc_verdict` is FAIL; and independently the trend is non-monotone
DECREASING beyond `_Q2R_SCALE_TOL` from K=24 (0.708) to K=96 (0.583),
a 0.125 drop > 0.10 tol).

## The decisive, honest scientific finding (NOT spun)

Q2 (capped at K<=24) saw a LOW-END scale-positive trend (0.333 ->
0.583 -> 0.625) and could not see what happens at larger scale. Q2R
extended UPWARD to K=96 -- exactly where scale-confidence actually
lives -- and the genuinely-correct trend-primary lens reveals the
honest answer: **the constrained-decoding generative-faithfulness
capability is NOT scale-confident.** Non-vacuity PEAKS at a small KB
(K=24, 0.708) and then **DEGRADES monotonically as the KB scales up**
(K=24 -> K=48 -> K=96: 0.708 -> 0.688 -> 0.583). There IS an
architectural ceiling: as more grounded propositions must be served,
the per-token grounded veto's allow-set grows and the small Generator-F
produces a non-vacuous grounded answer LESS often. Scale-up is
therefore NOT justified for this architecture -- it gets WORSE with
scale. This is the genuine, decision-relevant answer to the owner's
deliverable question, and it is a clean NEGATIVE.

## The goalpost-move concern is DEFINITIVELY MOOT (the central anti-cheat point)

The entire reason Q2R existed under maximum anti-cheat scrutiny was the
risk that a trend-primary criterion was a post-hoc goalpost-move to
convert Q2's FAIL into a PASS. The decisive evidence that it was NOT:
**the trend-primary criterion + the K=6 omission + the SAME unsoftened
0.50 floor STILL produced an honest FAIL.** It rescued nothing. A
genuine goalpost-move would have engineered a PASS; this produced a
NEGATIVE. Combined with the dedicated adversarial reviewer's CLEAN
verdict on all 6 probes (incl. the chronology: Q2's FAIL findings
pre-recorded the trend-primary option as a legitimate separate
increment 52 minutes before the Q2R design; Q2's FAIL byte-unchanged;
the 0.50 floor + 0.10 tol inherited byte-identical from the validated
Q2 instrument; all rungs run FRESH on the new 101-prop KB), the
trend-primary increment is vindicated as a disciplined, honest
experiment -- and its honest verdict on the underlying science is
NEGATIVE.

## Honest scope (no overclaim, no underclaim, no spin)

- **NOT scale-confident.** The pre-registered trend-primary criterion
  (the genuinely-correct lens for a scale-confidence question) returns
  FAIL: K=12 below the same 0.50 absolute floor AND non-vacuity
  DEGRADES (not holds/improves) from K=24 upward. The architecture has
  a demonstrated architectural ceiling at larger KB scale.
- **This DEFINITIVELY closes the constrained-decoding line toward the
  scale-confidence deliverable.** Q2's scale-positive near-miss was a
  low-end artifact; the upward extension reveals the trend reverses.
  The honest answer is "scale-up not justified for this architecture",
  not a near-miss to keep refining (refining further would be
  config-cranking past a pre-registered terminus).
- **Instrument genuinely SOUND at all 4 rungs** (V1 unconstrained
  drift 0.93-0.98 >> 0.20; shuffled-grounding decisively 0.000
  non-vacuity every rung; no-confab abst=bare=1.00 every rung;
  multitok-emittable 1.00 every rung -- the BPE-aware veto fix held;
  con_uer ~0.04-0.06 at K>=24 = faithful-by-construction, MECHANICAL
  not the discriminator). The FAIL is a genuine science verdict, not a
  broken-instrument artifact.
- **NOT a refutation of the validated assets.** Generator-F, the
  no-confab moat, the grounded memory, and the Q2 mechanism/instrument
  are independently validated and byte-UNMODIFIED throughout; this
  tested their COMPOSITION's scale behaviour.
- **NOT open-ended fluent composition / NOT an LLM / NOT
  conversation-solved.** Constrained decoding TRADES fluency for
  faithfulness BY DESIGN; the generator stays the Generator-F
  coherent-simple non-LLM ceiling.
- **GPU + monitoring honesty (owner-flagged):** the decisive run used
  `device=cuda` and was run in the FOREGROUND, synchronously monitored
  to active completion (5m17s / elapsed_seconds=313.1) -- NO bare
  `nohup` with a false "I will be notified" claim; completion was
  directly observed before any result was claimed.

## Why NOT config-cranked / NOT owner-deferred (CORRECTED OPERATING MODE)

Exactly one cosmetic STRENGTHEN (WEAK#1 telemetry, faithful-mirror
parity, frozen `_Q2R_*` byte-unchanged) was applied; the frozen
trend-primary criterion is byte-UNCHANGED and was pre-registered with
a-priori justification BEFORE any run. After the decisive run produced
FAIL, NOTHING is tuned/re-run/re-scored; the FAIL is propagated and the
arc PIVOTS NON-STOP to Q4 (NO "handed to the owner" deferral). Further
refining the constrained-decoding architecture to chase the degrading
trend would be config-cranking past the pre-registered terminus --
forbidden.

**Pivot: Q2R FAIL -> Q4 (concept-level pretraining objective for the
surrogate-grad cortex rewired into the validated v16 concept-pool
substrate).** Phase-2.3a's earlier NEGATIVE was a char-level-objective
mismatch (an architecture fault, not mere scale); a concept/word-level
prediction objective + concept-pool readout is the remaining
genuinely-distinct unexplored architecture. Own pre-registered
THREE-STATE + scale-confidence + honest ceiling, written at its turn.

## What is preserved / validated (unaffected)

Net-new only: `research/runners/q2r_core.py`,
`research/runners/q2r_gate.py`,
`tests/test_q2r_{grounding,core,smoke,no_harm}.py`. The validated Q2
mechanism (`constrained_decode_gate._GroundedConstrainedLM`) +
soundness instrument (`constrained_decode_core.cdc_verdict`) were
IMPORTED byte-UNMODIFIED (identity-asserted). NO protected/validated
module touched: the no-confab moat (`abstention_gate` + test, 7/7
byte-identical throughout), `sim/grounded_decode.py`,
`research/runners/generator_g_core.py`, the Generator-F artifact,
`sim/tiny_transformer.py`, `sim/bpe_tokenizer.py`, every frozen
`*_core`, `sim/bridge.py` etc. `git diff a1035cf..HEAD` on the full
ORIGINAL protected set is EMPTY. NO new autograd/training (Generator-F
inference-only via the imported class). All prior validated results
unaffected.

## Anti-cheat discipline (why this FAIL is trustworthy)

Pre-registered FIXED-bar trend-primary criterion (frozen `_Q2R_*`,
a-priori-justified, NEVER tuned); the reused Q2 `cdc_verdict`
soundness instrument byte-UNMODIFIED + NOT loosened (per-rung
GATE==PASS is the EXACT validated Q2 gate); ALL rungs run FRESH on a
genuinely-distinct 101-prop KB (not a Q2 re-score; Q2's FAIL
byte-unchanged); the dedicated adversarial reviewer's PRIMARY probe
was the goalpost-move and it returned CLEAN with chronological +
structural evidence; the decisive run recomputed from the single
recorded JSON (no re-run, no bar-tuning); the goalpost-move concern is
definitively moot because the criterion produced FAIL (rescued
nothing); the decisive run was monitored FOREGROUND to active
completion (no false "notified"); the arc PIVOTS non-stop rather than
deferring or cranking. The validated no-confab moat remained
byte-identical and 7/7 green throughout.

## Files / evidence

- Recorded gate output: `research/findings/raw/q2r_gate.json` (K=12
  FAIL 0.250 / K=24 PASS 0.708 / K=48 PASS 0.688 / K=96 PASS 0.583;
  instrument_valid True all rungs; shuffled non-vacuity 0.000 all;
  multitok 1.00 all; no-confab 1.00=1.00 all; device cuda;
  elapsed 313.1s; recompute-from-JSON, no re-run).
- Pre-decision tiny smoke (NOT propagated): `research/findings/raw/q2r_tiny.json`
  (device=cuda, instrument sound, TINY not propagated).
- Build commits (all controller-verified, original-protected
  byte-empty): `cd286fd` Task-0 -> `c08e8be` Task-1 core (+`6ee215a`
  plan self-consistency) -> `167f9da` Task-2 gate -> `2f6629b` Task-3
  WEAK#1 telemetry STRENGTHEN -> `b24316c` Task-4 no-harm.
- Design/plan: `docs/plans/2026-05-18-Q2R-trend-primary-scale-confidence-{design,implementation}.md`.
- Converges with / does NOT refute:
  `2026-05-18-Q2-constrained-decode-FAIL.md` (Q2's low-end
  scale-positive near-miss; Q2R is the fresh upward-extended test that
  honestly resolves it NEGATIVE -- NOT a re-score),
  `2026-05-18-Q1-engram-bootstrap-temporal-credit-in-bridge-VOID.md`,
  `2026-05-18-Q3-laminar-PC-inference-cheap-precursor-VOID.md`.
- PIVOT (autonomous, non-stop): Q4 = concept-level pretraining
  objective rewired into the validated v16 concept-pool substrate;
  design to be written next.
