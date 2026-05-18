# Q2 two-module per-token grounded CONSTRAINED DECODING — honest FAIL at the pre-registered scale-confidence criterion: instrument SOUND at all 3 rungs, the BPE-aware veto fix HELD, the signal is real and SCALE-POSITIVE (non-vacuity 0.333->0.583->0.625, K=12/24 PASS, shuffled-grounding decisively 0.000) — but the SMALLEST rung K=6 is below the frozen non-vacuity bar so the frozen "every-rung-PASS" criterion returns FAIL; NOT scale-confident, NOT spun, NOT bar-tuned, NOT re-run; autonomous NON-STOP pivot to Q3 (NO owner-deferral)

## TL;DR

Q1 (engram-bootstrap temporal-credit) honest-VOIDed and the arc pivoted
NON-STOP (no owner-deferral) to Q2: compose the project's TWO
independently-validated assets -- the trained coherent-simple
Generator-F (proposes tokens) and the multi-seed-validated
no-confabulation grounded memory (`abstention_gate` "650",
byte-UNMODIFIED) -- as a **per-token CONSTRAINED-DECODING** pair (the
grounded memory VETOES ungrounded next-tokens at EACH decode step).
Genuinely distinct from the Generator-G NEGATIVE (which measured ~89%
drift POST-HOC; Q2 prevents drift at generation time).

The full disciplined arc ran: design -> cheap falsify-first probe
**GREEN** (smell-tested HARDER than a FAIL; it surfaced that bare
`is_answered` is too weak -> a pre-registration STRENGTHEN to a
grounded-CONTENT-word non-vacuity bar, decided BEFORE any in-LM run) ->
writing-plans -> subagent-driven build (Task-0 grounding pin; Task-1
fully-specified frozen `_CDC_*` THREE-STATE + scale-confidence core,
15/15; Task-2 net-new per-token veto wrapper + kill-safe multi-rung
CLI) -> **DEDICATED ADVERSARIAL REVIEWER BEFORE Phase B** -> Phase B
no-harm -> controller-only decisive GPU run + MANDATORY anti-cheat
smell-test.

**The adversarial reviewer found a genuine science-invalidating HOLE
and it was FIXED, not abandoned (the corrected operating mode working):**
the per-token *word*-level veto was defeated by the BPE *subword*
tokenizer -- grounded multi-subword words (`max`=[314,502],
`friendly`=[204,310]) were STRUCTURALLY un-emittable and ungrounded
fragments leaked. Root-caused (word-vs-subword domain mismatch) and
fixed STRENGTHEN-only BEFORE the decisive run: **Fix A** = a BPE
prefix-automaton veto (proven on the real tokenizer: multi-subword
grounded words ARE emittable, ungrounded words NOT completable;
constrained ungrounded-entity-rate dropped from the subword-salad ~0.83
to **~0.04-0.06**); **Fix B** = an ADDITIVE frozen instrument-validity
floor `_CDC_MIN_MULTITOKEN_EMITTABLE=0.5` so a subword-defeated regime
is an honest VOID (cannot-test), NOT an ambiguous FAIL; **WEAK#2** =
RNG-permuted shuffle control. Existing frozen `_CDC_*` byte-UNCHANGED;
protected set byte-EMPTY; no-confab moat 7/7 throughout.

**Recorded decisive result (recomputed from the single recorded JSON
`research/findings/raw/q2_constrained_decode_gate.json`; NO re-run, NO
bar-tuning, NO config-crank; device=cuda; 72.4s), seeds 42-46, scale
ladder K in {6,12,24}:**

| Rung | GATE | instrument_valid | constrained non-vacuity (mean) | unconstrained_uer (V1) | shuffled non-vacuity | multitok-emittable | no-confab (abst=bare) |
|---|---|---|---|---|---|---|---|
| K=6  | **FAIL** | True | **0.333** (< 0.50 bar) | 0.917 | 0.000 | 1.00 | 1.00=1.00 |
| K=12 | **PASS** | True | **0.583** (>= 0.50) | 0.897 | 0.000 | 1.00 | 1.00=1.00 |
| K=24 | **PASS** | True | **0.625** (>= 0.50) | 0.916 | 0.000 | 1.00 | 1.00=1.00 |

Pre-registered `cdc_scale_confidence` => **scale_confident=False,
classification=FAIL** ("a rung GATE FAIL" -- the smallest rung K=6 is
below the frozen non-vacuity bar; the criterion requires EVERY rung to
PASS).

## Honest scope (no overclaim, no underclaim, no spin)

- **NOT scale-confident.** The pre-registered scale-confidence
  criterion (frozen BEFORE any run) is: every rung PASS AND non-vacuity
  non-decreasing up to `_CDC_SCALE_TOL` AND holds at the largest rung.
  K=6 FAILs the non-vacuity bar (0.333 < 0.50). Therefore the honest
  classification is **FAIL**. It is NOT re-scored, the bar is NOT
  lowered, K=6 is NOT dropped from the ladder, the run is NOT repeated
  -- doing any of those to manufacture a PASS is exactly the forbidden
  goalpost-move and is refused.
- **This is an HONEST, INFORMATIVE FAIL, not a "nothing works" FAIL
  (recorded as decision-relevant context, explicitly NOT spun as a
  win):** the instrument is genuinely SOUND at all three rungs (V1
  unconstrained genuinely drifts 0.90-0.92 >> 0.20; the
  shuffled-grounding control decisively fails with **0.000**
  non-vacuity at EVERY rung -- veto to a WRONG proposition yields zero
  grounded answers; no-confab bit-identical to the bare moat
  everywhere; the BPE-aware veto fix HELD with multitok-emittable=1.00
  everywhere so it is NOT subword-defeated; constrained
  ungrounded-entity-rate ~0.03-0.06 = faithful-by-construction, flagged
  MECHANICAL not the discriminator). The DISCRIMINATING signature
  (constrained NON-VACUITY) is REAL and **SCALE-POSITIVE**:
  0.333 -> 0.583 -> 0.625 monotonically increasing with KB size, with
  K=12 and K=24 both PASSing. The architecture does NOT exhibit an
  architectural ceiling/plateau -- the opposite: it IMPROVES with
  scale. The single reason for the FAIL is that the SMALLEST rung
  (K=6) does not clear the absolute 0.50 non-vacuity bar.
- **Honest methodological reflection (NOT a retroactive re-score):**
  the pre-registered criterion's "smallest-rung-must-also-PASS" clause
  produces a FAIL despite a favorable scale trend. A scale-confidence
  question is fundamentally about the TREND under scaling; a future,
  separately-pre-registered increment could legitimately test a
  trend-primary criterion (improvement + largest-rung value), decided
  fresh BEFORE its run. That is a Q3+ design consideration, NOT a
  reason to relitigate Q2's frozen criterion now.
- **NOT a refutation of the validated assets.** Generator-F (the
  coherent-simple non-LLM), the no-confab moat, and the grounded
  memory are independently validated and unaffected; this increment
  tested their COMPOSITION via per-token constrained decoding.
- **NOT open-ended fluent composition / NOT an LLM / NOT
  conversation-solved.** Constrained decoding TRADES fluency for
  faithfulness BY DESIGN; even the PASSing K=12/24 rungs are the
  Generator-F coherent-simple ceiling under a grounded veto, never
  spun as more.
- **GPU honesty (owner-flagged twice):** the run correctly placed the
  model on `device=cuda`, but Generator-F is a tiny TinyGPT (d_model
  256, 4 layers, vocab 513); the GPU forward is microseconds and
  wall-clock is dominated by the CPU-side per-token BPE-automaton veto
  + Python generation loop (whole decisive run = 72.4s, CPU/Python-
  bound). GPU utilization is near-zero BY THE NATURE OF THIS WORKLOAD
  (a 513-vocab toy transformer with a Python token-veto), NOT a
  misconfiguration. GPU matters for the spiking-bridge runs, not these
  validated-asset gates. Stated plainly, not hidden.

## Why NOT config-cranked / NOT owner-deferred (CORRECTED OPERATING MODE)

The cheap probe's pre-registration STRENGTHEN (grounded-content-word
bar) and the adversarial-review STRENGTHEN (Fix A BPE-aware veto + Fix
B additive VOID floor + WEAK#2) were all root-caused correctness
hardenings applied BEFORE the decisive run, with every existing frozen
`_CDC_*` value byte-UNCHANGED and transparently logged -- the discipline
working, not cranking. After the decisive run produced FAIL, NOTHING is
tuned/re-run; the FAIL is propagated and the arc PIVOTS NON-STOP to Q3
(NO "handed to the owner" deferral), per the owner-mandated CORRECTED
OPERATING MODE: an honest non-success is propagated then immediately
triggers the next genuinely-distinct architecture in the durable pivot
queue.

**Pivot: Q2 FAIL -> Q3 (Larkum laminar microcircuit using the
DURABLE-SOUND PC inference with a NON-PC-training-loop learning
signal).** The PC arc's durable positive (PC local update == backprop
gradient, cos~0.995, 5 seeds -- the Whittington-Bogacz equivalence
empirically held) means the PC *inference* is sound; only the PC
*training-loop accumulation* VOIDed. Q3 uses that sound laminar
predictive inference for hierarchical generation but learns via a
DIFFERENT validated signal (engram-bootstrap one-shot bind /
target-propagation), genuinely distinct from the PC-learning VOID. Own
pre-registered THREE-STATE + scale ladder + honest ceiling, written at
its turn.

## What is preserved / validated (unaffected)

Net-new only: `research/runners/constrained_decode_core.py`,
`research/runners/constrained_decode_gate.py`,
`tests/test_q2_{grounding,smoke,no_harm,veto_bpe}.py`,
`tests/test_constrained_decode_core.py`. NO protected/validated module
touched: the no-confab moat (`abstention_gate` +
`tests/test_abstention_gate.py`, 7/7 byte-identical throughout),
`sim/grounded_decode.py`, `research/runners/generator_g_core.py`
(REUSED byte-UNMODIFIED -- metric primitives), the Generator-F
artifact, `sim/tiny_transformer.py`, `sim/bpe_tokenizer.py`, every
frozen `*_core`, `sim/bridge.py` etc. `git diff 02addfa..HEAD` on the
full original protected set is EMPTY. NO new autograd/training (torch
inference-only reuse of the validated Generator-F). All prior validated
results unaffected.

## Anti-cheat discipline (why this FAIL is trustworthy)

Pre-registered FIXED-bar THREE-STATE + scale-confidence; the cheap
probe smell-tested harder than a FAIL and surfaced + pre-registration-
STRENGTHENED a too-weak bar BEFORE any in-LM run; the dedicated
adversarial reviewer found a REAL science-invalidating HOLE (BPE
subword-defeat) and it was root-caused + STRENGTHEN-fixed BEFORE the
decisive run (Fix A faithful veto proven on the real tokenizer; Fix B
makes subword-defeat an honest VOID; existing frozen bars
byte-unchanged); the decisive run recomputed from the single recorded
JSON (no re-run, no bar-tuning); the FAIL is NOT re-scored despite a
favorable scale trend (refusing the goalpost-move is the point of
freezing the criterion); the GPU-utilization reality stated plainly;
the arc PIVOTS non-stop rather than deferring. The validated no-confab
moat remained byte-identical and 7/7 green throughout.

## Files / evidence

- Recorded gate output: `research/findings/raw/q2_constrained_decode_gate.json`
  (K=6 FAIL 0.333 / K=12 PASS 0.583 / K=24 PASS 0.625; instrument_valid
  True all rungs; shuffled non-vacuity 0.000 all rungs; multitok 1.00
  all; no-confab 1.00=1.00 all; device cuda; 72.4s; recompute-from-JSON,
  no re-run).
- Pre-decision tiny smoke (NOT propagated): device=cuda, instrument
  valid, con_uer 0.056, multitok 1.00 -- the BPE-aware veto fix
  verified before the decisive run.
- Build commits (all controller-verified, protected byte-empty):
  `edddcec` Task-0 -> `cb3603b` Task-1 core -> `8b561642` Task-2 gate
  -> `8769470` Task-3 STRENGTHEN (Fix A+B+WEAK2) -> `461baa9` Task-4
  no-harm.
- Design/plan: `docs/plans/2026-05-18-Q2-two-module-constrained-decoding-{design,implementation}.md`
  (CORRECTED OPERATING MODE + pre-registered scale ladder + the
  cheap-probe + adversarial STRENGTHENs recorded).
- Converges with / does NOT refute: `2026-05-18-Q1-engram-bootstrap-
  temporal-credit-in-bridge-VOID.md`,
  `2026-05-18-predictive-coding-cheap-gate-VOID-with-durable-V1-positive.md`
  (Q3 builds on that durable PC-inference positive).
- PIVOT (autonomous, non-stop): Q3 = Larkum laminar microcircuit
  (durable-sound PC inference + non-PC-training-loop learning); design
  to be written next.
