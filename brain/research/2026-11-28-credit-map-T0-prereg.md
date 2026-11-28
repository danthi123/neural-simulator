# Capability: Sleep consolidation
# Record: brain/research/2026-11-28-local-rule-research-round.md

# T0 credit map prereg (2026-11-28): did the local rules move the adapters where gradient sleep's recall lives?

STATUS: REGISTERED 2026-11-28, after a fresh-context pre-data review (Registration review, end of file). Nothing has run.
Engine spec `tools/engine/specs/credit_map.yaml` is status REGISTERED.

Harness: `brain/experiments/credit_map.py` (imports `local_sleep`, `local_rule_v2`, `local_rule_v3`, `sleep_probe`; edits
none). Tests: `brain/tests/test_credit_map.py`. Kind: a DIAGNOSTIC, not a method screen. Backprop is an instrument only;
nothing here is proposed as the sleep rule.

## 1. The claim under test (research note section 1, verbatim in substance)

**Misdirected credit.** None of the four local rules moved the adapters along the direction that lowers the output error
at the answer word. The update that gives recall is small and specific (a few layers, the answer-predicting positions),
and each rule's error signal had near-zero projection onto it.

**Prediction (note).** Cosine under 0.1 in the layers that carry recall; recall concentrated in a minority of layers.
**Falsified if (note).** Cosine above 0.3 with no recall. Then direction is fine and the limit is step size or drift
budget (the 08-11 per-arm learning-rate artifact); the next step would be a per-layer lr sweep on v1, not a new rule.

## 2. Setup

0.1B (`rwkv7-g1d-0.1b-20260129-ctx8192.pth`), seed 7 only (the dev seed; no generalization claim; section 7), 240 steps,
the sleep_probe episodes (12 tellings, `build_arm_episodes(7)`), the sleep_probe keep texts (`GENERIC`), base state =
BIRTH, rank-8 adapters on every channel mix. Every all-layer arm starts from the same adapter init (`torch.manual_seed(7)`
then `PlasticCortex`). Gradient arms: `brain.sleep.sleep`, Adam lr 5e-4, keep weight 1 (sleep_probe's reference).

| arm | what | role |
|---|---|---|
| no_sleep | adapters at init (B = 0) | gated baseline |
| gradient | gradient sleep, all layers, all positions | the reference update |
| grad_reseed | gradient sleep, replay order of seed 8, same init | positive control of the cosine |
| grad_shuffled | gradient sleep on the shuffled-name episodes | test sanity; content-free direction (report) |
| v1 | `local_sleep`, transport feedback, KP, lr 0.01, decay 1e-3 (the 11-27 0.1B pick) | rule under test |
| v3 | `local_rule_v3.v3_sleep`, double-reading teacher, lr 0.3, keep 1 (0.1B best dev cell) | rule under test |
| grad_distill | `local_rule_v3.distill_sleep` (backprop to the double-reading teacher's logits) | report: bad target or bad rule |
| grad_ans | gradient sleep, all layers, replay CE on answer-predicting rows only | report: position concentration |
| grad_layer<i> | gradient sleep, adapters at layer i only, all positions (i = 0..L-1) | gated: the carrier map |
| grad_ans_layer<i> | the same, answer-predicting rows only | report |
| grad_init, grad_init_ans | -A0 @ dL/dB of the replay CE over every telling at init (all rows / answer rows), no night | report: first-order direction |
| cosine | per-layer cosines of the saved factors (pairs in `credit_map.PAIRS`) | instrument |

Answer-predicting rows: the logit rows that predict the tokens of the answer key in the telling (`answer_span` of v3).
Masked night (`masked_sleep`): `brain.sleep.sleep` with the replay CE on those rows; with every row it equals
`brain.sleep.sleep` (tested).

## 3. Read-outs (every arm)

- recall /12, paraphrase, false recall, generic KL: `sleep_probe.evaluate` (hippocampus off).
- **Graded read-out** (research note section 4): per fact, the ask() prompt `User: <question>\n\nAssistant:` then
  `" " + <the told word holding the key>` teacher-forced (from the key's start to the end of its word as told, e.g.
  " peanuts" for the key "peanut", so the scored tokens are the telling's word, not a re-tokenised stem); answer
  log-probability = summed log p of that word's tokens; rank of its first token (1 = top-1). Reported: mean log p, mean log p of the first token, mean and median rank, top-1 and top-10 counts. The
  bare-key frame is an instrument choice, the same for every arm; only differences to no_sleep are read.
- adapter norm, per-layer norm of the effective update.
- **Cosine**, per layer, in effective weight space dW = A @ B: key path, value path, and both concatenated ("kv",
  the verdict's number). Computed exactly from the factors (no product formed).

## 4. Carrier layers

gain_i = mean answer log p of grad_layer<i> minus no_sleep's. best = max_i gain_i. R = { i : gain_i >= 0.5 x best },
defined only when best > 0.1 nats. "Minority" = |R| < L / 2.

## 5. Verdict (exactly `credit_map.verdict`; tested)

1. **UNDEFINED** if any of: an arm missing (ANY arm of `all_arms(L)`, report-only arms included: a missing unit means an
   incomplete run, so the word is not read; "report-only" below means a report arm that ran but failed its engagement
   gate leaves the word); recall, adapter norm or graded mean log p non-finite on a gated arm (no_sleep,
   gradient, grad_reseed, grad_shuffled, v1, v3, every grad_layer<i>); an engagement gate fails (no_sleep norm 0;
   gradient, grad_reseed, grad_shuffled: norm > 0, the fixed-order replay CE over every telling at the night's end below
   its start, steps run, the layers given; every grad_layer<i>: norm > 0, that CE finite at start and end, steps run,
   the layers given, with NO decrease required (the map exists to find layers that do nothing; a flat layer is a
   reading, gain about 0, not a broken run); v1 norm > 0 with transport feedback; v3 norm > 0, fixed-
   order layerwise error end < start, teacher "double", every layer); gradient recall < 2; grad_shuffled recall >
   no_sleep + 1; no carrier map (best <= 0.1 nats); a cosine of v1, v3 or grad_reseed vs gradient non-finite in a layer of
   R; positive control: mean over R of cos(grad_reseed, gradient) < 0.5.
2. **FALSIFIED** if v1 or v3 has mean over R of cos(rule, gradient) > 0.3. Reading per rule: recall <= no_sleep + 1 ->
   "direction fine, no recall: step size or drift budget; next is a per-layer lr sweep on v1, not a new rule";
   else "aligned and recalling at this config". This deliberately widens the note's falsifier ("above 0.3 with no
   recall"): cosine above 0.3 refutes misdirected credit whatever the recall; the recall only picks the reading.
   One-sided: a negative cosine never falsifies.
3. **CONFIRMED** if v1 and v3 both have |cos(rule, gradient)| < 0.1 in EVERY layer of R, and |R| < L / 2. The
   absolute value is the note's claim, near-ZERO projection: a strongly negative cosine is not near zero.
4. Otherwise **UNDEFINED**, with the reason: |cosine| between the bars; a rule ANTI-ALIGNED (mean over R of cos(rule,
   gradient) < -0.3, reported as `anti_aligned` and named in the reason: the rule moves against the gradient, a
   different finding from misdirected credit, and its own next step); or recall not concentrated in a minority of layers.

Report-only (their engagement gates only their readings, never the word): grad_distill (reading: recall >= max(0.5 x
gradient, no_sleep + 2) -> "rule-limit" on v3's own target, else "target-limit"), grad_ans (the share of the all-position
log p gain kept with answer rows only), grad_ans_layer<i>, grad_init*, and every cosine not named in 5.1-5.3 (to
grad_shuffled, grad_init, grad_init_ans, grad_distill, grad_ans).

## 6. Why these thresholds

0.1 and 0.3 are the note's. The positive-control bar 0.5 sits above 0.3 so the falsifier is reachable: if two gradient
nights that differ only in replay order agree less than 0.5, accumulated-update cosine cannot tell aligned from
misaligned here, and the word is UNDEFINED, not CONFIRMED. The 0.1-nat floor on the best single-layer gain stops a
carrier map built on noise. Gradient recall >= 2 is v3's GRAD_MIN.

## 7. Limits written before data

- One seed (7), one scale (0.1B). Seed-7 gradient recall was 5 (note) and 3 (v3 run, commit 6ff41baa); six-seed mean
  3.83. A CONFIRMED is a credit map for the next method's aim, not a generalization claim.
- Accumulated Adam updates are not gradients; grad_init* give the first-order direction beside them (report).
- Single-layer nights use the all-layer lr (5e-4); a layer can look empty because its lr is wrong for one layer alone.
  Disclosed, not tuned (one night per layer).
- The 0.5 positive-control bar and the 0.1-nat floor are untuned (section 6); both fail toward UNDEFINED, never toward a
  wrong CONFIRMED.
- The graded read-out frame (bare told word right after "Assistant:") is not how the model would phrase an answer;
  absolute log p is not read, only differences to no_sleep.
- v1 at lr 0.01 is nearly inert (norm about 0.3 vs gradient about 16); cosine reads direction, not size.
- Effective-weight cosine weighs the key and value paths by their own update size.

## 8. Run

    POOL_NODE=1 python -m brain.experiments.credit_map run --model M.pth --seed 7 --out-dir D --tag t0
    python -m brain.experiments.credit_map score --dir D --tag t0 --out D/verdict.json

Engine: `python -m tools.engine credit_map` (one job, seed 7; smoke = no_sleep, gradient, v1, v3, grad_init
at 24 steps). About 2L + 8 nights (32 at 0.1B, L = 12). Time, from the v3 0.1B seed-7 units (240 steps, pool CPU
nodes, `local_rule_v3_01b/dev_units`): a gradient night plus its evaluate 218 s, a v3 night 75-87 s, no_sleep's
evaluate alone 38 s. So about 32 x 220 s + the graded read-outs = about 2-3 hours on one node, well under a pool-day. Units are atomic and resume; delta
factors (.pt) are written before each unit's json; the cosine unit runs last from them.

## Registration review

2026-11-28, fresh-context pre-data review (harness, tests, spec; tests run, no model): no BLOCKING defects, one
BLOCKING-before-registering item, eight NON-BLOCKING. Fixed before registering:
- BLOCKING-before-registering: CONFIRMED now needs |cos| < 0.1 in every carrier layer (a rule at -0.9 read as "near-zero
  projection" before); an anti-aligned rule (mean over R < -0.3) is reported and is UNDEFINED with its reason (5.3, 5.4;
  `verdict`, tested).
- Single-layer nights (grad_layer<i>, and the report arms grad_ans_layer<i>) no longer need a strict loss decrease; one
  flat layer no longer makes the run UNDEFINED. The all-layer gradient arms keep the strict decrease (5.1; tested).
- Graded read-out scores the told word holding the key (" peanuts", not the stem " peanut") (section 3; tested).
- 5.1 now says a missing report-only arm is UNDEFINED (incomplete run); 5.2 states that it widens the note's falsifier.
- Boundary tests added: gain exactly 0.5 x best is a carrier; best exactly 0.1 is no map; gradient recall 1 vs 2;
  grad_shuffled recall exactly no_sleep + 1 passes; positive control exactly 0.5 passes; |R| exactly L/2 is not a
  minority; |cos| exactly 0.1 and -0.1 are not under the bar; -0.35 is anti-aligned, not falsified.
- Run time derived from the v3 0.1B units (section 8): about 2-3 hours, not a pool-day.
- Untuned bars (0.5, 0.1 nats) and the read-out frame added to the limits (section 7). File renamed from -DRAFT; spec,
  harness header and spec test updated.
