---
type: finding
status: live
date: 2026-05-16
---

# Generator G1 — songbird controller (Approach B): honest NEGATIVE (pre-registered gate FAILED)

## TL;DR

The G1 cheap-first B-probe — a songbird-HVC sequential controller
trained ONLY by the sim self-comprehending its own babbled
productions (no external teacher, no corpus, no templates) over the
validated G.20 sparse-ensemble substrate — **FAILED its pre-registered
held-out anti-cheat gate.** This is a real, honest negative. The gate
was NOT tuned, the controller was NOT config-cranked, the abstention
floor was NOT lowered. It is the design's explicitly pre-registered
"FAIL" outcome and it is decision-relevant: it routes the path, by
evidence and cheapest-first, to the next mechanism.

## The pre-registered gate result (FIXED bars, never touched)

`song_g1_gate.py` on the trained checkpoint (epoch 59), evaluating
ONLY the 2 held-out propositions (NEVER trained), with the
sidecar-frozen control-calibrated floor `g1_abstain=72.0` (NOT the
literal 650; NOT recomputed), `meta_smoke=False` asserted, bars
`_G1_MARGIN=0.10` / `_G1_ABS_FLOOR=0.5` untouched:

| held-out prop | true_score | best_perm | gate_cleared | top_rate | verdict |
|---|---|---|---|---|---|
| int=4 `old hard` (bridgeC) | 0.000 | 0.000 | N (54.0 < 72.0) | 54.0 | FAIL |
| int=5 `ride smell` (bridgeB) | 0.000 | 0.000 | N (70.0 < 72.0) | 70.0 | FAIL |

Aggregate (`g1_verdict` on the means): mean_true 0.000, mean_best_perm
0.000, 0/2 cleared the frozen floor → **GATE: FAIL**.

## Mechanism (the honest, useful diagnosis)

Training ran the full pre-registered 60-epoch protocol. **`mean_reward
= 0.0000` for every single epoch** (the only value ever recorded);
temperature decayed fully 0.5 → 0.000; `SongHVC.W` never moved from
random init because the DA-gated `reinforce` is a strict no-op at
reward 0 and reward was never > 0. `n_gate_cleared` was 2–16 per epoch
— productions DID clear the abstention floor (they decode confidently
to *some* concept) but **never to the intended first concept**, so
`compose_reward` (score_order of the integrated decode vs the intended
order) was always 0 → no learning signal ever reached the controller.

This was foreshadowed honestly, pre-data, by the pre-registered Step-0
control calibration: even the *directly-ignited intended order*
(encoded mean 73.2) barely separated from the permuted/random
controls (control-max 72.0), AUC only 0.775. **The bottleneck is the
order-readout / order-discriminability of the integrated
single-concept self-comprehension judge on the existing
recognition-only substrate — not the songbird controller idea per
se.** A judge that cannot tell intended order from scrambled order
gives the controller no gradient; a controller with no gradient
cannot learn. This is a precise, architecture-level finding, not a
trainer bug: the trainer was reviewed across four focus areas, the
no-harm probe independently confirmed the substrate + decode work and
that a silent controller does not regress the validated path, and the
diagnosis is corroborated by the independent pre-registered Step-0
numbers.

## Anti-cheat discipline (this is a maxed-INTEGRITY negative)

The two-stage subagent review + load-bearing-gate discipline caught
and fixed FOUR distinct integrity issues **before any training datum
existed**, every one *strengthening* the gate's validity (never
weakening it), all documented as pre-registration corrections:

- **C1/C2/I1:** false-PASS holes in the pure `g1_verdict`/`score_order`
  (zero-permuted PASS, confabulation-blind scoring, `>`/`>=` boundary)
  — fixed; added an absolute 0.5 majority floor.
- **Correction 2:** the literal-650 abstention threshold was
  calibrated on a different (continuous-drive) regime than
  `self_comprehend`'s no-drive residual; applying it blindly risked a
  FALSE NEGATIVE — fixed by a pre-registered, control-distribution
  AUC-calibrated, regime-specific frozen floor.
- **Corrections 3/4:** no-harm-gate flakiness + an unmargined
  subject-qualification bug (near-650 straddlers tripping on substrate
  noise) — fixed with a principled cushion from the documented
  substrate variance.
- **Smoke/full isolation:** a sidecar-namespace footgun that could
  have gated the verdict on a smoke-calibrated floor — fixed with
  path isolation + cross-mode-reuse refusal.

650 never lowered. `_G1_MARGIN`/`_G1_ABS_FLOOR` never moved. The
controller was never config-cranked to chase reward. The protocol ran
to its pre-registered completion (60 epochs) rather than being cut
short. This is the strongest form of an honest negative: a
maxed-integrity, pre-registered FAIL.

## What this means and the decision-relevant next step

Per the design's pre-registered logic, FAIL ⇒ the bare controller
(Approach B) is insufficient and predictive-coding top-down (P) is
required. The *sharpened* diagnosis refines this into a cheapest-first,
evidence-routed sequence (pre-staged at
`docs/plans/2026-05-16-generative-G1-followup-branches-design.md`):

1. **G1.5 (next, cheap — days, no architecture rewrite):** the judge
   discards the order signal by taking `argmax` of the *final*
   residual. Decode the *trajectory* of transient concept activations
   across the production instead, scored vs the intended order, with
   the SAME pre-registered `g1_verdict` + permuted-ORDER control + a
   regime-recalibrated frozen floor. Tests whether the order signal
   exists in the substrate's dynamics but was being thrown away. If
   yes → controller viable with a better judge. If no → P justified by
   evidence, not assumption.
2. **G1.6 (if G1.5 shows cold-start, not discriminability, is the
   limit):** biologically-faithful subsong→plastic developmental
   scaffolding for the sparse-reward cold-start (tutor template = the
   grounded proposition the agent already stores; fully faded before
   the held-out eval).
3. **P (if G1.5/G1.6 exhaust):** the design's deep pre-registered
   branch — Rao-Ballard top-down generative + prediction-error so the
   judge becomes a sequence-likelihood generative model. Independently
   the more biologically-correct generative architecture.

A FAIL is not a dead end: it exhausts the cheap-controller hypothesis
and converges the path, by evidence, on the architecture that can
actually generate.

## The robust validated asset is unchanged

Generation remains unproven. The project's genuinely validated,
non-fragile contribution — the **trustworthy grounded continual
memory with no-confabulation abstention** (G.20 sparse ensemble,
160@100%/320@98.4% multi-seed; no catastrophic forgetting per
Marr/McClelland CLS) — is untouched by this negative; the no-harm
probe explicitly proved the songbird controller does not regress it.

## Files

- Controller/core: `sim/song_hvc.py`,
  `research/runners/song_g1_core.py` (+ `tests/test_song_hvc.py`,
  `tests/test_song_g1_core.py`)
- Adapter/probe/trainer/gate:
  `research/runners/song_g1_{ignite,noharm_probe,train,gate}.py`
  (+ `tests/test_song_g1_{ignite_smoke,gate}.py`,
  `tests/test_song_g1_noharm... ` raw)
- Evidence: `research/findings/raw/g11_bg/song_g1_train.log`,
  `song_g1.ckpt.npz` (+ `.meta.json` sidecar, smoke=False,
  g1_abstain=72.0), `song_g1_noharm.json`, `song_g1_gate.json`
- Design + plan + pre-staged branches:
  `docs/plans/2026-05-16-bidirectional-generative-conversational-agent-{design,implementation}.md`,
  `docs/plans/2026-05-16-generative-G1-followup-branches-design.md`
- Prior honest negatives this extends:
  `2026-05-16-generator-increment{1-foundation,2-distillation-NEGATIVE,3-capacity-scan-NEGATIVE}.md`
