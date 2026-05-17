# Generator P — Predictive-Coding Top-Down Generative Pathway — Design

> **For Claude:** REQUIRED NEXT SKILL: superpowers:writing-plans (then
> superpowers:subagent-driven-development). The design's pre-registered
> deep "FAIL" branch
> (`2026-05-16-bidirectional-generative-conversational-agent-design.md`),
> now justified by evidence (G1 + G1.5 negatives) and pruned-to
> (`2026-05-16-generative-G1-followup-branches-design.md`; G1.6 premise
> falsified). Same non-negotiable anti-cheat discipline: pre-registered
> permuted-control-gated honest probe, bars never tuned post-hoc,
> honest negative propagated if it fails.

## The evidence-converged problem statement

Five honest negatives converge on ONE precise cause:

- Inc-1/2/3 (char-level BPTT): generation as an isolated supervised
  next-token predictor — capacity not the bottleneck, memorization ≠
  generalization.
- G1 (bare songbird controller): zero learning signal — the
  self-comprehension judge cannot distinguish intended order from
  scrambled order on the substrate (Step-0 AUC 0.775, thin).
- G1.5 (order-sensitive trajectory readout): worse (AUC 0.40,
  anti-signal). Its pre-registered calibration also falsified G1.6:
  a **perfect, directly-ignited controller** (zero cold-start) still
  cannot be distinguished from scrambled order.

**Converged diagnosis:** the recognition-only G.20 substrate does not
encode *recoverable sequence order*. Igniting ensembles A→B vs B→A
leaves no reliably decodable bottom-up trace. No controller, readout,
or scaffolding can extract order information that the substrate's
bottom-up dynamics do not represent. The missing thing is not a
better *reader* of the substrate — it is a **generative model that
imposes and predicts order top-down**, whose prediction error is the
order-sensitive learning signal the bottom-up substrate cannot
provide. That is exactly predictive coding (Rao & Ballard 1999;
Friston free-energy/active inference; Bastos et al. 2012 canonical
cortical microcircuit) — and it is, independently, the most
biologically-correct generative architecture (the original design's
core thesis).

## Thesis (sharpened by the evidence)

Add a **top-down predictive-coding pathway over the validated concept
ensembles** that, conditioned on the sequence-so-far, *predicts the
next concept ensemble*; the discrepancy between its prediction and
the realized next ensemble is an explicit, order-sensitive prediction
error that (a) trains the generative weights and (b) at generation
time drives selection of the next ensemble (active inference: emit
the concept that minimizes predicted error w.r.t. the intended
proposition). Order is no longer something we *read out of* the
substrate — it is something the generative model *represents and
predicts*. Self-contained: the prediction error is internally
generated (the model predicting its own next state vs the realized
state); the "templates" are the grounded propositions the agent
already stores; no corpus, no external LLM, no templated UX.

## Architecture (net-new P layer; validated substrate UNCHANGED)

Three net-new components over the **unchanged** G.20 ensembles +
unchanged `song_hvc` controller + unchanged `song_g1_core` gate:

- **(P-R) Representation state** — a recurrent state region
  (`pc_state`) holding a distributed code of "sequence so far"
  (which concepts emitted, in what order). Updated each step from the
  realized ensemble activation. NMDA-stabilized recurrence (existing
  `--enable-pfc-nmda` machinery) for a persistent compositional
  buffer. STRICT separation-of-concerns guard (the v12/v13/v15 / G1
  "first do no harm" lesson): `pc_state` reads the ensemble pool but
  its only write into concept pools is the (P-T) top-down prediction
  current — never non-specific feedback.
- **(P-T) Top-down generative prediction** — a learned pathway
  `pc_state → predicted-next-concept-ensemble` (a distribution over
  the validated concept patterns), i.e. the generative model's
  prediction of the next ensemble given the sequence so far.
- **(P-E) Prediction-error units** — `error = realized_next_ensemble
  − predicted_next_ensemble` (Rao-Ballard residual; Bastos
  superficial-layer error analog). The error trains (P-T)/(P-R)
  weights (three-factor / surrogate-gradient on the small P layer
  only — NOT the substrate) and, at generation, is the signal the
  controller descends (emit the next concept whose realized ensemble
  minimizes prediction error toward the intended proposition =
  active-inference rollout).

**Generate (runtime, self-contained):** intention (grounded
proposition the agent holds) clamps the top of `pc_state`; roll out:
(P-T) predicts next ensemble → ignite the concept minimizing error
toward the intention via the existing write-only ignition →
`pc_state` updates from the realized ensemble → repeat. The ordered
sequence of ignitions is the produced utterance, decoded by the
existing A→W readout. No template/LLM/corpus at runtime.

**Train (self-supervised):** for each stored grounded proposition,
the model predicts each next concept from the prefix; prediction
error vs the realized intended next concept trains P (the order
signal P provides that the bottom-up substrate did not). Hippocampal
CLS (existing) guards against forgetting prior propositions.

## Pre-registered anti-cheat gate (SAME bars, SAME discipline)

Reuse `song_g1_core.g1_verdict`/`score_order`/`permuted_order_controls`
**UNMODIFIED** (`_G1_MARGIN=0.10`, `_G1_ABS_FLOOR=0.5` never touched).
New isolated namespace `song_pc.*` (reuse the proven smoke/readout
isolation + cross-mode-refusal machinery). Pre-registered
control-calibrated frozen abstention floor in the P-readout regime
(same control-max/AUC methodology that produced G1's 72.0; frozen,
never recomputed, never 650). **Load-bearing no-harm probe re-proof
REQUIRED before P training** (P adds pathways into concept pools via
(P-T) — the "first do no harm" guard is now *critical*, not
incidental: prove the validated W→A binding + abstention moat are
UNREGRESSED with P present-but-untrained before any training; if it
regresses, P's separation-of-concerns is wrong and must be fixed
before proceeding). Held-out novel propositions, permuted-ORDER
control, ≥10% margin + 0.5 floor. PASS ⇒ predictive-coding top-down
yields self-contained order-correct generation → G2 (scale,
multi-seed, cross-bridge, grammaticality, multi-turn). FAIL ⇒ the
honest, decision-relevant terminal conclusion: self-contained local
generative *production* is out of reach on this substrate/hardware
under no-cheating/local constraints — propagate honestly (no spin),
and the validated grounded continual memory + no-confabulation
abstention stands as the robust deliverable.

## Honest scope, risk, ceiling

This is the **largest** increment (a net-new predictive-coding layer
+ its training + active-inference rollout + gate), the original
design's months-class branch. It is high-risk research, but it is the
**principled, evidence-indicated** mechanism — it directly supplies
the one thing five negatives proved missing (a model that represents
and predicts order), not another way to read a substrate that does
not encode it. Risk is managed the same way the whole arc was: build
incrementally, the load-bearing no-harm probe gates training, the
pre-registered permuted-control gate decides, a maxed-effort FAIL is
itself decision-relevant (it would terminate the generative line
honestly and confirm the grounded-memory asset as the deliverable —
not a failure of method, a real finding). Bars never move; outcome
propagated whichever way it lands.

## Reuse surface (DRY — do NOT rebuild)

Validated substrate UNCHANGED: G.20 320-sparse ensembles +
`song_g1_ignite` write-only ignition + the validated comprehension
readout + `song_g1_noharm_probe` (re-run as the P training gate) +
`abstention_gate`. Reuse UNMODIFIED: `sim/song_hvc.py`,
`research/runners/song_g1_core.py` (verdict/score/permuted, bars
fixed), `sim/train_checkpoint.py` (kill-safe), the smoke/readout
namespace-isolation + cross-mode-refusal + sidecar-frozen-floor
machinery in `song_g1_train.py`/`song_g1_gate.py` (extend with a P
mode, same pattern; do NOT fork). NMDA persistence + region framework
(existing `--enable-pfc-nmda`, BrainRegion/RegionPathway). Net-new:
the `pc_state`/(P-T)/(P-E) predictive-coding layer + its
self-supervised trainer + active-inference rollout + the P gate
variant.

## Scientific basis

Rao & Ballard 1999 (predictive coding); Friston 2010 (free-energy /
active inference — generation = action that minimizes predicted
error); Bastos et al. 2012 (canonical microcircuit: distinct
prediction vs error populations); Keller & Mrsic-Flogel 2018
(predictive processing in cortex). Builds on the project's validated
Pulvermüller ensembles / Kanerva SDM / Tonegawa engrams /
Marr-McClelland CLS. Surrogate-gradient (Neftci 2019) for the small
P-layer weight training only (NOT the substrate).

## Out of scope (YAGNI)

No external LLM ever at runtime. No templated UX speech. No char-level
BPTT. No corpus-of-sequences. No rewrite of the validated substrate
(P is additive + no-harm-gated). G1.6 is NOT executed (its cold-start
premise was falsified by G1.5's direct-ignition calibration —
documented, evidence-based pruning).
