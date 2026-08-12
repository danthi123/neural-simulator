---
type: finding
status: contributing
date: 2026-08-12
mechanism: E2 internal worldview / affective world-model WIRED into the DEFAULT /api/brain-chat turn. The co-resident 2-channel spiking predictive-coding VALENCE forward model (state->pred_{pos,neg} learned transition; obs->surprise<-pred GABA_A inhibition; reuse-by-import from the E2 de-risk, 6/6 GO) makes the next-turn-affect prediction QUERYABLE ("what do you expect / how is this going" -> the two-pool spike-rate read) and fires a genuinely-SPIKING SURPRISE on an affect-trajectory violation (an honest "that shifts the mood unexpectedly" notice). Default-ON, moat-safe (only reads/notices), lesion-load-bearing, NO sim/ edit.
lane: Gate-B / E2 · Internal worldview / affective world-model
lane_ref: E2
verdict: GO / WIRED (production-integration). Single-process synchronous in-process verify on the real /api/brain-chat handler (SIM_BACKEND=numpy, GPU-free stub renderer, rf composer). 15/15 verify checks pass.
seed-waiver: production-INTEGRATION verify of an already-6/6-GO faculty (the E2 de-risk 2026-08-12-affective-world-model-spiking-valence-forward-model-6seed-GO.md, predicted-valence acc 1.00, expected 0 Hz vs violated 37-46 Hz, lesion 3/3). This doc verifies the deterministic WIRING glue on the real handler (single process, one seed=42 organ); the 6-seed evidence is the cited de-risk. Lesion + flag-off arms are decisive on the single wired seed.
artifacts:
  - research/findings/raw/_gateB_worldmodel_production_verify.json
---

# Gate-B / E2: a queryable spiking affective forward model on the default chat turn

**Status:** GO / WIRED. The brain now maintains a spiking affective FORWARD MODEL: it predicts the
interlocutor's next-turn valence (QUERYABLE — "what do you expect / how is this going?"), and fires a
genuinely-SPIKING SURPRISE when the actual next turn VIOLATES that prediction (an affective
prediction-error), surfacing an honest "that shifts the mood unexpectedly" notice.

## The wire (reuse-by-import; NO `sim/` edit)

`research/runners/worldmodel_production_organ.py` builds ONE co-resident 2-channel predictive-coding
valence circuit (`build_world_model_circuit` + `train_transition`, reused from the E2 de-risk), learns the
`state -> pred_{pos,neg}` transition, then FREEZES it, and selects (by the SPIKING read) a
positive-predicting and a negative-predicting state. On the turn: the current affective context (the
appraised valence sign) SELECTS a model state via a persistence prior; the two-pool spiking prediction
`sign(rate(pred_pos) - rate(pred_neg))` is the QUERYABLE expectation; a next turn whose observed valence
FLIPS the sign VIOLATES the held prediction, and the spiking surprise unit (`cp_firing_states[surprise]`,
obs excitation minus the predicted GABA_A inhibition) fires. `webapp/server.py brain_chat` answers a
"what do you expect / how is this going" query with the prediction read-out (early return, like the
feel-query), and on a normal turn prepends the honest surprise notice when the prediction is violated. It
only READS or NOTICES — never manufactures a fact, flips an abstain, or changes WHICH answer the recall
produced.

The prediction read + the mismatch read are the load-bearing SPIKING parts; the valence APPRAISAL and the
persistence state-SELECTION are declared host boundaries (the de-risk's legitimate environment boundary).

## Verify — 15/15 (real handler, numpy-CPU). Artifact `research/findings/raw/_gateB_worldmodel_production_verify.json`

<!--derived-->

(The numbers below are rounded reads of the cited verify artifact
`research/findings/raw/_gateB_worldmodel_production_verify.json`, whose full-precision values are the ground truth.)

- **QUERYABLE.** After a positive-affect turn, "what do you expect" -> "My affective forward model expects
  this to keep going positive (predicted next-turn valence +, pool-rate margin +411 Hz) ..." (worldmodel
  kind=query, pred_sign +1, margin 410.8). The prediction is read off the two spiking pools.
- **VIOLATION fires surprise.** A positive context followed by a negative turn ("I hate this it is bad and
  sad") fires the spiking surprise (24.3 Hz >= threshold 12.2, surprised=True) and PREPENDS the honest
  notice "That shifts the mood unexpectedly — my affective forward model had predicted this would keep
  going positive, and my prediction-error unit fired."
- **PERSISTENCE no surprise.** A negative context followed by another negative turn does NOT fire (kind=update,
  no notice) — the affect trajectory was expected.
- **LESION-LOAD-BEARING.** `BRAIN_WORLDMODEL_LESION=1` zeroes the learned `state->pred` transition:
  (a) the QUERYABLE prediction margin COLLAPSES (410.8 -> 0.0 through the handler); (b) on the production
  organ, the SAME expected observation that is 0.00 Hz (not surprised) intact fires 52.08 Hz (surprised)
  under lesion — the expected/violated separation collapses, so the discrimination is caused by the learned
  SPIKING prediction, not the host state/observation drive.
- **FLAG-OFF byte-identical.** `BRAIN_WORLDMODEL=0` -> worldmodel null on both a violation turn and a query;
  the query is NOT intercepted (falls through to the normal handler), and no notice is prepended.
- **NO-REGRESSION with ALL organs default-ON.** recall ("brain use spikes"), anaphora ("what does it eat"
  -> "The cat eats fish"), abstain, D2 surprise, D4 comprehension, and E1 metacog all hold.

## Honest residuals (declared — the mission's named next rungs, NOT faked)

- **GENERIC pos/neg pools — NOT bound to the ACTUAL interlocutor affect.** The model predicts a GENERIC
  valence. Binding the state + observation to the real interlocutor affect (the P0.3 valence latch + the W5
  ToM channel) so it predicts THIS person's next-turn affect is the NEXT RUNG (currently un-wired). This is
  the mission's explicitly-named residual.
- **HOST state SELECTION.** The persistence prior (a positive context selects a positive-predicting state)
  is a host mapping over the model's own learned predictions, not a learned conversational-state encoder.
- **FIRST-ORDER transition** (Markov-1, no history/context dependence — the HTM-TM high-order predictor is a
  named rung); **TEACHER-DRIVEN** (learned but not self-organized from conversation).
- **CO-RESIDENT** on its own forward-model bridge ALONGSIDE the recall composer — rides on the one-brain
  merge (burn-down #1), exactly as the affect/surprise/comprehension/metacog organs do.

A FUNCTIONAL affective-prediction read-out, NOT a claim of felt experience.

## Escape / lesion knobs

```
BRAIN_WORLDMODEL=0          # disable -> byte-identical oracle (no prediction, no surprise notice)
BRAIN_WORLDMODEL_LESION=1   # zero the learned transition -> the prediction collapses + expected fires (load-bearing)
```
