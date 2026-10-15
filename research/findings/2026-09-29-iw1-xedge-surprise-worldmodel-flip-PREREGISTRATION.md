---
type: preregistration
status: preregistered
date: 2026-09-29
lane: integration (wave 1, packet xedge-surprise-worldmodel; ledger row worldmodel-forward)
mechanism: one-brain cross-edge C1 -- the world model's own D2 prediction-error unit gates one Hebbian co-fire step
  of the live E2 organ's state->pred transition toward the observed valence on a surprising turn
  (research/runners/onebrain_xedge_surprise_worldmodel_production.py, BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL,
  anchor `_XEDGE_SWM_DEFAULT_ON`, which predates this wave); measured for the first time at reply level (LBF row
  xedge-surprise-worldmodel)
seeds: [42, 43, 44, 100, 101, 102]
verdict: PRE-REGISTRATION only. Filed with the row, before any gate-seed job exists. No result is claimed here.
---

# Packet xedge-surprise-worldmodel: flip cross-edge C1 default-on (pre-registration)

Shared steps (D, B, F, decision table, UNDEFINED rule): `2026-09-29-iw1-wave-battery-PREREGISTRATION.md`.

## Why this faculty (selection)

- **Reply impact (the open question).** The de-risk is GO 6/6 (`research/findings/raw/_crossedge_surprise_worldmodel_
  6seed.json`: the transition grows on a surprising sequence, not on an expected one, and not under the obs->surprise
  lesion). Wired default-OFF with a real-handler check on seed 42 (`2026-09-02-c1-surprise-worldmodel-wired-flip-
  ready.md`): byte-identical off, gated growth on real alternating traffic and none under the lesion, 0 of 38
  battery faculties regressed. It has never been measured at the reply. The reply that should carry it is the
  expectation read-out, which states the predicted sign and the pool-rate margin
  (`worldmodel_production_organ.expectation_readout`); the seed-44 arm moved the margin 396.5 -> 279.5.
- **Integration risk.** A flag with one reader and an anchor already exist; ARM A of the wiring finding verified
  byte-identical-off through the real handler. The finding itself says it "earns the flip's readiness".
- **Phase and stages.** One brain: faculties interact through synapses (non-negotiable 2); C1 is a cross-edge.
  What the brain says it expects becomes something it learned from its own surprise, not a build-time constant
  (the E2 residual "teacher-driven, not self-organized from conversation"). Stage 1: the novelty monitor and
  valence from context; Stage 2: affect bearing on cognition. The mission names an affective world model.
- **Checked.** Not retracted (docs/RETRACTED.md; `git log --all --grep` shows the C2 sibling's negative, not C1).
  Not on by default (`_XEDGE_SWM_DEFAULT_ON` False at P; no ledger row; C1's finding says to add one at the flip --
  this packet records it as a sub-rung of `worldmodel-forward` instead of a new row, so the faculty count is
  unchanged).

## Step R: reply level, capability, at P

Row `xedge-surprise-worldmodel` (`research/runners/lbf_rows/xedge_surprise_worldmodel.py`): session `iw1swm`, a
positive turn, eight alternating negative/positive turns (the C1 verify's own texts), then "what do you expect?".
Intact = flag `=1`; lesion = flag `=0` (kind whether-disable: the arms differ only in whether the gated update runs;
the module's neural lesion also silences the D2 notice, so it cannot attribute a reply change to the learning and is
not used). Fields: `answer`, `abstained`, `recalled_svo`. Tag `iw1-R-swm-<P7>`, six gate seeds, `--repeats 2`,
`tools/lb_shard.py jobs --no-fixes --faculties xedge-surprise-worldmodel --extra-env
BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL=1`.

- **R PASS:** `python tools/iw1_score.py R --packet xedge-surprise-worldmodel ...` reads all six seeds DEFINED and
  S2 `regressed`.
- **R FAIL:** any DEFINED seed reads `pass` -- a real negative: the learning does not reach the reply on this probe.
  It is then a trace-level mechanism, recorded as a NO-GO finding for its reply-level integration, not flipped.
- **UNDEFINED:** any seed not DEFINED after one re-run.
- Prediction: PASS on the margin text, if the gate opens on the gate seeds as it did on seed 42's real traffic.
  Not assured: a seed whose gate never opens reads `pass`.

## Packet-specific production-default criteria (in addition to the shared B1, B2, F2, F3)

- **F1:** in `iw1-flip-<F7>` the row `xedge-surprise-worldmodel` (REQUIRED_ENV lifted by the flip) is reply-level
  load-bearing on 6/6 seeds with no BRAIN_* flag set.

## Declared residuals and one production consequence

The teach direction and the gate threshold are declared host boundaries (the de-risk's own); the valence appraisal
feeding the model is host text appraisal. **The organ is process-shared** (`_get_worldmodel_organ()` takes no
session key): with the flag on by default, the model's transitions accumulate across every conversation a server
process serves. Every arm here is a fresh process, so that accumulation is not measured. The HANDOFF lists it as
NEEDS-CLAUDE for the merge review.
