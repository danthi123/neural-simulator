---
type: finding
status: go
date: 2026-09-16
mechanism: spiking-wire-in-flips-novelty-anaphor-qroute
lane: scaffold-retirement
seeds: [42, 43, 44, 100, 101, 102]
verdict: FLIPPED DEFAULT-ON — three spiking wire-ins (novelty->habituation, anaphor->CA3, question-route->WTA) pass
  the integrated /api/brain-chat no-regression battery (0 of 38 other default-on faculties regress, each flag
  ON-vs-OFF through the real handler) on top of their prior 6-seed focused load-bearing GO. Host paths kept as
  byte-identical `=0` opt-outs. B-curiosity `_XEDGE_CD6` no-regression ALSO passed but its flip is DEFERRED (see below).
runner: research/runners/onebrain_regression_battery.py
artifacts:
  - research/findings/raw/_regression_battery/battery_BRAIN_SPIKING_NOVELTY.json
  - research/findings/raw/_regression_battery/battery_BRAIN_SPIKING_ANAPHOR.json
  - research/findings/raw/_regression_battery/battery_BRAIN_SPIKING_QROUTE.json
  - research/findings/raw/_regression_battery/battery_BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6.json
external: NO-EXTERNAL-NEEDED -- production flip of already-GO'd, already-wired spiking mechanism de-risks
  (2026-09-09 novelty/anaphor/qroute wire-in findings); no new biological claim.
builds_on:
  - research/findings/2026-09-09-rank14-question-route-selection-wta-wirein-default-off.md
---

# Three spiking wire-ins flipped DEFAULT-ON (novelty / anaphor / question-route) — integrated no-regression GO

**One-line.** The 2026-09-09 wiring arc is completed: `BRAIN_SPIKING_NOVELTY`, `BRAIN_SPIKING_ANAPHOR`, and
`BRAIN_SPIKING_QROUTE` are flipped from default-OFF to default-ON, so the brain now decides word-novelty (synaptic
habituation), anaphora ("it/that/they" -> CA3 pattern-completion), and question-route (4-way lateral-inhibition WTA)
with SPIKING mechanisms by default instead of the host `set`/word-list/if-else shortcuts. Each was already 6-seed
focused-GO (load-bearing on + byte-identical off); the one gate remaining was the integrated no-regression soak,
which was RAM-blocked on the 46GB dev box and is now cleared on an AWS r7i.4xlarge (128GB, numpy).

## The gate that was blocking, now passed

`onebrain_regression_battery --flag <FLAG>` builds a fresh brain in the flag-ON arm and again in the flag-OFF arm
(same seed, identical background trajectory) and asserts every OTHER default-on faculty decides identically through
the REAL `webapp.server.brain_chat` handler — or names which regressed.

<!--derived-->
(values read directly from the cited artifacts, e.g.
`research/findings/raw/_regression_battery/battery_BRAIN_SPIKING_NOVELTY.json`,
`research/findings/raw/_regression_battery/battery_BRAIN_SPIKING_ANAPHOR.json`,
`research/findings/raw/_regression_battery/battery_BRAIN_SPIKING_QROUTE.json`.)

| flag | all_pass | faculties checked | regressed |
|---|---|---|---|
| BRAIN_SPIKING_NOVELTY | True | 38 | none |
| BRAIN_SPIKING_ANAPHOR | True | 38 | none |
| BRAIN_SPIKING_QROUTE | True | 38 | none |

Ran on the AWS r7i CPU instance (numpy; scipy present so the CPU path does not silently fall back to cupy); instance
launched, used, and TERMINATED (SG deleted, `.aws_gpu` cleared — no billing leak). ~$1-2, one-time.

## What changed (one line each, additive + reversible)

- `research/runners/spiking_novelty_habituation_organ.py:87` — `spiking_novelty_enabled()` default `"0"`->`"1"`.
- `research/runners/spiking_anaphor_detection_organ.py:87` — `spiking_anaphor_enabled()` default `"0"`->`"1"`.
- `research/runners/spiking_qroute_selection_organ.py:94` — `spiking_qroute_enabled()` default `"0"`->`"1"`.

Each keeps a **byte-identical `=0` opt-out** (verified: `BRAIN_SPIKING_NOVELTY=0` -> `enabled()` False), so the host
path is retained as an explicit oracle. The `*_LESION` flags are untouched.

## Honest scope — on_by_default, NOT (yet) the strict scaffold_retired counter

These flips make the three faculties **spiking-by-default** (`on_by_default: YES`). They do NOT increment the strict
`scaffold_retired` ledger counter, which (per this project's Check-D taxonomy + the 2026-08-26 wave-1/2/3 precedent)
requires the host scaffold to be genuinely removed/blocked, not merely default-off-able. The host `set`/word-list/
if-else code is retained here as the byte-identical opt-out, exactly as those prior flip waves retained theirs
(scaffold_retired stayed NO). So the substantive win is real — the brain does these by default now — while full
Check-D retirement (removing the host paths) is a named follow-on, not claimed here.

## B-curiosity (`_XEDGE_CD6`) — no-regression passed, flip DEFERRED (honest)

`BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6` ALSO returned `all_pass: True` (0/38 regressed). But its flip is NOT taken,
because the no-regression ran at the code's default `train_drive_scale=1.0` — which is the **read-isolation-corrected
NO-GO 3/6 calibration** that `_XEDGE_CD6_DEFAULT_ON=False` exists to respond to. B's load-bearing GO (6/6) was at
`train_drive_scale=1.5`, and that calibration is NOT the production default on main. So enabling B now would ship the
NO-GO calibration. Flipping B correctly needs the 1.5 retune ported to the module default AND a re-verify (both
no-regression and load-bearing) at 1.5 — banked as the next rung, not done here.

Functional read-outs only; no phenomenal-experience claim.
