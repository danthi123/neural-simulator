---
type: finding
status: contributing
date: 2026-09-16
mechanism: onebrain-xedge-curiosity-d6 base-rung (curiosity.ask -> d6.w0 cross-edge) default-ON flip
integration_faculty: onebrain-xedge-curiosity-d6
lane: memory + language (Live brain / chat)
seeds: [42, 43, 44, 100, 101, 102]
verdict: The onebrain curiosity->d6 cross-edge base rung, whose read-isolation-corrected NO-GO 3/6 had held it
  default-OFF since 2026-09-02, is FLIPPED DEFAULT-ON. Two gates cleared, both re-verified at the shipped config:
  (1) the faculty ITSELF reaches 6/6 GO at the train_drive_scale=1.5 retune through THIS module's own production
  wrapper self-test (not just the isolated runner), every seed lesion-attributable; (2) flipping it default-ON is
  answer-preserving in the integrated /api/brain-chat no-regression battery (all_pass, 0/38 faculties regress).
  4th of the 2026-09-16 default-on flip batch (after spiking novelty/anaphor/qroute, commit 1f444173).
runner: research/runners/onebrain_xedge_curiosity_d6_production.py
artifacts:
  - research/findings/raw/_onebrain_xedge_curiosity_d6_production_frozen_traindrivescale1.5_6seed.json
  - research/findings/raw/_regression_battery/battery_BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6.json
  - research/findings/raw/_regression_battery/arm_on_BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6.json
  - research/findings/raw/_regression_battery/arm_off_BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6.json
external: NO-EXTERNAL-NEEDED -- this is a production DEFAULT flip of an already-de-risked mechanism (the
  training-drive retune + the read-isolation fix are prior banked findings); no new mechanism or literature claim.
builds_on:
  - research/findings/2026-09-08-onebrain-crossedge-curiosity-to-d6wm-retuned-6-6-GO-via-training-drive-not-episode-count.md
  - research/findings/2026-09-02-onebrain-crossedge-curiosity-to-d6wm-read-isolation-fix-corrects-GO-to-NOGO-3-6.md
  - research/findings/2026-09-08-onebrain-curiosity-d6-semantic-drop-competitive-allocation-confound-CLOSED.md
---

# onebrain curiosity->d6 base rung flipped DEFAULT-ON (6/6 GO at the 1.5 retune + integrated 0/38 no-regression)

The 4th and last of the 2026-09-16 wiring-flip batch (the owner's #1 priority: FINISH the flips over more de-risk).
The base rung of the `onebrain-xedge-curiosity-d6` faculty — a live curiosity crave (this session's own recent
ASK-pool abstain read) measurably SUPPRESSING the rate of a d6 multi-referent-WM hold register through a frozen,
pre-grown spiking cross-edge — is now on by default (`_XEDGE_CD6_DEFAULT_ON` False->True). Its semantic-drop rung
(already default-ON in source but functionally inert behind the off base gate) becomes live with it.

## Why it was OFF, and why it is safe to turn ON now

It had been default-OFF since 2026-09-02: a read-isolation bug (an unrestored `_hard_reset` leaking C2 + NMDA-
recurrent state across the intact/lesion reads) had INFLATED the original GO 6/6 to a real NO-GO 3/6, and the flag
was correctly flipped OFF pending a real mechanism fix (a NO-GO faculty must not ship default-ON claiming to work).
That mechanism fix is the 2026-09-08 retune: INCREASING the training-time `ask` co-drive (not the episode count)
re-earns 6/6 GO with reads isolated.

## The two gates, both re-verified at the SHIPPED config (train_drive_scale=1.5)

(verdicts + counts read directly from the two cited raw artifacts, not from a run's stdout.)

1. **The faculty works at 1.5 through the PRODUCTION WRAPPER's own self-test** (not only the isolated de-risk
   runner). `research/findings/raw/_onebrain_xedge_curiosity_d6_production_frozen_traindrivescale1.5_6seed.json`:
   `train_drive_scale=1.5`, `n_go=6/6`. Every seed (42/43/44/100/101/102): `GO:True`, `lesion_attributable:True`,
   `frac_attributable_to_cross_edge=1.0`, `clears_registered_floor:True`, `no_signal_no_bias_ok:True`. The read is
   correct in SIGN and structure: holding a live crave shifts w0's held rate NEGATIVE (~ -0.01 to -0.017, <!--derived-->
   suppression) when `ask_held=True`, exactly 0.0 when `ask_held=False` (no signal, no bias), and collapses to 0.0
   under `BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6_LESION=1` (the cross-edge weight zeroed). Produced by the module's own
   official CLI (`--grow --semantic-drop --train-drive-scale 1.5 --seeds 42,43,44,100,101,102`).

2. **Flipping it default-ON is answer-preserving in the integrated system.** The /api/brain-chat no-regression
   battery ran on AWS r7i (128GB, numpy CPU path, scipy present so no silent cupy fallback), comparing the flag
   arm-on vs arm-off across the full production faculty set with the uncommitted flip live in the rsync'd code
   (source on the instance verified byte-matching the local edits before the run).
   `research/findings/raw/_regression_battery/battery_BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6.json`: `all_pass:True`,
   `n_faculties:38`, `n_regressed:0`, `regressed:[]` (26 probe turns incl. the `hold`/`held` D6 turns it fires on).

## Honest scope

This is a production DEFAULT flip, not a new capability. The load-bearing verdict is genuinely spiking (the frozen
cross-edge's own lesion-controlled weight drives both the w0 suppression and, via `apply_register_drive`, the
semantic-drop referent-set change). It stays `scaffold_retired:NO` / `retire_status:BLOCKED:neural-render`: the
reply-text qualifier is still a host f-string template and the crave-carryover is a coarse host binary — retiring
those needs the neural-render mouth + a continuously-decaying spiking crave, the named frontier. A same-landing
ledger correction also fixed a stale retire_status reason (it had claimed "gated behind still-off BRAIN_OPEN_ENDED
-> zero production execution"; in fact the branch is in the default strict path and the battery exercised it).

Escapes preserved: `BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6=0` (base rung opt-out, reproduces the read-isolation-
corrected calibration for an A/B); `BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6_SEMANTIC_DROP=0` (semantic-drop rung opt-out).
