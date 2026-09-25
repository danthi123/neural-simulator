---
type: finding
status: verified
date: 2026-09-17
mechanism: test the one named residual of the learned spiking constituent-boundary segmenter (GO 5/6) — that a
  deterministic verb->object pair (high transitional probability) has its boundary spike suppressed (Benjamin 2021, TP
  necessary-but-not-sufficient) — by (1) adding a verb->object-specific boundary diagnostic and sizing the residual
  across 6 seeds at baseline, and (2) a candidate lever: learned per-pre lateral competition (Harris successor-variety
  read off the STDP-trained stim->pred weights) that injects a residual boundary current where a token has multiple
  distinct predecessors
integration_faculty: language (comprehension — constituent/SVO boundary segmentation)
lane: language (comprehension)
seeds: [42, 43, 44, 100, 101, 102]
verdict: NO-GO on the lever, and the RESIDUAL IT TARGETED IS ESSENTIALLY NON-EXISTENT. The 6-seed OFF baseline
  verb->object boundary AUC is high on every seed (numbers in the body — well above the 0.85 gate and chance), so the
  base segmenter ALREADY detects verb->object boundaries; the Benjamin-2021 TP-invisibility concern is a theoretical one
  that does not bite on this data (the minimal function-word cue + the multi-predecessor structure already carry those
  boundaries). The learned lateral-competition lever HURTS: it raises verb->object recall but tanks the AUC
  monotonically with gain (collapsing to chance at the highest gain) by injecting indiscriminate boundary current
  (precision loss). Anti-cheats clean (no-learning and stream-scramble verb->object AUC both at chance, ON and OFF), so
  the diagnostic is a valid instrument. Net: the lever is banked NO-GO, and the finding
  STRENGTHENS the segmentation GO 5/6 — its one named caveat is not a real limitation. Additive, --seg-vo-competition
  default-OFF (byte-identical when off), wires nothing.
runner: research/runners/_learned_spiking_segmentation_derisk.py (--seg-vo-competition)
artifacts:
  - research/findings/raw/_learned_spiking_segmentation/verify_6seed_vo_baseline.json
external: NO-EXTERNAL-NEEDED — a residual-sizing + lever test on the project's own already-biology-bound segmenter
  (Benjamin 2021's TP-necessary-not-sufficient claim is the motivation being TESTED); the honest negative (the residual
  does not bite + the lever hurts) is the deliverable.
builds_on:
  - research/findings/2026-09-17-learned-spiking-constituent-boundary-segmentation-retires-host-segment-clause-GO.md
---

# Segmentation verb->object residual is essentially non-existent; the lateral-competition lever is NO-GO

The learned spiking constituent-boundary segmenter landed GO 5/6 with ONE named honest residual: a deterministic
verb->object pair (high transitional probability) could have its boundary spike suppressed by the predictive-coding
inhibition (Benjamin 2021 — pure TP is necessary but not sufficient). This sizes that residual and tests a lever.

## The residual, sized (from research/findings/raw/_learned_spiking_segmentation/verify_6seed_vo_baseline.json)
A new verb->object-only boundary diagnostic (verb->object transitions vs within-constituent negatives; sv-boundaries
excluded to isolate it), 6-seed OFF baseline:
- heldout_vo_boundary_auc per seed: 0.976, 0.976, 0.997, 0.971, 0.899, 0.972 (mean ~0.965, min ~0.899) — all well <!--derived-->
  above the 0.85 boundary gate and far above chance 0.5.
- overall heldout_boundary_auc per seed stays 0.935-0.998 (the banked GO, reproduced). <!--derived-->
So the base segmenter ALREADY detects verb->object boundaries on every seed. The named TP-invisibility residual is a
theoretical concern that does not materialize on this corpus.

## The lever, tested (seed-42 explore — single-seed, exploratory, NOT verdict-bearing)
Learned per-pre lateral competition (Harris successor-variety off the STDP weights), seed-42 gain sweep vs OFF (a
single-seed EXPLORE — no verdict is asserted on it; the pre-registered rule requires beating baseline on the explore
before any 6-seed spend, and it does not):
- OFF vo_auc 0.976 -> ON gain 20: 0.818, gain 45: 0.659, gain 90: 0.500 (collapse). vo_recall rises to 1.0 while AUC <!--derived-->
  falls: the bonus current adds boundary spikes indiscriminately (precision loss), it does not selectively recover a
  missed boundary because there is no missed boundary to recover. No 6-seed ON spent (fails the seed-42 explore).
- Anti-cheats: no-learning and stream-scramble verb->object AUC both 0.5 (ON and OFF) — the diagnostic is a valid
  instrument, so the lever's harm is real, not an artifact.

## What it means

Two clean outcomes: (1) the segmentation GO 5/6 is STRONGER than its own caveat implied — its one named residual
(verb->object TP-invisibility) does not bite on this data (base vo_auc 0.90-1.00 all seeds); (2) the learned
lateral-competition lever is a banked NO-GO — it hurts by adding indiscriminate boundary current when there is nothing
to fix. `--seg-vo-competition` stays additive/default-off. The comprehension segmentation path needs no verb->object
patch; the genuine open items elsewhere (the onebrain flip's curiosity-calibration fix, the supervised S2-template
lever) are where the frontier is. A wall was checked and found not to be a wall.
