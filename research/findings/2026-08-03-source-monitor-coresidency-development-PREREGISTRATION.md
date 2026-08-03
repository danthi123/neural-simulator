---
type: preregistration
status: live
lane: laneC
date: 2026-08-03
---

# Source-monitor co-residency development gate: preregistration

**Filed 2026-08-03 before seeds 214, 215, or 310 were run.** Calibration used
only seeds 212 and 213. Held-out seeds 311, 312, and 313 remain locked.

## Prediction

The co-resident episode-to-source pathway will learn seen, heard, and
self-generated source associations from physical afferent activity across
three fresh network initializations. Recall from episode activity alone will
preserve the correct source margin and will causally drive aPFC and ACC.

## Fixed Development Seeds

Run exactly seeds `214`, `215`, and `310`. Do not tune between seeds and do not
exclude a failed seed. A crash is a failed run until its cause is shown to be
infrastructure rather than the mechanism.

## Fixed Criteria

All criteria must pass on every development seed:

1. All 12 calibration structural, learning, source-swap, mixed-source, lesion,
   and propagation checks remain true.
2. Seen, heard, and self-generated recall each have a winner-minus-runner-up
   source-rate margin of at least `0.15`.
3. The episode-to-source lesion accounts for at least `90%` of intact source
   activity.
4. The source-to-ACC lesion accounts for at least `90%` of intact ACC activity
   while preserving at least `90%` of source activity.
5. An unexperienced, disjoint episode pattern produces exactly zero source
   spikes.
6. Learning-disabled weights remain exactly zero and produce exactly zero
   source spikes.
7. Inference still accepts episode activity only and exposes no source label,
   confidence scalar, answer, or response decision.

The overall verdict is GO only if all three seeds pass every criterion. Any
single failure is NO-GO and keeps held-out seeds locked.

## Interpretation

A GO would establish that this small learned source circuit is repeatable
enough to test in a minimal lived audio-visual episode loop. It would not prove
human source memory, language honesty, imagination monitoring, or open-ended
conversation. The explicit episode assembly, source-afferent identity, and
learning-window scaffolds would remain on the burn-down ledger.
