# HANDOFF packet: xedge-surprise-worldmodel (integration wave 1)

Run the shared flow in `../HANDOFF.md`; this file holds what is specific to this packet. Prereg:
`research/findings/2026-09-29-iw1-xedge-surprise-worldmodel-flip-PREREGISTRATION.md`.

**What flips.** The existing anchor `_XEDGE_SWM_DEFAULT_ON` in
`research/runners/onebrain_xedge_surprise_worldmodel_production.py` turns on
`BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL` (cross-edge C1): on a turn whose valence violates the affective forward
model's prediction, the model's own D2 prediction-error unit gates one Hebbian co-fire step of the live state->pred
transition toward what was observed. The model stops being frozen after build. Recorded as a sub-rung of ledger row
`worldmodel-forward` (no new row). Escape after the flip: `BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL=0`.

**Why it was chosen.** De-risk GO 6/6; wired default-OFF with a real-handler flip-verify (byte-identical off, gated
growth on real traffic, none under the lesion, 0 of 38 faculties regressed; seed 42); its own finding "earns the
flip's readiness". A cross-edge is the one-brain non-negotiable made concrete. It has never been measured at the
reply: step R is the first reply-level test, and it can fail honestly.

## Step R for this packet (shared HANDOFF sections 3-5)

Job file `jobs/R_swm_<P7>.txt` (6 lines), tag `iw1-R-swm-<P7>`, row `xedge-surprise-worldmodel`: a positive turn,
eight alternating negative/positive turns, then "what do you expect?" (the reply states the predicted sign and the
pool-rate margin). Lesion = the flag `=0` (whether-disable). Score: `tools/iw1_score.py R --packet
xedge-surprise-worldmodel ...`.

| R verdict | per-seed detail | do |
|---|---|---|
| PASS | 6/6 DEFINED, S2 `regressed` | into the flip list (if D0 and D_xedge-surprise-worldmodel pass) |
| FAIL | a DEFINED seed reads `pass` (the learning did not reach the reply on that seed) | not flipped. Draft `research/findings/<date>-iw1-xedge-surprise-worldmodel-reply-level-NO-GO-<n>of6.md` (provisional): per seed, the two expectation replies from the arm files, and "trace-level only on this probe" |
| UNDEFINED | a seed not DEFINED, or `UNDEFINED:*` | shared re-run rule once; still UNDEFINED: not flipped, NEEDS-CLAUDE "iw1 swm R UNDEFINED: <seed>: <fails>" |

## Production default (shared steps B and F)

F1: in `iw1-flip-<F7>` the row `xedge-surprise-worldmodel` must read reply-level load-bearing on 6/6.

## Finding template (after the shared section 7 reads PASS)

`research/findings/<date>-iw1-xedge-surprise-worldmodel-default-on-flip-validated-PROVISIONAL.md`, frontmatter
`integration_faculty: worldmodel-forward`, `status: provisional`; sections as in the d6 packet's template, plus the
process-shared caveat from the prereg copied verbatim.

## Dev seed 7 (Claude, 2026-09-29, local numpy, worktree at the packet commit)

SWM_DEV_PLACEHOLDER

## NEEDS CLAUDE (packet-specific)

1. Before the merge: the organ is process-shared (`_get_worldmodel_organ()` has no session key), so default-on means
   the world model learns across every conversation one server process serves. Accept, or require a per-session organ
   first. No step here measures cross-session accumulation.
