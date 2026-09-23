---
type: finding
status: live
lane: load-bearing
date: 2026-09-23
verdict: NO-GO
---

# Affect-marker SETTLE full-brain contrast: load-bearing 6/6 WITH the flag, but SETTLE-attributable on only 2/6 → PARTIAL (NO-GO for the flag as generally necessary) (2026-09-23)

The full-brain verification of `BRAIN_AFFECT_MARKER_SETTLE` (the deliberation-window + inter-turn-rest companion
processes from `2026-09-23-affect-marker-settle-deliberation-and-rest-oplevel-go-6seed.md`), scored by the
PRE-REGISTERED contrast gate (`research/runners/_affect_marker_settle_derisk.py --score-fullbrain`, registered in
ac02c209d; the "only s100 may read OFF load-bearing" identity rule was enforced in 2b238b74 before any row was read).

## Result — pre-registered verdict: PARTIAL (NO-GO for the flag)
<!--derived from research/findings/raw/_affect_marker_settle/fullbrain_contrast_verdict.json -->
Artifact: `research/findings/raw/_affect_marker_settle/fullbrain_contrast_verdict.json` (per-seed rows under
`lbf_on/` and `lbf_off/`; s43-s102 ran on AWS in parallel, s42 on pool41; all at revision 56f1abf5).

| seed | ON (SETTLE=1) load-bearing | OFF load-bearing | SETTLE credited |
|---|---|---|---|
| 42 | yes | no | yes |
| 43 | yes | no | yes |
| 44 | yes | yes | no |
| 100 | yes | yes | no |
| 101 | yes | yes | no |
| 102 | yes | yes | no |

Every row is valid (6/6 pairs: deterministic, null-control clean, env read back correctly). With SETTLE on, the
affect-marker is lesion-verified load-bearing on **6/6** seeds. With SETTLE off it is load-bearing on **4/6** (the same
4/6 the consistent all-fixes battery `allfixes2` measured without the flag). SETTLE is therefore responsible for the
difference on exactly the two seeds (42, 43) whose mood read sits on a register boundary — where the op-level diagnosis
said the 60 ms race could not resolve.

## Interpretation (honest)
- The op-level prediction (OFF load-bearing on 1/6) did NOT transfer to the full brain, where the marker already drives
  the reply on 4/6 without the flag. The pre-registered gate was written for that prediction, so it reads PARTIAL:
  the flag is not *generally* necessary, and it is reported as a NO-GO for the flag's attribution claim.
- Descriptively, the companion processes do exactly what a stabilizer should: they rescue the two boundary seeds and
  change nothing measurable on the four that already worked. Under the adequate probe with SETTLE opt-in the
  affect-marker is load-bearing 6/6 — measured in THIS contrast run, not yet inside a combined battery with every
  other fix flag (SETTLE could in principle interact with other faculties; unmeasured).
- Option-C: this does not change the shipped default (SETTLE is default-off; flipping it is owner-reserved).

## Host shortcuts (unchanged, declared)
S1 `_select()` np.argsort over pool rates + DEAD_MARGIN names the winner; S2 the drive is a host Gaussian population
code of the mood float. The host sets only the clock (deliberation window, rest).

## Honesty
Functional read-out only — the affect-marker lead ("Gladly! ...") provably depends on the spiking WTA's choice; no
felt/phenomenal claim.
