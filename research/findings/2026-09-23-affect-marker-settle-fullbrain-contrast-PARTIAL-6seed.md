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
ac02c209d). CORRECTION (merge review): the "only s100 may read OFF load-bearing" identity rule (2b238b74) was written
into the TEXT of the ac02c209d pre-registration, but the CODE enforcing it was added after the superseded rev-56e588d rows
had been seen by the fix agent (OFF s42/s43 not load-bearing; ON s42/s43/s101 load-bearing) — so "before any row was
read" is not accurate. The rule only makes the gate stricter (it can remove a GO, never add one), and no re-staged row
had been read when it landed.

## Result — pre-registered verdict: PARTIAL (NO-GO for the flag)
<!--derived from research/findings/raw/_affect_marker_settle/fullbrain_contrast_verdict.json -->
Artifact: `research/findings/raw/_affect_marker_settle/fullbrain_contrast_verdict.json` (per-seed rows under
`lbf_on/` and `lbf_off/`; s42 ran on pool41 and its sidecar records git_sha 56f1abf5, git_dirty=False; s43-s102 ran on
AWS from a `git archive` of 56f1abf5 deployed without a .git directory, so THEIR sidecars record git_sha 'unknown' —
their revision is asserted by the deploy procedure, not by provenance).

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
difference on two seeds (42, 43). Whether those are specifically the register-boundary seeds of the op-level diagnosis
is NOT established by this run (the full-brain rows do not record the mood read's distance to a boundary); it is a
hypothesis consistent with, not demonstrated by, these data.

## Interpretation (honest)
- The op-level prediction (OFF load-bearing on 1/6) did NOT transfer to the full brain, where the marker already drives
  the reply on 4/6 without the flag. The pre-registered gate was written for that prediction, so it reads PARTIAL:
  the flag is not *generally* necessary, and it is reported as a NO-GO for the flag's attribution claim.
- Descriptively, SETTLE rescues two seeds (42, 43) and changes nothing measurable on the four that already worked — the
  behaviour a stabilizer should have (boundary attribution unverified, see above). Under the adequate probe with SETTLE opt-in the
  affect-marker is load-bearing 6/6 — measured in THIS contrast run, not yet inside a combined battery with every
  other fix flag (SETTLE could in principle interact with other faculties; unmeasured).
- Option-C: this does not change the shipped default (SETTLE is default-off; flipping it is owner-reserved).

## Host shortcuts (unchanged, declared)
S1 `_select()` np.argsort over pool rates + DEAD_MARGIN names the winner; S2 the drive is a host Gaussian population
code of the mood float. The host sets only the clock (deliberation window, rest).

## Honesty
Functional read-out only — the affect-marker lead ("Gladly! ...") provably depends on the spiking WTA's choice; no
felt/phenomenal claim.
