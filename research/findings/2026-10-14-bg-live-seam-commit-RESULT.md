---
status: qualified
lane: load-bearing (the #1 metric, owner 2026-09-19)
date: 2026-10-14
type: finding
claim_check: measured (research.runners.host_share_bg_seam --score over the 42 landed gate shards at P=3e7a41d97818ad222adf1f05df83aac6db795883, seeds 42/43/44/100/101/102; score.json verdict "NO-GO" with gated keys G2a_neural_seam_holds_on_empty_turns=false and G3_value_lesion_moves_the_reply=false; G1, G2b, G4_commit_read_threshold_invariant, G5 true)
mechanism: BG live seam + spiking commit read (amendment 2026-10-14-bg-live-seam-commit) behind BRAIN_BG_COMMIT, six gate seeds
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_bg_live_seam/commit_gate/3e7a41d/score.json
  - research/findings/2026-10-14-bg-live-seam-commit-AMENDMENT.md
  - research/findings/2026-09-29-bg-live-seam-retirement-PREREGISTRATION.md
verdict: PROVISIONAL NO-GO. Six of seven keys as specified: G1, G2b, G4_commit_read_threshold_invariant (and the
reported-only G4_old_latch_read_threshold_invariant) true; G2a_neural_seam_holds_on_empty_turns false (empty_held
0.6667) and G3_value_lesion_moves_the_reply false (seam_lesion_held 0.2917 < 0.3 bar). <!--derived-->
---

# BG live seam + commit read at P=3e7a41d97 (amendment 2026-10-14-bg-live-seam-commit) -- PROVISIONAL NO-GO

Pre-registration: `research/findings/2026-09-29-bg-live-seam-retirement-PREREGISTRATION.md`, amended by
`research/findings/2026-10-14-bg-live-seam-commit-AMENDMENT.md`. Packet branch `research/bg-commit-gate`, pins
`docs/drafts/bg-commit/PINS.tsv` row P = 3e7a41d97818ad222adf1f05df83aac6db795883. Verdict and gates read from
the score artifact `research/findings/raw/_bg_live_seam/commit_gate/3e7a41d/score.json`; never recomputed.

## The two failed gated keys, with the amendment's sentence for each (Outcome -> verdict, verbatim)

- **G2a_neural_seam_holds_on_empty_turns = false.** "The commit read is invariant, but the seam's race does not
  hold empty turns or spares no question; the capability, not the read, fails."
- **G3_value_lesion_moves_the_reply = false.** "The reply does not depend on the value route through the commit
  stage often enough."

## gates (copy of score.json)

| key | value |
|---|---|
| G1_production_host_owns_the_reply | true |
| G2a_neural_seam_holds_on_empty_turns | false |
| G2b_neural_seam_never_holds_a_content_turn | true |
| G3_value_lesion_moves_the_reply | false |
| G4_commit_read_threshold_invariant | true |
| G5_bus_argmax_does_no_work | true |
| G4_old_latch_read_threshold_invariant (reported, not gated) | true |

## evidence

Empty-turn seam hold (G2a bar): 8/12 events held, pooled empty_held 0.6667. Per seed hsb_empty1/hsb_empty2
held: 42 (T/F), 43 (T/F), 44 (T/T), 100 (T/F), 101 (F/T), 102 (T/T).

Value-lesion reply change (G3 bar 0.3 per the amendment): seam_lesion_held on 7/24 events, pooled 0.2917 --
7 of 24 lesioned events still held the reply (hsb_empty1 on 42/43/100/102, hsb_empty2 on 44, hsb_hi on 43/102).

Other pooled numbers reported by score.json: F 0.875, S 0.0417, D 0.9583, content_held 0.0, old_read_agree
0.9167, co_commit 0.0, first12_change 0.2917, latch_thr_change 0.0417, commit_thr_change 0.0, commit_engaged 1.0,
sham_ok 1.0, membrane_ok 1.0, bus_unique_agree 1.0. production_seam_class: HOST-OWNED (replaceable).

All preconditions in score.json are ok=true (sham null on every target event, membrane range, event count, commit
op/organ seed, pythonhashseed). Canaries: host_share_bg_seam@3e7a41d97 PASS gate-grade (pool41, rc=0, wall=4409s).

## What this NO-GO does and does not say

The commit read is threshold-invariant (G4 true, old latch read agree too), the host owns the reply (G1), the
seam never holds a content turn (G2b) and the bus argmax does no work (G5): the read and its controls behave as
specified. The failure is the seam race holding empty turns often enough (G2a) and the value route's lesion moving
the reply often enough (G3) -- capability, not read. Nothing here is a verdict on a flip: no flag default was
touched (`BRAIN_BG_COMMIT`, `BRAIN_BG_NEURAL_SEAM`, `BRAIN_GNW_UNIQUE_IGNITION_READ` unchanged, per the HANDOFF).