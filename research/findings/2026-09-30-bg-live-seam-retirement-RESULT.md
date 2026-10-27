---
status: live
date: 2026-10-27
type: finding
runner: research/runners/host_share_bg_seam.py
claim_check: measured (research.runners.host_share_bg_seam --score over the 42 landed gate shards at P=0bc62cc318278839a77293358a6fa9ad9e36a516, seeds 42/43/44/100/101/102; score.json verdict "NO-GO" with G4_latch_read_threshold_invariant=false; G1, G2a, G2b, G3, G5 true; production_seam_class HOST-OWNED (replaceable))
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_bg_live_seam/gate/score.json
verdict: NO-GO. G4_latch_read_threshold_invariant false: latch read is not threshold-invariant; the host
read still owns part of the commit. G1, G2a, G2b, G3, G5 true.
---

# BG live seam retirement: NO-GO — the latch read is not threshold-invariant (G4); the host read still owns part of the commit

Six-seed gate (42/42 shards, seeds 42/43/44/100/101/102) scored per the HANDOFF Step 6 command from
`research/findings/raw/_bg_live_seam/gate/score.json`; verdict read from `score.json`, not recomputed.

## Gates
| gate | value |
|---|---|
| G1_production_host_owns_the_reply | true |
| G2a_neural_seam_holds_on_empty_turns | true |
| G2b_neural_seam_never_holds_a_content_turn | true |
| G3_value_lesion_moves_the_reply | true |
| G4_latch_read_threshold_invariant | false |
| G5_bus_argmax_does_no_work | true |

## Pooled and n
| key | pooled | n |
|---|---|---|
| F | 0.875 | 24 |
| S | 0.0417 | 24 |
| D | 0.9583 | 24 |
| cap | 1.0 | 18 |
| ns | 0.6667 | 24 |
| readout | 0.0833 | 24 |
| bus_disagree | 0.0 | 12 |
| latch_thr_change | 0.25 | 24 |
| first12_change | 0.1667 | 24 |
| membrane_ok | 1.0 | 72 |
| content_held | 0.0 | 6 |
| empty_held | 1.0 | 12 |
| sham_ok | 1.0 | 24 |
| bus_unique_agree | 1.0 | 12 |
| seam_intact_held_for_ns | 0.5417 | 24 |
| seam_lesion_held | 0.625 | 24 |

`production_seam_class`: HOST-OWNED (replaceable)

Reported-only numbers (not gated): S 0.0417, D 0.9583, first12_change 0.1667,
readout 0.0833.

## Caveats (fixed, per HANDOFF Step 8)
- wired, not learned (learned-content fraction 0)
- flags default OFF; flipping them is an owner decision
- the organ is its own SimulationBridge, not the one-brain substrate
