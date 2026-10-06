---
type: finding
status: live
date: 2026-10-02
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/host_share_bg_seam.py
artifacts:
  - research/findings/raw/_bg_live_seam/gate/score.json
---

# BG live-seam retirement: latch read is not threshold-invariant (G4); the host read still owns part of the commit

Artifact: research/findings/raw/_bg_live_seam/gate/score.json (label GATE; verdict NO-GO; status NO-GO; six-seed gate 0bc62cc31). Preregistered gates scored by the registered scorer; verdict and gates read from score.json, not recomputed.

## Verdict sentence (HANDOFF section 7)

latch read is not threshold-invariant (G4); the host read still owns part of the commit.

## Pooled and n (copied from score.json)

| field | pooled | n |
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

Reported-only numbers: S 0.0417, D 0.9583, first12_change 0.1667, readout 0.0833 (n = 24 each). Diagnostics: seam_held_attribution -0.1537751522983202 (the control exceeds the treatment).

Production seam class: HOST-OWNED (replaceable) (lane C class: HOST-OWNED (replaceable)).

## The six gates

| gate | holds |
|---|---|
| G1 production host owns the reply | true |
| G2a neural seam holds on empty turns | true |
| G2b neural seam never holds a content turn | true |
| G3 value lesion moves the reply | true |
| G4 latch read threshold-invariant | false |
| G5 bus argmax does no work | true |

G4 is the only gate that fails: the latch read changes on 0.25 of events when the threshold is varied, so the host read still owns part of the commit. All eight preconditions passed (sham null 1.0, membrane within range 1.0, event counts 24/12/6/24/24/12), so the verdict is a NO-GO, not an UNDEFINED.

## Fixed caveats

- wired, not learned (learned-content fraction 0).
- flags default OFF; flipping them is an owner decision.
- the organ is its own SimulationBridge, not the one-brain substrate.