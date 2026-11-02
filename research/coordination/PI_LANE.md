# Pi lane (pilot from 2026-10-31; lean method from 2026-11-02)

The supervisor feeds each `- [ ] ` line below to Pi with `tools/pi_driver/LANE_PROMPT.md`. Only Claude adds items.
The next item comes from `HANDOFF.md` or the board, as a build task for the new core (`brain/ARCHITECTURE.md`
section 4 milestones). Until Claude adds one, the lane is open and empty, and Pi waits.

## Archived (#203, frozen synapses; old track, closed by RULING pivot-1102)
Full text and evidence: `git show 6b4ff08e5:research/coordination/PI_LANE.md` (main) and the lane branch
`research/pi-lane` (its copy of this file holds the third item and the flip-smoke record).
- [x] #203 freeze: non-plastic synapses no longer drift, flag `cfg.enforce_plastic_mask_beyond_hebbian`, default OFF.
  Branch `research/pi-203-frozen-synapse-freeze` @ `fd42e0120`, merge_review APPROVED (2026-10-31).
- [x] #203 follow-up: the comprehension organ's frozen drift is the Hebbian loop (`sim/bridge.py` LTP + decay); fix
  on branch `research/pi-203-write-path-attribution`, default OFF, no reply-level change (2026-10-31).
- [x] #203 flip smoke (on `research/pi-lane`): finish only the GPU cells already in flight, land them, then stop
  (RULING pivot-1102). No flip, no new #203 work.

## Open
