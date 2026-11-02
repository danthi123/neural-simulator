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
- [ ] New core, weekly conversation test upgrade (`brain/eval/weekly.py`, milestone day 14, board #281). Today it
  sums the intact and lesion arms into one total, asks the same fixed questions every week, and credits degenerate
  replies (e.g. "Nova, Nova."). Change it so that: (1) the header prints one score line per arm (intact, hippocampus
  off) and the difference; (2) a third arm runs sleep between the two sessions (`brain/sleep.py`, adapters from
  `brain/plastic_cortex.py`) and then asks with the hippocampus OFF, so the transcript shows what sleep consolidated;
  (3) half the facts are drawn each week from a larger list, seeded by the ISO week, so the score cannot improve by
  memorising fixed questions (keep the other half fixed for week-to-week comparison); (4) a reply counts only if it
  contains the key word AND is not degenerate (repeated token runs, empty, or only the brain's own name). Add unit
  tests that run without the cortex weights (mock `Cortex`); real runs go to W via `run_gpu`. Read first:
  `brain/eval/weekly.py`, `brain/experiments/sleep_probe.py`, `brain/PREREQUISITES.md` (Sleep consolidation).
  Done = branch + tests passing + one real run on W attached to a merge_review checkpoint.
