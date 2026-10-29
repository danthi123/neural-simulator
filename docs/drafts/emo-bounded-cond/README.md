# emo-bounded-cond: drivers behind the bounded-conductance de-risk

These are the scripts that produced the raw outputs aggregated by `research/runners/emotion_category_bounded_aggregate.py`
(finding `research/findings/2026-10-26-emotion-organ-bounded-conductance-derisk.md`). They are copied verbatim from the
workstation scratch directory, so their paths (`/home/dant123/Projects/sim-rev/...`) are the workstation's and are not
portable; the raw per-seed outputs (about 20 MB) were not committed.

- `battery.py <flag 0|1> <tag> seeds...`: the 10-case organ battery (`emotion_category_dev.run_seed`) with the flag taken from the
  environment, plus a census of every simulation step of every bridge the seed builds and a log of each read's engagement fields.
- `stress.py <flag> inject|quench|pinned <tag> seeds...`: outlier-injection stress, quench-necessity lesion, pinned-failure replay.
- `launch_battery.sh`: 6 workers over seeds 301-360 and one over dev seeds 1-8. `handler_queue.sh`: the four serial seed-101
  Stage-2 emo shards (run from a scratch worktree at the Stage-2 pin with this branch's organ diff applied).
- `golden_main.py`: records the organ's reads from the unmodified file (main 011fa0036) for the byte-identity pin in
  `tests/test_emotion_category_bounded.py`. `overhead.py`: the cost of the census per step.
