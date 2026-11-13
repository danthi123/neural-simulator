# HANDOFF (overwrite each session; <= 40 lines)

- **TASK:** Sleep consolidation: told facts recalled from the cortex's own synapses after sleep, with the hippocampus
  off. Board #279 (and #280 high-use facts, #281 a lesion visibly changes replies).
- **TYPE:** build
- **DRIVER:** claude
- **STARTED:** 2026-11-02
- **TIMEBOX:** 2 days per problem (first decision point 2026-11-04: KEEP or KILL the text-replay method)
- **DONE MEANS:** six seeds, hippocampus off after sleep: true-replay recall above the no-sleep and shuffled arms on
  every seed, ordinary-text drift small, false recall not above shuffled; then sleep wired into the live brain
  (restart = sleep, adapters persisted) so the weekly transcript shows it.
- **MILESTONE:** day 14 (2026-11-16), brain/ARCHITECTURE.md §4. Day 7 (2026-11-09): first [weekly] run by Hermes.
- **STATE (11-13):** confidence gate at 1.5B, six seeds, AWS (`brain/experiments/results/2026-11-13-confidence-gate-1.5b.md`):
  the slept cortex's reply log-prob predicts its own errors (AUC 0.77), and gating reinstatement on it removes the
  cost (h 9.17 vs always-reinstate 7.67 of 12) but is NOT better than hippocampus-off after sleep (c 9.33). Registered
  G0-G4 held; three skeptics NARROWED it (bars compared only to f). No default changed. Weekly 11-13 ran on AWS
  (0.1B): after restart hippocampus-off 4/6, intact 2/6. Earlier: hippo-route NOT A LEAD (11-05, F3).
- **LIVE RUNS:** AWS mixed-day test (pre-registration `brain/research/2026-11-13-mixed-day-gate-1.5b.md`, jobs
  `brain/experiments/jobs/2026-11-13-aws-mixed-day.txt`); `bash tools/aws_brain_batch.sh status|pull`. W is dedicated to the owner's other work (owner 11-07/11-13): no W jobs (RULING
  w-dedicated-1113); only the mini-PCs and AWS (cap ~$50/day).
- **READY WORK:** AWS: none beyond the live run. Mini-PCs: none listed. Pi: `research/coordination/PI_LANE.md` (tautology scorer item, on
  `research/pi-lane-1105`, unmerged; land with a merge_review when convenient).
- **NEXT STEP:** pull + table the mixed-day run (`mixed_day_table`), skeptics, land; if LEAD (h beats c and f), build the gate in Brain (default
  OFF) and run it live, then the lesson/answer arbitration on the same signal.
- **DELEGATED:** Pi: lane items. Hermes: [weekly] is Claude-on-AWS while W is dedicated.
- **OWNER WORDS (2026-11-02, verbatim):** "Do all, ensure Hermes and Pi also follow our new methodology. And one
  caveat, I may not noticeably correct you so I don't want that as a benchmark of improvement. All I really care
  about is us making meaningful progress on this project, which has stalled for way too long as I continue to spend
  money on AI to drive it."
- **SPEED LEVERS (built):** AWS lane `tools/aws_brain_batch.sh`; Pi's Claude gated by live plan usage (`usage.ts`).
- **OPEN:** none blocking. Chores: repoint the board timer to a main-following checkout; pool40 host key (owner's call).
