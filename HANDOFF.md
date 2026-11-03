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
- **STATE (11-03 afternoon, branch `claude/landing-1103c`):** 1.5B live brain, six seeds (`results/2026-11-03-live-brain-1.5b.md`):
  hippocampus off after sleep 9.33/12 vs no-sleep 2.83 (6/6 seeds), shuffled 0; keep/kill = KEEP (11-04 decision made early).
  Hippocampal retrieval MISROUTES at 1.5B (right episode for 6/12 told questions, same on every seed) and lowers recall
  to 4.5 with it on. False recall still open (a 4.0 vs b 3.0 of 4). M2 lived_self LEAD at 1.5B (212/216 DEV, 140/144
  fresh, 6/6 seeds), pending a told-facts-under-gate test (the gated sets hold only never-told questions). M3 retrieval
  practice FAIL at 0.4B (gain -2.67). Earlier: cortex size decided 1.5B; 0.1B negatives mostly scale-bound. Brain defaults
  unchanged (all new switches OFF; default cortex still 0.1B until adoption steps land).
  Routing DIAGNOSED (branch `claude/hippo-route-1103`, `brain/research/2026-11-03-hippo-routing-1.5b.md`): the layer-0
  key S0 no longer separates tellings at 1.5B (forced 7/12); S_all 11/12, and at a 1.5B dev-fit bar 9 right / 0 wrong;
  oracle routing answers 10/12, so the gain is fine. Fix `Brain(key="S_all", key_bar="size")` (default OFF), prereg F1-F5.
- **LIVE RUNS:** W GPU: weekly test at 1.5B (`--shared-sleep`, out `_scratch/brainv2/weekly15/`). AWS: none; packet
  READY, not launched: `brain/experiments/jobs/2026-11-03-aws-hippo-route.txt` (6 jobs, ~2.5 A10G-h). Last batch (
  20261103-1224 pulled, instance terminated). Watcher fix landed (0a4737825): idle alerts no longer silenced 6 h by an
  earlier alert.
- **READY WORK (one next job per resource):** W GPU: free for other work after the weekly test (routing probe done on CPU).
  AWS: hippo-route packet; told-facts-under-gate test for M2 lived_self (six seeds). Mini-PCs: Pi-lane 12-fact rescoring item. Pi:
  `research/coordination/PI_LANE.md` open items. Claude: read the weekly transcript, land results.
- **NEXT STEP:** launch the hippo-route packet, read F1-F5 (`hippo_route_live_table`). Then merge `research/adopt-15b-steps-1103` and finish the 1.5B adoption plan; high-use facts and
  varied replay at 1.5B.
- **DELEGATED:** Pi: weekly test upgrade, then the varied-replay arm (`research/coordination/PI_LANE.md`). Hermes:
  [weekly] every Monday from 11-09; WORKLOG at most once a day (RULING overnight-1102).
- **OWNER WORDS (2026-11-02, verbatim):** "Do all, ensure Hermes and Pi also follow our new methodology. And one
  caveat, I may not noticeably correct you so I don't want that as a benchmark of improvement. All I really care
  about is us making meaningful progress on this project, which has stalled for way too long as I continue to spend
  money on AI to drive it."
- **SPEED LEVERS (built):** AWS lane `tools/aws_brain_batch.sh`; Pi's Claude gated by live plan usage (`usage.ts`).
- **OPEN:** none blocking. Chores: repoint the board timer to a main-following checkout; pool40 host key (owner's call).
