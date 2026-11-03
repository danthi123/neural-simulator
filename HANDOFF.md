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
- **STATE (11-03 pm):** 1.5B live brain, six seeds (`results/2026-11-03-live-brain-1.5b.md`): hippocampus off after
  sleep 9.33/12 vs no-sleep 2.83 (6/6 seeds), shuffled 0; KEEP. Hippocampus routing diagnosed: S0 key misroutes at
  1.5B (7/12 forced top-1; S_all 11/12; live path 9 right/0 wrong at the 1.5B bar); recall with it on is 4.5.
  Told-gate FAIL (`results/2026-11-03-told-gate-1.5b.md`): when the not-known gate fires, the lesson overrides a
  recalled fact (told 5.33 -> 0.83/12); gate quiet = harmless. Gate accuracy on told questions (misfire rate under the
  fitted bar, unmeasured) is the deciding prerequisite, plus lesson/answer arbitration so a confident recall wins.
  M2 stays a LEAD for never-told questions, NOT a default. Weekly 1.5B preview (Claude-run, not Hermes) recorded
  (`results/2026-11-03-weekly-1.5b-preview.md`): hippocampus off after sleep 5/6; intact degenerates on reinstated
  episodes; new: told facts loop in free text after sleep, self-name only in the hippocampus. M3 FAIL at 0.4B.
  Brain defaults unchanged (all new switches OFF; default cortex still 0.1B).
- **LIVE RUNS:** AWS: hippo-route packet (`brain/experiments/jobs/2026-11-03-aws-hippo-route.txt`, S_all key + 1.5B
  bar, arms f/g, 6 seeds, ~2.5 h, dead-man 240 min). W GPU: idle. Watcher fix landed (0a4737825).
- **READY WORK (one next job per resource):** W GPU: measure the familiarity gate's misfire rate on told questions at
  1.5B, then intrusion/looping of told facts in free text after sleep. Mini-PCs: Pi-lane 12-fact rescoring item. Pi:
  `research/coordination/PI_LANE.md` open items. Claude: land the hippo-route result when AWS finishes.
- **NEXT STEP:** read the hippo-route result and fix routing at 1.5B; then gate accuracy + arbitration for the lesson.
  Merge `research/adopt-15b-steps-1103`, finish the 1.5B adoption plan; high-use facts and varied replay at 1.5B.
- **DELEGATED:** Pi: weekly test upgrade, then the varied-replay arm (`research/coordination/PI_LANE.md`). Hermes:
  [weekly] every Monday from 11-09; WORKLOG at most once a day (RULING overnight-1102).
- **OWNER WORDS (2026-11-02, verbatim):** "Do all, ensure Hermes and Pi also follow our new methodology. And one
  caveat, I may not noticeably correct you so I don't want that as a benchmark of improvement. All I really care
  about is us making meaningful progress on this project, which has stalled for way too long as I continue to spend
  money on AI to drive it."
- **SPEED LEVERS (built):** AWS lane `tools/aws_brain_batch.sh`; Pi's Claude gated by live plan usage (`usage.ts`).
- **OPEN:** none blocking. Chores: repoint the board timer to a main-following checkout; pool40 host key (owner's call).
