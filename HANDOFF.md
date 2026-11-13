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
- **STATE (11-13):** mixed-day gate at 1.5B (`brain/experiments/results/2026-11-13-mixed-day-1.5b.md`): registered
  NOT A LEAD; a skeptic found the scorer credited general statements ("Oslo is the capital of Norway.") as answers.
  Scorer fixed (`strict_scorer.is_generic`); re-scored: mixed day VOID (hippocampus answers 0.83 of 6 new facts),
  11-13 f 7.67 -> 5.67, 11-05 g (routing fix, no sleep) 4.83 -> 0.50 = b. At 1.5B all told-fact answers come from
  sleep (c 9.33); reinstatement surfaces the right word but the cortex talks about it instead of answering.
- **LIVE RUNS:** none. W is dedicated to the owner's other work (owner 11-07/11-13): no W jobs (RULING
  w-dedicated-1113); only the mini-PCs and AWS (cap ~$50/day).
- **READY WORK:** CPU/mini-PC: re-score the other `correct_answer` results (0.1B live-brain, told-gate, weekly
  transcripts) with `brain/experiments/rescore_told.py`. Pi: `research/coordination/PI_LANE.md` (tautology scorer
  item, on `research/pi-lane-1105`, unmerged; land with a merge_review when convenient).
- **NEXT STEP:** make the hippocampus make the cortex ANSWER a fact told today: refit the reinstatement gain on unslept
  1.5B reads with the fixed scorer (gain 4 and the S_all size bar were fitted with the old scorer), pre-registered
  against b/g re-scored (0.50 of 12); AWS. The confidence gate waits until then.
- **DELEGATED:** Pi: lane items. Hermes: [weekly] is Claude-on-AWS while W is dedicated.
- **OWNER WORDS (2026-11-02, verbatim):** "Do all, ensure Hermes and Pi also follow our new methodology. And one
  caveat, I may not noticeably correct you so I don't want that as a benchmark of improvement. All I really care
  about is us making meaningful progress on this project, which has stalled for way too long as I continue to spend
  money on AI to drive it."
- **SPEED LEVERS (built):** AWS lane `tools/aws_brain_batch.sh`; Pi's Claude gated by live plan usage (`usage.ts`).
- **OPEN:** none blocking. Chores: repoint the board timer to a main-following checkout; pool40 host key (owner's call).
