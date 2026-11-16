# HANDOFF (overwrite each session; <= 40 lines)

- **TASK:** Sleep consolidation: told facts recalled from the cortex's own synapses after sleep, with the hippocampus
  off. Board #279 (and #280 high-use facts, #281 a lesion visibly changes replies).
- **TYPE:** build
- **DRIVER:** claude
- **STARTED:** 2026-11-02 · **TIMEBOX:** 2 days per problem (first decision point 2026-11-04: KEEP or KILL the text-replay method)
- **DONE MEANS:** six seeds, hippocampus off after sleep: true-replay recall above the no-sleep and shuffled arms on
  every seed, ordinary-text drift small, false recall not above shuffled; then sleep wired into the live brain
  (restart = sleep, adapters persisted) so the weekly transcript shows it.
- **MILESTONE:** day 14 (2026-11-16), brain/ARCHITECTURE.md §4. Day 7 (2026-11-09): first [weekly] run by Hermes.
- **STATE (11-15):** lesion item MET (`2026-11-15-sleep-lesion-weekly-1.5b.md`). **`self_completion` now ON by default**
  (`results/2026-11-15-self-completion-replication-1.5b.md`: seeds 300-305 GO; day-1 own name 24/24 vs 13/24 pooled;
  caveat: "I have one too" self-claims 6 vs 16 of 192, p~0.29; weekly `--no-self-completion` = flag-off arm).
  After the night "What's your name?" still recalls the sister telling: S0 loses on 6/6 slept homes (S0 key is the
  layer-0 attention delta, the night cannot reach it); S_all ranks self first 6/6 but clears its 0.22 bar 2/6
  (`2026-11-15-self-recall-after-night-1.5b.md`). Self episodes keyed by their question: steal every question,
  K2 0/12, NOT A LEAD (`2026-11-15-self-key-question-1.5b.md`; S0 is form-dominated). Teacher-night gate: own cues
  0/12 (cortex writes them to the brain, one greedy draw), never-told clean 1/87: no build
  (`2026-11-15-teacher-night-gate-1.5b.md`). **(11-14):** gain 0.5 on for 1.5B; key default KEEP S0 (S_all +1 fact,
  short of the bar, `2026-11-14-key-default-1.5b.md`); echo levers banked NO-GO; near-miss guard NOT A LEAD.
- **LIVE RUNS:** none. AWS and pool idle (pool homes `~/brain-lane/workSC_{off,on}_s{200-205,306-311}` kept).
- **POOL:** pool40/41/42 run brain/ jobs from `~/brain-lane` (venv `~/brain-lane/venv` on 40/41, `~/derisk-pool/sim/.venv`
  on 42; pattern `brain/experiments/jobs/2026-11-14-pool-*.sh`). pool40 host key re-trusted 11-14 (reinstall).
- **READY WORK:** answer-rate layout (`tools/answer_rate.py`). Pi: `PI_LANE.md`.
- **NEXT STEP:** (a) own name after the night: refit the S_all bar on SLEPT live reads at 1.5B (extend
  `brain/experiments/sall_live_bar_fit.py`: size sleep defaults, GPU, self cues; fit on DEV bank seeds 42-44), then a
  six-seed weekly S_all vs S0 (prereg first). (b) false recall after sleep, teacher night: a better cue source (the
  user's own-voice question, cortex-written; no "your"->"my" template) before any build.
- **DELEGATED:** Pi: lane items. Hermes: [weekly] is Claude-on-AWS while W is dedicated.
- **OWNER WORDS (2026-11-02, verbatim):** "Do all, ensure Hermes and Pi also follow our new methodology. And one
  caveat, I may not noticeably correct you so I don't want that as a benchmark of improvement. All I really care
  about is us making meaningful progress on this project, which has stalled for way too long as I continue to spend
  money on AI to drive it."
- **OPEN:** none blocking. Chores: board timer -> main-following checkout; brain/ lane JSONs stamp no host/sha.
