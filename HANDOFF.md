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
- **REORIENTED (owner 11-15):** "question whether you're ultimately building something that is focused on passing the
  tests you wrote rather than the broad objective". Audit: yes, partly; whole transcripts show told facts INTRUDING
  ("How are you feeling today?" -> "I drive a red Volvo."), hippocampus-off converses better. New primary measure:
  `brain/eval/open_conv.py` (held-out bank, new conversation per seed; recall / false / intrusions / needless recalls;
  read the transcripts). Stop polishing the scripted name line.
- **LIVE RUNS:** (1) pool40/41/42: own-cue frame + teacher gate (`~/brain-lane/ocf/`, prereg
  `research/2026-11-15-own-cue-frame-1.5b.md`), THEN open-conv baseline seeds 1100-1105 (`~/brain-lane/oc/oc_s*.md`,
  2 per node, CPU, hours). 0.1B smoke `~/brain-lane/oc/smoke_s7.md` on pool40. (2) AWS stage-1 S_all slept-bar fit
  (`jobs/2026-11-15-aws-sall-slept-bar-fit.txt`, from THIS worktree's `.aws_brain`; pull -> `sall_live_bar_fit fit`;
  low priority now: bank the result, judge any use of it on open_conv, not the weekly line).
- **POOL:** pool40/41/42 run brain/ jobs from `~/brain-lane` (venv `~/brain-lane/venv` on 40/41, `~/derisk-pool/sim/.venv`
  on 42; pattern `brain/experiments/jobs/2026-11-14-pool-*.sh`). pool40 host key re-trusted 11-14 (reinstall).
- **READY WORK:** answer-rate layout (`tools/answer_rate.py`). Pi: `PI_LANE.md`.
- **NEXT STEP:** read the open-conv baseline transcripts; then the first fix judged on open_conv: WHEN to recall
  (S0 question bar 0.12 lets almost any "?" pull an episode; recalled content then takes over the reply). Also run
  open_conv on AWS (6 seeds, ~1 h) for a GPU baseline. Bank the ocf/tg and S_all fit results briefly.
- **DELEGATED:** Pi: lane items. Hermes: [weekly] is Claude-on-AWS while W is dedicated.
- **OWNER WORDS (2026-11-02, verbatim):** "Do all, ensure Hermes and Pi also follow our new methodology. And one
  caveat, I may not noticeably correct you so I don't want that as a benchmark of improvement. All I really care
  about is us making meaningful progress on this project, which has stalled for way too long as I continue to spend
  money on AI to drive it."
- **OPEN:** none blocking. Chores: board timer -> main-following checkout; brain/ lane JSONs stamp no host/sha.
