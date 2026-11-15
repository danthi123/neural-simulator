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
- **STATE (11-15):** lesion item MET (`2026-11-15-sleep-lesion-weekly-1.5b.md`). **(11-14):** 1.5B answers facts told today before sleep: gain 0.5 flipped on for 1.5B
  (`results/2026-11-13-live-gain-1.5b.md`, GO; 0.1B keeps 4). Key default KEEP S0 (`2026-11-14-key-default-1.5b.md`).
  Echo: speaker contrast d + 3(d - d_self) works (oracle LEAD; live S0 NO-GO, routing-bound; live S_all GO NARROWED,
  `2026-11-14-sall-contrast-1.5b.md`: told 5.83 -> 6.83, one seed carries it, +2 false;
  report-only seeds 200-205: told +0.50, para +1.17, the aunt -> Oslo lure 6/6 vs 2/6).
  Mixed day with sleep: contrast NO-GO (`2026-11-14-sall-contrast-mixed-1.5b.md`, S_all told +0.00, S0 -1.17): echo
  levers all banked NO-GO in the live brain; package stays S0 at gain 0.5.
  Near-miss / frame guard in sleep (M4): NOT A LEAD (N1-N3 fail; switch stays OFF).
  Scorer: disclaimer + general talk no longer credited (`2026-11-14-disclaimer-rescore.md`); no verdict changed.
- **LIVE RUNS:** (1) AWS self-completion replication, seeds 300-305 (12 jobs, from THIS worktree's `.aws_brain`) ->
  `bash tools/aws_brain_batch.sh pull` -> `brain/experiments/self_completion_score.py <dir> 300 ... 305` -> C0-C3
  (`research/2026-11-15-self-completion-replication-1.5b.md`; GO -> flip `self_completion`). (2) Pool report-only seeds
  306-311 off/on (`~/brain-lane/outSC/`, done.txt). ssh: `SSH_AUTH_SOCK= ssh -o IdentitiesOnly=yes`.
- **POOL:** pool40/41/42 run brain/ jobs from `~/brain-lane` (venv `~/brain-lane/venv` on 40/41, `~/derisk-pool/sim/.venv`
  on 42; pattern `brain/experiments/jobs/2026-11-14-pool-*.sh`). pool40 host key re-trusted 11-14 (reinstall).
- **READY WORK:** answer-rate layout (`tools/answer_rate.py`). Pi: `PI_LANE.md`.
- **NEXT STEP:** (a) own name after the night: "What's your name?" recalls the SISTER telling on 6/6 seeds in session 2
  (`results/2026-11-15-self-completion-weekly-1.5b.md`; centred-cosine S0 key, "name" shared). No-training pool probe:
  day-1 tellings stored, S0 vs S_all (S_all self 1.67 vs 0.67 in `2026-11-14-key-default-1.5b.md`), off/on completion.
  (b) false recall after sleep: rank is not the lever (`results/2026-11-15-rank-diagnosis-1.5b.md`, MIXED); 10th method
  = hippocampus-as-teacher night (record A4 "distilled reinstatement", never built), no-training gate first.
  Day-1 own name: self-completion 6/6 vs 2/6 (NARROWED by one cat answer; replication live).
- **DELEGATED:** Pi: lane items. Hermes: [weekly] is Claude-on-AWS while W is dedicated.
- **OWNER WORDS (2026-11-02, verbatim):** "Do all, ensure Hermes and Pi also follow our new methodology. And one
  caveat, I may not noticeably correct you so I don't want that as a benchmark of improvement. All I really care
  about is us making meaningful progress on this project, which has stalled for way too long as I continue to spend
  money on AI to drive it."
- **OPEN:** none blocking. Chores: board timer -> main-following checkout; brain/ lane JSONs stamp no host/sha.
