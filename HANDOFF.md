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
- **STATE (11-14):** 1.5B answers facts told today before sleep: gain 0.5 flipped on for 1.5B
  (`results/2026-11-13-live-gain-1.5b.md`, GO; 0.1B keeps 4). Key default KEEP S0 (`2026-11-14-key-default-1.5b.md`).
  Echo: speaker contrast d + 3(d - d_self) works (oracle LEAD; live S0 NO-GO, routing-bound; live S_all GO NARROWED,
  `2026-11-14-sall-contrast-1.5b.md`: told 5.83 -> 6.83, one seed carries it, +2 false;
  report-only seeds 200-205: told +0.50, para +1.17, the aunt -> Oslo lure 6/6 vs 2/6).
  Mixed day with sleep: contrast NO-GO (`2026-11-14-sall-contrast-mixed-1.5b.md`, S_all told +0.00, S0 -1.17): echo
  levers all banked NO-GO in the live brain; package stays S0 at gain 0.5.
  Near-miss / frame guard in sleep (M4): NOT A LEAD (N1-N3 fail; switch stays OFF).
  Scorer: disclaimer + general talk no longer credited (`2026-11-14-disclaimer-rescore.md`); no verdict changed.
- **LIVE RUNS:** (1) AWS rank diagnosis (18 jobs, from this worktree's `.aws_brain`) -> `bash tools/aws_brain_batch.sh
  pull` -> rescore -> R0-R3 (`research/2026-11-15-rank-diagnosis-1.5b.md`). (2) AWS self-completion weekly (12 jobs,
  launched from worktree `_scratch/wt-selfc`: pull THERE) -> C0-C3 (`research/2026-11-15-self-completion-weekly-1.5b.md`).
  (3) Pool report-only self-completion seeds 200-205 off/on (`~/brain-lane/outSC/`).
  ssh: `SSH_AUTH_SOCK= ssh -o IdentitiesOnly=yes` (the Bitwarden agent hangs).
- **POOL:** pool40/41/42 run brain/ jobs from `~/brain-lane` (venv `~/brain-lane/venv` on 40/41, `~/derisk-pool/sim/.venv`
  on 42; pattern `brain/experiments/jobs/2026-11-14-pool-*.sh`). pool40 host key re-trusted 11-14 (reinstall).
- **READY WORK:** answer-rate layout (`tools/answer_rate.py`); day-1 own-name reinstatement (self episode recalled
  but not carried at gain 0.5). Pi: `PI_LANE.md`.
- **NEXT STEP:** milestone day 14: lesion item MET after sleep (`results/2026-11-15-sleep-lesion-weekly-1.5b.md`).
  Own name: day-1 cause found (`results/2026-11-15-own-name-day1-probe-1.5b.md`: self episode too weak at gain 0.5;
  completion-only store names it 12/12); score C0-C3, GO -> flip `self_completion`. Rank diagnosis: R3 -> hippocampus-
  as-teacher night (no-training gate first); R1/R2 -> M4 at that rank.
- **DELEGATED:** Pi: lane items. Hermes: [weekly] is Claude-on-AWS while W is dedicated.
- **OWNER WORDS (2026-11-02, verbatim):** "Do all, ensure Hermes and Pi also follow our new methodology. And one
  caveat, I may not noticeably correct you so I don't want that as a benchmark of improvement. All I really care
  about is us making meaningful progress on this project, which has stalled for way too long as I continue to spend
  money on AI to drive it."
- **OPEN:** none blocking. Chores: board timer -> main-following checkout; brain/ lane JSONs stamp no host/sha.
