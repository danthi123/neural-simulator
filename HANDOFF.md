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
- **STATE (11-14):** 1.5B answers facts told today before sleep: gain 0.5 flipped on for 1.5B
  (`results/2026-11-13-live-gain-1.5b.md`, GO; 0.1B keeps 4). Key default KEEP S0 (`2026-11-14-key-default-1.5b.md`).
  Echo: speaker contrast d + 3(d - d_self) works (oracle LEAD; live S0 NO-GO, routing-bound; live S_all GO NARROWED,
  `2026-11-14-sall-contrast-1.5b.md`: told 5.83 -> 6.83, one seed carries it, +2 false;
  report-only seeds 200-205: told +0.50, para +1.17, the aunt -> Oslo lure 6/6 vs 2/6).
  Dose mu 1.5: NOT A LEAD (`2026-11-14-sall-contrast-dose-1.5b.md`; lure 5/6; effects scale with mu, mu 3 stays). Switches stay OFF.
  Mixed day with sleep: contrast NO-GO (`2026-11-14-sall-contrast-mixed-1.5b.md`, S_all told +0.00, S0 -1.17): echo
  levers all banked NO-GO in the live brain; package stays S0 at gain 0.5.
  Scorer: a disclaimer + general talk naming the key is no longer credited (`2026-11-14-disclaimer-rescore.md`): no
  banked verdict changes; contrast's oracle edge shrinks (fresh +20 -> +15), part of it was echoes turned AI disclaimers.
- **LIVE RUNS:** none.
- **POOL:** pool40/41/42 run brain/ jobs from `~/brain-lane` (venv `~/brain-lane/venv` on 40/41, `~/derisk-pool/sim/.venv`
  on 42; pattern `brain/experiments/jobs/2026-11-14-pool-*.sh`). pool40 host key re-trusted 11-14 (reinstall).
- **READY WORK:** an answer-rate layout for unslept/mixed probes in `tools/answer_rate.py`. Pi: `research/coordination/PI_LANE.md`.
- **NEXT STEP:** the false-recall gate: on a slept 1.5B day about 7 of 12 never-told questions get a told fact
  (aunt -> Oslo 12/12); before-build first (`2026-11-13-confidence-gate-1.5b.md` is the record). Then the 1.5B cortex
  default (milestone 11-16) for the weekly transcript (needs a prereg). Land `claude/gain-refit-1113` on main.
- **DELEGATED:** Pi: lane items. Hermes: [weekly] is Claude-on-AWS while W is dedicated.
- **OWNER WORDS (2026-11-02, verbatim):** "Do all, ensure Hermes and Pi also follow our new methodology. And one
  caveat, I may not noticeably correct you so I don't want that as a benchmark of improvement. All I really care
  about is us making meaningful progress on this project, which has stalled for way too long as I continue to spend
  money on AI to drive it."
- **OPEN:** none blocking. Chores: repoint the board timer to a main-following checkout. Speed levers: AWS lane
  `tools/aws_brain_batch.sh`, Pi gated by `usage.ts`.
