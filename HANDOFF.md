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
  `2026-11-14-sall-contrast-1.5b.md`: told 5.83 -> 6.83, one seed carries it, +2 false). Switches stay OFF.
- **LIVE RUN:** AWS g6.xlarge, `brain/experiments/jobs/2026-11-14-aws-sall-contrast-mixed.txt` (prereg
  `brain/research/2026-11-14-sall-contrast-mixed-1.5b.md`, X0-X4; ~2.5 h, self-stopping). Next: `bash
  tools/aws_brain_batch.sh pull` once "stopped"; `*mix_s*.json` -> `results/2026-11-14-sall-contrast-mixed-1.5b/`;
  `python -m brain.experiments.mixed_contrast_table <dir>`. W dedicated to the owner (RULING w-dedicated-1113); AWS ~$50/day.
- **POOL:** pool40/41/42 run brain/ jobs from `~/brain-lane` (venv `~/brain-lane/venv` on 40/41, `~/derisk-pool/sim/.venv`
  on 42; pattern `brain/experiments/jobs/2026-11-14-pool-*.sh`). pool40 host key re-trusted 11-14 (reinstall).
- **READY WORK:** pool: scorer fix (reject a disclaimer + general talk naming the key; re-score 11-14 contrast results);
  an answer-rate layout for unslept/mixed probes in `tools/answer_rate.py`. Pi: `research/coordination/PI_LANE.md`.
- **NEXT STEP:** mixed-day verdict: GO -> land-result, flip the 1.5B package (S_all + size bar + contrast 3); then the
  false-recall gate. The 1.5B cortex default (milestone 11-16) puts this in the weekly transcript (needs a prereg).
- **DELEGATED:** Pi: lane items. Hermes: [weekly] is Claude-on-AWS while W is dedicated.
- **OWNER WORDS (2026-11-02, verbatim):** "Do all, ensure Hermes and Pi also follow our new methodology. And one
  caveat, I may not noticeably correct you so I don't want that as a benchmark of improvement. All I really care
  about is us making meaningful progress on this project, which has stalled for way too long as I continue to spend
  money on AI to drive it."
- **OPEN:** none blocking. Chores: repoint the board timer to a main-following checkout. Speed levers: AWS lane
  `tools/aws_brain_batch.sh`, Pi gated by `usage.ts`.
