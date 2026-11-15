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
  Mixed day with sleep: contrast NO-GO (`2026-11-14-sall-contrast-mixed-1.5b.md`, S_all told +0.00, S0 -1.17): echo
  levers all banked NO-GO in the live brain; package stays S0 at gain 0.5.
  Near-miss / frame guard in sleep (M4): NOT A LEAD (N1-N3 fail; switch stays OFF).
  Scorer: disclaimer + general talk no longer credited (`2026-11-14-disclaimer-rescore.md`); no verdict changed.
- **LIVE RUNS:** AWS g6.xlarge i-0f6697d03b73b2378, batch 20261115-0742: day-14 milestone weekly at 1.5B (7 jobs,
  `brain/research/2026-11-15-weekly-1.5b-milestone.md` W0-W3; dead-man 240 min). Pull, wk15_*.json are .md transcripts. Pool40/41/42: report-only weekly seeds
  200-205 (`jobs/2026-11-15-pool-weekly-1.5b.sh`, out `~/brain-lane/outWK/`). ssh: `SSH_AUTH_SOCK= ssh -o IdentitiesOnly=yes`.
- **POOL:** pool40/41/42 run brain/ jobs from `~/brain-lane` (venv `~/brain-lane/venv` on 40/41, `~/derisk-pool/sim/.venv`
  on 42; pattern `brain/experiments/jobs/2026-11-14-pool-*.sh`). pool40 host key re-trusted 11-14 (reinstall).
- **READY WORK:** answer-rate layout for mixed probes (`tools/answer_rate.py`); M4b prereg. Pi: `PI_LANE.md`.
- **NEXT STEP:** milestone day 14 (11-16): read the 1.5B weekly batch (W0-W3) with land-result; GO -> weekly default 1.5B.
  False recall after sleep: M4 near-miss guard NOT A LEAD (`results/2026-11-14-near-miss-guard-1.5b.md`: false
  6.83 -> 3.83 but told 7.17 -> 6.00, para -1.5, about half the removed intrusions became abstentions or inventions;
  lever count 7). Weaker-night control: SELECTIVE, narrowed (`results/2026-11-15-weak-night-control-1.5b.md`: a
  shorter night keeps false 9.00 vs guard 5.50 at matched slept told; ~60% of the gain is abstention). Next: M4b =
  guard minus told-topic cues, confabulation scored as false (prereg, AWS six seeds).
  Count invented answers as false in the next prereg. Land `claude/gain-refit-1113` on main.
- **DELEGATED:** Pi: lane items. Hermes: [weekly] is Claude-on-AWS while W is dedicated.
- **OWNER WORDS (2026-11-02, verbatim):** "Do all, ensure Hermes and Pi also follow our new methodology. And one
  caveat, I may not noticeably correct you so I don't want that as a benchmark of improvement. All I really care
  about is us making meaningful progress on this project, which has stalled for way too long as I continue to spend
  money on AI to drive it."
- **OPEN:** none blocking. Chores: board timer -> main-following checkout; brain/ lane JSONs stamp no host/sha.
