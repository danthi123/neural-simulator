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
- **LIVE RUNS:** (bg chain) M4b fix: nmft_s100/101/102 rerun with matched halves (`jobs/2026-11-15-aws-near-miss-ft-fix.txt`)
  -> pull -> rescore into `results/2026-11-15-near-miss-told-filter-1.5b/` -> `near_miss_table --on nmft`; then the
  rank diagnosis batch (`jobs/2026-11-15-aws-rank-diagnosis.txt`, 18 jobs, prereg `research/2026-11-15-rank-diagnosis-1.5b.md`
  R0-R3). Pool: weekly `--sleep-lesion` smoke s200/s201/s202 (`~/brain-lane/outSL/`, pool41/42/40).
  ssh: `SSH_AUTH_SOCK= ssh -o IdentitiesOnly=yes` (the Bitwarden agent hangs).
- **POOL:** pool40/41/42 run brain/ jobs from `~/brain-lane` (venv `~/brain-lane/venv` on 40/41, `~/derisk-pool/sim/.venv`
  on 42; pattern `brain/experiments/jobs/2026-11-14-pool-*.sh`). pool40 host key re-trusted 11-14 (reinstall).
- **READY WORK:** weekly `--sleep-lesion` arm (built, untested on weights: smoke, then prereg it as the #281 test); answer-rate layout (`tools/answer_rate.py`). Pi: `PI_LANE.md`.
- **NEXT STEP:** milestone day 14 NOT fully met (`results/2026-11-15-weekly-1.5b-milestone.md`): own name wrong on day 1
  on 4/6 (correct episode recalled, reinstatement too weak) and "Dan" after the night on 2/6. M4b: CPU NOT A LEAD;
  registered s100-s102 VOID (wrong halves), rerun. Then read the rank diagnosis: R3 (content) -> hippocampus-as-teacher
  night (no-training gate first); R1/R2 -> M4 at that rank.
- **DELEGATED:** Pi: lane items. Hermes: [weekly] is Claude-on-AWS while W is dedicated.
- **OWNER WORDS (2026-11-02, verbatim):** "Do all, ensure Hermes and Pi also follow our new methodology. And one
  caveat, I may not noticeably correct you so I don't want that as a benchmark of improvement. All I really care
  about is us making meaningful progress on this project, which has stalled for way too long as I continue to spend
  money on AI to drive it."
- **OPEN:** none blocking. Chores: board timer -> main-following checkout; brain/ lane JSONs stamp no host/sha.
