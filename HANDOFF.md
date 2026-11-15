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
- **LIVE RUNS:** AWS rank diagnosis batch (`jobs/2026-11-15-aws-rank-diagnosis.txt`, 18 jobs, i-020093192a5d5cbcb,
  launched 11-15 ~14:20, MAX_MIN 420) -> `bash tools/aws_brain_batch.sh pull` -> rescore -> read R0-R3
  (`research/2026-11-15-rank-diagnosis-1.5b.md`) against banked rank-8 nmoff/nmon. Pool: report-only weekly
  `--sleep-lesion` seeds 203/204/205 (pool40/41/42, `~/brain-lane/outSL/`).
  pool40 then: own-name day-1 probe (`research/2026-11-15-own-name-day1-probe-1.5b.md` D0-D4, `~/brain-lane/outON/`).
  ssh: `SSH_AUTH_SOCK= ssh -o IdentitiesOnly=yes` (the Bitwarden agent hangs).
- **POOL:** pool40/41/42 run brain/ jobs from `~/brain-lane` (venv `~/brain-lane/venv` on 40/41, `~/derisk-pool/sim/.venv`
  on 42; pattern `brain/experiments/jobs/2026-11-14-pool-*.sh`). pool40 host key re-trusted 11-14 (reinstall).
- **READY WORK:** answer-rate layout (`tools/answer_rate.py`); day-1 own-name reinstatement (self episode recalled
  but not carried at gain 0.5). Pi: `PI_LANE.md`.
- **NEXT STEP:** milestone day 14: "a lesion visibly changes replies" now MET after sleep
  (`results/2026-11-15-sleep-lesion-weekly-1.5b.md`, L0-L3 hold, night's synapses off -> 0/6 keyed on 6/6 seeds).
  Still not met: own name (day 1 right 2/6; "Dan" after the night). M4b final (matched six seeds): NOT A LEAD.
  Read the rank diagnosis: R3 (content) -> hippocampus-as-teacher night (no-training gate first); R1/R2 -> M4 at that rank.
- **DELEGATED:** Pi: lane items. Hermes: [weekly] is Claude-on-AWS while W is dedicated.
- **OWNER WORDS (2026-11-02, verbatim):** "Do all, ensure Hermes and Pi also follow our new methodology. And one
  caveat, I may not noticeably correct you so I don't want that as a benchmark of improvement. All I really care
  about is us making meaningful progress on this project, which has stalled for way too long as I continue to spend
  money on AI to drive it."
- **OPEN:** none blocking. Chores: board timer -> main-following checkout; brain/ lane JSONs stamp no host/sha.
