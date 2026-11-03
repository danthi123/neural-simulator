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
- **STATE (11-03, branch `research/landing-1103` -> main):** sleep is in the live `Brain` (restart = sleep, adapters
  persisted, lesions). Strict 12-fact test, six seeds, 0.1B: intact+sleep 6.5/12, no-sleep 6.33, hippocampus off after
  sleep 5.5, hippo+adapters off 0.17, shuffled 0: the trace is real; sleep adds paraphrase (+1.67), not recall, and adds
  false recall (3.0 vs 2.0 of 4). **CORTEX SIZE DECIDED: 1.5B** (lr 5e-4 / 480 / KL 3): sleep recall 8.83/12 vs 4.0 at
  0.4B, higher on 6/6 seeds, false recall 2.17 vs 2.0 (`results/2026-11-03-cortex-size/`); default still 0.1B until the
  adoption steps land. FAILS (pre-registered): abstention v1 and v2; high-use NO-GO; confab M1 (cue completion) and M3
  (retrieval practice) at 0.1B; `S_all` key in the live brain (refit bar 0.16 still FAIL); varied replay NOT SHOWN.
  PARTIAL/LEADS: confab M2 v2 (own turn only: "I don't know" 75% DEV / 82% fresh, 0/360 intrusions, general replies
  untouched; IMPROVED over v1, per-seed bar 2/6); first weekly (W45) rescored strictly: intact 3/12, sleep+hippo off 5/12; mood (negative darkens reliably, positive weak; 2/6 at 0.4B).
  Most 0.1B negatives are scale-bound (0.1B answers 6/12 even in context). Brain defaults unchanged (all new switches
  OFF). Tests: 92 passed (brain/tests + tests/test_brain_*, hippocampus_*, plastic_cortex, gpu_remote).
- **LIVE RUNS:** W GPU: M3 at 0.4B (`_scratch/brainv2/m3-04b/`). AWS: 1.5B packet (live 12-fact + M2 v1/v2, six seeds,
  `brain/experiments/jobs/2026-11-03-aws-15b.txt`; ~6 h, dead-man 420 min). Keep/kill memo: KEEP, flip on the 1.5B test.
- **READY WORK (one next job per resource; the attention watcher points here when one goes idle):**
  W GPU: after M3, the weekly test at 1.5B (`brain.eval.weekly --model 1.5b`). AWS: pull + score the 1.5B packet
  (memo flip rule). Mini-PCs: the Pi-lane 12-fact rescoring item. Pi: `research/coordination/PI_LANE.md` open items. Claude: read the weekly transcript, land results.
- **NEXT STEP:** (1) merge `research/adopt-15b-steps-1103` (episode-file layout conflicts with the key switch and M2's
  `source` field in `Brain.save/load`) and finish the 1.5B adoption plan; (2) at 1.5B rerun the confab methods (M1
  needs slept abstain-v2 1.5B brains; M2 with 2-3 lived lessons) and the 12-fact test; (3) 11-04: KEEP/KILL the
  text-replay method (record: recall real, paraphrase not recall, false recall open).
- **DELEGATED:** Pi: weekly test upgrade, then the varied-replay arm (`research/coordination/PI_LANE.md`). Hermes:
  [weekly] every Monday from 11-09; WORKLOG at most once a day (RULING overnight-1102).
- **OWNER WORDS (2026-11-02, verbatim):** "Do all, ensure Hermes and Pi also follow our new methodology. And one
  caveat, I may not noticeably correct you so I don't want that as a benchmark of improvement. All I really care
  about is us making meaningful progress on this project, which has stalled for way too long as I continue to spend
  money on AI to drive it."
- **SPEED LEVERS (built):** AWS lane `tools/aws_brain_batch.sh`; Pi's Claude gated by live plan usage (`usage.ts`).
- **OPEN:** none blocking. Chores: repoint the board timer to a main-following checkout; pool40 host key (owner's call).
