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
- **STATE (11-03, branch `research/integration-1103`, merged from four overnight branches + skeptic fixes):**
  Sleep is in the live `Brain` (restart = sleep, adapters persisted, `set_lesions`). 0.1B live smoke, hippocampus off,
  six seeds: correct-perspective recall 8/18 (substring 15/18 counted first-person echoes), consolidated lesion 0/18.
  Six-seed probe (0.1B): true 3-4/12 vs shuffled 0-1; 0.4B at 0.1B settings is under-trained (2.3/12 vs ceiling 11/12);
  kl 3 / 480 steps: more recall AND more false recall. FALSE RECALL is the defect and shows live ("Your horse is called
  Pixel" 4/6 seeds). Lead (not GO): question familiarity separates told from untold (hippocampal bar ~0.22, fitted
  in-sample, AUC 0.88; answer confidence fails, AUC 0.47). High-use facts: NO-GO at 0.1B (k=10 never 4/4; use effect
  weak/inconsistent; per-fact ceiling unmeasured). weekly.py now sleeps both arms. NOT merged (skeptics): meaning-selection
  (probe never runs `KeyedHippocampus.recall`), mood-state (6/6 PASS refuted, echo criterion fails 2/6), varied-replay (runs live).
- **LIVE RUNS (W may be off):** varied-replay on mini-PCs pool40/41/42 (`research/varied-replay-1103`; do not touch);
  GPU agent owns gpu_queue (cortex size, `research/cortex-size-1103*`). Pull mini-PC results: `rsync -a poolNN:brainv2/results/ _scratch/brainv2/pool_results/`.
- **NEXT STEP:** (1) abstention in the live brain: hippocampal question-familiarity gate (before-build: held-out bar, not
  in-sample) then run the `--dont-know` arm; (2) 12-fact live test (not 3) with a correct-perspective scorer and a shuffled arm;
  (3) cortex size operating point for 0.4B. 11-04: KEEP/KILL the text-replay method (record: recall real, false recall open).
- **DELEGATED:** Pi: weekly test upgrade, then the varied-replay arm (`research/coordination/PI_LANE.md`). Hermes:
  [weekly] every Monday from 11-09; WORKLOG at most once a day (RULING overnight-1102).
- **OWNER WORDS (2026-11-02, verbatim):** "Do all, ensure Hermes and Pi also follow our new methodology. And one
  caveat, I may not noticeably correct you so I don't want that as a benchmark of improvement. All I really care
  about is us making meaningful progress on this project, which has stalled for way too long as I continue to spend
  money on AI to drive it."
- **SPEED LEVERS (owner 11-02, built 11-02 night):** (A) `tools/aws_brain_batch.sh` runs brain/ jobs on one
  self-stopping AWS GPU (CUDA path smoke-tested on W); first use: the 0.4B six seeds if the mini-PCs are slow. (B) Pi's
  Claude calls are gated by live plan usage (`tools/pi_driver/extensions/sim-driver/usage.ts`, reviewed + fixed),
  daily backstops 20 checkpoints / 15 builds. Both land via `claude/overnight-1102` + `land_claude_branch.sh`.
- **OPEN QUESTIONS:** none blocking. Chores: W's primary checkout is on an old branch (no `tools/progress_report.py`);
  the board timer runs it from the stale lean-method worktree, so its brain-share line lags main: repoint it to a
  checkout that follows main. pool40's host key changed at its 10-02 reinstall (warning only; owner's call).
