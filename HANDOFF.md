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
- **STATE:** `brain/plastic_cortex.py`, `brain/sleep.py`, `brain/experiments/sleep_probe.py` (now `--kl-weight`,
  `--lr`). SIX SEEDS at 0.1B (`brain/experiments/results/2026-11-02-sleep-0.1b-six-seeds.md`): recall holds on every
  seed (true 3-4/12, shuffled 0-1, no-sleep 0; ceiling 6/12); paraphrase NOT shown (2.2 vs 1.3 shuffled); FALSE
  RECALL is the defect (any sleep 0 -> ~2/4, learned names leak into unrelated answers; true = shuffled on mean,
  above it on s100). Drift small (KL <= 0.04). A lead, not a GO.
- **LIVE RUNS (W is off, owner gaming):** mini-PCs pool40/41/42, user unit `brainv2-sleep` (linger on), code+venv+
  weights in `~/brainv2/` on each node, jobs `~/brainv2/jobs.txt`, results `~/brainv2/results/<name>.json`, log
  `~/brainv2/run.log`. Per node in order: 0.4B x2 seeds, 0.1B kl_weight 3.0 x2, 0.1B 480 steps x2 (pool40 42/43,
  pool41 44/100, pool42 101/102). Pull: `rsync -a poolNN:brainv2/results/ _scratch/brainv2/pool_results/`. Stage-2
  retake STOPPED 11-02 18:20 (owner: best judgement; 103 lines withdrawn, half-run s42/43/44, no verdict).
  Pi #203 GPU cells killed (old track).
- **NEXT STEP:** pull the overnight results. 0.4B: does recall approach its 11/12 ceiling? Does kl 3.0 or 480 steps
  cut false recall without losing recall? Then (before-build) the false-recall method: the brain must learn to say
  it does not know (record + prerequisites first: source monitoring/familiarity in the old record). Then varied
  replay (Pi builds the arm) and wiring sleep into `Brain` (restart = sleep, adapters persisted). 11-04: KEEP/KILL.
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
