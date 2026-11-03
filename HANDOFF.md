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
- **STATE (11-03, branch `research/integration-final-1103`; integration-1103 + live-brain-12facts, high-use-ceiling,
  meaning-selection-fix, mood-fix, abstain-in-brain; varied-replay NOT merged, runs live on pool40/41/42):**
  Sleep is in the live `Brain` (restart = sleep, adapters persisted, lesions). Strict 12-fact test, six seeds, 0.1B:
  intact+sleep 6.5/12, intact no-sleep 6.33, hippocampus off after sleep 5.5, hippo+adapters off 0.17, shuffled 0:
  the trace is real, but sleep adds paraphrase (+1.67), not recall, and adds false recall (3.0 vs 2.0 of 4). 3-fact smoke
  re-scored correct-perspective: 8/18 hippo off, 2 of 3 facts. NEGATIVES: abstention gate FAIL (false recall 63->34,
  46->38; signals do not generalise to new wording; gated replies are invented answers); high-use NO-GO (UseHippocampus
  +1 fact of 4, 4/4 on 1/6 seeds; 0.1B in-context ceiling 6/12). LEADS: `S_all` meaning key (6 criteria pass at 0.1B and
  0.4B, J 0.30 vs 0.17/-0.05, not in Brain); mood iteration 2 NOT PASS 6/7 (neg 6/6, pos 3/6, random also darkens).
  Brain defaults unchanged: abstain=False, affect not wired, default key. Scorer fixed: `strict_scorer.correct_answer`
  (echo of told sentence + tautologies void; first-person word alone does not) in sleep smoke and weekly.
  Tests: see TESTS line below. Old-track `test_learned_assoc_graph_agent` fails pre-existing.
- **LIVE RUNS (W may be off):** varied-replay on mini-PCs pool40/41/42 (`research/varied-replay-1103`; do not touch);
  GPU agent owns gpu_queue (cortex size, `research/cortex-size-1103*`). Pull mini-PC results: `rsync -a poolNN:brainv2/results/ _scratch/brainv2/pool_results/`.
- **NEXT STEP:** (1) abstention v2: a familiarity signal learned over question forms (not word overlap, not the "my X"
  frame), bars refit on fitting set only, eval on the same fresh sets; (2) cortex size operating point (0.4B/1.5B) for
  sleep, then re-run the 12-fact test with the corrected scorer; (3) live-brain test of `S_all` as the key. 11-04:
  KEEP/KILL the text-replay method (record: recall real, adds paraphrase not recall, false recall open).
- **DELEGATED:** Pi: weekly test upgrade, then the varied-replay arm (`research/coordination/PI_LANE.md`). Hermes:
  [weekly] every Monday from 11-09; WORKLOG at most once a day (RULING overnight-1102).
- **OWNER WORDS (2026-11-02, verbatim):** "Do all, ensure Hermes and Pi also follow our new methodology. And one
  caveat, I may not noticeably correct you so I don't want that as a benchmark of improvement. All I really care
  about is us making meaningful progress on this project, which has stalled for way too long as I continue to spend
  money on AI to drive it."
- **SPEED LEVERS (owner 11-02, built):** (A) `tools/aws_brain_batch.sh` runs brain/ jobs on one self-stopping AWS GPU.
  (B) Pi's Claude calls are gated by live plan usage (`tools/pi_driver/extensions/sim-driver/usage.ts`).
- **OPEN QUESTIONS:** none blocking. Chores: W's checkout is on an old branch, so the board timer's brain-share line
  lags (repoint to a main-following checkout); pool40's host key changed at its 10-02 reinstall (owner's call).
