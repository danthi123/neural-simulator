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
- **STATE:** `brain/plastic_cortex.py` (trainable cortex, equals pretrained at zero; tests pass), `brain/sleep.py`
  (strength-weighted interleaved replay + KL protection), `brain/experiments/sleep_probe.py`. Seed 7, 0.1B: true
  3/12 (reworded 1/12), shuffled 0/12, no-sleep 0/12, drift KL 0.03. In-context ceiling at 0.1B is 6/12, 0.4B 11/12.
  Record brief: `brain/research/2026-11-02-record-brief-consolidation-and-selection.md`.
- **LIVE RUNS:** six gate seeds of sleep_probe at 0.1B on W CPU (out: `/home/dant123/Projects/sim/_scratch/brainv2/
  sleep_probe_s<seed>_0.1b.json`, ~9 min each). Stage-2 retake on the pool (Hermes lands it).
- **NEXT STEP:** read the six seeds; if the effect holds, run 0.4B through the GPU queue (it yields the local model),
  then try varied replay (reworded rehearsal) for the paraphrase gap; then wire sleep into `Brain` (restart).
- **DELEGATED:** Pi: weekly test upgrade (`research/coordination/PI_LANE.md`, board #281). Hermes: [weekly] every
  Monday from 11-09; Stage-2 retake landing (`LOCAL_LLM_RUNBOOK.md` §4, RULING new-core-1102).
- **OWNER WORDS (2026-11-02, verbatim):** "Do all, ensure Hermes and Pi also follow our new methodology. And one
  caveat, I may not noticeably correct you so I don't want that as a benchmark of improvement. All I really care
  about is us making meaningful progress on this project, which has stalled for way too long as I continue to spend
  money on AI to drive it."
- **OPEN QUESTIONS:** none blocking.
