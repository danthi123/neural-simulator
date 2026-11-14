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
- **STATE (11-14):** the 1.5B brain now answers facts told today, before sleep. Live six-seed mixed day
  (`brain/experiments/results/2026-11-13-live-gain-1.5b.md`): gain 0.5 GO (L0-L4), new facts 3.50/6 vs 1.17 at the old
  gain 4 and 0.17 hippocampus-off; told 8.00/12 vs 4.83; no slept-fact cost vs c; skeptics NARROWED (false recall
  uninformative on this probe). **Flipped:** `Brain(gain="auto")` default = 0.5 at 1.5B, 4.0 elsewhere (0.1B unchanged;
  0.1B refit NOT A LEAD, `2026-11-13-gain-refit-0.1b.md`). Oracle refit: `2026-11-13-gain-refit-1.5b.md`. Scorer fix
  re-scored 0.1B live-brain and told-gate (awake hippocampus ~1.5/12, errata in those docs).
- **LIVE RUNS:** none (AWS batch pulled and terminated; ~$1 today). W is dedicated to the owner's other work (RULING
  w-dedicated-1113): only mini-PCs and AWS (cap ~$50/day). Pool: pool42 has `~/brain-lane` (brain/ + rwkv + 0.1B and
  1.5B models; venv `~/derisk-pool/sim/.venv`; run jobs as `cd ~/brain-lane/src && BRAIN_MODEL_DIR=~/brain-lane/models
  PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib <venv python> -m brain...`); pool41 has no torch (py3.10; installing
  needs the owner's OK for a download); pool40 host key changed (owner's call).
- **READY WORK:** pool42 (CPU, deterministic): oracle check of the S0 key at gain 0.5 vs S_all (which key to default at
  1.5B). AWS: the confidence gate on top of gain 0.5 (mixed day, arms c/r/h); S_all default at 1.5B. Pi:
  `research/coordination/PI_LANE.md` (on `research/pi-lane-1105`, unmerged).
- **NEXT STEP:** make S_all + size bar the 1.5B default if a six-seed live run with gain 0.5 beats S0 (pre-register;
  AWS), then the confidence gate for false recall. The weekly transcript is 0.1B: the 1.5B cortex default is the
  milestone item that makes this visible there.
- **DELEGATED:** Pi: lane items. Hermes: [weekly] is Claude-on-AWS while W is dedicated.
- **OWNER WORDS (2026-11-02, verbatim):** "Do all, ensure Hermes and Pi also follow our new methodology. And one
  caveat, I may not noticeably correct you so I don't want that as a benchmark of improvement. All I really care
  about is us making meaningful progress on this project, which has stalled for way too long as I continue to spend
  money on AI to drive it."
- **SPEED LEVERS (built):** AWS lane `tools/aws_brain_batch.sh`; Pi's Claude gated by live plan usage (`usage.ts`).
- **OPEN:** none blocking. Chores: repoint the board timer to a main-following checkout; pool40 host key (owner's call).
