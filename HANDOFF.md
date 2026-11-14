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
- **STATE (11-13):** reinstatement gain refit at 1.5B (`brain/experiments/results/2026-11-13-gain-refit-1.5b.md`):
  LEAD at gain 0.5 (oracle routing, unslept): held-out told 7/12 vs 1/12 at the live gain 4 (which mostly made the
  cortex talk ABOUT the word). Skeptics NARROWED (para 5/12 by hand); wrong-memory control injects nothing (0/24);
  finer grid peaks 0.5-0.75. Switch `Brain(gain="size")` built, default OFF. Earlier today: scorer fixed
  (`is_generic`); mixed-day gate VOID; at 1.5B told answers came only from sleep (c 9.33).
- **LIVE RUNS:** AWS g6.xlarge (g5 had no capacity), batch for `brain/experiments/jobs/2026-11-13-aws-live-gain.txt`
  (prereg `brain/research/2026-11-13-live-gain-1.5b.md`, L0-L4; ~2-3 h, self-stopping). Next session:
  `bash tools/aws_brain_batch.sh pull` (wait for "stopped"), copy `livegain_s*.json` from `_scratch/brainv2/aws/<batch>/`
  to `brain/experiments/results/2026-11-13-live-gain-1.5b/`, run `python -m brain.experiments.live_gain_table <dir>`.
  W is dedicated to the owner's other work (RULING w-dedicated-1113): only mini-PCs and AWS (cap ~$50/day).
  Pool: pool42 has `~/brain-lane` (brain/ + rwkv + 1.5B model; venv `~/derisk-pool/sim/.venv`); pool41 has no torch
  (py3.10; installing needs the owner's OK for a download); pool40 host key changed (owner's call).
- **READY WORK:** pool42: re-score the other `correct_answer` results (0.1B live-brain, told-gate, weekly) with
  `rescore_told.py`; the 0.1B gain (4) was fitted with the old scorer too: refit it the same way (hippo_gain_refit
  --model 0.1b, pool42). Pi: `research/coordination/PI_LANE.md` (on `research/pi-lane-1105`, unmerged).
- **NEXT STEP:** read the live gain verdict. GO -> land-result checks (skeptic, answer rate), then flip gain="size"
  for 1.5B and show it in the weekly transcript; then the confidence gate on top. NO-GO -> the routed/slept path is
  where it fails: read which (L1 vs L4).
- **DELEGATED:** Pi: lane items. Hermes: [weekly] is Claude-on-AWS while W is dedicated.
- **OWNER WORDS (2026-11-02, verbatim):** "Do all, ensure Hermes and Pi also follow our new methodology. And one
  caveat, I may not noticeably correct you so I don't want that as a benchmark of improvement. All I really care
  about is us making meaningful progress on this project, which has stalled for way too long as I continue to spend
  money on AI to drive it."
- **SPEED LEVERS (built):** AWS lane `tools/aws_brain_batch.sh`; Pi's Claude gated by live plan usage (`usage.ts`).
- **OPEN:** none blocking. Chores: repoint the board timer to a main-following checkout; pool40 host key (owner's call).
