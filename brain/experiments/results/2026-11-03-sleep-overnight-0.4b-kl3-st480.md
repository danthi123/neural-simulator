# Sleep: overnight 2026-11-02 on the mini-PCs (0.4B; 0.1B at kl_weight 3; 0.1B at 480 steps)

Runner: `tools/brain_batch_run.sh` (CPU, fp32) on pool40/41/42, `python -m brain.experiments.sleep_probe <args>`.
Raw: `sleep_overnight_1102/s<seed>_<condition>.json`. Hippocampus OFF for every read-out. Cells: recall/12 ·
paraphrase/12 · false recall/4 (true-replay arm | shuffled arm). Baseline 0.1B (240 steps, lr 5e-4, kl 1):
`2026-11-02-sleep-0.1b-six-seeds.md` (true 3.8 / 2.2 / 2.2; shuffled 0.3 / 1.3 / 2.2).

## 0.4B, 240 steps, lr 5e-4, kl 1

| seed | true replay | shuffled | drift (true) |
|---|---|---|---|
| 42 | 3 / 2 / 0 | 0 / 0 / 0 | 0.022 |
| 43 | 0 / 0 / 0 | 0 / 1 / 0 | 0.027 |
| 44 | 4 / 3 / 2 | 0 / 1 / 2 | 0.031 |
| 100 | 2 / 3 / 0 | 1 / 1 / 0 | 0.034 |
| 101 | 2 / 1 / 1 | 1 / 0 / 2 | 0.030 |
| 102 | 3 / 0 / 0 | 0 / 0 / 0 | 0.025 |
| mean | 2.3 / 1.5 / 0.5 | 0.3 / 0.5 / 0.7 | |

## 0.1B, 240 steps, kl_weight 3

| seed | true replay | shuffled | drift (true) |
|---|---|---|---|
| 42 | 3 / 2 / 1 | 0 / 0 / 0 | 0.006 |
| 43 | 4 / 3 / 4 | 1 / 2 / 4 | 0.010 |
| 44 | 3 / 3 / 0 | 0 / 0 / 2 | 0.014 |
| 100 | 5 / 4 / 1 | 0 / 0 / 2 | 0.015 |
| 101 | 2 / 2 / 3 | 1 / 1 / 3 | 0.016 |
| 102 | 2 / 1 / 0 | 0 / 1 / 0 | 0.009 |
| mean | 3.2 / 2.5 / 1.5 | 0.3 / 0.7 / 1.8 | |

## 0.1B, 480 steps, kl 1

| seed | true replay | shuffled | drift (true) |
|---|---|---|---|
| 42 | 8 / 5 / 4 | 0 / 0 / 4 | 0.035 |
| 43 | 5 / 5 / 4 | 1 / 1 / 4 | 0.023 |
| 44 | 3 / 3 / 2 | 0 / 1 / 3 | 0.033 |
| 100 | 3 / 1 / 3 | 0 / 2 / 1 | 0.025 |
| 101 | 5 / 2 / 4 | 1 / 1 / 4 | 0.031 |
| 102 | 3 / 2 / 1 | 1 / 0 / 1 | 0.026 |
| mean | 4.5 / 3.0 / 3.0 | 0.5 / 0.8 / 2.8 | |

## What it shows

- **0.4B at the 0.1B settings is under-trained, not better:** true recall 2.3/12 (seed 43: 0/12) against an
  in-context ceiling of 11/12, though false recall is low (0.5/4). The settings were tuned at 0.1B; the 0.4B needs
  its own operating point before any size verdict (being swept 11-03 on the GPU).
- **kl_weight 3 at 0.1B:** recall about the same (3.2 vs 3.8), drift lower, false recall a little lower (1.5 vs 2.2)
  but still well above no-sleep (0) and not below shuffled (1.8).
- **480 steps at 0.1B:** recall up (4.5; seed 42 reaches 8/12, above the 6/12 in-context ceiling measured with one
  telling), paraphrase up (3.0 vs 0.8 shuffled), but false recall up too (3.0/4, shuffled 2.8).
- **Pattern across all four conditions:** more consolidation brings more recall AND more confabulation of learned
  names onto untold questions. False recall is the method's main defect; it does not fall with a stronger KL guard
  alone. Work on it: the false-recall track (11-03).
