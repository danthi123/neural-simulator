# 2026-11-03 Cortex size: working sleep operating point at 0.4B, then 1.5B (milestone 2026-11-16)

All numbers below are read from the JSONs in this folder (`summarize.py <json>...` prints them). Hardware: RTX 3090,
fp16 weights, `--device cuda`, queued through `tools/gpu_queue.sh` (local-llm yielded). Probe: `brain/experiments/sleep_probe.py`
(arms no_sleep / sleep_true / sleep_shuffled; recall = of 12 told facts, hippocampus off; false = of 4 never-told questions;
gKL = drift on ordinary text). In-context ceilings (`results/2026-11-02-scale-ladder.txt`): 0.1B 6/12, 0.4B 11/12, 1.5B 11/12.

Command template (every run):
`python -u -m brain.experiments.sleep_probe --device cuda --model <pth> --steps S --seed N --lr LR --kl-weight KL --out <json>`

## 0. Is fp16/cuda the same condition as CPU fp32? (0.1B, seed 42, 240 steps, lr 5e-4, kl 1)
| run | recall | paraphrase | false | gKL | shuffled recall |
|---|---|---|---|---|---|
| CPU fp32 (`sleep_0.1b/sleep_probe_s42_0.1b.json`) | 4 | 2 | 1 | 0.0159 | 0 |
| cuda fp16 (`cuda_0.1b_s42.json`) | 3 | 1 | 0 | 0.0174 | 0 |

Same regime (single seed, one item apart on recall; not bit-identical). GPU 396 s vs CPU 628 s for the three arms.

## 1a. 0.4B grid, dev seed 7 (sleep_true arm; selection = recall minus false, gKL < 0.05)
| lr | steps | kl | recall | para | false | gKL | recall-false | gKL<0.05 |
|---|---|---|---|---|---|---|---|---|
| 1e-3 | 240 | 1 | 5 | 2 | 3 | 0.0425 | 2 | yes |
| 1e-3 | 240 | 3 | 2 | 0 | 0 | 0.0197 | 2 | yes |
| 1e-3 | 480 | 1 | 7 | 5 | 4 | 0.0350 | 3 | yes |
| **1e-3** | **480** | **3** | 6 | 6 | 2 | 0.0188 | **4** | yes |
| 2e-3 | 240 | 1 | 0 | 0 | 0 | 0.349 | 0 | no (diverged) |
| 2e-3 | 240 | 3 | 0 | 0 | 0 | 1.099 | 0 | no (diverged) |
| 2e-3 | 480 | 1 | 0 | 0 | 0 | 0.288 | 0 | no (diverged) |
| 2e-3 | 480 | 3 | 3 | 0 | 3 | 0.0398 | 0 | yes |

Chosen: lr 1e-3, 480 steps, kl 3. lr 2e-3 diverges (gKL 0.29-1.1) in three of four cells. Shuffled arm at the chosen cell:
recall 0, false 2, gKL 0.031. Single seed, 8 cells: selection noise is large (recall 6 vs 7 and false 2 vs 4 are one or two items).

## 1b. 0.4B, chosen setting, six seeds (lr 1e-3, 480 steps, kl 3; `six_0.4b_s*.json`)
| seed | recall | para | false | gKL | shuffled recall / false | elapsed s (3 arms) |
|---|---|---|---|---|---|---|
| 42 | 5 | 5 | 4 | 0.0188 | 0 / 1 | 1552 |
| 43 | 1 | 1 | 0 | 0.0100 | 1 / 1 | 1261 |
| 44 | 4 | 1 | 1 | 0.0150 | 0 / 0 | 654 |
| 100 | 3 | 1 | 1 | 0.0139 | 0 / 2 | 653 |
| 101 | 6 | 5 | 4 | 0.0202 | 1 / 1 | 859 |
| 102 | 5 | 5 | 2 | 0.0138 | 0 / 0 | 1243 |
| mean | **4.0**/12 | 3.0 | 2.0 | 0.0153 (max 0.0202) | 0.33 / 0.83 | 703-1552 (wall times vary with shared-machine load) |

no_sleep is 0/0/0 on all seeds. Versus the overnight 0.4B setting (240 steps, lr 5e-4, kl 1: mean 2.3/12, s43 0/12) mean true
recall rose 2.3 -> 4.0, but 4.0/12 is still 36% of the 11/12 in-context ceiling and false recall (mean 2.0 of 4) is not low.
Seed spread is wide (1 to 6).

## 2. 1.5B (24 GB; same 12 facts; chosen setting then lr halved)
Fit: fits. `nvidia-smi` showed ~19.7-19.8 GB total GPU use during sleep (sampled, not a logged peak; ~2.6 GB of that is the desktop).
Time: 1105-1863 s per seed for all three arms (about 25-30 min; the sleep_true arm alone ~10 min); at 0.4B 653-1552 s.

| setting | seed | recall | para | false | gKL | shuffled recall / gKL |
|---|---|---|---|---|---|---|
| lr 1e-3, 480, kl 3 | 42 | 6 | 5 | 3 | 0.0176 | 0 / **2.58** |
| lr 1e-3, 480, kl 3 | 43 | 0 | 0 | 0 | **0.393** (diverged) | 0 / 0.18 |
| lr 1e-3, 480, kl 3 | 44 | 0 | 0 | 0 | **0.175** (diverged) | 0 / 0.22 |
| **lr 5e-4, 480, kl 3** | 42 | 11 | 6 | 3 | 0.0158 | 0 / 0.014 |
| **lr 5e-4, 480, kl 3** | 43 | 3 | 4 | 0 | 0.0090 | 0 / 0.009 |
| **lr 5e-4, 480, kl 3** | 44 | 7 | 4 | 2 | 0.0141 | 0 / 0.013 |
| lr 5e-4 mean | | **7.0**/12 | 4.7 | 1.7 | 0.0130 | 0 |

lr 1e-3 diverges on 1.5B in 2 of 3 seeds (and in the shuffled arm of seed 42), so lr scales down with size; lr 5e-4 is stable
(gKL < 0.02) on all three. Runs for the lr 5e-4 setting were not tuned (one value tried).

## What this shows
- A working operating point exists at 0.4B: lr 1e-3, 480 steps, kl 3 gives true recall mean 4.0/12 vs 0/12 no-sleep and ~0.3/12
  for shuffled episodes, drift gKL ~0.015, on all six seeds. The earlier 0.4B under-training is confirmed and partly fixed.
- At 1.5B, sleep fits in 24 GB, takes about 25-30 min/seed and, at lr 5e-4, gives mean 7.0/12 (3, 7, 11) vs 4.0 at 0.4B,
  with lower false recall (1.7 vs 2.0) and drift, and shuffled recall 0. Same 1.5B in-context ceiling (11/12) as 0.4B.

## What it does not show
- 1.5B has three seeds, not six: no general claim. The 7.0 vs 4.0 difference is within the seed spread (0.4B: 1-6; 1.5B: 3-11).
- The 1.5B lr was not tuned beyond one halving, and the 0.4B grid was one dev seed, so the two sizes are not matched in tuning effort.
- Neither size is near the in-context ceiling in mean; no size reached a GO. Paraphrase and cued-recall/transfer were not
  varied here. False recall of 1-2 of 4 is a real defect at both sizes.
- fp16-vs-fp32 equivalence is one seed at 0.1B only.
- Peak VRAM was sampled with nvidia-smi, not measured by a logged peak.

## Recommendation for the 2026-11-16 milestone
Choose 1.5B as the cortex size, provisionally, at lr 5e-4 / 480 steps / kl 3: it fits the 3090, its sleep mean is higher
(7.0 vs 4.0 on the true-episode arm) with no extra drift, and its in-context ceiling equals 0.4B's, so it has the headroom.
The cost is ~2x wall per seed and a hard 24 GB fit while local-llm is yielded. Before declaring the size final, run
1.5B on seeds 100/101/102 at lr 5e-4 (about 90 min) to complete the six seeds; if that mean falls to <= 4, fall back to 0.4B,
which is the only size with six seeds today. Either choice still has a recall gap to the in-context ceiling that sleep tuning
(steps, replay content, adapter rank) has to close.
