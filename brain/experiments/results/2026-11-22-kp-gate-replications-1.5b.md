# K-P one-night gate replications, 1.5B, DEV seeds 319 and 321 (REPORT ONLY)

Prereg: `brain/research/2026-11-21-sleep-language-keep-prereg.md` (gate run 2, amendment A2; the gate was run on s316
and K-P launched gate-overridden). These two replications are report only: they change no verdict and no bar.
Method: one extra night (night 16) from each seed's `lifeS1F` night-15 home, B (no keep bank) vs K-P
(`keep_bank="prior"`), then `drift_read.py` on both resulting adapter sets. CPU fp32. s319 on W, s321 on pool40
(node-mxl1462drl). Raw JSON in the folder beside this file.

| seed | arm | held-out KL | held-out NLL | keep_ORDINARY KL | replayed-episode NLL | adapter norm |
|---|---|---|---|---|---|---|
| 319 | before (n15) | 2.933 | 4.747 | 0.142 | 1.400 | 69.1 |
| 319 | B n16 | 2.792 | 4.625 | 0.042 | 0.977 | 70.9 |
| 319 | K-P n16 | **0.762** | **2.707** | 0.130 | **0.444** | 68.9 |
| 321 | before (n15) | 2.858 | 4.674 | | 0.609 | |
| 321 | B n16 | 2.760 | 4.507 | 0.028 | 0.546 | 72.9 |
| 321 | K-P n16 | **0.813** | **2.737** | 0.139 | 0.500 | 70.6 |
| 316 (gate) | K-P vs B | 0.86 vs 2.80 | 2.81 vs 4.59 | | 0.451 vs 0.770 | |

Read: on both seeds, one K-P night cuts held-out KL by about 3.5x against B, and held-out NLL from about 4.6 to about 2.7.
Replayed-episode NLL is no worse than B (s319 better, s321 slightly better). This repeats the s316 gate on 3/3 seeds.
K-P gives up a little on its own three ordinary texts (KL 0.13-0.14 vs B 0.03-0.04): the prior bank replaces them in
the keep term. The bank costs 75-140 min of CPU sampling per night (`bank_s` 4553 on W, 8274 on pool40). s320 is
running on W. Caveat: one night only. The 15-day DEV lives (AWS, fp16) are the registered read.
