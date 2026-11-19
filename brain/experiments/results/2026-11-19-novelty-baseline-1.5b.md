# Curiosity baseline at 1.5B: the unchanged brain on the test bank, 4 samples per item, six seeds (2026-11-19)

`experiments/novelty_stage0.py --bank test --samples 4`, seeds 42/43/44/100/101/102, pool CPU fp32 (pool40/41/42).
864 replies, each from a fresh brain with Brain defaults. Two blind readers (Haiku + Sonnet, shuffled, class hidden;
agreement asks 0.99, invents 0.94, coherent 0.93). Counts where both agree. Labels: `2026-11-19-novelty-baseline-1.5b/blind_labels.json`.

| seed | asks what it is C / R / N (of 48) | invents a meaning C / R / N | coherent C / R / N |
|---|---|---|---|
| 42 | 1 / 4 / 6 | 6 / 18 / 9 | 41 / 47 / 39 |
| 43 | 1 / 5 / 6 | 8 / 17 / 14 | 44 / 45 / 42 |
| 44 | 1 / 4 / 4 | 7 / 23 / 14 | 41 / 45 / 44 |
| 100 | 1 / 4 / 5 | 8 / 22 / 10 | 47 / 44 / 42 |
| 101 | 0 / 4 / 7 | 10 / 23 / 12 | 43 / 44 / 42 |
| 102 | 3 / 5 / 7 | 8 / 21 / 14 | 44 / 43 / 42 |
| pooled (of 288) | 7 / 26 / 35 | 47 / 124 / 73 | 260 / 268 / 251 |

This is the OFF reference for the curiosity test: asking what a coined thing is 12% (common 2%), inventing a meaning for
it 25%. R "invents" is mostly real knowledge and is not a harm.
