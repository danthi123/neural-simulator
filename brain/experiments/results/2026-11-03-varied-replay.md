# Varied replay at 0.1B: six seeds (2026-11-03)

## Pre-registered analysis (committed before the six-seed aggregate was computed)

The runs were launched overnight from `research/varied-replay-1103` @ b8fd34840 (no criteria were committed then),
so this pre-registers the ANALYSIS, not the run. Seed 42's JSON was opened once before this file was written;
no other seed was looked at and no aggregate was computed. Code on all three nodes matches the branch
(md5 of `sleep_probe.py`, `facts_rewordings.py`, `sleep.py` checked on pool40/41/42).

Run: `python -m brain.experiments.sleep_probe --steps 240 --seed <s>` (0.1B, lr 5e-4, kl_weight 1.0), arms
no_sleep / sleep_true / sleep_varied / sleep_shuffled, six seeds 42/43/44/100/101/102 on pool40/41/42 (CPU).
All read-outs with the hippocampus OFF.

Skeptic's demand: the rewordings of items 5 (dog: "My dog goes by Biscuit.") and 12 (son: "My son was born in
March.") reuse the wording of their PARA question ("What's my dog called?" / "What month was my son born?"), so
paraphrase on those two items can be a surface match, not a reworded memory. Primary read-out is therefore
paraphrase on the CLEAN 10 items (items 1-4, 6-11, 1-indexed).

Criteria (varied replay "helps reworded recall" only if ALL hold):
1. Clean paraphrase (/10): sleep_varied > sleep_true on >= 5 of 6 seeds, and mean gain >= +1.0.
2. Clean paraphrase: sleep_varied > sleep_shuffled on 6 of 6 seeds (shuffled trains the same name words).
3. Recall on the original questions (/12) not hurt: varied mean >= sleep_true mean - 0.5.
4. False recall (/4, never-told questions) not worse: varied mean <= sleep_true mean + 0.5.
Also reported: the all-12 paraphrase, a per-item paraphrase table (seeds passing, per arm), drift (KL).
Outcome labels: all four hold = LEAD (a lead is not a GO; land-result skeptic + lesion still needed);
1 or 2 fails = NOT SHOWN; 3 or 4 fails with 1-2 holding = TRADE (reported as such).

## Result: NOT SHOWN (criteria 1 and 2 fail; 3 and 4 hold)

Raw: `2026-11-03-varied-replay/s<seed>_0.1b_varied.json` + `.log` (pulled from pool40: 42, 43; pool41: 44, 100;
pool42: 101, 102). Table: `python -I brain/experiments/results/2026-11-03-varied-replay/analyze.py`.
Sanity: no_sleep, sleep_true and sleep_shuffled reproduce the 11-02 six-seed table exactly (same seeds, same code).

Cells: recall /12 / paraphrase /12 / clean paraphrase /10 (items 5, 12 dropped) / false recall /4 / drift KL.

| seed | sleep_true | sleep_varied | sleep_shuffled |
|---|---|---|---|
| 42 | 4 / 2 / 2 / 1 / 0.016 | 6 / 5 / 5 / 1 / 0.027 | 0 / 1 / 0 / 1 / 0.015 |
| 43 | 4 / 4 / 4 / 4 / 0.027 | 6 / 3 / 2 / 3 / 0.029 | 1 / 2 / 2 / 4 / 0.026 |
| 44 | 4 / 3 / 3 / 1 / 0.030 | 4 / 2 / 1 / 2 / 0.036 | 0 / 2 / 1 / 3 / 0.030 |
| 100 | 3 / 1 / 1 / 2 / 0.033 | 1 / 0 / 0 / 0 / 0.031 | 0 / 1 / 1 / 0 / 0.036 |
| 101 | 4 / 2 / 2 / 4 / 0.038 | 4 / 5 / 5 / 4 / 0.050 | 1 / 1 / 1 / 4 / 0.038 |
| 102 | 4 / 1 / 1 / 1 / 0.023 | 4 / 4 / 3 / 0 / 0.027 | 0 / 1 / 0 / 1 / 0.023 |
| mean | 3.8 / 2.2 / 2.2 / 2.2 / 0.028 | 4.2 / 3.2 / 2.7 / 1.7 / 0.033 | 0.3 / 1.3 / 0.8 / 2.2 / 0.028 |

(no_sleep = 0 on every read-out, every seed.)

1. Clean paraphrase, varied > true: 3/6 seeds (42, 101, 102; lower on 43, 44, 100), mean gain +0.5. FAIL.
2. Clean paraphrase, varied > shuffled: 3/6 seeds (ties on 43 and 44, lower on 100). FAIL.
3. Recall: varied 4.2 vs true 3.8. HOLDS (no cost).
4. False recall: varied 1.7 vs true 2.2. HOLDS (not worse; the difference is inside seed noise).

All-12 paraphrase (3.2 vs 2.2) looks better than the clean-10 (2.7 vs 2.2): one point of the gap is item 5
(dog), which varied answers on 3/6 seeds and true on 0/6 - the leaked-wording item, as the skeptic predicted.

Per item (seeds passing out of 6; * = leaky, excluded above):

| item | true para | varied para | shuffled para | true recall | varied recall |
|---|---|---|---|---|---|
| 1 cat | 4 | 1 | 0 | 6 | 4 |
| 2 colour | 1 | 1 | 1 | 1 | 1 |
| 3 born | 0 | 0 | 0 | 2 | 2 |
| 4 sister | 4 | 1 | 1 | 2 | 0 |
| 5* dog | 0 | 3 | 0 | 0 | 0 |
| 6 car | 1 | 4 | 2 | 3 | 5 |
| 7 friend | 2 | 3 | 0 | 2 | 3 |
| 8 work | 0 | 3 | 0 | 0 | 3 |
| 9 brother | 0 | 1 | 1 | 0 | 1 |
| 10 allergy | 1 | 2 | 0 | 6 | 4 |
| 11 grandma | 0 | 0 | 0 | 1 | 2 |
| 12* son | 0 | 0 | 3 | 0 | 0 |

What it shows:
- Varied replay MOVES which facts consolidate (gains on car, friend, work; losses on cat, sister) rather than
  lifting paraphrase overall. Where it gains, recall on the original question rises too (item 8: 0 -> 3/6), so the
  gain looks like more of those facts getting stored, not better transfer across wordings.
- Paraphrase still mostly rides on recall: of clean item-seeds recalled, 11/23 (true) and 14/25 (varied) are also
  paraphrased; paraphrase WITHOUT recall is 2 in both arms. The surface form did not buy generalisation.
- Seed spread is wide (clean paraphrase 0-5 for varied), so 6 seeds x 10 items cannot resolve a +0.5 effect.
- Shuffled answers item 12 on 3/6 seeds: the "month" paraphrase is answerable by a swapped name word, another
  reason that item is excluded.

Verdict on the METHOD (teacher rewordings x 3 at 0.1B, 240 steps, same replay budget), not on the capability.
Against `brain/PREREQUISITES.md`: the 0.1B cortex's in-context ceiling is 6/12 and paraphrase needs the fact to
be stored first; at ~4/12 stored there is little left to generalise from. Next methods: the same arm at 0.4B
(ceiling 11/12) on the GPU/AWS lane, with the fixed rewordings of `origin/research/pi-sleep-varied-replay`
@ 6305bf2da (no PARA word planted in any item), so all 12 items count.
