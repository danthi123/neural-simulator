# 1.5B key default at gain 0.5: KEEP S0 (K2 fails by 0.17). S_all is ahead by about one fact, with no cost

Pre-registration `brain/research/2026-11-14-key-default-1.5b.md` (K0-K5, commit 0a0cbd9f, pushed before the runs).
Data `results/2026-11-14-key-default-1.5b/` (`A/` unslept day, pool40/41/42 CPU fp32, JSON `host` per file; `B/`
mixed day, AWS g6.xlarge L4 (g5 had no capacity), batch 20261114-0453, all rc 0; `table.txt`). Arms: c hippocampus
off, s S0 key at gain 0.5 (the current 1.5B default), r S_all + size bar at gain 0.5.

| part | arm | told (12) | slept (6) | new (6) | para (12) | false (12) | self (3) |
|---|---|---|---|---|---|---|---|
| A unslept | c | 0.17 | - | - | 0.17 | 0.17 | 0.00 |
| A unslept | s S0 | 4.83 | - | - | 1.50 | 0.33 | 0.67 |
| A unslept | r S_all | 5.83 | - | - | 2.17 | 0.83 | 1.67 |
| B mixed | c | 4.83 | 4.67 | 0.17 | 3.83 | 6.83 | 3.00 |
| B mixed | s S0 | 7.17 | 4.67 | 2.50 | 4.67 | 6.83 | 3.00 |
| B mixed | r S_all | 8.00 | 4.50 | 3.50 | 5.50 | 6.67 | 3.00 |

K0 holds, K1 holds (A: +1.00, r >= s on 6/6), **K2 fails** (B: +0.83 < 1, r >= s on 4/6), K3 holds, K4 holds
(A: 0.83 vs 0.33, at the bar), K5 holds. **VERDICT (registered): KEEP S0.** The default key does not change.

## What it shows

- The gain, not the key, was the big lever. With S0 at gain 0.5, a 1.5B brain that has not slept answers 4.83 of 12
  facts told that day (11-05, gain 4, re-scored: 0.50), and on a mixed day 2.50 of 6 new facts (gain 4 with S_all:
  1.17).
- S_all is ahead on every told measure (A +1.0, B +0.83; new facts +1.0) and costs nothing on slept facts, false recall
  (B) or self questions, but the registered margin was 1 fact and B fell short. The self-question risk from the
  identity check did not appear (A: S_all 1.67 vs 0.67; B: 3/3 for every arm, the slept cortex knows its name).
- Both keys still lose answers to first-person echoes: replies holding the fact word, B told, c/s/r = 35/49/55 of 72,
  against 29/43/48 credited ("My cat is called Pixel." to "What is my cat called?"). That perspective error is now the
  larger loss, not routing.
- Report-only extra seeds for part A (200-205, `A2/table.txt`, run after the registered part A): told s 4.67 vs r
  6.00, false 0.33 vs 0.33, self 0.67 vs 1.67. They replicate part A (S_all +1.3) and remove its false-recall
  borderline; they do not change the registered verdict, which part B decided.

Verdict is on S_all as the 1.5B default at this margin, not on the key. Next: the first-person echo (the brain
restates the user's sentence instead of answering it), and false recall on the slept cortex (c already says a told
fact for 6.83 of 12 never-told questions), the confidence gate's job.
