# High-use facts: does use strengthen consolidation? (2026-11-03, 0.1B, six seeds)

Milestone "high-use facts never wrong" (2026-11-16, board #280). Verdict: NO-GO on this method at 0.1B. A lead for
"recall rises with use" is not established either.

## What ran
Probe `brain/experiments/high_use_probe.py`. The 12 facts of `facts.py` (kept, not extended, so the known 0.1B in-context
ceiling of 6/12 applies); 4 facts told k=1, 4 told k=3, 4 told k=10. The fact-to-k assignment is a seeded permutation, so
each seed has a different set of facts at k=10. A fact told k times is one episode with strength k (sleep() samples in
proportion to strength, identical to k copies in the replay distribution). One sleep, 240 steps, lr 5e-4, kl 1, hippocampus
off. Arms: no_sleep; sleep_use (strength = k); sleep_equal (control, same steps, all strengths 1, same k labels).
Per seed ~25 min on CPU (4 concurrent, memcap 4 GB, OMP 4 threads):

    python -m brain.experiments.high_use_probe --model rwkv7-g1d-0.1b-20260129-ctx8192.pth --steps 240 --seed S --out sS.json

Seeds 42 43 44 100 101 102. Raw output: `s*.json`, per-seed lines: `aggregate.txt`.

## Results (recall of the 4 facts in each k group, mean over six seeds)
| arm | k=1 | k=3 | k=10 | total recall /12 per seed | false recall /4 | drift KL |
|---|---|---|---|---|---|---|
| no_sleep | 0 | 0 | 0 | 0 all | 0 | 0 |
| sleep_use | 1.33 | 1.33 | 2.17 | 6,6,5,9,1,2 (mean 4.8) | 2.8 | 0.029 |
| sleep_equal | 1.67 | 0.67 | 1.50 | 4,4,4,3,4,4 (mean 3.8) | 2.2 | 0.028 |

Paraphrase recall (mean of 4): use arm 1.17 / 0.83 / 1.00 for k=1/3/10; equal arm 1.17 / 0.50 / 0.50.

k=10 recall per seed in the use arm (of 4): 3, 3, 2, 3, 1, 1. Never 4/4 on any seed. Missed k=10 facts: s42 teal;
s43 biscuit; s44 cello, oslo; s100 march; s101 marguerite, volvo, cello; s102 lyon, volvo, oslo.

## What it shows
- "Never wrong" fails: k=10 facts reach 100% on 0 of 6 seeds (best 3/4), although each k=10 fact got ~43 replays vs ~4 for k=1.
- The use effect is weak and inconsistent (not established): in the use arm k=10 > k=1 on 4 seeds (s42, s43, s44, s101), ties on s100 (3 vs 3) and s102 (1 vs 1). Versus the equal control, the k=10 gain is +0.67 facts of 4 on average, driven by s42, s43, s44, s100;
  on s101 and s102 the use arm is WORSE than the control (1 and 2 total vs 4 and 4), so strength weighting concentrated replay
  and hurt overall recall. Mean total recall use 4.8 vs control 3.8 hides that spread.
- Paraphrase recall does not rise with use (about 1 of 4 in every group).
- False recall is higher with weighting (2.8 vs 2.2 of 4, learned names leak); drift is small and the same (KL 0.03).
- More replay did not make k=10 facts reliable at 0.1B, but this does NOT show that exposure is not the limit: the per-fact
  ceiling was not measured, the strength k was set by hand, and the hippocampus-on path was not tested. At 0.1B the in-context ceiling is 6/12, so at most half the facts
  are reachable at all; s101 recalled only 1/12 in total. Per-fact in-context ceiling was not measured per k group,
  so a k=10 miss is not proven to be a fact the cortex could never hold.

## What it does not show
No claim about 0.4B or the hippocampus-on path (hippocampus strength rises with use there, untested here). n=4 facts per cell, so
cell means are coarse. Not a verdict on use-weighted consolidation in general: only strength-proportional replay sampling at 240 steps.

## Next
(1) Measure the per-fact in-context ceiling at 0.1B and score k=10 against only the reachable facts. (2) Use-weighting by more steps
for high-k facts is the wrong lever; test per-episode loss weighting or a replay floor so every fact gets a minimum, with the false-recall
fix (the familiarity/"I do not know" prerequisite) first, since false recall grows with every added replay. (3) Rerun at 0.4B once the
0.4B recall (2.3/12) problem is understood.
