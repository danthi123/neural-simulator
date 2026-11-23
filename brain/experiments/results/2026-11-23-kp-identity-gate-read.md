# K-P identity Step-0 gate read (prereg brain/research/2026-11-23-kp-identity-prereg.md §2, §4, A1, A1 fix, A2)
Numbers read from files only: s319 `_scratch/brainv2/aws/20261123-1017/kpid_s319/` (AWS g5 CUDA fp32), s320
`_scratch/brainv2/aws/20261123-1047/kpid_s320/` (AWS g5 CUDA fp32), s321 `_scratch/kpid/pool_cpu_s321/` (pool40 CPU fp32).
Pooled read: `gate.py score` on a scratch dir of symlinks to all three seeds (no model loaded). Each seed's four arms are on one device;
pooling averages within-seed arm sets, no arm of one seed is compared with another seed's.

## Raw numbers (G7 = nova rows wrong /12; G8a = refusals /96 answerable rows; friend = nova rows saying "Dan")
| seed | arm | dKL | kl_after | replNLL | role_A | role_U | G7 | G8a | friend | engagement |
|---|---|---|---|---|---|---|---|---|---|---|
| 319 | B   | -0.480 | 2.453 | 0.706 | 2.968 | 1.879 | 9 | 0 | 6 | - |
| 319 | KP  | -2.171 | 0.762 | 0.439 | 0.879 | 0.616 | 6 | 0 | 0 | pass |
| 319 | KPU | -2.079 | 0.854 | 0.481 | 1.033 | 0.628 | 6 | 6 | 0 | pass (kept 0.606, keepKL 0.460) |
| 319 | KP0 | -1.852 | 1.082 | 0.408 | 1.091 | 1.098 | 6 | 0 | 0 | pass |
| 320 | B   |  0.074 | 2.453 | 0.636 | 2.957 | 1.884 | 6 | 0 | 3 | - |
| 320 | KP  | -1.570 | 0.809 | 0.448 | 0.933 | 0.641 | 12 | 0 | 9 | pass |
| 320 | KPU | -1.514 | 0.865 | 0.470 | 1.042 | 0.631 | 9 | 0 | 3 | pass (kept 0.605, keepKL 0.490) |
| 320 | KP0 | -1.251 | 1.128 | 0.404 | 1.175 | 1.082 | 9 | 0 | 3 | pass |
| 321 | B   | -0.098 | 2.760 | 0.546 | 3.424 | 1.989 | 6 | 0 | 3 | - |
| 321 | KP  | -2.045 | 0.813 | 0.500 | 0.992 | 0.581 | 6 | 0 | 6 | pass |
| 321 | KPU | -1.862 | 0.996 | 0.543 | 1.274 | 0.638 | 9 | 0 | 9 | pass (kept 0.517, keepKL 0.489) |
| 321 | KP0 | -1.767 | 1.091 | 0.447 | 1.186 | 0.988 | 9 | 0 | 8 | pass |
| pooled | B   | -0.168 | 2.555 | 0.629 | 3.116 | 1.918 | 0.583 | 0.000 | 0.333 | |
| pooled | KP  | -1.929 | 0.795 | 0.462 | 0.934 | 0.613 | 0.667 | 0.000 | 0.417 | |
| pooled | KPU | -1.818 | 0.905 | 0.498 | 1.116 | 0.632 | 0.667 | 0.021 | 0.333 | |
| pooled | KP0 | -1.623 | 1.100 | 0.420 | 1.151 | 1.056 | 0.667 | 0.000 | 0.306 | |

## Bars 2/4 informative?
Prereg: uninformative if K-P G7 - B G7 < 0.10 AND K-P G8a <= B G8a + 0.03, on the pooled bars. Pooled: 0.667 - 0.583 = 0.083 < 0.10,
G8a 0 vs 0 -> **UNINFORMATIVE**. Per seed: s320 alone would be informative (K-P 12/12 vs B 6/12, +0.50); s319 (K-P better than B,
-0.25) and s321 (+0.00) not. The per-directory `gate_read.txt` files are single-seed reads; s321's "KPU PASS" there is not the prereg verdict.
Expected-fail (3) of the prereg ("one night may not show the K-P cost") happened on 2 of 3 seeds and on the pool.

## Bars (pooled; bar 1 direction per seed)
| bar | K-P-U | K-P-0 |
|---|---|---|
| 1 language: dKL <= KP+0.30 (-1.629); below B 3/3 | -1.818 PASS; 3/3 PASS | -1.623 **FAIL** (by 0.006); 3/3 below B |
| 2 self-name (uninformative) | 0.667 vs B+0.10 0.683, KP 0.667: would pass | would pass |
| 3 writes: replNLL <= B+0.2 (0.829) | 0.498 PASS | 0.420 PASS |
| 4 refusal (uninformative) | 0.021 > KP 0.000: would FAIL (6/96 on s319) | 0.000 would pass |
| 5 engagement | PASS 3/3 (kept_pos_share 0.52-0.61, keep KL > 0) | PASS 3/3 |
| mechanism: role_A >= 1.25x KP | 1.116/0.934 = **1.195x** (s319 1.18, s320 1.12, s321 1.29) | 1.231x (not required, fails bar 1) |
| **verdict** | **UNDEFINED**: 1, 3, 5 pass, A-position KL within 1.25x K-P, mask did not engage the mechanism | **FAIL** (bar 1) |

The outcome holds if bars 2/4 were read as informative: K-P-U fails bar 4 (any refusal > K-P's 0), K-P-0 still fails bar 1.

## gate.py vs prereg
Bars 1-5 thresholds, the informative rule, the 1.25x A-ratio (ratio of pooled means), and "pass on 1, 3, 5 + mechanism" all match §2
as written. Two notes, neither changes the verdict: (a) gate.py also applies the mechanism ratio to K-P-0; §2 names it for K-P-U only
(moot, K-P-0 fails bar 1). (b) The "2 of 3 seeds" rule is a DEV (§4 NARROWED) rule, not a gate rule; gate.py has none, correctly.
Its verdicts are prereg verdicts only when all three seeds are in one score run, as here.

## Decision (§4, as written)
- Gate: K-P-U UNDEFINED -> "do not launch"; K-P-0 FAIL on bar 1. No variant passes -> **no DEV lives** of K-P-U or K-P-0.
- KEEP / H-role FALSIFIED / NARROWED / KILLED are DEV verdicts and are **not reached** (no lives). The one-night read does not
  test H-role: the mask left the A-positions near K-P's (1.19x), so the persona was not unpinned.
- Test2 §0 arm: **K-P, unchanged**, with its G7/G8 Expected-fail standing ("no KEEP by the timebox"). The timebox bars a second mask
  rule, label cap, bank size or keep weight in this window; any lever 2 goes through `before-build` (2nd-fix check) first.

## Refuters (2: sonnet numbers, Fable design; both NOT refuted) and narrowings
- Every number recomputed from the 36 per-seed files; the verdict follows §2/§4 as written.
- K-P-0's bar-1 FAIL is marginal: 0.0057 nats (pooled dKL -1.6231 vs limit -1.6288); per seed the K-P-0 minus K-P gap is
  0.319 / 0.319 (GPU) / 0.279 (CPU), so the device mix does not create the fail.
- "Persona not unpinned" is narrowed to: A-position KL rose 12-29% on 3/3 seeds with U-position KL flat (1.176x, 1.117x,
  1.285x; s321 alone clears 1.25x); the pooled rule the prereg prescribes gives 1.195x, so UNDEFINED stands.
- The UNINFORMATIVE call is one nova row from flipping (+0.083 vs 0.10); the informative branch gives the same outcome
  (K-P-U fails bar 4 on s319's 6/96 refusals; K-P-0 still fails bar 1). G7 moves in steps of 3/12, about 4 outcomes per arm-seed.
- A2 (15:20 UTC) was written after s321's K-P and K-P-U results existed (done 11:09Z, 12:22Z). It changed only the device of
  s319/s320 and re-ran all four arms there, bars unchanged: low threat, but not blind to one seed.
- Host is not recorded in these files (`tools/where_ran.py`: unknown); devices come from paths and the strategy field.
- Lever count on K-P's identity/refusal cost: lever 1 (mask) and 1b (frameless bank) spent. Lever 2 needs `before-build`
  (2nd-fix check) and, per the prereg, a judgment against scale and the Self-and-identity prerequisite first.
