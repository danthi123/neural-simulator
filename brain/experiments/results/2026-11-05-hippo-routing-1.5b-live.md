# Hippocampal routing fix at 1.5B, live brain, six seeds: NOT A LEAD (F3 fails); the cost left is reinstatement after sleep

Capability: memory selection (pattern completion). Pre-registration: `brain/research/2026-11-03-hippo-routing-1.5b.md`
(F1-F5, written and amended before any run). Data: `results/2026-11-03-hippo-routing-1.5b/live/hroute_s*.json`,
table `live/table.txt` (`python -m brain.experiments.hippo_route_live_table <dir> --glob 'hroute_s*.json'`).
Where ran: AWS g5 (NVIDIA A10G), instance i-00898921af308af13, batch 20261103-1735
(`_scratch/brainv2/aws/20261103-1735/run.log`; the JSONs record no host, so `tools/where_ran.py` says unknown).
1.5B cortex, `--steps 480 --lr 5e-4 --kl-weight 3`, `--fix-key S_all --fix-bar size`, seeds 42/43/44/100/101/102,
about 24 min per seed, `cuda_peak_bytes` 11.37 GB on every seed (< 24 GiB, all valid). Arms a-e match the
live15 means (a 4.50, b 2.83, c 9.33, e 0).

## Result (recall_correct of 12 told questions, per seed)

| arm | per seed | mean |
|---|---|---|
| a intact + sleep (S0 key) | 5 5 4 5 4 4 | 4.50 |
| b intact, no sleep (S0) | 2 3 3 2 4 3 | 2.83 |
| c hippocampus off after sleep | 10 8 7 11 9 11 | 9.33 |
| **f fix + sleep (S_all, 1.5B bar)** | 7 8 7 9 7 8 | **7.67** |
| **g fix, no sleep** | 4 5 5 4 6 5 | **4.83** |

Paraphrase: f 6.33, c 6.00, a 1.50. False recall (of 4): f 3.67, a 4.00, g 0.50, b 3.00.
Routing of told questions (right, wrong): g 9/0 on every seed; f 5-7/0; a and b 6/5 on every seed.

- F1 (instrument, arm g routes >= 8 right and <= 1 wrong on every seed): **holds** (9/0 x6, as the CPU measured).
- F2 (f - a >= 2, f > a on >= 5/6): **holds** (+3.17, 6/6).
- F3 (f >= c - 1): **fails** (7.67 < 8.33).
- F4 (false f <= a + 0.5): **holds** (3.67 <= 4.50).
- F5 (g - b >= 2, g > b on >= 5/6): **holds, exactly at the bar** (sums 29 vs 17, difference 12/6 = 2; 6/6). The
  first table printed False because it compared float means (1.9999999999999996); the table now compares exact
  means (`fractions.Fraction`), pinned by `brain/tests/test_hippo_route_live_table.py`, which fails on the old code.
  This changes no verdict.

Verdict as registered: **not a lead** for `key="S_all", key_bar="size"` (F3 fails). Brain defaults unchanged.

## What the run shows (diagnosis, not a criterion)

The fix removes the misrouting: no wrong episode fires in f or g on any seed, versus 5 of 12 with the S0 key, and
the hippocampus now helps instead of hurting (f > a and g > b on 6/6). What is left is a cost of reinstating the
RIGHT episode into a cortex that already learned it in sleep. Over the 72 told questions (12 x 6 seeds), joined
with arm c on the same questions:

| f after sleep | n | correct in f | correct in c (same questions) |
|---|---|---|---|
| right episode reinstated | 38 | 23 | 33 |
| nothing fired | 34 | 23 | 23 |

When nothing fires, f is the slept cortex alone (identical counts). When the right memory is reinstated on top of
consolidation, 10 answers are lost. This matches the weekly preview's "intact degenerates on reinstated episodes".
The reinstatement gain (4) was fitted on an UNSLEPT 1.5B cortex (`hippo_gain_probe.py`), where it is right; after
sleep the same memory arrives twice (adapters + reinstated delta). Caveat: f's own routing fires less after sleep
(5-7 right vs 9 unslept) because the bar was fitted on unslept reads (known limit in the pre-registration).

## Next (method, not capability)
The capability is not closed: the hippocampus must help, or at least not cost, after sleep. Candidate methods, to
pre-register before any run: (1) a reinstatement gain that falls with consolidation (fit on slept reads, dev bank);
(2) reinstate only when the slept cortex's own answer confidence is low, which is the same "do I know this" signal
the lesson arbitration needs (`HANDOFF.md`). (2) serves both problems; measure the cortex-confidence signal first.
Record for that design: `research/findings/2026-08-06-replay-consolidation-selective-CA1-to-cortex-reinstatement-research-gate.md`
(old track: complementary learning systems, selective reinstatement) and
`2026-08-20-sleep-replay-consolidation-6seed-GO-on-the-principle-host-hippocampus-scaffold.md`.
