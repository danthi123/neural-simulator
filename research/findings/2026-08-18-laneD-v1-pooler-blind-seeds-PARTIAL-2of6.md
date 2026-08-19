---
type: finding
status: contributing
date: 2026-08-18
mechanism: perception-v1-pooler-trace-invariance
runner: research/runners/_laneD_v1_pooler_trace_invariance_derisk.py
artifacts:
  - research/findings/raw/lanes/perception/v1_pooler_trace_sidecar_6seed_yaxis_localdiv_ncol240_k8.json
  - research/findings/raw/lanes/perception/v1_pooler_trace_sidecar_6seed_yaxis_localdiv_ncol240_k8_held2.json
---

# lane D perception: the V1-pooler trace-invariance operating point is TRACE-ROUTED-PARTIAL only 2/6 at blind seeds, and NO-GO 0/6 under a harder held-2-position task — the tuned op-point is NOT a robust GO

<!--derived-->
**One-line verdict.** The 2026-08-02 lane-D work found the default routed trace pass NO-GO and reached a best
operating point of TRACE-ROUTED-PARTIAL-2/3 on the scout seeds (42/43/44); 2026-08-07 then showed homeostatic
scaling REGRESSES it (not the missing companion process). This session took that same operating point (y-axis,
`local_orient_div`, `n_col=240`, `k_win=8`, `pool_lr_pot=0.08`, `pool_lr_depress=0.01`, `trace_decay=0.75`) to
6 seeds — adding three BLIND validation seeds (100/101/102) — and to a harder task. The result: at held-1-position
it is **TRACE-ROUTED-PARTIAL-2/6** (only seeds 42 and 44 pass; blind 100/101/102 and scout 43 all fail), and at
held-2-positions it collapses to **TRACE-ROUTED-NOGO 0/6** with the mean trace margin going negative. The 2/3 scout
was seed-lucky; this operating point does not generalise. Next step is the learned-decorrelation pooler lever (built
separately), not further op-point / pool-size tuning. No `sim/` edit — additive runner flags, `SIM_BACKEND=numpy`.

## Result 1 — held-1-position, 6 seeds: PARTIAL 2/6, blind seeds all fail

<!--derived-->
Artifact: `research/findings/raw/lanes/perception/v1_pooler_trace_sidecar_6seed_yaxis_localdiv_ncol240_k8.json`
(`summary.overall_verdict` = `TRACE-ROUTED-PARTIAL-2/6`, chance 0.3333). Per-seed GO
(`summary.per_seed_trace_go`) for seeds [42, 43, 44, 100, 101, 102] = [GO, no, GO, no, no, no]:

| seed | class | trace_go |
|---|---|---|
| 42  | scout | **GO** |
| 43  | scout | no |
| 44  | scout | **GO** |
| 100 | blind | no |
| 101 | blind | no |
| 102 | blind | no |

<!--derived-->
Group-level means (from `summary`): `v1_pooler_trace_heldpos_decode_mean` 0.3611 vs chance 0.3333 — only ~0.03 above
guessing; `v1_pooler_trace_margin_mean` 0.0673 does beat `shuffled_temporal_margin_mean` 0.0083, and the
`v1_complex_margin_mean` -0.0219 / `no_learning_margin_mean` 0.0019 controls stay flat, so the trace rule is doing
SOMETHING at the group level — but 4 of 6 seeds miss the per-seed GO gate, and all three held-out (blind) seeds fail.
The seed-42 GO is the strongest (per-seed trace held-position decode 0.5, held-train margin 0.1549, delta-vs-shuffled
0.1575). A 2/6 that is 2/3-on-scout and 0/3-on-blind is not a robust GO — it is an operating point that happened to
fit two of the three seeds it was tuned on.

## Result 2 — held-2-positions, 6 seeds: NO-GO 0/6, margin goes negative

<!--derived-->
Artifact: `..._ncol240_k8_held2.json` (same op-point, `n_held_pos=2` instead of 1;
`summary.overall_verdict` = `TRACE-ROUTED-NOGO`, `summary.per_seed_trace_go` all False). Under the harder
generalisation the invariance disappears entirely: `v1_pooler_trace_heldpos_decode_mean` 0.3194 (below the 0.3333
chance floor), and `v1_pooler_trace_margin_mean` -0.0491 — the held-to-train margin is now NEGATIVE, i.e. held-out
positions are LESS similar to the trained class than to other classes. Controls: `shuffled_temporal_margin_mean`
0.0063, `v1_complex_margin_mean` -0.0203, `no_learning_margin_mean` -0.0052. So generalising over two unseen
positions rather than one erases the effect; the trace pooler is not learning a position-invariant code, it is
fitting the single held-out position that sits nearest the training band.

## Where this leaves lane D <!--derived-->

<!--derived-->
Consistent with and extending the prior record:

- `2026-08-02-laneD-v1-pooler-trace-route-NOGO-default-3seed-needs-op-point-or-normalization.md` — default NO-GO,
  best op-point PARTIAL-2/3, seed-43 margin named as the remaining problem. At 6 seeds that op-point is 2/6 and the
  seed-43 failure persists, joined by all three blind seeds.
- `2026-08-07-laneD-v1-pooler-trace-homeostatic-scaling-REGRESSES-not-the-companion-process.md` — homeostatic scaling
  made the code LESS position-invariant. So op-point tuning, pool-size expansion, and homeostasis have each now failed
  to make the trace route robust.

<!--derived-->
The raw Foldiak trace rule at a tuned operating point is therefore banked NOT-a-robust-GO for position invariance on
this V1 front end. The next lever is the **learned-decorrelation pooler** (a competitive / common-mode-removing pooler
objective, being built separately) rather than more sweeping of `n_col` / `k_win` / learning-rate / homeostasis, all
of which are now measured insufficient. This is a lane-D perception de-risk, off the critical conversation path.

## Reproduce <!--derived-->

```bash
# held-1 (PARTIAL 2/6)
SIM_BACKEND=numpy .venv/bin/python -m research.runners._laneD_v1_pooler_trace_invariance_derisk \
  --seeds 42 43 44 100 101 102 --position-axis y --complex-norm local_orient_div --n-col 240 --k-win 8 \
  --pool-lr-pot 0.08 --pool-lr-depress 0.01 --trace-decay 0.75 \
  --out research/findings/raw/lanes/perception/v1_pooler_trace_sidecar_6seed_yaxis_localdiv_ncol240_k8.json
# held-2 (NO-GO 0/6): add  --n-train-pos 4 --n-held-pos 2  and the _held2 out path
```

GO gate (per seed, not met at the group level): held-position decode >= chance + margin, grouped trace margin beats
shuffled-temporal, V1-complex, and no-learning by their deltas, and per-image pixel scramble does not decode.
