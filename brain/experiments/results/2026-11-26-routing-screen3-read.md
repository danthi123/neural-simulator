# Routing screen 3 read. Part X: keeping permanent episodes out of the S_all recall competition is KILLED, because the freed rows fetch nothing. Part J: pending

Prereg `brain/research/2026-11-26-routing-screen3-prereg.md` (two-lens review amendment before data). Build
42fff1b23. Data are in `2026-11-26-routing-screen3/`: `x_s*.json`, `x_verdict.json`, `x_done.txt`, `run_x.sh`.
Offline on arm S's own night snapshots, pool42 (node-dl5d243), CPU fp32, at about 5 min per seed. DEV, 3 seen seeds,
fetch level only.

## Part X

### Validity
- **Reproduction.** K reproduced the banked S fetch on every row of every night: 0 void nights, 0 dropped rows on all
  3 seeds. Its counts equal the live ones:
  - right fetches: 18, the same quantity as the live intact-frozen right fetches (18).
  - fetches of any permanent episode: 65 = 24 / 20 / 21. The live count of 65 counted only the self-name episode,
    so the numbers match but the definitions differ slightly.
- **Engagement.** KX fetched a permanent episode on 0 rows. There are 130 self rows (told rows where K's weighted
  winner is permanent), above the 10 required.
- **First launch.** The first launch, at 00:34 UTC, failed at once (rc 1, no output), because it loaded the default
  0.1B cortex without `--model 1.5b`. It was rerun with the flag, and that rerun is the data
  (`x_done.failed0.txt` on the node).

### Bars (KX vs K, told rows)
| | right | wrong | wrong non-perm | no fetch | fetched permanent | fires (fire rows) |
|---|---|---|---|---|---|---|
| K s319 / s320 / s321 | 6 / 7 / 5 | 31 / 23 / 24 | 7 / 3 / 3 | 43 / 40 / 44 | 24 / 20 / 21 | 31 / 23 / 27 |
| KX s319 / s320 / s321 | 7 / 9 / 8 | 13 / 6 / 3 | 13 / 6 / 3 | 60 / 55 / 62 | 0 | 18 / 3 / 3 |

**Self rows (130).** On these rows KX goes to right on 6, to wrong on 8, and to no fetch on 116.

**Kills:**
- X1: right 24 < 28 (K 18).
- X3: wrong non-permanent rises by 6 on s319 (the maximum is 5).
- X5: self rows going right are 6/130, below 1/3.
- X7: told no-fetch rises by 17 / 15 / 18 (the maximum is 5).

X2 holds: right rises on 3 of 3 seeds. X4 holds: fires fall. X6 holds: self rows going wrong are 8/130.

**Decision: KILLED.** No live SX life runs.

### What it says
- **With the self episode removed, the right fetch mostly does not appear.** Of the 130 self rows, 14 change to an
  episode (6 right, 8 wrong) and 116 go to no fetch. The self episode was capturing those rows, but on most of them
  nothing else would have fired either.
- **Why nothing fires is not measured here.** The harness does not log the right episode's weight, and the slept bar
  and the margin rule are not separated. That the told episodes' weights are too low (screen 2: median 0.36 on the KR
  stores) is an inference, not an observation from this run.
- **Report only.**
  - Fires on fire rows (open, greet, never-told, untold near-duplicate) fall from 81 to 24 pooled.
  - Excluding the permanent episodes removes most of the S key's false fires.
  - Identity rows lose the self episode by construction. This is not counted here.

## Skeptic, part X (sonnet, fresh context, REFUTER-LENS block)
"KILLED is CONFIRMED." Every number matches the rescore. Three wording fixes were applied above: the 65's
definition, "never what blocked" and "too low weight". Lenses: (a) the switch engaged (KX fetched a permanent
episode 0 times against K's 65); (b) offline only, no flip, though told no-fetch rose by 50 pooled; (c) pool42 per
`x_done.txt`, with the inputs held on the node.

## Part J
Pending. The J matrices are computing on AWS g5 (batch 20261125-1937). Then the rank ceiling runs on the KR snapshots.
