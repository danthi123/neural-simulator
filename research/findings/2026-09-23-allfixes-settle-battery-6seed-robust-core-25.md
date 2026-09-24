---
type: finding
status: live
lane: load-bearing
date: 2026-09-23
seeds: [42, 43, 44, 100, 101, 102]
verdict: Robust core 25 of 26 exercised faculties with every merged fix flag plus BRAIN_AFFECT_MARKER_SETTLE on (6 seeds, one consistent run); every seed reads 25/26 (mean 0.962, SD 0.000). affect-marker-spiking-wta joins the core (6/6, was 4/6 in allfixes2); no faculty lost a seed.
artifacts:
  - research/findings/raw/_load_bearing/_shards/allfixes3settle/aggregate.json
  - research/findings/raw/_load_bearing/_shards/allfixes2/aggregate.json
---

# All-fixes + SETTLE adequate-probe battery, 6 seeds: robust core 25 of 26 (was 24) <!--derived-->

The combined measurement the allfixes2 finding named as next
(`research/findings/2026-09-23-allfixes-adequate-battery-6seed-robust-core-24.md`): the same battery, probes and fix flags,
plus `BRAIN_AFFECT_MARKER_SETTLE=1` (the D1 SETTLE mechanism: a 500 ms deliberation window and a 1000 ms inter-turn rest;
`research/findings/2026-09-23-affect-marker-settle-fullbrain-contrast-PARTIAL-6seed.md`).

## Configuration
- `tools/lb_shard.py jobs --tag allfixes3settle --extra-env BRAIN_AFFECT_MARKER_SETTLE=1` — 31 faculties x 6 seeds = 186
  shards, `--repeats 2`, `SIM_BACKEND=numpy`, every allfixes2 fix flag and adequate probe on.
- Compute: two AWS r7i.4xlarge (s42-s44 and s100-s102), 15 shards in parallel each, code snapshot of main with
  `data/corpus/`. As in allfixes2 the instances were synced without `.git`, so shard provenance records
  `git_sha: unknown`; the revision is asserted by the deploy procedure (main at launch, containing the SETTLE merge).

## Result
<!--derived from research/findings/raw/_load_bearing/_shards/allfixes3settle/aggregate.json -->
- **Exercised: 26 faculties on every seed. Robust core: 25 of 26.** Per-seed load-bearing fraction 25/26 = 0.9615 on all
  six seeds (mean 0.962, SD 0.000; allfixes2: 0.949 ± 0.018, robust core 24).
- `affect-marker-spiking-wta`: 6/6 (allfixes2: 4/6). This is the first time SETTLE's effect is measured inside the combined
  battery; it matches the separate contrast's 6/6-with-SETTLE.
- Every other faculty has the same load-bearing seed count as allfixes2; no faculty lost a seed and no exercised faculty
  has a dirty null control.
- Outside the core: `da-gated-encoding` 0/6 (its chat wiring is in a fix round; the merged DA v3 mechanism is not yet in
  the battery's path). Not exercised (thin probes, not in the denominator): `swap-drives-response`,
  `wm-binding-advanced` (adequate probes in fix rounds).

## Option-C report (owner-ratified pairing)
**Robust core 25/26 exercised under the adequate probes with every merged default-OFF fix flag plus SETTLE on**, paired with
the shipped default (~0.59 thin-probe, all flags off, 2026-09-20). Three of the fixes are now switched on by default on
branch `research/flip-validated-fixes` pending its production-default validation battery (running); SETTLE itself is a
candidate for the next flip batch now that it has a clean combined battery (its per-turn cost — a longer deliberation
window and rest — still needs measuring against the consumer-hardware reference before it ships).

## Honest residuals
- Code revision not recorded in shard provenance (see Configuration).
- One configuration, one probe set; 25 is not a production-default number.
- A 26/26 core needs `da-gated-encoding` to become load-bearing; `swap-drives-response` and `wm-binding-advanced` enter
  the denominator only once their adequate probes pass review.

## Honesty
Functional read-outs only: "load-bearing" means lesioning the faculty's spiking contribution provably changes the reply
with a clean null control. No felt or phenomenal claim.
