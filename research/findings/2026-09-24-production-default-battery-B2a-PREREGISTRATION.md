---
type: preregistration
status: preregistered
date: 2026-09-24
lane: load-bearing
mechanism: the #1 metric (lesion-verified load-bearing fraction) on the MERGED production default after batch 1
  (main @3d73f67df: three validated fixes default-ON, plus the 160 main commits since the flip battery's revision bd391aa31)
seeds: [42, 43, 44, 100, 101, 102]
verdict: PRE-REGISTRATION only. Filed before any b2a0924 shard exists. M1 is the commit that adds this file.
---

# Battery B2a: the load-bearing fraction on the merged production default (pre-registration)

## Why

The flip validation battery (`research/findings/2026-09-24-flip-validated-fixes-production-default-battery-GO.md`) measured
revision bd391aa31. Main now carries those flips plus 160 later commits (language, workspace, curiosity, working-memory,
perception, affect and other merges, all default-OFF or instrumentation). This battery measures the default production brain
as it actually ships, so tonight's headline is taken at the shipped revision and not inferred from an older one.

## Run (fixed)

```
python tools/lb_shard.py jobs --seeds 42 43 44 100 101 102 --tag b2a0924 --no-fixes --root ~/derisk-pool/revisions/<M1>
python tools/lb_shard.py aggregate --tag b2a0924 --seeds 42 43 44 100 101 102
```
Adequate probe set (the default), `--repeats 2`, numpy backend, no fix flag in the environment, every row in the registry at
M1. Shards run on the mini-PC pool and the AWS pool nodes, pinned to M1 by revision directory. Each shard's provenance records
the git SHA, backend and thread counts.

## Criteria (written before any shard exists)

- **R1 (no regression against the flip battery):** for every faculty in
  `research/findings/raw/_load_bearing/_shards/flipdefaults-adequate/aggregate.json`, n_load_bearing at M1 is not lower.
  numpy shards are deterministic, so no tolerance. A drop is attributed to the code merged since bd391aa31 and is bisected
  by merge before any further default flip.
- **R2 (complete):** no incomplete faculty, no dirty null control; a missing or UNRELIABLE shard makes that faculty
  UNDEFINED, never 0 and never a pass.
- **Reported (not gated):** robust core, union, mean fraction and SD over the six seeds, stated beside the thin shipped headline
  from the flip battery (0.603, robust core 15) <!--derived--> as the owner-ratified pair.
