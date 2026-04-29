# 2026-04-29 Overnight Progress Summary

**Status:** active, autonomous overnight session per user directive (no stopping until told).
**As-of timestamp:** ~02:40 EDT

## What landed (commits)

### 1. Catalog-driven remediation pass — 12 corrections (11 implemented + 1 design-doc deferral)

Plan: [`docs/plans/2026-04-29-catalog-remediation-pass.md`](../../docs/plans/2026-04-29-catalog-remediation-pass.md).

| Commit | Item | Description |
|---|---|---|
| `82b3d0d` | R1.1 | per-region E_inh override (MSN −60, SNc DA −55) |
| `a1765b0` | R1.2 | FSI cross-action wiring (replaces anatomically-backwards MSN→MSN) |
| `1521a9b` | R3.5 | sparse + decorrelated cortex→MSN (Bolam 2000) |
| `dfa9d15` | R3.10 | SNr→SNc disinhibition pathway |
| `b359bb1` | R3.7 | GPe split PV+ (prototypic) / PV− (arkypallidal) |
| `0e041e3` | R3.11 | striosome (patch) / matrix split |
| `35f1908` | R3.8 | HH_GPI_OUTPUT NaP+SK+Ih tuning |
| `8461a03` | R2.3 + R3.12 | striatal interneuron taxonomy + CA3 SWR framing |
| `bdb6452` | R3.6 | D1/D2 neuropeptide arms (KOR/NK-1/DOR/MOR) |
| `23b38fc` | R2.4 | asymmetric aversive reward magnitude |
| `befc1d0` | R3.9 | MSN KIR2/Kv2 design-doc deferral |

340 tests pass post-remediation.

### 2. New clusters (4 scaffolded + 1 designed)

| Commit | Cluster | Status |
|---|---|---|
| `2d8be00` | A — closed BG loop (`--enable-cluster-a-closed-loop`) | scaffolded; eval running |
| `01fddf4` | C v1 — tonic DA (`--enable-tonic-da`) | scaffolded; eval running |
| `3204c3e` | D v1 — hippocampus trisynaptic loop (`--enable-cluster-d-hippocampus`) | scaffolded; eval running |
| `b3f5f87` | C v2 — compartmentalized DA (`--enable-compartmentalized-da`) | scaffolded; eval queued |
| `dd14fed` | E — topographic maps | designed; deferred |

## What's running

Four chained background evals (each 6 runs × 1800 steps × multi-goal × 3 seeds):

1. **Cluster A eval** (`by1dv294e`) — baseline (no-A) vs +A, started 02:10. ETA done ~03:25.
2. **Combo eval** (`bki1vpptr`) — A+C+B.3 vs C-only, chained. ETA start ~03:25, done ~04:25.
3. **Cluster D eval** (`b6hu3ndsf`) — D-only vs A+D, chained. ETA start ~04:25, done ~05:25.
4. **C v2 eval** (`briwtq7yh`) — C v2 only vs A+C v2, chained. ETA start ~05:25, done ~06:25.

C v2 subagent (`af7014d0f2011f204`) completed at `b3f5f87` — implementation, 6 new tests, 50-step smoke all PASS.

## Eval matrix at completion (24 data points)

8 conditions × 3 seeds (42, 43, 44):

- baseline (post-remediation, no Cluster A/C/D)
- +A (closed loop)
- +C v1 (tonic DA)
- A + C v1 + B.3 (full triple)
- +D (hippocampus)
- A + D (closed loop + hippocampus)
- +C v2 (compartmentalized DA)
- A + C v2 (closed loop + per-action DA)

Aggregator: `python -m research.runners.aggregate_2026_04_29_evals`.

## Early result (n=1, seed 42 only)

Bit-identical between baseline and +Cluster A: both sum to **22.39**. Action logs DIFFER (first divergence at step 1) but the agent ends up at identical positions in the final 25% of each phase. This means the metric is partially insensitive to mid-phase trajectory at seed 42 — could be a metric saturation issue, or could mean Cluster A genuinely doesn't help at this seed. n=2 + n=3 (seeds 43, 44) will clarify.

## Methodology notes

- The "9.50 ± 0.85" baseline documented in CLAUDE.md (B.1+B.2 alone, multi-goal) does NOT reproduce at seed 42 in current code (gives 22.03–22.39 across runs). Bisect to pre-B.3 commit `714bc29` reproduced 21.22 — predates this session entirely. Some commit between when 9.50 was measured and the present has regressed multi-goal performance, but I haven't bisected to find it.
- For all current cluster comparisons, use **fresh current-code baselines** (per-condition runs in this eval matrix). The historical numbers in CLAUDE.md are advisory only.

## Decision tree (after evals land)

```
cluster A eval mean drop ≥ 1.0 vs baseline?
├── YES: tier-3 (6-seed) validation on +A; cheat-5 PARTIAL CLOSURE
├── PARTIAL: composability with C v1 helps?
│   ├── YES: tier-3 validation on A+C+B.3; cheat-5 PARTIAL CLOSURE
│   └── NO: dispatch C v2 (compartmentalized DA) implementation already in progress
└── NO: composability with C v1 / D / A+D might still help — check all
    └── If NO across all: implement C v2 + retest, then E (topography), then deeper investigation
```

## Files / commits ahead of origin

This session has produced 18+ commits on `main` from `9bb0371`:

- 11 R-pass commits + plan + propagation
- 5 cluster scaffold commits (A, C v1, D v1) + 3 design docs (A, C v1, C v2, D v1, E)
- Aggregator script + this summary

Branch is many commits ahead of origin/main; **NOT pushed** per project policy.

## Pending after evals

- Synthesize 18-data-point matrix → findings doc
- Update CLAUDE.md flagship recommendation if any cluster ships
- Decide on C v2 / E / Cluster D v2 (SWR generator) based on results

The user instruction is "no stopping until I tell you" — will continue iterating after eval completion.
