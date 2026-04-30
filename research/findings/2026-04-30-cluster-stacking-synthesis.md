# 2026-04-30 — Cluster-stacking strategy empirically falsified — synthesis

**TL;DR:** 9 cluster-stacking attempts past A+E (the operational ceiling
at 6.97 ± 0.83 multi-goal det / 3.31 ± 0.74 single-goal det) all NEUTRAL
or NEGATIVE. Five hurt cheat-5 outright; one (D v2) gives a variance
reduction without breaking the ceiling; rest neutral. A+E robustly
sits at the ceiling regardless of additional biology buildouts.

The cluster-stacking hypothesis (build out missing biology incrementally,
each piece adds complementary capacity) is empirically falsified for
the cheat-5 benchmark.

## The 9 attempts (chronological)

| # | Stack | Mechanism added | Result | vs A+E mean |
|---|---|---|---|---|
| 1 | A+D | Hippocampus trisynaptic loop | NEUTRAL | +0.65 |
| 2 | A+D+E | Hippocampus + topography | NEUTRAL | similar |
| 3 | A+F (F v1) | Cerebellum (Marr-Albus-Ito), reward-gated | NEUTRAL | +0.40 |
| 4 | A+E+F (F v1) | + topography | NEUTRAL | +1.05 |
| 5 | A+F v2 | CF-gated anti-Hebbian LTD (Albus 1971) | NEGATIVE | +14.80 |
| 6 | A+E+F v2 | + topography | NEGATIVE | +17.91 |
| 7 | A+E+D (sleep) | + sleep replay window | NEGATIVE | +22.35 |
| 8 | A+E+D+v2 | + SWR-gated CA3 plasticity | PARTIAL | +20.71 (vs +22.35 baseline; std cut) |
| 9 | A+E+C v2 | Compartmentalized DA (per-action channels) | NEGATIVE | +2.29 (multi) / +1.74 (single) |

A+E itself was Step 4 of the original buildout sequence. After A+E was
established, every additional cluster attempt has either done nothing
or actively hurt performance.

## What the cluster-stacking strategy assumed

- **Composition:** Each biology cluster (A=closed loop, B=striatal IN,
  C=DA modulation, D=hippocampus, E=topography, F=cerebellum) adds
  capacity that the others lack
- **Independence:** Failure modes between clusters are orthogonal
- **Net positive:** Combined effects > sum of parts

## What the data actually shows

- **Compositional failure.** Adding clusters past A+E doesn't compose;
  most stacks stay flat or regress. The mechanisms apparently interfere
  rather than complement at our scale.
- **A+E is doing all the work.** The closed BG loop (A: cortex→stn
  hyperdirect + thal→cortex feedback) provides credit-assignment
  topology + the topographic maps (E) provide spatial selectivity. Other
  clusters were trying to add credit-assignment mechanisms (compartment-
  alized DA, hippocampal replay) on top of one that already works.
- **Failure modes are scale-bound.** F v2 NEGATIVE was attributed to
  PF/PC scale mismatch (~64 vs biological ~150K). D v2 needed scheduled
  windows because endogenous CA3 bursts didn't fire at our scale. C v2's
  per-action DA likely needs a channelized cortex (which we don't have).
  At biological scale these mechanisms might function; at ~1500 neurons
  they don't.

## Why this matters

This is the third major strategic falsification on the project:
1. (Earlier 2026) Runner-side hacks for the silent-motor trap (V1-V7 in
   sessions D-I, all NEGATIVE). Pivot was the BG cascade architecture.
2. (2026-04-26) Refinement variants on Phase B (asymmetric DA, WTA,
   per-action DA, learned perception cold-start). All seed-dependent or
   NEGATIVE except surprise-LR-boost. Pivot was perception arc + curriculum.
3. (2026-04-30, this finding) Cluster-stacking. Pivot pending.

Each strategic falsification has unblocked a real architectural advance.
What's the pivot here?

## Three candidate pivots (with cost/benefit)

### Pivot A: Scaling

Scale the model 5-10×. Several clusters were specifically flagged as
"scale-bound" — F v2 because of the PF/PC mismatch, D v2 because CA3
recurrent dynamics don't self-sustain at 100 neurons. At biological
scales (CA3 ~150K, granule ~50M, cortex columns ~10K each), the
mechanisms might compose differently.

- **Cost:** 5-10× wall-clock per run. Tier-3 of any cluster goes from
  ~30 min → ~3-5 hours. Need to verify all kernels work at scale.
- **Benefit:** Tests whether failures are "bad biology" or "bad scale."
  Could re-enable failed clusters.
- **Confidence:** Medium. Some clusters might genuinely need scale
  (cerebellum's PF expansion is famously a scaling argument); others
  (compartmentalized DA) probably don't scale-fix.

### Pivot B: Harder benchmark

Cheat-5 multi-goal det has A+E at ~7.0 mean Manhattan distance — the
agent is near-optimal. Biology buildouts that would matter for *harder*
tasks (compositional planning, partial observability, longer credit
chains) don't show up because the task is already solved.

- **Cost:** Design + implement + validate a new benchmark. ~1-2 days.
- **Benefit:** If cluster benefits emerge on harder tasks, it confirms
  the buildout strategy was right but the benchmark wasn't testing it.
- **Confidence:** Medium-high. There's clear precedent in animal cognition
  research that biology shines under harder conditions.

### Pivot C: Replay content (smallest scope)

The SCIENCE_ROADMAP §4.7 already noted "content quality is the bottleneck"
for sleep replay. We tested gating (D v2) and found PARTIAL signal. The
*content* lever is untouched — reverse-order trajectories (Foster &
Wilson 2006), recency-weighted sampling, current-goal-only filtering.

- **Cost:** ~30-50 LOC. ~1-2 hours.
- **Benefit:** If sleep replay can be made beneficial (it currently
  HURTS A+E+D), then D becomes useful and might compose with other
  clusters differently.
- **Confidence:** Low. The fundamental issue may be that during cheat-5
  multi-goal, replay is replaying *stale* trajectories. Reverse order
  doesn't fix staleness.

## Pivot B attempted (2026-04-30): different benchmarks/metrics

Three harder-benchmark variants tested at A+E (n=3-6 each, sleep replay off):

| Schedule | Sum mean | Per-phase | Phases | Note |
|---|---|---|---|---|
| multi (corner, 4×450) — current | 8.45 | **2.11** | 4 | reference |
| random (uniform, 4×450) | 6.16 | 1.54 | 4 | EASIER (random ~5.5 vs corner ~10 from start) |
| multi-fast (corner, 8×225) | 11.05 | 1.38 | 8 | per-phase EASIER (sum bigger because more phases) |
| random-far (≥8 manhattan, 4×450) | 7.69 | 1.92 | 4 | similar to multi |

**Per-phase, the existing `multi` is already the hardest of the 4.** Corner
goals' long trajectories give the metric more room to accumulate distance.
The harder-benchmark hypothesis didn't pan out.

Adaptation-speed metric (first-quarter instead of final-quarter, computed
post-hoc from distance_log; now also emitted directly in phase_stats):

| Cond (multi) | First-Q sum | Final-Q sum |
|---|---|---|
| A+E | 8.13 ± 2.29 | 7.18 ± 1.58 |
| A+E+C v2 | 10.36 ± 2.04 | 9.26 ± 3.91 |
| Δ (v2 - AE) | +2.23 | +2.08 |

**C v2 hurts adaptation speed by the same amount as it hurts asymptotic
skill.** The biology buildouts don't help on adaptation-speed metric either.

So Pivot B as tried (different schedules + different metrics) doesn't reveal
hidden cluster benefits. The cluster-stacking falsification is robust across
multiple eval framings.

## Recommendation

**Pivot A (scaling) > Pivot B (more benchmark variations) > Pivot C (replay content).**

After Pivot B's null result, scaling becomes the highest expected-value
pivot remaining. The mechanism-level claims (cerebellar PF expansion, CA3
recurrent autoassociator, compartmentalized DA per striatal patch) are
explicitly scale-arguments. Testing them at biological scale (5-10× our
current ~1500 neurons) is the next falsifiable experiment.

Pivot B has the highest expected value: it would tell us whether
biology buildouts are inherently insufficient or whether the benchmark
just doesn't exercise them. If biology helps on harder tasks, the
cluster work wasn't wasted. If it still doesn't help, we have a stronger
falsification.

Pivot A is appealing but expensive in wall-clock. A 5-10× slowdown means
each experiment takes hours, which limits iteration speed. Worth doing
if pivot B succeeds and we want to tune at biological scale; not the
right next step.

Pivot C is cheap but the hypothesis is weak (content-quality fixes
don't address why sleep replay HURTS the active stack).

## What to ship as opt-in (regardless of pivot)

All cluster mechanisms are already shipped behind opt-in flags:

- `--enable-cluster-a-closed-loop` (used in flagship)
- `--enable-cluster-e-topography` (used in flagship)
- `--enable-cluster-d-hippocampus` (opt-in, NEUTRAL alone)
- `--enable-cluster-d-v2-swr` (opt-in, PARTIAL — variance reduction)
- `--enable-cluster-f-cerebellum` (opt-in, NEUTRAL)
- `--enable-cluster-f-v2` (opt-in, NEGATIVE — do not stack)
- `--enable-tonic-da` / `--enable-compartmentalized-da` (opt-in, NEGATIVE)
- `--enable-d1-d2-asymmetry` / `--enable-striatal-pv-fsi` / `--enable-tans` (B clusters, opt-in)

Flagship config remains:
```
--enable-msn-lateral-inhibition --enable-d1-d2-asymmetry
--enable-striatal-pv-fsi --enable-cluster-a-closed-loop
--enable-cluster-e-topography --deterministic
```

## Files

- All findings: `research/findings/2026-04-{27,28,29,30}-*.md`
- Detailed C v2 finding: `research/findings/2026-04-30-cluster-c-v2-results.md`
- Detailed D v2 finding: `research/findings/2026-04-30-cluster-d-v2-results.md`
- Detailed F v2 finding: `research/findings/2026-04-30-cluster-f-v2-results.md`
- Cluster strategy plan: `docs/plans/2026-04-28-cheat5-real-options-survey.md`
- Science roadmap: `docs/SCIENCE_ROADMAP.md` §4.7
