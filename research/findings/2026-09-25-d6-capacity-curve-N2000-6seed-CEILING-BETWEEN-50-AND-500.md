---
type: finding
status: live
date: 2026-09-25
lane: D6-learn-and-grow
mechanism: D6 capacity-curve scoring (D6 local Hebbian write vs host pattern copy vs write-freeze null)
seeds: [42, 43, 44, 100, 101, 102]
verdict: RECALL-HOLDS-TO-2000 for HEBB and COPY (by-construction, uninformative about capacity, per the prereg's
  own Amendment B); PARITY-BY-CONSTRUCTION at every level (HEBB is a near-copy of the host write, not evidence of
  independent learning); FREEZE null holds exactly (recall 0.0, both false rates 0.0, mean|w| 0.0) at every one of
  24 cells; per-turn cost CEILING-BETWEEN-50-AND-500 facts for both HEBB and COPY, measured entirely on shared
  pool-CPU / cloud-CPU nodes, never the project's single-consumer-RTX-3090 reference class.
runner: research/runners/d6_capacity_curve.py
builds_on:
  - research/findings/2026-09-23-d6-capacity-curve-PREREGISTRATION.md
  - research/findings/2026-09-23-d6-learn-through-use-v3-capability-gate-GO-6of6.md
artifacts:
  - research/findings/raw/_d6_capacity_curve/score.json
  - research/findings/raw/_d6_capacity_curve/s42_N5_HEBB.json
  - research/findings/raw/_d6_capacity_curve/s42_N500_HEBB.json
  - research/findings/raw/_d6_capacity_curve/s42_N2000_HEBB.json
  - research/findings/raw/_d6_capacity_curve/s100_N2000_HEBB.json
  - research/findings/raw/_d6_capacity_curve/s100_N2000_COPY.json
---

# D6 capacity curve: SCORING the 6-seed, N=2000 grid — CEILING-BETWEEN-50-AND-500 (cost); RECALL-HOLDS-TO-2000 (uninformative by construction)

**This is a SCORING pass, not a new run.** The 84-cell grid (6 seeds x N in {5, 50, 500, 2000} x arms HEBB/COPY/
FREEZE, plus a determinism replicate HEBB_REP at N=5 and 50) was harvested onto `main` at `fe1066f64` together
with its own `score.json`. This finding re-runs the pre-registered scorer against those same raw files
(`.venv/bin/python -m research.runners.d6_capacity_curve --score --arm-dir research/findings/raw/_d6_capacity_curve
--json <out>`, `CUDA_VISIBLE_DEVICES=""`) and reports what it says, with the per-seed detail and the honest limits
the committed `score.json` does not narrate. The rerun is structurally identical to the committed
`research/findings/raw/_d6_capacity_curve/score.json` (every key, every nested value, compared with `==` in
Python) — the scoring step is reproducible from the raw files alone. No new grid job ran for this finding.

## What was measured

Pre-registration: `research/findings/2026-09-23-d6-capacity-curve-PREREGISTRATION.md` (the runner's `score_grid`
implements its bands verbatim, per amendments A/B/C in that document's own AMENDMENT LOG). Per (seed, N) cell, the
scorer checks: all required arms exist, ran without error, taught exactly N facts, and share one substrate
threshold hash, one probe-spec hash, one facts hash and one `n_total` across arms (`cross_arm_identity`); the
FREEZE arm is a clean null; the HEBB lever moved; no store write occurred during the probe phase; and (at N=5, 50)
the HEBB_REP replicate agrees with HEBB on every probe decision. A cell that fails any of these is UNDEFINED, not
scored. **All 24 (seed, N) cells in this grid are `DEFINED`, all four `cross_arm_identity` checks hold, and
`freeze_null_ok` / `lever_moved` / `writes_during_probe_clean` are `true` on every one** (`score.json`,
`per_level.*.*.status` / `.cross_arm_identity` / `.freeze_null_ok` / `.lever_moved` /
`.writes_during_probe_clean`), and `rep_identical` is `true` on all 12 HEBB_REP cells at N=5 and N=50.

## The pre-registered verdict

Straight from `research/findings/raw/_d6_capacity_curve/score.json` (top-level keys):

| quantity | value |
|---|---|
| `level_label_HEBB` | RECALL-HOLDS at N=5, 50, 500, 2000 |
| `level_label_COPY` | RECALL-HOLDS at N=5, 50, 500, 2000 |
| `parity_HEBB_vs_COPY` | PARITY-BY-CONSTRUCTION at N=5, 50, 500, 2000 |
| `curve_HEBB` / `curve_COPY` | RECALL-HOLDS-TO-2000 |
| `level_label_HEBB_cost` / `level_label_COPY_cost` | COST-HOLDS at N=5, 50; COST-FAILS at N=500, 2000 |
| `curve_HEBB_cost` / `curve_COPY_cost` | **CEILING-BETWEEN-50-AND-500** |

Two curves, two different questions: the recall curve says the store answers correctly at every scale tested (an
expected, by-construction result — see below); the cost curve says the store stops being runnable as a live
chat-turn memory somewhere between 50 and 500 taught facts. The cost curve is the one that actually moves with N
and therefore the one this instrument can speak to.

## Per-seed table

Recall is 1.0 and both false-recall rates are 0.0 for HEBB and COPY on **every one of the 24 cells** — no seed,
no level, no arm deviates. Cost is the only axis that changes with N. `enc` = `encode_s_median` (seconds, per
fact taught); `rt` = `readtime_view_s_per_turn_projected` (seconds, the cost of re-reading every taught block
after one teach turn — median per-block read time x N).

| N | seed | HEBB enc (s) | HEBB rt (s) | COPY enc (s) | COPY rt (s) | HEBB cost | COPY cost |
|---|---|---|---|---|---|---|---|
| 5 | 42 | 0.2767 | 0.171 | 0.3234 | 0.2755 | HOLDS | HOLDS |
| 5 | 43 | 0.2991 | 0.184 | 0.2242 | 0.1725 | HOLDS | HOLDS |
| 5 | 44 | 0.2794 | 0.1745 | 0.2147 | 0.172 | HOLDS | HOLDS |
| 5 | 100 | 0.2753 | 0.173 | 0.217 | 0.1725 | HOLDS | HOLDS |
| 5 | 101 | 0.2802 | 0.174 | 0.2125 | 0.1725 | HOLDS | HOLDS |
| 5 | 102 | 0.2708 | 0.1695 | 0.2166 | 0.1755 | HOLDS | HOLDS |
| 50 | 42 | 0.43325 | 4.86 | 0.3292 | 3.0175 | HOLDS | HOLDS |
| 50 | 43 | 0.4392 | 3.0475 | 0.34375 | 3.055 | HOLDS | HOLDS |
| 50 | 44 | 0.4327 | 3.0325 | 0.3423 | 3.115 | HOLDS | HOLDS |
| 50 | 100 | 0.4153 | 2.87 | 0.32705 | 2.965 | HOLDS | HOLDS |
| 50 | 101 | 0.4283 | 2.96 | 0.3268 | 2.965 | HOLDS | HOLDS |
| 50 | 102 | 0.4356 | 3.09 | 0.343 | 3.025 | HOLDS | HOLDS |
| 500 | 42 | 2.1147 | 159.775 | 4.3831 | 677.9 | FAILS | FAILS |
| 500 | 43 | 9.65425 | 647.7 | 4.49525 | 662.325 | FAILS | FAILS |
| 500 | 44 | 6.18525 | 659.9 | 1.57605 | 165.2 | FAILS | FAILS |
| 500 | 100 | 6.2137 | 656.05 | 4.48825 | 669.575 | FAILS | FAILS |
| 500 | 101 | 6.3632 | 658.375 | 4.492 | 675.35 | FAILS | FAILS |
| 500 | 102 | 9.83325 | 718.875 | 6.568 | 1056.55 | FAILS | FAILS |
| 2000 | 42 | 9.19085 | 3240.7 | 33.4906 | 22433.4 | FAILS | FAILS |
| 2000 | 43 | 15.14865 | 2703.9 | 9.4624 | 4488.5 | FAILS | FAILS |
| 2000 | 44 | 9.24175 | 3078.9 | 6.8276 | 3425.5 | FAILS | FAILS |
| 2000 | 100 | 51.25485 | 26473.0 | 32.97025 | 28898.7 | FAILS | FAILS |
| 2000 | 101 | 39.39345 | 7479.7 | 7.1021 | 3489.0 | FAILS | FAILS |
| 2000 | 102 | 9.5617 | 3224.8 | 7.0265 | 4739.7 | FAILS | FAILS |

FREEZE is exact on every one of the 24 cells: `recall=0.0`, `novel_false_recall=0.0`, `nearmiss_false_accept=0.0`,
`mean_abs_w_probed=0.0` — see the FREEZE section below.

## What PARITY-BY-CONSTRUCTION means in the code, and what it does and does not license

**What the code does.** `research/runners/d6_hebbian_store.py::hebbian_encode` writes a fact by (1) composing the
same on-substrate bind/bundle output the host-copy path (`_write_block`) would read out and copy, but keeping it
as live oscillation instead of reading it to host; (2) driving the new block's D=128 readout cells through a
one-to-one **host-wired instructive ("teacher") pathway** installed fresh at unit weight for each write
(`rf_set_complex_weights(teacher)` — declared residual (a) in that module's docstring), while a **host phase-lock
loop** (`while _rf_counter % Pd != 0: _rf_advance_one()`, residual (b)) aligns the encoding window to the read's
reference phase; (3) accumulating a genuinely local, phase-coupled Hebbian correlation per trigger->readout
synapse, `dw_k = eta * z_post_k(t) * conj(z_pre(t))`, read off the bridge's own membrane-potential and
recovery-variable arrays each step; (4) **clamping every unfrozen synapse's magnitude to `W_MAX = 1.0`** (residual
(c)) — so the rule stores no graded strength, only a phase.

**Why that produces parity.** The postsynaptic target the Hebbian rule converges toward is not discovered — it is
imposed, through neural activity, by the host-wired teacher pathway; and the one free parameter that could
otherwise distinguish the two arms (synaptic magnitude) is a fixed constant shared by both arms. The result:
across every N tested, `encode_diag.n_saturated_min` is **128 of 128** synapses (every synapse in every taught
block, at N=5, 500 and 2000 alike — `s42_N5_HEBB.json`, `s42_N500_HEBB.json`, `s42_N2000_HEBB.json`,
`encode_diag`) and `encode_diag.mean_abs_w_mean` is **1.0**, matching COPY's own `summary.mean_abs_w_probed = 1.0`
(`s42_N5_COPY.json`). Both arms end each write holding a unit-magnitude complex weight at (very nearly) the same
phase — one placed there by a direct host copy, the other converged to by a local rule whose target and bound are
both host-supplied. **Parity in recall is therefore guaranteed by the write's design, not discovered by running
the grid.** This is exactly the prereg's own Amendment B(a): the accuracy gate here has no realistic failing
outcome, so a PASS is not evidence.

**What it does NOT license.** It does not license "HEBB is really learning something COPY does not" — the two
arms are constructed to land on effectively the same synaptic content, so a metric that cannot distinguish them is
not evidence that the local rule discovered anything independently. It does not license **self-organized**
(`docs/TERMS.md`): the postsynaptic target (the phase the plasticity converges to) is host-wired via the teacher
pathway, one of the two factors of the learning rule is therefore host-supplied, not discovered — the honest word
is host-instructed. It does not license **compositional**: this store is one disjoint trigger->readout block per
fact (`k_max = N + 16` sized from the start, disjoint synapse sets per block), a **localist**, one-unit-per-item
code, not a representation built from constituents.

**What it DOES license.** The local, phase-coupled Hebbian correlation computed from the bridge's own membrane
dynamics genuinely executes and genuinely converges, within one encoding window, to the pattern the teacher
pathway imposes — that is a real (if host-instructed) plasticity rule operating correctly on the substrate, not a
disguised host copy into the weight array (the weight is never written directly; it accumulates from
`z_post * conj(z_pre)` products the bridge computes itself). That is a nontrivial, verified engineering fact about
the write mechanism. It says nothing about whether the rule can do anything the direct copy could not.

**Direct answer to "is HEBB really learning anything COPY does not?"**: No, not by anything this instrument
measures. Recall is identical (1.0, every seed, every N) because both arms are built to converge on the same
synaptic content by two different routes. A genuine difference between a host-instructed local rule and a direct
host copy would have to show up on a task where the two routes diverge — degraded or partial teaching signal,
competing/interfering writes, or a superposed (non-localist) store where capacity can actually fail — and none of
those is what this grid tests.

## FREEZE as the null

`BRAIN_D6_HEBBIAN_FREEZE=1` runs the identical encoding episode (same compose, same instructive drive, same
trigger activation) with `eta=0`, so no synapse should change. On **all 24 cells**, FREEZE reads exactly
`recall=0.0`, `novel_false_recall=0.0`, `nearmiss_false_accept=0.0`, `abstain=1.0`, and
`mean_abs_w_probed=0.0` — not approximately zero, exactly zero, at every seed and every N from 5 through 2000
(`score.json`, `per_level.*.*.summary.FREEZE`; `freeze_null_ok=true` on all 24 cells). This is the cleanest part
of the grid: it confirms the write, not the probe or the routing, is what makes HEBB and COPY answer, at every
scale tested — the accuracy result above is not a probe artifact that would return "correct" regardless of
storage.

## Cost ceiling in consumer-hardware terms (one RTX 3090 chat turn)

The pre-registered cost bar (`research/runners/d6_capacity_curve.py`: `COST_ENCODE_MAX_S = 2.0`,
`COST_READTIME_MAX_S = 10.0`, `COST_MEM_GB = 24.0`) is stated directly in chat-turn terms: can one more fact be
taught, and can everything taught so far be read back, inside a turn's latency budget on the project's reference
hardware class (`project_consumer_hardware_reference_principle`: a single consumer RTX 3090)? Peak RSS never
approaches the 24 GB memory bar at any level — it stays at 126-232 MB across the whole grid
(`s42_N5_HEBB.json` through `s42_N2000_HEBB.json`, `peak_rss_mb`); **the ceiling is entirely a TIME failure, never
a memory one.**

In plain terms: at N <= 50 taught facts, teaching one more fact costs a few tenths of a second and reading
everything taught back costs 3-5 seconds — comfortably inside a live chat turn. At N=500, reading back what was
taught already costs 159.8-1056.6 seconds (2.7-17.6 minutes) per turn; at N=2000 it costs 2703.9-28898.7 seconds
(45 minutes to 8 hours). Both are categorically incompatible with a chat turn on any device, because the margin
over the 10-second budget is **16x at the smallest N=500 cell and up to 2890x at N=2000** — larger than any
plausible constant-factor speedup a GPU affords over this CPU measurement for a workload whose cost is driven by
re-reading O(N) stored blocks and by a substrate that grows by ~129 neurons per taught fact
(`n_total`: 6385 at N=5, 12190 at N=50, 70240 at N=500, 263740 at N=2000, `s42_N*_HEBB.json`). The failure is
architectural (linear-in-N per-turn cost, an ever-larger bridge), not a device-throughput artifact a faster card
would remove.

**Honesty about the measurement itself.** No cell in this grid ran on the project's consumer-GPU reference class.
Every job's `cost_hardware_class` is `"pool-cpu-shared"` (Amendment C(b)); the N=2000 cells alone landed on at
least three different, concurrently-loaded machines (`ip-172-31-47-37`, `node-dl5d243`, `node-dl4g243` —
`s42_N2000_HEBB.json` / `s42_N2000_COPY.json` / `s100_N2000_HEBB.json` `node.hostname`). That confounds any
attempt to rank HEBB against COPY by absolute time at N=2000: seed 42's HEBB ran on `ip-172-31-47-37`
(`encode_s_median=9.19085`) while its own COPY arm ran on `node-dl5d243` (`encode_s_median=33.4906`) — a 3.6x gap
that is at least partly which machine the job landed on, not the write mechanism, since COPY does strictly less
work than HEBB per fact (no Hebbian accumulation loop). **The ceiling's LOCATION (between 50 and 500) is robust to
this confound** — the fastest-observed N=500 cell (`s42` HEBB, `readtime_view_s_per_turn_projected=159.775`) is
still 16x over the 10-second bar, a margin far larger than the pool's node-to-node variance (roughly 1.6x on
`encode_s_median` at N=50 across all 6 seeds) — but the exact seconds reported here are `pool-cpu-shared` numbers,
not `"a single consumer RTX 3090's"` numbers, and should not be read as either.

**The retroactive cost note.** Seven of the eighteen N=2000 cells (`s100_N2000_{HEBB,COPY,FREEZE}`,
`s101_N2000_{HEBB,FREEZE}`, `s42_N2000_COPY`, `s43_N2000_HEBB`) carry a `cost_acknowledged` field added
**2026-09-25 at harvest**, stating plainly that no run-time cost projection was made before that cell was
launched, because the prereg's own N=2000 fit rule gates the launch decision on **memory** (drop N=2000 only if
the pre-launch resource-probe projects peak RSS past 13 GB or 120 GB) and explicitly does not treat wall time as a
drop criterion. Measured peak RSS at N=2000 stayed at 228.0-232.1 MB across every seed and arm
(`s42_N2000_HEBB.json` through `s102_N2000_COPY.json`, `peak_rss_mb`) — six to seven orders of magnitude under
either memory bound — so this procedural gap did not change whether N=2000 should have launched. It is recorded
here as an honest process deviation (no time projection preceded a set of jobs that individually ran 14.6-31.0
wall-hours on a shared pool), not a defect in the measurement those cells produced.

## What this does and does not show

**Shows:** recall holds at 1.0 for both HEBB and COPY through N=2000 facts, but this is a consequence of the
store's disjoint, one-block-per-fact design (Amendment B(a)), not a capacity result; HEBB and COPY converge on
near-identical stored synapses by construction (parity-by-construction), so the accuracy axis cannot separate a
host-instructed local rule from a direct host copy; the FREEZE null is exact at every scale tested, confirming the
write (not the probe or DG routing) drives the answer; the per-turn cost of this store crosses from usable to
unusable somewhere between 50 and 500 taught facts, on time alone, for both arms.

**Does not show:** whether the local Hebbian rule can do anything a direct copy cannot — that requires a task
where the two routes diverge, which this instrument does not construct; a capacity LAW (recall degrading with N)
— this store has none by construction, so it has nothing to compare against an LLM's parameter-count capacity
(that question belongs to the distributed-store lane,
`research/findings/2026-09-23-ca3-superposed-fact-attractor-capacity-PREREGISTRATION.md`); cross-fact
interference — the near-miss probe shares the taught fact's own DG shard by construction and does not test a
sibling fact's intrusion (prereg Amendment B(c), not re-litigated here); GPU/`SIM_BACKEND=cupy` throughput — every
cell in this grid ran `numpy` on shared CPU hosts.

One more honest limit worth naming even though it carries no threshold: `novel_shard_empty` (the fraction of
never-taught probes whose DG shard was empty before the substrate ever got a chance to answer) rises from 0.4 at
N=5 to 0.97 at N=2000 for seed 42 HEBB (`s42_N5_HEBB.json`, `s42_N2000_HEBB.json`, `summary.novel_shard_empty`).
As N grows, more and more of the "no false recall" result is explained by host DG-routing never reaching the
substrate decode at all, rather than the substrate correctly rejecting a novel query — a further reason the
recall/false-rate axis says less about the store's own discrimination as N grows, not more.

## Preconditions (the verdict above travels with these; a miss would make this scoring UNDEFINED, not a verdict)

- All 24 (seed, N) cells are `DEFINED`: every required arm present, no `error`, exactly N facts taught
  (`score.json`, `per_level.*.*.status`).
- `cross_arm_identity` (substrate-threshold hash, probe-spec hash, facts hash, `n_total`) matches across HEBB,
  COPY and FREEZE on all 24 cells (`score.json`, `per_level.*.*.cross_arm_identity`, all four keys `true`).
- `freeze_null_ok=true`, `lever_moved=true`, `writes_during_probe_clean=true` on all 24 cells (`score.json`,
  same paths).
- `rep_identical=true` on all 12 HEBB_REP cells at N=5 and N=50 (`score.json`,
  `per_level.{5,50}.*.rep_identical`) — determinism holds where the prereg requires it.
- The rescored grid is structurally identical (`==` in Python, every key and nested value) to the already-
  committed `research/findings/raw/_d6_capacity_curve/score.json` — the scoring step is reproducible from the raw
  files with no new compute.
- `tools/claim_check.py` passes on this document (every >=3-decimal number traces to one of the cited artifacts).

## Next step

The capacity-LAW question this instrument cannot answer — does recall degrade as a function of N, and at what
rate — belongs to the distributed-store lane
(`research/findings/2026-09-23-ca3-superposed-fact-attractor-capacity-PREREGISTRATION.md`,
`research/runners/ca3_superposed_fact_attractor.py`), which builds a store where recall CAN fail with scale and
fits a degradation curve from it. This grid's own next lever, if the D6 write is pursued further, is the one
Amendment B(a) already names: a task where the host-instructed teacher pathway and the direct copy diverge, since
this grid cannot distinguish them.

## Honesty

Functional read-outs only: "recall", "learns" and "answers" mean the probe's decoded answer matches the taught
fact, measured against the write-freeze null. No claim of felt experience, and no claim that the local Hebbian
rule discovered anything beyond what its host-wired instructive pathway and clamp already determine.
