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

**This is a SCORING pass, not a new run.** The 84-job grid (6 seeds x N in {5, 50, 500, 2000} x arms HEBB/COPY/
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

**What the scorer code actually checks.** `d6_capacity_curve.py::_parity_label` returns `"PARITY-BY-CONSTRUCTION"`
when, for all 6 seeds at a level, `score_cell`'s `parity` field is `True` — `parity = abs(recall_diff_HEBB_minus_COPY)
<= PARITY_TOL and dfalse <= PARITY_TOL` (`PARITY_TOL = 0.05`), computed from the RECALL and false-rate summary
numbers of each arm. **The scorer never loads or compares `store_synapses` / weight arrays between a HEBB job and
its COPY counterpart** — the `"BY-CONSTRUCTION"` suffix is a hard-coded relabelling introduced in Amendment B, not
a per-cell weight check the code performs. The claim that the two arms hold near-identical *synaptic content*
rests on ONE direct weight measurement, cited in the PREREGISTRATION's own Amendment B text (not stored in any
JSON artifact — a one-off reviewer comparison, not a runner output): s42 N=5, complex correlation 0.99996, max
`|dw| = 0.031`, all 128 synapses saturated at `W_MAX` — not on a check repeated across the grid. <!--derived-->

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
(`s42_N5_COPY.json`). Both arms end each write holding a unit-magnitude complex weight — one placed there by a
direct host copy, the other converged to by a local rule whose target and bound are both host-supplied. **Equal
MAGNITUDE at every N and seed is what the grid itself supports** (`encode_diag.n_saturated_min` / `mean_abs_w_mean`,
above); **near-identical PHASE ("very nearly the same phase") is the s42-N=5 Amendment B measurement, not something
this grid re-checks at every cell.** Parity in recall is therefore guaranteed by the write's design, not discovered
by running the grid. This is exactly the prereg's own Amendment B(a): the accuracy gate here has no realistic
failing outcome, so a PASS is not evidence.

**What the grid additionally supports (checked for this fix round, independent of `_parity_label`).** Comparing
every HEBB job against its COPY counterpart on the exact probe DECISIONS (`answer`, `yn`, `block_decode`) recorded
in the raw `probes.{taught,nearmiss,novel}` arrays, across all 24 (seed, N) cells: 0 mismatches over 4,590 probe
records (13,770 field comparisons). HEBB and COPY answer identically on every probe run in this grid — a
decision-level fact, not a phase measurement, and not something `_parity_label` itself checks.

**What it does NOT license.** It does not license "HEBB is really learning something COPY does not" — the two
arms are constructed to land on effectively the same synaptic content, so a metric that cannot distinguish them is
not evidence that the local rule discovered anything independently. It does not license **self-organized**
(`docs/TERMS.md`): the postsynaptic target (the phase the plasticity converges to) is host-wired via the teacher
pathway, one of the two factors of the learning rule is therefore host-supplied, not discovered — the honest word
is host-instructed. It does not license **compositional**: this store is one disjoint trigger->readout block per
fact (`k_max = N + 16` sized from the start, disjoint synapse sets per block), a **localist**, one-unit-per-item
code, not a representation built from constituents.

**What it DOES license — corrected.** The per-step correlation `z_post(t) * conj(z_pre(t))` is computed from the
bridge's own membrane-potential and recovery-variable arrays (`v`, `u`), read fresh at every one of the `Pd + 8`
steps of the encoding window (`d6_hebbian_store.py::hebbian_encode`) — the INPUTS to the rule are genuine,
evolving substrate activity, not a value read once and copied. But the accumulation loop itself
(`S += lr * post * np.conj(pre)`) and the magnitude clamp that follows it run in **host Python, in the runner
module** — not inside a `sim/` kernel — exactly as the module's own docstring already names (declared residuals
(a)-(c)). The resulting `w` is then written into the substrate by `OneBrainComposer._write_block`
(`one_brain_composer.py::_store_composite`), the SAME host function the COPY arm calls with
`self._compose_phases(...)` instead. So: the write is not a disguised copy of one *static* value (the inputs
genuinely accumulate over the window, and `w` is never assigned directly from `_compose_phases`'s own output the
way COPY's write is) — but the arithmetic that turns those genuine per-step reads into a weight update is host
computation over substrate state, not substrate-native computation. Under the project's brain-based-only standard
(CLAUDE.md non-negotiable 1), this write mechanism is a documented HOST SHORTCUT, exactly like the instructive
pathway and the clamp it already names — not an exception to them. The grid also does not verify the accumulation's
within-window CONVERGENCE dynamics (it records only end states via `encode_diag`). Calling the resulting
per-fact write mechanism itself "a verified engineering fact" is fair; calling it evidence the local rule operates
independently of host arithmetic, or calling its correctness "a verified engineering fact" without that
qualification, is not — it says nothing about whether the rule can do anything the direct copy could not.

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

**A correction to the prereg's declared zero-block value.** The PREREGISTRATION's declared-shortcut paragraph
states "a zero (frozen) block decodes to the codebook's first word (`ac00`, an action)". The raw FREEZE data
contradict this: the zero block decodes to a single word, repeated across all three roles, that is SEED-SPECIFIC
and never `ac00` in any of the 6 seeds checked (`probes.taught[0].block_decode`, N=5 FREEZE): `s42` -> `pt145`/
`pt145`/`pt145`; `s43` -> `ag193`; `s44` -> `ac44`; `s100` -> `pt081`; `s101` -> `ac44`; `s102` -> `ag108`. This
does not change the scored verdict — a single repeated word can never match both the taught agent cue and the
taught action cue (those word pools are disjoint), so FREEZE still cannot register a false accept regardless of
which fixed word the zero block decodes to, and FREEZE therefore cannot detect a partial or weak write — but the
prereg's specific `ac00` claim about which word the substrate actually reads out is wrong, and neither the prereg
nor the original version of this finding noted the discrepancy.

## Cost ceiling in pool-CPU terms (the single-consumer-RTX-3090 case is UNMEASURED)

The pre-registered cost bar (`research/runners/d6_capacity_curve.py`: `COST_ENCODE_MAX_S = 2.0`,
`COST_READTIME_MAX_S = 10.0`, `COST_MEM_GB = 24.0`) is stated in chat-turn terms: can one more fact be taught, and
can everything taught so far be read back, inside a turn's latency budget. **No job in this grid ran
`SIM_BACKEND=cupy`.** All 84 jobs ran `SIM_BACKEND=numpy` on shared pool-CPU / cloud nodes
(`env.SIM_BACKEND`; `cost_hardware_class="pool-cpu-shared"` on every job that reports one), single-threaded in
practice (`process_time_s / elapsed_s` is 0.86-1.03 across all 84 jobs — no multi-core speedup is in play).
Amendment C(b) (module docstring) states a cost verdict may only be labelled `"consumer-hardware"` when
`env.SIM_BACKEND == "cupy"`; none of these jobs qualify, so **this section reports the ceiling in pool-CPU terms
and treats the project's single-consumer-RTX-3090 reference class
(`project_consumer_hardware_reference_principle`) as UNMEASURED — not as a second label for the same numbers.**

Peak RSS never approaches the 24 GB memory bar at any level — it stays at 124.0-232.9 MB across the whole grid
(`s102_N5_HEBB_REP.json` to `s44_N2000_FREEZE.json`, `peak_rss_mb`); **the ceiling is entirely a TIME failure on
this hardware, never a memory one.**

In plain terms, on pool CPU: at N=5, reading everything taught back costs 0.17-0.28 seconds; at N=50 it already
costs 2.87-4.86 seconds — both comfortably inside a 10-second turn. At N=500, that read-back costs 159.8-1056.6
seconds (2.7-17.6 minutes) — 16x to 106x over the bar at the fastest and slowest cells; at N=2000 it costs
2703.9-28898.7 seconds (45 minutes to 8 hours) — 270x to roughly 2890x over the bar. **The ceiling's LOCATION on
pool CPU (between 50 and 500) is solid: every one of the 6 seeds fails at N>=500, including the fastest.** Whether
a single consumer RTX 3090 would still fail at N=500 is a genuinely open, UNMEASURED question, not a "categorical"
one: the margin at the fastest N=500 cell (`s42` HEBB, `readtime_view_s_per_turn_projected=159.775`) is only 16x
over the 10-second bar, and a 3090 against this single-threaded-numpy baseline could plausibly deliver a larger
constant-factor speedup than that on the dense array operations this workload performs — in which case N=500
would clear the bar on a 3090. Holding N=2000 under the bar needs a much larger margin (270x or more at the
fastest cell), which a constant-factor GPU speedup is less likely to close, but that too is UNMEASURED here.

**Why a GPU speedup would not simply divide these seconds by a constant and read the same crossing point off this
table.** The per-turn cost is not linear in N. The grid's own committed job list already says so:
`research/findings/raw/_d6_capacity_curve/JOBS.txt` states "Per-encode cost grows ~linearly with n_total (129
neurons per stored fact), so teaching N facts is O(N^2)." The read-time-view cost is `engram_read_s_median x
n_facts` (`d6_capacity_curve.py`), and `engram_read_s_median` itself grows with the substrate size
`n_total = 129*(N+16) + 3676` (6385 at N=5, 12190 at N=50, 70240 at N=500, 263740 at N=2000, `s42_N*_HEBB.json`,
`n_total`) — so the per-turn cost grows faster than linearly in N, not linearly as an earlier draft of this
finding stated. Because of that, a device that is faster per elementary array op does not shift the N-at-which-the-
bar-is-crossed by the same factor: the crossing point is UNMEASURED for a GPU, not derivable from the CPU curve by
a single division. This is offered as the reason "UNMEASURED, not FAILS" is the honest verdict for a GPU, not as a
fitted exponent — no GPU job exists in this grid to fit one from.

**Honesty about the node confound.** The N=2000 jobs alone landed on at least three different, concurrently-loaded
machines (`ip-172-31-47-37`, `node-dl5d243`, `node-dl4g243` — `s42_N2000_HEBB.json` / `s42_N2000_COPY.json` /
`s100_N2000_HEBB.json` `node.hostname`). That confounds ranking HEBB against COPY by absolute time at N=2000: seed
42's HEBB ran on `ip-172-31-47-37` (`encode_s_median=9.19085`) while its own COPY arm ran on `node-dl5d243`
(`encode_s_median=33.4906`) — a 3.6x gap that is at least partly which machine the job landed on, not the write
mechanism, since COPY does strictly less work than HEBB per fact (no Hebbian accumulation loop). The relevant
node/load spread is small at the short levels (`encode_s_median` at N=50: 1.05-1.06x within an arm, 1.34x across
HEBB and COPY) but much larger at the long levels where the exact ceiling seconds are read (`encode_s_median`
cross-arm: roughly 6.2x at N=500, 7.5x at N=2000; `readtime_view_s_per_turn_projected`: roughly 6.6x at N=500,
10.7x at N=2000 — an earlier draft of this finding cited "1.6x... at N=50" for this comparison, which used the
wrong level). **The ceiling's LOCATION (between 50 and 500) is robust to this confound** — the fastest-observed
N=500 cell is still 16x over the bar, larger than the node/load spread at that level — but the exact seconds
reported here are `pool-cpu-shared` numbers on visibly variable-load nodes, not a controlled benchmark, and no
single-consumer-RTX-3090 number exists in this grid.

**The retroactive cost note, and a correction to how an earlier draft of this finding characterized it.** Seven of
the eighteen N=2000 jobs (`s100_N2000_{HEBB,COPY,FREEZE}`, `s101_N2000_{HEBB,FREEZE}`, `s42_N2000_COPY`,
`s43_N2000_HEBB`) carry a `cost_acknowledged` field added 2026-09-25 at harvest, stating that "no run-time cost
projection was made before this cell was launched (the D6 prereg set per-chat-turn cost bars, not a compute-hours
projection)". **An earlier draft of this finding restated that more broadly as "no time projection preceded a set
of jobs" — that broader claim is FALSE.** `research/findings/raw/_d6_capacity_curve/JOBS.txt` (committed at
`f9ae852b7`, before the grid ran) states "N=2000 ~8 h (COPY ~6.5 h)", and every dispatched N=2000 line in
`research/queue/dispatch.log` carries `est_wall=8h` (HEBB/FREEZE) or `est_wall=6.5h` (COPY). A wall-time projection
did exist. The real process deviation is a 1.8-3.9x OVERRUN against that projection: the seven noted jobs ran
14.6-31.0 wall-hours against a 6.5-8 h estimate. What the raw note's own, narrower claim correctly points at still
stands: the prereg's LAUNCH-GATING criterion for N=2000 is memory only (drop N=2000 only if peak RSS is projected
past 13 GB or 120 GB), wall time was never a drop criterion, so no wall-time projection was REQUIRED to gate the
launch — one existed anyway, for scheduling, and it undershot by 1.8-3.9x. Measured peak RSS at N=2000 stayed at
227.8-232.9 MB across every seed and arm (`s42_N2000_HEBB.json` through `s102_N2000_COPY.json`, `peak_rss_mb`) —
roughly 57x under the 13 GB pool-node bound and roughly 528x under the 120 GB AWS bound (about 1.8 and 2.7 orders
of magnitude, not six to seven as an earlier draft stated), and roughly 106x under the 24 GB per-turn cost bar —
so the memory-gated launch decision itself was correct; only the independent wall-time ESTIMATE (never a gate)
undershot.

## What this does and does not show

**Shows:** recall holds at 1.0 for both HEBB and COPY through N=2000 facts, but this is a consequence of the
store's disjoint, one-block-per-fact design (Amendment B(a)), not a capacity result; HEBB and COPY converge on
near-identical stored synapses by construction (parity-by-construction), so the accuracy axis cannot separate a
host-instructed local rule from a direct host copy; the FREEZE null is exact at every scale tested, confirming the
write (not the probe or DG routing) drives the answer; the per-turn cost of this store crosses from usable to
unusable somewhere between 50 and 500 taught facts, on time alone, for both arms, on the pool-CPU hardware every
job in this grid actually ran on (the single-consumer-RTX-3090 case is UNMEASURED — see "Cost ceiling" above).

**Does not show:** whether the local Hebbian rule can do anything a direct copy cannot — that requires a task
where the two routes diverge, which this instrument does not construct; a capacity LAW (recall degrading with N)
— this store has none by construction, so it has nothing to compare against an LLM's parameter-count capacity
(that question belongs to the distributed-store lane,
`research/findings/2026-09-23-ca3-superposed-fact-attractor-capacity-PREREGISTRATION.md`); cross-fact
interference — the near-miss probe shares the taught fact's own DG shard by construction and does not test a
sibling fact's intrusion (prereg Amendment B(c), not re-litigated here); GPU/`SIM_BACKEND=cupy` throughput — every
job in this grid ran `numpy` on shared CPU hosts, so the single-consumer-RTX-3090 cost verdict is UNMEASURED, not
a value this grid computed and then declined to report.

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
- Provenance: all 84 raw job files carry a matching `*.json.prov.json` sidecar recording `git_sha =
  24231d6d6872a83cba9a0cf29612ef6aaa147fda` (post-dates Amendment C) on every one, `git_dirty=false` on every one,
  and `source_manifest_verified_at_start` / `_at_exit` both `true` on every one; `parse_errors=0` on every one of
  the 84 raw job files.
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
