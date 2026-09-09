---
type: finding
status: negative
date: 2026-09-08
mechanism: replay-cortical-consolidation-v9-true-sparsity-region-scaleup
runner: research/runners/_replay_cortical_consolidation_gate_v9_true_sparsity.py
builds-on: research/findings/2026-08-25-order-consolidation-v8-real-scale-ca1-port-6seed-GO.md
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/order_recalib/v9_true_sparsity_decisive_numpy.json
---

# Board #130's own next-named rung (region growth toward true biological sparsity) is NO-GO 6/6 — but the causal order-STDP mechanism itself stays intact; the failure is a fixed-threshold READOUT that never transfers across scale

<!--derived-->
**Verdict: NO-GO, 0/6 decisive seeds (numpy).** v8 (`2026-08-25-order-consolidation-v8-real-scale-ca1-port-6seed-GO.md`)
ported the balanced directed-sweep + order-STDP consolidation mechanism (board #130) onto a real-scale CA1
substrate but left one residual unclosed by its own admission: the assembly:region ratio (~33-67%) was an exact
duplicate of the hand-picked toy's, nowhere near the ~1-5% active biological sparse code, "and would need a
SEPARATE de-risk since it changes the assembly:region ratio the sparsify indegree formula [v8] assumes stays
fixed." This is that de-risk, at a first (2x) region-scale-up step (ratio 33.3% -> 16.7%). Result: **0/6 seeds
pass the full decisive gate**, but the underlying causal mechanism is NOT what fails — every seed's order-STDP
attribution is 1.0, the stdp-off power control collapses to exactly 0.0000, and the physical ordered cortical
trace is stronger than shuffled on all 6 seeds. What fails uniformly is a FIXED absolute correct-rate threshold
(0.015) baked into the shared `_calibration_verdict` gate that this region-scale-up was never checked against.

## What ran

<!--derived-->
`SIM_BACKEND=numpy .venv/bin/python -m research.runners._replay_cortical_consolidation_gate_v9_true_sparsity
--seeds 42 43 44 100 101 102 --out research/findings/raw/order_recalib/v9_true_sparsity_decisive_numpy.json`
(754.8s, single process, no GPU). Decisive artifact:
`research/findings/raw/order_recalib/v9_true_sparsity_decisive_numpy.json`.

## What v9 changed (isolated, one variable)

<!--derived-->
v9 decouples REGION size from ASSEMBLY size, something v1-v8 never exercised (region and assembly always moved
together by the same factor). Every assembly/overlap/FS/noise field stays at v8's exact absolute value
(ca3_assembly=600, ca1/cue/target_assembly=400, cue_overlap=150, sleep_noise_cells=300, n_target_fs=300); only
the four region fields grow 2x beyond v8 (n_ca3 1800->3600, n_ca1/n_cue/n_target 1200->2400). In-degree is looked
up from a FIXED absolute table (24/16/6, matching v8's own validated values) rather than recomputed as a
region-dependent ratio — verified byte-identical to v8 at `region_scale=1` (seed 42's order margin reproduced
v8's own table value, +0.0285277..., to 7 significant figures, confirming the sparsify rewrite introduces no
behavioural difference on its own). Everything else (directed-sweep replay, order-STDP, SFA eviction, learned
CA1->cortex reinstatement, every anti-cheat control) is v7/v8, byte-for-byte.

## Result (6 seeds, numpy; all counts direct reads of the cited artifact)

<!--derived-->
| seed | order margin | beats shuffled | both recovered | stdp owns order | attribution | trace I>S | A rate | B rate |
|---:|---:|:---:|:---:|:---:|---:|:---:|---:|---:|
| 42 | +0.0119 | yes | no | yes | 1.0 | yes | 0.0086 | 0.0186 |
| 43 | +0.0083 | no | no | no | 1.0 | yes | 0.0087 | 0.0134 |
| 44 | +0.0079 | no | no | no | 1.0 | yes | 0.0089 | 0.0122 |
| 100 | +0.0105 | yes | no | yes | 1.0 | yes | 0.0087 | 0.0172 |
| 101 | +0.0078 | no | no | no | 1.0 | yes | 0.0080 | 0.0152 |
| 102 | +0.0094 | no | no | no | 1.0 | yes | 0.0073 | 0.0138 |

<!--derived-->
Mean order margin +0.00931 (below the +0.01 bar on the aggregate, and 4/6 seeds individually below it too);
`n_beats_shuffled_order=2/6`; **`n_both_recovered=0/6`** — the SAME 7 checks
(`intact_partial_recovery`, `both_memories_recovered`, `intact_beats_no_sleep`, `schaffer_path_is_load_bearing`,
`cortical_plasticity_is_load_bearing`, `reinstatement_is_load_bearing`, `target_inhibition_improves_specificity`)
fail on **every single seed**, identically. That uniformity across 6 independent seeds is itself informative: this
is a systematic effect of the region-scale-up, not seed noise. Meanwhile `order_stdp_attribution=1.0` and
`stdp_off_order_margin=0.0000` (exact) on all 6, and `ordered_trace_stronger` (the probe-independent physical
sleep-time cortical weight delta) is `true` on all 6 — the causal fingerprint of order-sensitive consolidation
is fully intact and undiminished in its own right.

## Why (traced through the code, not inferred from the numbers alone)

<!--derived-->
`correct_rate` (the behavioral readout `both_memories_recovered` gates on) is the mean per-neuron firing rate of
the FIXED-size (400-cell) correct target ASSEMBLY — its definition does not itself depend on region size. But one
wiring pathway, `cortical_association` (v8's own docstring: "spans the FULL region, not an assembly"), connects
the WHOLE cue region to the WHOLE target region at a FIXED absolute in-degree (48) — the SAME value at
`region_scale=1` and `region_scale=2` by this de-risk's own design (see module docstring). The retrieval probe
drives only a small partial-cue SUBSET of cue cells. As `n_cue` doubles, the probability that any given target
neuron's 48 random converging cue-synapses include one of the FEW driven partial-cue cells drops by roughly the
same factor — diluting exactly the signal this one generic, non-assembly-scoped pathway carries into the correct
assembly. The assembly-SCOPED indexed pathways (CA3->CA1->cortex reinstatement, `_fact_shard`-style routing) are
architecturally insulated from this because their pre/post pools are assemblies, not full regions — consistent
with `intact_stdp_cortical_delta` (measured directly off the reinstatement synapses, assembly-scoped) staying
robust while the assembly's own downstream FIRING RATE (which also depends on `cortical_association`'s diluted
drive) does not.

<!--derived-->
This is the project's own "wall reframe" (CLAUDE.md, `2026-07-31-why-we-hit-walls-the-missing-companion-process.md`)
playing out exactly as described: a fixed absolute in-degree is the right call for an assembly-scoped pathway
(v8's own precedent, validated) but the wrong constant for the ONE pathway meant to represent bulk, non-specific
cortical association — its correct behaviour requires SOME compensation as the region grows (a homeostatic /
synaptic-scaling-style gain, or an in-degree law that preserves the driven-fraction probability rather than the
raw synapse count) that neither v1-v8 nor this v9 rung supplies, because region and assembly never moved apart
before now. Separately, the SAME fixed 0.015/0.03 additive-margin thresholds in the shared `_calibration_verdict`
gate (`research/runners/_replay_cortical_consolidation_gate_v5.py`) were calibrated for v7/v8's region density and
were never re-derived for a sparser one — every failing check here is of the shape `intact_recovery >= X + 0.015`,
so once the underlying correct-rate magnitude shrinks (for the mechanistic reason above), all of them fail
together, which is exactly the uniform failure pattern observed.

## What this does and does not establish

<!--derived-->
**Does not establish:** that a genuinely sparser (assembly:region ~1-5%) CA1 substrate can support this
consolidation mechanism's full behavioral gate. At `region_scale=2` (ratio 16.7%) it does not, 0/6.

<!--derived-->
**Establishes:** the order-sensitive STDP consolidation mechanism itself (order-STDP causally producing a
stronger physical cortical trace for ordered vs. shuffled replay) is NOT what breaks — its causal attribution
(1.0), its power control (exact 0.0000 collapse), and its probe-independent physical trace measurement
(`ordered_trace_stronger`) are all robust on every one of 6 seeds, unchanged in kind from v8. What breaks is (a)
the ONE full-region, non-assembly-scoped wiring pathway's fixed in-degree diluting signal as its candidate pool
grows, and (b) a battery of fixed additive-margin behavioral thresholds that were never re-derived for the new
region density. Per THE LAW (CLAUDE.md), this is a verdict on the METHOD (naive region growth + an unchanged
fixed in-degree + an unchanged fixed threshold), not on the underlying capability.

## Next mechanism (no-defer, named)

<!--derived-->
Two independent, compatible fixes, either sufficient to re-test: **(1)** give `cortical_association` specifically
a region-growth-COMPENSATING in-degree law (in-degree x driven-fraction held constant, i.e. in-degree scaling
with `n_cue` rather than fixed — the opposite of every OTHER, assembly-scoped pathway in this same substrate,
which should stay fixed) instead of applying the same "hold in-degree constant" rule this de-risk applied
uniformly; **(2)** re-derive the `_calibration_verdict` gate's fixed additive thresholds (0.015/0.03) as a
function of the substrate's own OPERATING POINT at the new scale (e.g. a fraction of the no-sleep/chance
baseline) rather than an absolute constant, so the gate measures the same RELATIVE effect at any region density —
matching this project's own `tools.lab` operating-point discipline. Both are scoped, mechanistic fixes to a named
cause, not a re-tune of this arc's decisive bar; board #130 stays open at v8's already-GO'd real-scale (unchanged)
result, with the further-sparsity rung now characterized rather than merely proposed.
