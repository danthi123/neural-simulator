---
type: finding
status: live
date: 2026-08-25
mechanism: replay-cortical-consolidation-v8-real-scale-ca1-port
runner: research/runners/_replay_cortical_consolidation_gate_v8_real_scale.py
builds-on: research/findings/2026-08-25-order-consolidation-recalib-balanced-directed-sweep-replay-6seed-GO.md
artifacts:
  - research/findings/raw/order_recalib/v8_real_scale_decisive_numpy.json
  - research/findings/raw/order_recalib/v8_real_scale_decisive_numpy.json.prov.json
---

# Board #130 next rung: the balanced directed-sweep + order-STDP consolidation mechanism PORTS to a real-scale CA1 substrate — 6/6 GO, unchanged margins, all anti-cheats intact

<!--derived-->
**Verdict: 6-seed GO (numpy), unchanged from v7's toy-scale result.** `intact_beats_shuffled_order` (margin >= +0.01)
passes 6/6, `both_memories_recovered` 6/6, the stdp-off power control collapses the ordered-vs-shuffled margin to
EXACTLY 0.0000 on all 6 seeds, the physical ordered cortical trace is stronger than shuffled on all 6, and every
per-seed check list is empty (`checks_failed: []` on every seed). Order margins +0.0166..+0.0295 — the SAME range as
v7's toy-scale +0.013..+0.030. This is the requested next rung on board #130: the mechanism was GO on a 72-neuron
hand-built calibration circuit (v7); this runner ports it, unchanged, onto a network 25x bigger in every population
and assembly dimension, with genuinely SPARSE (not all-to-all) wiring, and the GO holds without retuning anything.

## What "port to the real CA1 substrate" means here

<!--derived-->
Every version in this chain (v1..v7) runs the SAME real `sim.bridge.SimulationBridge` — no host-computed plasticity
anywhere — but the SAME small hand-picked population: n_ca3=72, n_ca1=n_cue=n_target=48, assemblies of 16-24 cells,
wired ALL-TO-ALL. That is a genuine spiking substrate, not a toy formula, but it is a calibration-scale circuit, not
anything close to a biologically plausible hippocampal population. "The real CA1 substrate" (the board task's
phrasing) is operationalised as: the SAME mechanism, at a network size in the same ballpark as this project's own
established "real" hippocampal-CA3 scale convention — `n_ca3=2000`, the production D5 episodic organ
(`research/runners/_episodic_dap_dialogue_memory.py`) — rather than the production D5 organ's own architecture
(D5 is CA3-only autoassociative BTSP storage with no CA1/cortex layering and no systems-consolidation transfer; this
mechanism specifically needs the CA3->CA1->cortex indexing/reinstatement pipeline v1..v7 built, which D5 does not
have). This is a **research-runner-scale port, not a production-integration** — matching this exact arc's own v7
framing ("Gate result (not yet production-integrated)").

## The two changes (nothing else in v7 touched)

<!--derived-->
**1. Every extensive (population/assembly) config field x25.** n_ca3 1800, n_ca1/n_cue/n_target 1200, n_target_fs
300, ca3_assembly 600, ca1/cue/target_assembly 400, cue_overlap 150, sleep_noise_cells 300 (the directed-sweep drive
window, scaled to keep the SAME fraction of an assembly driven per replay event). Every INTENSIVE parameter — per-
synapse weights, drive currents (pA), learning rates, STDP amplitudes/taus, SFA d_increment/a, event/step counts —
is inherited UNCHANGED from v7: these are biophysical properties of one neuron or synapse, not of population size,
and must not scale with N.

<!--derived-->
**2. Sparse wiring, indegree-matched to the toy.** v1-v7 wire every projection ALL-TO-ALL. At 25x bigger assemblies,
naive all-to-all would give each postsynaptic neuron 25x more converging synapses at the SAME per-synapse weight —
25x more aggregate drive, silently retuning the whole calibrated operating point (every drive current, learning
rate, and SFA eviction strength) instead of testing the mechanism on a bigger substrate. So every `_all_to_all` call
(the one choke point every wiring builder in v1/v2/v5/v5s/v6 routes through) is monkeypatched, for the duration of
one build, to a random sparse bipartite projection at indegree `round(len(pre)/25)` — the SAME idiom v7 itself
already uses to swap in the directed-sweep replay plan. Because every extensive field here is exactly
`toy_value * 25`, this recovers the EXACT toy in-degree for every one of the 9 wiring populations (ca3 recurrent 24,
ca3->ca1 24, ca1->cue 16, ca1->target/reinstatement 16, cortical_cue->target 48 — this one spans the FULL region,
not an assembly — target recurrent 16, target->FS 16, ca1->FS 16, FS->target 6), verified by hand against the toy's
own dense counts. The measured installed wiring (`inject_explicit_wiring: installed 158400 synapses across 9
populations`) is consistent with that design, not a full O(N^2) all-to-all count.

<!--derived-->
**The v5 "reinstatement_memory_specific" precondition, recomputed for sparse wiring.** `v5.build_bridge` asserts
the CA1->cortical_target reinstatement wire is memory-specific by comparing its edge COUNT to the dense all-to-all
product per memory — an equality that assumes dense wiring and would read False on every sparse build here even
though sparsification never creates a cross-memory edge (each `_all_to_all` call receives one memory's pre/post
pair at a time). Rather than weaken that precondition, this runner recomputes the actual invariant it protects
directly from the installed substrate — every synapse's real pre/post neuron ids under `INDEX_TARGET_GATE`, read
via `cp_connections.tocoo()` — and cross-checks each one against the two memories' actual ca1/target assembly
membership. This is a bridge-truth check, not a trusted-by-construction claim, and it passed on every seed/condition
(the precondition `earned.require` never fired).

## Result (numpy; provenance sidecar records argv + git SHA)

<!--derived-->
Decisive gate `research/findings/raw/order_recalib/v8_real_scale_decisive_numpy.json`, seeds 42/43/44/100/101/102,
identical GO bar to v7 (order margin >= +0.01, both memories recovered, stdp-off power control collapses the
margin, four causal lesions <= 0.005). Elapsed 815s (numpy, single process, no GPU) for the full 6-seed x 9-
condition x 2-arm (stdp-on/off) decisive gate — 108 bridge builds at real scale, ~7.5s/build.

| seed | order margin | STDP-off margin | both rec | trace I>S | lesions~0 | false recall | A rate | B rate | per-seed |
|---:|---:|---:|:---:|:---:|:---:|---:|---:|---:|:---:|
| 42 | +0.0285 | 0.0000 | yes | yes | yes | 0.044 | 0.037 | 0.054 | GO | <!--derived-->
| 43 | +0.0189 | 0.0000 | yes | yes | yes | 0.026 | 0.040 | 0.052 | GO | <!--derived-->
| 44 | +0.0198 | 0.0000 | yes | yes | yes | 0.025 | 0.040 | 0.048 | GO | <!--derived-->
| 100 | +0.0226 | 0.0000 | yes | yes | yes | 0.020 | 0.041 | 0.050 | GO | <!--derived-->
| 101 | +0.0166 | 0.0000 | yes | yes | yes | 0.032 | 0.040 | 0.050 | GO | <!--derived-->
| 102 | +0.0295 | 0.0000 | yes | yes | yes | 0.039 | 0.035 | 0.051 | GO | <!--derived-->

<!--derived-->
Every per-seed `checks_failed` list is empty, `order_stdp_attribution` reads 1.0 on every seed (the ordered-vs-
shuffled margin is fully attributed to the substrate's own STDP, since the stdp-off arm's margin is exactly zero),
and A/B correct rates sit in a narrower, lower band than the toy (~0.035-0.054 vs the toy's wider spread) —
consistent with a bigger, sparser target population diluting the correct-assembly firing rate per neuron while the
MARGIN (the load-bearing quantity) is preserved. Aggregated across all 6 seeds (`attributable_to`, `tools.lab`):
mean order margin +0.02265 vs mean stdp-off margin 0.0000 -> `aggregate_order_stdp_attribution = 1.0` — the SAME
whose-is-it question v7 already answers per-seed, now also asked and answered at the aggregate level.

## What this does and does not establish

<!--derived-->
**Establishes:** the order-sensitive consolidation mechanism (balanced directed-sweep replay + order-STDP + SFA
eviction + learned CA1->cortex reinstatement) is not an artifact of the toy's small, densely-wired calibration
circuit — it reproduces cleanly on a network 25x bigger with genuinely sparse, indegree-matched connectivity, with
every anti-cheat control intact and margins in the same range as the toy. The port required NO retuning of any
intensive parameter (weights, drive currents, learning rates, STDP/SFA amplitudes) — only extensive sizes and
wiring density changed.

<!--derived-->
**Does not establish (no-defer residuals, named explicitly, same discipline as v7):**
(1) **cupy confirmation is QUEUED, not yet landed** (`tools/gpu_queue.sh add`, depth 10 at queue time behind
in-flight GPU work) — this finding rests on numpy only, matching v7's own precedent (the toy arc's decisive artifact
was also numpy-first, cupy confirmed after). (2) The sparse wiring here is a FIXED random in-degree projection, not
a developmentally self-organized (activity-dependent pruning) connectivity — still a scheduled-at-build anatomy.
(3) The assembly:region-size RATIO is an exact duplicate of the toy's (~33-67% of a region is assembly membership)
— that is NOT the biological sparse code (~1-5% active) this project's other CA1 findings establish; a further
scale-up toward true sparsity (bigger region, same absolute assembly size) is the next rung, and would need a
SEPARATE de-risk since it changes the assembly:region ratio the sparsify indegree formula here assumes stays fixed.
(4) This is a gate-level result on a standalone research runner — NOT production-integrated (no wiring into
`webapp/server.py` / the D5 episodic organ was attempted or implied). (5) v7's own documented residual — the
behavioural margin is a recall-SPEED effect that attenuates at longer probe windows — was not re-swept at this
scale; it is inherited, unverified at the new size.
