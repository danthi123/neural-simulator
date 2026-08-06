---
type: finding
status: live
date: 2026-08-06
mechanism: replay-cortical-consolidation-v6-order-sensitive-STDP
runner: research/runners/_replay_cortical_consolidation_gate_v6_order_stdp.py
builds-on: research/findings/2026-08-06-replay-cortical-consolidation-v5-sfa-eviction-closes-interference-wall-order-control-is-next.md
supersedes-method: order-blind-rate-window-Hebbian-sleep-consolidation
artifacts:
  - research/findings/raw/replay_v5_sfa_order/replay_v6_order_stdp_calibration.json
  - research/findings/raw/replay_v5_sfa_order/replay_v6_order_stdp_calibration.json.prov.json
  - research/findings/raw/replay_v5_sfa_order/replay_v6_stdp_off_control.json
---

# Order-sensitive STDP consolidation CLOSES the replay-order control on BOTH calibration seeds — v6 is 2-seed GO with every v5 CLS signature and false-recall<0.15 intact

<!--derived-->
**Verdict: 2-seed GO (aggregate CALIBRATION_PROMISING; per-seed GO on 412 AND 413).** v5+SFA closed the interference wall but was NO-GO on the sole residual control `intact_beats_shuffled_order` — root cause: the sleep consolidation rule (rate-window Hebbian on a per-event down-state reset) is ORDER-BLIND, so permuting event order preserves the coactivity multiset and intact/shuffled learn the same trace (seed 413's order margin was NEGATIVE, −0.003). v6 adds the named surpass — order-sensitive spike-timing plasticity — and now BOTH seeds pass ALL frozen v5+SFA checks including `intact_beats_shuffled_order` and `false_recall_bounded<=0.15`, with no check failing on either seed.

## Mechanism (vs v5+SFA)

<!--derived-->
Two coupled, biology-grounded changes; everything else in v5+SFA is inherited unchanged. (1) ORDER-SENSITIVE PLASTICITY via the substrate's OWN STDP (`sim/kernels.fused_stdp_weight_update`, Bi&Poo 1998), not a host-computed timing rule: `enable_stdp` trains the cortical cue->target association by spike timing. STDP respects the per-pathway plasticity gate and the per-synapse plastic mask, and during sleep only `CORTICAL_GATE` is open, so STDP acts on cue->target ONLY. It is kept INERT outside sleep by never advancing `current_time_ms` in wake/probe (the documented bridge.py:9382 guard: a frozen clock makes every delta_t==0 and every update exactly 0.0), and `cp_last_spike_time` is cleared at sleep onset so no wake spike pairs across the phase boundary. STDP soft-bound is set to half the Hebbian weight scale (`stdp_w_max_scale=0.5`) so it nudges rather than saturates. (2) ORDER-CARRYING DYNAMICS: v5+SFA separates every replay event with a FULL fast-dynamics reset, which erases any dependence of event i+1 on event i and makes order invisible to any local rule. v6 keeps the membrane/conductance down-state boundary but PRESERVES `cp_last_spike_time` across it (`sleep_interevent_reset="timing"`), so STDP timing carries across the boundary and cross-event ADJACENCY becomes physically present. Ordered replay (adjacent events overlap — the `mean_adjacent_input_overlap` the frozen `temporal_control_changes_order` check already verifies is HIGH for intact, LOW for shuffled) sustains a coherent causal cue-before-target flow -> consistent LTP; shuffled replay breaks the adjacency -> weaker, undirected trace. The per-event coactivity multiset is no longer sufficient to determine the weights — adjacency (order) now is, exactly what the v5+SFA root-cause analysis said was required.

## Result

<!--derived-->
Artifact: `research/findings/raw/replay_v5_sfa_order/replay_v6_order_stdp_calibration.json` (NumPy backend; provenance sidecar records argv + git SHA). Nine conditions per seed through one persisted bridge. Calibration point: `stdp_a_plus=0.008, stdp_a_minus=0.03, stdp_w_max_scale=0.5, target_sfa_d_increment=180` (v5+SFA was d=120; the order-STDP trace is broader so the intrinsic one-of-N eviction evicts harder to hold false recall under 0.15 on the harder seed).

| seed | intact rec | false | weak | order margin (intact vs shuffled) | no_sleep | reinst_lesion | ca3_ca1_lesion | plast_off | STDP delta intact/shuffled | result |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 412 | 0.0736 | 0.1167 | 0.0319 | 0.0736 vs 0.0243 (+0.0493) | 0.000 | 0.000 | 0.000 | 0.000 | 2.973 / 0.645 | GO | <!--derived-->
| 413 | 0.0486 | 0.0916 | 0.0458 | 0.0486 vs 0.0347 (+0.0139) | 0.000 | 0.000 | 0.000 | 0.000 | 1.835 / 0.975 | GO | <!--derived-->

<!--derived-->
Every CLS signature is intact on both seeds: both memories recover, sleep reinstatement is memory-specific (match 14 vs mismatch 2 on 412, 15 vs 0 on 413), the target reinstates during sleep, and all four causal-lesion controls (no_sleep, reinstatement_lesion, ca3_ca1_lesion, cortical_plasticity_off) drop hippocampus-independent recall to exactly 0.000 — consolidated recall still flows only through the intracortical cue->target association built during replay. SFA eviction remains load-bearing (`target_sfa_lesion` false recall 0.129 / 0.159 vs intact 0.117 / 0.092), and target inhibition still improves specificity (inhibition-lesion false 0.500 / 0.407).

## STDP measurably created the directional trace (causal power control)

<!--derived-->
STDP potentiates the cue->target association ~4.6x more under ordered than shuffled replay on seed 412 (delta 2.973 vs 0.645) and ~1.9x more on seed 413 (1.835 vs 0.975): ordered replay builds a stronger directional trace, exactly the mechanism claimed. The decisive causal control (`replay_v6_stdp_off_control.json`): with the SAME v6 anatomy / SFA(d=180) / contiguous-replay reset but `stdp_sleep=False`, the ordered-vs-shuffled recovery margin COLLAPSES — seed 412 +0.0493 -> −0.0076 (shuffled beats intact), seed 413 +0.0139 -> +0.0035 (below the +0.01 bar). So the order margin is causally attributable to the order-sensitive STDP consolidation, not to the SFA eviction or the contiguous-replay reset (both present in the off control). This is a verdict FOR the order-STDP method on the replay-order capability.

## Decision and next

<!--derived-->
This closes the last residual control of the replay-consolidation arc at the 2-seed calibration bar: capability #4 (CLS replay consolidation) now passes every preregistered control on both calibration seeds. Preserve this runner + artifacts as the new baseline. NEXT: multiseed validation on the disjoint development seeds 414/415/410 (then held-out 417/418/419) BEFORE any generalization claim — seed 413's order margin (+0.0139) clears the +0.01 bar but is slim, so multiseed is the load-bearing confirmation. If multiseed holds, route the mechanism hands-off into the consolidation pipeline. Remaining scaffolds: host-defined wake episode populations and probe cues; fixed opponent/inhibitory membership; host-scheduled down-state boundaries and episode-agnostic CA3 background; host spike/weight measurement; fixed assembly anatomy; SFA (d/a) and STDP (amplitudes/bounds) set at build rather than developmentally tuned. This is a verdict on the ORDER method; it does not weaken any frozen v5 criterion (the entire v5+SFA verdict is inherited, only diagnostics added).
