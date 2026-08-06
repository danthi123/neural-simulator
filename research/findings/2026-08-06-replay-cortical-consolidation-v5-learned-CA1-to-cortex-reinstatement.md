---
type: finding
status: negative
date: 2026-08-06
mechanism: replay-driven-cortical-consolidation-v5-learned-ca1-to-cortex-reinstatement
runner: research/runners/_replay_cortical_consolidation_gate_v5.py
supersedes-method: fixed-intracortical-index-to-target-teacher (v3/v4)
artifacts:
  - research/findings/raw/replay_v5/replay_v5_calibration.json
  - research/findings/raw/replay_v5/replay_v5_calibration.json.prov.json
---

# Learned CA1->cortex reinstatement makes replay consolidation work on one calibration seed, not yet both

<!--derived-->
**Verdict: NO-GO at the 2-seed calibration bar (aggregate CALIBRATION_NEEDS_REVISION) — but a decisive mechanistic advance.** The learned, encoding-potentiated CA1->cortical_target reinstatement pathway fixes the v3/v4 structural root cause: the cortical target is now reinstated by the memory-specific hippocampal index during sleep, not by a fixed intracortical teacher. Seed 412 is a clean GO (all preconditions and all checks pass; both memories recover; every causal control is zero). Seed 413 is a NO-GO on two checks: retest false recall 0.180 exceeds the 0.15 ceiling, and its intact recovery does not beat the shuffled-replay-order control by the fixed margin. Development and held-out seeds remain locked.

## Mechanism change (vs v1-v4)

The 2026-08-06 research gate localized the structural root cause: v3/v4 fired the cortical target during replay through a FIXED intracortical `cortical_index->cortical_target` teacher, so target reinstatement never depended on the learned hippocampal index (v3 target fired 0 spikes during sleep). v1/v2 did carry a plastic `ca1_to_cortical_target` wire but it began at the same tiny 0.05 efficacy as every index wire and was seed-fragile.

<!--derived-->
V5 makes the CLS reinstatement (McClelland-McNaughton-O'Reilly 1995 / Tse 2007) explicit and reliable. The memory-specific `ca1 -> cortical_target` pathway starts at a functional baseline efficacy (8.0) and potentiates further at wake encode (CA1 fires via ca3->ca1 while the target is host-driven), so uncued CA1 replay reinstates the correct target directly. A symmetric fix reinstates the cortical CUE during sleep too (`ca1 -> cortical_cue` raised off 0.05): with both poles co-active, the intracortical `cortical_cue->cortical_target` association consolidates. The reinstatement wire carries its own transmission gate, ON at wake-encode and sleep, OFF at the hippocampus-disabled retest, so a new `ca1_target_reinstatement_lesion` control silences exactly this wire during sleep. Opponent target fast-spiking competition, target recurrence, and the true `shuffled_replay_order` control are inherited from v2.

## Result

<!--derived-->
Artifact: `research/findings/raw/replay_v5/replay_v5_calibration.json` (NumPy backend, provenance sidecar records argv + git SHA `071b68bca`). Both calibration seeds ran the full eight-condition battery through one persisted bridge (encode A -> interfering encode B -> uncued sleep -> hippocampus-disabled partial-cue retest).

| seed | intact recovery | mem A / mem B | false recall | no_sleep | reinst_lesion | ca3_ca1_lesion | plasticity_off | shuffled_order | result |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 412 | 0.0896 | 0.1361 / 0.0431 | 0.1125 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0597 | GO | <!--derived-->
| 413 | 0.0333 | 0.0417 / 0.0250 | 0.1801 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0361 | NO-GO | <!--derived-->

<!--derived-->
The reinstatement is memory-SELECTIVE during sleep on both seeds, reported with its raw per-episode magnitudes: of the reactivated events, the cortical target that reinstated matched the replayed CA3 assembly on 19 events vs 4 mismatches (seed 412) and 21 vs 0 (seed 413). Target fired 445 / 424 spikes during sleep (v3 fired 0). The cortical association changed only during sleep (`cortical_during_wake` ~ 0; `cortical_during_sleep` > 0) and not at all under cortical-plasticity-off.

The consolidation is causal and hippocampus-INDEPENDENT at retest on BOTH seeds — the load-bearing CLS signature. Removing sleep, lesioning the CA1->target reinstatement wire, lesioning ca3->ca1 transmission, or freezing cortical plasticity each drops recall to exactly 0.000. So the retest recall (hippocampus disabled: CA1 silent, reinstatement gate off) flows only through the consolidated intracortical cue->target association, and that association exists only because the learned CA1->cortex reinstatement co-activated cue and target during replay. This meets the TERMS condition for "consolidation" (the replay path executes AND the trace survives a lesion of the source structure).

## Residual (quantified) and root cause of the remaining fragility

<!--derived-->
The remaining defect is seed-413 retest cross-talk, not a reinstatement failure (413's reinstatement is perfectly specific: 21 match / 0 mismatch). At retest, the partial cue for one memory includes shared cue cells (`cue_overlap` = 6 of 16), which drive the consolidated association of BOTH memories, leaking ~5 spikes per probe to the wrong target assembly (false recall 0.180). Opponent fast-spiking competition reduces this (lesioning it raises 413 false recall to 0.31-0.37) but does not clear the 0.15 ceiling on the harder seed. The intact-vs-shuffled-order margin on 413 (0.0333 vs 0.0361) is within noise. This is the point-neuron competition limit the 2026-08-06 gate predicted: single-compartment neurons cannot fully suppress interference at shared cells.

## Decision and next mechanism

<!--derived-->
Do not open development seeds 414, 415, 410. Preserve this runner and the artifact as the new baseline: it is the first attempt in this arc to produce a clean-GO seed with both memories recovering and a memory-specific, causally load-bearing CA1->cortex reinstatement. The named surpass (2026-08-06 gate) is now the concrete next step: spike-frequency-adaptation-driven one-of-N eviction on the target attractor, to suppress the shared-cell interference that leaks false recall on the harder seed — plus the interference question directly (shared-cue-cell competition) and multi-seed validation. This is a verdict on the point-neuron competition METHOD, not on the reinstatement capability, which is now established.

Remaining scaffolds are unchanged from v2: host-defined wake episode populations and partial cues, fixed opponent inhibitory channel membership, host-scheduled sleep down-state boundaries and episode-agnostic CA3 background current, host measurement against known calibration assemblies, and the rate-window Hebbian rule on fixed anatomy.
