---
type: finding
status: negative
date: 2026-08-06
mechanism: replay-cortical-consolidation-v5-sfa-intrinsic-one-of-N-eviction
runner: research/runners/_replay_cortical_consolidation_gate_v5_sfa.py
builds-on: research/findings/2026-08-06-replay-cortical-consolidation-v5-learned-CA1-to-cortex-reinstatement.md
supersedes-method: point-neuron-opponent-FS-only-competition-at-retest (v5)
artifacts:
  - research/findings/raw/replay_v5_sfa/replay_v5_sfa_calibration.json
  - research/findings/raw/replay_v5_sfa/replay_v5_sfa_calibration.json.prov.json
---

# Intrinsic SFA one-of-N eviction CLOSES the shared-cue-cell interference wall on both seeds; the residual is the replay-ORDER control, not interference

<!--derived-->
**Verdict: NO-GO at the 2-seed calibration bar (aggregate CALIBRATION_NEEDS_REVISION) — but it decisively closes the named v5 wall.** Adding intrinsic spike-frequency adaptation (SFA) to the cortical-target attractor drives seed-413 retest false recall from 0.180 (v5) to 0.0797 and seed-412 from 0.1125 to 0.0658 — both far under the frozen 0.15 ceiling — and the new `target_sfa_lesion` control proves it is load-bearing (restoring RS-default adaptation returns false recall to the v5 values 0.1801 / 0.1125). Every other v5 check passes on BOTH seeds. The sole remaining failure, on both seeds, is `intact_beats_shuffled_order`: the replay-order control, which was ALREADY failing seed 413 in v5 and is a separate wall SFA does not target.

## Mechanism (vs v5)

<!--derived-->
SFA is INTRINSIC per-neuron biology, realised through the substrate's own Izhikevich adaptation on the `cortical_target` slice — the recovery variable u, incremented by `d` on every spike and relaxing at rate `a` (RS defaults d=100, a=0.03; set to d=120, a=0.02). No host code computes the eviction and no transmission gate can reach an intrinsic mechanism (the affect arc's 2026-07-31 lesson), so eviction is applied by writing `cp_izh_d_increment`/`cp_izh_a` and lesioned by restoring them. During a probe the correct target assembly is strongly, recurrently driven; the interfering assembly is driven only by the few shared cue cells (`cue_overlap` 6/16) with no recurrent support, so adaptation accumulates faster on the weak, unsupported assembly and silences it while the correct one rides through — a one-of-N eviction (Dehaene-Changeux 2011 metastability; Ecker 2022 adaptation-driven transitions) that sharpens the biased competition the opponent FS pool already begins.

## Result

<!--derived-->
Artifact: `research/findings/raw/replay_v5_sfa/replay_v5_sfa_calibration.json` (NumPy backend; provenance sidecar records argv + git SHA). Nine conditions per seed through one persisted bridge (v5's eight + the additive `target_sfa_lesion` power control). No frozen v5 criterion was weakened; `target_sfa_lesion` adds no threshold.

| seed | intact recovery | mem A / mem B | false recall | sfa_lesion false | no_sleep | reinst_lesion | ca3_ca1_lesion | plasticity_off | intact vs shuffled_order | result |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 412 | 0.0861 | 0.1264 / 0.0458 | 0.0658 | 0.1125 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0861 vs 0.0847 (+0.0014) | NO-GO | <!--derived-->
| 413 | 0.0410 | 0.0375 / 0.0444 | 0.0797 | 0.1801 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0410 vs 0.0354 (+0.0056) | NO-GO | <!--derived-->

<!--derived-->
The CLS consolidation signature is fully intact on both seeds: both memories recover, sleep reinstatement is memory-specific (match 19 vs mismatch 4 on 412, 19 vs 1 on 413), the target reinstates during sleep (411 / 369 spikes), and all four causal-lesion controls drop hippocampus-independent recall to exactly 0.000 — so consolidated recall still flows only through the intracortical cue->target association built by learned CA1->cortex reinstatement during replay. SFA is added on top of that mechanism, not in place of it.

## What SFA closes, and the residual (isolated + quantified)

<!--derived-->
CLOSED — the named v5 wall. Shared-cue-cell interference (v5's seed-413 root cause) is suppressed on BOTH seeds: false recall 0.0658 / 0.0797, each ~0.07-0.08 below the 0.15 ceiling, causally attributable to SFA (lesion restores 0.1125 / 0.1801; reduction 0.047 / 0.101). This is a verdict FOR the SFA-one-of-N-eviction method on the interference capability.

<!--derived-->
RESIDUAL — a DIFFERENT, deeper wall: `intact_beats_shuffled_order` (intact recovery must beat the shuffled-replay-order control by +0.01). It fails on both seeds (margins +0.0014, +0.0056). A joint sweep (d 110-190 x a 0.02-0.03, both seeds) shows it is UNSATISFIABLE together with false<=0.15: seed 412's order margin needs LOW adaptation (only d<=110 clears +0.01, where seed-413 false is still 0.19), while seed 413's order margin never exceeds ~+0.009 anywhere in the swept space. Two root causes. (1) SFA is an order-AGNOSTIC retest cleanup — it improves readout of any consolidated trace, ordered or shuffled, so strong SFA lifts the shuffled control's recall as much as intact (on 412 it inverts the v5 order margin from +0.030 to <=0). (2) The consolidation rule is rate-window Hebbian coactivity — shuffling event ORDER preserves the coactivity multiset, so ordered and shuffled replay consolidate near-identical weights; seed 413 got essentially zero order benefit even in v5 (intact 0.0333 < shuffled 0.0361).

## Decision and next mechanism

<!--derived-->
Do not open development seeds 414/415/410. Preserve this runner + artifact as the new baseline: SFA eviction is the established surpass for shared-cell interference. The residual is now precisely the replay-order-dependence of consolidation, and its root cause is the order-BLIND rate-window Hebbian rule, not the retest competition. Named next mechanism: an order-SENSITIVE (spike-timing-dependent / sequence-replay) plasticity rule during sleep consolidation, so ordered replay potentiates a directional cue->target trace that shuffled replay does not — restoring the `intact_beats_shuffled_order` margin at the source. Pair it with a mildly gentler SFA point (e.g. d=120) so the retest cleanup does not erase whatever order margin consolidation produces. This is a verdict on the point-neuron/rate-Hebbian ORDER method, not on the reinstatement or interference capabilities, both now established.

Remaining scaffolds are unchanged from v5, plus: SFA parameters (d_increment/a) are set on the target slice at build rather than developmentally tuned.
