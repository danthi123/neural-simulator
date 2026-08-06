---
type: finding
status: negative
date: 2026-08-06
mechanism: replay-cortical-consolidation-v6-order-sensitive-STDP
runner: research/runners/_replay_v6_multiseed.py
builds-on: research/findings/2026-08-06-replay-cortical-consolidation-v6-order-sensitive-STDP-closes-order-control-GO.md
supersedes-method: v6-2-seed-calibrated-operating-point-as-a-generalization-claim
artifacts:
  - research/findings/raw/replay_v5_sfa_order/replay_v6_multiseed_dev.json
  - research/findings/raw/replay_v5_sfa_order/replay_v6_multiseed_dev.json.prov.json
  - research/findings/raw/replay_v5_sfa_order/replay_v6_multiseed_dev_stdpoff.json
---

# Multiseed validation is NO-GO: the FROZEN v6 order-STDP operating point does NOT generalize past the 2 calibration seeds — capability #4 does NOT clear the reliability bar

<!--derived-->
**Verdict: MULTISEED_NO_GO on the development seeds (414/415/410) with the FROZEN v6 mechanism/config/evaluator — no seed passes.** The 2-seed calibration GO (412/413) does not transfer. Held-out seeds 417/418/419 stay SEALED (the aggregator refuses them without a development GO). The +0.01 order-margin bar does NOT hold across seeds, and false recall is far over the 0.15 ceiling on all three development seeds. This is a verdict on the 2-seed-CALIBRATED operating point (overfit), not on the reinstatement or interference capabilities, and not a new mechanism.

## Result (development seeds, frozen v6.GateConfig() defaults, evaluator inherited from v5+SFA)

<!--derived-->
Artifact: `research/findings/raw/replay_v5_sfa_order/replay_v6_multiseed_dev.json` (NumPy; provenance sidecar records argv + git SHA). One process ran all three seeds and collapsed the per-seed GO/NO-GO into one earned verdict.

| seed | intact rec | false recall | order margin (intact vs shuffled) | beats_order (+0.01) | false<=0.15 | per-seed |
|---:|---:|---:|---:|:---:|:---:|:---:|
| 414 | 0.0646 | 0.5000 | −0.0021 | FALSE | FALSE | NO-GO | <!--derived-->
| 415 | 0.0576 | 0.5000 | +0.0021 | FALSE | FALSE | NO-GO | <!--derived-->
| 410 | 0.0444 | 0.4615 | +0.0007 | FALSE | FALSE | NO-GO | <!--derived-->

<!--derived-->
Every development seed also fails `both_memories_recovered`, `weak_memory_present`, and `target_inhibition_improves_specificity` — the same stack that was clean on 412/413 is broadly seed-fragile here, not marginally short on one control. On calibration the same frozen config gave false recall 0.117 / 0.092 and order margins +0.049 / +0.014; the development seeds give false recall 0.46–0.50 and order margins ~0. The 2-seed calibration masked the fragility (the standing 6-seed rule: 2–3-seed indicators are unreliable; validate at 6+ before any generalization claim).

## Isolation — the false-recall blowup is NOT owned by the order-STDP

<!--derived-->
STDP-off reference on the SAME development seeds (`replay_v6_multiseed_dev_stdpoff.json`; v6 frozen config with `stdp_sleep=False`, i.e. the v5+SFA stack at the calibrated d=180 eviction, no order term): false recall is STILL 0.500 / 0.488 / 0.437 and order margins are ~0 (+0.000 / +0.0035 / +0.0028). So the primary blocker — retest false recall ~0.5 — is present WITHOUT the order-STDP: it is the underlying interference-control operating point (the SFA one-of-N eviction at d=180, and the reinstatement/opponent stack) tuned on 412/413 that fails to hold false recall on new seeds. The order-STDP still deposits a directional trace (nonzero sleep deltas), but its ordered>shuffled advantage is itself seed-fragile — on seed 414 SHUFFLED potentiates MORE than intact (sleep delta 4.35 vs 3.90), inverting the margin. The order-margin non-generalization is real but SECONDARY: with false recall already ~0.5, the interference control is the load-bearing failure.

## Decision and residual

<!--derived-->
Capability #4 (CLS replay consolidation) does NOT clear the reliability bar. Do NOT open held-out 417/418/419 (sealed until a development GO). Do NOT tune the frozen mechanism to these seeds (that would re-overfit the same way the calibration point did). The quantified residual: the whole cortical-consolidation interference control (SFA eviction d/a, opponent FS, reinstatement gains, and the STDP amplitudes/bounds) is set at BUILD on two calibration seeds and does not transfer — retest false recall swings from ~0.10 (calibration) to ~0.50 (development) at a fixed operating point, and the ordered-replay margin swings from +0.01…+0.05 to ~0. The named next mechanism (not built here): make the interference/eviction operating point EMERGE per-brain (a homeostatic set-point on retest false-firing / target activity, and a developmentally-tuned or activity-normalised SFA and STDP scale) rather than a fixed d/amplitude picked on two seeds — so the same rule self-calibrates across seeds instead of being hand-fit. The v6 order-STDP finding's 2-seed GO stands as reported (it explicitly named this multiseed run as the load-bearing confirmation); this finding records that the confirmation did NOT hold.

Scaffolds unchanged from v6, plus the now-explicit one this exposes: the interference/eviction/plasticity operating point is host-set at build on the calibration seeds, not self-organized per brain.
