---
type: finding
status: live
date: 2026-08-20
mechanism: idle-tick-replay-stabilization
lane: memory
seeds: [42, 43, 44, 100, 101, 102]
instrument: research/runners/_emergent_replay_specificity_derisk.py — emergent pattern-completion replay (plastic recurrent pre->pre assembly, untargeted noise-subset reactivation) + starting-weight-gated write, vs no-replay / no-presynaptic-reactivation / never-encoded controls, recall-after-delay
runner: research/runners/_emergent_replay_specificity_derisk.py
artifacts:
  - research/findings/raw/_emergent_replay_specificity.json
---
# 6-seed GO: emergent pattern-completion replay CLOSES the idle-consolidation specificity gap (learn-through-use advances)

Artifact: research/findings/raw/_emergent_replay_specificity.json

**One line.** The earlier rung-3 result (2026-08-20-idle-replay-trace-stabilization ... UNDEFINED) showed idle-tick
replay boosts recall directionally but NON-SPECIFICALLY — the same host-directed replay dose partially wrote a
never-encoded pathway (46-67% of the real trace). This closes it: make the replay CONTENT emergent — a plastic
recurrent pre->pre cell assembly, reactivated by UNTARGETED noise into a random ~18% subset, lets pattern-completion
recruit the rest ONLY when a real assembly was encoded. An unencoded pathway has baseline recurrent weights, so noise
does not pattern-complete it and its BTSP co-activation stays weak — it is not fabricated. 6-seed GO: G3 specificity
now clears on all six (moat_replay 3-16% of replay, vs 46-67% before), with G1 (replay beats no-replay) + G2 (lesion
vanishes) + PC (encoded assembly recruits more than unencoded under the identical dose) all holding.

## The mechanism (why emergence buys specificity)
- **Emergent replay.** The "pre" region carries plastic recurrent internal connectivity (`internal_density=1.0`) bound
  into a cell assembly during encode by an eligibility-trace Hebbian rule (mirrors the on-bridge BTSP kernel's own
  `etilde_pre * is_post` form). Idle reactivation drives only a random ~18% subset of pre with untargeted noise (the
  IDENTICAL draw for `replay` and the never-encoded `moat_replay`) and lets the recurrent loop pattern-complete the
  rest — real spiking propagation, not host-addressed reactivation of specific cells. An ENCODED assembly's
  potentiated loop recruits the rest of `pre_idx` (pre-activation replay=0.041 vs moat=0.023); an unencoded one cannot, <!--derived-->
  so its target-side BTSP co-activation stays too weak to write a trace.
- **Starting-weight-gated write.** A modest metaplastic gate suppresses the BTSP delta on target synapses that enter a
  tick below 30% of the tag-and-capture barrier (to 20% of their would-be delta), making the write more strongly
  conditional on a pre-existing tag. This is LOAD-BEARING, not insurance (see the honest note).

## 6-seed result (numpy; runner's own Verdict machinery; independently reproduced)
| seed | replay | noreplay | replay_nopre | moat_replay | moat/replay |
|---|---|---|---|---|---|
| 42 | 0.0550 | 0.0138 | 0.0175 | 0.0088 | 15.9% | <!--derived-->
| 43 | 0.0425 | 0.0063 | 0.0075 | 0.0025 | 5.9% | <!--derived-->
| 44 | 0.0387 | 0.0050 | 0.0075 | 0.0013 | 3.2% | <!--derived-->
| 100 | 0.0288 | 0.0063 | 0.0063 | 0.0037 | 13.0% | <!--derived-->
| 101 | 0.0400 | 0.0050 | 0.0063 | 0.0025 | 6.2% | <!--derived-->
| 102 | 0.0387 | 0.0075 | 0.0088 | 0.0050 | 12.9% | <!--derived-->
G1/G2/G3/PC require-checks + both separation controls PASS on all 6 -> VERDICT GO.

## Honest note (the 3-seed trap, caught in the act)
Mechanism 1 (emergent replay) ALONE passed a 3-seed pilot (42/43/44) but FAILED the full 6 — seed 102's moat_replay
reached 45.2% (over the 40% bar), seed 42 sat exactly at 40.0% with zero margin. This is the project's own "3-seed <!--derived-->
indicators unreliable" lesson: emergent content narrowed the residual substantially but did not by itself close it at
this network's scale; the metaplastic gate does real, non-trivial work in the final GO. Recorded in the artifact's
HONEST_NOTE. Also honest: the Hebbian assembly-potentiation rule + tag-and-capture maintenance + metaplastic gate are
runner-level weight-update MODELS, not yet `sim/` kernels (the WRITE, assembly formation, pattern-completion
propagation, and RECALL all run spiking on the real bridge).

## Scope / next rung
This closes the specificity residual that left the learn-through-use rung UNDEFINED — idle-tick replay now
strengthens a recent trace AND is specific (it does not fabricate unencoded associations), 6-seed. Next rung
(artifact NEXT_RUNG): port the Hebbian assembly rule + tag-and-capture + metaplastic gate to guarded default-off
`sim/` kernels; model pre-assembly forgetting; wire under `continuous_engine.py`'s idle tick (default-off first,
lesion-proven) so the brain genuinely LEARNS between turns; and isolate mechanism-1 vs mechanism-2 margin
(`--metaplastic-gate-frac 0`). Not wired live yet — the GO earns the wiring. (Agent-built, independently re-run +
reproduced; TERMS.md-checked: "stabilization".)
