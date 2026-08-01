---
type: finding
status: live
date: 2026-08-01
mechanism: affective-theory-of-mind
---

# W5 affective theory of mind — 6/6-seed GO, adversarially verified (2026-08-01)

## Result
A new lane-C de-risk, **W5 affective theory of mind**, closes at a **6/6-seed GO** (local CPU,
`SIM_BACKEND=numpy`, seeds 42 43 44 100 101 102, n_trials=60). The brain infers ANOTHER agent's EMOTION
(valence) from THAT agent's WITNESSED situation, held in an OTHER-tagged affect model that is DISSOCIABLE
from the system's OWN affect. It is the affective companion to W3 (the agent-keyed false-BELIEF register):
W3 turns the self-schema outward to model another agent's BELIEF; W5 turns the P0.3 affect region outward
to model another agent's FEELING. All four go-components True on every seed.

<!--derived-->
(Aggregate means/ranges below are rounded re-quotes of the full-precision values in the cited artifact.)
Aggregate (chance 0.5): other-attribution **1.000** · other|incongruent **1.000** vs EGOCENTRIC|incongruent
**0.000** · other-lesion **0.514** (collapsed) · scramble **0.483** (collapsed) · belief-track **1.000** vs
reality-baseline **0.000**.

Runner: `research/runners/_affective_tom_derisk.py` (NEW; reuse-by-import, NO `sim/` edit).
Artifact: `research/findings/raw/_affective_tom/summary_6seed.json`.

## Mechanism (brain-based; reuse-by-import, NO `sim/` edit)
Two P0.3 `AffectStateBrain` instances (verbatim import from `_affect_state_region_derisk.py`) on separate
numpy bridges — a clean self/other separation, the same "separate slot per agent" motif W3 uses for belief:
- **SELF affect model** — appraised on the SYSTEM's own situation valence.
- **OTHER affect model (OTHER-tagged)** — appraised on the OTHER agent's WITNESSED situation valence
  (F3 appraisal on the other-schema). Each region is the P0.3 opponent slow-NMDA attractor
  (affect_vplus/vminus + Namburi-Tye cross-inhibition); appraisal enters via the diffuse neuromodulator bus.
- **The emotional attribution / speech tone** = a SYNAPTIC read of the OTHER model's gated output: the affect
  state biases recall_pos (V+) vs recall_neg (V-) through the ONE `affect_out` transmission gate. `tone_sign =
  sign(rate(recall_pos) - rate(recall_neg))`; +1 = "share-joy/positive" tone, -1 = "comfort/negative" tone.
  This number is never host-set — it is a difference of two SPIKE RATES from the OTHER-tagged region.

## Why this is ToM and not egocentric projection (the load-bearing dissociation)
The trivial way to "report another's emotion" is to project your OWN affect. That FAILS whenever the other is
in a different situation. The decisive arm is the INCONGRUENT scenarios (self got good news, other got bad
news): the OTHER model attributes correctly (**other|incong = 1.000**), while using the SELF affect as the
attribution is exactly wrong (**egocentric|incong = 0.000**). Combined with the output-lesion and scramble
collapses below, this establishes a SEPARATE, other-driven, load-bearing affect attribution.

## Anti-cheat controls (all collapse, 6/6 seeds)
<!--derived-->
All values are rounded re-quotes from `research/findings/raw/_affective_tom/summary_6seed.json`
(aggregate + `per_seed`).
- **Other-lesion collapses** — lesion the OTHER model's OUTPUT (`set_affect_lesion(True)` -> `affect_out`
  gate = 0). The other pools keep appraising; only the gated read-out is severed -> attribution -> chance,
  mean **0.514** (per-seed 0.467-0.567, all <= 0.65). Proves the OUTPUT carries the attribution.
- **Scramble the other's witnessed valence** across trials -> attribution rides the wrong situation ->
  scored vs the TRUE other valence -> mean **0.483** (per-seed 0.400-0.533). Proves it rides the ACTUAL
  other-situation, not a fixed response.
- **Self/other dissociation** holds on every seed (other|incong 1.000, egocentric|incong 0.000).

## Characterization (the W3 x P0.3 integration, reported not gated)
On the false-belief-of-affect subset (the other WITNESSED a valence opposite to reality) the inferred emotion
tracks the other's BELIEF (witnessed), **1.000**, and a reality-appraised baseline would be WRONG, **0.000**.
This demonstrates the pipeline appraises the other's WITNESSED (believed) situation, so the other "feels
according to what THEY perceived" — the affective analogue of W3's false belief.

## Honest scope + caveats (the deliverable includes the boundary)
- **Valence only (good/bad).** Matches the P0.3 substrate's CHARACTERIZED bistable good/bad latch (P0.3 is a
  QUALIFIED-GO/BOUNDARY: robust valence sign, not a graded discrete-emotion circumplex). Fine discrete
  emotions need the SAME graded-circumplex surpass P0.3 already named (a line/bump attractor with SFA
  eviction / the dendritic substrate), NOT a new wall.
- **egocentric|incong = 0.000 is partly definitional** (incongruent trials have self_v = -other_v by
  construction, and the self model tracks self at 1.000). It shows projection would be exactly wrong; the
  substantive non-projective evidence is other|incong = 1.000 PLUS the lesion + scramble collapses.
- **The situation -> valence appraisal is the legitimate world/perceptual input** (P0.3's interface, DR-2
  learned-tag precedent). The ToM-specific neural work is (a) a SEPARATE OTHER-tagged affect state and (b) the
  synaptic tone read-out, verified load-bearing by the dissociation, lesion, and scramble.
- **A FUNCTIONAL affective-mentalizing correlate** (a separate, other-driven, dissociable affect attribution)
  — NOT a claim of phenomenal access to another mind's feelings.
- numpy-CPU read on real spiking Izhikevich bridges ("numpy" is the backend, not a host shortcut).

## Adversarial verification (verify-go; lenses with teeth, all PASS)
<!--derived-->
(All numbers in this section are re-quoted from the cited `summary_6seed.json` / the seed-42 reproduction.)
- **Reproducibility/power** — 6/6 seeds GO; seed 42 alone reproduces the 6-seed row byte-identical
  (attr 1.000, lesion 0.500, scramble 0.533). Effect >> seed-to-seed noise; no single seed carries it.
- **Gate-cheat** — the anti-cheat controls (lesion, scramble, egocentric) are INVOKED every trial and sit in
  `go_components`, not merely defined.
- **Control-integrity** — the lesion is ONE variable (`affect_out` gate on `other_brain` only). Arms are NOT
  both saturated: intact 1.000 vs lesion 0.514 is a responsive range, so the comparison has power.
- **MASS artifact** — the tone is a recall-pool mass comparison (recall_pos vs recall_neg), but the mass is
  EARNED: scramble (a permuted-target control) collapses it to 0.483, lesion to 0.514, and intact = 1.000
  requires the sign to FLIP with valence (no fixed pool bias; under lesion it reads chance).
- **Instrument-trust** — the runner's own verdict printed GO; `tools.verdict.Verdict` EARNED it (floor,
  dissociation, control-separation, per-seed collapse); the metric provably CAN read chance.
- **Seeding** — `AffectStateBrain` sets `cfg.seed = int(seed)` (`_affect_state_region_derisk.py:142`), NOT
  `actual_seed_used`; the reproduction is deterministic. `tools/engagement_check.py`: no void signature.

## Roadmap
Advances lane C (Self/Workspace/ToM), Stage-4 ToM ladder (affective rung), building on W3 (false-belief GO)
and P0.3 (affect-state QUALIFIED-GO). Runner is CPU, disjoint from the GPU lanes. The graded-circumplex
(discrete-emotion) surpass is shared with P0.3, not a new wall.
