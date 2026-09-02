---
type: finding
status: live
date: 2026-09-02
mechanism: crossedge-surprise-to-worldmodel-error-gated-update
lane: onebrain-integration
seeds: [42, 43]
artifacts:
  - research/findings/raw/_crossedge_surprise_worldmodel_smoke.json
runner: research/runners/_crossedge_surprise_worldmodel_derisk.py
builds_on:
  - research/findings/2026-09-02-onebrain-crossregion-integration-DESIGN-ranked-crossedges.md
  - research/findings/2026-09-01-surprise-episodic-encode-decision-crossedge-GO.md
---

# Cross-edge #1 (design rank 1) — D2 surprise gates an ERROR-GATED update of the E2 world-model's forward transition: 2-seed numpy SMOKE-GO, PARTIAL pending the 6-seed cupy soak

**One-line:** The E2 affective world-model's own declared residual is "TEACHER-DRIVEN: the transition is LEARNED
(Hebbian co-fire) but not self-organized from conversation" — its `state -> pred_{pos,neg}` valence transition is
trained ONCE by a host schedule then FROZEN, and nothing makes re-learning conditional on the model being wrong.
This de-risk makes the world-model's OWN spiking prediction-error unit (its `surprise_{pos,neg}` pools) the THIRD
FACTOR that gates plasticity on that existing transition: on a SURPRISING turn the gate opens, the state co-fires
with the pred pool matching the OBSERVED valence, Hebbian strengthens `state -> pred_{observed}`, then the gate
re-freezes. 2-seed numpy SMOKE-GO (2/2, seeds 42/43) on all four bars; a 6-seed cupy soak (42/43/44/100/101/102)
is QUEUED and is the decisive test. This is a de-risk hypothesis, **not** a GO/`wired`/`integrated` — PARTIAL
pending that soak. No production wire-in, no `sim/` edit, additive.

## 1. Verify-first (checked before building, not after)

The design finding (`2026-09-02-onebrain-crossregion-integration-DESIGN-ranked-crossedges.md`) ranks this edge #1
and states "The surprise<->world-model ... couplings are confirmed UN-BUILT." Re-checked here by grepping
`research/findings/` for `surprise->world-model` / `world-model error-gate` / `error-gated forward` (no prior
runner or finding). The two organs already co-reside (the two-pool migration GO); the shipped
surprise-gates-plasticity backbone (`2026-09-01-surprise-episodic-encode-decision-crossedge-GO.md`) is the pattern
reused. Confirmed un-built.

**Scope choice, declared:** the "D2 surprise" source used here is the world-model circuit's OWN internal
predictive-coding error unit (`surprise_pos`+`surprise_neg` in `build_world_model_circuit`) — the D2-CLASS
expectation-violation signal that lives inside the affective forward model — NOT the separate semantic-mismatch D2
organ (`surprise_production_organ.py`, over stored (agent,action)->patient facts). This keeps the edge
self-contained on the E2 circuit: the affective error that gates the update and the affective transition being
updated are on the same axis, which is exactly what "error-gated forward-model update" means. Merging the
separate semantic surprise organ as the gate source is a named, heavier follow-on (a cross-organ merge), not
attempted here.

## 2. The mechanism (brain-based; the third factor is a spike rate)

Per update turn on target state s0 (which the initial training taught to predict valence v0), with a persistently
surprising observation obs = -v0:

1. **Read the brain's own error** — cue s0 (establish the top-down prediction), drive s0 + the observed valence,
   read `surp_hz = rate(surprise_pos)+rate(surprise_neg)` off `cp_firing_states`. No host compare of observed vs
   predicted; the surprise is the un-inhibited channel's spikes.
2. **Third-factor gate** — the transition Hebbian window OPENS iff `surp_hz >= gate_threshold` (a build-time
   host-calibrated threshold on the spiking rate — the SAME declared boundary `surprise_production_organ` states:
   "the DECISION ... is a threshold on that spiking rate").
3. **Update (only if gated)** — co-fire state s0 with `pred_{observed}` (the environment delivers the observed
   valence as drive — the legitimate sensory boundary, identical to how `train_transition`/`run_update_on_error`
   already teach), Hebbian strengthens `state[s0] -> pred_{observed}` (bounded by `hebbian_max_weight`, with the
   circuit's Miller-MacKay afferent competition), then RE-FREEZE.

**Biology.** Predictive coding: the error signal drives learning in the generative model (Rao & Ballard 1999,
*Nat Neurosci* 2:79-87, PMID 10195184, verified via PubMed 2026-09-02, logged to
`research/queue/.external_searches.jsonl` lane `onebrain-integration`). Novelty/prediction-error gates plasticity
via the hippocampal-VTA DA loop (Lisman & Grace 2005, *Neuron* 46(5):703-713, PMID 15924857 — the same anchor the
shipped surprise->episodic edge cites). The surprise unit's firing IS the gating third factor.

**A genuine, emergent, self-limiting trajectory (not a host schedule).** Because the gate is driven by the neural
surprise, the update SELF-LIMITS: as `state->pred_{observed}` grows, cueing s0 begins to recall the observed pool,
which cancels the observation, so `surp_hz` FALLS below threshold and the gate self-closes. On seed 42 the surprise
trace ran `[24.3, 8.7, 3.1, 3.1, ...]Hz` against a threshold of 8.5 — the gate opened exactly 2 of 16 turns, and
those 2 credited steps already shifted the queryable prediction +125Hz toward the observation. This is
predictive-coding error-minimisation: the forward model re-learns until it agrees with the world, then stops.

## 3. 2-seed numpy SMOKE result (seeds 42, 43) — SMOKE-GO 2/2

(rounded from the cited smoke artifact; open the JSON for full precision. numpy CPU, ~13s/seed.)

<!--derived-->

| seed | v0 | gate thr (exp/vio Hz) | SURPRISING shift; w_obs; opens | EXPECTED | LESION | LES-FORCED | attributable (vs lesion / vs expected) | byte-off |
|---|---|---|---|---|---|---|---|---|
| 42 | +1 | 8.5 (0.0/24.3) | +125.3Hz; 0.057->0.318 (5.6x); 2/16 | shift +0.4, 0 opens, no wt change | shift -0.4, 0 opens, surp 0, 4608 edges zeroed | shift +396, 16 opens | 1.003 / 0.997 | weights+conn identical |
| 43 | -1 | 15.9 (0.0/45.5) | +124.0Hz; 0.034->0.282 (8.4x); 2/16 | shift +1.0, 0 opens, no wt change | shift -1.0, 0 opens, surp 0, 4608 edges zeroed | shift +417, 16 opens | 1.008 / 0.992 | weights+conn identical |

The four bars, each anchored on an UNAMBIGUOUS structural fact rather than a noisy read:

- **(a) EMERGENCE** — on the SURPRISING sequence the `state->pred_{observed}` transition weight GROWS from its
  near-zero post-train baseline (5.6x / 8.4x; `dw` well above the absolute floor) by the substrate's own Hebbian
  rule, gated open by the surprise, and the queryable prediction shifts +124-125Hz toward the observation (>> the
  5Hz floor, ~25x the few-Hz read-noise). Emergent, not a pre-set weight.
- **(b) ERROR-GATED SELECTIVITY** — the EXPECTED arm runs the IDENTICAL gating code but the observed valence
  CONFIRMS the prediction, so surprise stays silent (`surp_max=0`), the gate never opens, and the transition
  weight is EXACTLY unchanged. The ONLY difference from the surprising arm is whether the brain's error unit
  fired — so the learning is owned by the neural surprise, not by "the flag is on."
- **(b) LESION-VANISHES** — `SURPRISE_WORLDMODEL_LESION`: zeroing the `obs->surprise` sensory-drive edges (4608
  synapses) keeps surprise silent even on the surprising sequence, so the gate never opens and the weight is
  exactly unchanged. **Specificity:** on that SAME lesioned circuit, host-forcing the gate open still updates
  (shift +396/+417, 16 opens) — the transition pathway is intact, so the lost update is the severed surprise
  SIGNAL, not disabled plasticity. The lesion is verified to hold at measurement (`surp_max<1.0`).
- **(c) ATTRIBUTABLE** — the surprising-arm shift is attributable to the surprise-gated update: `attributable_to`
  reads ~1.00 vs the lesion (severed signal) AND ~0.99 vs the expected arm (same machinery, no surprise) on both
  seeds. (The vs-lesion fraction is marginally >1.0 because the lesion arm's read-noise moved slightly opposite —
  reported honestly by the helper, not clipped.)
- **(d) BYTE-OFF** — `SURPRISE_WORLDMODEL_UPDATE=0`: no update runs (the current frozen path), so the transition
  weights are EXACTLY unchanged and the circuit connectivity is byte-identical (asserted by exact weight + hash
  equality, `docs/TERMS.md` `byte-identical`). The edge adds NO synapse — it is a third-factor GATE on the
  existing transition, so additive-off is structural by construction.

## 4. Honest residuals (declared, not hidden)

- **NOT `self-organized` (per docs/TERMS.md).** The plasticity GATE is the brain's own spiking prediction-error
  (this is what the edge adds over the frozen organ), but the teach DIRECTION is the OBSERVED valence delivered
  as a sensory drive (environment boundary), and the gate THRESHOLD is a build-time host calibration. This is
  `host-supervised` / `error-GATED`, not self-organized. The target state and the turn schedule are host/teacher
  scaffold.
- **Internal affective surprise as the source**, not the separate semantic D2 surprise organ (§1) — a self-
  contained scope, with the cross-organ merge as the named follow-on.
- **The gate is a host threshold on a neural rate**, not yet an in-substrate neuromodulatory synapse. An actual
  gate population fed by a plastic `surprise->gate` cross-synapse (the shipped surprise->episodic-encode
  topology) whose firing gates the window is the further-biologized rung; here the third factor is read from the
  same predictive-coding surprise pool the circuit already computes.
- **Runner-level SMOKE only.** 2 seeds, numpy. Per `docs/TERMS.md` this is neither `GO` (the gate's own 6-seed
  verdict is not in) nor `wired`/`integrated`. The 6-seed cupy soak is QUEUED and decisive; report status as
  **PARTIAL-pending-6seed-cupy-soak**. No production wiring, no default flip, no `sim/` edit.

## 5. Files

`research/runners/_crossedge_surprise_worldmodel_derisk.py` (NEW) ·
`research/findings/raw/_crossedge_surprise_worldmodel_smoke.json` (smoke artifact + provenance sidecar). Reused,
unmodified by import: `research/runners/_affective_world_model_derisk.py` (`build_world_model_circuit`,
`train_transition`, `_drive_read`, `_hard_reset`, `_idx`, `_valence_map`), `tools/lab.py` (`attributable_to`,
`lever`). No `sim/` file touched; no `webapp/server.py` or `*_production_organ.py` edit. Functional read-outs
only; no phenomenal-experience claim.
