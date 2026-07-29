# Objrel structural-read surpass — plan + decision tree (2026-07-05)

**Goal.** A biological, learned, spiking read-out that reads grammatical **role
from structure** (not just word position) — so the conversational brain
comprehends object-relative clauses ("the ball **that** the dog chased", where
word order ≠ roles), a genuine rung toward LLM-equivalent comprehension. Owner
directive: grind to ground; position-solvable is NOT the real use case.

## What is precisely established (do not re-litigate)

- **The reservoir feature ENCODES objrel.** A shift-invariant **linear** argmax
  over the reservoir feature reads objrel slot-0 (THEME) at **1.00 on every
  seed**. So the information is present and separable — this is **not** the
  irreducible Mikulasch-Priesemann decorrelation wall.
- **The POSITIVE spiking WTA deploy loses it** (0–3% objrel, all floors,
  with/without competition, from-scratch delta, ridge weights) — a common-mode
  pedestal (positive-shift + `WS_ENS_FLOOR`) swamps the subtle structural margin
  that the shift-invariant linear read cancels.
- **The SIGNED conductance deploy CAN express it** but is **operating-point /
  seed fragile**: fixed ridge-signed = objrel slot-0 **0.75 / 0.00 / 0.50** on
  seeds 42/44/100 (overfit); the seed-44 harness is degraded even for canonical
  (0.28). From-scratch signed **delta rule → position** (canonical 0.75–0.97,
  objrel ~0) at all scales, because of a **7:1 slot-0 AGENT:THEME class
  imbalance** the per-example rule follows.

⇒ Two coupled obstacles: **(i) class imbalance** pulls the learned read to
position; **(ii) the signed deploy's operating point is seed-fragile.**

## The ladder (climb until objrel generalizes multi-seed + passes anti-cheat)

| # | Mechanism | Fixes | Cost | Status |
|---|---|---|---|---|
| 1 | **Balanced delta** — oversample objrel | (i) imbalance | runner-side | RUNNING (42/44/100, ~4:45 PM) |
| 2 | **Ridge-init + refine** | (i) start structural | runner-side | finicky (inherits ridge overfit); low priority |
| 3 | **Homeostatic operating point** | (ii) seed-fragility | runner-side | designed below; build if 1 is seed-fragile |
| 4 | **Phase-domain read** (FHRR / resonate-and-fire) | removes the common mode entirely | new mechanism | designed below; the deep pivot |

### Decision tree — on the balanced multi-seed result (rung 1)

- **All 3 seeds do BOTH** (canonical high + objrel slot-0 high) → run the
  **anti-cheats** (scramble → chance, syn-lesion → collapse, canonical control
  still high); if clean → **surpass**. Promote to `--mode c4` (learned-signed),
  wire into the RUNG B read-out, re-run the full anti-cheat suite.
- **Seed 42 only** (44/100 fail) → **seed-fragility confirmed** → build **rung 3
  (homeostatic op-point)**: the deploy works with the right operating point per
  draw; automate it biologically.
- **None do both** (still position, or the canonical/objrel tradeoff persists) →
  the signed *rate/count* deploy is fundamentally both-tasks-limited → escalate
  to **rung 4 (phase-domain read)**.

## Rung 3 design — homeostatic self-calibrating operating point

**Diagnosis.** The signed conductance subtraction `g_i·(E_i − v)` is subtractive
only when `v` sits near rest; a fixed `WS_ENS_FLOOR` puts `v` in the subtractive
regime for some reservoir draws and the shunting/saturating regime for others
(seed 44). The delta rule adapts **weights**, not the **operating point**.

**Mechanism (Turrigiano homeostatic scaling — Dale-legal, point-neuron-feasible,
runner-side).** Before training/deploy, per draw, calibrate the ens floor/gain
so the ensembles fire at a fixed **target rate** at a reference drive:
1. Drive each role ens at a reference (mean reservoir feature) input.
2. Measure its firing rate; adjust `WS_ENS_FLOOR` (or a per-ens gain) up/down by
   a homeostatic rule (`floor += η_h·(target − rate)`) until `rate ≈ target`.
3. Freeze the calibrated floor; then run the (balanced) signed delta rule.

This is *not* per-seed tuning of the answer — it calibrates the **operating
point** to a firing target (a biological set-point), draw-agnostic. Precedent:
the earlier signed arc named "self-calibrated bias delivery"; catalog
homeostasis (Turrigiano synaptic scaling); the project's validated read-out
normalization (spike-frequency adaptation + feedforward inhibition = 96% host).

**De-risk.** Add a `HOMEO_TARGET` pre-pass to `step8_learned_signed.py`;
multi-seed (42/44/100); PASS = objrel slot-0 high on ALL seeds (esp. 44, the
degraded harness) with canonical held. Anti-cheat: the calibration is driven by
a firing set-point, not by the objrel labels (no answer leakage).

## Rung 4 design — phase-domain structural read (the deep pivot)

**Diagnosis.** If rate/count reads fundamentally can't resolve the structural
margin under any operating point, the problem is the **coding domain**: a
common-mode pedestal swamps a rate margin. The composer hit this exact wall
(the `onoff` opponency SNR wall) and **escaped by moving to FHRR phase coding**
(resonate-and-fire phasor neurons + complex synapses): information in **phase**,
unit magnitude, **no common mode** to swamp.

**Mechanism.** Read the reservoir feature into **phase** (resonate-and-fire, the
project's `NeuronModel.RESONATE_AND_FIRE` + complex synapses already on the
bridge): the role decision becomes a **phase-coherence** winner (which role
phasor the feature aligns with), not a rate race. The structural margin survives
because there is no additive pedestal in the phase domain.

**De-risk.** Project the reservoir feature onto 3 role phasors via complex
synapses; winner = max phase coherence; multi-seed objrel. This reuses the
FHRR-on-bridge machinery (`rf_*` ops, `RFPhasorComposer`). Larger build → only
if rungs 1–3 fail; but it is the principled escape from the common-mode family.

## Parallel arcs (independent of objrel — ready to run)

- **A→W BRIDGE-C fix** (fully-spiking word production for transitive/PP
  constructions). The EMERGE-75 boundary is a **co-training** issue (3 high-freq
  prepositions {to,on,is} + 13 object nouns on one 16-pool bridge — the
  Goldilocks signature), so the fix is the named **EMERGE-75b pool-reassignment**:
  move {to,on,is} onto BRIDGE-F's free filler pools (it has 11), leave BRIDGE-C
  with just the 13 nouns, retrain both on GPU (now unblocked — `train_word_to_pool`
  verified on cupy 2026-07-05), update the `UnifiedNeuralSpell` dispatch.
  *Requires editing the validated EMERGE-68/75 runners additively → focused arc,
  not a risky parallel edit.*
- **RUNG B synaptic comprehension→composition hand-off** (the functional one
  brain): project the reservoir's role output → a role-ensemble region via the
  learned read-out as a fixed `RegionPathway` → WTA-select → feed the composer's
  parser-firing route, replacing the host `{role:word}` dict. CPU-feasible.
- **cupy-path `cp_traits=None`** in `build_unified_bridge` (task #5): unblocks
  the conversational bridge on GPU (relevant for 3090/cloud scaling).

## Discipline (why this plan is cautious)

Multi-seed + anti-cheat gate **every** claim — this session already caught two
overclaims (the population-lever reservoir-position confound; the
signed-conductance seed-42 result that failed multi-seed). A partial or
single-seed win is not a win.

## Provenance / artifacts

- Finding: `research/findings/2026-07-04-biological-learned-readout-delta-rule.md`
  (the c3 delta rule + the objrel isolation + the surpass attempts).
- Probes: `research/findings/raw/signed_conductance/step7_*.py` (isolation),
  `scratchpad/step8_learned_signed.py` (the learned-signed delta rule + balance +
  ridge-init flags).
- Read-out runner: `research/runners/_rungB1c_spiking_reservoir_synaptic_readout_derisk.py`
  (`--mode c3`); signed machinery: `research/runners/_rungB1c_signed_readout_derisk.py`.
- State: `research/findings/AUTONOMOUS_STATE.md` CYCLE 921.
