---
type: finding
status: live
date: 2026-08-13
mechanism: one-brain-merge
---

# Merge → production, RUNG 2: a 3rd FACULTY (reconsolidation) joins the shared pool byte-identically (GO); the next REGION-OWNING organ (comprehension) is a MEASURED dt BOUNDARY — the remaining organs each need per-region scoping of a faculty-load-bearing global flag

**Date:** 2026-08-13 · **Runner:** `research/runners/_onebrain_merge_rung2_verify.py` · **Artifact:**
`research/findings/raw/_onebrain_merge_rung2_6seed.json` (6 seeds 42/43/44/100/101/102, `SIM_BACKEND=numpy`).
**Additive only:** one default-preserving edit to `research/runners/_phaseB_multicue_competition_spiking_derisk.py`
(three kwargs `dt_ms`/`homeostasis`/`per_region_thresh`, defaults `0.5`/`False`/`False` → byte-identical to
before — the instrument that MEASURES the comprehension boundary). **NO `sim/` edit.** Builds on rung 1
(`2026-08-13-merge-production-integration-rung1-GO.md`: surprise + world-model on ONE shared bridge, byte-exact).

## What this rung asked, and the honest split answer

Rung 1 put the two MOST COMPATIBLE production organs (D2 SURPRISE + E2 WORLD-MODEL) on one shared spiking
bridge. Rung 2 asks: can MORE join? The answer splits cleanly, and both halves are measured 6/6:

| axis (6 seeds) | result | verdict |
|---|---|---|
| (A) rung-1 pair STILL byte-identical merged-vs-co-resident (surprise + world-model, max delta **0.0**) | 6/6 | GO (regression guard) |
| (B) RECONSOLIDATION faculty byte-identical on the shared pool (window read = merged surprise, max delta **0.0**) | 6/6 | GO |
| (B) reconsolidation faculty ALIVE on the merged bridge (window OPENS on contradiction, CLOSED on confirm) | 6/6 | GO |
| (C) COMPREHENSION well/ill AUC clears the 0.80 gate at the shared `dt=1.0` | **3/6** | **BOUNDARY** |
| (C) comprehension well/ill AUC clears 0.80 at its NATIVE `dt=0.5` | 6/6 | (native works) |

## (B) GO — a 3rd production FACULTY runs on the shared pool, byte-identically

The RECONSOLIDATION (belief-revision) organ (`reconsolidation_production_organ.py`) owns **no neurons of its
own**: its spiking reconsolidation WINDOW *is* the D2 surprise organ (a `cp_firing_states[surprise]` read; it is
"a co-resident belief-revision organ … owns NO new spiking circuit of its own"). So on the merged bridge the
belief-revision faculty runs on the SAME shared pool that carries surprise + world-model. Verified: its window
read (`opened` + `surprise_hz`) is byte-identical (**0.0**) merged-vs-co-resident, all 6 seeds, and the faculty
is alive (the two CONFIRM items keep the window closed, the two CONTRADICT items open it). This is a real
"more faculties on the ONE shared substrate" result — a faculty riding the merged pool WITHOUT adding a pool
member. (Byte-identity FOLLOWS from rung-1's surprise byte-identity — this is the same `judge` call — so it is a
composition-holds + faculty-alive confirmation, not a new mechanism; b3-noncontradiction likewise rides the
production recall composer, another faculty on a shared substrate, though on the recall bridge not yet merged
with this pool.)

## (C) BOUNDARY — why no additional REGION-OWNING organ joins the pool byte-identically without a sim/ edit

The rung-1 pair merges byte-identically for one reason: surprise + world-model are the ONLY two production
organs whose dynamics-config is IDENTICAL where it matters (`dt_ms=1.0`, IZHIKEVICH, homeostasis default ON, NO
parameter-heterogeneity, NO stdp/reward/neuromod, GABA_B inert). Every OTHER own-neuron production organ diverges
from that config on a **GLOBAL-only, faculty-LOAD-BEARING** flag — so putting it on the shared bridge at the
shared operating point CHANGES (degrades) its production read, which the mission forbids ("do NOT force a merge
that changes production reads"). The comprehension case is the one measured here in full:

- **COMPREHENSION** (the Wong-Wang `SpikingRoleCompetition` role monitor) sets `dt_ms=0.5`; the shared pool runs
  at `dt_ms=1.0`. The N-organ de-risk (`...-Norgan-GO.md`) reconciled the role WTA's COARSE binary selection to
  `dt=1.0` (6/6). But the PRODUCTION read is the GRADED well/ill role-resolution margin, and it does NOT survive
  the reconciliation: AUC at `dt=1.0` = **0.865 / 0.859 / 0.589 / 0.758 / 0.880 / 0.562** <!--derived--> (3/6 ≥ 0.80, mean
  ≈ 0.75) vs a perfect **1.000 on every seed** at its native `dt=0.5`. The Wong-Wang NMDA integration needs the
  finer step to keep the margin graded; forcing `dt=1.0` compresses well-vs-ill (well ≈ 0.14–0.18 vs ill ≈
  0.07–0.10). So comprehension cannot join at `dt=1.0` without degrading its faculty — and **per-region `dt`
  cannot be byte-identical** (the fused integrator steps ALL neurons at one `dt`; a per-region micro-step or a
  current-rescale is an APPROXIMATION, not the standalone `dt=0.5` trajectory bit-for-bit). This is a hard
  boundary, not a tuning gap.

The rest of the own-neuron organs, mapped by config (the load-bearing flag verified in each builder; the
comprehension row is the one MEASURED above):

| organ | own-neuron builder | diverging GLOBAL flag(s), load-bearing | per-region hook status |
|---|---|---|---|
| comprehension | `SpikingRoleCompetition` | `dt_ms` 0.5 (MEASURED: graded margin needs it) | **none possible byte-exact** (integrator is single-dt) |
| metacog | `build_metacog_bridge` | `enable_parameter_heterogeneity` True ("graded rate code REQUIRES het"); homeostasis OFF | needs NEW per-region param-het (name-keyed, mirroring `per_region_threshold_heterogeneity`) |
| pragmatic (ToM/RSA) | `_recursive_tom_rsa_derisk` | param-heterogeneity True; homeostasis OFF | same as metacog |
| affect (mood ladder) | `build_one_brain` | param-heterogeneity True; neuromodulator subsystem ON; homeostasis OFF | per-region param-het + per-region neuromod |
| causal (what-if) | `_causal_forward_model_derisk` | `enable_stdp`+`enable_reward_modulation` True; homeostasis OFF | per-region plasticity/reward scoping |
| curiosity | `build_curiosity_bridge` | stdp+reward+neuromodulator subsystem True; param-het True | per-region plasticity + neuromod |

**What already has a per-region hook (so is NOT the blocker):** NMDA (`BrainRegion.enable_nmda`, benign-union —
proven in `...-Norgan-GO.md`), threshold heterogeneity (`per_region_threshold_heterogeneity`), homeostatic
idle-drift (`per_region_homeostasis_isolation`), and homeostatic-threshold USE (the `cp_homeostasis_neuron_mask`
already gates which neurons apply adapted thresholds). **What is genuinely missing** and blocks the next cluster
is per-region **parameter-heterogeneity** (needed byte-exact for metacog/pragmatic/affect — the LARGEST cluster,
3 organs) and per-region **plasticity/neuromod** (causal/curiosity). `dt` is the one flag that CANNOT be made
per-region byte-exact.

## Named next rung (highest leverage)

Build a default-off, additive `sim/` flag `per_region_parameter_heterogeneity` — a name-keyed per-region param
substream mirroring the existing `per_region_threshold_heterogeneity` (lines ~3330–3380 of `sim/bridge.py`) — so
an organ's Izhikevich per-neuron jitter is invariant to co-residents. Combined with a per-region homeostasis
ENABLE (extend the existing `cp_homeostasis_neuron_mask` population), it unlocks the metacog/pragmatic/affect
cluster onto the shared pool WITHOUT changing their production reads. That is a distinct engine-feature lane;
this rung MAPS + QUANTIFIES it rather than forcing a bad merge.

## No regression (flag OFF = today)

- The one tracked edit (`_phaseB_multicue…` kwargs) is DEFAULT-PRESERVING: the production comprehension monitor
  calls `SpikingRoleCompetition(seed=…)` with no extra kwargs → `dt_ms=0.5`, `homeostasis=False`,
  `per_region_thresh=False` → the standalone build is byte-identical.
- `brain_chat_tui --smoke` is **byte-identical** to a stashed pre-change baseline (JSON verdict compared
  field-by-field, path-normalized: equal).
- `pytest tests/test_determinism.py -q` → **9 passed**.

## Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners._onebrain_merge_rung2_verify \
    --seeds 42,43,44,100,101,102 --out research/findings/raw/_onebrain_merge_rung2_6seed.json
```

## Honest scope / non-claims

- `wired: shared-substrate now serves 3 FACULTIES (surprise + world-model organs own the neurons; reconsolidation
  rides them) / on_by_default: NO (opt-in, `BRAIN_ONEBRAIN_MERGE` default-off, unchanged from rung 1) /
  scaffold_retired: none new.` Functional read-outs only; no phenomenal claim.
- **No NEW region-owning organ was migrated.** The pool still owns exactly the rung-1 pair's neurons; rung 2 adds
  a faculty that rides them (reconsolidation) and MAPS why the next region-owning organ needs an engine feature.
  This is the honest scope the mission sanctioned ("if that's too deep for this lane, MAP it precisely and
  migrate only the reconcilable organs") — the reconcilable-WITHOUT-a-sim-edit set beyond rung 1 is EMPTY.
- **The comprehension boundary is on the GRADED margin, not the binary WTA.** The Norgan 6/6 "role WTA functional
  at dt=1.0" (coarse selection) still holds; this refines it: the production faculty (well/ill AUC) does not.
- Comprehension's `dt=1.0` AUC still SEPARATES on 3/6 seeds — the residual is a genuine per-seed operating-point
  degradation, not a total collapse; it simply fails the pre-registered 0.80 / 5-of-6 gate, which is the bar for
  "the production read is unchanged".
