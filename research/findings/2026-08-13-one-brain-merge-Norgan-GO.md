---
type: finding
status: live
date: 2026-08-13
mechanism: one-brain-merge
---

# One-brain MERGE SCALES: 3 organs on one pool + DIFFERENT-builder organs (GABA_B + NMDA) via a config SUPERSET — both GO (5/6)

**Date:** 2026-08-13 · **Runner:** `research/runners/_one_brain_merge_Norgan_derisk.py` · **Artifacts:**
`research/findings/raw/_one_brain_merge_Norgan_6seed.json`, `research/findings/raw/_one_brain_merge_diffbuilder_6seed.json`
(6 seeds 42/43/44/100/101/102, `SIM_BACKEND=numpy`). **NO `sim/` edit** — additive, reuse-by-import; the guarded
engine flag `cfg.per_region_threshold_heterogeneity` already exists on `main`. This is a **de-risk**, not "closed": it
takes the named next rung after the 2-organ INIT-invariance de-risk
(`2026-08-13-one-brain-merge-CLOSED-per-region-threshold.md`, which merged two INSTANCES of ONE builder) — 2→N organs
and organs built by DIFFERENT builders.

## (a) N ORGANS — 3 expectation circuits on ONE shared spiking pool + a cross-organ DAG

Three expectation-circuit organs (suffix `_A`/`_B`/`_C`) on ONE `SimulationBridge` (one `cp_` array, **N=3168**
neurons, one step, one `cfg.seed`, `per_region_threshold_heterogeneity=True`). Cross-organ synapses form an
upper-triangular DAG `surprise_i → cue_j` for every pair i<j (the LC-NE / hippocampal novelty motif: organ i's
surprise gates organ j's recall).

| criterion (6 seeds) | result | verdict |
|---|---|---|
| ONE shared spiking pool (3 organs in one `cp_` array, N=3168) | 6/6 | GO |
| determinism (`cfg.seed`; build-twice incl. thresholds byte-identical) | 6/6 | GO |
| INIT byte-identity — every per-neuron array of every organ, merged-vs-standalone (max err **0.0**) | 6/6 | GO |
| all 3 cross pairs LOAD-BEARING (lesion → interaction collapses) | 6/6 | GO |
| all 3 organs FUNCTIONAL (surprise contradict/confirm separation) | 5/6 | (organ variance) |
| **STRUCTURAL 3-ORGAN MERGE** | **5/6** | **GO** |

- INIT byte-identity is EXACT **0.0** for all three organs, every per-neuron array, all 6 seeds — each organ's
  per-neuron init is invariant to its two co-residents. `--legacy-global-thresh` reproduces the divergence (BOUNDARY).
- Every cross pair is independently load-bearing, every seed: driving organ i into surprise (contradict) lifts
  organ j's recall by **+20.1…+36.6 Hz** intact vs **≤ +0.6 Hz** after zeroing `surprise_i → cue_j` (attribution
  100% to the cross synapse; Hebbian frozen during the read so the lesion holds).
- The one non-functional seed (100) is organ `_B`'s absolute surprise firing dipping below the pre-registered 5 Hz
  "real signal" floor (its ratio is degenerate, confirm≈0) — the expectation faculty's own per-seed operating-point
  variance (the standalone `_spiking_expectation_rpe` GO is likewise a 5-of-6-style gate), NOT a merge artifact: the
  merge-specific criteria (pool, determinism, INIT byte-exact, cross load-bearing) are all 6/6.

## (b) DIFFERENT BUILDERS — expectation (GABA_B) + Wong-Wang role WTA (NMDA) via a config SUPERSET

ONE bridge (**N=2088**) = the SURPRISE expectation organ (`build_expectation_circuit`, GABA_B subtractive
prediction) + the Wong-Wang `SpikingRoleCompetition` role monitor (`_phaseB_multicue_competition_spiking_derisk`,
NMDA mutual-inhibition WTA). Both builders' actual `BrainRegion`/`RegionPathway` specs are reused-by-import; the
config is the SUPERSET. Cross synapse `surprise_S → sel_agent` (surprise biases role assignment toward AGENT).

| criterion (6 seeds) | result | verdict |
|---|---|---|
| ONE shared spiking pool (both DIFFERENT-builder organs in one `cp_` array, N=2088) | 6/6 | GO |
| determinism | 6/6 | GO |
| INIT byte-identity of the CO-RESIDENT expectation organ (invariant to the role organ; max err **0.0**) | 6/6 | GO |
| role WTA organ FUNCTIONAL (competition selects the driven role) | 6/6 | GO |
| cross synapse LOAD-BEARING (`surprise_S → sel_agent`) | 6/6 | GO |
| expectation organ FUNCTIONAL (surprise separation) | 5/6 | (organ variance) |
| **STRUCTURAL DIFFERENT-BUILDER MERGE** | **5/6** | **GO** |

- The engine gates NMDA per-region: the run log confirms `NMDA per-region mask: 2 regions enabled (48 neurons)` —
  only the role organ's two `sel_` accumulators carry NMDA; `enable_nmda=True` does NOT add NMDA to the expectation
  organ. `enable_gabab=True` is inert for the role organ (no GABA_B synapses, `conductance_max=0`). So **GABA_B and
  the NMDA accumulator COEXIST in one bridge** — the mission's named config-superset requirement, satisfied.
- Surprise biases the role WTA toward AGENT by **+181…+217 Hz** intact vs **+0.0 Hz** lesioned (100% attributable),
  all 6 seeds. Role WTA selects the driven role with **~+460 Hz** margins, all 6 seeds.
- The one non-functional seed (102) is again the expectation organ below the 5 Hz floor (ratio 34.9x, confirm≈0),
  not a superset failure — every merge-specific criterion is 6/6.

## The config-superset conflict map (what coexists vs what a single bridge forces)

| global field | expectation | role | class | outcome |
|---|---|---|---|---|
| `enable_gabab` | True | False | BENIGN-UNION | superset True; inert for role (no GABA_B synapses) |
| `enable_nmda` | False | True | BENIGN-UNION | superset True; per-region mask → only role `sel_` carry NMDA |
| `dt_ms` | 1.0 | 0.5 | GENUINE-CONFLICT (single global) | reconciled at **1.0**; both organs functional (role 6/6, expect 5/6) |
| `enable_homeostasis` | True | False | GENUINE-CONFLICT (single global) | reconciled **ON**; role WTA still selects 6/6 |
| `hebbian_learning_rate` | 0.06 | 0.02 | TRAIN-CONFLICT | mitigated: role edges installed/gated, not co-trained here |
| `hebbian_max_weight` | 45 | 60 | TRAIN benign | superset = max = 60; both working ranges below it |

**Key result:** the two GENUINE single-valued conflicts (`dt_ms`, `enable_homeostasis`) turned out to be
RECONCILABLE at `dt=1.0`/`homeostasis=ON` — BOTH different-builder organs remain functional at that operating point
(role WTA 6/6, expectation 5/6, the 6th being intrinsic organ variance). **So no engine change is required for THIS
pair to co-reside and function.** The mapped next `sim/` step, should a FUTURE builder pair NOT reconcile at a shared
value, is **per-organ scoping of `dt_ms` and `enable_homeostasis`** (both are currently one global value per bridge) —
identified and quantified, not needed here.

## Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners._one_brain_merge_Norgan_derisk --mode norgan --n-organs 3 \
    --seeds 42,43,44,100,101,102 --out research/findings/raw/_one_brain_merge_Norgan_6seed.json
SIM_BACKEND=numpy python -m research.runners._one_brain_merge_Norgan_derisk --mode diffbuilder \
    --seeds 42,43,44,100,101,102 --out research/findings/raw/_one_brain_merge_diffbuilder_6seed.json
# INIT-invariance BOUNDARY reproduction (legacy single global threshold stream):
SIM_BACKEND=numpy python -m research.runners._one_brain_merge_Norgan_derisk --mode norgan --seed 42 --legacy-global-thresh
```

## Honest scope / non-claims

- **GO = a STRUCTURAL de-risk** (one shared pool + determinism + INIT byte-identity + load-bearing cross synapses +
  organs functional), 6-seed, pre-registered `≥5/6` gate. **NOT claimed:** exact byte-identity of the fully
  homeostatically-adapted PRODUCTION trained read — the 2-organ finding mapped that residual to the homeostatic
  companion process; the same bound applies and is not re-litigated here.
- **NOT "closed"** and **NOT integrated into production** — the production organs remain CO-RESIDENCY; this de-risks
  that a single shared spiking pool SCALES to N organs (3 shown) and to organs built by DIFFERENT builders
  (GABA_B expectation + NMDA Wong-Wang WTA).
- The role WTA functional test drives one role's cue and checks that role wins the mutual-inhibition competition
  (uses the init cue→role weight, no role training) — it tests the WTA decision on the merged substrate, not the
  full learned-validity parser. Functional read-outs only; no phenomenal claim.
