---
type: finding
status: live
date: 2026-08-13
mechanism: one-brain-merge
---

# One-brain MERGE de-risk: TWO organ bridges (surprise + recall) on ONE shared spiking substrate — structural GO, exact byte-identity gated on per-organ heterogeneity seeding (BOUNDARY)

**Date:** 2026-08-13 · **Runner:** `research/runners/_one_brain_merge_2organ_derisk.py` · **Artifact:**
`research/findings/raw/_one_brain_merge_2organ_6seed.json` (6 seeds 42/43/44/100/101/102, `SIM_BACKEND=numpy`).
**NO `sim/` edit; reuse-by-import.** **NO-EXTERNAL-NEEDED:** this is not a capability wall — the merge WORKS and
the one byte-identity residual has a concrete in-engine fix named below (a per-region RNG stream for the
threshold draw, the same shape as the engine's existing `per_parameter_heterogeneity_seed`), so no external
literature is load-bearing here.

## What this de-risks (roadmap §0.2 residual #1 — "one brain is CO-RESIDENCY, not one substrate")

The production brain is CO-RESIDENCY: each conversational organ (recall / surprise / comprehension / affect /
episodic / ...) is a SEPARATE spiking `SimulationBridge`. The GNW bus combines their READS via ignition, but the
organs do not share ONE neuron pool with cross-region synapses. Genuine cross-synaptic interaction was proven for
exactly one pathway (acquisition) and, WITHIN one faculty, for the WKV cortex's two internal bridges
(`2026-07-20-wkv-cortex-physically-merged-onto-one-bridge-GO`, byte-exact). This de-risk takes the next rung:
MERGE TWO DISTINCT ORGANS onto ONE `SimulationBridge` — one shared `cp_` neuron array + a genuine CROSS-ORGAN
synapse — and measure byte-identity, load-bearing, and determinism.

## The merge (the mission's named "recall + surprise" pair)

Both organs reuse the adversarially-verified D2 expectation-circuit primitives
(`_spiking_expectation_rpe_derisk`, 6/6 GO, lesion-decisive):

- **Organ A = SURPRISE** (expectation-violation): `cueA -(Hebbian topographic)-> patient_expected_A` (FS/PV-like,
  the recalled prediction, GABA_A subtractive) ; `patient_asserted_A -(exc)-> surprise_A`. Read = `surprise_A`
  windowed firing (CONFIRM cancels ~0 Hz; CONTRADICT/NOVEL fires).
- **Organ B = RECALL** (heteroassociative memory): `cueB -(Hebbian topographic)-> patient_expected_B`. Read =
  `patient_expected_B` firing (how strongly a cue recalls its stored patient).
- **CROSS-ORGAN synapse** (novelty/surprise gates recall — the LC-NE / hippocampal-novelty motif):
  `surprise_A -(exc)-> cueB`. When organ A is SURPRISED, its firing adds drive to organ B's recall cue -> organ B
  recalls MORE. Lesion the `surprise_A->cueB` edges -> organ B stops responding to A's surprise.

**Why this is a genuine MERGE, not co-location.** Both organs' regions are allocated in ONE `SimulationBridge`
(N = 2112 neurons: 1056 organ A + 1056 organ B) — asserted in code that ONE `cp_membrane_potential_v` array
holds BOTH organs, stepped by ONE `_run_one_simulation_step`, one `cfg.seed`. The cross-organ synapse is a real
edge in the ONE `cp_connections` matrix. The co-resident baseline is two separate bridges, two neuron arrays, no
shared matrix.

## Results (6 seeds 42/43/44/100/101/102; per-seed detail in the JSON)

| criterion | result | verdict |
|---|---|---|
| ONE shared spiking neuron pool (both organs in one `cp_` array) | 6/6 | **GO** |
| determinism (`cfg.seed`; build-twice byte-identical) | 6/6 | **GO** |
| CROSS-ORGAN synapse LOAD-BEARING (intact vs lesion interaction) | 6/6 | **GO** |
| byte-identical under HOMOGENEOUS threshold (cause-isolation control) | 6/6 | **GO** |
| EXACT byte-identical under PRODUCTION heterogeneity | 0/6 | **BOUNDARY** |
| **STRUCTURAL MERGE (pool + determinism + load-bearing + byte-clean-under-shared-heterogeneity)** | **6/6** | **GO** |

- **Cross-organ synapse is DECISIVELY LOAD-BEARING.** Organ B's recall rises when organ A is CONTRADICT
  (surprised) vs CONFIRM: interaction **+16.05 to +33.90 Hz intact** vs **−0.66 to −0.02 Hz** after zeroing the
  `surprise_A->cueB` edges — a **≥42x collapse** to ~0 (min over seeds; per-seed 42x/65x/379x/871x/586x/902x).
  This is genuine cross-region synaptic interaction between two organs on ONE substrate (not two co-located pools).
- **The merged organs stay FUNCTIONAL** (this is not a dead-organ byte-identity): organ A's surprise separation is
  **5.5x–51.8x** contradict/confirm on the merged bridge (the D2 faculty is intact after the merge), and organ B
  recalls at ~9–13 Hz.
<!--derived-->
- **Byte-identity — the mapped residual.** Merged-vs-co-resident read delta under the PRODUCTION (heterogeneous)
  config (`surprise_maxerr_hz`/`recall_maxerr_hz`/`surprise_merged`/`surprise_solo` per seed in the cited JSON):
  organ A surprise per-condition MEANS are byte-clean (the seed-42 `surprise_merged` vs `surprise_solo` triples
  agree to under 0.03 Hz), with a per-fact max |err| up to **2.66 Hz** on a single near-threshold fact; organ B
  recall differs by up to **2.02 Hz** (~15–20% of the ~9–13 Hz recall). This is NOT exact byte-identity under
  production heterogeneity.

## Root cause — isolated to ONE engine line

Every candidate was ruled out by direct measurement (`_one_brain_merge_2organ_derisk` diagnostics):

- **Per-neuron parameters** (izh a/b/k/vr/vt/c/d/vpeak/C), **initial state** (v, u), and the **sorted learned-weight
  distribution** of every organ-B pathway are **byte-identical** merged-vs-standalone.
- **Positions** differ (RNG-drawn over the larger pool) but do NOT feed the Izhikevich step — overwriting merged
  positions to the standalone values changes the read by 0.0.
- **The cross synapse does NOT leak** into an isolated read: cross_weight=0 and cross_weight=12 give identical
  organ-B recall when organ A is undriven.
- **The SOLE differing per-neuron array is `cp_neuron_firing_thresholds`** (`sim/bridge.py:2307`,
  `cp.random.uniform(...)` over ALL n neurons). It is drawn from the GLOBAL RNG stream, so organ B (the SECOND
  organ in the merged pool) lands at a SHIFTED stream position and receives a DIFFERENT — though equally valid —
  seeded heterogeneity than it would standalone. That (a) directly shifts organ B's firing and (b) during
  on-substrate Hebbian training changes the synaptic PLACEMENT of the same learned-weight distribution -> a
  divergent recall.

**Decisive cause-isolation control:** setting every threshold to one constant (removing the per-neuron RNG
heterogeneity), applied IDENTICALLY to merged and co-resident bridges, drives byte-identity to **EXACT 0.0** on
BOTH organs (6/6). So the threshold RNG stream is the SOLE byte-identity breaker; the merge mechanism
itself is byte-clean. (The homogeneous threshold also disables the surprise separation — the heterogeneity is
functionally required — so it is a cause-isolating control, not the production fix.)

## The named next mechanism (the deepest one-substrate step this scopes)

A mergeable substrate must initialize each organ's per-neuron heterogeneity from a **PER-REGION / PER-ORGAN seed
stream**, not the single global stream indexed by pool position, so an organ's seeded init is invariant to its
co-residents. The change is small and localized: draw `cp_neuron_firing_thresholds` (bridge.py:2307) per brain
region from a region-scoped RNG, exactly as the engine's existing `per_parameter_heterogeneity_seed` /
`_HET_PER_PARAM_SEED_STRIDE` machinery (bridge.py:3326-3386) already does for Izhikevich parameter heterogeneity.
With that in place, organ B's threshold slice equals its standalone pattern regardless of what shares the pool,
and — given every other per-neuron array and the weight distribution already match — the merge becomes exactly
byte-identical while preserving the production heterogeneity. This is an owner-scoped `sim/` edit (per-region RNG
for the homeostasis-threshold draw) and is the concrete first item for the deepest one-substrate step.

## Honest scope / non-claims

- **Two INSTANCES of the same builder.** Organ A and organ B both use the expectation-circuit primitives over
  different content/reads (surprise vs recall) — the mission's "recall + surprise" pair. Merging two organs built
  by DIFFERENT builders (e.g. surprise + the Wong-Wang `SpikingRoleCompetition` comprehension monitor) requires a
  config SUPERSET (GABA_B + NMDA-accumulator coexistence) and is the named larger next step; the config-superset
  merge is not attempted here.
- **Functional read-outs only.** The surprise "notice" and the recall read are honest functional signals; no
  phenomenal claim is made.
- This is a **de-risk**, not a closure. "One substrate" for the production organ set remains CO-RESIDENCY; this
  proves the FIRST genuine 2-organ merge (shared pool + load-bearing cross synapse) and scopes the exact residual.

## Anti-cheats (all hold)

- **Genuinely one pool:** asserted `len(cp_membrane_potential_v) >= n_A + n_B` and both organs' region indices
  fall in the one array (6/6).
- **Cross synapse load-bearing:** lesion (`surprise_A->cueB` zeroed) collapses the interaction to ~0 (6/6).
- **Byte-identity measured, not assumed:** merged-vs-co-resident, with the OTHER organ undriven so the cross is
  inert; the residual is quantified and root-caused, not hidden.
- **Determinism:** `cfg.seed` set on every build; build-twice byte-identical (`cp_membrane_potential_v` +
  `cp_connections.data` hashes, 6/6). No unseeded-substrate confound.
- **BRAIN-BASED:** `current_reward_signal == 0`; all reads are `cp_firing_states` reads; the cross-organ
  interaction is a real synapse, not a host coupling.
