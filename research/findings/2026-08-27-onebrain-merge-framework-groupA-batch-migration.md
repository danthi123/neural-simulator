---
status: live
type: finding
lane: integration
date: 2026-08-27
---

# One-brain merge FRAMEWORK — GROUP A batch migration (declarative registry + one batched verify)

Status: GO for the SUBSTRATE-INIT co-residence-invariance MIGRATION gate (7 organs registered, 6/6 seeds).
This is the O(N)->O(1) payoff the framework promised: ~13 candidate Group-A organs collapsed into a declarative
`OrganDescriptor` REGISTRY + ONE parameterized `verify(keys, seeds)`, run as a SINGLE 6-seed sweep instead of a
bespoke ~200-line class + a 6-seed verify per organ. Artifact:
`research/findings/raw/_onebrain_merge_groupA_6seed.json`.

Builds on the DESIGN + prototype skeleton (`2026-08-27-onebrain-merge-framework-DESIGN.md`,
`research/runners/onebrain_merge_framework.py`) and the twopool proof
(`2026-08-27-onebrain-twopool-merge-substrate-byte-identity-6seed-GO.md`, 6/6 GO for 4 organs). It packages that
proven form as a registry and scales it.

## What "migrated byte-identically" means here (the honest boundary — read FIRST)

The gate is **SUBSTRATE-INIT CO-RESIDENCE-INVARIANCE**: each organ's region-slice per-neuron init arrays
(firing thresholds, membrane v, recovery u, the 8 Izhikevich params, the 2 gate masks) are byte-IDENTICAL when
the organ is built ALONE on the shared superset config (`coresident`) vs when it is built co-resident with the
other six (`merged`). It proves adding the other organs does NOT perturb an organ's substrate. This is exactly
the claim the twopool derisk gated for 4 organs; here it is the batched, registry-driven form for 7.

Three things it is NOT, stated so the record cannot over-read it:

- It is **NOT organ-read / answer equivalence.** NONE of the 13 Group-A shipped classes takes a `shared=`
  substrate today (confirmed by reading all 13; the word "shared" elsewhere is "process-shared singleton" or an
  architectural phrase). So the shipped read pipelines were NOT run on the pool. Adding a `shared=` kwarg per
  organ + a MergedPool read surface is the named NEXT rung (the twopool ORGAN-READ verify is its 4-organ proof).
- It is **NOT integration.** Byte-identity-in-isolation deliberately FORBIDS the cross-region synapses that ARE
  the one-brain goal (DESIGN §4). A pool with zero cross-edges is MIGRATED, not INTEGRATED. The F-gate is next.
- It is **NOT byte-identity to each organ's PRE-migration standalone.** Most Group-A builders set
  `per_region_threshold_heterogeneity=False`; the shared pool needs it TRUE (the name-keyed seam is what makes
  co-residence safe). So an organ's slice on the pool differs from its own historical standalone by exactly that
  seam — which IS the migration (the organ moves onto the seam config), the same move pool-1 already made.

## The engine, as built

`research/runners/onebrain_merge_framework.py`:
- `OrganDescriptor` (`:41`) — declarative record: `key`, `spec_fn` (seed -> regions/pathways/meta, reused BY
  IMPORT from the organ's own de-risk builder), `config` (unioned; a clash raises `MergeConflict`), `param_het`
  (organ uses parameter-heterogeneity -> reconcile via the name-keyed per-region seam), plus the deferred
  `post_build`/`freeze_regions`/`isolation`/`idx_fn`/`read_fn`/`organ_cls`/`supports_shared` fields for the
  organ-read + integration rungs.
- `merge_organs(descriptors, seed, config_descriptors=None, legacy=False, force_het_off=False)` (`:255`) — the
  N-organ generalization of the bespoke `MergedSubstrate.ensure_built` + twopool `build_pool`: spec-union ->
  name-disjointness (raises, naming the colliding organs) -> config-union (raises `MergeConflict` on a real
  clash) -> per-region param-het seam (GLOBAL off keyed on `config_descriptors`, per-region mask keyed on the
  instantiated descriptors) -> one bridge -> generic gain-0 freeze -> snapshot. `MergedPool` exposes the
  `.bridge/.cfg/.xp/.ensure_built()/.read_isolation(key)/.idx(key)` surface the shipped organs' `shared=` path
  expects (so `desc.organ_cls(seed, shared=pool)` will run UNMODIFIED once the organs carry a `shared=` kwarg).
- `substrate_byte_identity(merged, coresident, regions)` (`:308`) — the migration gate, promoted from the
  twopool `byte_identity`.
- `config_descriptors` vs `descriptors` split is load-bearing: the CORESIDENT baseline unions the FULL registry's
  config but instantiates only one organ's regions, so a non-zero slice delta isolates CO-RESIDENCE from config.

`research/runners/onebrain_merge_verify.py` — the batched, registry-driven `verify(keys, seeds)`. Per organ,
per seed: (1) substrate-init byte-identity merged-vs-coresident; (2) the LEGACY discriminator (name-keyed seams
OFF must DIVERGE); (3) for a param-het organ, a HET-LOAD-BEARING control (the per-region mask cleared must move
the izh params -> the reconciliation is not a vacuous all-zero het). Verdict wired via `tools.verdict.Verdict`;
`--keys all` == the Group-A batch.

## Result — 7/7 Group-A organs, 6/6 seeds (42,43,44,100,101,102)

| organ | regions | param-het | substrate byte-identical | legacy diverges | het load-bearing (izh Δ range) |
|---|---|---|---|---|---|
| causal_whatif | 1 (`evt`) | no | 6/6 (Δ=0) | 6/6 | n/a |
| comprehension | 12 (role/cue pools) | no | 6/6 (Δ=0) | 6/6 | n/a |
| self_schema | 3 (workspace/…/self_schema) | yes | 6/6 (Δ=0) | 6/6 | 6/6 (Δ 74–90) |
| source_provenance | 8 (episode/prov/…) | yes | 6/6 (Δ=0) | 6/6 | 6/6 (Δ 117–137) |
| curiosity | 5 (cue/striosome/snc/ask) | yes | 6/6 (Δ=0) | 6/6 | 6/6 (Δ 85–140) |
| prospective_memory | 4 (cortex_ctx/dlpfc_wm/rel_*) | no | 6/6 (Δ=0) | 6/6 | n/a |
| d6_multiref_wm | 31 (w0..w29 + fs) | no | 6/6 (Δ=0) | 6/6 | n/a |

Merged pool N≈4968 neurons (numpy CPU, bit-exact). **Every** organ's slice is byte-identical merged-vs-coresident
(substrate Δ=0 on every seed) AND **every** organ diverges under the seams-off discriminator (per-organ legacy Δ
in the 90–143 range across seeds) — the byte-identity is non-vacuous for each organ individually, not merely for
the pool. The three param-het organs carry genuine, load-bearing per-region heterogeneity (izh Δ 74–140 vs the
mask-cleared control) that is itself byte-identical merged-vs-coresident — the exact twopool reconciliation, now
declarative. Verdict via `tools.verdict.Verdict`: substrate byte-identity 42/42, legacy diverges 6/6, param-het
load-bearing 18/18, per-organ GO 42/42 -> GO.

## Deferred to Group B/C (honest, with the seam each needs)

Six of the 13 candidates do NOT register as a single-pool substrate slice; each is a real finding about an engine
seam, not a failure hidden:

- **b3_noncontradiction** — STATELESS; owns no substrate. Rides the live composer's spiking polarity recall via a
  `recall` callable. Nothing to co-locate.
- **reconsolidation** — owns no circuit; reuses the D2 SURPRISE organ's slice + rewrites the composer store. Its
  substrate migrates WHEN surprise does.
- **repair** — no class; functions composing the D4 COMPREHENSION organ. Its substrate == comprehension's.
- **d3_discourse_event_register** — MULTI-bridge: FOUR FS-WTA discretizer bridges + a host rate-RNN. Not one
  shared-pool slice; needs a multi-bridge seam.
- **d5_episodic** — heavy OWN-pool: ~2000-neuron CA3 with two-compartment apical dendritic-dAP + slow-NMDA
  reverberation + BTSP formation. Group-C own-pool + apical/NMDA-slow seam.
- **affective_tom** — OU + NEUROMODULATOR-subsystem seam: `enable_ou_process=True` + a bespoke `appraisal`
  neuromodulator triad drives the read. Group-B OU/neuromod seam.

So: 13 candidates -> **7 migrated (substrate-init byte-identity, 6/6)**, 3 ride an already-registered organ, 3
need a documented engine seam. Note the design's original Group-A list assumed several of these were
declarative-NOW; reading the shipped code corrected that (param-het/OU/neuromod/multi-bridge), which is itself
the value of running the batch.

## Next rungs (NOT done here — this is the migration de-risk, not the production flip)

1. ORGAN-READ byte-identity: add a `shared=` kwarg to each Group-A class (minimal, byte-identical-when-None) +
   the MergedPool read surface, then run each shipped read/answer merged-vs-coresident (the twopool ORGAN-READ
   verify is the pattern).
2. Fold the 4 already-merged pool organs (surprise/world-model/metacog/pragmatic) into the SAME registry,
   retiring `MergedSubstrate`/`MergedSubstrate2` to a thin `merge_organs([...])` call.
3. Group-B seams (STP / OU+neuromod / apical-NMDA-slow / multi-bridge), each unlocking its organ(s).
4. INTEGRATION phase: cross-region edges under the F-gate (faculty-still-works + interaction-is-real-and-lesionable
   + no-runaway + moat/honesty). This is the actual one-brain goal; byte-identity is the SAFETY gate beneath it.
5. The production flip (retire MergedSubstrate*, change server defaults) is explicitly NOT done here.

Scaffold residuals carried per descriptor for the self-organization burn-down (DESIGN §6): the hand-declared GNW
assembly loops (self_schema), the host-injected DA sign at train time (causal_whatif), and — pool-wide — the
name-keyed seams + gain-0 freeze themselves, which a faithful end state would let the substrate develop and damp
intrinsically rather than hand-set.
