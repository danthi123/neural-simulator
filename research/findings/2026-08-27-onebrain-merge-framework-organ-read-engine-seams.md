---
status: live
type: finding
lane: integration
date: 2026-08-27
---

# One-brain merge framework — ORGAN-READ byte-identity EXTENDED to 5/7 (source_provenance + causal_whatif via engine seams)

Status: GO. This extends the framework's ORGAN-READ co-residence-invariance gate from 3 organs
(`2026-08-27-onebrain-merge-framework-organ-read-extension.md`: self_schema + d6_multiref_wm + comprehension) to
FIVE — closing the two organs whose read needed an ENGINE SEAM rather than a pure frozen forward pass:
`source_provenance` (a LEARNED, context-gated opponent trace — its read ENCODES then recalls) and `causal_whatif`
(a DIRECTED spiking forward model — its read TRAINS temporal-order STDP + phasic-DA then reads). Both now RUN
their real read pipeline on the ONE wired pool byte-identically, 6 seeds. The remaining two (curiosity,
prospective_memory) stay substrate-init GO with their reads honestly deferred (precise blockers below). NO `sim/`
edit. All runs are tiny numpy nets (N=4968 merged) on the CPU.

Artifact (carries a `tools.verdict.Verdict` `preconditions` block):
`research/findings/raw/_onebrain_merge_organread_engineseams_6seed.json` (the full `--keys all` 6-seed sweep).

## What now passes (6 seeds = 42,43,44,100,101,102)

- SUBSTRATE-INIT byte-identity: 7/7 Group-A organs, 42/42 organ-seeds (no regression).
- ORGAN-READ byte-identity: `self_schema` + `d6_multiref_wm` + `comprehension` + **`source_provenance`** +
  **`causal_whatif`**, all `read_maxerr == 0.0` AND `answer_same == True` on every seed (30/30 read-organ-seeds),
  co-resident with ALL 7 Group-A organs on ONE `SimulationBridge` (N=4968) — not the read-organs alone, and with
  the read organs run in-order on the SHARED bridge (each organ's read after the others').
- Legacy discriminator diverges 6/6; param-het reconciliation load-bearing on every param-het organ; every organ's
  per-seed GO gate 42/42; POOL ALL-GO True.
- Both closes are NON-DEGENERATE at the pool operating point (dt=1.0, homeostasis OFF), verified merged AND
  coresident:
  - `source_provenance`: the perceived-vs-generated opponent sign is 100% correct (acc 1.0) with min normalized
    discriminability d ≈ 0.61–0.75 across seeds — a real reality-monitoring separation, not a vacuous all-zero read.
  - `causal_whatif`: fwd-prediction acc 1.0; the unseen 2-step consequence D fires at ≈98–100 Hz vs off-chain 0 Hz
    (predicts_D); the DO-intervention separates cause from correlation cleanly (Y|do(C) ≈ 167 Hz vs Y|do(X) 0 Hz);
    the chain edge grows w_AB 0.2→≈10 while the spurious X→Y is pruned to ≈0.5 and the direct A→D stays unlearned
    (≈0.2). These match the STANDALONE de-risk byte-for-byte — the pool build-time train reproduces it exactly.
- Byte-identical-when-`shared=None`: `ProvenanceBrain(seed, shared=None)` is a purely ADDITIVE change (the entire
  standalone `__init__`/encode/recall path is unchanged; the `shared=` branch returns early). `causal_whatif` adds
  NO edit to its de-risk at all — its read organ + explicit_wiring_fn live entirely in the framework file.

## How the two engine-seam organs closed

Both are BUILD-TIME-PLASTICITY-then-FROZEN-READ, a new shape beyond the three frozen-forward-pass organs. The
shared seam pattern: the pool config keeps every plasticity flag OFF (unions with the frozen organs); the read
organ flips the needed flags TRUE for a BUILD-TIME step ONLY (the flags are read live per step), runs a UNIVERSAL
gain-0 freeze of every NON-organ edge (only the organ's own plastic edges can move → the mutation is confined to
its slice + co-residence-invariant), then FREEZES (flags OFF) and reads. The whole step runs in the pool's
`read_isolation` so co-resident slices restore, and the read organ RESTORES every config scalar / timing counter /
gain array it touched so no other organ's read on the shared bridge is perturbed.

- `source_provenance` — `ProvenanceBrain` gained `shared=None`: when a pool is injected it adopts `pool.bridge`,
  discovers its episode/ctx_*/prov_*/inh_* slice from `region_indices_dict`, sets the gate defaults + zeros the
  prov traces. The read organ (`_SourceProvReadOrgan` in the framework) runs the 8-item Hebbian encode at build
  time under `enable_hebbian_learning=True` + the gain-0 freeze, then the recall read is a clean FROZEN forward
  pass. Its `enable_nmda=False` reconciles via the per-region mask (its regions opt OUT); `param_het` via the
  name-keyed per-region seam. THE RESET-BASELINE FIX (the load-bearing subtlety): the read must reset to the pool's
  PRISTINE settle-to-rest snapshot (`pool.snap`), NOT the bridge as-is — a co-resident organ's earlier read leaves
  residuals that, captured as "rest", made the encode ORDER-dependent (failed 2/6 seeds until fixed).
- `causal_whatif` — the evt region carries NO RegionPathways (its xblock edges are injected separately), so the
  organ supplies an `explicit_wiring_fn` regenerating the cross-block edges per-region-seamed (same
  `RandomState(seed+17)` + loop order as `build_forward_model`). The read organ (`_CausalReadOrgan`) runs the
  temporal-order-STDP + phasic-DA train at build time under `enable_stdp`+`enable_reward_modulation`=True + the
  gain-0 freeze, then the frozen substrate reads (forward prediction / unseen-consequence rollout / DO-intervention)
  — all `cp_firing_states` reads. THE ALLOCATION FIX (the load-bearing subtlety): the plasticity STATE arrays
  (`cp_eligibility_trace`, `cp_last_spike_time`) are allocated at BUILD only when the flags are on; the frozen pool
  leaves them None, so toggling the flags at train-time SILENTLY no-ops (the DA three-factor rule needs the
  eligibility trace) — the organ allocates them for the train and restores them to None after.

## The framework NMDA-mask reconciliation seam (a new co-residence seam)

The engine builds a per-neuron NMDA mask the moment ANY region opts in (`BrainRegion.enable_nmda=True`), after
which regular NMDA applies ONLY to masked neurons; with global `enable_nmda=True` but NO region opting in it falls
back to GLOBAL NMDA (v1 back-compat). That fallback is co-residence-DEPENDENT: an organ whose regions all opt OUT
(source_provenance, causal_whatif) gets NO NMDA when co-resident with an NMDA organ (masked out) but SPURIOUS
global NMDA when ALONE on the enable_nmda=True superset config. `MergedPool.ensure_built` now PINS it — global-on +
no-opt-in installs an ALL-ZERO mask (no neuron gets NMDA) — so a no-NMDA organ's slice reads byte-identically alone
vs co-resident. Organs that WANT global NMDA opt every region in via `region_flags` (d6), so a mask is already
built and this never fires; and any pool that already has an opting-in region (self_schema present) skips it. Init
arrays are untouched, so substrate-init byte-identity is unchanged.

## Honest boundary — unchanged: still the MIGRATION gate, not INTEGRATION

Byte-identity-in-ISOLATION forbids the cross-region interaction that IS the one-brain goal: a pool with zero
cross-synapses is MIGRATED, not INTEGRATED. The co-resident-alone baseline is the pool-built-alone organ on the
superset config, NOT the organ's pre-migration standalone — the claim is co-residence-invariance of the read at the
shared operating point (dt=1.0, homeostasis OFF, noise OFF), not identity to the un-merged organ. For
`causal_whatif` there is a SECOND declared residual: the production what_if/why NL ANSWER is rendered by a live
`RFPhasorComposer`; the organ-read gate closes the SUBSTRATE causal VERDICT (predicts_D / directed-ratio /
cause-separation / DO-intervention — composer-INDEPENDENT), and the composer-grounded NL rendering rides the
composer-grounding burn-down (it does NOT block the migration gate).

## The remaining two organ-read deferrals — precise blockers (NOT a substrate limit)

- `curiosity` — the read is a FROZEN forward pass, but its ONLY read-load-bearing dependency is the `curiosity`
  NEUROMODULATOR SUBSYSTEM (`enable_neuromodulator_subsystem` + a from_novelty rule driving `excitability_drive` on
  `group:ask` — the ASK pool has NO afferent pathway, firing only via the modulator), PLUS `enable_ou_process`
  (default ON, index-order RNG → co-residence-DEPENDENT). Seam: enable+register the curiosity modulator on the pool
  (a global-subsystem config seam) + force OU off (a per-neuron-seeded OU stream is a `sim/` edit). The neuromod
  subsystem's internals live in `sim/` — deferred for owner review.
- `prospective_memory` — the ONE genuinely-remaining organ. The FORMATION-mutation half is now a SOLVED pattern
  (source_provenance's Hebbian encode + causal_whatif's STDP+DA train both close via the toggle + gain-0-freeze +
  plasticity-array-allocation seam). The TWO residual blockers: (1) DEPTH — the read runs on a 3-class hierarchy
  (base → HomeostaticProspectiveMemory → SFANmdaProspectiveMemory) that builds its own bridge AND runs a
  multi-stage homeostat+plateau CALIBRATION in `__init__`; a `shared=` injection must thread the bridge + index
  maps through all three and re-home the calibration on the pool slice. (2) MULTI-TURN HOLD — fire_on_cue is not a
  single drive→settle→read: the self-sustaining cortex↔dlpfc attractor + per-neuron SFA trace must persist UNRESET
  across an unbounded number of INDEPENDENT production turns, which is at odds with read_isolation's per-call
  snapshot/restore (closable by spanning ONE guard across a read's turns, but the calibrate-then-multi-turn-read
  structure over the deep hierarchy is the intricate part). Deferred for SCOPE, not a substrate limit.

## Files changed

- `research/runners/onebrain_merge_framework.py` — the NMDA-mask reconciliation seam in `MergedPool.ensure_built`;
  source_provenance read plumbing (`_SOURCE_PROV_CONFIG`, `_source_prov_organ`/`_SourceProvReadOrgan`/
  `_source_prov_reads`/`_source_prov_answer`); causal_whatif read plumbing (`_CAUSAL_CONFIG`, `_causal_wiring`,
  `_causal_organ`/`_CausalReadOrgan`/`_causal_reads`/`_causal_answer`); both descriptors gain
  config/organ_cls/read_fn/answer_fn/explicit_wiring_fn/supports_shared=True; `GROUP_A_ORGANREAD_DEFERRED` reduced
  to the 2 genuinely-remaining organs (source_provenance + causal_whatif removed; pmem note refined).
- `research/runners/_laneC_source_provenance_opponent_derisk.py` — additive `shared=None` on `ProvenanceBrain`
  (`_attach_shared`: adopt the pool bridge, discover the slice, reset-baseline from `pool.snap`; byte-identical when
  None).

NO `sim/` edit. All runs are tiny numpy nets (N<=4968) on the CPU.
