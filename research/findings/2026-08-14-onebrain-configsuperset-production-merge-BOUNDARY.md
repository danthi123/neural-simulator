---
type: finding
status: live
lane: gap#5
date: 2026-08-14
mechanism: one-brain-merge
---

# One-brain config-SUPERSET production merge (surprise GABA_B + Wong-Wang comprehension NMDA): BOUNDARY on `enable_homeostasis`

**Date:** 2026-08-14 · **Verdict:** BOUNDARY (no reconcilable operating point over the (dt, homeostasis) 2×2) ·
**Status:** honest-negative with a mapped `sim/` engine feature · **Backend:** numpy (bit-exact CPU) · 6 seeds
(42,43,44,100,101,102) × 4 cells.

**Runner:** `research/runners/_one_brain_merge_configsuperset_production_derisk.py` ·
**Artifact:** `research/findings/raw/_one_brain_merge_configsuperset_6seed.json` ·
**Reproduce:**
```
SIM_BACKEND=numpy OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -m \
  research.runners._one_brain_merge_configsuperset_production_derisk \
  --seeds 42,43,44,100,101,102 --cells 0.5:True,0.5:False,1.0:True,1.0:False \
  --out research/findings/raw/_one_brain_merge_configsuperset_6seed.json
```

## What this is

The named "larger next step" of `2026-08-13-one-brain-merge-2organ-BOUNDARY.md`: merge two DIFFERENT-builder
PRODUCTION organs onto ONE spiking `SimulationBridge` (one `cp_membrane_potential_v`, one step, one `cfg.seed`)
via a config SUPERSET — the D2 SURPRISE organ (`build_expectation_circuit`; GABA_B/GIRK subtractive prediction)
and the D4 Wong-Wang COMPREHENSION role monitor (`SpikingRoleCompetition`; NMDA-slow recurrent WTA read by
`ComprehensionProductionOrgan`) — at the PRODUCTION answer-preservation bar (the bar pool #1/#2 flips met). This
is a de-risk (reads driven on the merged bridge Norgan-style); the production `shared=` wiring + DEFAULT flip is
the gated follow-on. NO `sim/` edit; two additive default-preserving edits only (below).

The merged pool is 2088 neurons: surprise organ = 1056 (`cue_S`/`patient_expected_S`/`patient_asserted_S`/
`surprise_S`), role organ = 1032 (`sel_agent`/`sel_patient`/`sel_FS_*` + 8 `cue_{cue}_{sgn}` pops). The config
superset carries `enable_gabab=True` AND `enable_nmda=True`; the per-region NMDA mask restricts NMDA to exactly
the 48 `sel_agent`+`sel_patient` neurons (verified 6/6, `n_nmda==n_sel==48`), GABA_B (`gabab_conductance_max=0`)
is inert for the role organ. GABA_B and NMDA coexist as independent per-neuron conductances that interact only
through the shared membrane voltage.

## Verdict: `go_by_cell` = 0/6 in every cell → BOUNDARY

| cell (dt, homeo) | GO/6 | comp byte-id | comp AUC≥0.80 | comp ans-preserved | surp byte-id | surp ans-preserved | cross LB |
|---|---|---|---|---|---|---|---|
| 0.5, **True**  | 0 | 6 | 2 (AUC 0.759–0.843, mean 0.801) <!--derived--> | 0 | 4 | 2 | 5 |
| 0.5, **False** | 0 | 6 | 6 (AUC 1.000) | 6 | 6 | 0 (surprise SILENT) | 0 |
| 1.0, **True**  | 0 | 5 | 0 (AUC 0.463–0.602, mean 0.536) <!--derived--> | 0 | 4 | 2 | 5 |
| 1.0, **False** | 0 | 6 | 6 (AUC 1.000) | 6 | 6 | 0 (surprise SILENT) | 0 |

Merge-specific infrastructure axes are 6/6 in EVERY cell: one shared pool, determinism (build-twice byte-id on
`cp_membrane_potential_v`+`cp_neuron_firing_thresholds`), GABA_B+NMDA coexistence (per-region NMDA mask = 48),
surprise-slice INIT byte-id vs a standalone `build_expectation_circuit` (`surp_init_err = 0.0`, the per-region
threshold fix holds under the superset), and read-isolation (comp read leaves the surprise slice bit-for-bit
unchanged; the surprise-read guard restores the role slice bit-for-bit). The merge itself is clean.

## The single binding conflict is `enable_homeostasis` (not `dt_ms`)

The 2×2 isolates the conflict cleanly, and it is DIAGONAL on the global homeostasis flag:

- **Comprehension needs homeostasis OFF.** Its Wong-Wang graded well/ill AUC is a perfect 1.000 at BOTH dt=0.5
  and dt=1.0 when homeostasis is OFF (6/6, answer byte-identical to the shipped native read). Turning homeostasis
  ON degrades the graded sel-pool margin: AUC drops to mean 0.801 at dt=0.5 (clears 0.80 on only 2/6 seeds) and <!--derived-->
  to mean 0.536 (≈chance) at dt=1.0, and the comprehended answer is preserved on 0/6 seeds in both homeo-ON cells. <!--derived-->
- **Surprise needs homeostasis ON.** Its predictive-coding mismatch circuit is TUNED at its native operating
  point (`build_expectation_circuit` runs `enable_homeostasis` at the `CoreSimConfig` default True): at homeo=ON
  it separates cleanly (confirm≈0 Hz, contradict≈5–6 Hz) at BOTH dt values. At homeo=OFF the intrinsic thresholds
  never adapt down to the drive, so the whole circuit is SILENT (recall=0, confirm=contradict=novel=0 Hz;
  `surprise_functional` 0/6) — no surprise signal, and the cross synapse cannot bias the role (cross LB 0/6).

So `dt_ms` is NOT the binding constant: each organ tolerates BOTH dt=0.5 and dt=1.0 at its required homeostasis
setting. (This refines the rung-2 prediction that dt was the crux; on this substrate, with the installed cue
weights + per-region-threshold heterogeneity, comprehension's AUC is dt-robust at homeo=OFF.) The lone shared
value the fused engine must pick — a single global `enable_homeostasis` for all 2088 neurons — cannot satisfy
both organs at once. No cell reconciles them: {homeo=OFF → comp✓ surp✗} × {homeo=ON → comp✗ surp✓}.

Homeostasis is not only an operating-point conflict but a COUPLING channel. At homeo=ON the merge stops being
fully byte-clean: surprise byte-id drops to 4/6 in BOTH homeo-ON cells and comprehension byte-id to 5/6 at
(dt=1.0, ON) (it still holds 6/6 at (dt=0.5, ON)). The global homeostatic threshold normalization lets a
co-resident's firing perturb the other organ's read within the step; snapshot/restore read-isolation restores
state AFTER a read but cannot undo a within-read global-normalization coupling. At homeo=OFF the merge IS fully
byte-clean (comprehension + surprise byte-id 6/6 in both cells). This is a second, independent reason the unblock
must be per-region.

## Anti-cheats (each verified, not asserted)

- **Genuinely one pool:** `len(cp_membrane_potential_v)=2088 ≥ n_surprise(1056)+n_role(1032)`; every surprise +
  role region index falls in the one array (6/6 all cells).
- **No weight transport / no host gradient:** the role cue→role weights are the comprehension organ's OWN frozen
  learned synapses (`_install_learned_cue_role` copies them BY NAME and freezes the plasticity gates);
  `enable_stdp=False`, `enable_reward_modulation=False`; surprise wiring is topographic block-diagonal.
- **No reward leak:** `current_reward_signal==0.0`, `reward_baseline==0.0` (asserted every cell).
- **Brain-based reads only:** the comprehension margin is `cp_firing_states` off `sel_agent`/`sel_patient`; a
  tripwire replaces the host `_semantic_contrast` dot-product with a raising sentinel on every read view — it is
  never called. Surprise is windowed `surprise_S` firing. The cross coupling is a real `cp_connections` synapse
  `surprise_S→sel_agent` (load-bearing 5/6 at homeo=ON: intact +1…+60 Hz sel_agent bias vs ~0 decoupled; 0/6 at
  homeo=OFF only because surprise is silent there, not because the synapse is inert).
- **Byte-id isolates the merge from the operating point:** each byte-id is a FRESH decoupled twin (cross=0) vs
  the intact merged bridge (cross=40), differing ONLY in the one coupling edge — no in-place CSR mutation (the
  `_install_block_diagonal_full` toggle does NOT round-trip cleanly and was discarded for byte-id).
- **Determinism:** `cfg.seed=cfg.heterogeneity_seed=cfg.ou_seed=seed`; build-twice byte-identical (6/6).
- **Read-isolation:** generalized `MergedSubstrate.read_isolation` over arbitrary region masks (6/6 both organs).

## Biology grounding (cites that RESOLVE)

- **GABA_B/GIRK + NMDA coexist as independent conductances on one cortical neuron, summing only at the membrane.**
  GABA_B (metabotropic) → GIRK inwardly-rectifying K⁺, a slow hyperpolarizing (subtractive) conductance with
  reversal `E_gabab ≈ −90 mV` (the GIRK potassium reversal) and `τ ≈ 150 ms`, added as `I_gabab = g_gabab·(E_gabab
  − V)`. In-repo resolved anchor: `research/findings/2026-06-08-gabab-girk-conductance-design.md`. NMDA is
  ionotropic, a slow voltage-dependent depolarizing current carrying the Wong-Wang accumulator. Distinct
  receptors, channels, reversals → they sum at the membrane, no direct interaction; one bridge carrying both
  currents per-neuron is faithful.
- **Wong-Wang NMDA-dominated recurrent excitation as the graded decision/accumulator variable** (the role WTA):
  Wong & Wang 2006, *J. Neurosci.* 26(4):1314–1328, cited in
  `research/runners/_phaseB_multicue_competition_spiking_derisk.py:11,166` ("Wong-Wang mutual inhibition",
  "NMDA-slow recurrent").
- **Empirical coexistence already in-repo:** `2026-08-13-one-brain-merge-Norgan-GO.md` — "NMDA per-region mask:
  2 regions enabled (48 neurons)"; this rung reproduces it (48/48) 6/6.

## The honest negative and the mapped `sim/` engine feature

Biology runs INTERACTING processes and we implement ONE, substituting a static bound for the rest. Here the
constant we replaced is the **single global `enable_homeostasis`** (and, secondarily, the single global `dt_ms`):
the fused Izhikevich integrator steps ALL neurons with one homeostasis flag and one timestep. Intrinsic/homeostatic
plasticity is a genuine per-region companion process — the surprise circuit's operating point is set by it (its
weak topographic recall + asserted drive reach threshold only because thresholds adapt), while the Wong-Wang
comprehension WTA's graded margin is corrupted by it. A single global flag cannot host both.

**The mapped unblock is a per-region `enable_homeostasis`** — each `BrainRegion` opts in/out of intrinsic-homeostatic
threshold adaptation, exactly as the per-region NMDA mask (`BrainRegion.enable_nmda`) and
`per_region_threshold_heterogeneity` already scope those mechanisms. The existing
`per_region_homeostasis_isolation` flag is NOT this: it only freezes IDLE co-resident regions from drift; it
cannot let an actively-firing region undergo homeostasis while a co-resident opts out. With per-region homeostasis
(surprise ON, role OFF) at the shared dt=0.5 — where the merge is already byte-clean and both organs' reads are
individually functional — this pair is expected to reconcile. A per-region sub-stepping integrator (per-region
`dt_ms`) is the second, weaker owner-scoped lane (dt is not binding here, but a dt-invariant surprise rate-window
read would remove even the residual dt sensitivity).

This is a distinct, precisely-named, owner-scoped `sim/` feature lane — not a defeat. The config-superset merge
INFRASTRUCTURE is DONE and byte-clean (one pool + determinism + GABA_B/NMDA coexistence + byte-clean reads +
read-isolation, all 6/6 at homeo=OFF); the residual is a single engine primitive.

## What stays a host shortcut after this rung (declared residual)

The comprehension organ's cue LEXICON (ANIMACY / VERB_SELECTS) is the toy 2-noun transitive scope (its declared
vocab ceiling), and this rung is a DE-RISK (reads driven on the merged bridge Norgan-style). The PRODUCTION
`shared=` wiring of `ComprehensionProductionOrgan` onto the pool (mirroring surprise/metacog `get_organ(shared=…)`)
+ the DEFAULT flip is the follow-on production rung, gated on this pair reconciling — which is gated in turn on the
per-region `enable_homeostasis` engine feature named above.

## Files

- New runner: `research/runners/_one_brain_merge_configsuperset_production_derisk.py`
- Additive edit (default-preserving): `research/runners/_spiking_comprehension_monitor_derisk.py` — `_build_comp`
  gains `dt_ms`/`homeostasis`/`per_region_thresh` kwargs (threaded to `SpikingRoleCompetition`, which already
  accepts them); defaults reproduce the standalone monitor bit-for-bit (every existing caller passes seed only).
- Learned-weight install hook `_install_learned_cue_role` lives in the new runner (no edit to
  `_one_brain_merge_Norgan_derisk.py`; `build_merged_diffbuilder` is reused unchanged).
- Artifact: `research/findings/raw/_one_brain_merge_configsuperset_6seed.json`.
