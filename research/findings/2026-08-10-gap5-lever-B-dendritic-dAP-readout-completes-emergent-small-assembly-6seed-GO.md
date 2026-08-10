---
type: finding
status: live
date: 2026-08-10
mechanism: gap5-dendritic-dAP-readout-completion
lane: F (gap#4/#5 episodic composition seam)
instrument: per-seed control-comparison suite in research/findings/raw/_gap5_dapB/dapB_6seed.json — the LINEAR (coincidence-off) point-neuron control (must still fail = lever load-bearing), the mossy-LESION membership attribution (100% DG-derived), the permuted-cue + silent-rest + no-encoding + recurrence-zero collapses, and the genuine-formation weight check separate WHERE the completion comes from (the per-cell dendritic plateau reading a BTSP-formed recurrent weight), not merely its size
artifacts:
  - research/findings/raw/_gap5_dapB/dapB_6seed.json
  - research/runners/_gap5_dendritic_dap_readout_completion_derisk.py
---

# gap#5 LEVER B GO (6-seed) — the intrinsic per-cell dendritic dAP READOUT completes the EMERGENTLY-SELECTED small ~23-cell BTSP assembly CUE-SPECIFICALLY, where the RECURRENT slow-NMDA reverberatory attractor could not

<!--derived-->

**2026-08-10.** The gap#5 composition SEAM (544c0b742 / cff6a8e2f): the three individually-GO pieces (emergent-DG
SELECTION · BTSP FORMATION · slow-NMDA COMPLETION) do NOT compose, because the emergently-selected assemblies are SMALL
(~14-33 cells) and the RECURRENT completion is NON-SPECIFIC on them (perm≈nocue≈cue) — a ~23-cell set is too small for a
RECURRENT bistable attractor at any inhibition (cue-completion and self-ignition share the within-assembly recurrent
gain; a 9-point density×ff grid found no window). **LEVER B replaces the recurrent-attractor completion READ with an
INTRINSIC per-cell dendritic dAP READOUT bistability that is SIZE-INDEPENDENT** — each CA3 cell's apical dendrite holds
its own bistable UP/DOWN latch (`enable_two_compartment_dap`: `fused_coincidence_plateau` regenerating on `cp_v_apical`
+ self-regen + KIR down-state), so cue-ignition is decoupled from a large recurrent population. This is the 2026-07-08
R-iii dAP completion (GO 0.571 vs LINEAR 0.007 on HAND-INSTALLED attractors,
[`2026-07-08-riii-onsubstrate-dendritic-dAP-completion-SURPASS-6seed.md`](2026-07-08-riii-onsubstrate-dendritic-dAP-completion-SURPASS-6seed.md))
applied to the EMERGENT small assembly, and read via the DECOUPLED apical-state read that the 2026-07-18 magnitude-capped
payoff named as its cheapest untested next-mechanism (#1).

This is a **READOUT-bistability lever, EXPLICITLY DISTINCT from the tested-NEGATIVE dendritic deep-CREDIT / BDSP rule**
([`2026-05-17-dendritic-credit-assignment-NEGATIVE.md`](2026-05-17-dendritic-credit-assignment-NEGATIVE.md)): the plateau
READS a BTSP-formed recurrent weight; there is no hidden-credit learning (BDSP learning is OFF throughout).

## Result — 6/6 GO on the APICAL read (emergent membership · lever load-bearing · all anti-cheats pass)
<!--derived-->
Numbers below are from `research/findings/raw/_gap5_dapB/dapB_6seed.json` (per-seed full precision; table rounds for
presentation, AGG is the mean over seeds); run
`SIM_BACKEND=cupy python -m research.runners._gap5_dendritic_dap_readout_completion_derisk --seeds 42 43 44 100 101 102
--densities 0.5 --wmax 100 --kthresh 15 30`. Operating point: n_ca3=2000, density 0.5, BTSP w_max 100 (the SMALL-weight
dAP regime — see below), plateau_strength 30, apical_R 0.15, k_thresh 15, self_regen 2.0, KIR 1.0, up_thresh −20 mV.
The GO bar (per seed, apical read): held_cue≥0.20 AND ≥3×held_perm AND ≥3×held_nocue AND held_nocue≤0.10, with the
LINEAR control still failing.

```
seed  sizes         APICAL cue  perm  nocue  no-enc  rec-zero  w_within  genuine  kt  LINEAR-ctrl(cue/perm)  GO
 42   [33,13,21]      0.517     0.000 0.000   0.000   0.000      87      True     15   0.000/0.000 (FAILS)   ✓
 43   [21,25,17]      0.293     0.000 0.000   0.000   0.000      84      True     15   0.000/0.000 (FAILS)   ✓
 44   [32,15,16]      0.438     0.000 0.000   0.000   0.000      82      True     15   0.000/0.000 (FAILS)   ✓
100   [22,25,21]      0.545     0.000 0.000   0.000   0.000      83      True     15   0.000/0.000 (FAILS)   ✓
101   [23,24,24]      0.444     0.000 0.000   0.000   0.000      81      True     15   0.000/0.000 (FAILS)   ✓
102   [31,26,25]      0.830     0.000 0.000   0.000   0.000      81      True     15   0.000/0.000 (FAILS)   ✓
AGG                   0.511     0.000 0.000   0.000   0.000                            (all fail)          6/6
```

The held-out (non-cued) assembly members reach the apical UP state at 0.29-0.83 (agg 0.511) from a partial cue, with the permuted
cue, the silent rest, the no-encoding baseline, and the recurrence-zeroed matrix ALL at exactly 0.000 — a genuine,
cue-triggered, silent-at-rest, structure-dependent completion. This clears the strict joint bar (≥0.20) that the
2026-07-18 soma-read payoff on PRE-ASSIGNED assemblies capped at 0.156.

## The anti-cheat suite (the instrument — WHERE the completion comes from, not its size)
<!--derived-->
Per-seed, from the same artifact.
1. **EMERGENT membership 6/6** — the assemblies are DG-SELECTED (mossy detonator), NOT hand-set: sizes are the emergent
   ~13-33 (not the readout's 0.18·N=360 pre-assigned mask), Jaccard vs the random-permutation set is low, and the
   **mossy-LESION collapses the membership 100%** (Verdict control: intact ~69 cells vs lesion 0, |sep|=69; the DG→CA3
   detonation is load-bearing).
2. **LEVER LOAD-BEARING (the key control)** — the LINEAR (coincidence-off) point-neuron read of the SAME BTSP-formed
   weights FAILS at every seed (cue 0.000). The dendritic dAP readout is what completes, not the weights alone — the
   recurrent read that the seam finding showed is non-specific here does not complete via the point soma either.
3. **cue-specific** — permuted cue 0.000 (cue ≥3× satisfied vacuously with perm=0), silent-rest nocue 0.000
   (no self-ignition / limit-cycle artifact — the bistable DOWN state is stable).
4. **BTSP formation genuine** — w_within grew from the fused_btsp_update RULE (≈85 from a 1.5 baseline) with cross/
   non-member dW ≈ 0 (specificity by construction, isolated per-assembly episodes); no hand-set constant. no-encoding
   baseline = 0.000 (BTSP is load-bearing).
5. **recurrence-zero** — zeroing ca3→ca3 collapses completion to 0.000 (it is the recurrent structure the plateau reads,
   not cue re-drive).
6. **plasticity FROZEN** at recall; BDSP/BTSP/STDP/Hebbian all OFF during the read; OU membrane noise OFF (isolates the
   DETERMINISTIC per-cell bistability).

## Why it works where the recurrent attractor and the soma-read did not
<!--derived-->
- **Size-independence.** The recurrent reverberatory attractor needs a large mutually-recurrent population to hold a
  bistable high state; a ~23-cell set is below that. The dAP latch is INTRINSIC to each cell — a held-out cell ignites
  when its own within-assembly recurrent input (from co-firing cue partners) crosses the plateau threshold, regardless
  of population size.
- **The decoupled apical-state read.** The 2026-07-18 payoff capped the SOMA-firing read at 0.156 because a strong
  apical→soma read re-closed the assembly's within-member recurrent loop (self-ignition). Here the completion is read
  as the apical UP-state fraction with WEAK apical↔soma coupling (apical_g_couple 0.3): the plateau HOLDS the memory
  without the soma firing hard enough to re-drive the loop. The SOMA read here is 0.000 at every seed — confirming the
  decoupling: the memory lives in the apical latch, exactly as the payoff's next-mechanism #1 predicted.

## What the de-risk had to discover (two engineering seams, both quantified)
<!--derived-->
From the diagnostic sweeps (scratchpad, not committed; the operating point is reproduced by the runner).
1. **Scale mismatch → numerical divergence.** The BTSP weights that make the slow-NMDA reverberatory attractor work are
   ~9000; routing 9000-weight synapses through the coincidence plateau at the R-iii apical_R=50 makes `apical_R ×
   I_coincidence` jump thousands of mV/step → the forward-Euler apical ODE DIVERGES (measured p50 −400..−1500 mV, max
   +800 mV). **The dAP readout requires a SMALL-weight regime** (w_max≈100 → w_within≈85, apical_R≈0.15,
   plateau_strength≈30). The specificity is a RATIO (within-weight ≈85 vs baseline 1.5), achievable at any scale, so
   forming BTSP at w_max=100 costs nothing.
2. **Per-step vs summed drive.** The static c_drive (all cue cells firing at once) is ≈161 on held cells vs ≈11 for a
   permuted cue (15× separation), but the plateau reads `c_drive = co_matT @ prev_firing_states` — only cells that fired
   the PREVIOUS step. Per-step c_drive on held is p50 0 / p90 93 / max 349 (sparse gamma volleys; cue cells fire
   ~3%/step, held soma stays silent). k_thresh=15 (between the baseline synapse 1.5 and a within-assembly synapse 85)
   catches the volleys and rejects the permuted cue. A carryover BUG was fixed: `hard_silence` does not reset
   `cp_v_apical` or the plateau conductance, so a prior read's self_regen latch contaminated the next (every read read
   UP); each recall episode now starts from the apical DOWN state at rest.

## Honest residuals (per THE LAW — each launches the next method, none is a wall)
<!--derived-->
- **k_thresh sensitivity.** k_thresh=15 is the working point at all 6 seeds; k_thresh=30 gives 0 on some seeds (the
  volley amplitude varies with the emergent size). The next companion process is a HOMEOSTATIC per-cell plateau
  threshold (an intrinsic-excitability set-point, cf. Kopsick divisive homeostasis) so k_thresh self-adjusts to the
  per-cell drive instead of a global constant — the same "replace the constant with the process the animal runs
  alongside it" pattern.
- **Density 0.5.** Demonstrated at ca3_density 0.5 (good within-assembly fan-in); the recurrent path failed at 0.5 too,
  so this closes the seam AT the density where it was open, but robustness across the 0.12-0.5 range the seam swept is
  the next sweep (at 0.12 many held cells have zero within-assembly cue inputs — a fan-in limit no readout can beat, so
  the mossy/recurrent density is itself a substrate parameter to characterize).
- **Read-during-cue vs hold-after-offset.** The completion is read while the partial cue is still driven (the plateau is
  sustained by the sparse cue volleys + self_regen between them). Whether the apical UP-state HOLDS after full cue
  offset (a true standalone attractor memory) is the stronger claim; the self_regen/KIR bistability is built for it and
  the hold-test scaffolding is in the runner — the next measurement.
- **This is a de-risk GO, not a "closed" capability.** The shipped default completion path is still the recurrent
  slow-NMDA read; wiring the dAP readout into the end-to-end loop as the default is the integration step.

## Files
<!--derived-->
Runner: [`research/runners/_gap5_dendritic_dap_readout_completion_derisk.py`](../runners/_gap5_dendritic_dap_readout_completion_derisk.py)
(reuses the emergent-membership + mossy-lesion from `_gap5_emergent_end_to_end_episodic_loop_derisk.py`, the BTSP
formation `form_btsp_multi` + masks `make_readout` from `_gap5_btsp_forms_nmda_slow_reverberatory_derisk.py`, and the
two-compartment dAP `_build` from `_riii_ca3_coincidence_completion_derisk.py`; NO `sim/` edit). Artifact:
`research/findings/raw/_gap5_dapB/dapB_6seed.json`. Biology: Major-Larkum-Schiller (NMDA spike / dAP), Sanders 2013
(KIR "perfect couple" down-state), Bittner-Magee 2017 / Milstein-Magee 2021 (BTSP one-shot). Prior:
[`2026-07-08-riii-onsubstrate-dendritic-dAP-completion-SURPASS-6seed.md`](2026-07-08-riii-onsubstrate-dendritic-dAP-completion-SURPASS-6seed.md)
(dAP on hand-installed), [`2026-07-18-gap5-CA3-bistable-dendrite-payoff-bistability+specificity-solved-magnitude-capped.md`](2026-07-18-gap5-CA3-bistable-dendrite-payoff-bistability+specificity-solved-magnitude-capped.md)
(soma-read cap 0.156 on pre-assigned + the decoupled-read next-mechanism), the seam runners (544c0b742 / cff6a8e2f).
