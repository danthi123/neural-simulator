# gap#5 candidate #3 (Kamondi ramp-phase-advance) — 6-seed HONEST NEGATIVE on point neurons; the mechanism is intrinsically DENDRITIC (2026-07-24)

## Result (6-seed, pool, numpy)
`_gap5_ramp_phase_advance_readout_derisk --seeds 42 43 44 100 101 102 --n-ca3 2000 --rest-steps 2200`: **GO 0/6**.
The test is SOUND — every control collapses as designed (`shuffled_collapses`, `reverse_collapses`,
`basket_off_collapses` all True; `frozen_ok` True) — and the setup is live (theta-basket reference fires; the ramp
drives assembly-0; the decoupled forward-asymmetric store is correct: w_within≈208, adj_fwd≈38.6, adj_rev≈5.0,
ratio 7.71). But the **main condition also fails**: `main_phase_order=false`, `per_cycle_support=false`, per-assembly
phase-lock resultant **R≈0.015–0.07 (mean ≈0.055)** — i.e. essentially NO phase locking, so the assemblies do not
occupy clean forward-advancing theta phases. `per-asm active [[0,0,0]]`; per-cycle forward_frac 0.000.

## Diagnosis — this is the POINT-NEURON LIMIT family, not an operating point
The runner's auto-verdict suggests operating-point tweaks (stronger sel_inhib_spare/adj-fwd, per-cycle re-seed,
wider theta window). But **R≈0.05 is not "weak locking" — it is ~no locking at all**, which points to a structural
cause, not a strength knob. The Kamondi (1998) single-cell-pacemaker mechanism that produces phase precession is
**intrinsically DENDRITIC**: a ~1 s dendritic depolarizing ramp drives a *voltage-dependent dendritic oscillation
slightly faster than the somatic theta*, and THAT intrinsic frequency offset is what advances the spike phase each
cycle (Buzsáki *Rhythms of the Brain* pp. 319, note 88; Kamondi et al. 1998, intradendritic recordings). A **point
neuron (Izhikevich) has no dendritic compartment** — it cannot host the sub-threshold dendritic oscillation, so a
depolarizing ramp just raises its rate uniformly across theta phases (R≈0), never producing a phase-advancing pace.
This is the SAME family as the project's documented point-neuron limits (whitening/decorrelation are analog/dendritic,
Mikulasch-Priesemann; graded divisive-normalization). Candidate #3 asked a point substrate to do a dendritic
computation.

## Per THE LAW — the METHOD is banked, the CAPABILITY stays OPEN
The imaginative-replay / theta-ordered-sequence capability remains open. The banked negatives for gap#5's readout are
now: (A) spontaneous bistable ignition = NO; (B) targeted DG-detonator ignition = NO; **(C) Kamondi ramp phase-advance
on POINT neurons = NO (this finding).** The through-line hardens: on a point substrate neither ignition NOR intrinsic
phase-precession orders the chain.

## Next (research-gate dispatched, NOT a blind operating-point sweep)
Two candidate directions the gate must rank (read the sources, don't assume):
1. **Run the Kamondi ramp on the project's DENDRITIC substrate** (the two-compartment / dendritic-plateau path already
   built — `sim/dendritic_*`, `_dendrite_stage1_onbridge_graded_plateau`), where an apical dendritic ramp CAN host the
   faster-than-theta oscillation. This tests whether phase precession needs the dendrite (the hypothesis above).
2. **A NETWORK-level ordering mechanism that does NOT need intrinsic precession** — theta-nested gamma where
   feed-forward (basket) inhibition timing + the forward-asymmetric links sort assemblies into successive gamma slots
   within the theta cycle (Lisman-Idiart theta-gamma phase coding; the ordering comes from inhibition-release timing,
   not an intrinsic dendritic pacemaker). This could work on point neurons if the gamma-slot mechanism carries it.
The gate reads Buzsáki (theta-gamma), Kandel Ch 54, Lisman-Idiart, and the catalog to rank these + specify the cheap
de-risk. NO `sim/` edit in this arc (the ramp is external current; theta-basket reuses the FS route).

## UPDATE — the DENDRITIC Kamondi ramp ALSO fails 6-seed (option ② on the two-compartment substrate, 3090, 2026-07-24)
The research gate's parallel diagnostic (`_gap5_ramp_dendritic_phase_advance_derisk`, NO `sim/` edit — reuses the guarded
two-compartment apical path) tested whether a REAL apical compartment restores the phase-precession the point soma
couldn't. **6-seed HONEST NEGATIVE 0/6.** The apical substrate is present (apical_present 6/6; ramp targets the apical;
basket reference fires), but the readout produced NO clean forward-ordered phase-locking: **MAIN mean R ≈ 0.103 vs the
point-neuron control ≈ 0.104 — the dendritic compartment did NOT lift phase-locking above the point neuron.**
per_asm_active is the familiar dichotomy (`[6,6,6]`/`[4,4,4]` diffuse co-fire on some seeds, `[0,0,0]` no-ignition on
others); forward_frac 0.292 vs chance 0.222 (not a clean order). Per-seed R noisy + inconsistent (0.03–0.36, and seed
43's point-neuron control 0.368 > its dendritic 0.056).

**⇒ The Kamondi single-cell-pacemaker phase-advance is banked NEGATIVE on BOTH point-neuron AND two-compartment
substrates.** This CONFIRMS the research gate's load-bearing call: the intrinsic-pacemaker marginal phase-advance is a
FRAGILE order carrier (even with a real dendrite), and the ROBUST ordering primitive is the NETWORK gamma-slot / cued
theta-disinhibition sweep (option ①, riding self-avoidance + the adjacent-vs-skip structure, NOT the intrinsic
oscillation) — which is the top pick and still running. Method banked; capability OPEN; the answer rides on ①.
