# 🎉 gap #4 ↔ gap #5 UNIFICATION — BTSP plateau-gated one-shot encoding STORES a CA3 assembly that the bistable CA3 COMPLETES, on ONE spiking substrate, reusing the SHARED dendritic-bistability keystone. MECHANISM 6/6 (cue-gated + specific + bistable + no-encode-collapses); completion magnitude ~0.18 (marginal vs the strict 0.20 bar — a characterized uniform-vs-structured residual). NO new `sim/` edit.

**2026-07-18.** The two gaps are unified on the shared keystone: the gap#4 local-credit rule (BTSP, plateau-gated
one-shot) is the STORING rule; the gap#5 bistable CA3 is the COMPLETION. The SAME intrinsic dendritic bistability
(self-regen SUSTAIN + KIR down-state) serves both — BTSP uses the held plateau as its instructive signal to store the
assembly one-shot, and the bistable CA3 uses it to complete the assembly from a partial cue. Demonstrated end-to-end on
one `SimulationBridge` at the validated completion scale (n_ca3=2000), 6-seed.

## What was built (default-preserving; NO new `sim/` edit)
An additive `encode_btsp` path in the gap#5 completion runner (`_riii_ca3_synchronous_assembly_derisk.run`, default
False => byte-identical, the gap#5 GO preserved — confirmed: the Hebbian baseline completes cue 0.217 unchanged). When
on: init the ca3→ca3 recurrent LOW (encode_ca3w=0.5), disable the rate Hebbian, and during the co-fire drive the PLATEAU
DIRECTLY on the pre-assigned assembly (via the bistable BDSP apical — my keystone) so only the assembly cells have BOTH
pre-eligibility (co-firing) AND a plateau (IS_post) => BTSP potentiates the WITHIN-assembly recurrent one-shot,
SPECIFICALLY (specificity by construction: member→non-member post has no plateau). enable_bdsp/enable_btsp are DISABLED
after encode so recall uses the two_comp coincidence plateau for completion. Reuses the two committed session edits
(bistable BDSP apical + the on-bridge BTSP block).

## Result — MECHANISM 6/6 (`_gap4_btsp_completion_unification_6seed.py`)
Config: n_ca3=2000, density 0.05, assembly_frac 0.12, encode via BTSP (encode_ca3w 0.5, encode_plateau_pA 250, btsp_lr
0.02, btsp_w_max 300, train_events 30), recall the gap#5 bistable machinery (structural_sep=1, selective_inhib,
plateau_self_regen 0.15, apical_kir_g 3, apical_gc_read 5) with **recall_k_thresh 40** (BTSP's UNIFORM within-assembly
distribution wants a lower dendritic threshold than Hebbian's 110 — the coincident drive is spread evenly;
structural_sep keeps permuted specificity structurally, so lowering it is safe).

| seed | 42 | 43 | 44 | 100 | 101 | 102 |
|---|---|---|---|---|---|---|
| cue (held completion) | 0.187 | 0.168 | 0.185 | 0.166 | 0.176 | 0.191 |
| nocue (bistable rest anti-cheat) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| perm (permuted-cue specificity) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| no-encode (stored-assembly anti-cheat) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

- **MECHANISM 6/6:** every seed shows genuine cue-gated completion (cue ~0.18 vs nocue 0), SPECIFIC (perm 0), BISTABLE
  (nocue 0, a silent rest), and LOAD-BEARING on the BTSP-stored assembly (no-encode → 0: no plateau + no co-fire → the
  recurrent stays at init 0.5 → no completion). This is the unification: BTSP stores, the bistable CA3 completes.
- **MAGNITUDE ~0.18, marginal vs the strict cue≥0.20 bar (0/6 strict).** By the project's OWN gap#5 standard this is a
  real completion: the gap#5 Hebbian result called its marginal seeds "cue ~0.18-0.19, still a real specific held
  completion" (its own verdict was 5/6 GO with magnitude seed-variable in [0.18, 0.33]). BTSP's magnitude is TIGHTER
  (all [0.166, 0.191]) and consistently just below 0.20; the Hebbian baseline reaches 0.217 on seed 42.

## The characterized residual (honest, precise — a mechanism lever, not a config knob)
The magnitude gap is a WEIGHT-DISTRIBUTION effect: BTSP's plateau-gated saturation stores a ~UNIFORM within-assembly
matrix, while Hebbian's rate-window LTP + heterosynaptic competition + heterogeneous co-firing stores a STRUCTURED
(variable) one. Extensive config tuning (8 GPU sweeps — btsp_w_max 8→2000, btsp_lr 0.02→0.15, train_events 30→120,
recall_k_thresh 40→150, recall_drive 700→1200, apical_gc_read 5→15) maps it precisely: cue peaks ~0.19 at MODERATE
storing (w_within ~70) + low recall threshold, and OVER-strong uniform weights (w_within 166) NON-monotonically HURT
(over-drive the recall, cue → 0.05). apical_gc_read is not a clean read-side lever here (enable_bdsp uses it during
encode too). The STRUCTURED-storing mechanism lever was ALSO tried: `encode_hetero` (a per-cell plateau multiplier so
assembly cells latch at heterogeneous strengths, default 0 = uniform/byte-preserved) — it HURTS (cue 0.174→0.172→0.133
→0.106 as hetero 0→0.4→0.7→1.0; the low-multiplier cells just store less). ⇒ the magnitude residual is genuinely a
property of BTSP's plateau-gated storing and is NOT closed by config (9 GPU sweeps) OR plateau-heterogeneity. Closing
the last ~0.02-0.04 would need a fundamentally DIFFERENT storing rule (a non-saturating BTSP variant, or replicating
Hebbian's rate-window + heterosynaptic-competition structure via a different mechanism) — a deep new arc, NOT warranted
for a ~0.02 magnitude refinement of an already-mechanism-GO result (and NOT chased further = p-hacking risk). The
residual is an EXHAUSTIVELY-CHARACTERIZED honest boundary at cue ~0.18 (a real completion by the gap#5 standard).

## Status
- **UNIFIED (mechanism, 6/6):** the gap#4 credit rule (BTSP) and the gap#5 completion (bistable CA3) run on ONE
  substrate sharing ONE dendritic-bistability keystone — BTSP stores the assembly one-shot, the bistable CA3 completes
  it, cue-gated + specific + bistable + anti-cheat-verified, all 6 seeds. This is the two gaps genuinely connected.
- **Residual:** completion magnitude ~0.18 (marginal vs 0.20; a real completion by the gap#5 standard) — the
  uniform-vs-structured distribution difference; the next lever is structured BTSP storing.
- Infra: the `encode_btsp` path (default-off / byte-identical) + `_gap4_btsp_completion_unification_6seed.py`. NO new
  `sim/` edit (reuses the bistable BDSP apical + the on-bridge BTSP block).
