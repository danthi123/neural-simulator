# gap#5 emergent-DG sparse-pattern-separation — research gate: the diffuseness is a CA3 failure (DG is fine), root-caused to a biologically-wrong DENSE mossy fan-out + all-or-none recurrent; R1 = reproduce + scale-bisect + sparse detonator (no `sim/` edit)

**2026-07-21.** After the per-pathway-STP mossy-detonator BOUNDARY, the research-gate discipline fired (a-1 RAG + external
biology + ranked de-risks, read-only). Decisive, precise diagnosis + a ranked cheap-first path.

## DIAGNOSIS — a HYBRID gap, localized to CA3 (the DG is fine)
Three-stage localization (from our own measurements):
- **DG output: sparsity INTACT** (~20%, ~60/300 cells, reproducible Jaccard 1.00, separated J 0.07-0.18). "The residual is NOT a DG problem."
- **Mossy fan-out: sparsity PARTIALLY LOST** — density **0.10** = each DG cell → **200 of 2000 CA3**, **16× denser** than the biological detonator (Acsády 1998: ~15 giant boutons/granule, ~50 MF inputs/CA3 cell). Our fan is *dense-and-moderate*; biology is *sparse-and-GIANT*. A sparse DG code (~60 cells) is SMEARED across a broad CA3 seed — the opposite of the detonator's job (transmit sparse→sparse, NOT generate sparsity).
- **CA3 recurrent+inhibition: sparsity CATASTROPHICALLY lost (the knife-edge)** — the bistable recurrent is ALL-OR-NONE: a target-sparse 6-40 set (0.3-2% of 2000) expects only `40×0.05≈2` within-set recurrent inputs (marginal for self-sustenance) → either the un-potentiated sparse set can't reach the plateau's coincident drive (→ transient decays → silent) OR gain is raised and the regenerative plateau ignites globally (→ 2000-avalanche). Two stable fixed points, no sparse one between.

**Why generic "add feedback inhibition" is already spent:** global feedback inhib SATURATES at sparsity 0.21 (byte-identical weight 120 vs 250) and is NON-SELECTIVE (suppresses members too, ratio 1.16×); E%-max needs a per-gamma RESET our substrate lacks (the input gamma-pulse is INERT); subtractive somatic inhibition can't quench a regenerative dendritic plateau (Kopsick 2024: the fix is assembly-SELECTIVE learned E→I, not more weight).

**The decisive surpassability clue:** a sparse-separated-stable selection PROVABLY EXISTS at **n_ca3=400** (6-seed GO,
`2026-07-19-...-SELECTION-de-risked-GO`: 10-37 cells, sep 0.04-0.16, Jaccard 0.94-1.00) and FAILS at n_ca3=2000. The two
differences: (i) SCALE (a fixed-FRACTION mossy fan scales the detonation with N — density 0.10 → 40 cells at N=400 but
200 at N=2000), (ii) RESET discipline (the GO used a full post-build snapshot/restore between presentations; the committed
path does only `reset_steps=10`). AND we don't currently reproduce our own best result (the GO rested on deleted scratchpad).

## RANKED PATH (research gate)
- **R1 (recommended first — runner-config, NO `sim/` edit, ~3-5 GPU-h):** restore the snapshot/restore reset discipline
  (reuse EMERGE-61 `_restore_state`/`from_host`, byte-identical, public-attribute writes) + SCALE-BISECT n_ca3 ∈
  {400,700,1000,1400,2000} at the `EMERG` op-point, folding in a 2×2 co-sweep {mossy dense (d0.10/w200) vs SPARSE
  DETONATOR (d0.02/w1000, "few-but-strong" — ~40 synapses/DG cell, N-independent count)} × {recurrent w4.0 vs w2.5}.
  ⇒ either RECOVERS the 6-40 sparse-separated-stable assembly at scale (deliverable), or shows the stable window CLOSES
  at a critical N (a clean result that justifies R4/BTSP rather than building it blind). Anti-cheats: mossy-LESION → 0,
  PERMUTED-input overlap <0.13, read the NATURAL ≥θ assembly (NOT top-k). Mission-fit: neutral (a reproduction/diagnostic).
- **Downstream (only if R1 shows the window genuinely closes):** the sparse-detonator N-independent fan + assembly-selective
  inhibition + the one-shot BTSP keystone (the gap#4↔#5 unification, which currently works on PRE-ASSIGNED assemblies) to
  carve the within-set recurrent basin so the sparse set self-sustains — the genuine mechanism gap.

## Read-out
The gap#5 emergent-DG is SURPASSABLE (it works at N=400), not a substrate wall — the honest first move is a
reproduction + scale-bisection + the biologically-correct sparse detonator (fix the fixed-fraction fan-out error), which
gates whether the residual is a cheap scale/reset artifact or the genuine BTSP keystone. R1 dispatched (waits for the
GPU). NO `sim/` edit in R1.
