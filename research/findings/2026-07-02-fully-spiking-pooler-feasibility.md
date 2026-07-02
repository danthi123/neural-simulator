# Fully-spiking kWTA pooler — FEASIBILITY confirmed (the EMERGE-33/34 "rate-reference" is replaceable): a coincidence-driven column layer on the bridge forms similarity-preserving codes AND supports on-bridge inheritance, with NO numpy kWTA. Full 6-seed GO pending the robustness treatment. NO `sim/` edit.

**2026-07-02 (autonomous; the EMERGE-33/34 fully-spiking follow-on).** Scratch-probe feasibility (not yet a committed runner). Reuse the validated coincidence drive (`build_pool_bridge` machinery, `coincidence_weighted_drive`); NO `sim/` edit; CPU numpy-backend.

## The honest-scope note it closes (conceptually)
EMERGE-33/34 form the emergent superordinate with a **numpy** competitive Spatial Pooler (kWTA + boosting) — a rate-reference for the representation step — then run the inheritance on the real spiking bridge. The open follow-on was: make the pooler's k-winners-take-all **spiking**. This probe demonstrates the fully-spiking pooler is FEASIBLE.

## What was tried + found
- **A naive spiking WTA** (feature cells → column cells via standard conductance-based SYNAPSES + a column→FS→column lateral-inhibition WTA, reusing EMERGE-11's FS pattern) — **the columns do NOT fire** (0/40) even at feature-drive 500 + 2× feedforward weight. Diagnosis: standard synaptic current with naive weights doesn't reach Izhikevich threshold; EMERGE-11's WTA works because it drives the column cells DIRECTLY.
- **The fix — drive the columns via the VALIDATED coincidence pathway** (option b): each column = a cell that samples a random subset of feature cells through coincidence synapses; a column responds when ≥ act_th of its features are active (the `coincidence_weighted_drive` that already fires reliably across EMERGE-9..34). Result:
  - **Similarity-preserving codes:** input A vs a similar A′ (3 of 4 features shared) → column-response overlap **0.56**; A vs a dissimilar B → **0.00**. Sparse response (6/40). This is the pooler's core property (random-projection LSH-like), fully on the spiking substrate.
  - **Inheritance on the spiking-pooler codes:** teach a property on training objects' column responses (the committed `sim/` three-term kernel) → a held-out object inherits via the overlapping response. Real **1.0 on 2/3 seeds** (mean 0.83; seed 44 weak at 0.5).

## Status + what's needed for a clean 6-seed GO (EMERGE-35)
FEASIBLE and promising, but not yet a clean GO: the 2-category / 1-held-out-per-category setup is coarse (the permuted-features control is noisy — 0.0/1.0/1.0 — and one seed is weak), exactly the small-setup noise the control-validity methodology finding warns about. The completion (EMERGE-35, a committed runner) needs the same robustness treatment applied to EMERGE-33/34: **more members + ≥3 held-out/category** (finer metric), **>2 categories** (lower chance, wider margin), tune the coincidence act_th / column count / feature-subset size for a stable response, and gate on the **input-destruction permuted-features** control + the deterministic lesion. Optionally add an FS-WTA to enforce exact k-sparsity + Hebbian on the feat→col projection to make the columns SELF-ORGANIZE (rather than a fixed random projection).

## Significance
The fully-spiking pooler is not a wall — the coincidence drive (already validated across the EMERGE chain) gives a spiking, similarity-preserving column layer that supports inheritance, replacing the numpy kWTA. The remaining work is the robustness/tuning treatment for a clean multi-seed GO, not a new mechanism.

## Artifacts
Scratch probe (inline); to be promoted to `research/runners/_emerge35_fully_spiking_pooler_derisk.py` with the robustness treatment. Prior: `2026-07-02-emerge33-spatial-pooler-emergence-GO.md`, `2026-07-02-emerge34-perception-grounded-emergence-GO.md`, `2026-07-02-anti-cheat-control-validity-methodology.md`.
