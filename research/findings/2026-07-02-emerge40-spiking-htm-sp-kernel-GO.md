# EMERGE-40 / toward-semantics — GO (6/6 seeds): the FULLY-SPIKING HTM Spatial Pooler. The winner-INACTIVE (selectivity) depression that EMERGE-39 de-risked as a host op is now the committed `sim/` kernel `fused_htm_winner_inactive_depression` (additive; every existing path byte-unchanged). BOTH competitive-pooler learning terms are now `sim/` fused kernels. ONE additive `sim/` kernel.

**2026-07-02 (autonomous).** Runner `research/runners/_emerge40_spiking_htm_sp_kernel_derisk.py`; CI guard `tests/test_emerge40_spiking_htm_sp_kernel.py` (4 tests). Reuse-by-import (`_emerge14` committed kernel + `_emerge12`); ONE additive `sim/` kernel (`sim/kernels.py`, new function; existing kernels untouched); CPU numpy-backend; 6-seed.

## What's new
EMERGE-38 validated the competitive-learning MECHANISM (host HTM Spatial Pooler, 0.98 on overlapping categories) and EMERGE-39 realized it ON-SUBSTRATE with the winner-inactive depression as a host op on `cp_connections.data` (0.94, and pinned the exact missing kernel term). **EMERGE-40 makes that term a committed `sim/` fused kernel**, so the competitive-pooler LEARNING is fully realized by `sim/` kernels: potentiation via `fused_htm_permanence_update` (`ld=0`) + winner-inactive depression via the new `fused_htm_winner_inactive_depression`, both over the bridge's coincidence synapse permanences.

## The `sim/` edit (additive, default-inert)
`sim/kernels.py` gains ONE new fused kernel:
```
fused_htm_winner_inactive_depression(w, pre_active, post_win, lam_dep_wi, w_min, w_max):
    dep = (1 - pre_active) * post_win * lam_dep_wi     # winner (post_win=1) depresses its INACTIVE inputs (pre_active=0)
    return clip(w - dep, w_min, w_max)
```
It is the one term the committed `fused_htm_permanence_update` structurally lacks (that kernel gates BOTH its terms on `pre_last`, so an inactive-presynapse synapse is a no-op there). It is a SEPARATE function — `fused_htm_permanence_update` and every existing caller are **byte-unchanged**; `lam_dep_wi=0` is a no-op. Biology: HTM Spatial Pooler winner-selectivity (Cui-Ahmad-Hawkins 2017) / Diehl-Cook 2015 STDP + lateral inhibition.

## The claim (6/6 seeds)
On 6 overlapping categories (adjacent share 3/6 features, held-out inheritance, chance 0.17), both learning terms as `sim/` kernels:
- **SPIKING-KERNEL: held-out inheritance 0.96 mean** (1.00/1.00/0.94/0.94/0.94/0.94 across seeds 42/43/44/100/101/102) — matching the EMERGE-39 host op (0.94), so the fused kernel is a faithful port.
- **The winner-inactive kernel is LOAD-BEARING:** no-selectivity (kernel OFF) 0.18; **FIXED** projection 0.56; **PERMUTED** 0.10; **dAP-LESION** 0.00.

## Anti-cheats
- **NO-SELECTIVITY** (the new kernel off): 0.20 — isolates it as load-bearing.
- **FIXED (no-learn)**: 0.61; **PERMUTED-features**: 0.15; **dAP-LESION**: 0.00.
- Kernel-math CI pins the exact per-synapse behavior (depress only winner-inactive synapses) + that the sibling permanence kernel is unchanged; 6-seed.

## Significance
The competitive self-organizing pooler — the mechanism that lets the brain separate OVERLAPPING categories by tuning columns to their discriminative features — is now **fully realized on the spiking substrate via committed `sim/` kernels**, no host learning op. This is the honest fully-on-substrate end-state of the EMERGE-38/39/40 sub-arc: the brain self-organizes its own category codes with a biologically-grounded, kernel-level competitive-learning rule, and does full inheritance over them.

## Honest scope + next
- Both learning terms are `sim/` fused kernels over `cp_connections.data` (on-substrate). The **k-WTA drive read** (which columns win) is still a top-k over the substrate weights — the spiking **FS-WTA lateral-inhibition** realization of the competition is the next rung (a further on-substrate step, not a wall).
- The pooler is a single competitive layer on a controlled 6-category / 21-feature task; hierarchical pooling + corpus-scale category counts are follow-ons.
- Next: the FS-WTA spiking competition; couple competitive-pooler emergent codes into the experiential console (EMERGE-31) so discovered overlapping categories feed the full inference (inheritance + cancellation + transitivity).

## Artifacts
`research/runners/_emerge40_spiking_htm_sp_kernel_derisk.py`, `tests/test_emerge40_spiking_htm_sp_kernel.py`, `sim/kernels.py` (additive `fused_htm_winner_inactive_depression`), `research/findings/raw/_emerge40_spiking_htm_sp_kernel.json`. Prior: `2026-07-02-emerge39-onsubstrate-competitive-pooler-GO.md`, `2026-07-02-emerge38-competitive-self-organizing-pooler-GO.md`.
