---
type: finding
status: live
mechanism: sparse-engram-pattern-separation
lane: breadth-crux / catastrophic-forgetting
date: 2026-08-08
---

# Teacher-loop breadth crux: a NEURAL k-WTA sparse-gated readout allocation SEPARATES the engrams but does NOT raise sequential retention over the dense readout — honest negative with teeth

**Date:** 2026-08-08 (run 2026-08-09) · **Status:** single-seed SMOKE, **HONEST NEGATIVE with teeth** ·
**Backend:** numpy (the OnBridge Izhikevich net is tiny — 67 neurons — and launch-bound, so CPU is faster than the
3090 here; GPU cupy path verified to run) · seed 42 · artifact
`research/findings/raw/teacher_loop_sparse_readout_s42.json`.

## The wall this attacks (finding fcdc2fd2)

Sequentially teaching N distinct facts into ONE brain by corrective e-prop retains ~1 fact: every fact is learned
perfectly the moment it is taught (immediate held-out ~0.995), then OVERWRITTEN (`frac_recalled ~ 1/N`; the
interleaved control — teacher re-presents old facts — retains 8/10 at N=10 on the SAME net, so CAPACITY is
adequate). fcdc2fd2 located the failure as **sequential interference on the shared leaky readout**: the Bellec
readout `logit_k = Σ_j W_{jk} r_j` reads a DENSE last-hidden eligibility, so teaching fact i moves W over every
unit r activates, dragging down earlier facts' readout weights.

## The mitigation built (brain-based, additive, default-off)

`SparseReadoutEpropNet(OnBridgeEpropNet)` (runner-side subclass, **NO sim/ edit**) adds a **k-winners-take-all
competition** on the last-hidden units so only the k units a percept drives most **above their own homeostatic
baseline** pass to the readout — a dentate-gyrus / cerebellar-granule sparse-coding motif (Marr 1971; Treves-Rolls;
O'Reilly-McClelland CLS). The gate acts on BOTH the forward logit and the e-prop readout gradient, so a masked-out
unit neither contributes nor updates.

**The allocation is NEURAL, not a host table (grep-verifiable).** Winners come from `np.argpartition` over the
neurons' OWN centered eligibility `dev = r_raw − mu`; the class label `y` never enters `_sparse_mask` /
`_readout_feature`. It is a competition (feedback inhibition), not a fact→slot allocator.

## The two companion-process lessons that had to be paid to make the code even INPUT-DEPENDENT

<!--derived-->
| winner-selection signal | winner-overlap (Jaccard) | acquisition | verdict |
|---|---|---|---|
| raw activity `r_raw` (top-k) | **1.00** | 1.00 | same units win for EVERY percept — no separation |
| standardized `(r_raw−mu)/sigma` | **1.00** | collapses to chance | sigma-floor (1e-6) units blow up, always win |
| **centered deviation `r_raw−mu`** | **0.20–0.33** | 0.98 | input-dependent — distinct subspaces per fact |

The separation is not the top-k operator; it is the per-neuron homeostatic baseline the top-k is measured
against — exactly the "what else does the real system run alongside this that we replaced with a constant" term.
And a SECOND companion process had to be frozen: e-prop's hidden-layer plasticity (`hidden_lr_scale=5`) reshapes
the reservoir during teaching and COLLAPSES the sparse code (winner-overlap climbs 0.33 → 0.55 on the trained net),
so the pattern-separation layer is a FIXED random expansion (`freeze_hidden`, biologically apt — DG/granule
expansion is structural; plasticity is at the readout) and learning is readout-only.

## Result — the mechanism SEPARATES but does NOT surpass the wall (seed 42, N=10, hidden=64, k=6)

<!--derived-->
| metric | DENSE readout (baseline) | SPARSE k-WTA (mitigation) |
|---|---|---|
| `frac_recalled` @ N=10 | 0.40 | **0.40** (no gain) |
| mean retained acc @ N=10 | 0.450 | 0.378 |
| immediate acquire acc | 0.99 | 0.98 (held) |
| winner overlap (Jaccard) | — | 0.202 (distinct subspaces) |
| off-flag byte-identical to parent | — | True |

k-sweep (load-bearing): `k=64(dense) 0.40 · k=12 0.40 · k=3 0.50` — only the sparsest k nudges +0.10 at a single
seed. `attributable_to(sparse, dense)` returns 0% (the retention is present identically in the control).

## Teeth — what holds and what does NOT

- (off byte-identical) **HOLDS** — logits identical to the parent with the flag off (mechanism is additive).
- (acquisition held) **HOLDS** — 0.98–0.99; the sparsity does not break learning.
- (separation neural + load-bearing) **HOLDS** — overlap 0.20 with the centered-deviation competition; remove that
  companion term (raw or standardized) → overlap 1.00. No fact→slot table.
- (retention RISES vs dense) **FAILS** — the decisive A/B: sparse does NOT beat the dense readout (both 0.40; graded
  <!--derived--> 0.378 vs 0.450, rounded from `mean_retained_acc` in the cited artifact). The method does not rise
  toward the 8/10 interleaved ceiling.

## Why it fails — what the CAPABILITY actually needs (this is the deliverable)

1. **Winner instability under percept noise** — within-referent winner stability is only ~0.6, so test-time
   winners differ from training-time winners; the engram is re-competed each percept rather than COMMITTED.
2. **The softmax output-side all-vs-all suppression re-introduces interference.** Even with disjoint winners,
   `d = softmax − onehot(y)` actively suppresses every non-target class column on the active units. A LOCAL
   target-only potentiation delta (tested, `local_readout`) removes that suppression but then loses calibrated
   acquisition (acq → 0.61) — the global normalization biology does not run is exactly what was holding
   acquisition together.
3. **A frozen reservoir + homeostatic standardization already gives the DENSE readout comparable separation**, so
   sparsity has little headroom. (Freezing the reservoir alone lifts retention off the trainable-hidden ~1/N wall —
   but that is the freezing, not the sparsity; frozen dense ties frozen sparse.)

## The verdict as a method-not-capability call (per the LAW)

The k-WTA sparse-gated readout allocation is a valid, neural, load-bearing pattern-separator that does **not**, by
itself, close the sequential-interference wall on this substrate. The residual it maps — a COMMITTED (stable)
engram + a plasticity rule that is local (non-suppressive) yet still calibrated — points the breadth capability at
the OTHER workstream: **self-generated sleep-replay consolidation** (reactivate the brain's own engrams to
interleave internally, no teacher re-presentation), and/or a committed-allocation variant that stabilises winners.
Banked as a characterised negative, not a stop.

## Reproduce

```
# single-seed SMOKE (numpy; fast — tiny launch-bound net):
SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
  python -m research.runners._teacher_loop_sparse_readout_derisk --seed 42 \
    --n-max 10 --milestones 1 5 10 --hidden 64 --kwta-k 6 --epochs 30 --settle-steps 25 \
    --n-draws 20 --k-sweep 64 12 3 --out research/findings/raw/teacher_loop_sparse_readout_s42.json
# 6-SEED (the A/B must hold 6/6 to claim a positive; here it holds 0/1):
SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
  python -m research.runners._teacher_loop_sparse_readout_derisk --seeds 42 43 44 45 46 47 \
    --n-max 10 --milestones 1 5 10 --hidden 64 --kwta-k 6 --epochs 30 --settle-steps 25 --n-draws 20
```
