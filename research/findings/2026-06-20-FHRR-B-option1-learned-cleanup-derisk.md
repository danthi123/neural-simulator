# FHRR-B Option 1 — learned iterative cleanup over a fixed bind: GO (the bundle-inverse's learnable half generalizes)

**Date:** 2026-06-20
**Runner:** `research/runners/_phaseB_fhrr_b_learned_iterative_cleanup_derisk.py`
**Raw:** `research/findings/raw/_fhrr_b_iter_cleanup_run.log` (gitignored `.log`; data reproduced in the table below)
**Backend:** numpy/CPU (cheap-first; no `sim/` edit)

> **Provenance note:** the de-risk subagent (`a530b1bbfbd2ed7bd`) completed the full D_h capacity sweep × 6
> seeds but then RESTED, and its run process hung ~11 min past the conclusive sweep (no log output) — the
> controller killed the stuck process and wrote this verdict directly from the committed run log. Every number
> below is from the log.

## VERDICT: **GO** — a LEARNED cleanup over a FIXED self-inverse bind recovers HELD-OUT (never-bundled) role-filler bindings from a bundle and GENERALIZES, at sufficient hidden capacity. This overturns the prior "a from-scratch learned bundle-unbind fails" — the *learnable* half of the bundle-inverse (the decomposition/cleanup) is genuinely learnable + generalizing.

**Definitions (plain):** a *bundle* = a superposition of several role-filler bindings into one fact vector
(e.g. agent⊗dog + action⊗run + patient⊗fast). The *bundle-inverse* = recovering one role's filler from that
superposition. The prior finding (`2026-06-16-onsubstrate-learned-binder-single-attr-GO-bundling-NEGATIVE.md`)
showed a from-scratch learned unbind of a bundle fails. Option 1 keeps the **bind** a fixed self-inverse
structure and **learns the cleanup** (a resonator-network-style iterative decomposition: unbind-with-fixed-role
→ project onto the learned codebook → re-estimate) — the part a bundle genuinely *can* have.

## Capacity sweep × 6 seeds (CLEAN decorrelated stream codes, between-cos mean 0.047)

held-out = recovery on NEVER-bundled role-filler combinations (leakage-asserted train/test disjoint).

| D_h | ITER held-out (per seed 42/43/44/100/101/102) | ITER mean | single-pass mean | lesion | perm-role |
|---|---|---|---|---|---|
| 64  | 0.752 / 0.697 / 0.838 / 0.491 / 0.833 / 0.867 | 0.746 | 0.640 | 0.03–0.08 | 0.00–0.08 |
| 128 | 0.944 / 1.000 / 1.000 / 1.000 / 0.735 / 0.667 | 0.891 | 0.804 | 0.03–0.16 | 0.00–0.23 |
| **256** | **1.000 / 1.000 / 1.000 / 1.000 / 1.000 / 1.000** | **1.000** | 0.978 | 0.00–0.21 | 0.00–0.07 |

**Headline:** at D_h=256 the learned iterative cleanup recovers held-out bundle bindings at **1.000 on every
seed**, vs the prior learned NEGATIVES (additive 0.193, learned-linear 0.056, learned-dendritic 0.168) and the
FRLF single-pass base 0.639 — and approaches the fixed-±1 self-inverse ceiling (0.989, the positive control).
The capacity (D_h=256) is the dominant lever; the ITERATIVE form adds robustness — it **rescues the hardest
seeds** where the single pass drops (seed 102: single 0.867 → iter 1.000; seed 100 @ D_h=128: 0.806 → 1.000).

## Anti-cheats (all pass)

| anti-cheat | result | reading |
|---|---|---|
| prior learned NEGATIVE controls (additive / learned-linear / learned-dendritic) | 0.193 / 0.056 / 0.168 — all far below | the harness reproduces the documented negatives; Option 1 genuinely beats them |
| fixed-±1 self-inverse POSITIVE control | 0.989 | the achievable ceiling; Option 1 (1.000 @ D_h=256) reaches it |
| FRLF single-pass baseline | 0.639 | the learned ITERATIVE cleanup beats single-pass, esp. at low capacity / hard seeds |
| **lesion** (scramble the cleanup) | **0.00–0.21 collapse** 6/6 | the learned cleanup is LOAD-BEARING |
| **permuted-role** | **0.00–0.07 collapse** 6/6 | the recovery is role-structured, not a shortcut |
| leakage-asserted held-out (train/test bundle-disjoint) | held-out == generalization | not memorization |
| moat (familiarity gap) | **+0.41 to +0.50** every seed | the no-confab abstention margin is intact (never weakened) |
| decorrelated codes (the separate code axis) | clean stream codes (between-cos 0.047) | the code-correlation wall is a SEPARATE, already-solved axis (PPMI stream cortex) — not conflated |

## Interpretation — what this closes, and the residual

FHRR-B (replace the composer's host-DESIGNED exact-inverse bind algebra with a learned binder) has **shrunk to
two separable pieces**:

1. **The cleanup / decomposition half — NOW GO (this de-risk).** Recovering a binding from a bundle is a
   learnable, generalizing read-out (iterative cleanup + hidden capacity). The host exact-inverse *read-out* is
   replaceable by a learned one. A genuine brain-based reduction.
2. **The bind FORM itself (the fixed self-inverse) — the residual.** The bind primitive stayed fixed here.
   Whether the FORM is *learnable* is the deep open question — pursued next via Option 2 (deep/hidden-layer
   learned binder, the dendrite's legitimate re-entry in hidden layers) and Option 3 (orthogonal per-attribute
   role tags / tensor-product). Per the owner's rule this is NOT classified as a boundary — it is the next
   mechanism to try. The honest alternative the scoping raised, to be resolved empirically by pushing 2/3 (NOT
   assumed): a fixed self-inverse bind may be the *correct biological structural primitive*
   (binding-by-coincidence / dendritic product), in which case closing FHRR-B = learned codes (done) + learned
   cleanup (this GO) + a fixed structural bind.

## Reproduce

```bash
python -m research.runners._phaseB_fhrr_b_learned_iterative_cleanup_derisk \
    --seeds 42,43,44,100,101,102 \
    --out research/findings/raw/_fhrr_b_iter_cleanup_run.log
# capacity sweep D_h ∈ {64,128,256}; see the runner's argparse for the full flag set
```

## Next

- **Option 2** (deep/hidden-layer learned binder — does the bind FORM become learnable with hidden layers +
  e-prop/dendritic credit assignment?).
- **Option 3** (orthogonal per-attribute role tags / TPR — does it dissolve the two-attribute boundary?).
- A spiking confirm of the Option-1 cleanup on the bridge (hand-controlled, not auto-launched).
