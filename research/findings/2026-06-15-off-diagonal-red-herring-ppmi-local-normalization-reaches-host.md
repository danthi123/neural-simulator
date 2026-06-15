# The off-diagonal was (largely) a red herring: PPMI — a feedforward LOCAL normalization — reaches host and generalizes on the real corpus, no learned cross-neuron decorrelation needed

**Date:** 2026-06-15
**Cycle:** 88 (autonomous; owner steered toward option (a) — a functional cortex without curated concepts)
**Status:** STRONG numpy finding, multi-seed, skeptically verified. **Needs the on-bridge confirmation before redirecting the build.** Potentially a major simplification of the owner-chosen dendritic-plus-lateral path.

---

## The context

The owner chose option (a): push past the blocker to a functional learned cortex that recovers REAL text-corpus category structure WITHOUT curated concepts. The 8-probe arc (CYCLES 80–87) had concluded the blocker was the **off-diagonal cross-neuron decorrelation** — reachable offline (rank-8 ZCA whitening +0.497) but not by any tested local online rule (the similarity-matching lateral plateaus +0.35 / collapses; the diagonal dendritic gain caps +0.22) — and that the only remaining route was a months-scale dendritic-plus-lateral build.

While de-risking that build, an interneuron-whitening probe surfaced a contradicting number, and a skeptical follow-up confirmed it.

## The finding (multi-seed, skeptically verified)

`research/runners/_phaseB_ppmi_centering_verify_derisk.py` (3 seeds, real corpus, host PPMI+SVD ceiling +0.442):

| representation (cosine vs S_true) | Pearson | note |
|---|---|---|
| raw counts (centered) | +0.166 | — |
| log counts (centered) | +0.179 | **the encoding the prior bridge/_phaseB arc used** |
| **PPMI (uncentered)** | **+0.502** | reaches host |
| **PPMI (centered)** | **+0.518** | = the "offline optimum"; *higher* than the whitened rank-8 ZCA (+0.497) |
| online running-mean centering(PPMI) | +0.510 | ≈ batch → centering is online-local |
| centered-PPMI **held-out generalization** | **0.859** (chance 0.125) | a real structured code, not a similarity artifact |
| permuted-S_true (anti-cheat) | −0.009 | clean |

**PPMI is a feedforward LOCAL normalization** — `PPMI = ReLU( log(count) + log(T) − log(per-concept total) − log(per-hub total) )`:
- `log(count)` = Weber-Fechner / dendritic compression — local.
- `− log(per-hub total)` = per-hub frequency normalization = the running per-hub mean in log space = the **shipped `input_mean_adapt` primitive** — local.
- `− log(per-concept total)` = per-concept divisive normalization (each concept normalized by its own total input) = **Carandini-Heeger divisive normalization** — local (feedforward inhibition / shunting proportional to total drive).
- `ReLU` = the spiking rheobase threshold — local.

No learned cross-neuron weights, no decorrelation, no whitening, no dendrite-plus-lateral. The "off-diagonal wall" was a wall for the **learned-decorrelation** approach (similarity-matching, dendritic lateral) — which is the *wrong* approach: **whitening over-processes** (the SM whitens → +0.35; PPMI just normalizes → +0.52).

## Decomposition (which local operations carry it — honest, corrected)

| PPMI piece | centered | uncentered |
|---|---|---|
| log only | +0.183 | — |
| per-hub normalization only | +0.179 | weak |
| per-concept normalization only | +0.530 | **+0.269** (generalizes 0.672) |
| **full PPMI (both)** | **+0.518** | **+0.502** (generalizes 0.859) |

The per-concept normalization *centered* looked like a single silver bullet (+0.530), but **uncentered it is only +0.269** — so the robust recipe is the **full PPMI (both marginal normalizations)**, not a single one. The per-concept (total-input) normalization is a significant contributor and is exactly the piece the per-hub-focused dendritic D1/D2 mechanism missed (D1's per-hub divisive gain = +0.179). (Corrected from an initial over-read of the centered row-only number — the anti-shortcut rigor check caught it.)

## Why this wasn't seen before (reconciliation)

1. The bridge/_phaseB arc fed **log** input (CYCLE 80 noted SIMMATCH_LOG = +0.088), never the per-concept+per-hub-normalized PPMI. log centered = +0.179; PPMI = +0.502. The encoding was the gap.
2. The arc framed +0.518 as the **offline optimum a learned cortex must converge to**, and chased it with the similarity-matching network (which *whitens*) and the dendritic gain (per-hub only) — both the wrong operation. PPMI is a **feedforward normalization**, not a learned decorrelation; it reaches the optimum directly.
3. The dendritic D1/D2 diagnosis focused on the **per-hub** common-mode (the right intuition, wrong marginal); the **per-concept** divisive normalization is the larger lever and is equally local.

## Is this a "functional cortex without curated concepts"?

Yes, on the owner's standard: it recovers REAL category structure from REAL co-occurrence (no curated concepts), and it *generalizes* (0.86 held-out). The experience-dependent part (the per-hub frequency, learned as a running mean over exposure = `input_mean_adapt`) is genuine learning-from-experience; the rest is standard local feedforward normalization. It is *much* simpler than a learned cross-neuron decorrelator.

## Honest scope + the decisive next test

- This is a **numpy** result. The decisive question is whether the **spiking bridge** realizes PPMI-equivalent drive at host level. The prior bridge reached +0.155 with **log** input + E/I + per-hub centering; the missing piece is the **per-concept divisive normalization** (+ the proper PPMI encoding). The on-bridge test: feed PPMI-shaped drive (log + per-hub `input_mean_adapt` + per-concept total-input divisive normalization + threshold) and measure the cortex-code structure vs host. Spiking realization has losses (the prior E/I+spiking retained ~50% of the numpy diagonal), so the bridge number may be marginal — but the target rises from "an unreachable +0.52 needing months of dendrite-lateral work" to "a local feedforward normalization the substrate already has most of."
- **Recommendation for the owner:** before committing to the months-scale dendritic-plus-lateral off-diagonal build, run the cheap on-bridge PPMI-normalization test. If the bridge reaches host with local normalization, the off-diagonal build is unnecessary; if it falls short, we have a sharp, characterized target (the spiking-realization loss, not a missing mechanism).
## Reconciliation with the off-diagonal mechanism scope (it landed; it strengthens this)

A parallel deep-research scope on the off-diagonal *mechanism* (`2026-06-15-dendritic-predictive-coding-offdiagonal-mechanism-spec.md`) landed simultaneously and **independently identified the same Oja-W-collapse** ("the vicious cycle"). Its fix — error-gated learned decorrelation (Mikulasch-Priesemann / Duong-Lipshutz) — reaches the **whitened** rank-8 ZCA target (+0.49). But two facts make the feedforward-PPMI route superior for the cortex:
1. **The "+0.518 offline optimum" the whole arc chased is `pca_lowrank_sim(Xppmi, 64)` = centered PPMI with ALL components kept — it never whitened.** The arc mis-attributed a *feedforward-normalization* optimum to a *learned-decorrelation* problem. PPMI (+0.518) > the whitened ZCA (+0.49).
2. PPMI is feedforward + local; the decorrelation mechanism is a learned recurrent circuit. The simpler one wins for producing the structured cortex.

So the months-scale dendritic-plus-lateral off-diagonal build is **(largely) unnecessary for the generalizing cortex** — its goal ("a learned, semantically-structured cortex that generalizes", per the D2 decision doc) is reached by PPMI local normalization. The decorrelation mechanism scope is retained as a sound fallback, but it targets a lower ceiling than PPMI.

## The honest residual — the sharpened blocker (do NOT over-claim this solves everything)

PPMI codes are **CORRELATED** — that is *precisely why they generalize* (similar concepts get similar codes). The project's downstream **binder/composer** (the VSA exact-inverse algebra, and the learned binder) was validated on **DECORRELATED** codes; the 2026-06-11 cortex-fork found the learned binder NEGATIVE on correlated codes. So:
- This finding solves the **cortex / representation** blocker (recover a generalizing structure from real experience, no curated concepts) — cheaply, locally, no dendrite.
- It **sharpens, not closes**, the remaining blocker: **binding/composing the correlated generalizing codes.** Whitening/decorrelating the codes for the binder would *destroy* the generalization (that is the fundamental representation-vs-binding tension the project has circled). The genuine open problem is a binder that operates on correlated semantic codes — the real "(B)" frontier, now correctly located *at the binder, not the cortex*.

This is a sharper, more accurate map of the blocker than "the cortex needs the off-diagonal." Recommended sequence for the owner: (1) confirm PPMI normalization on the bridge (cheap, decisive); (2) re-target the deep work at **binding correlated codes**, not at decorrelating the cortex.

## Artifacts

- `research/runners/_phaseB_ppmi_centering_verify_derisk.py` + `research/findings/raw/_phaseB_ppmi_centering_verify.{json,txt}` (the skeptical verification + decomposition)
- `research/runners/_phaseB_interneuron_whitening_derisk.py` + raw (the interneuron-whitening probe that surfaced the contradiction — NEGATIVE for whitening, which is the point)
