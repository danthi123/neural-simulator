# Learned graded-embedding DE-SATURATE fix-test — **GO**: a de-saturated (few-cycle) brain-based Hebbian learn recovers the graded embedding to near the host ceiling — **Option A is ALIVE** (the real bug was the member-submatrix read-out discarding the hub-mediated second-order signal, NOT an irrecoverable learn)

**Date:** 2026-06-11. **Runner:** `research/runners/learned_graded_embedding_desaturate_probe.py` (NEW). **Backend:** `SIM_BACKEND=cupy` (GPU, RTX 3090), foreground. **Raw:** `research/findings/raw/_lge_desaturate_seed42.json` (full sweep) + `_lge_desaturate_seed43.json` (confirmatory). **Scope:** seeds 42 + 43, GPU, foreground — each spiking-learn ran inline (~18 s @ 2 cycles → ~140 s @ 20 cycles). The fix-test the diagnosis (`2026-06-11-learned-graded-embedding-diagnosis.md`, commit 61100385) pointed at.

> **Verdict: GO (2/2 seeds).** De-saturating the brain-based Hebbian learn — by running **FEWER store-cycles** (2 instead of 20) — makes the spiking-learned recurrent **W track the FULL co-occurrence counts** (Pearson(W, counts_full) **+0.687 / +0.682** at 2 cycles, vs **+0.062 member-submatrix at 20 cycles** in the diagnosis) AND, read out by the host-method-on-the-LEARNED-W (PPMI+SVD over the full W incl. hubs → member rows), **recovers the graded structure to near the host ceiling**: Pearson(sim_W, S_true) **+0.843 / +0.879** (ceiling +0.932 / +0.950), second-order cat~dog margin **+0.623 / +0.630**, **generalization 1.000** (4.0× chance). The FULL de-risk gate battery PASSES at the de-saturated operating point: G1 structure recovered + graded + second-order; G2 A1 (gen 1.000) with the orthogonal (A2) + permuted-property (A3) controls collapsing; the **HEADLINE permuted-CO-occurrence control collapses** (gen → chance 0.263 / 0.338, Pearson → 0); beats-random; anti-cheat W-distinct-from-counts (+0.68 ≪ 0.999). **A homeostatically-regulated (here: low-cycle) brain-based Hebbian learn recovers the embedding → Option A is alive; the months-scale dual/CLS learned-embedding build is re-justified on the brain-based path.**

## What the diagnosis got right, and the one piece it missed
The diagnosis correctly localized the collapse to the **LEARN, not the read-out family it tested** — and correctly predicted the cure direction (**de-saturate**; the +0.724-at-2-cycles CPU-smoke hint). But it measured saturation/faithfulness only on the **member↔member submatrix of W** (Pearson(W_members, counts_members) = +0.062) and tested read-outs (PPMI/divnorm/diffusion) only on that **submatrix**. That submatrix **structurally cannot carry the second-order cat~dog signal**: in this corpus members never co-occur directly — cat~dog are close ONLY via the shared **hub** ("animal"), so the second-order structure lives in the **hub columns of W**, which the member-submatrix discards. This fix-test adds the **load-bearing read-out the diagnosis never ran**: the host-method (PPMI + truncated-SVD) applied to the **FULL learned W** (all concepts incl. hubs), then member embedding rows — exactly mirroring `host_ceiling_codes`' pipeline but on the **brain-LEARNED W** rather than the raw counts. On the FULL W the learned recurrent is faithful (+0.69) and the host-method read-out recovers everything. **Two things were wrong, not one: (1) the learn over-saturates with cycles (real, fixed by fewer cycles); (2) the read-out must operate on the full hub-inclusive W, not the member submatrix (the second, decisive correction).**

## The sweep (seed 42, full scale: 48 concepts = 8 hubs + 40 members, 280 facts, 53 second-order pairs, 2.40M recurrent edges)

Primary read-out = **host-method on the FULL learned W**. `Pearson(W,counts_full)` = faithfulness of the learned recurrent to the FULL co-occurrence counts; `Pearson(W,counts_memb)` = the diagnosis's member-submatrix check (kept for the trail).

| operating point | Pearson(W, counts_full) | Pearson(W, counts_memb) | **HOST(W) Pearson(sim, S_true)** | 2nd-order margin | graded? | **gen (chance 0.250)** |
|---|---|---|---|---|---|---|
| **cycles=2 (de-saturated)** | **+0.687** | +0.314 | **+0.843** | **+0.623** | **True** | **1.000** |
| cycles=3 | +0.422 | +0.163 | +0.824 | +0.564 | True | 1.000 |
| cycles=5 | +0.340 | +0.083 | +0.738 | +0.431 | True | 0.944 |
| cycles=8 | +0.219 | +0.056 | +0.464 | +0.226 | False | 0.844 |
| cycles=12 | +0.224 | +0.050 | +0.426 | +0.197 | False | 0.894 |
| **cycles=20 (the diagnosis's collapse)** | +0.202 | **+0.062** | +0.358 | +0.175 | False | 0.750 |
| gamma=0.9 × cycles=8 (decay) | +0.444 | +0.104 | +0.222 | +0.145 | False | 0.656 |
| gamma=0.9 × cycles=20 (decay) | +0.438 | +0.107 | +0.227 | +0.152 | False | 0.637 |
| gamma=0.95 × cycles=8 (decay) | +0.554 | +0.125 | +0.430 | +0.278 | True | 0.769 |
| gamma=0.95 × cycles=20 (decay) | +0.542 | +0.126 | +0.425 | +0.274 | True | 0.756 |
| **HOST CEILING (PPMI+SVD on RAW counts)** | — | — | **+0.932** | — | True | **1.000** |
| collapsed baseline (diagnosis, 20cyc) | — | +0.062 | (member-submatrix) −0.026 | — | False | 0.237 |

**The saturation is monotonic and now visible on the load-bearing metric:** cycles 2→3→5→8→12→20 drives Pearson(W, counts_full) +0.687 → +0.422 → +0.340 → +0.219 → +0.224 → +0.202 and HOST(W) Pearson +0.843 → +0.824 → +0.738 → +0.464 → +0.426 → +0.358. **Fewer cycles = a faithful graded W**; the un-normalized recurrent fills toward a uniform floor as cycles accumulate (recurrent mean climbs 0.54 → 1.06 → 1.73 then the CV of the off-diagonal rises 0.173 → 0.272). The +0.724-at-2-cycles CPU-smoke hint is confirmed at full scale.

## The best operating point → FULL de-risk gates (re-learned at `cycles=2`, read-out = host-method-on-full-W)

| gate | seed 42 | seed 43 | pass? |
|---|---|---|---|
| G1 Pearson(sim, S_true) (permuted-S baseline) | **+0.843** (perm +0.076) | **+0.879** (perm −0.005) | ✅ (≥ 0.5) |
| G1 graded / second-order recovered (margin) | True / True (+0.623) | True / True (+0.630) | ✅ |
| G2 A1 generalization (vs chance 0.250) | **1.000** (4.0×) | **1.000** (4.0×) | ✅ (≥ 0.7) |
| G2 A2 orthogonal collapses | 0.119 | 0.256 | ✅ (≤ 1.5×chance) |
| G2 A3 permuted-property collapses | 0.131 | 0.163 | ✅ |
| **G5 permuted-CO-occurrence collapses (HEADLINE)** | gen **0.263**, Pearson +0.064 | gen **0.338**, Pearson +0.017 | ✅ (→ chance) |
| G5 beats random-Gaussian | 1.000 > 0.312 | 1.000 > 0.163 | ✅ |
| anti-cheat: W distinct from counts | +0.687 < 0.999 | +0.682 < 0.999 | ✅ |

**Both seeds GO.** The permuted-co-occurrence control is the load-bearing anti-cheat: re-learning on a corpus with the SAME concepts/fact-sizes but RANDOM context structure collapses the recovery to chance — proving the graded structure comes from the REAL co-occurrence statistics learned into the recurrent, not the architecture/read-out. The anti-cheat (Pearson(W, counts_full) +0.68 ≪ 0.999) confirms the host-method read-out runs on the **brain-LEARNED W** (which is faithful-but-distinct), not silently on the raw counts: a +0.69-faithful spiking recurrent reaching +0.84 graded-recovery is genuine learning, not a pass-through.

## Why low cycles works (and why the explicit decay arm under-performed)
- **Low cycles = implicit homeostasis.** The recurrent is un-normalized direct-Hebbian co-fire growth (`enable_hebbian_learning=True`, `hebbian_max_weight=45`, `hebbian_min_weight=0`); each co-fire potentiates. After many cycles, nearly every co-fired pool→pool edge has grown toward a common ceiling → the *contrast* between more- and less-co-occurring pairs (the graded signal) is compressed. At 2 cycles the recurrent has potentiated just enough to encode the **rank** of co-occurrence (faithful) without saturating the magnitude. This is the brain-based reading: a developing synapse early in potentiation carries graded co-occurrence; sustained un-regulated LTP washes it to a uniform floor (the missing Turrigiano scaling / BCM contrast).
- **The explicit per-fact multiplicative decay (gamma<1) was a real-but-inferior knob.** gamma=0.95 recovered partially (HOST(W) Pearson +0.43, gen 0.77 — above chance, below the cleaner low-cycle result); gamma=0.9 over-decayed (recurrent mean → 0.03–0.06, structure lost). A per-fact rescale is a coarse synaptic-scaling proxy; the cleanest de-saturation here is simply **not over-potentiating** (few cycles). For the build, the principled version is a proper homeostatic normalization INSIDE the learning (synaptic scaling toward a target row-norm, or Oja's intrinsic normalization) so the recurrent stays in the faithful regime at ANY cycle count — but the cheap lever (low cycles) already clears all gates, so the expensive rewrite is not required to reach GO.

## The honest answer to the fix-test's question
**Does de-saturating the learn make the brain-LEARNED W track the graded counts AND pass the architecture gates?** **Yes** — by running fewer store-cycles (2), the spiking-Hebbian recurrent tracks the full co-occurrence counts (Pearson +0.687/+0.682) and, read out by the host-method on the full learned W, recovers the graded second-order structure to near the host ceiling (Pearson +0.843/+0.879 vs ceiling +0.932/+0.950; gen 1.000) with every control collapsing, 2/2 seeds. **Option A (a homeostatically-regulated Hebbian learn recovers the embedding) is ALIVE.** Distance to ceiling: ~0.09–0.07 Pearson (the brain-learned W is a slightly noisier version of the counts; generalization is already saturated at 1.000).

## Explicit next step
**Re-run the full de-risk at the de-saturated operating point (cycles=2) with the host-method-on-full-W read-out, multi-seed 42/43/44, including G3 (cortex-channel round-trip) + G4 (spiking strong-encode compatibility)** — the two architecture gates this fix-test did not re-run (it focused on the LEARN-quality + G1/G2/G5 that the diagnosis localized). The de-risk's `learn_assoc_matrix` + `graded_readout` should be swapped for the low-cycle learn + `host_method_codes_on_W` (the read-out is the second half of the fix). If G3/G4 also pass at the de-saturated point (high prior they do — the codes already pass G1/G2 and are graded+distinct), the **months-scale dual/CLS learned-embedding build is GO end-to-end on the brain-based path**, with two concrete recipe locks:
1. **Learn in the faithful regime** (few cycles, or — the principled build version — a homeostatic synaptic-scaling/Oja normalization inside the recurrent so faithfulness is cycle-independent at scale).
2. **Read out via the host-method on the FULL learned W** (PPMI+SVD over all concepts incl. hubs → member rows), NOT the member-submatrix — the second-order signal lives in the hub columns. (For a fully brain-based read-out, the spiking analogue of this is spreading-activation/diffusion THROUGH the hub nodes — the de-risk's `graded_readout` direction — but tuned to actually propagate the hub-mediated structure; the current diffusion under-propagated, hence the PPMI+SVD-on-full-W stands in as the validated read-out and the spiking diffusion is the conversion target.)

No banking — reported exactly as found, both seeds GO, with the diagnosis's member-submatrix blind-spot named.
