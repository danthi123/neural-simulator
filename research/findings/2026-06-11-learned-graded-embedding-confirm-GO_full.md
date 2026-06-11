# Learned graded-embedding CONFIRMATION — **GO_full**: the de-saturated brain-based learn passes the COMPLETE de-risk battery (G1–G4) end-to-end, multi-seed; the brain-based diffusion read-out partially recovers but stays a documented stand-in (a build-time refinement)

**Date:** 2026-06-11. **Runner:** `research/runners/learned_graded_embedding_confirm_probe.py` (NEW). **Backend:** `SIM_BACKEND=cupy` (GPU, RTX 3090), **foreground/synchronous** (multi-seed run 192.2 s inline; each cycles=2 spiking-learn ~16–19 s + the permuted-co-occurrence re-learn + the spiking G4 encode per seed). **Raw:** `research/findings/raw/_lge_confirm_multiseed.json` (GPU, seeds 42/43/44, all gates + G4) + `research/findings/raw/_lge_confirm_smoke.json` (numpy harness-wiring smoke, seed 42). **Scope:** the FOCUSED confirmation the desaturate-GO finding (`2026-06-11-learned-graded-embedding-desaturate-GO.md`, commit 1febdd20) explicitly queued — it validated the LEARN-quality + G1/G2/G5 but did NOT re-run G3/G4 and the recovering read-out was the host stand-in.

> **Verdict: GO_full (3/3 seeds 42/43/44).** At the recovered recipe (**cycles=2** de-saturated learn + read-out = **host-method PPMI+SVD on the FULL hub-inclusive learned W**), the COMPLETE de-risk battery PASSES on the brain-LEARNED graded codes, multi-seed: **G1** structure recovered (Pearson(sim, S_true) **+0.843 / +0.879 / +0.881**; ceiling +0.932 / +0.950 / +0.941; second-order cat~dog margin +0.62 / +0.63 / +0.65; graded True); **G2** generalization **1.000** (4.0× chance) with orthogonal (A2) + permuted-property (A3) controls collapsing; **G3** cortex-channel round-trip closes (Pearson **+1.000**, permuted ≈0, binding identity **1.000**); **G4** spiking strong-encode compatible (DG between-cos −0.0025 / −0.0009 / +0.0019 → decorrelated, repro 1.000 → graded-cortex + decorrelated-DG COEXIST); **G5** permuted-CO-occurrence control collapses (gen → chance 0.263 / 0.338 / 0.175, Pearson → 0); beats-random; anti-cheat W-distinct-from-counts (+0.69 ≪ 0.999). **The dual/CLS learned-embedding works END-TO-END on the brain-based path → the months-scale build is justified end-to-end.** The **brain-based diffusion read-out** partially recovers (best, full-column variant, Pearson **+0.49 / +0.54 / +0.63**, generalization **1.000** all seeds) but does NOT clear the G1 cosine-structure bar (second-order margin ~+0.04 vs the +0.10 bar) → it **stays a documented stand-in**; the host-method-on-W is the validated read-out and the spiking diffusion is a **build-time refinement**.

## What this confirmation closed (the two pieces the desaturate fix-test left open)
The desaturate GO validated the LEARN-quality (Pearson(W, counts) tracking) + G1/G2/G5 with the host-method-on-W read-out. It did **NOT**:
- **(a)** re-run **G3** (the cortex-channel round-trip) or **G4** (the spiking strong-encode compatibility) on the LEARNED graded codes — those had passed only on SYNTHETIC graded codes in the architecture-proof (commit 343c721d). This confirmation runs them on the **brain-LEARNED** graded codes → **both pass, 3/3**.
- **(b)** test whether a genuinely **BRAIN-BASED** read-out can replace the host PPMI+SVD stand-in — the diffusion the desaturate fix shipped UNDER-PROPAGATED. This confirmation sweeps a tuned spreading-activation diffusion **through the hubs** (steps × alpha × {member-column, FULL-column}) → it **partially recovers but does not close**; characterized below.

## PRIMARY read-out (host-method PPMI+SVD on the FULL learned W) — the full gate battery, multi-seed

| gate | seed 42 | seed 43 | seed 44 | pass (all seeds)? |
|---|---|---|---|---|
| **G1** Pearson(sim, S_true) (permuted-S baseline) | **+0.843** (perm +0.076) | **+0.879** (perm −0.005) | **+0.881** (perm −0.046) | ✅ (≥ 0.5) |
| **G1** graded / 2nd-order recovered (margin) | True / True (+0.623) | True / True (+0.630) | True / True (+0.650) | ✅ |
| **G2 A1** generalization (vs chance 0.250) | **1.000** (4.0×) | **1.000** (4.0×) | **1.000** (4.0×) | ✅ (≥ 0.7) |
| **G2 A2** orthogonal collapses | 0.119 | 0.256 | 0.237 | ✅ (≤ 1.5×chance) |
| **G2 A3** permuted-property collapses | 0.131 | 0.163 | 0.212 | ✅ |
| **G3** cortex-channel round-trip (permuted) | **+1.000** (perm +0.006) | **+1.000** (perm −0.035) | **+1.000** (perm −0.023) | ✅ (≥ 0.7, ≫ permuted) |
| **G3** binding identity | 1.000 | 1.000 | 1.000 | ✅ (≥ 0.9) |
| **G4** DG decorrelation (between-cos) | −0.0025 | −0.0009 | +0.0019 | ✅ (≤ 0.10) |
| **G4** DG reproducibility | 1.000 | 1.000 | 1.000 | ✅ (≥ 0.90) |
| **G4** graded-cortex + decorrelated-DG COEXIST | True | True | True | ✅ |
| **G5** permuted-CO-occurrence collapses (HEADLINE) | gen 0.263, P +0.064 | gen 0.338, P +0.017 | gen 0.175, P −0.008 | ✅ (→ chance) |
| **G5** beats random-Gaussian | 1.000 > 0.312 | 1.000 > 0.169 | 1.000 > 0.169 | ✅ |
| anti-cheat: Pearson(W, raw_counts_full) | +0.687 | +0.682 | +0.706 | ✅ (< 0.999, distinct) |
| **per-seed verdict** | **GO_full** | **GO_full** | **GO_full** | **GO_full 3/3** |

Toy corpus: 48 concepts (8 hubs + 40 members), 280 facts, 53 second-order pairs (cat~dog members that never directly co-occur — close ONLY via the shared hub). Learner: `LearnedAssocGraph` (n_pool=2000, pattern_size=100), 2.76M recurrent edges, cycles=2. G4 spiking encoder: `StrongDGEncoder` (DG=600, CA3=300, drive 800 pA, k=40).

**Both the round-trip (G3) and the strong-encode (G4) close on the brain-learned graded codes exactly as they did on synthetic graded codes.** G3 reaches a perfect Pearson +1.000 (the cortex code is cleanly reinstated after the spiking-Hopfield identity recall; the permuted baseline ≈0 confirms it's the recovered identity, not the architecture). G4 confirms the strong-vs-graded tension is resolved as the dual design predicted: the DG decorrelates ANY input (between-cos ≈0) and the encode is perfectly reproducible (1.000) **regardless of the graded-ness of the cortex code driving it** — graded-cortex and decorrelated-DG coexist as linked populations.

## BRAIN-BASED diffusion read-out (close the stand-in residual?) — partial recovery, stays a stand-in

On the SAME learned W (no host PPMI+SVD), a tuned spreading-activation diffusion through the hubs was swept: diffusion steps ∈ {2,3,4,6} × alpha ∈ {0.5,0.7,0.9} × {member-column, FULL-column (hubs included)} variants.

| best per seed | read-out | Pearson(sim, S_true) | generalization | 2nd-order margin | passes G1+A1? |
|---|---|---|---|---|---|
| seed 42 | `steps2_alpha0.5_fullcols` | +0.489 | **1.000** | +0.043 | ❌ (G1=False, A1=True) |
| seed 43 | `steps2_alpha0.5_fullcols` | +0.543 | **1.000** | +0.035 | ❌ (G1=False, A1=True) |
| seed 44 | `steps2_alpha0.5_fullcols` | +0.634 | **1.000** | +0.038 | ❌ (G1=False, A1=True) |

- **The FULL-COLUMN variant is decisive** (and confirms the diagnosis's hub-column insight independently): reading each member's diffused association profile over **ALL columns incl. the hubs** gives Pearson +0.49…+0.63, while the **member-column** restriction (the desaturate fix's `graded_readout`) *collapses to negative Pearson* (−0.10…−0.23 at low alpha) — because the cat~dog second-order signal lives in the hub columns, which the member-submatrix discards. The hub-inclusive read is the right brain-based direction.
- **But the diffusion read-out plateaus below the G1 cosine-structure bar.** Generalization is already saturated at **1.000** (the property-inference task is solved — the diffused codes are functionally sufficient), but the second-order cosine **margin** (~+0.04) does not clear the +0.10 graded bar, so G1 (Pearson ≥ 0.5 AND graded AND second-order-recovered) is not met. Diffusion smooths the association profile but does NOT sharpen the within-vs-between *contrast* the way PPMI's marginal-division + SVD does (PPMI removes the high-frequency-hub common-mode; raw diffusion leaves it in, compressing the contrast). Gap to the host-method-on-W stand-in: +0.25…+0.35 Pearson; to the ceiling: +0.31…+0.44.
- **Conclusion:** the brain-based diffusion read-out **partially recovers (it generalizes perfectly) but does NOT close the cosine-structure residual** → the host-method PPMI+SVD-on-W **stays the validated read-out (a documented stand-in)**, and the fully-spiking read-out is a **build-time refinement**. The principled spiking analogue of PPMI's marginal-division is divisive normalization (Carandini–Heeger) on the recurrent BEFORE the spreading read — i.e. a diffusion that first removes the hub common-mode (the missing step). The diagnosis already showed `divnorm`/`PPMI`-on-W recovers on the saturated W; combining divisive-normalization with the full-column spreading read is the concrete brain-based read-out to build/test next, but it is NOT required to reach GO (the host-method-on-W stand-in clears everything).

## Decision logic (as specified)
- **GO_full** if all gates G1–G4 pass at cycles=2 multi-seed with the host-method-on-W read-out → **MET (3/3 seeds).** The dual/CLS learned-embedding works END-TO-END on the brain-based path.
- **BOUNDARY** only if G3/G4 failed on the learned graded codes → **NOT triggered** (both passed cleanly, 3/3).

**⇒ GO_full.** The months-scale dual/CLS learned-embedding build is end-to-end justified on the brain-based path. The read-out status: **host-method-on-W is the validated read-out (a documented stand-in)**; the brain-based spreading-activation diffusion is **a build-time refinement** (it generalizes perfectly but needs a divisive-normalization step to sharpen the cosine structure to the G1 bar).

## Anti-cheat / honesty ledger
- **The read-out runs on the brain-LEARNED W, not the raw counts.** Pearson(W, raw_counts_full) = +0.687 / +0.682 / +0.706 ≪ 0.999 (3/3): a +0.69-faithful spiking recurrent reaching +0.84…+0.88 graded recovery is genuine learning, not a pass-through of the host ceiling.
- **The HEADLINE permuted-CO-occurrence control collapses** (re-learn at cycles=2 on a corpus with the SAME concepts/fact-sizes but RANDOM context → gen to chance, Pearson to 0, not graded) — the graded structure comes from the REAL co-occurrence statistics learned into the recurrent.
- **G2/G3/G4 with their existing controls all collapse** (orthogonal A2, permuted-property A3, G3 permuted-S baseline ≈0, G4 driven on the assigned ensembles independent of the cortex code).
- **The host PPMI+SVD on RAW counts is the labelled CEILING ONLY** (+0.932/+0.950/+0.941); it is never the deliverable.
- **Distance to ceiling:** ~0.07–0.10 Pearson (the brain-learned W is a slightly noisier version of the counts; generalization is already saturated at 1.000).
- **No banking beyond what was measured.** GPU, foreground, 3 seeds, all gates including the spiking G4. The brain-based diffusion's failure to close the cosine-structure bar is reported as found (it does generalize perfectly — that nuance is stated, not hidden).

## Explicit next step (the build is end-to-end justified → scope it)
The months-scale dual/CLS learned-embedding build is GO end-to-end on the brain-based path, with the read-out status precisely characterized. Two recipe locks (from the desaturate GO, now confirmed with G3/G4):
1. **Learn in the faithful regime** — few cycles (cycles=2 here), or — the principled build version — a homeostatic synaptic-scaling / Oja normalization INSIDE the recurrent so faithfulness is cycle-independent at scale (so the build doesn't depend on hand-picking a cycle count).
2. **Read out via the host-method on the FULL hub-inclusive learned W** (PPMI+SVD over all concepts incl. hubs → member rows) as the **validated stand-in**, with the **fully-spiking read-out** (divisive-normalization-then-spreading-activation through the hubs — the missing common-mode-removal step the raw diffusion lacks) as the **build-time refinement to convert the read-out to brain-based** (it already generalizes 1.000; it needs the contrast-sharpening step to clear the G1 cosine bar).

Build scope: (i) the homeostatic recurrent (synaptic scaling / Oja) for cycle-independent faithfulness at full vocab; (ii) the divisive-normalization spreading read-out to retire the host stand-in; (iii) scale the toy corpus to the production concept set and re-confirm G1–G4. The architecture (generalization + cortex-channel + strong-encode) is now confirmed end-to-end on the brain-learned graded codes — the open work is the two read-out/learn refinements + scale, not the architecture.
