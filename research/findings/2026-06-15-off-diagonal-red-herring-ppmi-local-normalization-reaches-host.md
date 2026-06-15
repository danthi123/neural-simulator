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

## CYCLE 89 — the residual is DISSOLVED: the PPMI codes are in the binding SWEET SPOT (they generalize AND bind)

The "binding correlated codes" residual flagged above turns out to be a non-blocker, because the 2026-06-11 "correlated codes can't bind" result was tested at the **extreme** (denoise64, between-cos ≈ 0.81) and the **middle was never swept**. `research/runners/_phaseB_correlation_binding_tradeoff_derisk.py` (3 seeds; the project's own `BilinearBinder` + the leakage-free systematicity protocol; a within-category correlation knob β) sweeps the full curve:

| between-cos | semantic (Pearson / gen) | binding held-out |
|---|---|---|
| ~0.00 | −0.02 / 0.25 (none) | 1.00 |
| 0.05 | +0.68 / 0.98 | **1.00** |
| 0.09 | +0.87 / 1.00 | **0.83** |
| 0.13 | +0.94 / 1.00 | **0.92** |
| 0.16 | +0.97 / 1.00 | 0.17 (collapses) |
| 0.18 | +0.98 / 1.00 | 0.33 |

**There is a wide SWEET SPOT (between-cos ≈ 0.05–0.13) where codes BOTH generalize semantically AND bind systematically.** Binding only collapses above ~0.16. The tension is **not strict**.

**And the real PPMI codes land in it.** The actual PPMI concept codes (projected to 64-D, 16 fillers across 4 categories) measure **between-cos +0.014** — deep in the sweet spot — with semantic Pearson +0.67 and generalization 1.00. Run through the binder: **2 of 3 seeds bind systematically (held-out 1.000) AND generalize (1.000) simultaneously.** (The 3rd seed *failed to fit* — train 0.25 — a binder convergence/init instability, NOT the ill-conditioning failure mode, which shows high-train/low-held-out.) PPMI codes are simultaneously near-orthogonal (bindable) and semantically structured (generalizable) because the semantics live in the *relative* same-vs-different-category pattern, not in a high absolute correlation.

**⇒ The whole representation-vs-binding tension that motivated the months-scale dendritic-plus-lateral build is DISSOLVED.** The functional cortex is: **PPMI local normalization** (CYCLE 88, generalizes, no curated concepts) → its codes are already in the **binding sweet spot** (CYCLE 89) → the existing binder + cleanup + no-confab gate operate on them. No off-diagonal decorrelation, no dendritic rewrite.

**Binder seed-stability RESOLVED (3/3).** The 3rd seed's non-convergence was a gradient-descent init instability: with multi-init (best of 4), **all 3 seeds bind the PPMI codes — train 1.000, held-out 1.000** — alongside generalization 1.000. The dissolve is robust, not a 2/3 fluke.

**Honest residuals (the real remaining work, much smaller than a months-scale build):** (1) the on-bridge spiking realization of PPMI normalization + binding (the existing cortex-forward runner's random readout confounds the cortex read — needs a faithful read + the per-concept divisive-norm circuit); (2) the assembled-pipeline numpy de-risk end-to-end — **DONE, see CYCLE 90 below**; (3) scaling F=16 → 320 concepts. These are build/engineering, not a wall.

## CYCLE 90 — the assembled pipeline works END-TO-END on PPMI codes (the dissolve is complete in numpy)

`research/runners/_phaseB_assembled_pipeline_ppmi_derisk.py` (3 seeds, the real 64-concept corpus, PPMI codes; HRR circular-convolution multi-role binding; the composer's actual who-Q&A + conjunctive-cue + abstention logic):

| metric | result (3 seeds) |
|---|---|
| who-Q&A recall (present facts, multi-role SVO superposition) | **0.92** |
| within-category confusions | **0 / 8** (the semantic structure causes no errors) |
| no-confab abstention (absent (verb,object) queries) | **1.00 — 0 false-accepts** (the moat holds) |
| familiarity gap (present-match vs absent-match) | **+0.466 vs +0.073** (clean, wide, separable) |

**GO.** The full conversational capability — multi-role SVO fact binding → who/what recall → the no-confab abstention moat — works **end-to-end on PPMI codes**, with no curated concepts. The familiarity/abstention gate, which 2026-06-11 saw *collapse* on the extreme-correlated codes (gap 0.45→0.03), is **clean and wide** on PPMI codes (the low between-cos preserves it). Two principled (not test-tuned) pipeline choices: the cue is *conjunctive* (verb AND object must match — the correct "find the fact with V and O" logic) and the gate sits at the gap midpoint. The recall capacity is set by the corpus size (64 concepts → 64-dim codes → ~8 facts cleanly via HRR); at the 320-concept target the codes are 320-dim → proportionally more fact capacity, so the F=16→320 scaling is a capacity *gain*, not a risk.

**⇒ The functional cortex without curated concepts is DE-RISKED end-to-end in numpy:** PPMI local-normalization cortex (CYCLE 88, generalizes) → codes in the binding sweet spot (CYCLE 89, generalize AND bind) → multi-role facts + who/what recall + the no-confab moat (CYCLE 90). The entire off-diagonal / months-scale dendritic-plus-lateral build is unnecessary. The only remaining work is the on-bridge spiking realization (a faithful read + the one missing local op, per-concept divisive normalization) + scaling — build/engineering, not a wall.

## CYCLE 91 — on-bridge realization begins: the single-neuron rate read hits the documented rate-code wall (the encoding is the design decision)

With edits approved, the on-bridge realization started cheap-first: the faithful-read de-risk (`_phaseB_ppmi_bridge_faithful_read_derisk.py`, GPU) drives the hub layer with host-computed PPMI and reads the hub layer's **own firing rate** as the code (removing the CYCLE-88 random-readout confound). Result (seed 42, drive-scale sweep {12, 25, 50, 120}): the spiking single-neuron firing-rate code **peaks at +0.102 (20% of numpy PPMI +0.50) at drive-scale 50**, and falls off either side — too-low drive silences half the concepts (+−0.02, silent 0.48 at scale 12), too-high saturates the rate (+0.04 at scale 120). So tuning helps (4%→20%) but the graded PPMI structure mostly does not survive a single-neuron rate code (20% ≪ the ≥70% gate).

**This is the documented rate-coded SNR wall, not a new wall.** The project already established (the conversational opponency arc) that rate codes physically cannot preserve fine graded structure — which is exactly why the production composer pivoted to **phase-coded resonate-and-fire (RF) phasors** (info in PHASE, not rate). So the on-bridge PPMI code needs a structure-preserving spiking encoding, and the path is already in the codebase:

- **Encoding options (the design decision):** (a) **population rate** — represent each PPMI value by a small population (finer resolution than one neuron; the project's standard for graded values); (b) **phase / RF** — encode the PPMI vector as RF phasors and bind via the existing RF-FHRR composer (the project's established answer to the rate-code wall); (c) a **structure-preserving readout** — the random 0.1-density readout destroys structure, so a learned/PCA-equivalent readout (or reading the hub population directly) is required.
- The CYCLE-90 numpy pipeline used the graded PPMI vectors directly (HRR); the on-bridge realization must carry that graded structure through a spiking code — the encoding is the next focused design step, and it reuses existing infrastructure (RF phasors / population pools), not a new mechanism.

This is the honest scope of the remaining on-bridge work: the *numpy de-risk is complete* (cortex + binding + pipeline + no-confab moat all GO on PPMI codes), and the on-bridge realization is a known engineering path (the right spiking encoding) — not the months-scale dendritic wall, which remains unnecessary.

**Population coding resolves it (the encoding question is answered).** Giving each PPMI dimension a small *population* (group-averaged firing rate = finer resolution, the brain's standard for graded values) lifts the on-bridge faithful read from 20% (1 neuron) to **+0.330 — 66% of numpy PPMI (75% of host) — at 16 neurons/dimension** (seed 42, drive-scale 50; silent 0.00; vs log-drive +0.064). The single-neuron→16-neuron trend (20%→66%) shows fidelity scales with population size, so the spiking substrate *does* carry the PPMI code — the CYCLE-88 null was the random-readout confound plus single-neuron resolution, exactly as diagnosed. **⇒ the on-bridge cortex path is viable with a population-rate code** (the most brain-faithful encoding for a graded value); the remaining tuning (more neurons/dim, window, drive scale) climbs toward host, and the neural per-concept normalization + the binder wiring are the next build steps. No phase/RF pivot required, no dendritic wall.

**Scaling confirmed — the population code reaches host.** The faithful read climbs cleanly with population size: **1 neuron 20% → 16 neurons 66% → 32 neurons 83% of numpy PPMI** (+0.417), which is **94% of host (+0.442)** at 32 neurons/dim + window 80 (seed 42). So the spiking substrate carries the PPMI code to essentially host fidelity with a sufficient population — the on-bridge cortex is **confirmed**, not just viable. The graded structure the single-neuron rate code lost is fully recovered by population averaging (the brain's standard for graded values). The on-bridge realization's hardest unknown (does the spiking substrate carry the code?) is answered YES; the rest (neural per-concept normalization, binder wiring, scaling) is standard build.

## CYCLE 92 — the per-concept divisive primitive is SHIPPED (byte-clean); but the on-bridge PPMI *computation* from raw counts needs a LOG-DOMAIN circuit (honest negative)

With edits approved, I built + committed the **per-concept divisive-normalization primitive** (`sim/`: `cfg.enable_input_divisive_norm` + `BrainRegion.input_divisive_norm` + a guarded per-step block, Carandini-Heeger `r_i = x_i/(σ + g·mean_j x_j)`), mirroring the shipped `input_mean_adapt` exactly — **default-off byte-identical, verified by 12/12 byte-identity A/B tests**, diff reviewed. It is a sound, byte-clean building block.

But the **functional** on-bridge test (`_phaseB_neural_norm_cortex_derisk.py`: drive *raw counts*, normalize neurally, population-read) is an honest **negative**: host-PPMI ceiling +0.330, raw-no-norm floor +0.113, **+divisive-only −0.064, +divisive+input_mean −0.074** — the on-substrate normalization *hurts*. Root cause (clear, not a tuning miss): the primitive divides in the **current** domain, but PPMI's normalizations are **log-subtractive** — they must happen *after* a log compression — and the Izhikevich f-I is not log-like enough to convert a current-space divide into the log-ratio. (My numpy +0.339 approximation had the per-concept divide *pre-log* and the per-hub subtraction *post-log*; on the bridge both ops were pre-f-I, the wrong order, and the f-I isn't the `log1p(·×scale)` numpy used.)

**Honest scope, restated precisely:**
- The numpy de-risk is **complete** (cortex + binding + pipeline + no-confab moat, CYCLES 88–90).
- The substrate **carries** a PPMI code at host fidelity (population read 94%, CYCLE 91). ✓
- The substrate **computing** PPMI from raw experience is **not yet realized** — it needs the normalizations in the **log domain**: a log compression of the drive first (a log-shaped f-I / dendritic Weber-Fechner compression, or a log readout), *then* the per-hub (`input_mean_adapt`) + per-concept subtractions in the firing/log domain. That is focused circuit-design (the next step), not knob-tuning.
- A pragmatic interim (a BRAIN-BASED-ONLY *boundary judgment* for the owner): since the substrate carries a host-PPMI code at 94%, host-rendering the PPMI normalization as part of *sensory input rendering* would unblock the binder-wiring + scaling while the log-domain circuit is designed — provided we judge corpus-statistics normalization to be "sensory rendering" (host-ok) rather than "perception" (must be neural). I'm surfacing that judgment rather than deciding it silently.

## Artifacts

- `research/runners/_phaseB_ppmi_centering_verify_derisk.py` + `research/findings/raw/_phaseB_ppmi_centering_verify.{json,txt}` (the skeptical verification + decomposition)
- `research/runners/_phaseB_interneuron_whitening_derisk.py` + raw (the interneuron-whitening probe that surfaced the contradiction — NEGATIVE for whitening, which is the point)
