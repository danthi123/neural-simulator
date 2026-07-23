# Perception closure scoping — what it takes to move perception 🟨 → done, and where the honest boundary is

**Date:** 2026-07-23
**Type:** READ-ONLY scoping (this doc is the only write). No code edited, no build, no GPU job.
**Verdict up front:** The perception **front end** (retina → Gabor V1 → phase-pooled complex) is validated and
richly used; the deployed **learned ventral hierarchy** (`cortex_v2` → `cortex_it` STDP) is *unvalidated as feature
learning* and its navigation contribution is *unquantified* (the Cluster-K-v2 headline ran with the heuristic ON).
Perception is a **supporting system, off the current critical path** — the right target is a **SCOPED closure
(≈ measurements + two already-de-risked lifts, likely no `sim/` edit)** plus an **honestly-marked boundary** (rich
natural-image object recognition / IT invariance-at-scale), **NOT** an open-ended deep-vision build.

---

## 1. DIAGNOSIS — the perception stack as it actually is, per layer

Source: `sim/visual_cortex.py`, `research/runners/g11_bg_runner.py` (~2690–2930, 4690), `tests/test_visual_cortex.py`,
and the perception findings (Cluster-K-v2, genfrontier Option-B, EMERGE-34/36/53, B1, N2/N7, 2026-07-11).

| Layer | What it is in code | On-substrate vs host | Validated? |
|---|---|---|---|
| **Retina** (2×32×32 = 2048 ON/OFF) | `render_gridworld_to_image` (nav) / `build_shape_set` (objects) paints the world/object into an ON/OFF image → `image_to_retina_drive` → retina region ext-drive | **Host render = legitimate** (environment rendering the sensory input; N2 verdict DEFENSIBLE) | Unit: shape/layout tests |
| **V1 simple** (8 ori × 4 freq × 16×16 = 8192) | `build_v1_simple_weights` + `apply_v1_gabor_weights` overwrite the `retina→cortex_v1_simple` weights with **fixed Gabor** — 32 unique templates × 256 positions (527K values). Pathway is declared `plastic=True`/gated `visual_cortex_v1` but deployment overwrites and **never refines**. | **Op on-substrate** (synaptic matvec → V1 spikes); **STRUCTURE host-designed** (the Gabor formula = the B1 criterion-2 residual) | Unit: orientation tuning, forward-fires (`test_v1_orientation_tuning_after_gabor_init`, `test_visual_cortex_neurons_fire_when_retina_driven`). Self-org **de-risked GO in numpy** (B1); **on-bridge lift NOT done** |
| **V1 complex** (8 × 16×16) | fixed phase/frequency pooling → position/phase-invariant | On-substrate, **fixed** | By construction |
| **V2** (256, plastic recurrent) | `cortex_v1_complex → cortex_v2`, STDP, gate `visual_cortex_v2` "always open" | On-substrate STDP | **NOT validated as feature learning** |
| **IT** (`cortex_it`, 64/128, plastic recurrent) | `cortex_v2 → cortex_it`, STDP, gate `visual_cortex_it` "always open"; feeds ventral "what" readouts (`cortex_it→striosome_value` critic, `cortex_it→language_output` grounding) | On-substrate STDP | **NOT validated as feature learning** |
| **V4** | skipped (no color in gridworld) | — | N/A (deliberate) |

**Where the "what/where" split lives (it is genuinely wired):**
- **Dorsal "where"** = `sc_retina` → `sc_map` (spiking superior-colliculus retinotopic orienting map; pop-vector/divnorm/WTA
  → `cortex_{N,E,S,W}`). This path is the *more developed / de-risked* one (N1 spiking SC).
- **Ventral "what"** = retina → V1 → V2 → `cortex_it` (object code feeding the value critic + the language/grounding readout).
  This is the *under-developed* half — its object-recognition (V2/IT) is exactly the unvalidated part.

**The load-bearing honest finding — the deployed V2/IT is NOT what the validated perception work uses.** Every validated
"perception-grounded" result — genfrontier Option-B (`2026-06-16`), EMERGE-34/36/53 (`2026-07-02`), the fully-spiking
codon (`2026-07-11`) — runs **V1(Gabor) → V1-complex(phase-pool)** as a *rate reference front end*, then a **separate
competitive pooler / Marr-Albus coincidence codon** for category discovery. **None of them uses the deployed
`cortex_v2`→`cortex_it` STDP regions.** So the validated ventral representation is the *pooler codon over V1 features*,
not the deployed learned hierarchy. The deployed V2/IT STDP is orphaned + unproven.

**What "partial" (🟨) actually refers to — five distinct residuals, not one:**
1. **V1 structure is host-designed** (B1) — de-risked GO in numpy (mechanism A learned bank OSI 1.0 / RSA-to-host 0.988;
   mechanism B dev-random 0.97); the on-bridge realization is not done.
2. **V1 front end is a RATE reference** in the validated perception uses — a fully-spiking Gabor V1 is not deployed in
   those consoles (EMERGE-36 spiking-codon-izes the *pooler*, not V1).
3. **The deployed learned ventral hierarchy (V2/IT STDP) is unvalidated** as feature learning and possibly inert.
4. **Navigation contribution is unquantified.** The Cluster-K-v2 headline (2.97 ± 0.12) was **measured with
   `--heuristic-strength 1.0` ON** (2026-07-16 correction: the flag that zeros the heuristic was absent from the run's
   own recorded command); the visual pathway's independent nav contribution is unknown, and it *may* have been silently
   inert via `except KeyError: pass  # Gate not present (no IT→cortex synapses)`.
5. **Object recognition is trivial** — a single clean bright blob (nav) or well-separated oriented bars (categories); no
   clutter, occlusion, multi-object, scale/rotation, or natural images.

**What is ALREADY adequate (do not re-litigate):** N2 (goal rendered into the retina) and N7 (innate Gabor V1) are both
**DEFENSIBLE** under the brain-based bar (legitimate environment render + innate-V1 synaptic prior; the strict-bar
enrichment is the L.05 retinal-wave developmental version = residual #1). And the *capability that matters for the
mission* — **SEE an object → discover its visual category → reason/talk about a held-out perceived object** — is already
demonstrated end-to-end, on the spiking bridge, 6-seed, with per-image-scramble + RSA anti-cheats (EMERGE-34/36/53,
2026-07-11). Perception already serves its two real uses.

---

## 2. What "CLOSED" could mean, at two levels

### (a) SCOPED closure — adequate for the current uses + an honest boundary (THE PRAGMATIC TARGET)

"Perception is done *for what the project needs it for*": (i) a navigation cue the neural visual hierarchy perceives,
and (ii) an object-grounding front end whose codes carry visual-category similarity for the emergent-semantics arc.
Both are already largely served. Scoped closure = discharge the *tractable* residuals and *honestly mark* the rest:
- Realize **B1 on-bridge** (self-organized/dev-random V1 RF bank replacing the host Gabor formula) — residual #1.
- **Validate-or-retire V2/IT** — either prove the deployed ventral hierarchy learns invariant category features on-bridge,
  or honestly retire it to "unvalidated/inert" and standardize the grounding path on the validated V1→pooler codon
  (residual #3).
- **Quantify the nav contribution** (one heuristic-off run) — residual #4.
- Optionally couple the **spiking V1** into the perception-grounded console (residual #2).
- **Mark residual #5 (rich object recognition) as the deliberate deferred boundary.**

**Cost:** modest — mostly measurements + two *already-de-risked* lifts, reuse-by-import, **likely no `sim/` edit**.
Estimate ~3–6 focused runner-days of build + a handful of GPU-hours, cleanly parallelizable.

### (b) FULLER learned ventral hierarchy (THE BIGGER BUILD)

A genuinely trained V1→V2→IT recognizing richer objects — clutter, occlusion, multi-object, position/scale/rotation
invariance — trained on natural-image patches (CIFAR/ImageNet), producing Tanaka IT object-identity tuning, DiCarlo
"untangled" invariant manifolds, an HMAX-style S/C hierarchy (Riesenhuber-Poggio). This is the "real visual scene
understanding" the Cluster-K-v2 finding speculated about.

**Cost:** weeks-to-months and a *new research problem*: it needs a training corpus, and it needs **on-bridge STDP
feature learning at scale that the project has never validated** (V2/IT feature emergence is exactly the unproven piece).
It is off the critical path — the mission (fluent grounded conversation + the 5-gap cluster) does not need rich vision;
it needs *category-structured object codes*, which the V1→pooler stack already supplies.

---

## 3. RANKED cheap-first steps for the scoped closure (reuse existing machinery)

Ordered by information-per-GPU-minute, cheapest/most-decisive first.

1. **Quantify the nav contribution (1 run, measurement).** Re-run the Cluster-K-v2 config at
   `--heuristic-strength 0` (+ log stdout, which the original k_v2 artifacts lacked) to separate "the visual cortex
   navigates" from "the heuristic navigates," and to confirm the IT→cortex synapses are present (not silently inert via
   the `except KeyError: pass` path). **This is the single cheapest move and it resolves residual #4 outright.**
   Reuses: the existing `--enable-visual-cortex` stack. No new code.
2. **Lift B1 to the bridge (already-GO de-risk → on-substrate).** Replace the `apply_v1_gabor_weights` overwrite
   (`g11_bg_runner.py:4690`) with either (A) open `visual_cortex_v1`, drive retinal-wave-like patterned input, refine
   `retina→cortex_v1_simple` with the existing on-bridge **rate-Hebbian + homeostasis** rule (NOT symmetric STDP,
   per CYCLE-95), then freeze the gate; or (B) inject a `devrandom_rf_bank(seed)` through the same
   `set_pathway_weights` API. Score the learned V1 **firing** codes vs the host bank. Residual #1; **no `sim/` edit
   expected** (the pathway is already `plastic=True`/gated; the numpy ceiling is the B1 GO de-risk).
3. **Validate-or-retire V2/IT (the true "partial").** A focused on-bridge de-risk: do the deployed `cortex_v2`/`cortex_it`
   STDP regions produce **position-invariant, category-selective** codes when the visual hierarchy is exposed to the
   Option-B / EMERGE-34 object set? If GO → the deployed ventral hierarchy is validated as a learned "what" stream.
   If NEGATIVE → honestly retire cortex_v2/it STDP to "unvalidated/inert" and standardize the grounding path on the
   validated **V1→pooler codon** (what EMERGE already uses). Either outcome closes residual #3 with an honest verdict.
4. **Couple the spiking V1 into the perception-grounded console** (residual #2). EMERGE-36 already has the spiking
   Marr codon; wire the spiking Gabor V1 front end (from step 2's bridge-lift) into the SEE→discover→talk console so
   the whole path is spiking end-to-end incl. V1. Reuse-by-import.
5. **(Optional, bridges toward (b)) richer objects** — render multi-pixel textured objects among distractors to stress
   V2/IT recognition. This is a *capability addition*, not a cheat removal; it is the entry point to the fuller build
   and should be gated behind a fresh decision (perception re-entering the critical path).

---

## 4. Recommended CHEAP-FIRST de-risk + its anti-cheat controls

**Recommended de-risk = step 3 (validate-or-retire V2/IT on-bridge)** — because that is *the actual "partial"*: the front
end is already validated and the nav-measurement (step 1) is one run, but the deployed learned ventral hierarchy is the
unproven, orphaned piece that decides whether the deployed "what" stream is real or should be standardized onto the
pooler. It reuses the exact anti-cheat kit the EMERGE perception work already validated:

- **Lesion (mechanism-ablation):** silence `cortex_v2`/`cortex_it` (or turn coincidence off) → held-out
  category-inheritance / IT-code separation collapses to chance. (EMERGE dAP-lesion → 0.00.)
- **Per-image pixel scramble (input-destruction, LOAD-BEARING):** shuffle each object's pixels independently → destroys
  within-category *visual* similarity → IT category structure collapses (intact 1.00 vs scramble ~0.56 in EMERGE-34/36).
  This isolates the *visual shape* as the cause, not an injected label.
- **RSA pixel-provenance (label-free):** correlate the off-diagonal of the IT-**firing** cosine matrix with the raw-pixel
  cosine matrix (never touching labels). Intact tracks pixels (EMERGE r ≈ 0.83; Option-B r = 0.99); scramble collapses (≈ 0).
- **Position-invariance (the DiCarlo "IT untangles" property — the one V2/IT is *supposed* to add over V1):** present the
  same object at several retinal positions → IT code should stay stable (within-object-across-position cosine ≫
  across-object), where raw V1 (retinotopic) does not. This is the discriminating test that V2/IT do something V1 cannot.
- **No-learning control:** frozen (un-trained) V2/IT → no category structure / no invariance (the flat baseline).
- **6-seed** (dev 42/43/44 + blind 100/101/102), like-for-like against the frozen-V2/IT baseline.

If step 3 is judged too heavy to open now, the **fallback cheapest single decision-useful de-risk is step 1** (one
heuristic-off nav run): it costs a single GPU run and resolves whether the deployed visual pathway does anything at all
for navigation — the highest-value information per minute in the whole stack.

---

## 5. Effort/cost, concurrency, and the recommendation

**Effort/cost.**
- Scoped closure (steps 1–4): ~3–6 runner-days of build + a handful of GPU-hours; reuse-by-import; **no `sim/` edit
  expected** (B1's on-bridge lift and the V2/IT de-risk both ride existing plastic/gated pathways + existing kernels).
- Fuller build (b): weeks-to-months, a new corpus, and unproven on-bridge STDP feature-learning at scale.

**Concurrency with language / gap#5.** **Cleanly concurrent.** Perception is a *disjoint region subsystem* (retina/V1/V2/
IT + the dorsal SC) from the conversational/gap#5 stack; the only shared resource is the bridge, and steps 1, 3, 4 are
read/measure + reuse-by-import, step 2 touches only the perception-local `retina→V1` plastic pathway. No `sim/` edit is
expected, so there is no protected-code contention with the language work. These can run as their own parallel track
(subagents / background GPU runs) without blocking gap#5. **Caveat:** the GPU is currently busy (a training run + a CPU
de-risk live) — step 1/2/3's GPU runs should be *queued*, not started now; the runner build + numpy smoke can proceed in
parallel immediately.

**Recommendation — SCOPED closure + honest boundary; do NOT open the deep-vision build.**
Perception already adequately serves its two real uses (navigation cue perception; object grounding for the emergent-
semantics arc), and the mission-relevant capability (perceive → discover category → reason/talk) is already demonstrated
end-to-end and spiking. Closing it "for real" therefore means: (1) **measure** the nav contribution (heuristic-off),
(2) **lift B1 on-bridge** (self-organize/dev-random the V1 bank — the one genuine structure residual, already de-risked
GO), (3) **validate-or-retire V2/IT** with the EMERGE anti-cheat kit and record an honest verdict, and (4) optionally
**spiking-V1 the console**. Then **mark residual #5 — rich natural-image object recognition, IT invariance-at-scale,
Tanaka/DiCarlo/HMAX-grade vision — as the deliberate deferred boundary**: a real open problem, off the critical path,
re-entered only if a future task demands rich scene understanding. The honest boundary here is *object-recognition
richness*, not a substrate limit — the substrate + mechanisms are in hand; the deferral is a priority call, not a wall.

---

## References (verified)

- Code: `sim/visual_cortex.py` (gabor_kernel 39–73, build_v1_simple_weights 76–152, render 155–207,
  apply_v1_gabor_weights 223–310); `research/runners/g11_bg_runner.py` (visual regions ~2690–2760, pathways ~2813–2930,
  gabor apply ~4690, ventral readouts `cortex_it→striosome_value` ~511–524 / `cortex_it→language_output` ~2929);
  `tests/test_visual_cortex.py` (14 unit tests: Gabor tuning, forward-fires, wiring, gates — no V2/IT feature-learning test).
- Design: `docs/plans/2026-05-01-cluster-k-visual-cortex-hierarchy.md` (V1 fixed Gabor / V2-IT plastic / V4 skipped).
- Nav result + confound: `2026-05-01-cluster-k-v2-breakthrough.md` **+ its 2026-07-16 correction** (heuristic ON; visual
  contribution unquantified; possible silent inertness).
- Perception validations (all use V1→pooler, not deployed V2/IT): `2026-06-16-generalization-optionB-visual-similarity.md`
  (RSA r=0.99), `2026-07-02-emerge34-perception-grounded-emergence-GO.md`,
  `2026-07-02-emerge36-spiking-perception-pipeline-GO.md`,
  `2026-07-02-emerge53-perception-grounded-conversation-GO.md`,
  `2026-07-11-EMERGENT-fully-spiking-perception-codon-drives-the-ladder-6seed.md`.
- B1 (V1 self-org residual): `2026-06-21-B1-v1-gabor-selforg-scoping.md`, `2026-06-21-B1-v1-gabor-selforg-derisk.md`
  (GO, numpy; on-bridge lift documented, no `sim/` edit).
- Brain-based verdicts on perception: `2026-06-10-N2-N7-characterization-and-honest-nav-cheat-ledger.md`
  (N2/N7 DEFENSIBLE; N1/N5 still host).
- Biology: Hubel-Wiesel 1962 (V1 simple cells, catalog E.08); Kandel 6e Ch 22 (visual processing) + Ch 49 (retinal waves,
  catalog L.05); Riesenhuber-Poggio 1999 (HMAX); Tanaka 1996 / DiCarlo-Cox 2007 (IT object identity + untangling);
  Olshausen-Field 1996 (sparse coding → Gabor basis); Zylberberg et al. 2011 (SAILnet, spiking local Gabor emergence).
</content>
</invoke>
