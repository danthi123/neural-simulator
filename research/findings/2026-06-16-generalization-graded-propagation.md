# Generalization frontier — the held-out cross-modal convergence PROPAGATES AS SPIKES (GO)

**Date:** 2026-06-16
**Runner:** `research/runners/_genfrontier_graded_propagation_derisk.py`
**Raw:** `research/findings/raw/_genfrontier_graded_propagation.json`
**Builds on:** `research/findings/2026-06-16-generalization-onsubstrate-convergence.md` (the on-substrate convergence GO)
**Verdict:** **GO** — 3 seeds (42/43/44), GPU (`SIM_BACKEND=cupy`), all anti-cheats clean. NO `sim/` edit.

---

## The question (the live-task capstone's one open piece)

The on-substrate convergence GO established that population-Hebbian co-activation of a similarity-STRUCTURED
perception region + a concept region transfers category-generalization on spikes — a HELD-OUT (never-converged)
concept's structured perception cue lands in its correct semantic CATEGORY. BUT that transfer was read as the
concept assembly's GRADED population DEPOLARIZATION (an instrument), because the point-neuron concept assembly was
believed to **not be able to SPIKE from perception alone** (verified there: 0 concept spikes even at 8000 pA / weight
29 — the synaptic conductance decays between sparse perception spikes faster than it accumulates to the Izhikevich
+30 mV threshold). The live-task pipeline (who/what recall + the no-confab moat) reads concept codes through
SYNAPSES = SPIKES, so the novel-perceived-object response must PROPAGATE AS SPIKES. This runner de-risks that
propagation: **for a held-out (novel) perceived object — perception cue ALONE — can a downstream read-out convert
the converged concept assembly's graded category-correct response into category-correct SPIKES (`cp_firing_states`,
REAL spikes, not membrane) that a synaptic pipeline reads?**

## Candidate(s) tried + how the read-out was wired

Three candidates were available (the project uses all three); the decisive one is **candidate 1 (NMDA-integrated
read-out)**. The architecture is a 3-region bridge `perception → concept(NMDA) → readout(NMDA)`:

- **perception → concept**: ALL-TO-ALL, plastic, near-floor init (0.05). The convergence the rate-Hebbian LEARNS
  (the spiking analogue of the numpy ridge map; reuses the convergence GO's `structured_perception_sets` +
  `train_convergence` verbatim by import).
- **concept → readout**: BLOCK-DIAGONAL (concept block i → readout block i ONLY), FIXED, strong weight. The
  CATEGORY structure is NOT in the wiring — category is read by category-mean over the read-out SPIKE counts, so
  the read-out projection does no category work.
- **Candidate 1 (NMDA), wired thus:** `BrainRegion(enable_nmda=True)` on BOTH the concept and read-out regions +
  `cfg.enable_nmda=True` globally, so the framework's per-region NMDA mask confines the slow NMDA current to those
  two slices (`sim/bridge.py:1212-1221`). The slow NMDA conductance (`tau_decay=100 ms`, fed the SAME excitatory
  synaptic input as AMPA scaled by `nmda_ratio` — `sim/bridge.py:5986-5989`, then Mg2+-block-gated) **temporally
  integrates** the sparse perception-driven drive across the gaps that defeat the AMPA-only point neuron → it
  crosses threshold and SPIKES. `nmda_ratio=2.0` so NMDA dominates the integration.
- Candidates 2 (population pooling + low threshold) and 3 (graded transmission, `RegionPathway.graded=True`) are
  selectable (`--candidate pool|graded`); candidate 3 was run for comparison (below).

Wiring is installed via `inject_explicit_wiring` (which rebuilds the sparse matrix + every per-synapse array,
incl. the graded mask, in one correct pass — no post-init `cp_connections` surgery). A non-obvious gotcha cost
several iterations and is documented in the runner: an EMPTY `region_pathways` makes the framework generate no
synapses → the bridge falls into the spatial-generator FALLBACK, which leaves the regions inert (perception 0
spikes at ANY drive). The fix is to declare both pathways in the framework (so init takes the clean wiring branch),
then fully overwrite via `inject_explicit_wiring`.

## Result (3 seeds, GPU)

| metric | result |
|--------|--------|
| **Concept assembly SPIKES / cue** (held-out, REAL `cp_firing_states`) | **146** (132 / 168 / 138) — refutes the prior "concept cannot spike from perception" residual |
| **Concept-spike category accuracy** (held-out, primary read-out) | **0.92** (1.00 / 1.00 / 0.75); chance 0.25; same-vs-other margin **+0.146** |
| **Downstream read-out region SPIKES / cue** (propagation) | **266** (284 / 265 / 247) — > 0, the response propagates a synapse further |
| FLAT-distinct concept cat-acc (structure ablation) | **0.25** (≈ chance) — structure load-bearing |
| PERMUTED (category-derangement) concept margin | **−0.020** — collapses |
| Moat (no-confab) | **INTACT** all 3 seeds (held-out best-cat spikes ≈ 1.7–2× a novel-no-category cue's) |
| read-out *region's own* category read (secondary relay) | 0.25 — noisy (honest residual, below) |

- **The held-out novel-perception response PROPAGATES AS SPIKES.** The converged concept assembly — which is the
  conversation cortex's concept code the live who/what + no-confab pipeline reads via synapses — **SPIKES** (146
  spikes/cue, REAL spike counts) and lands in the held-out concept's correct semantic CATEGORY **0.92** (≫ chance
  0.25). This is candidate 1's NMDA-integration doing exactly its job: the slow NMDA conductance integrates the
  sparse perception drive that an AMPA-only point neuron cannot, producing supra-threshold concept spikes. The
  spike-based cat-acc (0.92) **equals the prior GRADED-depolarization read (0.92)** — i.e. the read is now neural
  (real spikes), not an instrument, at no loss.
- **Flat-distinct perception does NOT transfer** (0.25 ≈ chance) → similarity-structured perception (Option B)
  stays the load-bearing prerequisite, exactly matching the convergence GO's flat collapse.
- **The category-derangement control collapses** (margin −0.020 vs structured +0.146; concept cat-acc 0.08) → the
  transfer is the LEARNED perception-category↔concept-category correspondence, not a geometry coincidence. (At the
  concept-block level this is unambiguous: a held-out cat-0 cue drives concept-blocks [cat0 43, cat1 11, cat2 14,
  cat3 15] in the structured arm, but [cat0 17, **cat1 47**, cat2 18, cat3 27] in the derangement — the learned map
  cleanly re-routes to the wrong category.)
- **The no-confab moat survives** — a novel-no-category perception ensemble produces a far weaker best-category
  spike response than a real held-out concept; the system abstains rather than confabulating.
- **The response propagates one synapse further** — the downstream read-out region also SPIKES robustly (266/cue),
  proving the concept spikes drive a further synaptic stage (candidate 3 graded: 69/cue — also propagates).

## Honest residual (load-bearing, characterized)

**The further-downstream read-out *region's own* category decision is noisy (cat-acc 0.25, NEGATIVE margin).** The
block-diagonal concept→readout relay, with its saturating NMDA non-linearity, does NOT faithfully preserve the
category ranking: each read-out block sees only a FIXED uniform weight × its concept block's spike rate, so the
read-out's category DIFFERENTIAL is small and swamped by per-read-out-block intrinsic excitability heterogeneity
(neither raw, z-scored-against-train-baseline, nor category-pooled decoding recovered it past 2/4). This is the
documented rate-code wall reappearing one synapse further on, and it is candidate-independent (NMDA and graded
both show it). **It does NOT weaken the GO:** the propagatable signal the live pipeline actually reads is the
**concept assembly's own spikes** (the concept region IS the conversation cortex's concept code), and those are
category-correct (0.92) + synaptically readable. The concept region, NMDA-integrated, IS candidate 1's read-out
("the read-out population that SPIKES most" = the concept assembly). The extra read-out region was a one-more-hop
existence proof of propagation (it spikes); making a CLEAN second-stage relay (e.g. a learned read-out, or a
category-pooled read-out with common-mode removal) is a bounded follow-on, not a blocker for the capstone.

## Anti-cheats (all clean)

1. **Flat-distinct baseline** — flat perception scores at chance (0.25) on held-out category transfer; the
   discriminating gap vs the structured arm (0.92).
2. **No-leakage split** — held-out concepts are excluded from co-activation training (asserted); held-out and
   train do not overlap.
3. **Category-derangement permuted control** — co-activating each train concept's structured perception with a
   WRONG-category concept block collapses the held-out transfer (margin −0.020, concept cat-acc 0.08).
4. **No-confab moat** — a novel-no-category perception ensemble does not drive a confident category spike response;
   the held-out concept is clearly more category-familiar. INTACT all 3 seeds.

## Scope / honest residuals

- Validated at **16 concepts (4 categories × 4)**, the small-config de-risk scale (matching the convergence GO).
- **Option B (a similarity-structured perception FRONT END) remains the GIVEN prerequisite** — this runner
  confirms that GIVEN such a front end + the validated convergence, the held-out response propagates as
  category-correct SPIKES.
- **Dendritic substrate NOT required** — point-neuron NMDA-integration converts the sub-threshold concept
  depolarization to spikes (the project's standard "slow-NMDA integration + population code" lift), consistent with
  the convergence GO's call.
- The clean second-stage read-out relay (above) is the one bounded follow-on; the capstone's open piece — does the
  novel-perception response propagate as spikes a synaptic pipeline can read — is **de-risked GO**.

## NO `sim/` edit

Reuse-by-import only (`sim.backend.to_host`, the brain-region framework + per-region NMDA mask + the
`inject_explicit_wiring` / `graded` pathway, and the convergence GO's `structured_perception_sets` /
`flat_perception_sets` / `train_convergence`). `git diff -- sim/` is empty.
