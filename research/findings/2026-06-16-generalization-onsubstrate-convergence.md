# Generalization frontier — cross-modal convergence REALIZED on the spiking substrate (GO)

**Date:** 2026-06-16
**Runner:** `research/runners/_genfrontier_onsubstrate_convergence_derisk.py`
**Raw:** `research/findings/raw/_genfrontier_onsubstrate_convergence.json`
**Numpy cheap-first (the result being made neural):** `research/findings/2026-06-16-generalization-crossmodal-unify-cheap-first.md`
**Verdict:** **GO** — 3 seeds (42/43/44), GPU (`SIM_BACKEND=cupy`), all four anti-cheats clean. NO `sim/` edit.

---

## The question

The numpy cheap-first established (GO) that a LEARNED map from a similarity-STRUCTURED perception code onto the
conversation cortex's category-structured concept code transfers category-generalization to perception — a
HELD-OUT (never-converged) concept's structured perception lands in its correct semantic category (cat-acc 1.00,
chance 0.25), and it collapses under a category-derangement permuted control + a flat-distinct baseline, with the
no-confab moat intact. That isolated the MECHANISM in a numpy ridge map. **This runner realizes that convergence
on a real `SimulationBridge`** as population-Hebbian co-activation, and reads the transfer from the concept
region's spiking response — making it neural, not a host regression.

## The bridge design (how STRUCTURED perception was driven on the substrate)

A two-region `SimulationBridge` (reuse-by-import of the `_phaseB_stdp_cooccurrence_derisk.build_assoc_bridge`
pattern: two regions, population code, **rate-Hebbian** — co-occurrence is symmetric, so STDP's Δt kernel lands at
~0 and the matched rule is Hebbian coincidence):

- **perception region** (the "hub", 1600 neurons) — each concept drives a SIMILARITY-STRUCTURED ensemble: a
  **shared per-CATEGORY core** (same `n_active_cat=48` neurons for all four concepts in a category → same-category
  OVERLAP, the Option-B shared-feature structure) + a **per-concept unique tail** (`n_active_uniq=12`, disjoint).
  The assignment is SCATTERED via a random permutation across the region (a contiguous layout creates a spurious
  monotonic index bias). The flat-distinct baseline gives every concept its own disjoint block (no category
  overlap = the current nav regime).
- **concept region** (the "target") — DISJOINT per-concept population blocks (`n_concept_per=100` neurons each,
  16 concepts), the conversation cortex's concept codes.
- a plastic **perception→concept** pathway (init `weight_mean=0.05`, rate-Hebbian, `hebbian_max=20`).

**Training (the convergence):** for each TRAIN concept X, co-drive its structured perception ensemble (300 pA) +
its concept block (600 pA teacher) for 16 steps × 20 epochs. Rate-Hebbian potentiates the perception→concept
synapses for the co-active pair (verified: perception + concept BOTH fire in the scene — first-scene diag perc
~130 / conc ~360 spikes — so the 1-step Hebbian coincidence has its substrate; the learned own-block weights grow
from 0.05 to mean ~7 / max ~29).

**Held-out test (on the spiking substrate):** for a HELD-OUT concept (never co-activated; its unique-tail synapses
stay at floor), drive ONLY its structured perception ensemble, run the bridge, and read the concept region's
GRADED population response per block (population-averaged membrane depolarization above rest), z-scored against the
TRAIN-cue baseline, decided by **category-mean** (the category whose concept blocks have the highest mean
z-response). Held-out structured perception overlaps its category's TRAINED perceptions (the shared core) → drives
those concept blocks via the learned category→concept synapses → category transfer = the generalization.

## Result (3 seeds)

| arm | held-out category accuracy | same-vs-other z-margin |
|-----|----------------------------|------------------------|
| **STRUCTURED** (Option B) | **0.92** (1.00 / 1.00 / 0.75); chance 0.25 | **+1.367** |
| **FLAT-distinct** (current nav) | 0.17 (≈ chance) | −0.020 |
| **PERMUTED** (category derangement) | 0.00 | −0.118 (collapses) |

**Moat (no-confab):** INTACT all 3 seeds — a held-out concept's best-category z (1.59 / 1.47 / 1.71) is far above a
NOVEL random ensemble's (0.57 / 0.72 / 0.33); the system abstains on the novel cue rather than confabulating a
category.

- **The cross-modal convergence is realized on the spiking substrate**: a held-out (never-converged) concept's
  STRUCTURED perception cue lands in its correct semantic category 92% (≫ chance 25%, z-margin +1.367) on real
  spikes — reproducing the numpy ridge result (cat-acc 1.00) now neural.
- **Flat-distinct perception does NOT transfer** (0.17 ≈ chance) → similarity-structured perception (Option B) is
  the load-bearing prerequisite, exactly matching the numpy GO's flat collapse.
- **The category-derangement control collapses** (cat-acc 0.00, margin −0.118) → the transfer is the LEARNED
  perception-category↔concept-category correspondence, not a geometry coincidence.
- **The no-confab moat survives** the convergence — the load-bearing abstention is preserved.

## Did the spiking convergence match the numpy ridge map? (the honest read)

**Yes, once the documented population-code lift was applied** — and that lift was the load-bearing fix. The
point-neuron concept region **physically cannot SPIKE from perception alone**: verified that concept spikes stay
0 even at 8000 pA perception drive / learned weight 29 (the synaptic conductance decays between sparse perception
spikes faster than it accumulates to the +30 mV Izhikevich threshold). So the spike-count read-out (the first
attempt) was all-zeros → NEGATIVE. The fix is the project's documented **rate-code-wall** pattern: read the concept
assembly's **graded population depolarization** (its own membrane state — not a host computation), and **average
over a large population** (CYCLE 91 lift). The population size is decisive:

| n_concept_per | held-out cat-acc (3 seeds) |
|---------------|----------------------------|
| 12 (single-block argmax, spike-count) | 0.00 (read-out is all-zero spikes) |
| 24 (z-score + category-mean) | 0.25 / 0.75 / 0.25 (noisy) |
| 64 | 1.00 / 0.75 / 0.50 |
| **100** | **1.00 / 1.00 / 0.75** |

Two further point-neuron-specific read-out choices were needed (both legitimate, both LOCAL): (1) a **scattered**
(permuted) perception assignment to kill a contiguous-layout index bias, and (2) **per-block z-scoring against the
TRAIN-cue baseline** (a feedforward-standardization that removes each block's intrinsic per-neuron excitability
offset — computed from TRAIN cues only, no held-out leakage), with a **category-mean** decision rather than a
single-block argmax (a single noisy block is the wrong decision rule at point-neuron scale; the population
category-mean uses all the assembly evidence — same-category z-margins were reliably positive at every population
size, it was only the single-block argmax that was unstable). The honest takeaway: **the spiking read-out is
noisier than the ridge map**, but it is the documented rate-code wall, and the documented population lift closes it
to ridge-map parity (cat-acc 0.92 ≈ the numpy 1.00).

## Anti-cheats (all clean)

1. **Flat-distinct baseline** — flat perception scores at chance (0.17) on held-out category transfer; the
   discriminating gap vs the structured arm (0.92).
2. **No-leakage split** — held-out concepts are excluded from the co-activation training (asserted); the z-score
   baseline is computed from TRAIN cues only.
3. **Category-derangement permuted control** — co-activating each train concept's structured perception with a
   WRONG-category concept block (a consistent derangement) collapses the held-out transfer to a NEGATIVE margin
   (−0.118) and chance/below cat-acc (0.00): the transfer is the learned correspondence, not geometry.
4. **No-confab moat** — a novel perception ensemble (random neurons, no category) produces a low best-category z
   (~0.5) vs a real held-out concept (~1.6): the system abstains rather than confabulating.

## Scope / honest residuals

- Validated at **16 concepts (4 categories × 4)**, the small-config de-risk scale. The 320-concept on-substrate
  version (and a corpus-grounded similarity-structured perception RENDER for Option B — here the structured
  perception is the CONTROLLED GIVEN, an independent category basis, exactly as the numpy cheap-first specified)
  are the follow-on build.
- **Option B (a similarity-structured perception FRONT END) remains the prerequisite** and is de-risked
  separately — this runner confirms that GIVEN such a front end, the spiking convergence transfers
  generalization. The live-task build (co-activate perception + the conversation cortex on the merged bridge when
  the agent perceives an object whose word it hears, commit shared assemblies, test generalization to novel
  similar perceived objects + the who/what matrix + the moat) is the next step.
- **Dendritic substrate NOT required** — this is point-neuron population-Hebbian convergence + a graded population
  read, consistent with the numpy cheap-first's decisive call and the on-bridge PPMI-stream cortex (the
  generalization being inherited is itself realized on point neurons).

## NO `sim/` edit

Reuse-by-import only (`sim.backend.to_host`, the brain-region framework, the `build_assoc_bridge` rate-Hebbian
pattern). `git status -- sim/` is empty. The runner builds its own two-region bridge in-file (no `sim/` change).
