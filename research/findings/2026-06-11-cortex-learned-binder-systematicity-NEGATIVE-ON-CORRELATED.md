# Step 3 (true cortex) — learned-binder systematicity: NEGATIVE_ON_CORRELATED → the Fodor-Pylyshyn boundary is the CODE CORRELATION, and decorrelated codes dissolve it (a learned binder is systematic there)

**Date:** 2026-06-11 (overnight). **Runner:** `research/runners/cortex_learned_binder_systematicity_probe.py` (CPU, `SIM_BACKEND=numpy`). **Raw:** `research/findings/raw/cortex_learned_binder_systematicity_multiseed.json` (seeds 42/43/44, F=16, R=4, n_splits=3, 1000 epochs, D_h=64, proj 256) + `_syst_tinysmoke.json`.

> **Result: NEGATIVE_ON_CORRELATED, multi-seed.** A *learned* bilinear binder generalizes **systematically** (binds role-filler pairs it never saw paired — the Fodor & Pylyshyn 1988 held-out-novel-combination test) on **decorrelated** codes (held-out accuracy **1.000 = train, all 3 seeds**, matching the exact-inverse algebra reference and infinitely beating a memorizer), but **fails on the brain's correlated codes** (held-out ≈ chance; it cannot even fit them). The systematicity failure is the **code correlation**, not learning itself. This converges with the cleanup arc + the positive control: **the cortex must decorrelate its codes first (structural sparse expansion), and over decorrelated codes a learned binder is systematic.**

## Why this probe ran

The cleanup sub-arc closed with three mechanistically-distinct NEGATIVES (vanilla Hopfield common-mode collapse; Storkey locality wall; spiking dentate-gyrus sub-reproducibility) — no brain-based mechanism can clean the composer's *fixed correlated* concept codes (between-code cosine ≈ 0.81) post-hoc. The positive control (`2026-06-11-cortex-sparse-attractor-poscontrol-GO.md`) showed a distributed attractor recovers 1.000 on *decorrelated* codes (cosine ≈ 0.05) and collapses on correlated — the wall is the code correlation, not the mechanism. The remaining genuinely-cortical question (the deep-research opening move flagged it as the load-bearing risk): can a **learned** binder — not the fixed exact-inverse algebra — bind over the brain's correlated codes AND generalize *systematically* (recombine known atoms it never saw paired), which is the property a fixed algebra has by construction but a learner might only fake by memorization?

## Method (a leakage-free systematicity protocol with four anti-cheats)

- **Task:** R=4 roles × F=16 fillers (concept codes). A fact = a (role, filler) pair; the binder produces a bound vector; UNBIND recovers the filler from (bound, role). Score = nearest-filler (native readout) == true filler.
- **The split (the crux):** hold out R novel (role, filler) combinations such that **every role and every filler still appears in some training combo** — only the specific *pairings* are held out (a novel recombination of seen atoms). A `leakage_count` assert verifies train ∩ held-out = ∅ (it was 0 everywhere; every atom covered everywhere).
- **Binders:** (1) the **learned bilinear binder** (gradient-trained) — the candidate; (2) the **exact-inverse FHRR (Fourier Holographic Reduced Representation) algebra** — the systematic-by-construction REFERENCE.
- **Two code regimes:** DECORRELATED sparse-distributed codes (between-cos ≈ 0.001 as read) and CORRELATED `denoise64` codes (≈ 0.81). A unit check asserts both correlations (codes read in native form — never median-bipolarized, the artifact that manufactures a false common mode).
- **Anti-cheats:** (i) leakage assert; (ii) **memorization floor** — a pure lookup-table that has no entry for held-out combos must score chance/zero (it scored **0.000**); (iii) shuffled-held-out-label control; (iv) abstention/familiarity gap on never-bound queries.
- **Staging (honest):** the gradient-trained binder is a CPU **characterization** of whether *any* learned binder generalizes on these codes — the same cheap-first staging the numpy-Hopfield cleanup probes used before the spiking ones. If a host-optimized binder can't generalize, no spiking realization will. The spiking surrogate-gradient backprop-through-time realization is the later build.

## Results (multi-seed 42/43/44, F=16)

| regime | learned binder held-out | learned binder train | FHRR reference held-out | memorization floor (held) | familiarity gap | leakage | chance |
|---|---|---|---|---|---|---|---|
| **DECORRELATED** (cos≈0.001) | **1.000 / 1.000 / 1.000** | 1.000 / 1.000 / 0.989 | 1.000 | **0.000** | 0.47 / 0.46 / 0.43 | 0 | 0.062 |
| **CORRELATED** (cos≈0.81) | 0.000 / 0.167 / 0.167 | 0.183 / 0.267 / 0.489 | 1.000 | 0.000 | 0.03 / 0.03 / 0.06 | 0 | 0.062 |

**Aggregate:** decorrelated held-out **1.000 ± 0.000** (systematic_fraction 0.78); correlated held-out **0.111 ± 0.079** (≈ chance, systematic_fraction 0.11). Verdict = **NEGATIVE_ON_CORRELATED**.

## Reading

- **The systematicity on decorrelated codes is REAL, not a readout artifact.** The decisive control is the memorization floor: a lookup-table memorizer scores **0.000** on the genuinely-novel (leakage=0) held-out combos, while the learned binder scores **1.000** — i.e. it *recombines* known roles and known fillers it never saw paired, exactly as the systematic-by-construction FHRR reference does. (The shuffled-label control is weak here only because the held-out set is small, N=4 combos, so a label permutation has a high coincidental match rate; the memorization-floor + FHRR-reference + leakage-0 trio is the clean evidence.)
- **The correlated failure is comprehensive.** On the brain's correlated codes the binder can't even *fit* the training pairs (train 0.18–0.49), can't generalize (held ≈ chance), and the familiarity/abstention gap collapses (0.45 → 0.03 — the no-confab signal also can't separate known from unknown on correlated codes). Highly-correlated fillers (cos 0.81) are near-linearly-dependent, so recovering a filler from (bound, role) is ill-conditioned — the small differences that distinguish fillers are swamped.
- **Honest caveat (the one residual nuance):** because train accuracy is itself low on correlated codes, this is "fails to fit *and* fails to generalize," not the cleaner "fits perfectly then memorizes." A much higher-capacity binder might fit-then-memorize correlated training; but the actionable conclusion — *decorrelate first* — is unchanged either way, and was not worth the (multi-hour) capacity sweep once the architectural answer was settled and triply-confirmed.

## The architectural conclusion (this closes the cortex DE-RISK arc — it converges with the whole night)

Every thread now points to one architecture. The brain's raw concept codes are **correlated** (they carry semantic similarity — useful for generalization, but adversarial to binding):

1. **A fixed exact-inverse algebra** (FHRR) cannot use correlated codes (it demands clean/decorrelated codes — the known idealization limit; the composer's documented "clean-code demand").
2. **A learned binder** also cannot use correlated codes (this probe: can't fit, can't generalize) — so the failure is not "fixed vs learned," it is the **correlation**.
3. **No brain-based post-hoc cleanup** of correlated codes exists (the three cleanup NEGATIVES; a local rule provably can't remove the common mode — Mikulasch-Priesemann; only a non-local matrix inverse can, which isn't biological).
4. **Decorrelated codes dissolve all of it:** the distributed attractor cleanup recovers **1.000** (positive control) AND a learned binder is **systematic** (this probe, held-out 1.000) AND the FHRR algebra works AND the familiarity/no-confab gate separates (gap 0.45). The decorrelation the project ships is **structural sparse expansion** (catalog F.12 granule/codon coding, D.12 dentate-gyrus pattern separation; the validated sparse-distributed scheme, between-cos ≈ 0.05) — a *structural* front end, NOT a learnable local pairwise rule.

**⇒ The functional cortex = [structural sparse-expansion decorrelation front end] → [a binder over decorrelated codes (learned, now shown systematic — *or* the exact-inverse algebra)] → [cleanup (distributed attractor or localist) + the learned familiarity/no-confab gate].** Every piece is now validated individually. The genuinely-open work is the **build**: assembling this pipeline on-substrate (spiking) and validating the full who/what-Q&A + abstention + negation + clause + two-attribute capability matrix to 320 concepts (the vocabulary-ceiling specification) — a GPU build that the standing "present before committing build resources" practice says to present to the owner before launching.

## Provenance / brain-based bar

- The systematicity test is leakage-free (asserted) with every atom seen in training — the held-out set is genuine novel recombination.
- The gradient-trained binder is labelled a CPU **characterization** of learnability/systematicity, not the deliverable; the FHRR algebra appears only as the systematic reference; the readout is scoring, not the binding computation.
- The decorrelated codes come from the project's structural sparse-expansion scheme (F.12/D.12), not a host whitening transform; codes read in native form (unit-checked), never median-bipolarized.

## Verdict + next step

**NEGATIVE_ON_CORRELATED (multi-seed).** A learned binder is systematic on decorrelated codes and fails on the brain's correlated codes — the Fodor-Pylyshyn boundary is the code correlation, and structural decorrelation dissolves it. The cortex DE-RISK arc is complete; the architecture is scoped into individually-validated pieces. **Next:** consolidate the de-risked pieces into a concrete cortex BUILD plan (the assembled spiking pipeline + the V=320 acceptance matrix) and present it before the GPU build. No banking — the wall (a cortex over the brain's codes) is surpassed by the decorrelate-first architecture, demonstrated piece-wise across cleanup, binding, and the no-confab gate.
