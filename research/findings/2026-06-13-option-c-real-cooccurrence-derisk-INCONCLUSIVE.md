# Option C de-risk (learn graded semantics from REAL TinyStories co-occurrence): INCONCLUSIVE — the host ceiling control fires; the cheap real-text setup carries SYNTAGMATIC (scene) structure, not the PARADIGMATIC taxonomy → build the production cortex on Option B (curated); Option C is a genuine follow-on, not a blocker

**Date:** 2026-06-13. **Runner:** `research/runners/option_c_real_cooccurrence_derisk.py` (commit `70e89ffc`). **Backend:** `SIM_BACKEND=cupy` (GPU, 3-seed 42/43/44). **Raw:** `research/findings/raw/_option_c_real_cooccurrence_multiseed.json` + `.log`. **Design:** `docs/plans/2026-06-13-option-c-real-cooccurrence-derisk-design.md`. **Source:** `data/corpus/tinystories.txt` (1.59M tokens), 8-category × 8-word independent taxonomy.

> **Verdict: INCONCLUSIVE for the brain-based mechanism** (the automated label was `BOUNDARY_weak_graded`, triggered by a hair — gen 0.309 vs the 1.2×chance=0.30 threshold; the honest controller read, with a follow-up full-context diagnostic, is INCONCLUSIVE). The brain-based spiking-Hebbian learn recovered essentially **no** taxonomic structure from real TinyStories co-occurrence (Pearson(learned, taxonomy) **+0.015**, generalization ≈ chance). **But the host PPMI+SVD ceiling — the load-bearing disambiguator — ALSO failed its bars** (restricted +0.126; full-context +0.263, nn-same 0.50), so the test does NOT carry a clean enough signal to cleanly *attribute* the brain-based failure to the mechanism. Root cause (diagnosed): **TinyStories windowed co-occurrence is SYNTAGMATIC (scene/theme) structure, not the PARADIGMATIC (substitutability) structure the category taxonomy encodes** — even the gold-standard host only partially recovers it. **⇒ build the production cortex on Option B (curated semantic sub-taxonomy, validated). Option C (learned-from-experience) remains genuinely open — a follow-on needing a corpus/measure where the host clearly recovers the paradigmatic taxonomy — NOT a blocker.**

## Results (3 seeds, real TinyStories co-occurrence vs the INDEPENDENT taxonomy)

| Gate | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| **G1** Pearson(S_learned, S_true) | +0.014 | +0.013 | +0.019 |
| G1 2nd-order margin | +0.130 | +0.149 | +0.247 |
| **G2** generalization (chance 0.25) | 0.219 | 0.338 | 0.369 |
| G2 A2 orthogonal collapses / A3 perm-property collapses | True / True | True / True | True / True |
| **HEADLINE** permuted-co-occurrence collapses | True | True | True |
| beats random baseline | False | True | True |
| **HOST CEILING passes** (the disambiguator) | **False** | **False** | **False** |
| host Pearson / gen | +0.126 / 0.525 | +0.126 / 0.487 | +0.126 / 0.594 |
| `s_true_independent` (the crux) | True | True | True |

- **Brain-based: essentially at chance.** Pearson +0.015 (≈ 0), generalization 0.31 (chance 0.25). It did not recover the category structure.
- **Host ceiling: also below bars, but partial.** Restricted-vocab Pearson +0.126 / gen 0.535; the brain-vs-host gen gap is **+0.227** (the host extracts ~2× more from the same data) — so there *is* a real gap, but it cannot be cleanly read as "the mechanism can't" because the host itself fell short.

## The diagnostic that decides the interpretation (cheap pure-numpy, full context)
The de-risk restricted co-occurrence to the 64 target words. To rule out "the vocab restriction destroyed the shared-context (paradigmatic) signal," I ran a host PPMI+SVD over a **target × FULL-context** matrix (64 targets × the 5,000 most-frequent context words — the standard distributional-semantics setup) over all 1.59M tokens:
- **Pearson(sim, taxonomy) = +0.263, within-between margin +0.075, nearest-neighbour same-category 0.50** (chance 0.125). Better than the restricted +0.126 — but **still does not cleanly recover the taxonomy.**
- The per-category nearest neighbours are the tell: **`dog→cat`, `mom→dad`, `run→jump` cluster correctly** (these categories genuinely substitute in similar contexts), but **`red→ball`, `ball→dog`, `house→cat`, `apple→bell`, `hand→kite` do NOT** — they bind to their *scene-mates* (a red ball, a dog with a ball, a cat in the house). **TinyStories co-occurrence is dominated by syntagmatic scene structure, which only partially aligns with the paradigmatic taxonomy.**

## What this means (honest)
- **The host-ceiling control did exactly its job.** Without it, the brain-based +0.015 would have read as a clean "the point-neuron mechanism can't learn graded semantics from real experience." The ceiling reveals the cheap real-text setup carries only a weak, partially-mismatched signal — so we do **not** overclaim a clean mechanism negative. This is the design's load-bearing disambiguator working as intended.
- **The result is genuinely inconclusive for the mechanism question.** There is a real brain-vs-host gap (the host gets animals/family/actions; the brain-based gets nothing), suggesting the point-neuron Hebbian learn is weaker — consistent with the prior (the Mikulasch-Priesemann point-neuron limit) — but the test can't isolate it from the weak/mismatched data signal.
- **Build decision: Option B.** Per the bounded plan (one falsification → build on C if GO, else B), Option C did not produce a GO, so the production cortex builds on **Option B** (the curated within-cluster semantic sub-taxonomy — validated, ready, the brain-based learn reproduces the curated structure). The curated corpus is legitimately "the agent's structured experience" (host code provides the environment/experience; the learn over it is brain-based).
- **Option C is a real follow-on, not a blocker.** A fair Option-C test needs a corpus/measure where the host PPMI+SVD *clearly* recovers a paradigmatic reference (e.g., a larger/more-varied corpus, a substitutability/second-order co-occurrence measure rather than raw windowed counts, or accepting the categories that do cluster — animals/family/actions). That is a separate research arc; it does not gate the production build.

## Cost + scope
~1 GPU-hour (3-seed) + a ~3-min numpy diagnostic resolved the corpus-source question for a multi-day build — the cheap-first de-risk worked as intended. **No `sim/` edits** (reuse-by-import). The crux held throughout (`s_true_independent=True` every seed — the reference taxonomy was never corpus-derived). Toy scale (64 words) — but the host-ceiling-also-fails result is the decisive signal that this cheap setup can't fairly test the mechanism, independent of scale.

## Conclusion + next
Option C de-risk = INCONCLUSIVE (the cheap real-text setup carries syntagmatic, not paradigmatic, structure — the host ceiling control fires). **Build the production cortex on Option B.** With the corpus-source decided (B), the 32-bridge fan-out multi-seed GO, route A, and the production vocab all in place, **the multi-day production build is the remaining step — the owner's explicit-go gate.** Option C (learned-from-experience similarity) is logged as a genuine follow-on research arc (it needs a corpus/measure where the host clearly recovers the taxonomy). No banking — the negative is reported honestly as inconclusive, with the host-ceiling control preventing a false clean-mechanism-negative claim.
