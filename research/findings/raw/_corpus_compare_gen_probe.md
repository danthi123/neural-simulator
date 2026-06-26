# Corpus-comparison generalization probe (CPU numpy) — VERDICT

**Date:** 2026-06-26
**Type:** Cheap CPU de-risk (the gate specified in `_corpus_richness_gen_lever_scoping.md` §(c)).
**Constraints honored:** CPU + numpy ONLY (no cupy/GPU; `SIM_BACKEND=numpy`); NO `sim/` edits; no git commit.
**Runner:** `research/runners/_corpus_compare_gen_probe.py` · **Data:** `research/findings/raw/_corpus_compare_gen_probe.json`
**Wall-clock:** 8.7 s (CPU, deterministic).

---

## TL;DR — VERDICT: NOT a demonstrated lever on this thin sample (the corpus-content hypothesis is refuted a 4th time)

**On a SIZE-MATCHED ~620K-token sample, at the BOTH-corpora INTERSECTION vocab, with the SAME a-priori taxonomy
labels, the thin local Simple-Wiki sample (`wikitext.txt`) does NOT raise the intrinsic numpy category-structure
ceiling above TinyStories — it is materially LOWER on both taxonomy scales:**

| taxonomy | intersection | TinyStories Pearson | wikitext Pearson | Δ (wiki − tiny) | call |
|---|---|---|---|---|---|
| TAXONOMY_8x8 (64w/8cat) **[under-powered]** | 39 words / 7 cats | **+0.5469** | **+0.3517** | **−0.1952** | tiny higher |
| TAXONOMY_40x8 (320w/40cat) | **132 words / 34 cats** | **+0.2636** | **+0.1595** | **−0.1041** | tiny higher |

(gen + ratio_vs_chance, deterministic, 3 seeds, std 0.0000: 8x8 tiny gen 0.821 @5.74× / wiki 0.564 @3.95×; 40x8
tiny gen 0.386 @13.14× / wiki 0.205 @6.95×.)

**Both anti-cheats PASS.** Derangement (shuffle the a-priori labels) collapses Pearson to ~0 on every arm (8x8:
tiny −0.0331, wiki +0.0322; 40x8: tiny −0.0083, wiki +0.0080) → the numbers reflect REAL category structure, not
an artifact. S_true is the independent a-priori taxonomy block matrix throughout (never corpus-derived).

**⇒ The recommendation: DO NOT download the full Simple-Wiki dump *for the generalization-lever reason*.** On this
local sample, corpus richness is not a demonstrated category-structure lever — consistent with the
3×-already-refuted corpus-content hypothesis (the C0 control: the 320 gen number is gated by SCALE ~0.30 Pearson +
the SPIKING READ-OUT ~0.145 Pearson, NOT corpus flatness). **CAVEAT (load-bearing, below): a major confound + the
thin sample mean this does NOT *refute* the full dump — it just removes the cheap evidence that a richer corpus
*would* lift the ceiling.** Simple-Wiki remains a real **KNOWLEDGE-BREADTH** lever (more clusterable concepts past
TinyStories' ~680), which is a separate, sound reason to add it — just not a generalization fix.

---

## What was run (the apples-to-apples design)

Reuse-by-import of the C0 template (`_curriculum_gen_C0_substrate_vs_scale.run_seed_numpy_exact`, byte-for-byte
pipeline body): `M[target,hub] += 1` exact integer co-occurrence over a WM window=2, then
`double_center(log1p(M*100))`, then `heldout_generalization` + `_pearson_vs_Strue` against the a-priori taxonomy.
`N_HUB=500` (each corpus uses its OWN top-500 frequent context words as hubs).

Three design points make it apples-to-apples (all in the runner docstring + JSON):
1. **SIZE-MATCH** at the token-stream level. wikitext (619,738 `[a-z]+` tokens) is the smaller corpus, so BOTH
   are truncated to that common budget. Isolates corpus QUALITY from corpus SIZE.
2. **RE-CHUNK** both flat token streams into equal-length 400-token pseudo-stories — identical WM-window
   treatment, and it neutralizes wikitext's 0 `<|endoftext|>` delimiters (it would otherwise stream as one
   giant document). window=2 co-occurrence is window-local, so a chunk boundary only drops the 2 straddling
   pairs — identical for both corpora. Both → 1,550 pseudo-stories.
3. **INTERSECTION VOCAB** — score only taxonomy words present in BOTH size-matched corpora at freq ≥ 20; same
   a-priori cat_ids; empty categories dropped + ids densified so chance = 1/#non-empty-cats is correct.

**Intersection sizes (reported per scoping (c) step 2):**
- 8x8: **39/64 words, 7/8 categories** → **UNDER-POWERED** (below the ≥40-word / ≥6-category threshold; flagged
  loudly). 25 of the 64 children's-story words (e.g. several toys/foods) are too rare in the wiki sample.
- 40x8: **132/320 words, 34/40 categories** → adequately powered (132 words across 34 categories).

The 320-scale arm is the powered, load-bearing comparison; the 64-scale arm is under-powered but agrees (same
sign, larger gap).

---

## ⚠️ The load-bearing CONFOUND — read before acting on the number

The gap is **partly confounded with per-word sampling density**, which I must report honestly rather than
attribute the whole −0.10 to "flatter category structure":

- At MATCHED total size, the taxonomy words occur **~6× more often in TinyStories** than in the wiki sample:
  over the 132-word 320-intersection, the summed target-word frequency is **TinyStories 60,552 (mean 458.7/word)
  vs wikitext 10,436 (mean 79.1/word)**. The exact-count pipeline therefore accumulated **187,599 vs 35,635
  co-occurrence updates** — wikitext's codes are built from ~5× fewer counts.
- Examples (tiny / wiki): dog 1879/20, tree 1501/53, car 580/26, red 723/91, water 601/151, run 370/173, river
  69/45, king 80/57. (22/132 intersection words ARE more frequent in wiki — mostly the encyclopedic ones — so it
  is not a one-sided vocab; but the children's-story-centric taxonomy is sampled far more densely by TinyStories.)

**Interpretation:** the question (c) asks specifically about *matched corpus SIZE* (anti-cheat 8: "hold size to
isolate within-category density"), and at matched size the wiki sample gives a LOWER recoverable category
structure for THIS taxonomy. That is the honest answer to the question as posed. **But the mechanism is at least
partly "this children's-vocabulary taxonomy is simply rarer in encyclopedic text at matched size" (thinner
per-word Zipf depth), NOT necessarily "wiki's category co-occurrence is intrinsically flatter."** A
per-word-count-matched comparison (or a Wiki-appropriate taxonomy) could read differently. So the clean,
defensible claim is the narrow one: **this thin Simple-Wiki SAMPLE does not lift the numpy ceiling for the
existing TinyStories-curated taxonomy at matched size** — not the broad "Wikipedia has worse category structure."

This is exactly why the verdict is "not a *demonstrated* lever on this sample" (evidence absent), **NOT** "a
richer corpus is refuted" (which this thin, confounded sample cannot establish).

---

## Anti-cheats (scoping (d)) — status

1. **Corpus = environment** ✓ — both are natural linguistic corpora; the numpy exact-count "cortex" does the
   learning; no LLM-derived codes, no hand-engineered category injection.
2. **S_true independent / a-priori** ✓ — built from `cat_ids` (the fixed taxonomy) only, identical across corpora.
3. **Pearson(cos,S_true) as the chance-independent tell** ✓ — reported with gen + ratio_vs_chance. (Note the gen
   ratios: wiki 320 is 6.95× chance — well above chance, just BELOW TinyStories' 13.14×.)
4. **Vocab declared a-priori** ✓ — the TAXONOMY_8x8 / TAXONOMY_40x8 labels are fixed; the intersection rule
   (freq ≥ 20 in both) is mechanical, not gen-tuned.
5. **Size-matched** ✓ — both truncated to 619,738 tokens.
6. **DERANGEMENT control** ✓ — Pearson collapses to ~0 on all four arms (≤ |0.033|), proving real structure.
7. **Determinism** ✓ — exact-count is deterministic; std 0.0000 across seeds 42/43/44 (the seeded story-order
   permutation changes order, not counts). The derangement is a single seed-42 run (only needs to show collapse).

---

## Decision (per scoping (c) + (e))

**The §(c) decision rule:** "If richer-corpus numpy Pearson does NOT beat TinyStories numpy Pearson, the corpus
is not the lever and the run is saved." → **wikitext Pearson < TinyStories Pearson on both scales ⇒ on this
sample the corpus is NOT the demonstrated generalization lever; the multi-hour richer-corpus GPU run is NOT
justified *by the generalization argument*.**

**Honest framing for the owner (per the scoping's mandated reframe):**
- The **generalization NUMBER** (0.15 at 320 on the spiking bridge) is gated by **scale (metric granularity,
  ~0.30 Pearson) + the spiking read-out (corr(M,C)=0.756, ~0.145 Pearson)** — proven by the C0 control and now
  reinforced: even the *intrinsic numpy ceiling* is not higher for the wiki sample. A richer corpus cannot lift a
  number the read-out caps, and on this evidence it does not lift the numpy ceiling either.
- The **corpus is still a KNOWLEDGE-BREADTH lever** (TinyStories caps clusterable concepts at ~680; reaching
  ~1,000–2,000 needs Simple-Wiki/BabyLM). If breadth is the goal, adding Simple-Wiki is sound — but it should be
  framed as "more concepts," not "higher generalization."
- **If the corpus-as-generalization-lever question is to be settled definitively** (beyond this thin sample), the
  cheap next step is NOT the GPU run — it is a **per-word-count-matched** numpy re-probe on the full Simple-Wiki
  dump (still CPU, free): match per-word occurrence counts (not just total size) so the comparison isolates
  within-category co-occurrence *shape* from Zipf depth, and/or score a Wiki-appropriate taxonomy. Only if THAT
  shows a numpy-ceiling lift does a GPU corpus run earn its hours.

**Provenance trail (full, per anti-cheat 7):** the corpus-content hypothesis was already refuted three times
(yardstick-swap 0.153→0.167; content-vocab filter →0.125; the C0 decomposition scale ~0.30 / spiking ~0.145).
**This probe is the 4th independent line of evidence against it** — and the first apples-to-apples corpus-vs-corpus
test at matched size + matched a-priori labels. Caveat preserved: the full dump (~15× larger, deeper Zipf) +
per-word-count matching could differ; this thin, density-confounded sample is evidence-of-absence, not refutation.

---

## Artifacts
- Runner: `research/runners/_corpus_compare_gen_probe.py` (CPU numpy; reuse-by-import of the C0 pipeline body)
- Data: `research/findings/raw/_corpus_compare_gen_probe.json`
- Reuses: `_curriculum_gen_C0_substrate_vs_scale.run_seed_numpy_exact`,
  `dendritic_d1_learn_graded_structure_derisk.{_cos_sim,_pearson_vs_Strue,heldout_generalization}`,
  `option_c_real_cooccurrence_derisk.{TAXONOMY_8x8,taxonomy_to_vocab_categories}`,
  `stream_taxonomy_320.TAXONOMY_40x8`, `corpus_stream.iter_stories`,
  `_phaseB_online_stream_cortex_derisk.{WINDOW,EMA_ALPHA,double_center}`,
  `option_c_stageB_fair_test.STOPLIST`.
