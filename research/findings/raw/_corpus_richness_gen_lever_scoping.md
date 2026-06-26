# Corpus richness as a generalization lever for the stream cortex — scoping (READ-ONLY)

**Date:** 2026-06-26
**Type:** RESEARCH-GATE scoping (read-only; NO edits/runs/webapp/GPU — GPU busy with a training run). Deep-research on a NEW-direction lever (which corpus maximizes distributional category structure at substrate scale).
**Question:** *What text corpus gives the RICHEST category structure for the spiking stream-cortex's GENERALIZATION, at the substrate's scale (~320–2000 concepts), local-first (no >24 GB VRAM, one RTX 3090)?*
**Builds on (verified this session):** `_curriculum_gen_miss_DEEP_scoping.md` (2026-06-25) + the **C0 control that ran AFTER it** (`_curriculum_gen_C0_substrate_vs_scale.json`), `_curriculum_gen_miss_REAL_scoping.md`, `_knowledge_scaling_first_chat_scoping.md`.

---

## TL;DR — the recommendation, and the load-bearing caveat that reorders it

**RECOMMENDATION (corpus, cheapest-first):**
1. **DO NOT commit a multi-hour richer-corpus training run yet.** The premise of this question — *"richer corpus → stronger category structure → higher Pearson generalization"* — is **partly confounded** and was **NOT** the dominant cause of the current gen miss. A C0 control run (3 seeds, completed 2026-06-25, *after* the deep scoping was written) **decomposed the 0.91→0.15 gen collapse**: ~**0.30** of the Pearson drop is **scale** (64/8-cat → 320/40-cat, present even in a *noise-free numpy exact count*) and ~**0.145** is the **spiking read-out fidelity** (corr(M,C)=0.756). The corpus/vocab hypothesis is now **refuted a third time** (same TAXONOMY_40x8, same normalization, only substrate+scale changed). **So a richer corpus is a real lever for KNOWLEDGE BREADTH, but it is NOT the binding constraint on the current 320 generalization number — and a richer corpus cannot lift a number the spiking read-out is capping.**
2. **The CHEAP de-risk that MUST run first** (CPU, no GPU, minutes): the **host numpy exact-count corpus-comparison probe** — derive codes from a small Simple-English-Wikipedia (or CHILDES) sample with the *validated numpy pipeline* (`M[a,b]+=1`), score Pearson(cos, S_true) on the **same a-priori TAXONOMY_40x8/8x8 labels**, and compare to the TinyStories numpy baseline (0.513 @64 / 0.215 @320). This isolates the **corpus's intrinsic category-structure ceiling** from the spiking-substrate confound with ZERO GPU. **If richer-corpus numpy Pearson does NOT beat TinyStories numpy Pearson, the corpus is not the lever and the run is saved.**
3. **IF that probe shows a richer corpus lifts the numpy ceiling**, the ranked corpus choice is:
   - **#1 (cheapest, already 50% local): expand the cached `wikitext.txt` to the FULL Simple-English-Wikipedia dump (~23.9 M words, ~120 MB).** `corpus_stream.py`'s multi-file path (`iter_stories_multi`/`load_token_stream_multi`/`normalize_corpus_paths`) **already supports it** + the curriculum runner already has `--corpus-paths`. Biggest breadth-per-byte; taxonomic statements ("a dog is an animal") sharpen entity-category co-occurrence.
   - **#2 (the developmentally-principled one): the BabyLM-10M "strict-small" corpus** (CHILDES + Gutenberg children's stories + OpenSubtitles + Simple-Wiki + dialogue), single OSF download. CHILDES is the strongest *category-structure-per-word* source in the literature (noun/nominal-dense, repetitive) — and a published result shows animals/foods/body-parts clusters emerge automatically from ~5.3 M words of it.
   - **#3 (foundational tier, later): BabyLM-100M.** Local-feasible to STREAM (sequential, bounded memory) but the per-bridge wall-clock + the open spiking-read-out cap make it premature.

**Bottom line:** the *right corpus for breadth* is **TinyStories (clean codes) + full Simple-English-Wikipedia (breadth/facts)** — the combined corpus the prior scoping already recommended — with **BabyLM-10M (CHILDES-heavy)** as the principled alternative if the numpy probe shows CHILDES's nominal density wins on category structure. **But the generalization NUMBER is gated by substrate+scale, not corpus** — so run the **CPU numpy corpus-comparison probe FIRST**; it decides whether corpus richness is even the lever before any GPU.

---

## (a) DIAGNOSIS — what corpus property drives distributional category structure, and why TinyStories is weak on it

### What actually drives recoverable category structure (distributional semantics → spiking cortex)

Generalization here = a held-out concept's code lands in its a-priori category by code similarity (`heldout_generalization`, `dendritic_d1_learn_graded_structure_derisk.py:187` returns `correct/Nc, 1/len(cats)`; the chance-independent tell is `_pearson_vs_Strue(cos, S_true)` `:134`). Distributional semantics (Harris/Firth — "a word is known by the company it keeps") makes **co-occurring words similar**. The corpus properties that produce recoverable *category* structure, in priority order:

1. **Within-category co-occurrence density** — members of a category must genuinely SHARE context (animals appear with animals, foods with foods). This is THE driver. A category is recoverable iff its members' context fingerprints overlap MORE with each other than with out-of-category words. **Function words and cross-domain adjectives ("very", "can", "hot", "big") co-occur with *everything* → near-uniform codes → NOT a recoverable cluster** (this is correct distributional behaviour, not a cortex failure; confirmed by the content-filter refutation below).
2. **Content-word ratio** — the fraction of high-frequency tokens that are entities/verbs (clusterable) vs function/adjective/emotion words (flat). Low content-ratio means the most-frequent words (learned first, also the hub-context dimension) are the flat ones, which **homogenize the genuine entity codes too**.
3. **Topical diversity / taxonomic statements** — a corpus that says "a dog is an animal", "a hammer is a tool" gives SHARP entity→category co-occurrence; a narrative corpus where all animals appear generically together "as pets in a play scene" gives only blurry, generic co-occurrence.
4. **Per-word sample sufficiency (Zipf depth)** — each category member needs enough occurrences (the project's proven floor is **freq ≥ 50**, `stream_taxonomy_320.py:28`) for a stable fingerprint. This trades off against breadth at fixed corpus size.

### Why TinyStories is weak (measured this session, confirming the prior scoping)

`data/corpus/tinystories.txt` (8.0 MB): **1,591,863 tokens, 7,349 unique types, 1,816 types f≥50, 9,828 `<|endoftext|>` docs.** Three weaknesses:

- **(i) Low content-ratio at the top.** The frequency-top-320 over the g20 taxonomy is **48% (154/320) distributionally-flat adjective/function/emotion words** (`_curriculum_step1_320_real_corpus_seed42.json` `per_category_count`: texture_adj 27, abstract_relations 21, emotion 21, time 19, spatial 19, …). The most-frequent content-ish words are `hey, day, big, very, happy, there, can, did, too, could` — the first genuine entity (`bird`) is rank 10. (My own top-30 measure: `the and a to was they he it she said … day … big … happy …`.)
- **(ii) Narrow, category-flat topics.** It is children's stories: animals-as-pets, toys, play, food-at-meals, simple feelings. NO geography, science, professions, history, how-things-work. Entities co-occur only in generic play scenes — blurry within-category structure. The validated taxonomy doc is explicit (`stream_taxonomy_320.py:46-53`): *"40 CLEAN 8-word semantic categories ARE achievable at the ≥50 floor … but the corpus is close to its ceiling there … Pushing meaningfully past 40 clean categories would require dipping below freq 50 or admitting abstract/function words."*
- **(iii) Hard content-vocab ceiling ≈ 320–680.** 681 g20-content words clear f≥50 (per the prior scoping's measure); admitting f≥20 gives ~895. So TinyStories ALONE caps the clean content vocab at the low end of the ~320–2000 target.

### ⚠️ THE LOAD-BEARING CAVEAT — the gen miss is NOT (mostly) the corpus

The question's premise ("TinyStories' weak category structure → low generalization → richer corpus fixes it") is **the hypothesis the project has now refuted three times.** The decisive evidence is the **C0 control** (`_curriculum_gen_C0_substrate_vs_scale.json`, 3 seeds, pure numpy exact count, ZERO GPU, run *after* the DEEP scoping):

| arm | substrate | vocab | n_hub | gen | **Pearson(cos,S_true)** | chance |
|---|---|---|---|---|---|---|
| validated-64 | numpy exact count | TAXONOMY_8x8 (64, 8-cat) | 500 | 0.906 | **+0.513** | 0.125 |
| **numpy-320** | **numpy exact count** | **TAXONOMY_40x8 (320, 40-cat)** | 300 | **0.453** | **+0.215** | 0.025 |
| numpy-320 (n_hub 500) | numpy exact count | TAXONOMY_40x8 | 500 | 0.488 | +0.234 | 0.025 |
| spiking-bridge-320 | **spiking** read-out | TAXONOMY_40x8 / freq-320 | 300 | **0.15** | **+0.07** | 0.033 |

**Pearson decomposition of the +0.513 → +0.07 collapse** (from the C0 JSON `decomposition_pearson_cos_Strue`):
- **scale cost (64→320): ~0.30** — present even in a *noise-free integer count* (numpy-320 only reaches +0.215). This is the 40-category metric granularity + thinner per-concept context, NOT the corpus and NOT the substrate.
- **spiking read-out cost (320): ~0.145** — numpy-320 +0.215 minus bridge-320 +0.07. The on-bridge population block-mean read of rate-Hebbian weights (corr(M,C)=0.756, not 1.0) blurs exactly the fine off-diagonal similarity generalization needs.
- **thin hub (300→500): ~0.02** — minor.

**And the corpus-content hypothesis itself was directly refuted:** the `--vocab-filter content` arm (drop the 48% flat words, keep entities+verbs, 100% gen-usable) gave gen **0.125 / Pearson +0.059** (`_curriculum_step1_contentfilter_seed42.json`) — *below* the 0.153 it was meant to fix. So removing the flat words did NOT help.

**⇒ The current 320 gen number is capped by SCALE (metric/granularity, ~0.30 Pearson) + the SPIKING READ-OUT (~0.145 Pearson), not by corpus flatness.** A richer corpus can improve the *intrinsic numpy ceiling* (the +0.215 number — IF a richer corpus has denser within-category co-occurrence) and **buys real KNOWLEDGE BREADTH** (more clusterable concepts past TinyStories' ~680), but it **cannot lift a generalization number the spiking read-out is the binding constraint on.** This is the single most important fact for this question, and it **reorders the recommendation**: prove the corpus lifts the *numpy* ceiling (CPU, free) before spending GPU, and recognize the corpus is primarily a *breadth/scale* lever, secondarily a *category-structure* lever, and NOT the fix for the 0.15 spiking number.

---

## (b) RANKED corpus options (cheapest-first)

Each: category structure provided · size · local-feasibility on one 3090 · how to obtain/stream (does `corpus_stream.py` already support it?) · expected generalization lift.

### #1 — Expand the cached `wikitext.txt` to the FULL Simple-English-Wikipedia dump  ★ cheapest, biggest breadth-per-byte

- **Category structure:** real-world entities, places, professions, science, taxonomic statements ("X is a Y"). Sharpens entity→category co-occurrence that TinyStories' narrative flatness blurs. The current 4 MB `wikitext.txt` (627 K tokens, **21,552 unique types** — 3× TinyStories' breadth already) is Zipf-thin (only 349 content words clear f≥50); the FULL dump fixes the depth.
- **Size:** full Simple-English-Wikipedia ≈ **23.9 M words / ~120 MB / ~249K articles** (web-verified). ~15× the cached shard, ~15× TinyStories tokens. Lifts content-words-f≥50 into the thousands while keeping breadth → enables the ~1,000–2,000 concept target.
- **Local-feasibility:** trivial. 120 MB text streams in bounded memory (`iter_stories`). VRAM is per-bridge (~4.9 GB / 320-bridge), unaffected by corpus size. Wall-clock is window-budget-bound, not corpus-size-bound.
- **Obtain/stream:** download one dump (Kaggle "plain-text-wikipedia-simpleenglish", or HuggingFace `rahular/simple-wikipedia`, ~120 MB), drop at `data/corpus/simplewiki.txt`. **`corpus_stream.py` ALREADY supports it** — `normalize_corpus_paths` + `iter_stories_multi`/`load_token_stream_multi` (`:131-186`) aggregate frequency+co-occurrence across the UNION; the curriculum runner already exposes `--corpus-paths` (`:826`). A file with 0 `<|endoftext|>` delimiters streams as one document (correct for window-local co-occurrence). **Zero new code.**
- **Expected gen lift:** **uncertain but the most likely positive** — taxonomic density should raise the *numpy* category-structure ceiling above TinyStories' +0.215@320. On the SPIKING run, gen rises only to the extent the read-out preserves it (still capped by the ~0.145 substrate cost). **Run the CPU numpy probe (§c) to measure the ceiling lift FIRST.**

### #2 — The BabyLM-10M "strict-small" corpus  ★ the developmentally-principled choice; best category-structure-per-word

- **Category structure:** the strongest per-word in the literature. **CHILDES (child-directed speech) is noun/nominal-dense and repetitive** — a published result (Huebner & Willits, Frontiers 2018, "Structured Semantic Knowledge Can Emerge Automatically from Predicting Word Sequences in Child-Directed Speech") shows **hierarchical category clusters (animals, foods, body-parts) emerge automatically from ~5.3 M words of CHILDES**, at **70–74% pairwise category accuracy** (best "days" ~90%, worst "times" ~50%) — direct evidence that child-directed speech carries recoverable category structure at small scale. BabyLM-10M mixes CHILDES + Gutenberg children's stories + OpenSubtitles dialogue + Simple-Wiki + BNC dialogue → diverse syntactic/semantic constructions.
- **Size:** 10 M words (the strict-small track; designed to a child's ~per-year input). Single packaged download.
- **Local-feasibility:** trivial (10 M words ≈ tens of MB; streams in bounded memory).
- **Obtain/stream:** single download from **OSF `https://osf.io/ad7qg/`** (preprocessing at `github.com/babylm/babylm_data_preprocessing`). Needs a one-time concatenate-into-one-`.txt` + lowercase (the `re.findall(r"[a-z]+")` tokenizer assumes lowercase, `corpus_stream.py:21-22`). After that, `corpus_stream.py`'s multi-file path handles it; or add it to `--corpus-paths` alongside TinyStories. ~hours of light CPU prep, no new mechanism.
- **Expected gen lift:** **the best bet for category-structure quality** (CHILDES's nominal density + repetition is exactly the within-category-density driver §a-1). The numpy probe should be run on a CHILDES sample specifically to test whether it beats Simple-Wiki on Pearson. Still subject to the spiking-read-out cap on the bridge run.

### #3 — BabyLM-100M (the full "strict" track)  ★ foundational tier, premature now

- **Category structure:** richest of all (CHILDES 29M + Gutenberg 26M + OpenSubtitles 20M + Simple-Wiki 15M + BNC 8M + Switchboard 1M = 100M; 70% child-oriented, 58% transcribed speech). Adult-vocab category breadth.
- **Size:** 100 M words. Single OSF download (same repo).
- **Local-feasibility:** **STREAM yes, RESIDENT no.** Streaming is bounded-memory (`iter_stories`), so corpus size is never a VRAM wall. The scale ceiling is per-bridge VRAM (~2K concepts ~7 bridges ~22 GB resident; the existing `TieredSynapseStore` pages past that) + wall-clock (a few overnight runs). **No >24 GB VRAM wall** (matches `feedback_long_local_runs_ok_confirm_cloud_cause` — cloud only for VRAM, not wall-clock).
- **Expected gen lift:** marginal over BabyLM-10M for the *first-chat* ~1–1.5K tier; the breadth tail is rarely probed early. **Premature** until (i) the numpy probe confirms corpus is a lever AND (ii) the spiking read-out cap is lifted or accepted.

### Not-recommended at this stage
- **Full English Wikipedia** (~5 B words, 24 GB compressed): adult-encyclopedic, NOT developmentally-plausible, vast Zipf tail dilutes per-concept density at the target scale, large prep. Over-kill; Simple-Wiki is the right Wikipedia.
- **A bespoke "taxonomic-statement" synthetic corpus** ("a dog is an animal" × N): would maximize within-category co-occurrence but risks the **environment-vs-cognition anti-cheat** (the corpus must be a natural linguistic ENVIRONMENT the brain hears, not an LLM/hand-engineered category injection). Simple-Wiki gives the taxonomic statements *naturally*.

---

## (c) The CHEAPEST-FIRST de-risk — a CPU numpy corpus-comparison probe (NO GPU, minutes)

**This is the single decisive measurement and it must run before ANY multi-hour GPU run.** It isolates the *corpus's intrinsic category-structure ceiling* from the spiking-substrate + scale confounds, using the validated noise-free numpy pipeline that the C0 control already established as the clean baseline.

**Probe design (reuse-by-import; a tiny host harness, NO `sim/` edit, NO bridge, NO GPU):**

1. Take a **small sample** (~1.5–2 M tokens, matched to TinyStories' size so size is controlled) of each candidate corpus: a Simple-Wiki sample, a CHILDES sample (BabyLM-10M's CHILDES portion), and the existing TinyStories (the baseline).
2. For EACH corpus, run the **validated numpy exact-count pipeline** verbatim (the C0 baseline): `M[target, hub] += 1` over WM window=2, then `double_center(np.log1p(M * 100.0))` (`_phaseB_online_stream_cortex_derisk.py:99,103`), `N_HUB=500`.
3. Score on the **SAME a-priori labels** for the words that exist in all corpora — use **TAXONOMY_8x8 (64, the 0.513 anchor)** AND **TAXONOMY_40x8 (320, the +0.215 anchor)** (`stream_taxonomy_320.py`). Report `heldout_generalization` gen + **Pearson(cos, S_true)** + `ratio_vs_chance` (`dendritic_d1_learn_graded_structure_derisk.py:134,187`).
4. **The comparison:** richer-corpus Pearson vs TinyStories Pearson, at matched size + matched vocab + matched a-priori labels. The ONLY varied axis is the corpus.

**Decision:**
- **If richer-corpus numpy Pearson > TinyStories numpy Pearson (materially, e.g. +0.215 → +0.35 @320):** corpus richness IS a real category-structure lever → proceed to the combined-corpus GPU run (#1 first), with eyes open that the spiking run will still be capped by the ~0.145 read-out cost (so target the numpy-ceiling-relative lift, not 0.80 absolute).
- **If richer-corpus numpy Pearson ≈ TinyStories numpy Pearson:** **the corpus is NOT the generalization lever** (the structure ceiling is the same; TinyStories was rich enough). The binding constraints are scale (metric granularity) + the spiking read-out — and the multi-hour richer-corpus run is **saved**. The corpus then matters only for KNOWLEDGE BREADTH (more clusterable concepts), which is a separate, real reason to add Simple-Wiki but NOT a generalization fix.

**Why this and not a GPU run:** the C0 control already proved the numpy substrate carries the structure (numpy-320 +0.215, ratio 18×chance) and the spiking read-out loses ~0.145 of it. Re-running on the bridge would re-confound corpus with the read-out. The CPU numpy probe is the apples-to-apples corpus comparison — **minutes, free, decisive.** (The two prior corpus-focused scopings predicted ≥0.80 confidently and were each refuted; do NOT predict — measure the numpy ceiling.)

**Secondary cheap check (also CPU, free):** corpus content-statistics on the candidate dumps before learning — content-words-f≥50 count, per-a-priori-category member count (≥4–6/category needed), top-50 content-ratio. This bounds *how many clusterable concepts* each corpus reaches (the breadth lever) independent of the gen probe. (The prior scoping measured this for the cached shards: TinyStories 681 / wikitext-4MB 349 content-words-f≥50.)

---

## (d) ANTI-CHEATS (mandatory — circularity hides in corpus + curriculum construction)

1. **Corpus = ENVIRONMENT, NOT cognition (the BRAIN-BASED-ONLY standard).** The corpus is the linguistic environment the brain HEARS; the stream cortex (online Hebbian co-occurrence in synapses) does the learning. NO LLM derives codes; NO hand-engineered category injection. A bespoke "a dog is an animal × N" corpus is borderline (engineered structure) — prefer Simple-Wiki where taxonomic statements occur *naturally*. (memory `project_communicable_brain_not_rag`.)
2. **The gen reference S_true MUST stay INDEPENDENT / a-priori, NEVER corpus-derived** (`option_c` design SS1; the load-bearing correctness property). Score against TAXONOMY_8x8/40x8 category blocks. The corpus sets WHICH words + their co-occurrence codes; the category labels come from the independent taxonomy. The numpy probe must use the SAME a-priori labels across all corpora (only the corpus varies).
3. **Generalization measured by Pearson(cos, S_true) as the chance-independent tell, NOT just nearest-category accuracy.** Chance differs across 8-cat (0.125) vs 40-cat (0.025); the raw gen number is partly a chance artifact (the C0 control showed numpy-320 ratio 18×chance is HIGHER than numpy-64's 7.25×, yet absolute gen is lower). The load-bearing target is Pearson lift (validated band +0.40–0.51), with `ratio_vs_chance` + derangement reported.
4. **The vocab/content filter must be declared a-priori, NOT tuned on gen** (`CONTENT_G20_DOMAINS` is fixed before the run, `_curriculum_step1_320_real_corpus.py:163`). A corpus swap must not be paired with a gen-maximizing vocab search.
5. **The no-confab moat MUST hold at scale on every arm** — every never-stored cue ABSTAINS (0 false-accepts); a single fabricated certainty is a HARD STOP (`measure_recall_and_moat`). Recall ≥ 0.95 stays. The moat held at 320 real-corpus (recall 1.0, 0-FA) — re-assert on any richer-corpus run. (memory `feedback_moat_not_hard_lossy_memory_ok`: moat a plus, never weakened.)
6. **Frozen-brain control** (plasticity OFF → corr(M,C)~0, recall<0.5, gen≤learned) on every arm — proves codes are LEARNED, not smuggled. **Derangement control** (shuffle labels → gen collapses to ~chance) — proves the number reflects real structure. Both already PASS at 320; re-assert.
7. **PROVENANCE — report the full refutation trail** alongside any new number: the original 0.153 (all/sharding), 0.167 (coherent), content-filter 0.125, AND the C0 decomposition (scale ~0.30 / spiking ~0.145 / hub ~0.02 in Pearson). The corpus-content hypothesis was refuted three times — do NOT hide that. A richer-corpus run's honest framing is "corpus lifts the numpy category-structure ceiling by X / buys Y more clusterable concepts", NOT "corpus fixes the 0.15 gen" (which it cannot, given the substrate cap).
8. **Size-matched comparison in the probe** — compare corpora at matched token budget so the lift is corpus *quality*, not corpus *size*. (A bigger corpus trivially gives more f≥50 words; the probe must hold size to isolate within-category density.)

---

## (e) RECOMMENDATION (which corpus, which de-risk first)

**De-risk FIRST (this is the gate):** the **CPU numpy corpus-comparison probe** (§c) — derive codes with the validated numpy exact-count pipeline from a *size-matched* sample of **Simple-English-Wikipedia** and **CHILDES (BabyLM-10M's portion)**, score **Pearson(cos, S_true)** on TAXONOMY_8x8 + TAXONOMY_40x8, compare to the TinyStories numpy baseline (+0.513@64 / +0.215@320). **Minutes, no GPU, decisive.** It answers the question's premise directly: does a richer corpus raise the *intrinsic* category-structure ceiling? If not, no corpus run is worth the GPU.

**Corpus, IF the probe shows a lift:**
- **Primary recommendation: TinyStories (cached, clean codes) + the FULL Simple-English-Wikipedia dump (#1 — ~23.9 M words, ~120 MB, already half-local as `wikitext.txt`, already supported by `corpus_stream.py`'s `--corpus-paths`).** Cheapest, biggest breadth-per-byte, natural taxonomic statements, zero new code. This is the combined corpus the prior `_knowledge_scaling_first_chat_scoping.md` already recommended for breadth.
- **Strong alternative if the probe shows CHILDES wins on category structure: BabyLM-10M (#2)** — the developmentally-principled, noun-dense, literature-validated category-structure source (animals/foods/body-parts cluster from ~5.3 M CHILDES words; 70–74% category accuracy). One OSF download + light lowercase prep.
- **BabyLM-100M (#3): defer** — local-streamable, no VRAM wall, but premature until the corpus-is-a-lever question and the spiking-read-out cap are settled.

**The honest reframe the owner should hear:** the **corpus is a KNOWLEDGE-BREADTH lever** (TinyStories caps at ~680 clusterable concepts; reaching ~1,000–2,000 NEEDS Simple-Wiki/BabyLM — that part of the question's premise is sound and the corpus recommendation stands for breadth). But the **generalization NUMBER (0.15 at 320) is gated by SCALE (metric granularity, ~0.30 Pearson) + the SPIKING READ-OUT FIDELITY (corr(M,C)=0.756, ~0.145 Pearson), NOT by corpus flatness** — proven by the C0 control + the content-filter refutation. So a richer corpus is the right move for *breadth and a likely numpy-ceiling lift*, but the path to a higher *spiking-bridge* generalization number runs through **read-out fidelity** (more windows / bigger population n_per / bigger hub — though the n_per=32/n_hub=500/300K-window hi-fi attempt **OOM'd at ~20.6 GB**, so the fidelity lever has its own VRAM cost to engineer), and the **scale/metric granularity** (curriculum category structure), *not primarily* through the corpus. **Run the free CPU numpy probe; let it decide whether corpus richness is the lever before any GPU.**

---

## Sources / artifacts (read-only, verified this session)

**In-repo (load-bearing):**
- `research/runners/corpus_stream.py` — streaming loader; `iter_stories`/`load_token_stream` (`:53,:96`) bounded-memory single-file; **`normalize_corpus_paths`/`iter_stories_multi`/`load_token_stream_multi` (`:131-186`) = the ALREADY-BUILT multi-file UNION path** (comma/os.pathsep-separated; a single bare path is byte-identical to the legacy run); tokenizer `re.findall(r"[a-z]+")` assumes lowercase (`:34,:48`). Default `data/corpus/tinystories.txt`.
- `research/runners/_curriculum_step1_320_real_corpus.py` — the Step-1 runner; `derive_curriculum_from_corpus` + `--vocab-filter {content,all,curated}` (`:238,:807`); `COHERENT_G20_DOMAINS`/`CONTENT_G20_DOMAINS` a-priori filters (`:145,:163`); `read_codes` `double_center(log1p(M·100))` (`:464-468`); `measure_generalization` + BOTH gen references (`:624`); `--corpus-paths` combined-corpus arg (`:826`); `_make_svo_facts`/`measure_recall_and_moat` (moat) (`:494,:549`).
- `research/findings/raw/_curriculum_gen_C0_substrate_vs_scale.json` — **THE DECISIVE C0 CONTROL** (3 seeds, pure numpy exact count, ZERO GPU): numpy-64 gen 0.906 / Pearson **+0.513**; numpy-320 gen 0.453 / Pearson **+0.215** (ratio 18.1×chance); numpy-320-nhub500 +0.234; spiking-bridge-320 +0.07. Decomposition: **scale ~0.30, spiking read-out ~0.145, hub ~0.02** (Pearson). Verdict: *"the 'it's the vocab' diagnosis is REFUTED a THIRD time … same a-priori TAXONOMY_40x8, same normalization, only substrate+scale changed."*
- `research/findings/raw/_curriculum_step1_contentfilter_seed42.json` — the content-vocab arm (drop flat words, 100% gen-usable): gen **0.125** / Pearson **+0.059** (BELOW the 0.153 it was meant to fix) — the corpus-content hypothesis directly refuted.
- `research/findings/raw/_curriculum_step1_320_real_corpus_seed42.json` — the `all` provenance run: gen 0.153, Pearson +0.070, corr(M,C) 0.756, recall 1.0 (48/48), moat 0-FA, VRAM 4911 MB, top25 = hey/day/big/very/happy…, 48% flat categories.
- `research/findings/raw/_curriculum_320_hifi_seed42.log` — the C2 fidelity-lever attempt (n_per=32, n_hub=500, 300K windows, content vocab) **OOM'd at 20.6 GB** (`cupy.cuda.memory.OutOfMemoryError`) — the on-bridge fidelity lever did NOT complete; its VRAM cost is a real engineering constraint. (`_curriculum_320_hifi24_seed42.log` = 30 lines, barely started.)
- `research/runners/_phaseB_online_stream_cortex_derisk.py` — the VALIDATED numpy 0.91 pipeline: `N_HUB=500` (`:47`), `TAXONOMY_8x8` (64, `:43`), `M[…]+=1` exact count (`:99`), `double_center(log1p(M·100))` (`:103`) — the EXACT pipeline the §c CPU probe reuses.
- `research/runners/stream_taxonomy_320.py` — `TAXONOMY_40x8` (40×8=320 balanced coherent CONTENT words, freq≥50); docstring `:46-53` = the TinyStories ~40-clean-category corpus ceiling (the breadth-cap fact); `:16-23` = abstract/function words deliberately excluded.
- `research/runners/dendritic_d1_learn_graded_structure_derisk.py` — `heldout_generalization` (`:187`, returns `correct/Nc, 1/len(cats)` → chance=1/#cats), `_cos_sim` (`:128`), `_pearson_vs_Strue` (`:134`) — the gen metric, identical import in all runners + the probe.
- `data/corpus/tinystories.txt` (8.0 MB, 1.59 M tok, **7,349 types**, 1,816 f≥50, 9,828 docs — measured) + `data/corpus/wikitext.txt` (4.0 MB, 627 K tok, **21,552 types**, 1,595 f≥50, **0 `<|endoftext|>`** — measured; the Simple-Wiki *sample* to expand).

**Literature (web, verified this session):**
- Huebner & Willits 2018, *"Structured Semantic Knowledge Can Emerge Automatically from Predicting Word Sequences in Child-Directed Speech"*, Frontiers in Psychology / PMC5827184 — **hierarchical category clusters (animals, foods, body-parts) emerge automatically from ~5.3 M words of CHILDES; 70–74% pairwise category accuracy** (best "days" ~90%, worst "times" ~50%). Direct evidence child-directed speech carries recoverable category structure at small scale. (Notes CHILDES's constrained activities *should* hinder learning yet structure emerges anyway; does not isolate content-word density as an independent variable.)
- BabyLM 2024 *"Findings of the Second BabyLM Challenge"* (arxiv 2412.05149) — strict-100M composition: **CHILDES 29M + Project Gutenberg children's stories 26M + OpenSubtitles 20M + Simple-Wiki 15M + BNC dialogue 8M + Switchboard 1M = 100M words, 70% child-oriented, 58% transcribed speech.** Download: **OSF `https://osf.io/ad7qg/`**, preprocessing `github.com/babylm/babylm_data_preprocessing`. Strict-small = 10M track. Budget grounded in child input (~2–7M words/yr → ~84M by age 12).
- Simple-English-Wikipedia: full dump **≈ 23.9 M words / ~120–171 MB / ~249K articles** (Kaggle plain-text-wikipedia-simpleenglish; HuggingFace `rahular/simple-wikipedia`). Full English Wikipedia ≈ 5 B words / 24 GB compressed (over-kill, not developmentally-plausible).
- BabyLM design principle (Call for Papers, arxiv 2301.11796): source selection = plausibility of child exposure (transcribed/child-directed speech) + diversity of syntactic/semantic constructions.

**Prior scopings (this builds on / corrects):**
- `_curriculum_gen_miss_DEEP_scoping.md` (2026-06-25) — pinned the substrate (numpy-exact vs spiking-population) + scale as the likely dominant cause over vocab; designed C0. **NOTE: written BEFORE the C0 run; C0 has since CONFIRMED its prediction (scale ~0.30 + spiking ~0.145, vocab refuted).**
- `_curriculum_gen_miss_REAL_scoping.md` (2026-06-25) — the corpus-flatness/content-vocab hypothesis (predicted content-filter → ≥0.80; **refuted**: content-filter gave 0.125).
- `_knowledge_scaling_first_chat_scoping.md` (2026-06-25) — the combined-corpus (TinyStories + Wikipedia) breadth recommendation + the ~1,000–1,500-concept first-chat target + the resource envelope (3090 ceiling ~2K resident, tiering past, no VRAM wall, cloud only for VRAM not wall-clock). This doc's #1 corpus recommendation = its combined-corpus, with the substrate caveat added.

**Memories:** `project_communicable_brain_not_rag` (corpus = environment; BRAIN does cognition, no LLM-free-generate; moat = never assert a fabricated fact), `feedback_moat_not_hard_lossy_memory_ok` (moat a plus, never weakened; lossy OK if it buys scaling), `feedback_long_local_runs_ok_confirm_cloud_cause` (cloud only for >24 GB VRAM, not wall-clock — relevant to BabyLM-100M streaming feasibility), `feedback_deep_research_at_roadblocks` (this scoping; deep-research-first at a new direction).
