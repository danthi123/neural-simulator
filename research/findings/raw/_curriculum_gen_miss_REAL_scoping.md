# Foundational-curriculum Step-1 generalization miss is REAL (not the yardstick) — read-only scoping

**Date:** 2026-06-25
**Type:** RESEARCH-GATE scoping (read-only; NO edits/runs/webapp). Standing-practice deep-research at a MULTIPLY-CONFIRMED boundary BEFORE building a fix.
**Trigger:** the cheap measurement-fix predicted by the prior scoping (`_curriculum_gen_miss_scoping.md`, "swap the gen reference → predicted ≥ 0.80") **FAILED**. Re-measuring the real-corpus-320 codes against the VALIDATED coherent reference moved generalization only **0.153 → 0.167** (Pearson(cos,S_true) **+0.070 → +0.082**). The wrong-yardstick hypothesis is **REFUTED** — the gen miss reproduces on BOTH references, so it is the **CODES**, not the reference. Diagnose WHY real-corpus-320 doesn't generalize when the validated run did, before building a fix.
**Result JSONs:** `_curriculum_step1_320_real_corpus_seed42.json` (gen 0.153, sharding ref); the coherent-ref 0.167 is the prompt-reported re-measure.
**Runner:** `research/runners/_curriculum_step1_320_real_corpus.py`

---

## TL;DR verdict — the dominant cause changed; it is REAL, and it is the VOCAB, not the reference

**The prior scoping was over-confident and is now corrected by its own anti-cheat #1 (the reference must be independent — it was, and swapping it did NOT fix the miss).** The decisive, source-verified fact:

> The SAME winning pipeline (count-Hebbian `M[t,h]` + `double_center(log1p(M·100))`) on the SAME real TinyStories corpus scores generalization **0.91 / Pearson +0.513** on the **64 hand-curated co-occurrence-COHERENT words** (`TAXONOMY_8x8`, CYCLE 94) but **0.15–0.17 / Pearson +0.07–0.08** on the **frequency-selected top-320 words**. Normalization is byte-identical; the learn rule family is the same; the corpus is the same. **The only material variable is WHICH WORDS are in the vocabulary.**

**Dominant cause = (b) the CORPUS-FREQUENCY VOCAB SELECTION, quantified:** the frequency-derived 320 is **48% adjective / function / emotion words** (`texture_material_adj` 27, `abstract_relations` 21, `emotion_states` 21, `time_words` 19, `spatial_words` 19, `question_discourse` 14, `color_adj` 12, `quantity` 11, `size_shape_adj` 10 = **154/320**), and its **most-frequent words — learned first — are `hey, day, big, very, happy, there, can, did, too, could`** (function words + adjectives). These high-frequency words **share context with everything**, so a distributional cortex (correctly) gives them similar, near-uniform codes — and because they ARE the most frequent words, they also dominate the hub-context dimension and **homogenize the codes of the genuine entities too**. This is **not** the yardstick (proven: the coherent reference, scored over the 66% gen-usable coherent subset, still reads 0.167 / +0.082). It is the codes — and the codes are bad **because the vocab is dominated by distributionally-flat words**, which is a **property of "top-N-by-frequency over TinyStories," not of the cortex.**

**Decisively NOT the cause:** (a) SCALE — RULED OUT (the validated 320 used a curated balanced semantic vocab and held); (c) NORMALIZATION — RULED OUT (byte-identical `double_center(log1p(M·100))`, line-by-line confirmed, AND the same pipeline gives 0.91 on the curated vocab); (d) n_hub/window — minor.

**VERDICT: CLOSEABLE, and the fix is a CURRICULUM/VOCAB-CONSTRUCTION fix, not a new mechanism — but with a real, honest scope caveat.** The cheapest decisive option is to **build the curriculum from a corpus-frequency-ranked list that is FILTERED to co-occurrence-coherent content words** (drop the adjective/function/emotion-flat categories; keep entities + verbs), which is exactly what made `TAXONOMY_8x8`/`TAXONOMY_40x8` work — a balanced, coherent **content-word** vocab — while keeping the words corpus-derived (frequency-ranked within the coherent set, NOT hand-curated meanings). **Honest residual the owner must accept:** "real-corpus AND top-by-frequency AND clusters-into-the-a-priori-categories" cannot all three hold over TinyStories — the most frequent words ARE distributionally flat. The generalizable foundational vocab is the corpus-frequent **content/entity** subset, not the raw frequency top-N. The recall-1.000 + moat-0-FA WIN stands regardless (reference-independent; confirmed).

---

## 1. DIAGNOSIS — isolate the gen miss, QUANTIFIED (the dominant cause CHANGED from the prior scoping)

### The decisive controlled comparison (3 runs, all REAL TinyStories co-occurrence — verified in source)

| run | vocab | n_hub | n_per | learn rule | read-out | gen | Pearson(cos,S_true) |
|---|---|---|---|---|---|---|---|
| `_phaseB_online_stream_cortex_derisk` (CYCLE 94) | **`TAXONOMY_8x8` (64 curated coherent)** | 500 | 1 (numpy) | count `M[t,h]+=1` | `double_center(log1p(M·100))` | **0.91** | **+0.513** |
| `_curriculum_step1_320_real_corpus` (this run) | **freq-top-320 of `g20` taxonomy** | 300 | 16 | rate-Hebbian (bridge) | `double_center(log1p(M·100))` | **0.153** (shard) / **0.167** (coherent) | **+0.070 / +0.082** |
| `option_c_real_cooccurrence_derisk` | `TAXONOMY_8x8` (64 curated) | — | — | Oja homeostatic | divnorm_spreading | 0.308 | +0.015 |

The first two rows isolate the cause: **same corpus, same read-out normalization (byte-identical), same learn-rule family (Hebbian count) — only the VOCAB differs (64 curated-coherent → 320 frequency-selected), and gen collapses 0.91 → 0.15.** The read-out is verified identical (`_phaseB_online_stream_cortex_derisk.py:103` and `_curriculum_step1_320_real_corpus.py:370` both `double_center(np.log1p(M * 100.0))`; both import the SAME `heldout_generalization` / `_cos_sim` / `_pearson_vs_Strue` from `dendritic_d1_learn_graded_structure_derisk.py`).

### Cause (a) — SCALE (64 → 320). **RULED OUT.**

The validated stream cortex **already scaled to 320 and held** — but on a **curated balanced semantic vocab** (`stream_taxonomy_320.TAXONOMY_40x8`, 40 categories × 8 coherent words, freq≥50): on-bridge recall 1.00 + moat 0-FA + (the gate's) gen, 150K windows, 9920 neurons (CYCLE 96, `2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md:203-220`). So **320 per se is not the wall.** What changed between CYCLE-96-320 (held) and Step-1-320 (missed) is **only the vocab construction**: curated-balanced-coherent → frequency-top-N-over-the-sharding-spec.

### Cause (b) — CORPUS-FREQUENCY VOCAB SELECTION. **DOMINANT (quantified).**

The frequency-derived 320 (verified from the result JSON's `curriculum_developmental_order` + `per_category_count`):

- **48% (154/320) are in distributionally-FLAT categories** — adjectives/function/emotion words that modify or accompany *different* nouns and share no mutual co-occurrence neighbourhood:

  | flat category | count in chosen-320 | sample members |
  |---|---|---|
  | `texture_material_adj` | **27** | hot, cold, warm, wet, dry, soft, hard, clean, shiny… |
  | `abstract_relations` | **21** | very, can, did, too, could, would, because, if, also… |
  | `emotion_states` | **21** | happy, sad, scared, excited, proud, angry, glad… |
  | `time_words` | **19** | day, then, when, again, always, now, after, soon… |
  | `spatial_words` | **19** | there, back, down, around, outside, inside, near… |
  | `question_discourse` | **14** | hey, what, who, how, why, okay, yes, please… |
  | `color_adj` / `size_shape_adj` / `quantity` | 12 / 10 / 11 | red/blue; big/little/tiny; many/more/two… |

- **The TOP of the curriculum — the words learned FIRST and most — is dominated by these flat words:** `hey`(19729), `day`(15337), `big`(12427), `very`(9499), `happy`(9243), `play`, `little`(8401), `there`(7691), `can`(4466), `did`(4098), `too`(4023), `sad`(3987), `could`(3954). The first genuine entity is `bird` at rank 10.

- **These high-frequency flat words ARE the most-frequent words, so they also overlap the role of the hub-context dimension and act as shared context for everything → they homogenize the entity codes too.** This is why even restricting the score to coherent words doesn't recover it (below).

- **The frequency-vocab and the validated curated vocab are only 51% the same word-set** (162/320 overlap; 158 freq-words absent from `TAXONOMY_40x8`; 158 curated words too rare to be top-320). So the frequency selection produces a **materially different, flatter vocabulary** than the one the validated runs used.

- **Uneven category fill** (a facet of (b)): `texture_material_adj` 27 vs `fish_reptiles`/`hand_tools` 2; only 30/32 categories present (`machines`, `vegetables` absent). The entity categories that COULD cluster are thin (`mammals` 11, `birds` 4, `fruits` 3, `insects` 3).

**The yardstick-swap proof that (b) — not the reference — dominates:** re-scoring against the coherent reference (validated `TAXONOMY_40x8` + coherent g20 entity/verb domains), over the **66% (211/320) gen-usable coherent subset**, gives gen **0.167** / Pearson **+0.082** — barely above the sharding number. If the cause were the reference (the prior hypothesis), restricting to coherent labels would have lifted gen toward 0.80. It did not. **⇒ the coherent-subset words ALSO fail to cluster, because their codes are washed out by the 48% flat high-frequency words that dominate the learning + the hub context.** Pearson +0.08 vs the validated +0.41–0.52 is a near-orthogonal code-similarity structure — a CODE failure, not a label failure.

### Cause (c) — NORMALIZATION / PPMI. **RULED OUT (twice).**

(i) Byte-identical read-out in both runs (`double_center(np.log1p(M*100.0))`, line-confirmed). (ii) The same normalization gives **0.91 on the curated vocab** — so it is not the limiter. The prior scoping's ruling stands and is strengthened: PPMI/normalization is not the cause. (NB: the `double_center(log1p)` IS the validated log-domain PPMI-equivalent local normalization, CYCLE 88/93b — both runs use the full validated normalization, not a reduced version. The prior-scoping "byte-identical" claim is CORRECT.)

### Cause (d) — n_hub (300 vs 500) / window / n_per. **MINOR.**

n_hub 300 (320-run) vs 500 (64-run): a smaller hub pool over a 5× larger target vocab gives each concept a thinner context fingerprint — a plausible few-points contributor, NOT a 0.91→0.15 collapse. window=2 identical. n_per 16 (320, population) vs 1 (64, numpy single) should HELP the 320 run (population averaging lifts fidelity, CYCLE 91), so it cannot explain the miss. These are worth a cheap sweep but are not the dominant cause.

### Dominant-cause ranking (pinned, corrected from the prior scoping)

1. **(b) CORPUS-FREQUENCY VOCAB SELECTION — DOMINANT.** 48% distributionally-flat adjective/function/emotion words (the most-frequent ones, learned first, also the hub context) → codes homogenized → Pearson +0.07–0.08 on BOTH references; even the 66% coherent subset fails. This is a property of "top-N by raw frequency over TinyStories," not the cortex.
2. **(d) n_hub 300 vs 500 + thin entity categories — minor.** A few points, not the collapse.
3. **(a) scale — RULED OUT** (curated-320 held).
4. **(c) normalization — RULED OUT** (byte-identical AND gives 0.91 on the curated vocab).

**The wrong-yardstick hypothesis (prior scoping) is REFUTED and explicitly retracted here.** The prior scoping correctly ruled out (c) and correctly identified the `g20` sharding taxonomy as a poor reference — but it OVER-predicted that swapping the reference would clear 0.80. It would not, because the codes themselves are flat (the 48% function/adjective words wash them out). The reference was *a* problem; the VOCAB is *the* problem.

---

## 2. REFRAME via biology + the validated CYCLE-88→96 arc (what EXACTLY the validated run did that this doesn't)

**The validated arc already answered "can a distributional cortex generalize from real text" — YES — but it always learned from a CURATED, co-occurrence-COHERENT CONTENT-WORD vocabulary, and it DELIBERATELY EXCLUDED the flat words this run includes.**

The validated reference's own design docstring is explicit (`stream_taxonomy_320.py:16-23`): *"Each category must be a REAL semantic domain whose members genuinely SHARE co-occurrence context… ABSTRACT / FUNCTION words (thought, something, about, when) are deliberately EXCLUDED — they don't form clean co-occurrence clusters and would only add noise."* And `option_c_real_cooccurrence_derisk` design SS1 builds `S_true` ONLY from an a-priori taxonomy of **content words**. The validated 0.91 was obtained on a vocab from which exactly the words that dominate this run (function/adjective/emotion) were filtered OUT.

**The biology (the load-bearing reframe):** distributional semantics (Harris/Firth — "a word is known by the company it keeps") makes **co-occurring words similar**. Function words ("can", "very", "too", "if") and cross-domain adjectives ("hot", "big", "happy") **co-occur with everything**, so their distributional signature is **near-uniform** — biologically and information-theoretically they do NOT form a semantic cluster. Convergence-zone / hub-and-spoke semantic biology (Patterson–Lambon Ralph ATL; Pulvermüller distributed cortical word ensembles, catalog **G.20**) represents category structure **carried by shared sensory/linguistic context** — entities cluster because they appear in entity contexts. **A child's semantic cortex does NOT cluster "texture adjectives" or "function words" as a category either.** So the gen miss is the *correct* behaviour of a distributional cortex on a flat-word-heavy vocab — and the fix is to give the brain the **content-word** vocabulary that DOES carry recoverable category structure (entities + verbs), exactly as the validated runs did, while keeping it corpus-derived.

**What the validated run did that this one doesn't (the precise diff):**
1. **Vocab = curated co-occurrence-coherent CONTENT words** (entities/verbs, ~8/category, balanced) — NOT top-N-by-raw-frequency (which is 48% flat words).
2. **Function/adjective/emotion words filtered OUT of the target set** (they are CONTEXT, i.e. hubs, not concepts to cluster).
3. Balanced category fill (8/category) — not 27-vs-2.
Everything else (corpus, learn rule, normalization, gates) is the same. The Step-1 "engineering piece" (`derive_curriculum_from_corpus`, frequency-rank the `g20` taxonomy members) reproduced (1)–(3) WRONG: it ranked by raw frequency over a taxonomy that *contains* the flat categories, so the flat words won the top slots.

---

## 3. RANK cheap-first options (closing the gen lift) — these fix the VOCAB, not the reference

Ordered cheapest-first. The headline insight: **filter the corpus-frequency curriculum to co-occurrence-coherent CONTENT words** (the validated recipe), keeping words corpus-derived (frequency-ranked within the coherent set), NOT hand-curating meanings.

| # | Option | What it does | Cost | Expected on gen |
|---|---|---|---|---|
| **1** | **Frequency-rank within the COHERENT content categories only** (drop `texture_material_adj`/`abstract_relations`/`emotion_states`/`time_words`/`spatial_words`/`question_discourse`/`color_adj`/`size_shape_adj`/`quantity` from the TARGET set; keep entity + verb categories; take the top-N by frequency *within* them). The dropped words become HUBS (context), not concepts. | Aligns the corpus-derived vocab with the validated content-word recipe; words stay frequency-ranked (corpus-derived), not hand-picked meanings. | **minutes edit + 1 GPU re-run** (~75 min, or instant re-score if a code dump exists). The `_coherent_category_map` already exists in the runner — reuse it to FILTER the curriculum, not just to re-score. | **Predicted ≥ 0.80** — this is the validated recipe (curated coherent content words) made corpus-frequency-ranked. The decisive test. |
| **2** | **Re-balance the kept categories** (cap members per category at ~8–10 so no category is 27; ensure ≥4–6 per kept category; this is what `TAXONOMY_40x8` enforces). | Removes the uneven-fill facet of (b). | **minutes edit + same re-run as #1.** | Combined with #1, should clear 0.80 cleanly + tighten multi-seed variance. |
| **3** | **Reproduce the VALIDATED `stream_taxonomy_320.TAXONOMY_40x8` vocab directly** (it is already curated, balanced, freq≥50, validated 320). Stream the real corpus to those 320 words. | The cleanest apples-to-apples reproduction of the validated 320 GO on the real-corpus pipeline; confirms the pipeline, isolates that ONLY the frequency-selection was the issue. | **1 GPU re-run.** | **Predicted ≥ the validated number** (this is literally the validated vocab). The control that proves the pipeline is fine. HONEST: this is curated, not corpus-derived — use it as the CONTROL, with #1 as the corpus-derived production path. |
| **4** | **A richer / fact-denser corpus** (BabyLM-10M or Simple-English-Wikipedia layered in) | TinyStories is narrative and category-flat (animals all appear as pets in play scenes); a fact-denser corpus (Simple-Wiki: "a dog is an animal", taxonomic statements) gives sharper entity-category co-occurrence. Addresses whether TinyStories is *intrinsically* too flat for FINE entity separation even on content words. | **corpus download + loader + a longer GPU run** (~hours). | **Uncertain lift** — only worth it IF #1–#3 under-clear (i.e. if even coherent content words don't separate on TinyStories). The catalog/BabyLM scoping already ranks Simple-Wiki as the fact-density complement. |
| 5 | **n_hub 300 → 500 (match the 64-run) + window/n_per sweep** | Bounds cause (d). | **minutes + 1 re-run.** | **Small** (a few points); fold into #1's re-run, not a standalone fix. |

**Recommended cheap-first bundle:** **#1 (coherent-content-word curriculum) + #2 (re-balance) + #3 (the curated-320 control), all in ONE 3-seed GPU pass** (the runner already loads the corpus + has `_coherent_category_map`; the edit is to FILTER `derive_curriculum_from_corpus` to coherent content categories + cap per-category, then ALSO run the curated `TAXONOMY_40x8` as the control arm). #4 (richer corpus) and #5 (n_hub) are follow-ons only if #1–#3 under-clear. Bump n_hub to 500 in the same run for free.

**Is TinyStories intrinsically too category-flat (sub-question b)?** The evidence says **NO for the validated content-word vocab** (TinyStories + 64 curated content words → 0.91; + 320 curated content words → CYCLE-96 GO), but the open question is whether the *frequency-ranked content subset* (option #1) separates as well as the *hand-curated* one — i.e. whether the corpus-frequent entities (dog/cat/bird/ball/tree/park…) cluster as cleanly as the hand-balanced set. #1 vs #3 measures exactly this gap. If #1 lands materially below #3, that is the honest "frequency-selection-vs-curation" residual, and #4 (richer corpus) is the lever. The `option_c` 64-curated `BOUNDARY_weak_graded` (gen 0.308, host ceiling 0.535) is a CAUTION FLAG: it used a LOSSIER learn+readout (Oja+divnorm), but its low host ceiling (PPMI+SVD only 0.535/Pearson 0.126 on the 64 curated words) hints TinyStories co-occurrence for the FULL 64 (incl. colors/body) is harder than the `_phaseB` 0.91 suggests — so the read-out pipeline matters and #1 must use the WINNING (count + double-center) pipeline, which the runner already does.

---

## 4. ANTI-CHEATS + cheap-first de-risk + GO bars + VERDICT

### Anti-cheats (all mandatory; the curriculum construction is exactly where circularity hides)

1. **The gen reference MUST stay INDEPENDENT / a-priori (not corpus-derived).** Whether scoring against `TAXONOMY_40x8` (option #3) or the coherent-content categories (#1), `S_true` = the a-priori category-block matrix, NEVER the corpus co-occurrence (the load-bearing correctness property, `option_c` design SS1). Filtering the curriculum to coherent content categories restricts WHICH words are learned/scored — it does NOT smuggle the corpus into `S_true` (the labels are still the a-priori taxonomy).
2. **The vocab filter must NOT be tuned on the gen score.** The "coherent content categories" set must be fixed a-priori (entities + verbs; drop adjective/function/emotion) — declared BEFORE the run, not searched to maximize gen. Report the exact dropped/kept category list. (The runner's `COHERENT_G20_DOMAINS` is already an a-priori declaration — reuse it verbatim, don't re-tune it.)
3. **No loosening of any other bar.** recall ≥ 0.95 and **moat 0 false-accepts** MUST still hold on the new (smaller, content-word) vocab. They are reference-independent (read the codes, not `S_true`) — assert they stay ≥ the Step-1 1.000 / 0-FA. (Honest note: a content-word-only vocab gives FEWER noun/verb facts to bind; confirm n_facts is still meetable — the runner's `_make_svo_facts` already filters to nouns/verbs, so a content-word vocab should HELP recall, not hurt it.)
4. **Frozen-brain control** (`plasticity_on=False`) MUST still show competence-flat (corr(M,C)~0, recall<0.5, gen≤learned) — already PASS in Step-1; re-assert.
5. **Derangement control** (shuffle category labels → gen collapses to ~chance) on the new vocab/reference — confirms the lift is real structure, not a denser/sparser reference inflating chance. Report gen AND ratio_vs_chance (chance = 1/n_categories changes when categories are dropped).
6. **PROVENANCE — report the frequency-top-320 number (0.153/0.167) ALONGSIDE the new number.** The finding is "the corpus-FREQUENCY top-320 is 48% distributionally-flat words a cortex cannot cluster (gen 0.15–0.17, Pearson +0.07–0.08 on BOTH references — a clean result about what raw frequency selects); the corpus-frequency CONTENT-word vocab (entities+verbs) generalizes (gen X) — the validated recipe, corpus-derived." Do NOT hide the 0.15 (it is the scientifically-correct measurement of "top-N-by-raw-frequency" and the reason the curriculum construction must filter to content words). **Foreground the REFUTED prior hypothesis: the reference-swap did NOT fix it; the vocab did.**

### Cheap-first de-risk

**One 3-seed GPU pass** (seed 42 first; 43/44 if it clears) of `_curriculum_step1_320_real_corpus.py` with `derive_curriculum_from_corpus` FILTERED to coherent content categories + per-category cap (option #1+#2), PLUS a control arm streaming the curated `TAXONOMY_40x8` (#3). Bump n_hub 300→500. ~75 min/arm GPU (the runner already loads the corpus + the coherent map). If a Step-1 code dump exists, the coherent-content RE-SCORE is an instant host check first (re-score the EXISTING 320 codes over only the coherent-content rows) to bound the lift before any GPU.

### GO bars

- **Primary (the fix is real):** generalization ≥ 0.80 on the **independent a-priori content-word reference** (coherent-content categories and/or `TAXONOMY_40x8`), 3 seeds, with derangement collapse + frozen-flat + recall ≥ 0.95 + moat 0-FA all holding. Pearson(cos,S_true) ≥ +0.40 (the validated band; the load-bearing tell, vs the current +0.08).
- **Control confirmation (pipeline is fine):** the curated-`TAXONOMY_40x8` arm (#3) reproduces ≥ the CYCLE-96 validated gen — proving ONLY the frequency-selection was the issue.
- **Frequency-vs-curation residual (honest):** if the corpus-frequency content-word arm (#1) lands materially below the curated arm (#3), report the gap as the "raw-frequency-vs-curated-content" residual and trigger option #4 (richer corpus) as the next lever.

### VERDICT

**CLOSEABLE — and the dominant cause is REAL (the codes), localized to the VOCAB-CONSTRUCTION, not the reference, not a substrate/learning boundary.** The standing-practice deep-research overturned the prior scoping's comfortable "swap the yardstick" verdict (which its own anti-cheat #1 then refuted): the gen miss reproduces on BOTH references (Pearson +0.07/+0.08), so it is the codes — and the codes are flat because the **corpus-frequency top-320 is 48% distributionally-flat adjective/function/emotion words** (the most-frequent words, learned first, also the hub context), which a distributional cortex correctly cannot cluster. The SAME pipeline + SAME corpus + SAME (byte-identical) normalization gives gen **0.91** on the **64 curated co-occurrence-coherent CONTENT words** — proving the mechanism, normalization, and scale are all fine; ONLY the frequency-driven vocab selection (which the Step-1 "corpus-derived curriculum" engineering piece introduced) is the cause. **The fix is to frequency-rank within co-occurrence-COHERENT CONTENT categories (entities + verbs) — the validated recipe, kept corpus-derived — not a new mechanism, not the dendritic rewrite.** A ~minutes-edit + 1–3-seed GPU re-run (the runner already has the coherent map + corpus loader).

**Honest scope (the genuine residual the owner must accept):** "real-corpus AND top-by-raw-frequency AND clusters-into-a-priori-categories" cannot all three hold — the most frequent TinyStories words ARE distributionally flat (that is a true property of the corpus + distributional semantics, not a failure). The generalizable foundational vocabulary is the corpus-frequent **content/entity** subset (~52% of the top-320 are content words; option #1 keeps those), not the raw frequency top-N. If, after #1, the corpus-FREQUENCY content-word vocab still under-clears vs the CURATED content-word vocab (#3), THAT is the real "frequency-selection-vs-curation" boundary, and a richer/fact-denser corpus (Simple-Wikipedia / BabyLM-10M, #4) is the precise next lever — a corpus property, addressable, not a substrate wall. **The recall-1.000 + moat-0-FA WIN stands regardless** (reference-independent; the codes are distinguishable enough to bind, retrieve, and abstain at 320 real-corpus concepts — only the FINE category-similarity structure needed for generalization is washed out by the flat words).

**This is NOT a fire-the-gate-to-build-a-new-mechanism: the mechanism (Hebbian co-occurrence + log-double-centre normalization + population code) is validated and reproduces 0.91 on a coherent content-word vocab. It is a curriculum-construction fix gated behind the standing anti-cheat that the gen reference + the vocab filter be independent + a-priori.**

---

## Sources / artifacts (read-only, verified this session)

- `research/findings/raw/_curriculum_step1_320_real_corpus_seed42.json` — the result (gen 0.153 sharding, Pearson +0.070, corr(M,C) +0.756, recall 1.0 (48/48), moat 0-FA (0/48), frozen-flat PASS, VRAM 4911 MB). `per_category_count` (texture 27 / abstract 21 / emotion 21 / time 19 / spatial 19 …), `top25` (hey/day/big/very/happy…), `tail10` (≥169). The coherent-ref 0.167 / Pearson +0.082 is the prompt-reported re-measure (no separate JSON read this session).
- `research/runners/_curriculum_step1_320_real_corpus.py` — `_category_map` (`:104`, `g20_vocab_spec_2048`), `_coherent_category_map` + `COHERENT_G20_DOMAINS` (`:130-164`, the a-priori coherent map — REUSE to FILTER, not just re-score), `derive_curriculum_from_corpus` (`:200`, frequency-rank over the FULL `g20` taxonomy = the bug), `read_codes` (`:370`, `double_center(np.log1p(M*100.0))`), `measure_generalization` + `--gen-reference` (`:526`, the 2026-06-25 reference-swap that did NOT fix it).
- `research/runners/_phaseB_online_stream_cortex_derisk.py` — the VALIDATED **0.91 / +0.513** run: `TAXONOMY_8x8` (64 curated content words, `:43`/`:122`), REAL TinyStories (`load_token_stream`, `:56`), count `M[t,h]+=1` (`:99`), `double_center(np.log1p(M*100.0))` (`:103`) — the SAME read-out as the 320-run, proving the vocab is the only material difference.
- `research/runners/option_c_real_cooccurrence_derisk.py` — `TAXONOMY_8x8` (`:108`); the INDEPENDENT-`S_true` correctness property (design SS1, `:31-37`, `:224-226`); the `BOUNDARY_weak_graded` 64-word real-corpus result (`_option_c_real_cooccurrence_multiseed.json`: pearson_struct +0.015, gen 0.308, host ceiling Pearson +0.126 / gen 0.535) — a LOSSIER (Oja+divnorm) pipeline; caution that TinyStories co-occurrence for the full 64 incl. colors/body is non-trivial, so #1 must use the WINNING count+double-center pipeline (it does).
- `research/runners/stream_taxonomy_320.py` — the VALIDATED 320 reference + vocab: `TAXONOMY_40x8` (`:83-147`, 40×8 balanced coherent CONTENT words, freq≥50); docstring (`:16-23`) states the coherence requirement + that abstract/function words are DELIBERATELY EXCLUDED (the exact words this run includes).
- `research/runners/learned_graded_cortex_fair_test.py` — `build_real_corpus` (real TinyStories, `TAXONOMY_8x8`) — confirms the 0.91 run's batch reference is real-corpus, not synthetic.
- `research/runners/dendritic_d1_learn_graded_structure_derisk.py` — `heldout_generalization` (`:187`), `_cos_sim` (`:128`), `_pearson_vs_Strue` (`:134`) — the identical gen metric imported by BOTH the 0.91 run and the 320-run.
- Findings: `2026-06-15-biology-faithful-online-stream-cortex-reaches-target.md` (gen 0.91 / Pearson +0.513, on the 64 curated content words); `2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md` (CYCLE-96 curated-320 GO; `:191-201` notes g20 is GRAMMATICAL not semantic — the early warning that frequency-over-g20 would be flat); `2026-06-15-off-diagonal-red-herring-ppmi-local-normalization-reaches-host.md` (local normalization is the gen lever — confirmed present in both runs).
- Prior scoping (REFUTED-and-corrected here): `research/findings/raw/_curriculum_gen_miss_scoping.md` (predicted the reference-swap clears 0.80; it moved 0.153→0.167 — the wrong-yardstick hypothesis is refuted; the VOCAB, not the reference, is the cause).
- Scaling scoping: `research/findings/raw/_foundational_curriculum_scaling_scoping.md` (§5 Step-1 "vocab derived from the corpus by frequency, NOT the hand-curated taxonomy" = the source of the flat-vocab bug; §1 ranks Simple-Wiki / BabyLM as the richer-corpus lever for option #4).
- Catalog: `E:\Documents\Projects\sim-catalog\references\feature-catalog.md` (G.20 Pulvermüller distributed cortical word ensembles — category structure = shared linguistic context; entities cluster, function/adjective words do not).
