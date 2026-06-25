# Foundational-curriculum Step-1 generalization miss (0.153 vs 0.80) — read-only scoping

**Date:** 2026-06-25
**Type:** RESEARCH-GATE scoping (read-only; NO edits/runs/webapp). Standing-practice deep-research at a boundary BEFORE building a fix.
**Trigger:** Step-1 (320 concepts, REAL TinyStories, frequency-derived curriculum, 150K windows, GPU, seed 42) landed: **recall 1.000 CLEARED, moat 0-FA CLEARED, derangement+frozen controls PASS — but generalization 0.153 MISSED (bar 0.80)**. Scope the gen miss before building a fix.
**Result JSON:** `research/findings/raw/_curriculum_step1_320_real_corpus_seed42.json`
**Runner:** `research/runners/_curriculum_step1_320_real_corpus.py`

---

## TL;DR verdict

**CLOSEABLE, cheaply — and the dominant cause is a REFERENCE/CURRICULUM-CONSTRUCTION mismatch, NOT a learning limit, NOT the tail, NOT missing PPMI, NOT scale.** The Step-1 run measured generalization against the **`g20_vocab_spec_2048` taxonomy** — a 32×64 spec built for a *different purpose* (multi-bridge concept **sharding**), whose categories are **not co-occurrence-coherent in TinyStories** (e.g. `texture_material_adj` = {hot, cold, fast, slow, clean, dirty, …} — adjectives that modify *different* nouns and never co-occur with *each other*) and which the frequency-selection populated **unevenly** (27 members in one category, 2 in another; 30/32 categories present). The validated stream-cortex that scored generalization **0.91 (CYCLE 94, numpy) / 0.45 on the on-bridge subset (CYCLE 96)** measured against a **completely different, purpose-built reference**: `stream_taxonomy_320` / `TAXONOMY_8x8` — **balanced 40×8 (and 8×8) categories hand-curated so each category's members genuinely share story context** (animals = dog/cat/bird/fish; food = apple/cake/bread/milk). A distributional/co-occurrence cortex CAN recover the second one (members co-occur → similar codes) and provably CANNOT recover the first (members don't co-occur). The read-out normalization is **byte-identical** between the two runs, so PPMI/normalization is ruled out.

**The single highest-leverage fix is to swap the generalization REFERENCE** to a co-occurrence-coherent semantic taxonomy (`stream_taxonomy_320`, already curated, validated, freq≥50, balanced) and/or **report a corpus-derived-taxonomy cluster-purity** as a second, reference-independent gen number. This is a ~minutes-of-edit + 1-bridge re-measure (or even a pure re-score of the already-learned codes IF they were cached) — not a new mechanism. **Honest caveat:** the *learning* is genuinely good (`corr(M,C)=+0.756`); the recall-1.000 WIN stands regardless of the gen reference. The gen miss is "we asked the brain to cluster by a structure the corpus does not contain," not "the brain failed to learn."

---

## 1. DIAGNOSIS — isolate the gen miss, quantified

### The numbers (seed 42, from the result JSON)

| signal | value | reads as |
|---|---|---|
| `corr(M,C)` (learning fidelity) | **+0.756** | the bridge learned the co-occurrence faithfully (healthy; cf. CYCLE-96 capstone +0.885 at 30K windows, this is 150K) |
| recall (who/what) | **1.000** (48/48) | CLEARED — the codes are distinguishable enough to bind + retrieve facts |
| moat false-accepts | **0** | CLEARED — no-confab holds at 320 real-corpus concepts |
| **generalization (held-out cat)** | **0.153** | MISSED (bar 0.80); but **4.59× chance** (chance 0.0333 = 1/30 categories) |
| derangement gen (shuffled labels) | 0.009 | control collapses correctly (gen is real signal, not an artifact) |
| Pearson(cos, S_true) | **+0.070** | the learned code-similarity barely tracks the a-priori category blocks |
| frozen-brain gen | 0.009 | control PASS (codes are LEARNED, not smuggled) |

The gen number is **above chance and the controls are clean** — so the cortex DID learn *some* structure that aligns with the reference (4.6× chance), it just doesn't reach 0.80. The question is what caps it. Four candidate causes were posed; here is the isolation.

### Cause (a) — REFERENCE mismatch (the a-priori taxonomy ≠ what a co-occurrence cortex can cluster). **DOMINANT.**

The load-bearing fact, verified in source:

- **Step-1 reference** = `g20_vocab_spec_2048.ALL_CLUSTERS_2048` (`_curriculum_step1_320_real_corpus.py:104-113`, `_category_map`). This spec's docstring (`g20_vocab_spec_2048.py:1-30`) states its purpose: *"32 hand-curated semantic super-clusters of exactly 64 mutually-similar concepts … Why semantic-cluster sharding (NOT part-of-speech): the dual/CLS … only works when a bridge's 64 concepts genuinely cluster in concept space. So each bridge must hold a TAXONOMIC cluster."* It is built for **multi-bridge SHARDING capacity**, not for co-occurrence-clusterability.
- **The validated gen runs** (`option_c_real_cooccurrence_derisk.TAXONOMY_8x8`, `stream_taxonomy_320.TAXONOMY_40x8`) use a **different reference, purpose-built for THIS metric**. `stream_taxonomy_320.py:16-23`: *"Each category must be a REAL semantic domain whose members genuinely SHARE co-occurrence context in children's stories — animals {dog, cat, bird}, food {apple, cake} … Words that co-occur with similar contexts will get similar learned codes; the metric then tests whether that learned similarity recovers the a-priori category blocks. ABSTRACT / FUNCTION words … are deliberately EXCLUDED — they don't form clean co-occurrence clusters and would only add noise to S_true."*

The mismatch is mechanical and visible in the chosen-320 category contents:

| Step-1 category (member count in chosen 320) | sample members | co-occur with **each other** in stories? |
|---|---|---|
| `texture_material_adj` (**27**) | hot, cold, warm, fast, slow, new, old, clean, dirty, wet, dry, soft, hard, sweet, strong, kind, nice, good, bad | **NO** — each modifies a *different* noun (hot soup / fast car / clean room); they share no co-occurrence neighbourhood |
| `abstract_relations` (**21**) | and, or, but, if, because, can, will, do, very, too, also, only, not, same, other | **NO** — relation/function words, scattered everywhere |
| `spatial_words` (**19**) | up, down, in, out, on, under, here, there, near, far, inside, outside | **NO** — distributed across all contexts |
| `time_words` (**19**) | now, then, before, after, today, soon, always, never, morning, night | **NO** — distributed |
| `motion_verbs` (**20**) | run, jump, walk, play, eat, sleep, … | **PARTIALLY** — co-occur with overlapping agents/scenes |
| `mammals` (**11**), `furniture` (**13**), `weather_nature` (**13**) | dog/cat/bear; table/chair/bed; rain/sun/snow | **YES** — the few coherent ones |

A distributional cortex assigns **similar codes to words that share context**. Two texture adjectives DON'T share context with each other, so the cortex (correctly) does NOT place them near each other — yet the `g20` reference demands they cluster. The metric therefore penalizes the cortex for a structure the corpus does not contain. The validated 8×8/40×8 reference deliberately uses only categories whose members DO co-occur (animals appear together as pets; food appears together at meals), so the same cortex scores 0.91.

**Quantitative tell that (a) dominates:** `Pearson(cos, S_true)=+0.070` is near zero. In the validated arc the same Pearson is **+0.41–0.52** (CYCLE 88–94). The gap is not noise on a good reference — it is a structurally near-orthogonal reference. The big categories that drive the metric (texture 27, abstract 21, spatial 19, time 19 = **86 of 320 concepts, 27%**) are precisely the co-occurrence-incoherent ones.

### Cause (b) — the thinly-learned long-frequency TAIL (~169 counts). **MINOR contributor, not dominant.**

The chosen vocab spans freq 19,729 → 169 (`freq_range`); the tail-10 are 169–173. The proven learnability floor is freq≥50 (the 64-tier's least-frequent concept fired 48×; the curated 40×8 enforces ≥50 with median ~100+). So at **169, the tail is ABOVE the floor** — thinly learned but not unlearned, and `corr(M,C)=+0.756` confirms the matrix is faithful overall. The tail would shave a few points off a *good* reference, but it cannot explain a collapse from 0.80→0.15. (The de-risk in §5 includes a high-freq-subset check to bound this precisely; the prediction is that it lifts gen only modestly on the `g20` reference because the *dominant* cap is the incoherent big categories, which include high-freq words like "hot/cold/big".)

### Cause (c) — missing PPMI normalization / per-hub double-centring. **RULED OUT.**

Byte-identical normalization in both runs:
- Step-1 `read_codes`: `code = double_center(np.log1p(M * 100.0))` (`_curriculum_step1_320_real_corpus.py:286`).
- Validated stream-cortex: `code = double_center(np.log1p(M * 100.0))` (`_phaseB_onbridge_stream_cortex_derisk.py:164`).
- Same `double_center` (per-row + per-column mean subtraction = the log-domain PPMI-equivalent local normalization, CYCLE 88/93b).
- Same `heldout_generalization` / `_pearson_vs_Strue` / `_cos_sim` (imported from `dendritic_d1_learn_graded_structure_derisk.py` by BOTH).

So the Step-1 run is NOT missing the validated normalization. Cause (c) is not in play.

### Cause (d) — scale (64 curated → 320 frequency). **PARTIALLY folded into (a); not an independent learning-limit.**

The validated stream cortex *already scaled to 320* (CYCLE 96, `stream_taxonomy_320`, 150K windows, 9920 neurons) with recall 1.00 + moat — at 320, on the curated semantic reference. So 320 per se is not the problem. What changed between CYCLE-96-320 and Step-1-320 is **the curriculum construction**: CYCLE-96 used the curated balanced 40×8 semantic taxonomy; Step-1 used **top-320-by-frequency over the 32×64 sharding spec** → uneven category membership + incoherent big categories. So "scale" here is really "the frequency-derived vocab populated an unbalanced, co-occurrence-incoherent reference" = a facet of (a)+(b), not a fresh capacity wall.

### Dominant-cause ranking (pinned)

1. **(a) REFERENCE mismatch — DOMINANT.** The `g20` sharding taxonomy is not a co-occurrence-clusterable structure; 27% of concepts sit in big incoherent categories; Pearson +0.07 vs the validated +0.41–0.52.
2. **(b) tail — minor.** Above the floor (169 > 50); trims a few points, cannot explain the collapse.
3. **(d) curriculum imbalance — a facet of (a).** Frequency-selection over a sharding spec made the reference uneven; folded into (a).
4. **(c) normalization — ruled out** (byte-identical).

---

## 2. REFRAME via biology + the validated stream-cortex

**The validated stream cortex already answered this question — the Step-1 run just changed the yardstick.** CYCLE 88–96 established (numpy + on-bridge, multi-seed): online Hebbian co-occurrence + running-frequency log-double-centring **reaches the host PPMI target (+0.513 vs +0.502) and generalizes 0.91** *on a co-occurrence-coherent semantic taxonomy*. Off-diagonal decorrelation was a confirmed red herring (CYCLE 88) — generalization comes from **local feedforward normalization preserving the similarity structure the corpus contains**, not from cross-neuron whitening. So the mechanism is GO; the variable that moved is the reference.

**The biology:** distributional semantics (Harris/Firth: "a word is known by the company it keeps") is exactly what a Hebbian co-occurrence cortex implements — and it makes **co-occurring words similar**. Convergence-zone / hub-and-spoke semantic-hub biology (Patterson–Lambon Ralph anterior-temporal lobe; Pulvermüller distributed cortical word ensembles, catalog G.20) says category structure that the brain represents is **the structure carried by shared sensory/linguistic context** — animals cluster because they appear in animal contexts. A category like "texture adjectives," whose members share no linguistic context with each other, is **not a thing a distributional cortex (or a child's ATL) represents as a cluster**. So the `g20` reference is asking for a non-biological clustering; the curated semantic reference asks for the biological one. **The reframe: the gen reference, not the cortex, is the issue** — and the fix is to measure against the structure the corpus+biology actually produce.

**Is the on-bridge 320 also subject to the budget caveat?** CYCLE 96 noted absolute on-bridge fidelity is window-budget-bounded (its on-bridge gen read 0.45 at 30K windows on the curated reference, with `corr(M,C) 0.885` showing faithful learning; numpy reached 0.91). Step-1 ran **150K windows** (5× more) and got `corr(M,C) 0.756` — faithful, slightly below the 30K-capstone's 0.885 likely because the *frequency* vocab includes the incoherent/tail words that dilute the correlation. So budget is not the limiter at 150K; the reference is.

---

## 3. RANK cheap-first options (closing the gen lift)

Ordered cheapest-first; each is a real candidate, with the expected lift.

| # | Option | What it does | Cost | Expected on `gen` |
|---|---|---|---|---|
| **1** | **Swap the gen REFERENCE to the curated semantic taxonomy** (`stream_taxonomy_320.TAXONOMY_40x8`; restrict the curriculum to its 320 words, or score gen only over the chosen words that fall in a coherent semantic category). | Measures gen against the structure a co-occurrence cortex CAN recover (the validated reference) — apples-to-apples with the 0.91/0.45 result. | **minutes-to-1-bridge** (re-score cached codes if available; else 1 GPU re-run ~75 min). NO new mechanism. | **Predicted ≥ 0.80** (this is literally the reference the validated run cleared). The decisive test. |
| **2** | **Report a CORPUS-DERIVED-taxonomy cluster-purity as a SECOND, reference-independent gen number** (cluster the learned codes — e.g. k-means / agglomerative on `cos` — and measure cluster purity / silhouette against the a-priori labels of whatever coherent subset exists; OR report the held-out gen of the *natural* clusters the codes form). | Answers "do the codes form *clean clusters at all*?" without imposing the `g20` block structure. Anti-cheat: the cluster *count* / labels must not be tuned on the test (see §4). | **minutes** (pure host re-score of learned codes). | A high cluster-purity with low `g20`-Pearson is the **direct evidence that (a) dominates** (the codes cluster well — just not into `g20`'s blocks). |
| **3** | **Restrict the curriculum to a co-occurrence-coherent, BALANCED subset** (drop the 4 big incoherent categories — texture/abstract/spatial/time — and the function-word-heavy ones; keep entity/animal/food/body/color/place/motion categories at ~8 members each). | Removes the 27% of concepts the metric can't reward; aligns the frequency-derived vocab with a clusterable reference. | **1 GPU re-run** (smaller/cleaner 320). | **Lifts gen materially**; combined with #1 should clear 0.80. |
| **4** | **More windows for the tail** (raise `max_windows` 150K→300K, or down-weight/cap the rare tail). | Bounds cause (b). | **1 GPU re-run, ~2.5 hr.** | **Small** on the `g20` reference (the cap is the incoherent big categories, not the tail). Only worth it on a coherent reference. |
| 5 | **High-freq-subset-first diagnostic** (score gen on the top-K most-frequent chosen concepts only, K∈{64,128,160}). | Isolates (b) quantitatively: if gen on the high-freq subset still misses, the tail is exonerated and (a) is confirmed. | **minutes** (re-score). | Diagnostic, not a fix; informs whether #4 is worth running. |

**Recommended cheap-first bundle:** **#1 (swap reference) + #2 (cluster-purity) + #5 (high-freq-subset), all as a re-score of the Step-1 learned codes if they were cached** (the runner reads codes from the bridge; if a code dump exists, this is a pure-host minutes-long re-measure with NO GPU). If codes were not cached, **#1 on a single bridge** (≤75 min GPU) is the one decisive run. #3/#4 are the follow-on builds only if #1 surprisingly under-clears.

---

## 4. ANTI-CHEATS + cheap-first de-risk + GO bars + verdict

### Anti-cheats (all mandatory — the gen reference is exactly where circularity hides)

1. **The gen reference MUST be INDEPENDENT / not corpus-derived (not smuggled).** The headline guard, and the one the original design (`option_c_real_cooccurrence_derisk` design SS1) enforces: `S_true` comes ONLY from the a-priori taxonomy, NEVER from the corpus co-occurrence. Swapping to `stream_taxonomy_320` keeps this (it's a hand-curated independent semantic taxonomy, asserted distinct from the corpus). **For option #2 (corpus-derived cluster-purity) the independence is provided differently:** the *clustering* is corpus-derived but the *labels it's scored against* (purity) must be the independent a-priori labels — and the cluster count/granularity must be fixed a-priori (e.g. = number of coherent categories), NOT searched to maximize the score. Report both the chosen granularity and a sweep to show it wasn't cherry-picked.
2. **No loosening of any other bar.** recall ≥ 0.95 and **moat 0 false-accepts** must still hold under the new reference/curriculum (they are reference-independent — recall/moat read the codes, not `S_true` — so they should be byte-unchanged; assert it).
3. **Frozen-brain control** (`plasticity_on=False`) must still show competence-flat (codes LEARNED, not smuggled) — already PASS in Step-1; re-assert on any re-run.
4. **Derangement control** (shuffle category labels → gen collapses to ~chance) on the NEW reference — confirms the lift is real structure, not a denser reference inflating chance. (Note: a coherent reference with fewer, balanced categories changes `chance` = 1/n_cat; report `gen` AND `ratio_vs_chance`.)
5. **Provenance:** the `g20`-reference gen (0.153) must be REPORTED alongside the new number — the swap is a *correctly-scoped* re-measure, not hiding the miss. The finding is "the codes cluster by co-occurrence-coherent semantics (gen X on the validated reference), and the `g20` sharding taxonomy is the wrong yardstick for a distributional cortex (gen 0.153, Pearson 0.07) — which is itself a clean result about what the substrate represents."

### Cheap-first de-risk

**Re-score the Step-1 learned codes (or 1 fresh bridge) against `stream_taxonomy_320.TAXONOMY_40x8`** (option #1) + **report corpus-derived cluster-purity** (option #2) + **high-freq-subset gen** (option #5), seed 42 first, then 43/44 if it clears. Pure-host if codes cached; else ≤75 min GPU for one bridge.

### GO bars

- **Primary (the fix is real):** generalization ≥ 0.80 on the **independent co-occurrence-coherent semantic reference** (`stream_taxonomy_320`), 3 seeds, with derangement collapse + frozen-flat + recall ≥ 0.95 + moat 0-FA all still holding.
- **Diagnostic confirmation (cause (a) dominates):** corpus-derived cluster-purity is HIGH (codes form clean clusters) while the `g20`-reference Pearson stays ~0.07 — i.e. the codes cluster well, just not into `g20`'s blocks.
- **Tail bound (cause (b)):** high-freq-subset gen on the `g20` reference is still well below 0.80 → the tail is exonerated, (a) confirmed.

### VERDICT

**CLOSEABLE-AND-CHEAPLY. The gen miss is a reference/curriculum-construction mismatch, not a substrate or learning boundary.** The dominant cause (quantified) is (a): the `g20_vocab_spec_2048` *sharding* taxonomy is not a co-occurrence-clusterable structure (27% of concepts in big incoherent categories — texture/abstract/spatial/time adjectives & function words that don't co-occur with each other; Pearson +0.07 vs the validated +0.41–0.52), and the frequency-selection populated it unevenly. The validated stream cortex scored generalization **0.91** on a *purpose-built, balanced, co-occurrence-coherent* semantic taxonomy (`stream_taxonomy_320`/`TAXONOMY_8x8`) using **byte-identical** learning + normalization. The fix is to **measure against that reference (and/or report reference-independent cluster-purity)** — a re-score / single-bridge re-run, not a new mechanism. The tail (169 > the 50 floor) and PPMI (byte-identical) are not the cause; PPMI is ruled out outright.

**Honest scope:** the recall-1.000 + moat-0-FA WIN stands regardless — the codes are good enough to bind, retrieve, and abstain at 320 real-corpus concepts. The "generalization 0.153" is a *correct measurement against the wrong yardstick*. If, after swapping to the coherent reference, gen STILL misses 0.80 at 320 from the real corpus, *that* would be the genuine real-corpus-320 gen boundary (and the precise next question would be window-budget for the tail + a balanced curated-vs-frequency curriculum) — but the evidence (Pearson 0.07 on an orthogonal reference, vs 0.41–0.52 + 0.91 gen on a coherent one, same mechanism) strongly predicts the swap clears it.

**This is a documentation/measurement fix gated behind the standing anti-cheat that the gen reference be independent — exactly the property that was preserved-but-mis-chosen. NOT a fire of the research gate to build a new mechanism; the mechanism is already validated.**

---

## Sources / artifacts (read-only, verified)

- `research/findings/raw/_curriculum_step1_320_real_corpus_seed42.json` — the result (gen 0.153, corr(M,C) 0.756, recall 1.0, moat 0-FA, Pearson(S,S_true) +0.070, chance 0.0333, ratio 4.59×).
- `research/runners/_curriculum_step1_320_real_corpus.py` — `_category_map` (`:104`, uses `g20_vocab_spec_2048`), `derive_curriculum_from_corpus` (`:116`), `read_codes` normalization (`:286`, `double_center(log1p(M*100))`), `measure_generalization` (`:420`).
- `research/runners/g20_vocab_spec_2048.py` — the Step-1 reference; docstring (`:1-30`) declares it a 32×64 SHARDING taxonomy; `CLUSTER_TEXTURE_MATERIAL_ADJ`/`_ABSTRACT_RELATIONS`/`_SPATIAL_WORDS`/`_TIME_WORDS` (`:385-474`) = the co-occurrence-incoherent big categories.
- `research/runners/stream_taxonomy_320.py` — the VALIDATED reference; `TAXONOMY_40x8` (`:83-147`) balanced 40×8, freq≥50, hand-curated for co-occurrence coherence; docstring (`:16-23`) states the coherence requirement + excludes abstract/function words.
- `research/runners/option_c_real_cooccurrence_derisk.py` — `TAXONOMY_8x8` (`:108-117`); the INDEPENDENT-`S_true` correctness property (design SS1, `:160-171`, `:224-226`); second-order-pairs (within-cat words with ZERO direct co-occurrence) construction.
- `research/runners/_phaseB_onbridge_stream_cortex_derisk.py` — `build_stream_bridge` + the byte-identical `code = double_center(np.log1p(M*100.0))` read-out (`:164`); same `heldout_generalization` import (`:43`).
- `research/runners/dendritic_d1_learn_graded_structure_derisk.py` — `heldout_generalization` (`:187`), `_cos_sim` (`:128`), `_pearson_vs_Strue` (`:134`) — used by BOTH runs (identical metric).
- Findings: `2026-06-15-biology-faithful-online-stream-cortex-reaches-target.md` (gen **0.91**, Pearson +0.513, on the curated reference); `2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md` (CYCLE-96 320 capstone, on-bridge gen 0.45 at 30K windows, curated reference; line 198: g20-was-grammatical-then note); `2026-06-15-off-diagonal-red-herring-ppmi-local-normalization-reaches-host.md` (decorrelation red herring; local normalization is the gen lever).
- Scoping that produced Step-1: `research/findings/raw/_foundational_curriculum_scaling_scoping.md` (§4 anti-cheat #3 cited the 0.91; §5 Step-1 said "vocab derived from the corpus by frequency, NOT the hand-curated taxonomy" — the source of the mis-chosen reference).
- Catalog: `E:\Documents\Projects\sim-catalog\references\feature-catalog.md` (G.20 Pulvermüller distributed cortical word ensembles; convergence/decorrelation entries A.11/A.12).
