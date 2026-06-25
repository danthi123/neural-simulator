# Knowledge scaling for an EXCELLENT first chat — corpus + scale + sequence + readiness bar (READ-ONLY scoping)

**Date:** 2026-06-25
**Type:** RESEARCH-GATE scoping (read-only; NO edits/runs/webapp). Standing-practice deep-research BEFORE committing the big 24/7 develop-loop training run.
**Trigger:** owner priority — scale the brain's KNOWLEDGE (the foundational curriculum) FIRST, build the chat console AFTER, so the FIRST chat is an EXCELLENT, natural-feeling first impression (not thin/shallow). The conversational MECHANISM is GO (the discursive engage-and-discuss turn, scoped CYCLE 577; recall 1.000, moat 0-FA). The gating work is purely the KNOWLEDGE: corpus richness + concept scale.
**Builds on (verified):** `_foundational_curriculum_scaling_scoping.md` (the LINEAR-not-V² feasibility + multi-bridge route + resource envelope) and `_curriculum_gen_miss_REAL_scoping.md` (the corpus-richness diagnosis: TinyStories' most-frequent words are distributionally FLAT; the content-vocab fix; the open frequency-vs-curation residual → richer corpus is the lever). This doc is the **first-chat framing**: what corpus + how many concepts + in what order + a measurable "first-chat-ready" bar.

---

## TL;DR — the five deliverables

1. **CORPUS:** **Start TinyStories-only (the content-vocab gen-confirm in flight decides the floor), then LAYER IN the already-downloaded Wikipedia corpus as the breadth/fact source.** TinyStories gives dense, clean per-concept co-occurrence (the validated 0.91-generalization path) but a NARROW child's-world topic set; `data/corpus/wikitext.txt` (already present: **21,552 unique types vs TinyStories' 7,349**) gives 3× topic breadth + real-world facts. An excellent first chat needs BOTH: TinyStories for code quality, Wikipedia for "relate to ~any everyday thing." **Quantified recommendation below (§1).**
2. **SCALE:** **~1,000–1,500 concepts is the first-impression target** (relate to most everyday things + a few facts each). NOT hundreds (320 alone is thin/single-domain), NOT tens-of-thousands (the far ceiling). This is **3–5 bridges, ~15–25 GB resident, local-feasible on the 3090.** The 3090 ceiling is **~2K concepts (~7 bridges) resident at once**; beyond that the existing synapse tiering pages bridges (they never interact during learning) — so even the far tier is local. **TinyStories alone caps clean content concepts at ~320–680** (documented corpus ceiling) → reaching ~1–1.5K NEEDS the combined corpus.
3. **SEQUENCE:** gen-confirm (in flight) → Step-2 multi-bridge gate (4-bridge, ~1,280 concepts, the decisive de-risk) → corpus-choice fork (TinyStories-only vs +Wikipedia, decided by gen-confirm + Step-2) → the long 24/7 develop-loop training run at the chosen scale.
4. **READINESS BAR (the concrete "build-the-console-now" trigger):** **vocab ≥ 1,000 grounded concepts spanning ≥ 8 everyday domains; ≥ ~3 facts/concept (≥ 3,000 facts); generalization ≥ 0.80 (coherent reference, derangement-collapse); moat 0-FA (frozen-flat at scale); AND a scripted DISCURSIVE-turn quality check passes** (10 everyday prompts each get a ≥ 2-proposition, ≥ 2-type, verified-or-flagged paragraph — not a thin 1-fact answer, not an abstain). Measurable, not a guess.
5. **RESOURCE/WALL-CLOCK + CEILING:** calibrated from the ACTUAL Step-1 run — **~4.9 GB + ~76 min per 320-concept bridge at 150K windows** on the 3090. ~1–1.5K concepts ≈ **3–5 bridges ≈ ~4–6 GPU-hr** (one overnight). The 24/7 develop-loop (accumulate + persist + no-forget) is the intended mode. **3090 ceiling: ~2K concepts resident (VRAM), unbounded with tiering (wall-clock-bound). Cloud is NOT needed — no >24 GB VRAM wall.**

---

## 1. THE CORPUS QUESTION — is TinyStories rich enough, or is Wikipedia / BabyLM needed?

### 1a. The measured corpus facts (this session, both cached shards)

| Corpus (the cached shard in `data/corpus/`) | size | tokens | unique types | types f≥50 | types f≥100 | **g20-content words f≥50** | g20-content f≥20 |
|---|---|---|---|---|---|---|---|
| **`tinystories.txt`** | 8.0 MB | 1.59 M | **7,349** | 1,816 | 1,106 | **681** | 895 |
| **`wikitext.txt`** | 4.0 MB | 627 K | **21,552** | 1,595 | 726 | **349** | 556 |

(Both already present — `wikitext.txt` was downloaded 2026-06-22. NB: `wikitext.txt` currently has **0 `<|endoftext|>` delimiters**, so the streaming reader treats it as one document — a trivial loader detail, not a content issue.)

### 1b. The honest read: TinyStories is rich ENOUGH for clean codes, NOT for topic breadth

Three load-bearing facts decide this:

- **(i) TinyStories supports clean content codes to ~320–680 concepts, but that's its CEILING.** The validated taxonomy doc (`stream_taxonomy_320.py:45-54`) is explicit: *"40 CLEAN 8-word semantic categories ARE achievable at the ≥50 floor… but the corpus is close to its ceiling there… Pushing meaningfully past 40 clean categories would require dipping below freq 50 or admitting abstract/function words."* The measured 681 content-words-f≥50 is the soft outer edge (admitting f≥20 thinner words gives ~895). **So TinyStories ALONE caps the clean content vocab at ~320–680** — enough for a single-domain demo, NOT for "relate to ~any everyday thing."

- **(ii) TinyStories topics are NARROW and weird-flat** (the owner's "fish sings pink" concern is real). It is children's stories: the world is animals-as-pets, toys, play, food-at-meals, simple feelings. There is no geography, no science, no professions, no history, no how-things-work. A first chat about "what is a computer / a mountain / a doctor / electricity" finds NOTHING in a TinyStories-only brain. AND its narrative flatness means even its entities co-occur in generic play-scenes (animals all appear together as pets), which is exactly why the FINE category-similarity needed for generalization was the open question (`_curriculum_gen_miss_REAL_scoping.md` option #4: *"TinyStories is narrative and category-flat… a fact-denser corpus (Simple-Wiki: 'a dog is an animal', taxonomic statements) gives sharper entity-category co-occurrence"*).

- **(iii) Wikipedia gives 3× the topic breadth + real fact structure, but is Zipf-thin per-word at small scale.** `wikitext.txt` has **21,552 unique types** (3× TinyStories) — real-world entities, places, concepts, taxonomic statements. But at only 627K tokens, only 349 content-words clear f≥50 (the breadth is spread thin). A LARGER Simple-English-Wikipedia dump (10s of M words, the prior scoping's option 3) would lift the f≥50 content count into the thousands while keeping the breadth — that is the real lever for "any everyday topic."

### 1c. The decisive in-flight result + the QUANTIFIED recommendation

**The pivot is the content-vocab gen-confirm currently running** (`_curriculum_step1_320_real_corpus --vocab-filter content`, CYCLE 576). It tests whether the corpus-frequency CONTENT-word vocab (entities+verbs, the flat adjective/function words demoted to hubs) clears generalization ≥ 0.80 on TinyStories. Two outcomes, two corpus paths:

- **If content-vocab CLEARS (gen ≥ 0.80, ~= the curated 0.91 control):** TinyStories-derived content codes generalize. → **TinyStories is the code-quality FLOOR** (clean codes to ~320–680 concepts), and Wikipedia is layered in PURELY for topic BREADTH past ~680 (to reach ~1–1.5K). The combined corpus is the production curriculum.
- **If content-vocab UNDER-clears vs the curated control** (the documented "frequency-vs-curation residual"): TinyStories' narrative flatness limits even content-word fine-separation. → **Wikipedia is needed EARLIER** (for sharper entity-category co-occurrence: "a dog is an animal" taxonomic statements), not just for breadth. The combined corpus becomes load-bearing for code quality too.

**QUANTIFIED RECOMMENDATION (corpus ladder for the first chat):**

| Rung | Corpus | Concept reach | Why | Status |
|---|---|---|---|---|
| **A (floor)** | **TinyStories alone** (cached, 1.59M tok) | ~320–680 clean content concepts | dense per-concept co-occurrence; the validated 0.91-gen path; clean codes | **HERE NOW** (gen-confirm in flight) |
| **B (first-chat target)** | **TinyStories + Wikipedia** (both cached; expand Wikipedia to a ~10–50M-tok Simple-Wiki dump) | **~1,000–1,500 concepts** | TinyStories code quality + Wikipedia topic breadth + fact density → "relate to ~any everyday thing" | **the recommended first-chat corpus** |
| C (foundational) | + BabyLM-10M (CHILDES + books + Wiki + dialogue) | ~5K–8K core | the principled developmental scale; multi-source | far tier, gated on B GO |

**Bottom line on the corpus question:** **TinyStories is NOT rich enough alone for an excellent first chat** (narrow child's-world topics, ~680-content-word ceiling). **The recommendation is the COMBINED corpus** — TinyStories (already cached) + Wikipedia (already cached; expand to a larger Simple-Wiki dump). TinyStories supplies clean codes; Wikipedia supplies the everyday-topic breadth + fact density that makes a first chat feel like it knows the world. BabyLM-10M is the *foundational* tier above the first-chat bar, not needed for the first impression. **The gen-confirm result decides whether Wikipedia is needed for code QUALITY (under-clear) or only for BREADTH (clear) — but either way the combined corpus is the first-chat production curriculum.**

---

## 2. THE SCALE TARGET — how many concepts for an excellent first impression?

### 2a. The target: ~1,000–1,500 grounded concepts across ≥ 8 everyday domains

The owner's bar is "relate to ~any everyday thing + discuss it richly." Calibrate from three anchors:

- **320 is thin** — it's ~40 categories, single-domain-ish (the validated tier), enough for a demo but a first chat about an off-domain topic finds nothing. The discursive-turn scoping is explicit: *"the meaning-of-life turn from a 24-fact brain will assemble a few adjacent grounded fragments… genuinely discursive, but shallow. Depth and substance EMERGE as the curriculum grows."* 320 is the shallow end.
- **~1,000–1,500 is the first-impression sweet spot** — enough breadth that ~any everyday noun/verb/place/concept the user mentions has a grounded code + a few facts + graph-adjacency for the (D) discuss path. This is the "core 2K content words" band the prior scoping names, taken at its lower (first-chat) edge. It covers everyday entities (animals, food, body, household, vehicles, nature, buildings, tools, clothing), common verbs, and — via Wikipedia — places, professions, simple science/world facts.
- **5K–30K is the far ceiling** (full BabyLM / adult vocab) — NOT the first impression; the foundational/far tier, scaled later over the develop loop.

**Why ~1–1.5K and not more for the FIRST chat:** richness in the discursive turn comes from (a) having a grounded code for the topic, (b) ≥ a few facts about it, and (c) graph-adjacency for elaboration. ~1–1.5K concepts × ~3 facts each ≈ ~3–5K facts gives every common topic a non-empty (C)-certain gather + (D)-adjacency set. Past that, marginal first-impression value drops (the user rarely probes the rare tail in a first chat) while wall-clock grows. **~1–1.5K is the minimum that doesn't feel thin, at the lowest cost.**

### 2b. Tie to the resource envelope (the validated multi-bridge route)

From the ACTUAL Step-1 run (`_curriculum_step1_320_real_corpus_seed42.json`) + the sparse-distributed capacity curve:

| First-chat scale | bridges (@≤320/bridge) | pool neurons (≈) | **VRAM resident** | local on 3090? |
|---|---|---|---|---|
| 320 (validated) | 1 | 9,920 | **~4.9 GB** (measured) | yes, trivially |
| **~1,000** | **~3–4** | ~30–40K | **~15–20 GB** | **yes — resident, no tiering** |
| **~1,500** | **~5** | ~46K | **~22–25 GB** | **at the 24 GB line** — resident OK at ~1,280 (4 bridges, ~13–20 GB); ~1,500 may need tiering |
| 2,048 | ~7 | ~80K | ~22 GB | the resident ceiling |
| 5,000+ | ~16+ | ~184K+ | >24 GB | NEEDS tiering (page bridges; they don't interact) |

**The 3090 ceiling is ~2K concepts (~7 bridges) resident at once.** The first-chat target (~1–1.5K) sits comfortably below it — **4 bridges (~1,280 concepts) fit resident with headroom**; ~1,500 (5 bridges) is at the line and the existing `TieredSynapseStore` covers any overflow. **Cloud is NOT needed** — there is no >24 GB VRAM wall for the first-chat tier (per `feedback_long_local_runs_ok_confirm_cloud_cause`: cloud only for a genuine VRAM wall, not wall-clock).

**The corpus CAPS the scale before VRAM does.** TinyStories alone tops out at ~320–680 clean content concepts; reaching ~1–1.5K REQUIRES the combined corpus (§1). So the scale target and the corpus recommendation are coupled: **~1–1.5K concepts is feasible ONLY with TinyStories + Wikipedia.**

---

## 3. THE DEPENDENCY-ORDERED SEQUENCE (de-risk → train)

Each rung gates the next; cheapest-decisive first. (Rung 0 is in flight; do not re-run it.)

**Rung 0 — gen-confirm (IN FLIGHT, controller-managed; do NOT touch).** `_curriculum_step1_320_real_corpus --vocab-filter content` (+ the `curated` control arm). Decides the corpus floor: does the corpus-frequency CONTENT-word vocab clear gen ≥ 0.80 on TinyStories? **GATE:** gen ≥ 0.80 coherent-reference, derangement-collapse, recall ≥ 0.95, moat 0-FA, frozen-flat. **Outcome routes Rung 2's corpus choice (§1c).** ~76 min/seed already budgeted.

**Rung 1 — corpus-loader + curriculum plumbing for the combined corpus (CPU/cheap, ~hours engineering — NOT this scoping).** Three small engineering pieces (the prior scoping's §3, all reuse-shaped): (a) add `<|endoftext|>` handling / a doc-splitter for `wikitext.txt` (it has none); (b) wire `derive_curriculum_from_corpus` + the hub set to read MULTIPLE corpus files (TinyStories + Wikipedia merged frequency); (c) the content-vocab filter (already prepped, CYCLE 575). No new mechanism. *Prerequisite for Rung 2's combined-corpus arm.*

**Rung 2 — the DECISIVE multi-bridge gate: ~1,280 concepts, 4-bridge ensemble (1 GPU, ~5–6 GPU-hr).** Stream the **combined** corpus to ~1,280 corpus-frequency content concepts across 4 sparse bridges (≤320/bridge), via `g20_multibridge --sparse` + the `g20_vocab_spec_2048` sharding. **This is the scaling claim's decisive test** (multi-bridge linear-in-count beyond 320, from a real combined corpus, moat + generalization intact). **GATE (3 seeds, VRAM + wall-clock REPORTED):** per-bridge discrimination ≥ 95%; cross-bridge who/what recall ≥ 0.90; **moat 0-FA across the 1,280-concept space**; generalization ≥ 0.80 + derangement-collapse; frozen-brain competence-flat; VRAM ≤ 24 GB resident. **This rung directly tests the first-chat scale (~1,280 ≈ the ~1–1.5K target).**

**Rung 3 — corpus-choice fork (decided by Rung 0 + Rung 2, no extra run if both clean).** If TinyStories-content cleared at Rung 0 AND the combined corpus holds at Rung 2 → lock the combined corpus at ~1–1.5K. If either under-clears → the Wikipedia weight / a larger Simple-Wiki dump is the lever (a corpus-mix sweep, cheap). *This is a decision node, not necessarily a new run.*

**Rung 4 — the long 24/7 develop-loop training run at the first-chat scale (overnight–days, LOCAL, the intended mode).** `_longitudinal_develop_loop_gpu` over the combined corpus at ~1–1.5K concepts: accumulate + converse + consolidate (no-forget) + persist (per-day bundles → the watchable console capstone). **GATE = the READINESS BAR (§4).** ETA: building ~1–1.5K codes ≈ 4–6 GPU-hr of stream-learning; the develop-loop wraps it in simulated days for the watch-and-talk capstone (a compressed "week"/"month" of accumulation, the artificial-life axis). **Then build the chat console** (the discursive turn is already scoped + Stage-0 de-risk dispatched).

**Why this order:** Rung 0 (free, in flight) sets the corpus floor; Rung 2 is the one ~5–6-hr run that proves the actual first-chat scale + the moat at 4× vocab from a real corpus — flip the decision before committing days of develop-loop GPU. Rungs 1/3 are cheap engineering/decision nodes between them.

---

## 4. THE READINESS BAR — concrete, measurable "first-chat-ready"

So we KNOW when to build the console rather than guessing. **ALL must hold** (3 seeds where applicable):

### 4a. Knowledge scale + breadth
- **Vocab ≥ 1,000 grounded concepts** (the brain HAS a learned code for each — `grounded` dict size, from heard concepts, not just a vocab counter).
- **Breadth ≥ 8 everyday domains** populated (animals, food, body/people, household/objects, nature/places, vehicles, tools/clothing, + Wikipedia-sourced world topics) — so ~any everyday thing has a code. (Measured: ≥ 8 of the coherent categories have ≥ 4 grounded members.)
- **Fact density ≥ ~3 facts/concept (≥ 3,000 stored facts)** — so a topic question has a non-empty (C)-certain gather AND (D)-adjacency for elaboration (not a one-liner).

### 4b. Knowledge QUALITY (the anti-cheats — never relaxed)
- **Generalization ≥ 0.80** on the INDEPENDENT coherent reference, with **derangement-collapse** (shuffle labels → collapses to ~chance). Pearson(cos,S_true) ≥ +0.40 (the validated band).
- **Moat 0 false-accepts at scale** — every never-stored cue ABSTAINS; **frozen-brain competence-flat** (plasticity OFF → corr(M,C)~0, recall < 0.5 → the brain LEARNED the knowledge, not smuggled). A single fabricated certainty is a HARD STOP.
- **Recall ≥ 0.95** (who/what) on stored facts.
- **Gen reference INDEPENDENT/a-priori** (never corpus-derived) + the vocab filter NOT tuned on gen (the standing curriculum anti-cheats).

### 4c. The CONVERSATION-QUALITY check (the actual first-impression test — the new bar)
Run the scoped **DiscursiveTurn** (CYCLE 577, Stage-0 GO) on **10 scripted everyday first-chat prompts** spanning the domains (e.g. "what is a dog", "tell me about cars", "what's the weather like", "what is a mountain", "what do you think about cats", an open "what is happiness", a phatic "hi", a "tell me more" follow-up, a who/what on a flagged proposition, an unknown word). **PASS criteria:**
- **≥ 8/10 produce a ≥ 2-proposition, ≥ 2-TYPE paragraph** (certain + flagged/discuss/phatic) — engages + discusses, NOT a thin 1-fact answer, NOT a bare abstain.
- **The open question** ("what is happiness") gets a (D) discuss paragraph (adjacent grounded + flagged), not an abstain.
- **The follow-up** ("tell me more") increases depth on the held topic.
- **MOAT (hard):** every CERTAIN proposition re-parses to a STORED fact; every FLAGGED proposition's who/what ABSTAINS + is never stored; 0 fabricated-fact assertions across all 10.
- **Reads as natural** (subjective owner check — the deliverable is a transcript the owner judges "genuinely good first impression").

**The trigger:** when 4a + 4b + 4c ALL hold at ~1–1.5K concepts → **build the console.** Until then, keep scaling knowledge. This is the measurable line the owner asked for (vocab + facts + generalization + a sample-conversation quality check via the discursive turn).

---

## 5. RESOURCE / WALL-CLOCK ESTIMATE per rung + the 3090 ceiling

Calibrated from the ACTUAL Step-1 run (`_curriculum_step1_320_real_corpus_seed42.json`): **320 concepts, 150K windows → 9,920 neurons, VRAM 4,911 MB (~4.9 GB), learn 4,534 s (≈ 75.6 min), build 11.5 s.** (Matches the prior scoping's ~84 min / ~4.9 GB calibration.)

| Rung | What | VRAM | Wall-clock (3090) | Cloud? |
|---|---|---|---|---|
| 0 gen-confirm | 320, 1 bridge, content-vocab + curated control | ~4.9 GB | ~76 min/seed (in flight) | no |
| 1 plumbing | combined-corpus loader + curriculum | ~0 (CPU) | ~hours engineering | no |
| **2 multi-bridge gate** | **~1,280, 4 bridges, combined corpus** | **~13–20 GB resident** | **~5–6 GPU-hr** (3 seeds → fold; ~75 min/bridge × 4 + scoring; one overnight) | **no** |
| 3 corpus fork | decision (+ optional mix sweep) | — | cheap | no |
| **4 develop-loop run** | **~1–1.5K first-chat scale, accumulate+persist+no-forget** | **~15–25 GB** (≤2K resident; tiering past) | **~4–6 GPU-hr stream-learning**, wrapped in simulated days for the watch-and-talk capstone (a compressed week/month) | **no** |
| C far tier (later) | BabyLM-10M ~5–8K, 16–25 bridges | >24 GB → **synapse tiering** (page bridges; they don't interact during learning) | days, staged | **only for faster turnaround, never for VRAM** |

**THE 3090 CEILING (honest):**
- **VRAM:** ~2K concepts (~7 bridges, ~22 GB) resident at once. The first-chat tier (~1–1.5K) is BELOW this → fully resident, no tiering. Past ~2K, the existing `TieredSynapseStore` pages bridges in/out — bridges NEVER interact during learning (each learns its own disjoint concepts), so VRAM is **not a true wall**; the substrate scales to 5K–30K LOCAL with tiering.
- **Wall-clock:** the real constraint. Linear in (bridges × windows). First-chat ~1–1.5K ≈ one overnight (~4–6 GPU-hr). The full BabyLM far tier ≈ a few overnight runs — exactly the develop-loop's intended 24/7 operating mode (the owner accepts long local runs with an ETA; the per-day-bundle capstone makes it watchable).
- **The corpus, not the GPU, caps the FIRST-chat scale** — TinyStories alone tops at ~680 content concepts; reaching ~1–1.5K needs the combined corpus (§1).

**⇒ Cloud is NOT needed for the first chat.** No >24 GB VRAM wall at ~1–1.5K. Cloud (H100, ~3–5× faster) would only *cut wall-clock* on the far BabyLM-100M tier — never enable the first-chat tier, which is local-overnight.

---

## 6. THE GENUINE OPEN RISK (the honest-negative trigger)

Two coupled risks, both surfaced by Rung 0 + Rung 2:

1. **Code QUALITY from a real combined corpus.** The gen-confirm (Rung 0) tests whether corpus-frequency CONTENT words generalize on TinyStories. If they under-clear vs the curated control, that's the documented **frequency-vs-curation residual** — and the precise lever is a richer/fact-denser corpus (Wikipedia weight ↑ / a larger Simple-Wiki dump). If even the combined corpus under-clears at ~1,280 (Rung 2), THAT maps the real corpus-quality boundary for the substrate (honest deliverable; reframes toward the learned-cortex frontier for richness).
2. **Multi-bridge fidelity at the first-chat scale.** Whether the multi-bridge route preserves per-bridge ≤320 discrimination + the moat + generalization at ~1,280 from a real (noisier) corpus, and whether cross-bridge routing (which bridge holds a queried concept, cross-bridge associative retrieval) degrades. Rung 2 is designed to surface exactly this. NO-GO at Rung 2 = a characterized boundary, not a failure.

Neither is expected to block (the 320 tier is GO, the mechanism is validated, the combined corpus is downloaded) — but both are the things to WATCH, and either NO-GO is the scientific deliverable per the BRAIN-BASED-ONLY standard.

---

## 7. VERDICT

**The path to an excellent first chat is KNOWLEDGE-bound, local-feasible, and well-scoped — gated on one ~5–6-hr decisive run, not days of guessing.**

- **CORPUS:** TinyStories (cached) for clean codes + **Wikipedia (cached) for breadth/facts** = the combined first-chat corpus. TinyStories alone is too narrow (child's-world topics, ~680-content-word ceiling). The gen-confirm in flight decides whether Wikipedia is needed for code QUALITY or only BREADTH; either way the combined corpus is the production curriculum. (BabyLM-10M is the foundational tier above the first-chat bar.)
- **SCALE:** **~1,000–1,500 grounded concepts** across ≥ 8 everyday domains — the first-impression sweet spot (relate to ~any everyday thing + a few facts each). 320 is thin; 5K–30K is the far ceiling. ~1–1.5K = 3–5 bridges, ~15–25 GB, local on the 3090. **The corpus, not VRAM, caps the first-chat scale** (needs the combined corpus to exceed ~680).
- **SEQUENCE:** gen-confirm (in flight) → Rung-1 combined-corpus plumbing → **Rung-2 the decisive 4-bridge ~1,280-concept gate** → corpus fork → the 24/7 develop-loop run at ~1–1.5K → build the console.
- **READINESS BAR:** vocab ≥ 1,000 / ≥ 8 domains / ≥ 3 facts-per-concept (≥ 3,000 facts) + generalization ≥ 0.80 (coherent, derangement-collapse) + moat 0-FA (frozen-flat) + **the 10-prompt discursive-turn quality check (≥ 8/10 mixed-type, verified-or-flagged, natural)**. Measurable — build the console when ALL hold.
- **RESOURCE:** ~4.9 GB + ~76 min per 320-bridge (measured); ~1–1.5K ≈ ~4–6 GPU-hr (one overnight). **3090 ceiling ~2K concepts resident; unbounded with tiering (wall-clock-bound). Cloud NOT needed — no >24 GB VRAM wall for the first chat.**

**Honest where-the-ceiling-is:** the FIRST-chat scale (~1–1.5K) is comfortably inside the 3090 and the combined corpus. The 3090's hard limits are ~2K concepts RESIDENT (VRAM) — relieved to 5K–30K by the existing synapse tiering (bridges don't interact during learning) — and wall-clock (a few overnight runs for the full BabyLM far tier). The genuine residual risk is corpus code-QUALITY at scale (the frequency-vs-curation residual; richer corpus is the lever) and multi-bridge cross-routing fidelity — both surfaced by the in-flight gen-confirm + the Rung-2 gate, both honest-negatives if they don't clear.

---

## Sources / artifacts (read-only, verified this session)

**In-repo (load-bearing):**
- `research/runners/_curriculum_step1_320_real_corpus.py` — the Step-1 runner; `derive_curriculum_from_corpus` + `--vocab-filter {content,all,curated}` (the gen-fix, `:235`/`:794`); `_coherent_category_map` + `COHERENT_G20_DOMAINS`/`CONTENT_G20_DOMAINS` (`:163`/`:142`/`:160`); `read_codes` `double_center(log1p(M·100))` (`:455`); measured VRAM/wall-clock fields.
- `research/findings/raw/_curriculum_step1_320_real_corpus_seed42.json` — the ACTUAL resource numbers: **320 concepts, 150K windows, 9,920 neurons, VRAM 4,911 MB, learn 4,534 s (~76 min), recall 1.0, corr(M,C) 0.756, gen 0.153 (`all` vocab; content-fix in flight)**, freq_range [19729, 169].
- `research/runners/corpus_stream.py` — the bounded-memory streaming loader (`iter_stories`/`load_token_stream`, `--corpus-path`); default `data/corpus/tinystories.txt`.
- `data/corpus/tinystories.txt` (8.0 MB, 1.59M tok, 7,349 types, 681 g20-content f≥50) + `data/corpus/wikitext.txt` (4.0 MB, 627K tok, **21,552 types**, 349 g20-content f≥50; **0 `<|endoftext|>` delimiters** — loader detail).
- `research/runners/stream_taxonomy_320.py` — `TAXONOMY_40x8` (40×8 = 320 curated coherent content words, freq≥50); docstring `:45-54` = the **TinyStories ~40-clean-category corpus ceiling** (the load-bearing scale-cap fact); `:16-23` = coherence requirement + abstract/function words excluded.
- `research/runners/_longitudinal_develop_loop_gpu.py` — `StreamCortex` (`M[Nt,n_hub]` linear-in-vocab `:191`/`:254-256`; `hear_day` `:218`; `read_codes` `:249`); `_GPU_SYLLABUS` (`:84`, the hand-authored syllabus to generalize to corpus-derived). `_longitudinal_develop_loop.py` — `GradedCurriculum` (`:144`, `vocab_through_day`/`full_vocab` — the cumulative vocab axis).
- `research/runners/concept_pool_sparse_distributed.py` — `build_sparse_pool_bridge`/`generate_sparse_patterns` (`:53`/`:137`); `research/runners/g20_multibridge.py` — `--sparse` multi-bridge loader; `g20_vocab_spec_2048.py` — 32×64 = 2048 sharding spec (data-complete).
- `sim/synapse_storage.py` (`TieredSynapseStore`, idle/pressure eviction — the >2K-concept VRAM relief); `sim/lineage.py` (atomic persistence + per-day bundles); `sim/auto_growth.py` (`TierPromoter`).

**Prior scopings (this doc builds on; verified):**
- `research/findings/raw/_foundational_curriculum_scaling_scoping.md` — the LINEAR-not-V² feasibility, multi-bridge route (64@100%/320@98.4%/bridge), corpus ladder (TinyStories/BabyLM/Simple-Wiki), resource envelope (~84 min/320-bridge, ~2K resident ceiling, tiering past, cloud-not-for-VRAM).
- `research/findings/raw/_curriculum_gen_miss_REAL_scoping.md` — the corpus-richness diagnosis: TinyStories' most-frequent words are distributionally FLAT (48% adj/function/emotion); the content-vocab fix; the OPEN frequency-vs-curation residual (option #4 = richer corpus = the lever); **the recall-1.000 + moat-0-FA WIN stands regardless** (reference-independent).
- `research/findings/raw/_curriculum_gen_miss_scoping.md` — the (refuted) wrong-yardstick hypothesis (the reference-swap moved 0.153→0.167 only; it's the VOCAB/codes, not the reference).
- `research/findings/raw/_communicable_discursive_turn_scoping.md` — the discursive engage-and-discuss turn (the conversational MECHANISM, GO/scoped); **"RICHNESS is knowledge-gated… depth and substance EMERGE as the curriculum grows"** (the load-bearing link between this scoping's knowledge axis and the conversation quality).

**Memories:** `project_communicable_brain_not_rag` (the engage-and-discuss north-star; moat = never-assert-a-fabricated-fact; the BRAIN does cognition, no LLM-free-generate), `project_foundational_curriculum` (referenced; corpus = the linguistic ENVIRONMENT, brain learns via its own stream cortex), `feedback_long_local_runs_ok_confirm_cloud_cause` (cloud only for >24 GB VRAM, not wall-clock), `feedback_moat_not_hard_lossy_memory_ok` (moat a plus not a hard gate; lossy OK if it buys scaling).
