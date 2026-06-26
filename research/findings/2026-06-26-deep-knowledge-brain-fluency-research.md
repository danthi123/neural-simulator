# Deep-knowledge + brain-native fluency — deep-research + design-input pass

**Date:** 2026-06-26
**Type:** read-only deep-research + design doc (NO `sim/`/runner edits, no commits). The controller trust-but-verifies the load-bearing claims and designs the build from this.
**Trigger:** the project's standing practice (deep research + catalog review BEFORE a new-direction / mechanism-class build). This is exactly that: the first-chat console works mechanically (10/10 rubric, moat 0-leak) but has three verified gaps that are each a *new mechanism class*: shallow knowledge, templated fluency, ~15 s/turn latency (`AUTONOMOUS_STATE.md` CYCLE 612; `2026-06-26-first-chat-ready-bar.md`; `2026-06-26-breadth1454-window-sweep.md`).

---

## 0. Terms (defined once)

- **Stream cortex** — the project's on-bridge Hebbian co-occurrence learner: it "hears" a corpus word-by-word in sliding windows and accumulates a co-occurrence weight matrix `M` (target-concepts × hub-neurons). `read_codes()` log-double-centres `M` into a per-concept **grounded phasor code** (a unit-magnitude vector whose information is in the *phase*). (`_curriculum_step1_320_real_corpus.py:365` `StreamCortexBridge`, `:460` `read_codes`.)
- **Grounded code** — the per-concept phasor `{word: phases[D]}` produced by the stream cortex; the input the composer binds. `D` is the phase dimension (64 or 128).
- **PPMI** — Positive Pointwise Mutual Information, `max(0, log p(a,b)/(p(a)p(b)))`; the standard distributional-semantics weighting for word association / **selectional preference** (how strongly a verb "selects" a noun argument). The project's association graph `P` is this. PMI between a verb and a noun *is* the selectional-preference strength ([Wikipedia/PMI](https://en.wikipedia.org/wiki/Pointwise_mutual_information); [PPMI overview](https://www.emergentmind.com/topics/positive-pmi-ppmi)).
- **VSA / FHRR** — Vector Symbolic Architecture / Fourier Holographic Reduced Representation. Role–filler **binding** (agent⊗AGENT + action⊗ACTION + patient⊗PATIENT, superposed) into a single fixed-width composite, **unbound** by the (near-)exact phasor inverse, **cleaned up** to the nearest codebook entry. The project's `RFPhasorComposer`. The known scaling law: retrieval degrades roughly linearly with the number of items bound into one composite unless dimension/redundancy grows ([Schlegel et al., VSA comparison, AI Review 2022](https://link.springer.com/article/10.1007/s10462-021-10110-3); [HDC/VSA survey, ACM CSUR 2023](https://dl.acm.org/doi/10.1145/3558000)).
- **No-confab moat** — the project's hard invariant: a query whose cue is not a stored fact returns `None` (abstain), never a fabrication. In code it is the `query_patient/query_agent/ask_yes_no → None` path plus the DiscursiveTurn's type-aware VERIFY gate (`_discursive_turn_stage0_derisk.py:402`).
- **SVO fact** — a `(agent, action, patient)` triple, e.g. `(dog, chase, cat)`. The project's atomic relational unit.

---

## 1. Diagnosis — what is actually wrong, mechanism by mechanism

The console pipeline is sound; the *content feeding it* is impoverished. The three gaps are causally **chained**: bad knowledge → bad fluency; latency is largely a separate engineering axis.

### GAP 2 (root cause): the stored "facts" are random recombinations, and the relational graph is noise on the rare tail

Two independent defects, both verified in code:

1. **The fact-base is fabricated, not corpus-grounded.** `_make_svo_facts` (`_curriculum_step1_320_real_corpus.py:494`, used by `first_chat_console.py:154`) builds each "fact" by **uniform random sampling**: `a = nouns[rng.randint(...)]; v = verbs[rng.randint(...)]; p = nouns[rng.randint(...)]` (lines 521–523). Nothing reads what the corpus *says*. So the brain's "knowledge" is 24 arbitrary noun-verb-noun triples that happen to use learned words. It can recall them (and abstain correctly on the rest — the moat holds), but they are not *true*, so the chat cannot be about anything real.

2. **The co-occurrence relations are reliable only for frequent concepts.** The stream cortex learns a genuine co-occurrence matrix, but at 1,454 concepts in a single bridge at `n_per=10`, the rare tail's codes blur. CYCLE 612 verified: `dog→cat`, `fire→cook/oven`, `run→catch` are meaningful; `curry→bull`, `king→knife`, `school→lamb` are noise, many saturated at cos 1.00. The window-sweep (`2026-06-26-breadth1454-window-sweep.md`) diagnosed the mechanism precisely: **over-training densification** — past ~8 K windows `M` goes fully dense (frac-nonzero → 1.0), the distinguishing structure washes into a uniform background, recall collapses 0.958@7K → 0.208@150K. A single bridge has a hard **recall-vs-coverage trade-off**: enough windows to cover the rare tail densifies and destroys the frequent core.

The corpus itself is **fully adequate** for real facts: `data/corpus/tinystories.txt` (8 MB) and `data/corpus/simplewiki.txt` (143 MB) are natural-language prose ("the loud dog... I'll protect you"; "Tom lost his red ball"), `<|endoftext|>`-delimited, with real subject-verb-object structure on every line. The pipeline simply never extracts it — `corpus_stream.iter_stories` tokenizes with `re.findall(r"[a-z]+", story)` (`corpus_stream.py:49`) into a **bag of words for co-occurrence only**; sentence/clause boundaries and argument structure are discarded.

### GAP 3 (downstream of GAP 2): fluency is templated, and the verb is chosen by the noisy graph

Two layers, both verified:

1. **The renderer only orders a pre-decided triple — it is a fixed SVO frame.** `NeuralSerialOrderRenderer.render(frame_concepts, spell)` (`neural_serial_order_renderer.py:60`) takes an *already-chosen* `(agent, action, patient)` and produces the spiking rate-ranked order, then `" ".join`s the spelled words. The neural part is the **parallel→serial conversion** (Grossberg competitive queuing; legitimately a brain operation, catalog G.07/H.19). But it cannot choose *which* words go in the frame, cannot vary the construction (no "the X that Y'd", no adjectives, no connectives), and the frame template is fixed SVO. This is the correct, validated *serialization* primitive — but it is not *grammatical encoding*.

2. **"curry describes pine" comes from the proposer trusting a noisy graph.** The novel/discuss channel proposes a triple by sampling action then patient weighted by PPMI relatedness (`_genfrontier_b2_generative_replay_derisk.py:_weight_partner:203`, `_sample_weighted:236`), gated by `_plausible = _related(a,ac) and _related(ac,p)` (`:179`). That gate **is** the standard selectional-preference test — the *mechanism is right*. But when `P` is noise for rare concepts (curry~describes~pine all spuriously "related"), the gate passes garbage, and "describes" is a high-frequency verb that co-occurs with everything. So fluency is stilted **because the knowledge underneath it is wrong**, not because the generator is broken. Fix GAP 2 and most of GAP 3 follows; the residual GAP-3 work is *constructions beyond the single SVO frame*.

### GAP 1 (largely separate): latency ~15 s/substantive turn

Verified bottleneck (`first_chat_console.py`, CYCLE 612): `propose_candidates_about(topic, n_attempts=500)` (`_communicable_turn_stageA_derisk.py:311`) samples up to 500 triples and, for each *novel + plausible* one, runs `_contradicts → composer.ask_yes_no` (`:346`) — a full composer **resonate** per candidate. Several uncached topics per turn × a resonate-per-candidate ≈ the ~2 s/topic cost, plus per-flagged-proposition moat-audit resonates. A per-topic cache exists (`:284`/`:316`) but only helps *within* a turn / repeated topics. **GPU is not the fix**: the CYCLE-612 probe (`_firstchat_gpu_latency_probe.py`) found the GPU `OneBrainComposer` co-resident bridge (~54 K neurons) is ~1 s/query — *slower* than the small-bridge CPU `RFPhasorComposer` for the DiscursiveTurn's many-ops-per-turn pattern. The small CPU composer is the right substrate; the latency win is in **doing fewer resonates** (cache/index/cap), not bigger hardware.

---

## 2. Question 1 — KNOWLEDGE ACQUISITION at scale: ranked options

**Goal:** give the brain *meaningful* facts/relations across ~1,454+ concepts, sourced from the corpus's real sentences, stored so the brain can recall + generalize them, with the moat intact.

### Catalog + Kandel grounding

- **G.13 Wernicke's area — auditory→semantic mapping** (`feature-catalog.md:2786`; Kandel 6e Ch 55 pp 1384–1385). Sim status **missing**; **prerequisite explicitly "semantic memory store."** This is the gap we are filling.
- **G.11 Dual-stream model** (`:2762`; Hickok & Poeppel; Kandel Ch 55 pp 1380–1387): ventral stream = sound→meaning (comprehension/semantic), dorsal = sensorimotor (production). Knowledge lives on the ventral side. Sim status **missing** — the project's stream cortex + composer is the functional ventral-stream stand-in.
- **D.01 Episodic memory → consolidation** (`:1085`; Kandel Ch 52 pp 1296–1302) and **D.02 Relational binding ("memory space", Eichenbaum–Cohen)** (`:1098`; Kandel pp 1301–1302): events as **items-in-context**, networked via overlapping events to support flexible (incl. transitive) inference. Sim status **missing as a system** but the substrate exists (regions, plasticity gates, the composer's role-filler binding). **A fact-base of `(agent, relation, patient)` triples that share arguments IS an Eichenbaum–Cohen relational network** — `(dog,chase,cat)`, `(cat,chase,mouse)`, `(hawk,chase,fox)` form a relational graph the `query_chain` multi-hop walk already traverses.
- **D.14 engram-tagging** (Tonegawa; the shipped `bridge.start_engram_recording/commit_engram_tag/stimulate_tag`, `sim/bridge.py:3352/3381/3466`) — a complementary *episodic* binding for facts that should be stored as a co-fired ensemble rather than a VSA composite.
- **Semantic-memory generalization is the hub-and-spoke story** (Lambon Ralph et al.; [Nature Rev Neurosci 2016](https://www.nature.com/articles/nrn.2016.150); [Patterson et al. hub-and-spoke](https://www.researchgate.net/publication/301244457)): the anterior temporal lobe **hub** integrates modality-specific **spokes** into coherent concepts and computes *semantic* (not superficial) generalizations. The project's prior generalization arc already converged on this (cross-modal Hebbian convergence; `CLAUDE.md` "GENERALIZATION across SIMILAR concepts"). Relevant here: a *meaningful* fact-base + similarity-structured codes is what lets `query_patient` on an unseen-but-related cue land in the right neighbourhood.

### Literature grounding (fact extraction + biological plausibility)

- **Open Information Extraction (OpenIE)** — extracting `(arg1, relation, arg2)` triples from free text in an unsupervised, domain-independent way is a mature field; the dependency-parse + verb-frame approach is standard ([oie-resources](https://github.com/gkiril/oie-resources); [Neural OpenIE](https://www.researchgate.net/publication/334117206)). A relational triple `(entity, relation, entity)` is exactly the knowledge-graph atom.
- **Usage-based / item-based acquisition (Tomasello)** — children build abstract constructions out of concrete **verb-island** patterns heard in adult speech, via statistical learning over the distributional/frequency properties of the input ([Tomasello, First Steps](https://terpconnect.umd.edu/~israel/Tomasello-FirstSteps-01.pdf)). This is the biological licence for: *extract the concrete `(a,v,p)` patterns the corpus actually contains, store them, generalize later.* The brain's knowledge IS its accumulated heard usage, not invented combinations — which is precisely what `_make_svo_facts`'s random sampling violates.
- **Selectional preference via PMI** is the established distributional measure ([distributional semantics survey, arXiv:1203.1858](https://arxiv.org/pdf/1203.1858)); the project's `_plausible` gate already uses it. The lever is making `P` reliable (coverage), not changing the gate.

### Ranked options

**Option K1 — Corpus SVO fact-extraction → real fact-base (RECOMMENDED, cheap-first).**
Parse the corpus sentences into real `(agent, action, patient)` triples restricted to the learned vocab, count them, keep the high-frequency / high-confidence ones, and store *those* (replacing `_make_svo_facts`'s random sampling). Two extraction tiers, both host-side **environment preprocessing** (legitimate per BRAIN-BASED-ONLY — this is preparing the curriculum, exactly like rendering a retinal image; the brain still *stores/recalls/generalizes* via spikes/binding):
- **K1a (cheapest):** a lightweight verb-centred window heuristic over the existing token stream — for each known verb token, take the nearest preceding known-noun (subject) and nearest following known-noun (object) within the clause; count `(subj, verb, obj)`. Pure-Python, reuses `corpus_stream` tokenization, no new dependency. Noisy but corpus-true and frequency-rankable.
- **K1b (better, available now):** real dependency parse — `spaCy 3.8.11` **is installed** (verified) and currently **unused for this**. `nsubj`/`dobj`/`pobj` extraction yields clean SVO triples; lemmatize and intersect with the learned vocab. This is the standard OpenIE-grade extractor; ~minutes to run on TinyStories, longer on full Simple-Wiki (chunk it).
- **Storage:** drop the resulting top-N triples straight into `RFPhasorComposer.store(a, v, p)` (`rf_phasor_composer.py:528`) — no API change. **Pros:** the chat becomes about real things ("dogs chase cats", "birds fly"); the moat is unchanged (absent cues still abstain); reuses 100% of the composer + DiscursiveTurn; immediately fixes most of GAP 3 (the proposer now recombines *real* arguments). **Cons:** extraction quality is the new variable; needs the anti-cheat that stored facts are corpus-attested (§5).

**Option K2 — Multi-bridge concept split for relational COVERAGE (RECOMMENDED, the breadth lever).**
The window-sweep's own honest-scope conclusion: a single bridge can't hold 1,454 distinct codes *with coverage* without densifying. The validated fix is the **320-tier recipe that held recall at 150 K**: ~5 bridges of ~290 concepts each at the proven `n_per=16–24`. `g20_multibridge.py --sparse` already does cross-bridge sparse storage + recall (per CLAUDE.md: 5 bridges × 64 = 320 @ 98.4%; 160-concept ensemble end-to-end). Each bridge keeps its frequent-core fidelity; the union covers the full 1,454 with *reliable* co-occurrence on each tier. **Pros:** directly fixes the noisy-rare-tail half of GAP 2; linear in bridge count; reuses the sparse ensemble machinery. **Cons:** cross-bridge fact storage (a fact spanning two bridges) uses the engram-multitag path, not the single composer — more orchestration; per-bridge train ~17 min (320-tier) so ~5 bridges ≈ 1.5 h.

**Option K3 — Engram-tagged episodic facts (catalog D.14) for the relational store.**
Store each extracted fact as a co-fired engram (`commit_engram_tag`) spanning the agent + relation + patient pools; recall by `stimulate_tag`. **Pros:** biology-faithful episodic→semantic path (D.01); persists through checkpoints; complements the VSA composite for facts that don't fit one composite. **Cons:** the VSA composite is already the project's validated who/what/abstain substrate; engram-tagging is a *parallel* store with its own recall semantics — adds a second mechanism. **Rank: secondary** — use only for cross-bridge facts (K2) or if the composite KB hits the linear-degradation wall.

**Option K4 — bigger develop-loop syllabus (the cumulative-knowledge vehicle).**
The develop-loop's `GradedCurriculum` (`_longitudinal_develop_loop.py:144`) already uses **real, sensible hand-authored** SVO facts (`("dog","eat","apple")`, `("fox","chase","rabbit")` — `_GRADED_SYLLABUS:94`), grown day-over-day with zero catastrophic forgetting (6/6-seed GO). It caps at ~24 vocab / ~11 facts purely because the syllabus is hand-written and tiny. **Feed K1's extracted fact-base into the syllabus** and the develop-loop becomes the persistent, resumable, never-restart vehicle for the whole knowledge base. **Rank: the integration target for K1**, not a standalone knowledge source.

**Option K5 — LLM-authored curriculum (the documented offline path).** The grounded-language faculty (`2026-06-23-grounded-lang-*`) re-encodes a Claude-authored *offline* curriculum (recall 1.0, abstain 0-FA). This is a legitimate knowledge *source* but it is authored knowledge, not the brain learning from the corpus, and the project's north-star is the latter. **Rank: fallback** if corpus extraction proves too noisy at scale; note it is offline (no runtime LLM), so it does not violate the LLM-minimal goal.

**Knowledge verdict:** **K1b (spaCy SVO extraction) + K2 (multi-bridge coverage), integrated into K4 (develop-loop syllabus)**. K1 gives *true* facts (fixes the fabrication half of GAP 2 and most of GAP 3); K2 gives *reliable relations across the full vocab* (fixes the noisy-tail half); K4 makes it cumulative/persistent. K3/K5 are reserves.

---

## 3. Question 2 — BRAIN-NATIVE FLUENT GENERATION (no LLM): ranked options

**Goal:** go beyond the fixed SVO template toward varied, meaningful sentences produced by the brain from meaning.

### Catalog + the canonical production models

- **G.10 Language as hierarchical symbolic system** (`feature-catalog.md:2750`; Kandel Ch 55 pp 1370–1372): finite units → infinite utterances via syntactic rules. Sim **missing**.
- **G.12 Broca's area — speech production + grammatical processing** (`:2774`; Kandel Ch 55 pp 1382–1384): maps stored word-forms to articulation; supports grammatically complex sentences. This is the grammatical-encoding seat. Sim **missing**.
- **G.07 Pre-SMA/SMA — internally generated sequences** (`:2710`) / **H.19 premotor sequential action** (`:3110`): the **parallel→serial conversion** the project already realizes in `NeuralSerialOrderRenderer` (Grossberg competitive queuing). This is the *late* stage of production — it serializes an already-assembled message.
- **Levelt's blueprint** ([Levelt, Roelofs & Meyer, BBS 1999](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/theory-of-lexical-access-in-speech-production/7E4A98E8791AB85397761DAAB35288AA); [WEAVER++ PNAS 2001](https://www.pnas.org/doi/10.1073/pnas.231459498)): **conceptualization → lemma selection (retrieval by spreading activation, verified by a production rule) → grammatical/syntactic encoding → phonological encoding**. Maps cleanly onto the project: conceptualization = the brain's chosen proposition (the proposer/composer); lemma selection = the concept→word A→W read-out; grammatical encoding = *the missing piece* (currently the fixed SVO frame); phonological = the word string. The project already does Levelt's stages **except grammatical encoding beyond one frame**.
- **Hagoort's MUC model** ([Hagoort 2013, Front. Psychol.](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2013.00416/full); Neurobiology of Language ch 28): **Memory** (stored lexical/構 structural knowledge, temporal cortex) + **Unification** (assembling lexical building blocks into a structured whole, Broca BA44/45) + **Control** (selecting context-appropriate output, social/joint-action). The "Unification" component is precisely what binds a chosen set of words into a grammatical structure — the function the project needs above the SVO template.
- **Chang–Dell–Bock dual-path connectionist sentence production** ([Chang, Dell & Bock 2006, Psych Rev](https://www.eva.mpg.de/documents/AmericanPsychologicalAss/Chang_Becoming_PsychRev_2006_1555016.pdf)): a connectionist model that learns syntactic abstractions and **generalizes constructions** via error-based implicit learning (the same learning that explains structural priming). Its **message → sequence** mapping with a separate "meaning" and "sequencing" path is the most directly implementable biologically-plausible route to *learned, varied constructions* (vs one hand-coded frame). This is the principled target if multi-frame syntax is prioritized.

### Ranked options

**Option F1 — Fix the input, keep the SVO frame (RECOMMENDED FIRST; nearly free).**
Most of the *felt* stiltedness ("curry describes pine") is GAP 2, not the renderer. With K1's real facts, the SVO frame producing "the dog chases the cat" reads fluent and meaningful. **Add only a determiner/agreement polish** (host-side surface morphology on the neurally-ordered triple — "a/the", verb inflection by number) which is the **body emitting motor output**, not cognition (legitimate per BRAIN-BASED-ONLY; the *word choice + order* stay neural). **Pros:** zero new mechanism; immediately lifts perceived fluency once K1 lands. **Cons:** still one construction type (declarative SVO).

**Option F2 — Frame inventory + neural frame selection (RECOMMENDED SECOND; reuses existing primitives).**
Add a small inventory of construction frames (declarative SVO; existential "there is a X"; attributive "the X is ADJ"; relative "the X that V'd"; conjunction "X and Y") and let the brain **select the frame** by the same SpikingSpeakAccumulator/worth machinery that already selects propositions, then fill + serialize each frame with the existing renderer. The renderer already orders arbitrary frame_concepts; frames differ only in their primacy gradient + slot set. **Pros:** real construction *variety* with the validated serialization + selection primitives; the moat is unchanged (each frame still VERIFY-gated). Catalog: this is MUC "Unification" as *frame assembly*. **Cons:** frames are hand-authored (a small, honest scaffold, like the develop syllabus); not *learned* syntax.

**Option F3 — Learned dual-path grammatical encoder (the principled, deeper build).**
Implement a Chang-style message→sequence learner (the project has the BPTT-SNN machinery from the generative-sequence arc, `project_generative_sequence_frontier.md`): train it on the corpus's real sentences (now that K1 extracts them) to map a meaning representation (the bound proposition) to a word sequence, learning constructions + agreement from data. **Pros:** genuinely *learned*, generalizing, varied syntax — the real Broca/MUC-Unification analogue; aligns with the owner-approved generative-sequence frontier. **Cons:** a mechanism-class build (weeks); needs its own de-risk; the moat must be enforced *outside* the generator (VERIFY-gate every output against the stored fact-base — the generator supplies *form*, the fact-base supplies *truth*, exactly the grounded-language decoupling). **Rank: the Stage-2-deep target**, gated on K1 producing enough clean sentence data.

**Option F4 — spiking-Qwen LLM fluency faculty (FALLBACK ONLY).**
The documented spiking-Qwen faculty (`2026-06-23-grounded-lang-INTEGRATION-GO.md`) supplies *fluency only*; the brain supplies knowledge + the VERIFY moat caught a real hallucination. **Where it is genuinely unavoidable:** if the requirement is *open-ended, fully grammatical, human-quality prose over arbitrary topics* — long varied sentences with subordination, discourse connectives, register — no brain-native mechanism in or near the project reaches that bar, and F4 is the only option that does. **Where it is NOT needed (prefer brain-native):** short, true, on-topic utterances about stored facts (F1/F2 cover these). Note F4 is the *fluency* faculty wrapped by the moat, **not** the free-generate cheat (which the DiscursiveTurn header explicitly rejects, `:49`). Use F4 as an *optional render backend* behind the same VERIFY gate, never as the knowledge source.

**Fluency verdict:** **F1 (free, ride on K1) → F2 (frame inventory, reuse selection+serialization) → F3 (learned dual-path, the deep build)**. F4 stays a clearly-bounded, moat-wrapped fallback for open-ended prose, explicitly not on the critical path.

---

## 4. Reusable project machinery (read-only survey; absolute paths)

| Need | Tool / API | File:line | Notes |
|---|---|---|---|
| Corpus → vocab + codes | `derive_curriculum_from_corpus`, `StreamCortexBridge.hear_corpus/read_codes`, `--save-codes` | `research/runners/_curriculum_step1_320_real_corpus.py:238,365,428,460` | produces the `.npz` brain (`vocab`, `grounded`, `code`, `M`, `cat_ids`). Combined-corpus aware. |
| Corpus tokenization | `iter_stories`, `iter_stories_multi`, `normalize_corpus_paths` | `research/runners/corpus_stream.py:53,155,131` | `re.findall(r"[a-z]+")`, `<|endoftext|>`-split. **Bag-of-words only — no sentence/dep structure** (the GAP-2 extraction point). |
| **Fact storage + recall + moat** | `RFPhasorComposer.store/query_patient/query_agent/ask_yes_no/render_fact/query_chain/elaborate/update_on_mismatch` | `research/runners/rf_phasor_composer.py:528,681,664,736,758,721,795,583` | `grounded_codes={word:phases}` injects learned codes. Moat = `→ None` on absent cue. Drop-in target for K1's facts. |
| On-bridge spiking composer | `OneBrainComposer` (RFPhasor-API-compatible) | `research/runners/one_brain_composer.py:107` | ~54 K-neuron co-resident bridge; **~1 s/query — slower than the CPU composer for many-ops turns** (CYCLE 612). GPU only when one big batched op dominates. |
| Discursive turn | `DiscursiveTurn.discuss`, `CommunicableTurn.propose_candidates_about` | `research/runners/_discursive_turn_stage0_derisk.py:471`; `_communicable_turn_stageA_derisk.py:311` | typed-proposition assembly + type-aware VERIFY moat (`:402`). The `propose_candidates_about` resonate-per-candidate is the GAP-1 cost. |
| First-chat console | `build_brain_on_codes`, rubric | `research/runners/first_chat_console.py:95,154` | loads the 7K brain + wires DiscursiveTurn; **uses `_make_svo_facts` random facts** (the GAP-2 site to replace). |
| Brain-native word order | `NeuralSerialOrderRenderer.order/render` | `research/runners/neural_serial_order_renderer.py:50,60` | competitive-queuing serialization of a *given* frame; the F2 fill-and-serialize primitive. |
| Multi-bridge breadth | `g20_multibridge.py --sparse`, `build_sparse_pool_bridge`, `SharedPoolMember` | `research/runners/g20_multibridge.py` | 5 bridges × 64 @ 98.4%; cross-bridge engram multitag. The K2 coverage vehicle. |
| Develop loop + persistence | `GradedCurriculum`, `DevelopState`, `BridgeLineage.save/load` | `research/runners/_longitudinal_develop_loop.py:144,209`; `sim/lineage.py:140` | hand-authored **real** SVO syllabus (`:94`), grows + persists with zero forgetting (6/6 GO). The K4 cumulative vehicle. `current.simstate.h5` + `metadata.json` + `history/`. |
| Episodic binding | `start_engram_recording/commit_engram_tag/stimulate_tag` | `sim/bridge.py:3352,3381,3466` | Tonegawa D.14; persists in checkpoint. K3 reserve. |
| Novel-proposition generator | `GenerativeReplayProposer.propose`, `_plausible`, `_weight_partner` | `research/runners/_genfrontier_b2_generative_replay_derisk.py:250,179,203` | PPMI selectional-preference sampling — *the mechanism is right*; it inherits the graph's noise. |
| Parsing libs (for K1b) | `spaCy 3.8.11`, `nltk 3.9.4` | installed, **currently unused for SVO** | enables clean `nsubj/dobj` extraction host-side. |

**Bottom line:** every consumer of facts already exists and needs no API change. The missing object is a **corpus→fact extractor** (host-side preprocessing) plus the **multi-bridge coverage** wiring and a **frame inventory**. This is composition, not new substrate — consistent with reuse-by-import / no-`sim`-edit.

---

## 5. Recommended STAGED cheap-first plan (each stage = a usable brain)

The framing matches the owner's cumulative directive: **every stage yields a deployable brain on the lineage; the next stage builds on it without retraining.** All stages are reuse-by-import; no `sim/` edits anticipated.

### Stage 0 — latency relief (independent, ~hours; do in parallel, ships first)
Make the *current* 7K-brain console usable for the owner's first chat while Stage 1 builds.
- **Cheapest de-risk first:** profile `propose_candidates_about` and (a) cap `n_attempts` adaptively (stop after K accepted candidates), (b) **persist the per-topic candidate cache to disk** keyed by `(brain-hash, topic)` so repeated topics across sessions are instant, (c) index the composer KB by `(agent, action)` so `query_patient` is a dict lookup before any resonate. Target: <3 s/turn on the existing brain.
- **Usable brain produced:** the 7K brain, same knowledge, responsive.
- **Builds-on:** all later stages inherit the faster turn loop.

### Stage 1 — KNOWLEDGE (the headline; ~1–2 days)
**Stage 1a (cheapest de-risk, run FIRST):** on the *existing* 7K brain, run K1b (spaCy) over TinyStories, extract `(subj, verb, obj)` triples restricted to the 1,454 vocab, count, keep the top-N attested triples, store *those* via `comp.store(...)` instead of `_make_svo_facts`. Re-run the 10-prompt rubric. **GO test:** the chat now states *corpus-true* facts ("dogs chase cats"); moat still 0-leak; the proposer recombines real arguments (GAP-3 stiltedness measurably drops). This is a few hours and decides whether extraction quality is sufficient before any retrain.
- **Stage 1b (coverage):** stand up K2 — ~5 sparse bridges of ~290 concepts at `n_per=16–24` over the combined corpus (the 320-tier recipe that held recall at 150 K), giving reliable relations across the full 1,454. Extract facts per bridge + cross-bridge.
- **Stage 1c (cumulative):** feed the extracted fact-base into the develop-loop syllabus (K4) and persist via `BridgeLineage` — the brain now *accumulates* corpus knowledge day-over-day with zero forgetting, resumable.
- **Usable brain produced:** a 1,454-concept brain that knows *real* facts with reliable relations, on the lineage.
- **Builds-on Stage 0** (fast turns) and is the substrate Stage 2 generates from.

### Stage 2 — BRAIN-NATIVE FLUENCY (~days, after Stage 1 facts exist)
- **Stage 2a (free):** F1 — confirm the SVO frame now reads fluent on real facts; add host-side determiner/agreement surface polish (body-level).
- **Stage 2b (cheap de-risk):** F2 — add a 4–5 frame inventory + neural frame selection (reuse SpikingSpeakAccumulator + the serial-order renderer per frame). **GO test:** ≥2 construction types appear in a sample chat, each VERIFY-gated, moat intact.
- **Stage 2c (deep, optional/owner-gated):** F3 — train the dual-path BPTT-SNN grammatical encoder on the corpus sentences Stage 1 extracted; VERIFY-gate every output against the fact-base. **GO test:** novel grammatical constructions, generalize to unseen argument fillers, 0 fabrication (the LESION arm caught by VERIFY).
- **Usable brain produced:** the Stage-1 brain that *speaks varied, fluent, true* sentences.
- **Builds-on Stage 1** (its facts + extracted sentences); the fact-base is unchanged, so no retraining of knowledge.

### Stage 3 — SPEED at scale (ongoing)
- Training throughput: the densification ceiling (`2026-06-26-breadth1454-window-sweep.md`) is handled by K2's multi-bridge split (each bridge trains in the structured regime); fact-extraction throughput is a one-time host preprocess (chunk Simple-Wiki).
- Interaction: extend Stage 0's caching/indexing; keep the **CPU small-bridge composer** as the substrate (GPU only for a genuinely batched single op — characterize per-op: GPU wins when one op processes many facts at once; CPU wins for the DiscursiveTurn's many small sequential ops).
- **Usable brain produced:** the Stage-2 brain, fast + scalable. No knowledge/fluency regression.

**Why this is never-wasted:** Stage 0 ships immediately; Stage 1's fact-base + lineage is the durable asset every later stage reads; Stage 2 only *adds* a render path over the same facts; Stage 3 only *speeds* the same brain. A week of Stage-1 training persists on the lineage and is resumed, never restarted (the develop-loop's verified property).

---

## 6. Anti-cheat controls per stage

- **Stage 0 (latency):** byte-identical answers before/after caching (the cache changes *speed*, not *which* candidates/decisions) — assert the rubric transcript is unchanged. No moat impact (no new content path).
- **Stage 1 (knowledge) — the load-bearing controls:**
  - **Corpus-attestation:** every stored fact must be traceable to ≥1 (or ≥N) corpus sentence (log the source sentence + count). A fact with 0 attestation is the `_make_svo_facts` failure mode and must be rejected. (Distinguishes *learned* from *fabricated*.)
  - **Moat unchanged (HARD):** absent-cue `query_patient/ask_yes_no → None`; 0 false-accepts at 1,454, re-verified after storing real facts. The moat is *content-agnostic*, so real facts must not weaken it.
  - **Extraction-quality control:** a held-out hand-labelled sample of corpus sentences → precision/recall of the extractor; report it (don't gate the chat on magnitude, but know the noise floor).
  - **Multi-bridge specificity:** cross-bridge recall must collapse under a bridge-scramble control (the cross-bridge link is real, not coincidence) — the existing g20 ensemble anti-cheat.
- **Stage 2 (fluency):**
  - **VERIFY gate on every utterance (HARD):** a CERTAIN sentence must re-parse to a STORED fact; a flagged/novel sentence must abstain on a who/what + never be stored (the DiscursiveTurn `:402` invariant, re-asserted on the new frames/generator).
  - **LESION anti-cheat:** sever the brain's proposition / fact-base and let the renderer (or F3 generator) free-generate → VERIFY must reject the fabricated output (the header's `:54` control, run on the new construction types).
  - **Construction provenance:** for F2, the frame *selection* must be the spiking accumulator's firing (read `cp_firing_states`), not a host `if`; for F3, a shuffled-training-corpus control must collapse the learned constructions (proving they are learned from data).
- **Stage 3 (speed):** answer-identity across CPU/GPU substrates for any op that is moved (the project's standing CPU-oracle-vs-GPU parity check); no rubric regression.

---

## 7. Honest open risks

1. **Extraction noise (Stage 1a).** A heuristic/shallow-parse SVO extractor over child-story prose will produce some wrong triples (coreference "he/she", light verbs "is/has", multi-clause sentences). Mitigation: frequency-threshold + the corpus-attestation control + lemma/vocab intersection; report precision on a labelled sample. spaCy (K1b) substantially de-risks this vs the pure heuristic (K1a). **The moat protects against the *consequence* — a wrong-but-attested fact is still abstained-against if not queried, and never fabricated — but a confidently-stored wrong fact is still wrong.** This is the genuine residual: the brain will be only as truthful as the extractor.
2. **VSA composite scaling (Stage 1b/c).** Retrieval degrades ~linearly with facts-per-composite ([VSA survey](https://link.springer.com/article/10.1007/s10462-021-10110-3)). A few thousand facts across one composite KB will hit this; the mitigation is the multi-bridge split (K2) + per-`(agent,action)` indexing + possibly engram-tagged storage (K3) for overflow. Needs a capacity de-risk (how many real facts can one composer hold at recall ≥0.95?).
3. **F2 frames are a scaffold, not learned syntax.** Honest scope: a hand-authored frame inventory gives *variety*, not *productive grammar*. The genuinely-learned path is F3 (dual-path), which is a real mechanism-class build with its own variance. Don't overclaim F2 as "the brain learned grammar."
4. **F3 / generative syntax is unproven on this substrate.** The generative-sequence frontier (`project_generative_sequence_frontier.md`) is owner-approved but the *grammatical-encoding-from-meaning* version is new; the categorical-novelty gap (`2026-06-22-generation-novelty-categorical-gap-MEASURED.md`) is a documented prior negative in the neighbourhood. Treat F3 as research-gated.
5. **Generalization stays substrate-capped.** Per `2026-06-26-gen-readiness-bar-recalibration.md`, the point-neuron read-out caps semantic generalization (~+0.065 Pearson @320); meaningful facts + similarity-structured codes *help the recall neighbourhood* but do not lift the fundamental cap. Knowledge depth ≠ generalization; keep them separate in claims.
6. **The LLM-fluency line (F4) is a real temptation.** The owner's directive is brain-native; F4 is genuinely better at open-ended prose. The discipline is to use it *only* behind the VERIFY moat as an optional render backend for open-ended turns, never as the knowledge source or the default — and to be explicit in any result about which path produced the text.

---

## 8. One-paragraph recommendation for the controller

The console works; the *content* is hollow. Do **Stage 0** (cache/index the resonate-bound turn → <3 s, ships the current brain immediately) in parallel with **Stage 1a** (the cheapest decisive de-risk: spaCy-extract real `(subj,verb,obj)` triples from TinyStories on the existing 7K brain, store *those* instead of `_make_svo_facts`'s random recombinations, re-run the rubric). If Stage 1a's chat states corpus-true facts with the moat intact, scale knowledge via **Stage 1b/c** (the 320-tier multi-bridge split for reliable relations across all 1,454, fed into the develop-loop syllabus on the lineage for cumulative, resumable, zero-forgetting knowledge). Fluency then mostly follows for free (**F1**); add a small **frame inventory (F2)** for variety using the already-validated selection + serial-order primitives; reserve the **learned dual-path encoder (F3)** as the deeper, research-gated build and the **spiking-Qwen (F4)** strictly as a moat-wrapped fallback for open-ended prose. The single highest-leverage, lowest-cost action is **Stage 1a** — it converts the brain from "knows 1,454 words, 24 made-up facts" to "knows real things the corpus says," which is the difference between a vocabulary and knowledge.
