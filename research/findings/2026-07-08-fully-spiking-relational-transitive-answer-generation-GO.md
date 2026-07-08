# Fully-spiking GENERATION of the real-corpus RELATIONAL (transitive SVO) answer (GO, 3-seed): "the dog eats the cat" — the C_TRANS slot ORDER produced ON SPIKES by the EMERGE-72/74 signature-keyed registry producer, and EVERY word (incl. the 3sg verb surface) spelled ON SPIKES by a new 3-bridge A→W (BRIDGE-3 = 3sg verb forms). Extends the property-answer generation to the transitive construction — the biggest spoken-expressivity jump. Gate-first moat intact; NO `sim/` edit.

**Date:** 2026-07-08
**Runners:** `research/runners/_realcorpus_spiking_broca_relational_answer_derisk.py` (de-risk); `_realcorpus_train_breadth_aw3.py` (BRIDGE-3 trainer); `_realcorpus_multi_bridge_speaker.py` (3-bridge dispatch). CI guard `tests/test_realcorpus_unified_console.py::test_relational_answer_generated_on_spikes`. numpy. NO `sim/` edit.
**Verdict:** GO (3-seed) — the transitive answer's structure + every word are produced on spikes, moat holds.

## Why this ran (the fluency fork, extended to the transitive)
The property answer ("the X can Y", F_MODAL) already generates fully on spikes (committed). The transitive relational answer ("the dog eats the cat", C_TRANS) is the biggest remaining spoken-expressivity jump — arguments AFTER the verb. Its residual was precisely pinned: the EMERGE-74 C_TRANS slot ORDER is already spiking-proven, and the subject/object/the are covered by the existing breadth A→W — the ONLY gap was the **3sg verb surface** ("eats"), which BRIDGE-1/2 (both at 16-word capacity) do not hold. Frequent inflected forms are lexically stored (Pinker's words-and-rules), so a dedicated A→W bridge for the core-SVO 3sg surfaces is biologically defensible.

## What was built
- **BRIDGE-3** (`_realcorpus_train_breadth_aw3.py`): a 16-word concept-pool A→W over the 9 core-SVO 3sg verb surfaces (eats/chases/sees/likes/wants/hugs/finds/holds/kicks) + "a" + 6 more animal subjects (wolf/rabbit/lion/fox/mouse/owl). Trained on GPU (1500s, 16/16 words).
- **`MultiBridgeFrameSpeaker`** (`_realcorpus_multi_bridge_speaker.py`): generalizes the EMERGE-68 two-bridge dispatch to N bridges. Combined vocab = 46 words across BRIDGE-1/2/3; **all-word spell accuracy 46/46 on spikes** (every 3sg verb reads correctly).
- **The relational producer**: the C_TRANS construction is MINED from the corpus stream (EMERGE-74's `build_stream_svo`); the FILLERS are the console's facts. `RegistryBrocaProducer(cq, spell=multi_speaker.spell)` renders the slot order on spikes and spells every slot (incl. the 3sg VERB surface) via the A→W. Gate-first (abstain → never invoked → moat).

## The result — de-risk, 3-seed (42/43/44)
```
reason(dog,  eat,   cat)    -> "the dog eats the cat"      [exact, order+words ON SPIKES]
reason(wolf, chase, rabbit) -> "the wolf chases the rabbit"[exact]
reason(fox,  see,   bird)   -> "the fox sees the bird"     [exact]
reason(bear, like,  fish)   -> "the bear likes the fish"   [exact]
ABSTAIN                     -> producer NOT invoked (0 productions)  [gate-first moat]
VERDICT: GO (4/4 exact, all seeds)
```
- The **C_TRANS SLOT ORDER** ("the" → subject → 3sg-verb → "the" → object) is the spiking rate-ranking of the registry producer (competitive queuing + EMERGE-61 wash-out).
- Every **WORD** — including the 3sg verb surface ("eats", "chases") — is spelled on spikes by the 3-bridge A→W read-out.
- The **gate-first moat** holds: 4 productions for 4 answers, 0 on the abstain.
- The 3sg inflection follows EMERGE-74's standard: `emerge_v3` produces the surface, which is then spelled on spikes (morphology-on-spikes is the deeper documented residual, shared with EMERGE-74).

## Console integration
`UnifiedTalkableConsole(..., spiking_gen=True, multi_bridge=True)` builds the multi-bridge speaker + the C_TRANS producer; `_speak_svo` routes the relational answer through it when every filler is spellable (else the host-template fallback). Default OFF is byte-unchanged (`_svo_producer is None`). CI guard: teach "dog eat cat" → ask "what does the dog eat?" → "the dog eats the cat" (order + words on spikes), moat unaffected.

## What this establishes
The talkable brain's TRANSITIVE relational answer is now generated fully on spikes (structure + words) — extending the fluency fork from the property answer to the core SVO construction, the biggest spoken-expressivity jump. Transformer-free, gate-first moat intact. Follow-on: the who/yes-no relational answers route through the same producer (same `_speak_svo`); the describe contrast connective; ditransitive (EMERGE-77 n_slot_pools lever); morphology-on-spikes (the shared EMERGE-74 residual).

## Files
`research/runners/_realcorpus_train_breadth_aw3.py`, `_realcorpus_multi_bridge_speaker.py`, `_realcorpus_spiking_broca_relational_answer_derisk.py`; `_realcorpus_unified_talkable_console.py` (`multi_bridge` flag + SVO producer + `_speak_svo` routing); `tests/test_realcorpus_unified_console.py`. Reuses EMERGE-72/74 (registry producer), EMERGE-61 (wash-out), the concept-pool A→W. Prior: the property-answer generation `2026-07-08-fully-spiking-generation-real-corpus-property-answer-GO.md`, `2026-07-08-flagship-console-property-answer-generated-on-spikes-GO.md`.
