# Conversation-depth: brain-based audit + burndown to fully-spiking-one-brain (2026-06-27)

Owner directive (this session): the END RESULT must be FULLY SPIKING on the ONE-BRAIN shared substrate
(non-negotiable); the PATH per-capability is my efficiency call — scaffold-first-then-clean OR
biological-from-start, whichever is cheaper. This doc is the living inventory + the burndown sequence.
Memory: `feedback_end_state_fully_spiking_one_brain_path_by_efficiency`.

## Per-component audit (VERIFIED = read in code this session; RECORD = per the project's validated findings)

| Component | What the CONSOLE runs today | Spiking form exists? | Status |
|---|---|---|---|
| Role-filler **bind / unbind** | numpy phasor algebra (`RFPhasorComposer`, `enable_substrate_store=False`) [VERIFIED] | YES — RF resonate-and-fire + complex synapses on-bridge; onebrain GPU path; byte-equivalent [RECORD] | OPERATION — easy converge |
| **Serial-order** (word order) | numpy `FrameCQ` reimpl of the validated spiking CQ [VERIFIED `argstructure_composer.py`] | YES — the 6/6-GO spiking competitive-queuing renderer [RECORD] | OPERATION — easy convert |
| **No-confab moat** | host familiarity check / numpy cleanup [RECORD] | YES — spiking Bogacz-Brown familiarity gate + spiking NEF cleanup, validated [RECORD] | OPERATION — easy converge |
| **Cleanup** (nearest-concept) | numpy argmax (console) [RECORD] | YES — spiking Izhikevich WTA / NEF cleanup, default-on in onebrain [RECORD] | OPERATION — easy converge |
| **Comprehension parser** | the BridgeParser (Hebbian-learned, on-bridge) for SVO; **regex** for wh-questions [VERIFIED `wh_question_parser.py`] | parser=spiking; wh-parse=host | MIXED |
| **Verb-frame structure** (`FRAME_LEXICON` go→GOAL) | hand-authored host dict [VERIFIED] | NO — learned grammar is the frontier | STRUCTURE — scaffold-then-learn |
| **wh→role map** (`WH_ROLE_CANDIDATES`) + **closed-class words** (`_BARE_LEAD` "to the") | hand-authored host dicts [VERIFIED] | NO | STRUCTURE — scaffold-then-learn |
| **Working memory** | `OrderedPositionWM` fixed-slot spiking buffer [RECORD] | YES (spiking) | OK |
| **Entity-instance / file-card** (Tier 1, in flight) | engram-barcode + DG/CA3 + WTA = spiking; the DRT surface-ref index = host | partial | MIXED (Tier 1) |

## The burndown sequence (cheap-first, per the path-by-efficiency directive)

**Bucket A — OPERATIONS, easy converge (the spiking form already exists + is byte-equivalent; wiring, not research):**
1. Console composer → the spiking RF substrate / onebrain (`enable_substrate_store=True` or the onebrain path). The
   project already ships this on GPU; the console runs numpy-reference only for CPU speed/portability. Converge the
   GPU/prod console onto spiking; keep numpy-reference as the CPU test-oracle (per `feedback_close_arcs_to_full_capacity`).
2. `FrameCQ` numpy → the validated spiking competitive-queuing renderer (`neural_serial_order_renderer`).
3. Moat + cleanup → already have spiking forms default-on in onebrain; ensure the console path uses them on GPU.
   *These are not a frontier — they are the standing one-brain consolidation (`project_one_brain_integrated_pipeline_and_cleanup`).*

**Bucket B — STRUCTURE, scaffold-then-LEARN (the deep frontier; scaffold now is far cheaper than learn-from-start):**
4. Verb-frame lexicon, wh→role map, closed-class words: today hand-authored. The learned version = the brain
   self-organizing grammar from the corpus — the BPTT-SNN scale + dendritic frontier (`feedback_spiking_structure_must_self_organize`,
   `project_generative_sequence_frontier`). Scaffold now (days), learn later (months). This is the genuine Tier-3 wall.

**Bucket C — MIXED (close opportunistically):**
5. wh-question parse (regex → on the spiking parser); the Tier-1 DRT file-card (host index → spiking pointer binding).
6. **Tier-1 which-X candidate SCORING** — a host loop today (the WIN — codes/binding/abstain — is spiking RF); → the de-risked spiking **biased-competition WTA** (`biased_competition_buffer.py`). Tier 1's mechanism (token = type code ⊕ a spiking D.14 barcode; facts bound via RF) is otherwise brain-based.

## Efficiency rationale (why scaffold-first here)

- STRUCTURE (Bucket B) is cheap to scaffold (host dicts, done) and expensive to learn (the deep frontier). Blocking the
  capabilities on the learned-grammar problem would stall the whole roadmap for months. Scaffold-then-learn is the
  efficient path, and the scaffolds are TRACKED here so they retire, not linger.
- OPERATIONS (Bucket A) already have validated spiking forms — converging them is wiring (the one-brain consolidation
  that is already the project's north-star), not a frontier. Do these as the cheap, high-value early burndown.

## Honest note

The console you chat with today is a numpy REFERENCE of a spiking architecture plus host-authored linguistic
structure. It computes what the validated spiking substrate computes (byte-equivalent), but is not itself firing
neurons in the CPU path, and the grammar is not self-organized. Neither is a result-faking cheat; both are tracked
shortcuts on the path to fully-spiking-one-brain. The no-confab moat is preserved throughout.
