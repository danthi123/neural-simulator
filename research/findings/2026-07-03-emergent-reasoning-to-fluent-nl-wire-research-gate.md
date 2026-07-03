# Research gate — wiring the EMERGENT grounded REASONING (EMERGE-51..55) to FLUENT NL: CHEAPLY WIREABLE (a gate-first adapter, NOT a new mechanism). Wernicke-decides → Broca-articulates. Controller-reviewed read-only deep-research.

**2026-07-03 (autonomous).** Read-only scoping subagent (grep + read across the emergent-substrate console + the grounded-language / fluid-conversation faculty). The standing "deep-research FIRST at a new direction" gate for the north-star direction (memory `project_fluid_llm_like_conversation_priority`): make the emergent substrate's conversation FLUENT (not templated).

## Isolate the residual — it's an ADAPTER, not a mechanism
- **EMERGE-51..55 produces** a structured inference decision: `ask_can(member, prop)` returns abstain (moat) | inherited class default | member-specific exception — i.e. `(gate_decision, subject, property, source)`.
- **The fluent faculty (grounded-language / fluid-conversation Phase 2/3) takes** exactly: a GATE-FIRST boolean (answer vs abstain) + the grounded fact as an SVO triple, rendered by the RA-fine-tuned 21M via `facts: {the subject verb property} question: {..} answer:` → fluent answer → post-hoc VERIFY. If abstain, the generator is NEVER invoked (no-confab by construction).
- **The mapping is 1-to-1.** The only gap: EMERGE returns templated English strings; unwiring them into the structured tuple is a ~20-line parse/refactor. **No new mechanism, no dendritic circuit, no new learning rule.**

## Reframe via biology
Wernicke (comprehension + semantic retrieval + the DECISION) → Broca (articulation of a gated message). The project's gate-first design IS this division: the brain decides + supplies the grounded fact; the generator renders-only, never free-generating. Biologically sound + matches the owner's "minimize the transformer" principle (dual-stream language model; catalog G.11/G.13; Damasio convergence).

## Ranked cheap-first de-risks
1. **Rung 1 (CPU-native, cheapest, FIRST):** wire the EMERGE console's gated-fact output → a structured `(gate, subject, property)` tuple → a STUB renderer (the existing `TemplateStubFaculty`), moat preserved (abstain → renderer never invoked). NO GPU, NO checkpoint. A scripted 8–10-question demo. Proves the adapter + moat end-to-end.
2. **Rung 2 (GPU, ~30 min):** replace the stub with the RA-fine-tuned 21M (ckpt `gen_tinystories_ra_ft.ckpt.pt` — EXISTS in the repo); render the EMERGE gated fact fluently; match the Phase-2/3 reference (5/5 grounded + 3/3 gate-first moat).
3. **Rung 3 (integration):** merge into `_fluidconv_chat_repl.py` so `can a penguin fly?` (EMERGE) + `what does a dog eat?` (existing) both work with a consistent moat + fluency.

## Verdict
**CHEAPLY WIREABLE — adapter + architecture only.** EMERGE inference output READY; fluent faculty input spec READY; mapping trivial (1-to-1); RA ckpt in repo; moat preserved by the gate-first construction; biology aligned. First de-risk = **Rung 1 (CPU-native)**. Honest scope: this is wiring-not-mechanism (two orthogonal subsystems — on-brain spiking reasoning + a fine-tuned-ANN articulator — handed off via the gated-fact tuple; the generator ANN remains a tracked temporary scaffold per the fully-spiking end-state, its spiking-forward conversion already validated at 88.6M).

## Next
Build Rung 1 (EMERGE-56): the CPU-native gated-fact adapter proving EMERGE-reasoning → gate-first fluent-render, moat preserved; 3-seed / scripted; then Rung 2 (GPU RA-render) + Rung 3 (FluidChat integration). Cite the exact reuse: `_emerge51_experiential_conversational_console.py` (ask_can) + `_fluidconv_phase2_ra_finetune.py` / `_fluidconv_chat_repl.py` (the faculty + gate-first moat).
