# Fluid conversation — Phase 16 GO: grounded DISCOURSE PLAN → connected prose (multi-fact synthesis, cheap-first #1)

**2026-07-01 (autonomous; the research-gate cheap-first #1 from `2026-07-01-multi-fact-synthesis-frontier-scoping.md`,
controller-verified).** The synthesis frontier's scoping verdict: "DISCUSS lists facts" is a DISGUISED boundary — the
grouped rendering is ALREADY NLG synthesis (aggregation + referring-expression, per Levelt/Reiter-Dale), and the ~70%
cheap residual is **discourse connectives + same-subject/same-verb aggregation** over the brain's grounded facts. This
builds + de-risks + WIRES the deterministic plan-then-realize renderer. Reuse-by-import; **NO `sim/` edit**; **NO
train**; the no-confab moat holds **by construction** (every asserted fact is a stored triple).

## Result — GO (Phase-16 de-risk, then wired live into the console)
`_fluidconv_phase16_discourse_plan_derisk.py` — `plan_discourse(topic, facts)` turns a topic's grounded facts into ONE
connected prose (aggregate same-relation/same-verb + Joint/Elaboration connectives), and `compare_discourse(x, y)`
fires **checkable** Contrast/Additive connectives. All scenarios GO: DEPTH (fewer sentences than facts + ≥1 aggregated
clause + ≥1 connective), GROUNDED (0 invented tokens), CONNECTIVE-CORRECT (Contrast "but" IFF patients differ; Additive
"and so does" IFF shared verb+patient), LESION (empty → hedge).

**Live in the console** (`_fluidconv_chat_repl.py` `_discuss`/compare now import `plan_discourse`/`compare_discourse`):
- *"tell me about the elephant"* → **"An elephant is a mammal; it is grey and has trunk and tusk."** (was three
  sentences: "An elephant is a mammal. It is grey. It has trunk and tusk.")
- *"tell me about the dog"* → **"A dog is big. It eats meat, chases cat and likes bone."** (was four list-y sentences).
- *"tell me about the bird"* → **"A bird eats seed, likes worm and lives tree."**
- *"compare dog and cat"* / *"how are dogs and cats different?"* → **"the dog eats meat, but the cat eats fish. the dog
  chases cat, but the cat chases mouse. the dog likes bone, but the cat likes milk."** — genuine multi-fact comparison,
  every Contrast entailment-checked (the patients actually differ).
- Both offline self-checks (`--demo` + `--instance-demo`) STILL green (no regression); "compare"/"different" added to
  the question-trigger set so a compare-request (no wh-word) routes as a query, not a statement.

## Why this is synthesis, not a cheat (per the BRAIN-BASED-ONLY standard)
The connective/aggregation layer is **body/surface realization** of brain-supplied content, defensible on three counts
argued in the scoping doc: (1) it asserts **no new fact** — every proposition is a retrieved, VERIFY-clean stored
triple; (2) every connective is **entailed** by the grounded facts via a checkable predicate (Contrast IFF patients
differ, etc.) — a deterministic transform of brain content, not host cognition; (3) content SELECTION + ORDERING (the
macroplanning that IS cognition) stays brain-native (the neighbourhood retrieval + dlPFC ranking). This mirrors the
project's already-accepted grey-area call — the neural serial-order renderer (word ORDER is brain-produced). The wrong
hypothesis (condition the 21M on N facts, free-generate a paragraph) is exactly what MEASURED confabulation on this
model; plan-then-realize (Levelt macroplan→microplan→realize; Reiter-Dale; the field's plan-guided-SLM faithfulness
fix) moves structure INTO a checkable plan and asks the generator for LESS.

## Honest ceiling
- The connective INVENTORY (Joint/Elaboration/Contrast/Additive) + the entailment predicates are **host-authored** (a
  residual host structure, like the FRAME_LEXICON); the fully-brain-based Broca connective producer + self-organized
  RST relations are the deep follow-on (tracked, not on the cheap-first path).
- The topic's own action facts now render via grounded TEMPLATE (aggregated + connected), not the FT generator — the
  template is fluent + connected + guaranteed grounded (the FT single-fact fluency was giving isolated sentences with
  artifacts; the connected template is a strict readability + safety win). The FT remains for the single-turn `_answer`.
- **The genuine wall (unchanged, routed around not solved):** free single-pass abstractive synthesis + open-world
  cross-fact INFERENCE on a ~21M model (catalog G.09 constructive recombination = a new mechanism class; a documented
  free-analogy NO-GO). The honest posture at it: checkable-inference-only + grounded connectives + say-where-knowledge-
  ends.

## Where this sits
Closes the ~70% cheap residual of the multi-fact-synthesis frontier — "it lists facts" → connected, contrastive,
grouped grounded prose. Next (gated, lower priority): #2 richer checkable inference (gist/shared-property beyond
compare); #3 a plan-guided fine-tune for genuine single-pass fluency (moat + fallback as the safety net) — owner-steer,
since #1 already delivers the connected-prose flavor with zero risk.

**Artifacts:** `research/runners/_fluidconv_phase16_discourse_plan_derisk.py` (+ `plan_discourse`/`compare_discourse`,
imported by the console); result `research/findings/raw/_fluidconv_phase16_discourse_plan.json`; scoping
`2026-07-01-multi-fact-synthesis-frontier-scoping.md`.
