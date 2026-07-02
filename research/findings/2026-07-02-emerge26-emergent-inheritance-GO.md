# EMERGE-26 / toward-semantics — GO (6/6 seeds): the FIRST inference-BEYOND-told-facts on the emergent brain. Collins-Quillian property INHERITANCE (with cancellation) EMERGES on the spiking HTM cortex with NO explicit inference engine — a never-taught member inherits its class property from a shared superordinate code × the next-state predictor. NO `sim/` edit.

**2026-07-02 (autonomous; the open-world-semantics research gate's recommended de-risk).** Runner `research/runners/_emerge26_emergent_inheritance_derisk.py`; CI guard `tests/test_emerge26_emergent_inheritance.py` (4 tests). Reuse-by-import (`_emerge14` on-bridge learner + `_emerge12` priming); NO `sim/` edit; CPU numpy-backend; 6-seed (42/43/44/100/101/102).

## The claim
Teach ONLY class-level facts — "a BIRD flies", "a FISH swims" — and one member-specific fact ("a penguin walks"). Then, all on the emergent spiking HTM cortex, 6/6 seeds:
- **INHERITANCE beyond told facts (1.00):** never-taught members answer their class property — `robin/sparrow/canary → flies`, `trout/salmon → swims` — although each member's own property was NEVER taught, only the class's. This is inference beyond told facts.
- **CANCELLATION (1.00):** `penguin → walks` — the member-specific fact beats the inherited class default (flies). The discriminating Collins-Quillian cancellation (a more-specific stored property wins).
- **MOAT (1.00):** `novel → ABSTAIN` — a concept with no superordinate drives no class pathway, so it abstains, never confabulating a property.

## The mechanism — inference EMERGES, no inference engine (the research-gate reframe)
The open-world-semantics research gate (`2026-07-02-open-world-semantics-knowledge-acquisition-research-gate.md`) reframed "semantic inference": relational/semantic inference **emerges** from *overlapping/shared codes × a next-state predictor*, with NO explicit inference engine — and the substrate already has both. Each concept = a three-block sparse code: a **content** block (the specific concept, 3 cols) + a shared **superordinate** (is-a) block (robin/sparrow/canary/penguin all share BIRD's 2 cols; trout/salmon share FISH). The class fact is taught by potentiating the SUPERORDINATE block → the property (`BIRD-cols → flies`) via the committed `sim/` three-term kernel. Querying a member presents its content+superordinate cells; the shared BIRD cells prime "flies" through the learned class pathway → the member inherits, though its own content was never bound to any property.

**Cancellation via graded drive:** the member-specific pathway (penguin's 3 content cols → walks) out-DRIVES the inherited class pathway (BIRD's 2 super cols → flies) — measured directly: the inherited property charges the apical to ~+5 mV (drive 2), the specific to ~+20 mV (drive 3). A graded-magnitude read (argmax over each property's mean apical drive, abstain below a rest floor) picks the specific over the inherited — Collins-Quillian's "most specific stored property wins", emergent from the drive asymmetry, not a hand-coded rule.

## Anti-cheats (all airtight, 6/6)
- **DERANGED-SUPERORDINATE** (members share the WRONG is-a block, BIRD↔FISH swapped): inheritance collapses to **0.00** — isolating the shared is-a code as the cause of inheritance (not chance, not a per-member pathway).
- **dAP-LESION** (coincidence off): collapses to **0.00** — the inference is genuinely the bridge's dendritic-plateau recurrence + the `sim/` kernel.
- **HELD-OUT**: the members' properties are NEVER taught (only the class + the one penguin-specific fact) — so the correct answer cannot come from a memorized member→property pathway.
- **MOAT 1.00** (confabulation 0); 6-seed unanimous.

## Significance — the substrate INFERS over relational structure
This is the first **inference-beyond-told-facts** on the emergent spiking brain: the cortex answers questions whose answers were never stored, by inheriting over an is-a hierarchy — the hallmark of semantic cognition (Collins-Quillian 1969; Rogers-McClelland / Lambon Ralph 2017; Saxe-McClelland-Ganguli 2019: taxonomic inheritance emerges from feature-prediction). It reuses exactly the machinery already validated (EMERGE-17 shared "family"/is-a micro-columns + the HTM next-state predictor); the only change is repurposing the shared block as an is-a superordinate and reading the graded drive. NO `sim/` edit.

## Honest scope + next
- **The substrate INFERS over relational structure that is host-DESIGNED** (the shared is-a codes are hand-assigned). A GO proves inference-OVER-structure, NOT acquisition-OF-structure-from-experience. That deferred residual (R-c in the research gate) — the is-a codes must EMERGE from co-occurrence/perception statistics — is the genuinely-irreducible next research direction, surpassable via the project's existing PPMI stream cortex + replay (the gate's paths d/e). It should be gated only now that the substrate's ability to infer is proven.
- Named next inference steps (build, compose GO pieces): **transitive / multi-hop taxonomy** (robin → BIRD → ANIMAL → breathes, inheritance up a chain) and **transitive relational inference** (recombining overlapping learned pairs, hippocampal-style — catalog D.02).

## Artifacts
`research/runners/_emerge26_emergent_inheritance_derisk.py`, `tests/test_emerge26_emergent_inheritance.py`, `research/findings/raw/_emerge26_emergent_inheritance.json`. Prior: `2026-07-02-open-world-semantics-knowledge-acquisition-research-gate.md`, `2026-07-02-emerge25` (console), `2026-07-02-emerge24-online-growth-GO.md`.
