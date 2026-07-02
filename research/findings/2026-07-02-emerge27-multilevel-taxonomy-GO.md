# EMERGE-27 / toward-semantics — GO (6/6 seeds): MULTI-LEVEL taxonomic inheritance. A concept inherits properties from MULTIPLE levels of its is-a hierarchy at once, and a member-specific cancellation at one dimension does not block inheritance at another — the full Collins-Quillian hierarchical structure, emergent on the spiking HTM cortex, NO inference engine, NO `sim/` edit.

**2026-07-02 (autonomous).** Runner `research/runners/_emerge27_multilevel_taxonomy_derisk.py`; CI guard `tests/test_emerge27_multilevel_taxonomy.py` (4 tests). Reuse-by-import (`_emerge14` + `_emerge12`), extends EMERGE-26; NO `sim/` edit; CPU numpy-backend; 6-seed.

## The claim
Hierarchy ANIMAL > {BIRD, FISH} > {robin, penguin, trout}. Teach ONLY level facts — "an ANIMAL breathes" (top), "a BIRD flies" / "a FISH swims" (mid), and one specific "a penguin walks". Then, on the emergent spiking HTM cortex, 6/6 seeds:
- **Multi-level inheritance:** `robin → breathes` (from ANIMAL, **2 levels up**) **AND** `flies` (from BIRD, 1 level up); `trout → breathes + swims`. A concept inherits from EVERY level of its taxonomy at once (RESP-inherit **1.00**, LOCO-inherit on held-out members **1.00**).
- **Dimension isolation / cancellation:** `penguin → breathes + walks` — the specific "walks" CANCELS the inherited flies at the LOCOMOTION dimension, while the RESPIRATION inheritance "breathes" (from ANIMAL) SURVIVES untouched (dim-isolation **1.00**). This is the full Collins-Quillian structure: cancellation is per-property, not global.
- **Moat 1.00:** a concept with no ancestors abstains on both dimensions (no confabulation).

## The mechanism (EMERGE-26 extended to nested levels)
Each concept's code = its CONTENT block + ALL its ancestor SUPERORDINATE blocks (robin = content + BIRD + ANIMAL). A level fact is taught by potentiating that level's block → the property (`ANIMAL-cols → breathes`) via the committed `sim/` three-term kernel. Querying a concept presents its content + every ancestor block; each level's block primes its property, so the concept inherits from every level simultaneously. The read is PER DIMENSION (respiration {breathes} / locomotion {flies,swims,walks}) by argmax over that dimension's apical DRIVE, abstaining below the rest floor — so a stronger member-specific pathway (penguin's 3 content cols → walks, drive 3) cancels the inherited default (BIRD's 2 cols → flies, drive 2) WITHIN its dimension, while other dimensions inherit untouched.

## Anti-cheats (all airtight, 6/6)
- **DERANGED-ANCESTORS** (concepts share the WRONG mid-level, BIRD↔FISH swapped; top ANIMAL preserved): mid-level (locomotion) inheritance collapses to **0.00** — isolating the mid is-a code as the cause.
- **dAP-LESION** (coincidence off): collapses to **0.00**.
- **HELD-OUT**: the members' own properties are never taught (only the levels' + the one penguin-specific); the answers cannot come from memorized member→property pathways.
- **MOAT 1.00**; 6-seed unanimous.

## Significance
This is the full hierarchical Collins-Quillian semantic network on the emergent spiking brain: a concept answers questions about properties stored at ANY level of its taxonomy (2-hop-up respiration + 1-hop-up locomotion), and specific facts override inherited defaults per-property (penguin doesn't fly, but still breathes). It composes EMERGE-26's single-level inheritance with nested ancestor codes — reuse-by-import, NO `sim/` edit. Grounded in Collins-Quillian 1969 (hierarchical semantic memory) and the feature-prediction view (Rogers-McClelland; Saxe-McClelland-Ganguli 2019: taxonomic inheritance emerges from shared features × prediction).

## Honest scope + next
- The is-a hierarchy is host-DESIGNED (ancestor blocks hand-assigned). This is inference-OVER-structure, NOT acquisition-OF-structure-from-experience — the deferred R-c residual (the hierarchy must EMERGE from co-occurrence/perception statistics via the PPMI stream cortex + replay), the next deep-research gate.
- Named next build: transitive relational inference (recombining overlapping learned pairs, hippocampal — catalog D.02), and coupling the inheritance read-out into the EMERGE-25 conversational console (answer "does a robin breathe?").

## Artifacts
`research/runners/_emerge27_multilevel_taxonomy_derisk.py`, `tests/test_emerge27_multilevel_taxonomy.py`, `research/findings/raw/_emerge27_multilevel_taxonomy.json`. Prior: `2026-07-02-emerge26-emergent-inheritance-GO.md`, `2026-07-02-open-world-semantics-knowledge-acquisition-research-gate.md`.
