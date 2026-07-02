# EMERGE-37 / toward-semantics — GO (6/6 seeds): CANCELLATION on EMERGENT codes. The full Collins-Quillian inference (inheritance + specific-override cancellation) works on codes LEARNED FROM EXPERIENCE (co-occurrence), not just hand-assigned ones — tying the inference arc (EMERGE-26) to the emergence arc (EMERGE-30). NO `sim/` edit.

**2026-07-02 (autonomous).** Runner `research/runners/_emerge37_cancellation_emergent_codes_derisk.py`; CI guard `tests/test_emerge37_cancellation_emergent_codes.py` (3 tests). Reuse-by-import (`_emerge14` + `_emerge12`), composes EMERGE-30 + EMERGE-26; NO `sim/` edit; CPU numpy-backend; 6-seed.

## The claim (6/6 seeds)
Members co-occur with context tokens (the committed `sim/` three-term kernel learns member-content → context on the bridge); the shared context cells are the EMERGENT superordinate (never labeled). A class property is taught on those emergent cells ("bird-context → flies"); a member-specific override is taught directly on one member's content ("robin → walks"). Then, on the spiking bridge:
- **Cancellation 1.00:** `robin → walks` — the member-specific fact CANCELS the inherited class default (flies), on the LEARNED code.
- **Inheritance 1.00:** `sparrow/canary → flies`, `trout/salmon → swims` — the non-overridden members inherit the class property via the LEARNED (from co-occurrence) grouping.

## Mechanism
Querying a member: the DIRECT pathway (member-content → property, 1-hop) competes with the INHERITED pathway (member-content → emergent-context → property, 2-hop); a graded-drive read takes the strongest. The member-specific direct fact out-drives the inherited default (a directly-taught, more-direct pathway) → the override wins for robin; for members with no override, only the inherited (2-hop via the learned grouping) is present → they inherit. This is EMERGE-26's Collins-Quillian cancellation, but on the emergent (learned-from-experience) codes of EMERGE-30.

## Anti-cheats (6/6)
- **PERMUTED-CONTEXT** (input-destruction: scrambled co-occurrence → no emergent grouping): inheritance collapses to **0.50 (chance)** on every seed — isolating the LEARNED grouping as the inheritance cause (while the content-direct override survives, as expected — it doesn't depend on the grouping).
- **dAP-LESION** (coincidence off): collapses to **0.00**.
- 6-seed unanimous.

## Significance
Cancellation is not tied to hand-assigned codes: the substrate does the FULL Collins-Quillian inference (inherit the class default; a specific fact cancels it per-member) over structure DISCOVERED from experience. This ties the inference triad (EMERGE-26/27/28) to the emergent-structure arc (EMERGE-30..36): the brain discovers categories from experience AND does full inheritance-with-cancellation over them, on one spiking brain, NO `sim/` edit.

## Honest scope + next
- The context tokens are the environment (legitimate); the grouping is discovered (the permuted-context control isolates it). Single-override on a 2-category setup; multi-override / multi-level-emergent cancellation are follow-ons.
- Next: the competitive self-organizing pooler; couple the full inference (inheritance + cancellation + transitivity) over emergent codes into the experiential console.

## Artifacts
`research/runners/_emerge37_cancellation_emergent_codes_derisk.py`, `tests/test_emerge37_cancellation_emergent_codes.py`, `research/findings/raw/_emerge37_cancellation_emergent_codes.json`. Prior: `2026-07-02-emerge30-emergent-superordinate-GO.md`, `2026-07-02-emerge26-emergent-inheritance-GO.md`.
