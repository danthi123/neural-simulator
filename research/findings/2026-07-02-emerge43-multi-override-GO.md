# EMERGE-43 / toward-semantics — GO (6/6 seeds): MULTI-OVERRIDE scales. Many member-specific exceptions (one per category) coexist with class inheritance over the pooler-discovered overlapping categories — each overridden member answers its OWN exception (no cross-bleed), non-overridden members inherit. The category-code + member-identity-ensemble representation (EMERGE-42) handles the realistic many-exceptions case. NO `sim/` edit.

**2026-07-02 (autonomous).** Runner `research/runners/_emerge43_multi_override_derisk.py`; CI guard `tests/test_emerge43_multi_override.py` (3 tests). Reuse-by-import, composes EMERGE-38 (competitive pooler) + EMERGE-42 (member-identity cancellation); NO `sim/` edit; CPU numpy-backend; 6-seed.

## The claim (6/6 seeds)
6 overridden members (one per discovered category), each carrying a DISTINCT exception fact, over 6 overlapping categories:
- **override-acc 0.97 mean** (1.00×5 + 0.83): each overridden member answers ITS OWN exception (`0_0→OVR:0_0`, …, `5_0→OVR:5_0`) — no exception bleeds to another member or category.
- **inheritance 0.99 mean** (1.00×5 + 0.94): the many overrides do NOT disrupt inheritance — non-overridden members still inherit their category's class property.
- **Anti-cheats collapse:** PERMUTED-features 0.33 (no discoverable categories), dAP-LESION 0.00.

## Mechanism
Each member = a shared **category code** (the pooler-discovered codon) + a unique **member-identity ensemble** (EMERGE-42). Class properties are taught on the codons (inherited via shared columns); each overridden member's exception is taught on ITS OWN identity ensemble (member-unique → keyed to that member alone). Querying a member primes its codon (→ inherited class default) + its identity ensemble (→ its own exception if any); a graded-drive read takes the strongest. Because each exception lives on a member-unique ensemble, N exceptions coexist without cross-talk — the identity ensemble is the per-member key, and the shared codon still carries the class default for everyone.

## Significance
Real knowledge has many exceptions (many entities each with their own facts). EMERGE-42 showed one override coexisting with inheritance; EMERGE-43 shows the member-identity mechanism SCALES to many coexisting exceptions, cleanly, on the pooler-discovered categories — the realistic case for grounded conversation about many entities.

## Honest scope + next
- Composes EMERGE-38 pooler + EMERGE-42 member-identity cancellation; one exception per category (6 total). Multi-level-emergent cancellation, transitivity over discovered categories, and coupling into the interactive experiential console (EMERGE-31) remain the follow-ons.
- The pooler LEARNING is a rate-reference (fully-on-substrate at EMERGE-39/40; k-WTA spiking at EMERGE-41); inheritance/cancellation run on the spiking bridge over the discovered codons.

## Artifacts
`research/runners/_emerge43_multi_override_derisk.py`, `tests/test_emerge43_multi_override.py`, `research/findings/raw/_emerge43_multi_override.json`. Prior: `2026-07-02-emerge42-pooler-discovered-categories-reason-GO.md`.
