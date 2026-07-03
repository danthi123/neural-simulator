# EMERGE-43 / toward-semantics — GO: MULTI-OVERRIDE scales, with GENUINE HELD-OUT inheritance. Many member-specific exceptions (one per category) coexist with class inheritance over the pooler-discovered overlapping categories — each overridden member answers its OWN exception (no cross-bleed), and HELD-OUT members that were NEVER taught the class property still inherit it via the shared pooler codon (genuine generalization, not direct retrieval). The category-code + member-identity-ensemble representation (EMERGE-42) handles the realistic many-exceptions case. NO `sim/` edit.

**2026-07-02 (autonomous; corrected 2026-07-02 per adversarial audit — see "Correction" below).** Runner `research/runners/_emerge43_multi_override_derisk.py`; CI guard `tests/test_emerge43_multi_override.py` (3 tests). Reuse-by-import, composes EMERGE-38 (competitive pooler) + EMERGE-42 (member-identity cancellation); NO `sim/` edit; CPU numpy-backend.

## Correction (adversarial audit, 2026-07-02) — closed the same hold-out hole as EMERGE-42
The original CLASS teaching loop taught the class property on ALL 9 members/category, then `inheritance_acc()` tested a subset (the last-3 non-override members) of those TAUGHT members — so "inheritance" measured DIRECT RETRIEVAL of a fact the member had been taught, not generalization. **Fix:** the held-out set (the last-3 non-override members per category, matching `inheritance_acc`'s selection) is computed in `__init__` and EXCLUDED from the CLASS teaching loop. Held-out members carry NO class fact of their own; they can only answer correctly by inheriting via the shared pooler codon. The multi-override teaching (each override on its own member-identity ensemble) is unchanged. The GO SURVIVES the corrected, honest test (below).

## The claim (3-seed 42/43/44, corrected honest metric)
6 overridden members (one per discovered category), each carrying a DISTINCT exception fact, over 6 overlapping categories; inheritance now on **held-out, never-taught members**:
- **override-acc 1.00 mean** (1.00 / 1.00 / 1.00): each overridden member answers ITS OWN exception (`0_0→OVR:0_0`, …, `5_0→OVR:5_0`) — no exception bleeds to another member or category.
- **held-out inheritance 1.00 mean** (1.00 / 1.00 / 1.00): members never taught the class property inherit it via the shared codon — genuine generalization — and the many coexisting overrides do NOT disrupt it.
- **Anti-cheats collapse:** PERMUTED-features 0.15 mean (0.06 / 0.17 / 0.22 — no discoverable categories → no shared codon to inherit through), dAP-LESION 0.00 mean.
- GO gate (override ≥ 0.85, held-out inh ≥ 0.80, inh ≥ permuted+0.30, inh ≥ lesion+0.30) cleared with margin on all three seeds.

Note: the held-out numbers are actually CLEANER (1.00 flat) than the pre-correction figures (0.97 / 0.99 with a 0.83 and a 0.94 seed) — the pre-fix metric was a mix of direct-retrieval and inheritance; the corrected metric isolates pure held-out inheritance through the shared codon, which the mechanism does cleanly.

## Mechanism
Each member = a shared **category code** (the pooler-discovered codon) + a unique **member-identity ensemble** (EMERGE-42). Class properties are taught on the codons of the TAUGHT members (inherited via shared columns); the held-out members are taught NOTHING but still share the codon. Each overridden member's exception is taught on ITS OWN identity ensemble (member-unique → keyed to that member alone). Querying a member primes its codon (→ inherited class default) + its identity ensemble (→ its own exception if any); a graded-drive read takes the strongest. Because each exception lives on a member-unique ensemble, N exceptions coexist without cross-talk — the identity ensemble is the per-member key, and the shared codon still carries the class default for everyone, including members never explicitly taught.

## Significance
Real knowledge has many exceptions (many entities each with their own facts). EMERGE-42 showed one override coexisting with inheritance; EMERGE-43 shows the member-identity mechanism SCALES to many coexisting exceptions, cleanly, on the pooler-discovered categories — the realistic case for grounded conversation about many entities. With the corrected metric, the inheritance side is a GENUINE generalization claim: held-out members never taught the class property inherit it purely through the shared pooler codon, and the many overrides do not corrupt that shared structure.

## Honest scope + next
- Composes EMERGE-38 pooler + EMERGE-42 member-identity cancellation; one exception per category (6 total). Multi-level-emergent cancellation, transitivity over discovered categories, and coupling into the interactive experiential console (EMERGE-31) remain the follow-ons.
- The pooler LEARNING is a rate-reference (fully-on-substrate at EMERGE-39/40; k-WTA spiking at EMERGE-41); inheritance/cancellation run on the spiking bridge over the discovered codons.

## Artifacts
`research/runners/_emerge43_multi_override_derisk.py`, `tests/test_emerge43_multi_override.py`, `research/findings/raw/_emerge43_multi_override.json`. Prior: `2026-07-02-emerge42-pooler-discovered-categories-reason-GO.md`.
