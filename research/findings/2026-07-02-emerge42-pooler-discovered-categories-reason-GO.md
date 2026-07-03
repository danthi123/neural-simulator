# EMERGE-42 / toward-semantics — GO (6/6 seeds): the DISCOVERED categories REASON. The competitive self-organizing pooler (EMERGE-38..41) discovers OVERLAPPING categories from experience, and the FULL Collins-Quillian inference (class inheritance + member-specific-override cancellation) runs over the LEARNED codons on the spiking bridge. Composes the pooler arc with EMERGE-37 cancellation. NO `sim/` edit.

**2026-07-02 (autonomous).** Runner `research/runners/_emerge42_pooler_inference_derisk.py`; CI guard `tests/test_emerge42_pooler_inference.py` (3 tests). Reuse-by-import (`_emerge14` + `_emerge12`), composes EMERGE-38 (competitive pooler) + EMERGE-37 (cancellation on emergent codes); NO `sim/` edit; CPU numpy-backend; 6-seed.

## The claim (6/6 seeds)
On 6 OVERLAPPING categories (adjacent share 3/6 features), the competitive self-organizing pooler discovers the category structure from experience, and the full inference runs over the learned codons:
- **CANCELLATION 1.00** (all 6 seeds): the overridden member answers its SPECIFIC fact (OVR), not the inherited class default.
- **INHERITANCE 0.99 mean** (1.00/1.00/1.00/1.00/0.94/1.00): non-overridden members inherit their category's class property via the discovered overlapping-category columns.
- **Anti-cheats collapse:** PERMUTED-features 0.33 (the pooler can't discover categories), FIXED (no-learn) 0.41, dAP-LESION 0.00.

## Mechanism (the key insight: a member = category code + identity code)
The pure category pooler COLLAPSES same-category members onto (near-)identical codons — that is exactly what makes inheritance generalize, but it leaves no member-specific substrate for an override. The fix (biologically faithful): a member is represented by BOTH a shared **category code** (the pooler codon, for inheritance) AND a unique **member-identity ensemble** (for member-specific facts):
- the competitive HTM Spatial Pooler (EMERGE-38..41) learns a category codon per member (shared within a category);
- the **class** property is taught on the codons (category → inherited by all category-mates via the shared columns);
- the **override** property is taught on the overridden member's **identity ensemble** (unique to it → keyed to that member alone, no bleed);
- querying a member primes its category codon (→ inherited class default) + its identity ensemble (→ its override, if any); a graded-drive read takes the strongest. The override, being taught more strongly + on the member-unique cells, out-drives the inherited default for the overridden member; every other member has no override → inherits.

Two implementation facts pinned along the way: (1) teaching the override on the codon BLEEDS to the whole category (same-category codons are near-identical) — it must be on the member-unique cells; (2) a single identity cell cannot clear the coincidence-plateau threshold (drive 1.0 < 1.5) — the identity ensemble needs ≥2 cells (here 3) to plateau and out-drive.

## Anti-cheats (6/6)
- **PERMUTED-features** (input-destruction): inheritance collapses to 0.33 — no discoverable category structure → the pooler can't group members.
- **FIXED (no-learn)** projection: 0.41 — the untuned baseline can't cleanly separate overlapping categories.
- **dAP-LESION** (coincidence off): 0.00.
- Cancellation 1.00 + inheritance 0.99; 6-seed unanimous.

## Significance
This ties the whole competitive-pooler arc (EMERGE-38..41: discover overlapping categories, fully-on-substrate) to the inference arc (EMERGE-26/37: inheritance + cancellation): the brain **discovers** overlapping categories from experience AND does the **full Collins-Quillian inference** (inherit the class default; a specific fact cancels it per-member) over the self-discovered structure — with the member-specific representation (category code + identity ensemble) that makes both inheritance AND per-member override coexist. A materially richer, biology-faithful semantic substrate for grounded conversation, on one spiking brain, no transformer.

## Honest scope + next
- The pooler LEARNING is a rate-reference (realized fully-on-substrate at EMERGE-39/40; k-WTA spiking at EMERGE-41); the inheritance/cancellation run on the spiking bridge over the discovered codons. Single override on a 6-category setup; multi-override / multi-level-emergent cancellation / transitivity over discovered categories are follow-ons.
- Next: couple this into the interactive experiential console (EMERGE-31) so a user can teach overlapping categories by co-occurrence and query inheritance + override live; then transitivity over discovered categories.

## Artifacts
`research/runners/_emerge42_pooler_inference_derisk.py`, `tests/test_emerge42_pooler_inference.py`, `research/findings/raw/_emerge42_pooler_inference.json`. Prior: `2026-07-02-emerge41-fs-wta-kwinners-GO.md`, `2026-07-02-emerge38-competitive-self-organizing-pooler-GO.md`, `2026-07-02-emerge37-cancellation-emergent-codes-GO.md`.
