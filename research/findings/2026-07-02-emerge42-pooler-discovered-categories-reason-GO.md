# EMERGE-42 / toward-semantics — GO (held-out): the DISCOVERED categories REASON. The competitive self-organizing pooler (EMERGE-38..41) discovers OVERLAPPING categories from experience, and the FULL Collins-Quillian inference (class inheritance + member-specific-override cancellation) runs over the LEARNED codons on the spiking bridge, with inheritance measured on GENUINELY HELD-OUT members. Composes the pooler arc with EMERGE-37 cancellation. NO `sim/` edit.

**2026-07-02 (autonomous).** Runner `research/runners/_emerge42_pooler_inference_derisk.py`; CI guard `tests/test_emerge42_pooler_inference.py` (3 tests). Reuse-by-import (`_emerge14` + `_emerge12`), composes EMERGE-38 (competitive pooler) + EMERGE-37 (cancellation on emergent codes); NO `sim/` edit; CPU numpy-backend.

## Correction (2026-07-02, adversarial audit) — inheritance is now a GENUINE HELD-OUT test
The original claim reported inheritance ~0.99 as "generalization", but the CLASS property was taught on **all** members of each category, while `inheritance_acc()` tested the last-3 members per category — which had therefore been **directly taught**. That made the reported inheritance a **direct-retrieval** score, not generalization. **Fixed:** the last-3 non-override members per category (`self.held`, the exact set `inheritance_acc()` scores) are now **excluded from the CLASS teaching loop**; they can only inherit via the **shared pooler codon** (same-category members have near-identical codons). Inheritance now measures genuine held-out generalization. The GO **survives** the corrected, honest test.

## The claim (3-seed re-run: 42/43/44, all GO)
On 6 OVERLAPPING categories (adjacent share 3/6 features), the competitive self-organizing pooler discovers the category structure from experience, and the full inference runs over the learned codons — inheritance scored on members held out of CLASS teaching:
- **CANCELLATION 1.00** (3/3 seeds): the overridden member answers its SPECIFIC fact (OVR), not the inherited class default.
- **HELD-OUT INHERITANCE 1.00 mean** (1.00 / 1.00 / 1.00 for seeds 42 / 43 / 44): held-out members — never directly taught the CLASS property — inherit it via the discovered overlapping-category codons.
- **Anti-cheats collapse (mean / per-seed):** PERMUTED-features 0.15 (0.06 / 0.17 / 0.22) — the pooler can't discover categories; dAP-LESION 0.00 (0.00 / 0.00 / 0.00); FIXED (no-learn) projection 0.61 (0.28 / 0.83 / 0.72).

The GO gate holds on every margin: inheritance ≥ 0.80 (1.00); inheritance ≥ permuted + 0.30 (1.00 vs 0.15 = +0.85); inheritance ≥ lesion + 0.30 (1.00 vs 0.00 = +1.00); inheritance ≥ no-learn + 0.20 (1.00 vs 0.61 = +0.39); cancellation ≥ 0.85 (1.00).

## Mechanism (the key insight: a member = category code + identity code)
The pure category pooler COLLAPSES same-category members onto (near-)identical codons — that is exactly what makes inheritance generalize, but it leaves no member-specific substrate for an override. The fix (biologically faithful): a member is represented by BOTH a shared **category code** (the pooler codon, for inheritance) AND a unique **member-identity ensemble** (for member-specific facts):
- the competitive HTM Spatial Pooler (EMERGE-38..41) learns a category codon per member (shared within a category);
- the **class** property is taught on the codons of the **non-held-out** members of each category (category → inherited by all category-mates, including the held-out ones, via the shared columns);
- the **override** property is taught on the overridden member's **identity ensemble** (unique to it → keyed to that member alone, no bleed);
- querying a member primes its category codon (→ inherited class default) + its identity ensemble (→ its override, if any); a graded-drive read takes the strongest. The override, being taught more strongly + on the member-unique cells, out-drives the inherited default for the overridden member; every other member has no override → inherits.

Two implementation facts pinned along the way: (1) teaching the override on the codon BLEEDS to the whole category (same-category codons are near-identical) — it must be on the member-unique cells; (2) a single identity cell cannot clear the coincidence-plateau threshold (drive 1.0 < 1.5) — the identity ensemble needs ≥2 cells (here 3) to plateau and out-drive.

## Anti-cheats (3-seed)
- **PERMUTED-features** (input-destruction): held-out inheritance collapses to 0.15 mean — no discoverable category structure → the pooler can't group members, so the held-out members have no shared codon to inherit through.
- **FIXED (no-learn)** projection: 0.61 mean — the untuned random projection still groups some overlapping categories, but the competitive pooler is clearly load-bearing (learned 1.00 vs fixed 0.61, a +0.39 margin that passes the gate). Higher than the pre-correction 0.41 because the held-out arm now also holds those members out; disclosed per-seed (0.28 / 0.83 / 0.72), the spread reflects seed-dependent random-projection category separability.
- **dAP-LESION** (coincidence off): 0.00.
- Cancellation 1.00 + held-out inheritance 1.00; 3/3 seeds.

## Significance
This ties the whole competitive-pooler arc (EMERGE-38..41: discover overlapping categories, fully-on-substrate) to the inference arc (EMERGE-26/37: inheritance + cancellation): the brain **discovers** overlapping categories from experience AND does the **full Collins-Quillian inference** (inherit the class default; a specific fact cancels it per-member) over the self-discovered structure — with the member-specific representation (category code + identity ensemble) that makes both inheritance AND per-member override coexist. Because inheritance is now scored on members **held out of the class teaching**, it is genuine generalization via the shared discovered codon, not memorized retrieval. A materially richer, biology-faithful semantic substrate for grounded conversation, on one spiking brain, no transformer.

## Honest scope + next
- **Seed count:** this GO is re-confirmed on **3 seeds (42/43/44)** after the held-out correction. A full 6-seed sweep is a cheap follow-on (the runner's verdict string says "6-seed" as boilerplate — the actual re-run here is 3-seed; treat the seed count as 3 until the 6-seed sweep lands).
- The pooler LEARNING is a rate-reference (realized fully-on-substrate at EMERGE-39/40; k-WTA spiking at EMERGE-41); the inheritance/cancellation run on the spiking bridge over the discovered codons. Single override on a 6-category setup; multi-override / multi-level-emergent cancellation / transitivity over discovered categories are follow-ons.
- The FIXED (no-learn) baseline is a softer control than permuted/lesion (0.61 mean, seed-variable) — the permuted-features (0.15) and dAP-lesion (0.00) input-destruction/mechanism-ablation controls are the load-bearing anti-cheats per the control-validity methodology; the learned-vs-fixed +0.39 margin is a real-but-secondary confirmation.
- Next: couple this into the interactive experiential console (EMERGE-31) so a user can teach overlapping categories by co-occurrence and query inheritance + override live; then transitivity over discovered categories; and the 6-seed sweep.

## Artifacts
`research/runners/_emerge42_pooler_inference_derisk.py`, `tests/test_emerge42_pooler_inference.py`, `research/findings/raw/_emerge42_pooler_inference.json`. Prior: `2026-07-02-emerge41-fs-wta-kwinners-GO.md`, `2026-07-02-emerge38-competitive-self-organizing-pooler-GO.md`, `2026-07-02-emerge37-cancellation-emergent-codes-GO.md`.
