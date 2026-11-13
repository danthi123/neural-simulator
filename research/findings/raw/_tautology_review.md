# MiMo second opinion on the self-predicated-tautology change (2026-11-13)

Reviewed `research/findings/raw/_tautology_review.diff` (code + tests only). Verbatim reply (the code diff itself is
in that file; this is the review text, saved because Claude's merge review asked for it):

- **CONFIRMED DEFECT (line 27, `_self_predicated`)** (first framing, self-retracted in the next paragraph): the slice
  arithmetic `t[pos[0] + n:p]` — MiMo then restated this as correct and moved to a different claim, below.

- **CONFIRMED DEFECT (line 27, `_self_predicated`)**: the substring match `re.finditer(r"\b" + re.escape(key.lower()), t)`
  is not restricted to whole words, so a longer word starting with the key counts as a second occurrence:
  `is_tautology("Pixel is pixelated.", "pixel")` — "pixelated" at position 8 counts as an occurrence, slice " is "
  contains "is", so the clause is flagged, but "Pixel is pixelated" is not a tautology. Claims this contradicts
  "holds the key at least twice" and diverges from `has_key`'s word semantics.

- **CONFIRMED DEFECT (line 26)**: `t = clause.lower().replace("’", "'")` normalizes curly apostrophes in the clause but
  the key is used as `key.lower()`; a key with a typographic apostrophe ("o’connor") would never match. Dormant
  inconsistency, not behavior-preserving.

- **OK**: the between-occurrences span arithmetic (`pos[0]+n:p` is left-inclusive, correctly excludes the first
  occurrence's characters while retaining a possessive `'s`).
- **OK**: the `_clauses` refactor — the split regex is identical to the inlined one in `is_echo` and
  `wrong_perspective`, behavior-identical.
- **OK**: the two existing tautology rules preserved unchanged in logic.

## Disposition (Pi, same day)

1. Prefix claim: **kept as built, ruled on by Claude in the merge review** — "holds the key" uses the module's
   documented key-as-word-start semantics (`has_key`: "peanut matches peanuts"; rule 1's `\w*`). Claude ruled (verdict
   question 2): keep word-start semantics — the old rule 1 already flags "Pixel is pixelated." (its second key has no
   closing `\b`), so whole-word matching only in the new rule would make the two rules disagree, and it would break
   plural keys ("Pixels are called Pixels.").
2. Apostrophe claim: **fixed** — the key is normalized exactly like the clause (`62d83b752`). No recorded key contains
   an apostrophe (facts.py), so no score changes; the rescore rerun stayed byte-identical.
