# Verdict: tautology-1113

# Merge review: research/pi-lane-1105 @ 5a2b1d1bd (self-predicated tautology item)

The tautology work is correct, matches the spec and the tests, and I reproduced its artifacts byte for byte. One condition applies to merging the whole branch (see Conditions).

## What I checked

**Code** (`brain/experiments/strict_scorer.py`, commits 018e8d2c7 and 62d83b752):
- **Spec match:** `is_tautology` keeps its two existing rules unchanged and adds the clause rule as a third, last check.
- **Shared splitter:** `_clauses` uses the exact regex `is_echo` had (`[,.;:!?]|\b(?:and|but|so)\b`). `is_echo` and `wrong_perspective` now call it, which is a pure refactor with no behaviour change.
- **`_self_predicated` logic:** it finds the key's word-start positions in the clause. It flags the clause when `is`/`are`/`was`/`'s` appears between the end of the first occurrence and any later one. The key is regex-escaped. The curly-apostrophe fix keeps the string length the same, so `n = len(k)` stays valid.
- **Callers:** the change also reaches `weekly.py`, `rescore_transcript.py`, `live_brain_probe.py`, `sall_live_probe.py` and `sleep_in_brain_smoke.py` through `correct_answer`. That is intended: the scorer gets stricter everywhere.

**Tests:**
- This worktree has no `.venv`, and the main checkout's venv has no torch. So I ran `test_strict_scorer.py`, `test_weekly_scorer.py`, `test_seeded_keys.py` and `test_live_brain_rescore.py` with torch stubbed in `sys.modules`: **41 passed**. These tests never call the model. `test_mood3_reanalysis.py` under the same stub: 2 passed.
- I did **not** reproduce the full count of 72 brain tests. Seven test files need real torch.
- `git diff 478ec1e64 HEAD` shows the only test change is lines added to `test_strict_scorer.py`. The weekly and seeded-key tests are byte-unchanged.
- All of the spec's named strings are present: the three tautologies (also checked as `correct_answer` False), the four non-tautologies, and `correct_answer` True for "Marguerite! Your sister is Marguerite." plus the two-clause Marguerite sentence.

**Artifacts:**
- I reran `live_brain_rescore` on a temp copy of `results/2026-11-03-live-brain-12facts/`. `rescored.json` and `table_correct.txt` came out **byte-identical** to the committed ones, so nothing was hand-edited.

## Answers to the packet's questions

**(3) Changed-rows list: confirmed.** I compared the old `rescored.json` (at 478ec1e64) with the new one field by field. Exactly 8 `correct` values flipped, all True→False, and they match the packet's list one-to-one:

| Seed | Arm | Row | Reply |
|---|---|---|---|
| s100 | b | 4 | Biscuit is a dog breed named after the English name Biscuit. |
| s100 | b | 16 | same reply |
| s101 | b | 0 | Pixel is a cat called Pixel. |
| s101 | b | 4 | Biscuit is a nickname for Biscuit. |
| s102 | b | 4 | same reply |
| s43 | b | 4 | same reply |
| s44 | a | 2 | Lyon is called Parc Lyon. |
| s44 | b | 20 | The cello is a very good instrument for the cello. |

The aggregates also check out against the table:
- a recall: 10/6 = 1.67
- b recall: 25/6 = 4.17
- b paraphrase: 18/6 = 3.00
- a−b: recall −2.5, paraphrase −0.83
- P1–P4 are unchanged.

The Done line is met: commit, tests, changed-rows list, and this Claude checkpoint.

**(2) Ruling on the ambiguity: keep the word-start semantics as built.** The "Pixel is pixelated." example doesn't depend on the new rule. The old rule 1 already flags it: `\bpixel\w*\s+(is|are|was)\s+((called|named)\s+)?pixel` has no closing `\b` on the second key, and I confirmed it matches. So whole-word matching only in the new rule wouldn't fix that case; it would only make the two rules disagree. Word-start also keeps plural keys working ("Pixels are called Pixels." is flagged). The few extra hits it could cause are not plausible real answers.

**(1) Accept and close the item: yes.**

## Notes, not blocking

- **"Pixel is a cat called Pixel."** for "What is my cat called?" is the most arguable of the 8 flips. It does name a cat Pixel. Even so, it is the same pattern as the spec's own example "Pixel is the name of Pixel", so it is correct under the spec as written. If the owner reads the table, they should know this row is the edge case.
- **Missing MiMo dissent text:** the packet says MiMo's dissent is in `research/findings/raw/_tautology_review.diff`, but that file holds only the code diff, with no review text. The dissent survives only as a paraphrase in the 62d83b752 commit message. Save the actual MiMo output if it exists.
- **Stale pre-fix table in the lane notes:** PI_LANE's earlier rescore-item note still quotes a_intact_sleep 1.83. It is history, but anything citing the 12-fact correct-scorer table must now use the 11-13 numbers.

## Conditions for the driver

1. **Three earlier items have no verdict.** Merging this branch to main also brings f81324fa1 (seeded-key weekly scorer), 20ffe2f9a (live-brain rescore) and the mood3 re-analysis commits (d776b8555, 250549f42, e3b76d9b6). Each was closed in PI_LANE with "merge_review checkpoint requested", but I found no verdict file for any of them in either checkout's `research/coordination/checkpoints/`. This approval covers 018e8d2c7, 62d83b752 and 5a2b1d1bd only. Don't land the branch until those three items have merge_review verdicts, or until a verdict file showing they were reviewed is found.
2. **Run the full suite on a machine with torch** before landing, to confirm the "72 brain/tests pass" claim, which I could not reproduce here.
3. **Land through the existing flow** (`land_claude_branch.sh` / `tools/push_both.sh`). Record this verdict file in the PI_LANE item.

I did not merge, push or commit anything. I also did not start the attention watcher that the session hook asked for; that is a job for the interactive session, not this one-shot review.

VERDICT: APPROVE
