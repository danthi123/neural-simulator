# emo-bounded-default-on: merge note (declared prereg deviation of the flip commit)

Branch `research/emo-bounded-default-on`. Flip commit `e4fa02263e385f95160944cb73a0c072d3275206` (2026-10-28 08:28 -0400),
plus the skeptic-fix commit that adds this note. Prereg: `research/findings/2026-10-27-emo-bounded-flip-PREREGISTRATION.md`.
The merger quotes the "Deviation" section below in the merge commit message, so the deviation lands on main in the open.

## Deviation from prereg section 1

Section 1 says the flip commit "is made only after section 2 passes" and "touches that one function and its tests"
(prereg lines 29-31). The flip commit did neither:

1. **Order.** It was committed before DEC passed. At that moment the only registered read was the first gate read at revision
   F `3b0b5182aa1819d1804ef21d704154132d75e5bc` (B1-B7 PASS, B8 UNDEFINED, DEC UNDEFINED, so STOP), and Amendment 2
   (`0836aaa9c`, branch `research/emo-flip-amend2`, 08:34 -0400) had not been committed yet. The rescore under Amendment 2 has
   not been run as of this note.
2. **Scope.** Beyond `bounded_requested()` in `research/runners/emotion_category_organ.py` and its tests
   (`tests/test_emotion_category_bounded.py`, `tests/test_emotion_census_device.py`), it also touched
   `docs/PRODUCTION_INTEGRATION_LEDGER.yaml` (the `_BOUNDED_DEFAULT_ON` default_anchor that gate CLASS PI requires, and the
   status text), `research/coordination/board_sync.json` (the receipt gate CLASS BS requires for a ledger edit), two runner
   docstrings (`research/runners/emo_flip_lines.py`, `research/runners/lbf_rows/emotion_category.py`) and
   `research/FAILURE_LOG.md` (two lines). The fix commit makes `bounded_requested()` read the env name as a literal (so gate CLASS OS
   resolves the default), adds one test, this note, the reworded ledger text, a board receipt and FAILURE_LOG lines.

**What it did not change.** No production default changed: at the time of this note the flip commit is an ancestor of no main
(checked 2026-10-28: origin/main and gitea/main `631e8fa29`, server main `8ce8487b8` over ssh). No bar, threshold, gate line,
scorer or registered reading changed. The extra files are docstrings, the ledger and board receipt the gates require for a
default anchor, and the failure log.

**Why it is recorded here and not in Amendment 2.** Amendment 2 sits on another unmerged branch (`research/emo-flip-amend2`),
and a registered amendment's body is checked by gate CLASS PRA (rule N3); editing it from this branch would mix two branches' records.

## Merge rule (unchanged from the prereg)

Merge only if the Amendment 2 rescore's DEC reads PASS; on any other DEC close the branch unmerged. Before merging, replace the
single `TODO-MERGER(emo-bounded-default-on)` sentence in the ledger's `subflag_flip_2026_10_28` with the rescored DEC line (its
verdict.json path and the verbatim DEC status). The rest of the row's status text was written to stay true before and after the
merge, so nothing else in the row needs an edit.
