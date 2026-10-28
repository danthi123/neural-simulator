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

## The DEC (2026-10-28): PASS, so the branch is ready to merge

**DEC PASS at F 3b0b5182a under Amendment 2 (post-data, B8 admission rule only), scored by Claude 2026-10-28 on S main
6dfa01e1e: B1-B8 PASS, B8 ON load-bearing 6/6, OFF 5/6; verify-go panel (3 lenses) survived.** Verbatim scorer line:
`DEC PASS -> flip the default ON before the 11-21 re-pin`.

- **Committed artifact.** `6dfa01e1e` is on neither origin nor gitea, and the only committed score file at origin/main
  `04680400c` was the registered 23:59 read (B8 UNDEFINED, DEC UNDEFINED). The rescore was therefore re-run on this branch,
  as a module, on the clean tree `2123e2c68`, with the registered read's arguments:
  `research/findings/raw/_emo_bounded_flip/gate_score_amend2_3b0b5182aa1819d1804ef21d704154132d75e5bc.json` and its
  `.prov.json` (run_id 1793195766-2231076, git_dirty false). Same verdict: bars DEFINED 8 of 8, DEC PASS.
- **The ledger.** The `TODO-MERGER` sentence in `subflag_flip_2026_10_28` is replaced by the DEC line above, the artifact path
  and the verbatim scorer line; no `TODO-MERGER` marker is left in the ledger.
- **Panel verdicts (verify-go, 3 lenses; its findings grouped by topic).**
  - Reproduction: REPRODUCED. The scorer gives the same verdict in a fresh detached worktree of origin/main `04680400c` (also
     gitea/main), CPU numpy, rc 0; the no-pin `--only B8` read gives the same 6/6 and 5/6; the 37 tests then present passed.
  - Hand recompute and luck: all 12 B8 cells recomputed from the arm files agree with their lb.json verdicts (complete, null
     clean, lesion reproduced, non-feel turns unchanged). No ON seed is a near miss: intact joy gate counts 46 to 52 against a
     threshold of 18, the read invariant at x0.5 and x2, and the lesion drives the gate count to 0. Each seed's load-bearing
     result rests on one probe turn (emocat_feel), the row's thin-probe design; the arms are deterministic. OFF s101
     not load-bearing matches the 10-26 result (OFF s101 names joy None on three hosts).
  - Instrument and provenance: every sidecar is git_sha F, git_archive, manifest verified at start and exit, numpy, not dirty;
     the ON sidecars carry the treatment key and the OFF ones do not; seven mutated copies each read UNDEFINED with a named
     reason; the row data was only ever added, each cell one pool job, all before the registered read.
- **Required fixes from the panel, made on this branch.** (1) B8 checked engagement on the intact arms only; it now reads
  every arm, and a lesion arm must show commit_lesioned True and its tag's flag state (`2123e2c68`). (2) Under a treatment the
  pin rule read only the recorded process env; it now also reads each arm's argv `--env` (`2123e2c68`). (3)
  `docs/drafts/emo-bounded-cond/handler_queue.sh`'s OFF arm left the flag unset; it now sets `=0` (`2123e2c68`). (4) This
  artifact, (5) the ledger line, and (6) the addendum to `research/coordination/emo-flip-gate-RESULT.md`. Neither reader
  fix changed a cell: the 24 lesion files read commit_lesioned True and the flag state of their tag; the intact arms' argv
  `--env` is `{}` and the lesion arms' is exactly `{"BRAIN_EMOTION_CATEGORY_COMMIT_LESION": "1"}`.
- **Non-blocking notes from the panel (none changes the verdict).** Amendment 2 is post-data (under the old rule the ON
  cells were UNDEFINED by construction; the cells' LB status is the same with and without `--pin`). B8 takes its verdict from
  the lb.json row fields rather than recomputing it from the arm files; the hand recompute agrees on all 12 cells.
  `tools/where_ran.py` collects only `*.json`, so it misses the 12 extension-less `.rep0` artifacts (their sidecars were read by
  hand: same host as their cell). A worker part file (`gate_on/s44/emo__permuted.json.part.json` and its sidecar) was committed
  in `69253e318`; its answers equal the final shard's and no scorer reads it.
- **Bar-sanity records.** The scorer moved, so the B4 and B6 records (which hash it) were re-derived with the same specs, n 1000
  and seed 0, as Amendment 2 did; see `docs/drafts/emo-bounded-flip/bar_sanity_record_b4.json` and `_b6.json`.
