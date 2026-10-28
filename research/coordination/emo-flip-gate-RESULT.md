# emo-flip-gate RESULT (2026-10-27/28, gate revision F=`3b0b5182a`)

> **Superseded by the Amendment 2 rescore (2026-10-28), see the addendum at the end: B1-B8 PASS, bars DEFINED 8 of 8,
> DEC PASS, default flipped ON on `research/emo-bounded-default-on`.** The table below is the registered 23:59 read and stays
> as it was read.

Scored by `python -m research.runners.emo_flip_score --pin <F>` (Amendment 1's combined read);
artifact `research/findings/raw/_emo_bounded_flip/gate_score_3b0b5182aa1819d1804ef21d704154132d75e5bc.json`
(provenance stamped, run_id 1793159784-401057). Data: emo__ matrix 70/70 json (7 seeds x 5 arms x on/off,
incl dev s7), lb shards on+off s42-s44/s100-s102; 72/72 gate lines drained; all paired.

| bar | verdict | reading |
|---|---|---|
| B1 | PASS | n_correct 6 of 6 defined seeds |
| B2 | PASS | lesion_affect n_lesion_effect 6, n_reproduced 6 of 6 (attributable_to 100% each seed) |
| B3 | PASS | n_permuted_tracks 6 of 6 |
| B4 | PASS | 30 ON shards, 150 feel reads: 0 violations, 0 undefined |
| B5 | PASS | OFF intact named categories = 10-26 gate on 6 of 6 seeds, 0 mismatch |
| B6 | PASS | 30 pairs: 240 non-feel whole-record + 150 feel pre-clause compares, 0 diffs |
| B7 | PASS | flip_check 30 pairs (FEEL-NOT-APPEND-ONLY flag not read here: B6's), answer_rate paired OK 6 seeds |
| B8 | **UNDEFINED** | ON row: load-bearing 0 of 6 (need 6/6); OFF beside it 5 of 6. All 6 ON cells fail the pin: sidecars carry stray BRAIN_* env key `BRAIN_EMOTION_CATEGORY_BOUNDED` |

**bars DEFINED: 7 of 8. DEC = UNDEFINED -> STOP with the data. Default NOT flipped (stays OFF).** (The registered 23:59
read. Superseded: the Amendment 2 rescore below reads B8 PASS and DEC PASS.)

## Why B8 is undefined (deterministic, not transient)

`tools/lb_shard.py` `cell_prov_fails()` (line 271) checks `lb.json.prov.json` with
`allow_brain_keys=set()` - NO BRAIN_* env key may appear - and arm sidecars with
`PROV_ARM_ALLOWED_BRAIN_KEYS` (no BOUNDED). But the prereg's B8-ON row *is* the treatment: it must run with
`BRAIN_EMOTION_CATEGORY_BOUNDED=1` in env. Everything else in the pin passes on every cell (git_sha == F in
full, source_kind == git_archive, both source_manifest verifications true, SIM_BACKEND numpy) - the only
failure is the env-key allowlist. The DEC's remedy (re-run each UNDEFINED cell once at the same revision,
then rescore) is deterministic-futile: the same required env reappears, the same refusal returns, a
still-UNDEFINED cell never counts. Fixing the allowlist (e.g. admit BOUNDED for the ON tag's cells) is a
tools/scorer edit = Claude's (section 2); no re-run was spent. `--pin` reading and engagement checks were
never reached on the ON cells (UNDEFINED precedes them); the OFF row reads 5 of 6 load-bearing (one cell
not load-bearing is a data result, not a pin failure - see score json).

Posted as NEEDS `[emo-flip-gate]` RESULT. Prereg: `research/findings/2026-10-27-emo-bounded-flip-PREREGISTRATION.md`.

## Addendum (2026-10-28): the Amendment 2 rescore, DEC PASS

Prereg Amendment 2 (POST-DATA, `0836aaa9c`, merged to main at `04680400c`) changed only how B8's row cells are admitted under
`--pin`: the ON tag's sidecars must carry `BRAIN_EMOTION_CATEGORY_BOUNDED="1"`, the OFF tag's must not carry it, and a cell is
read only when complete. No cell was re-run and no bar changed.

| bar | verdict | reading |
|---|---|---|
| B1-B7 | PASS | as in the table above |
| B8 | **PASS** | ON row: load-bearing 6 of 6 seeds (need 6 of 6); OFF beside it 5 of 6 (s101 not load-bearing, as in the 10-26 result: OFF s101 names joy None) |

**bars DEFINED: 8 of 8. DEC PASS -> flip the default ON before the 11-21 re-pin.**

- First read by Claude on the server's main `6dfa01e1e` (that commit is on neither origin nor gitea). The verify-go panel
  (3 lenses) reproduced it on origin/main `04680400c` in a fresh worktree and hand-recomputed all 12 B8 cells from the arm files.
- Committed artifact: `research/findings/raw/_emo_bounded_flip/gate_score_amend2_3b0b5182aa1819d1804ef21d704154132d75e5bc.json`
  with its `.prov.json` (run_id 1793195766-2231076, git_sha `2123e2c68`, git_dirty false, host dant123-wk, numpy). It was run
  as a module with the same arguments as the registered read, on branch `research/emo-bounded-default-on` after the panel's
  two reader fixes: B8 now checks every arm's engagement (lesion arms included), and under a treatment the pin rule also
  reads each arm's argv `--env`. Neither fix changed a cell.
- The earlier artifact `gate_score_3b0b5182aa1819d1804ef21d704154132d75e5bc.json` stays as the registered 23:59 read
  (B8 UNDEFINED, DEC UNDEFINED).
- The flip: `_BOUNDED_DEFAULT_ON = True` in `research/runners/emotion_category_organ.py` (branch `research/emo-bounded-default-on`;
  ledger row `emotion-category`, `subflag_flip_2026_10_28`). It takes effect on main when that branch merges.
