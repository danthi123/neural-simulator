---
type: preregistration
status: preregistration
date: 2026-10-27
packet: emo-bounded-flip
runner: research/runners/stage2_exit_gate.py (group emo), tools/lb_shard.py + research/runners/lbf_rows/emotion_category.py, tools/answer_rate.py
scorer: research/runners/stage2_exit_gate.py --aggregate, research/runners/emotion_category_dev.py --flip-check, tools/lb_shard.py aggregate, tools/answer_rate.py paired
---
# Emotion-category organ: bounded conductance becomes the production default (PREREGISTRATION, 2026-10-27)

flip: yes
host-decision: none the flag only selects the engine's own exponential-Euler conductance guard inside an existing organ; it adds no host rule, word list or table and decides no content (an integrator option)
bar-sanity: docs/drafts/emo-bounded-flip/bar_sanity_record_c4_emo.json
smoke-record: research/findings/smoke/emo-bounded-flip-s78.json faculty=emotion-category
smoke-record: research/findings/smoke/emo-bounded-flip-affect-coloring-s78.json faculty=affect-coloring

Drafted by the workstation driver on Claude's instruction (RULING `emo-bg-merge-1026` item (2), the C4 retake path), before any
gate-seed run. The bars in section 2 are the ruling's own words, asserted as substrings of the runbook entry when this file was
built; Claude diffs them before any gate-seed data. The flag, the organ change behind it and its de-risk are
`research/findings/2026-10-26-emotion-organ-bounded-conductance-derisk.md` (merged default OFF, main `1dbc75b96`). No gate seed
(42, 43, 44, 100, 101, 102) has been run for this flip; the only data below are dev seeds 7 and 8.

## 1 What flips

| flag | old default | new default | anchor |
|---|---|---|---|
| `BRAIN_EMOTION_CATEGORY_BOUNDED` | OFF | **ON** | `research/runners/emotion_category_organ.py` `bounded_requested()` (today `_truthy(BOUNDED_ENV)`, unset reads False) |

The flip commit does not exist yet and is not part of this file. It is made only after section 2 passes, touches that one
function and its tests, and reads an UNSET env as True while an explicit `0`/`false`/`no`/`off` stays the escape back to the
unbounded organ. The escape is pinned by a test in that commit, not claimed here. The organ is already default ON
(`BRAIN_EMOTION_CATEGORY`, `webapp/server.py` `_EMOTION_CATEGORY_DEFAULT_ON`); this prereg flips only how its conductance
step is integrated. Mechanism and the numbers behind it: the de-risk finding above (conductance-term overshoot of the
explicit-Euler step; held-out 60/60 ON vs 58/60 OFF; out-of-range neurons 0 ON vs 19/60 seeds OFF).

Expected drift, never a failure: every emotion read now carries five engagement keys, `bounded_flag`, `bounded_env`,
`bounded_readback`, `oor_count` and `oor_count_session`, even with the flag off. A whole-record compare of an emo or feel
turn across the merge differs by exactly those keys. Every comparison below drops them (at any depth) before it compares.

## 2 Bars, fixed before any gate-seed data

The ruling, verbatim (`research/coordination/LOCAL_LLM_RUNBOOK.md` section 6, `emo-bg-merge-1026`, item (2)):

> (2) FLIP PIPELINE for BRAIN_EMOTION_CATEGORY_BOUNDED (the C4 retake path), standard procedure (smoke-first, decision table). Driver runs the dev 7/8 smoke (answer_rate paired record), then drafts the flip prereg (flip: yes) with THESE bars verbatim; Claude diffs the bars before any gate-seed data. On the six gate seeds through the Stage-2 runner's emo group (intact_a, intact_b, permuted, lesion_affect + .rep0), flag ON: C4 correct 6/6 (all four categories named on every seed); lesion_affect changes the named category on 6/6 and reproduces; permuted tracks 6/6; every ON feel read has bounded_readback True and oor_count_session 0. Flag OFF arms on the same seeds must reproduce the 10-26 gate's emo result (s101 joy None) as instrument sanity. Non-feel turns identical ON vs OFF apart from the five keys; feel text before the category clause identical; answer_rate shows no FLIP-HURTS-ANSWERS. The emotion-category faculty's load-bearing row stays load-bearing with the flag ON. Pass -> flip the default ON before the 11-21 re-pin; anything else -> STOP with the data.

The same bars, split into clauses. Each clause is a word-for-word substring of the ruling; the split changes nothing.

- **B1.** "C4 correct 6/6 (all four categories named on every seed)"
- **B2.** "lesion_affect changes the named category on 6/6 and reproduces"
- **B3.** "permuted tracks 6/6"
- **B4.** "every ON feel read has bounded_readback True and oor_count_session 0"
- **B5.** "Flag OFF arms on the same seeds must reproduce the 10-26 gate's emo result (s101 joy None) as instrument sanity."
- **B6.** "Non-feel turns identical ON vs OFF apart from the five keys; feel text before the category clause identical"
- **B7.** "answer_rate shows no FLIP-HURTS-ANSWERS"
- **B8.** "The emotion-category faculty's load-bearing row stays load-bearing with the flag ON."
- **DEC.** "Pass -> flip the default ON before the 11-21 re-pin; anything else -> STOP with the data."

Scoring of each clause (how it is read; the bar text is above and is not changed by this):

- **B1, B2, B3** (flag ON): `python -m research.runners.stage2_exit_gate --aggregate <ON dir> --seeds 42,43,44,100,101,102`,
  read at the `emo` group only (the other four groups have no shards in these directories, so their missing-shard lines and the
  stage verdict are expected and not read). B1 is the group's `n_correct` 6/6, B2 its `n_lesion_effect` and `n_reproduced` 6/6,
  B3 its `n_permuted_tracks` 6/6.
- **B4** (flag ON): every `feel_*` response of every ON shard (all five arms) carries `emotion_category.bounded_flag` True,
  `bounded_env` "1", `bounded_readback` True and `oor_count_session` 0. A reader for this is owed (section 6).
- **B5** (flag OFF): the same `--aggregate` on the OFF directory reproduces the 10-26 gate's emo result, read from
  `research/findings/raw/_stage2_exit/gate/aggregate.json` (`emo` C4 FAIL, correct 5/6, seed 101 intact `joy` named None): the
  per-seed `intact` named categories of the OFF shards equal that file's on all six seeds. This is an instrument check, not a
  capability bar; a mismatch makes the OFF half UNDEFINED and the ON half is not read until it is explained.
- **B6** (ON vs OFF, same seed, group and arm): `python -m research.runners.emotion_category_dev --flip-check <ON dir> <OFF dir>
  --seeds ...` flags `FLIP-CHANGES-NONFEEL` (a non-feel turn's `answer`, `abstained` or `recalled_svo` differs) and
  `FEEL-NOT-APPEND-ONLY` (a feel turn's ON text is not the OFF text plus exactly one category clause). Both must be absent. The
  ruling's wording is wider than that scorer's three fields (the whole non-feel record, apart from the five keys); the whole-record
  compare is owed (section 6) and is read as well when it exists.
- **B7**: `--flip-check` (above) also reports `FLIP-HURTS-ANSWERS` over every turn of the Stage-2 shards, and `python
  tools/answer_rate.py paired --on-tag emo-bounded-flip-gate-on --off-tag emo-bounded-flip-gate-off --faculty emotion-category`
  (the six gate seeds) reads the row cells. A FLIP-HURTS-ANSWERS from either is reported and is not read as a pass.
- **B8**: `tools/lb_shard.py` runs the `emotion-category` row (`research/runners/lbf_rows/emotion_category.py`, committed
  `ec8fe543f`) at the six gate seeds with the flag ON, `--no-fixes --probe-set thin`, and the row must read reply-level
  load-bearing (`docs/TERMS.md` `load-bearing`: the lesion changes `answer`, the null is clean, the lesion reproduces) in every
  DEFINED cell, 6 of 6 seeds. The same row with the flag OFF runs on the same seeds and is reported beside it, so "stays" has a
  comparison. The row's lesion is the organ's own readout to commit-gate cut (`BRAIN_EMOTION_CATEGORY_COMMIT_LESION`), not the
  Stage-2 gate's `lesion_affect` (`BRAIN_AFFECT_LESION`), which also moves the production ladder's sentence before the clause.
  "6 of 6" is this file's reading of "stays load-bearing" and is the first thing for Claude to confirm or change.

## 3 Protocol

**Gate revision F.** The tip of `main` after Claude merges this file. Every gate line pins F (the pool's revision directory
`~/derisk-pool/revisions/<F>`, as the Stage-2 lines do). Gate seeds 42, 43, 44, 100, 101, 102 only, and only from lines emitted
against this committed prereg.

**Stage-2 emo lines.** `python -m research.runners.stage2_exit_gate --emit-pool-jobs <F> --seeds 42,43,44,100,101,102
--no-informational`, filtered to group `emo`: arms `intact_a`, `intact_b`, `permuted`, `lesion_affect`, `lesion_affect.rep0`,
6 seeds, 30 lines per flag state. ON: `BRAIN_EMOTION_CATEGORY_BOUNDED=1` added to the line's `env` block and `--out` under
`research/findings/raw/_emo_bounded_flip/gate_on/`. OFF: the variable absent (unset, not "0", as the 10-26 handler shards were)
and `--out` under `.../gate_off/`. Nothing else differs: the runner's frozen harness (numpy, one BLAS thread, stub renderer,
tiny-demo +LTM, `BRAIN_CHAT_SEED`, `PYTHONHASHSEED=0`), `mem_gb` 9 under a 10 GB memcap. 60 lines.

**Row lines.** `python tools/lb_shard.py jobs --seeds 42 43 44 100 101 102 --tag emo-bounded-flip-gate-on --no-fixes --probe-set thin
--faculties emotion-category --extra-env BRAIN_EMOTION_CATEGORY_BOUNDED=1` and the same with tag `...-gate-off` and no
`--extra-env`: 12 lines, each four brain builds (intact twice, lesion twice). Total 72 lines; the ASK states the emitted TSV's
count before queueing.

**Failure handling.** Pass means B1 to B8 all hold. Anything else: STOP with the data (the ruling's words); no retuning on gate
seeds and no second flag. A missing or UNDEFINED cell is re-run once at the same revision, logged before its result is read; a
still-UNDEFINED cell never counts.

## 4 Dev-seed smoke (seeds 7 and 8), recorded before this file

Smoke-first (RULING `smoke-first-1018`): the flag ON against the flag unset, on the two dev seeds, scored by
`tools/answer_rate.py paired --seeds 7,8`, in the `_load_bearing/_shards` layout the scorer reads. Revision `ec8fe543f` (the
`emotion-category` row). Tags `emo-bounded-on-ec8fe54` / `emo-bounded-off-ec8fe54`, env otherwise production defaults (`--no-fixes
--probe-set thin`), `--repeats 2`, CPU numpy, under `mem_ok`/`memcap`.

| record | faculty dir | seed | flag ON answered | flag unset answered | status |
|---|---|---|---|---|---|
| `research/findings/smoke/emo-bounded-flip-s78.json` | `emotion-category` | 7 | 2/2 | 2/2 | ok |
| `research/findings/smoke/emo-bounded-flip-s78.json` | `emotion-category` | 8 | 2/2 | 2/2 | ok |
| `research/findings/smoke/emo-bounded-flip-affect-coloring-s78.json` | `affect-coloring` | 7 | 2/2 | 2/2 | ok |
| `research/findings/smoke/emo-bounded-flip-affect-coloring-s78.json` | `affect-coloring` | 8 | 2/2 | 2/2 | ok |

The two probes per side are the `intact_a` and `intact_b` arms of that cell. Neither record carries FLIP-HURTS-ANSWERS (`flags`
is empty in both), and no probe on either side is an abstention or an empty reply. All eight cells ran rc=0 (shards committed in
`7d7c365c4` under `research/findings/raw/_load_bearing/_shards/emo-bounded-on-ec8fe54/` and `.../emo-bounded-off-ec8fe54/`, with
sidecars). Read beside the records, not as bars:

- The `emotion-category` row names `joy` on both seeds with the flag ON and with it unset, in both intact arms. With the flag ON
  every feel read shows `bounded_flag` True, `bounded_env` "1", `bounded_readback` True and `oor_count_session` 0 (both seeds);
  unset it shows False, none, False. On seed 8 with the flag unset `oor_count_session` is 1 (the unbounded organ's out-of-range
  excursion that the de-risk finding reports on 5 of 8 dev builds; joy is still named there, which is why dev seeds do not
  discriminate).
- The row reads reply-level load-bearing (`load_bearing` true, null clean, lesion reproduced) in all four `emotion-category`
  cells; the lesioned reply differs from the intact one only by the category sentence ("No single emotion channel latched on my
  affect ladder." for "The emotion channel that latched on my affect ladder is joy.").
- The `affect-coloring` row reads `pass` (not load-bearing) in all four cells on its default `well` probe, flag ON and unset
  alike: that probe carries no affect, which is why the instrument has its own `LB_AFFECT_DRIVE_PROBE` remap to the `emo` turn
  for this row. It is used here for its reply (a non-feel affect surface), not for its load-bearing status.

## 5 What this does not claim

- No gate-seed result and no verdict; the smoke is two dev seeds and only shows the flag does not make a reply disappear.
- The `answer_rate` probe of the `emotion-category` row is its feel turn, which the handler answers from the inner-state read-out
  before any recall path runs; it can show a broken or empty read, not a recall regression. That is why the second record is the
  `affect-coloring` row (a non-feel affect reply, the `well` turn): it asks whether the flag leaks outside the feel path.
- The de-risk's own caveats stand (finding above): seeds 101, 202, 242, 308, 348 and the held-out 301-360 are design seeds of the
  repair; the pinned failure rate is about 4% of builds, so a 6/6 gate pass alone proves little. The evidence for the flip is the
  paired organ-level comparison and the injection stress in that finding plus the bars above, not the gate alone.
- The category is a functional read of which channel latched (CLAUDE.md non-negotiable 5), never a claim of felt experience.

## 6 Owed before any gate line is queued

- The bar-sanity record cited in the header is the registered `emo` group verdict under the real aggregate, regenerated
  2026-10-27 from `docs/drafts/stage2-exit/bar_sanity_spec_s2_c4_emo.py` because the Stage-2 records predate edits to
  `research/runners/stage2_exit_gate.py` (verdict OK: the perfect brains pass, every null brain fails). It covers B1 to B3 only.
  B4 to B8 have no bar-sanity record: no scorer reads B4 or the whole-record form of B6 yet, so the record cannot be made until
  one exists.
- A scorer for B4 (the engagement keys on every ON feel read) and the whole-record compare of B6 (apart from the five keys),
  with tests and a bar-sanity record each, filed as an amendment before gate lines are emitted.
- Gate revision F, and the emitted TSV.

## Amendment 1 (2026-10-27): the readers section 6 owed, their bar-sanity records, B8 confirmed, gate revision F

bar-sanity: docs/drafts/emo-bounded-flip/bar_sanity_record_b4.json
bar-sanity: docs/drafts/emo-bounded-flip/bar_sanity_record_b6.json

This amendment changes no bar. Sections 1 to 5 stand word for word; B1 to B8 and DEC are the ruling's words. It discharges
section 6 (the readers, their tests and bar-sanity records, gate revision F, the emitted TSV) and fixes two readings of a
scorer, before any gate-seed run. No gate seed has been run for this flip.

**Scorer commands** (repo root, as modules so provenance sidecars are written; tests: `tests/test_emo_flip_score.py`, 24).
- All eight bars and DEC in one read: `python -m research.runners.emo_flip_score --on-dir research/findings/raw/_emo_bounded_flip/gate_on
  --off-dir research/findings/raw/_emo_bounded_flip/gate_off --out research/findings/raw/_emo_bounded_flip/verdict.json --pin <F, full sha>`.
  It prints PASS, FAIL or UNDEFINED per bar with the shard and turn behind each, then `DEC`. Defaults are this protocol's: row tags
  `emo-bounded-flip-gate-on` / `-off` under `research/findings/raw/_load_bearing/_shards`, the 10-26 reference
  `research/findings/raw/_stage2_exit/gate/aggregate.json`. `--pin` makes each row cell's sidecars a `git_archive` measurement of F
  (lb_shard's B2b rule; a cell that fails it is UNDEFINED); without it that check is not made.
- The pieces, as the prereg named them, read the same cells: B1 to B3 `python -m research.runners.stage2_exit_gate --aggregate
  <ON dir> --seeds 42,43,44,100,101,102` (emo group; the combined scorer calls the same function); B5 the same on the OFF
  directory against the reference; B7 `python tools/answer_rate.py paired --on-tag emo-bounded-flip-gate-on --off-tag
  emo-bounded-flip-gate-off --faculty emotion-category` and `python -m research.runners.emotion_category_dev --flip-check <ON>
  <OFF> --seeds 42,43,44,100,101,102` (its `FLIP-HURTS-ANSWERS` only, reading 1 below); B4, B6, B8 `--only B4`, `--only B6`, `--only B8`.
- Dev-seed read for `tools/gate_preflight.py`: the same command with `--seeds 7 --only B1,B2,B3,B4,B6` over dev-seed shards (B5 has no
  10-26 reference at a dev seed).
- Status rule for every bar (how a cell that cannot show the bar is read; no threshold moves): FAIL if any defined cell breaks the
  bar; else UNDEFINED if any shard, turn, key or pair is missing, errored or unreadable (never a pass); else PASS. DEC is PASS only
  when all eight read PASS, and then reads "flip the default ON before the 11-21 re-pin"; anything else reads "STOP with the data"
  and names the FAIL and the UNDEFINED bars. A B5 mismatch is UNDEFINED (the OFF half does not reproduce the 10-26 gate) and holds
  the ON half unread, as section 2 says.

**Two readings fixed here.**
1. B6, the feel half. Section 2 reads `FEEL-NOT-APPEND-ONLY` of `--flip-check` as the feel-text check. That flag asks whether the ON
   text is the OFF text plus exactly one clause, which holds when the OFF arm has no clause (the organ flag `BRAIN_EMOTION_CATEGORY=0`).
   Here both arms append a clause ("... is joy." or "No single emotion channel latched ..."), so it fires on every feel turn of
   a perfect brain, and B6 would be unreachable. The ruling's own words are read instead: the feel text before the category clause
   is identical ON vs OFF (the clause may differ), by `score_b6`; pinned by `test_the_prereg_flip_check_feel_flag_fires_on_a_perfect_pair_which_is_why_b6_does_not_read_it`.
   The non-feel half is the whole record: every field at every depth of every non-feel turn, after the five engagement keys are
   dropped at any depth from both sides; a sixth `bounded_*` key is a difference; the first differing path is printed. That
   subsumes `FLIP-CHANGES-NONFEEL` (answer, abstained, recalled_svo only). On the six 10-26 gate shards `intact_a` and `intact_b`
   differ at 0 paths whole-record, so a deterministic build can pass it.
2. B4 checks the four fields section 2's scoring text lists (`bounded_flag` True, `bounded_env` "1", `bounded_readback` True,
   `oor_count_session` the integer 0) on every `feel_*` response of every ON shard. B8 adds a validity condition and no bar: a flag-ON
   row cell whose intact arms do not show `bounded_flag` and `bounded_readback` True is UNDEFINED, not a measurement of the flag ON.

**Bar-sanity records** (`tools/bar_sanity.py`, n = 1000 per generator, seed 0, decide = the scorer itself on synthetic shards with
the production reply surfaces; specs `docs/drafts/emo-bounded-flip/bar_sanity_spec_b4.py` and `_b6.py`):

| bar | record | verdict | perfect brains | null brains |
|---|---|---|---|---|
| B4 | `bar_sanity_record_b4.json` | OK | 3, each passes 1.000 | 19, each passes 0.000 |
| B6 | `bar_sanity_record_b6.json` | OK | 4, each passes 1.000 | 20, each passes 0.000 |

A first trial of the B6 spec read UNREACHABLE for the perfect brain; that was the spec's own synthetic builder desynchronising its
random stream between the ON and OFF builds when the organ latched, fixed in the builder before the records above. B1 to B3 stay
covered by `bar_sanity_record_c4_emo.json`. B5, B7 and B8 read existing instruments (an aggregate comparison, `flip_check` plus
`answer_rate`, and the lb_shard row records); they have no record here and are not claimed to.

**B8 confirmed.** Claude confirmed on 2026-10-27 that section 2's reading of "stays load-bearing" as "6 of 6" seeds stands.

**Gate revision F.** F is `7bf5bd7dd355b79c269b5b204c20465e02b10443`, the commit that contains the scorers (it supersedes section 3's
"the tip of main after Claude merges this file"). It does not contain current main, and the pool provisioner refuses a revision that does
not; so when this branch is merged, the F put in the lines is the tip of main that contains it, which must carry
`research/runners/emo_flip_score.py` byte-identical (sha256 `5d5f96d76eaab47aeb7fa8cffe3595cd4efb9faca59b18168a41e7d6257239cf`),
and the harness files the lines run (`research/runners/stage2_exit_gate.py` sha256 `67bfd30b66d266748d9e06b0ef393d23ab03f5be3b22b2e84831ec094616a177`,
the emotion-category organ, `tools/lb_shard.py`, `tools/answer_rate.py`, `research/runners/load_bearing_fraction.py`, the
emotion-category lbf row) unchanged from F. Checked 2026-10-27: at the server's main `d3bb83adb`, each of those files is identical to F's.

**Gate lines.** `research/coordination/emo-bounded-flip-gate-lines.txt`, 72 pool lines (TSV: command, then the `--checked` reason that
names `prereg research/findings/2026-10-27-emo-bounded-flip-PREREGISTRATION.md`), emitted by `python -m research.runners.emo_flip_lines
[<F>]` with `<sha>` left as a placeholder: 60 Stage-2 emo lines (6 gate seeds x 5 arms, ON and OFF; the runner's own lines with `--prereg`
and `--out` edited and, ON only, `BRAIN_EMOTION_CATEGORY_BOUNDED=1` added to the env block) and 12 row lines (`tools/lb_shard.py jobs`
as section 3 gives it, wrapped in `mem_ok.sh 9 1` and `memcap.sh 12`). The Stage-2 lines declare `mem_gb=9` under a 10 GB memcap, as the
10-26 gate lines did; the row lines declare `mem_gb=9` as well (the same tiny-demo brain) under a 12 GB cap, as earlier lb_shard row lines,
because the battery's parent process shares the arm subprocess's cgroup and the dev smoke recorded no peak for this row. Nothing is
queued by this amendment. Before `pool_queue.sh` accepts a gate-seed line it still needs a passing pool canary of
`research.runners.stage2_exit_gate` and of `research.runners.load_bearing_fraction` at F, and a `gate_preflight` read of this prereg at F.
