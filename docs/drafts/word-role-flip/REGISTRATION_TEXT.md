---
type: draft
status: draft
date: 2026-10-16
packet: word-role-flip
runner: research/runners/lbf_rows/word_role_parse.py, research/runners/lbf_rows/word_role_commit.py, research/runners/iw1_default_identity.py, research/runners/iw2_default_identity.py
scorer: tools/answer_rate.py, research/runners/iw1_default_identity.py --compare, research/runners/iw2_default_identity.py --compare
---
# word-role flip: registration text held by Claude (to be committed as research/findings/2026-10-16-word-role-flip-PREREGISTRATION.md once its bar-sanity records exist)

> **PROMOTED 2026-10-16** to `research/findings/2026-10-16-word-role-flip-PREREGISTRATION.md` (with the per-row F1
> engagement rule and the row lift, RULING `wrc-f1a-engagement`). That file is the registration; this draft is history.

Written by Claude before any gate-seed run of this flip. It promotes the reviewed packet
`docs/drafts/word-role-flip/PACKET.md` (branch `research/wrc-flip-packet`) as approved by RULINGs
`wrc-packet-review-1014`/`1014b`, `wrc-packet-answers-1014`, `wrc-f3-1015`, `wrc-f3-verdict-1015` and
`wrc-prereg-go-1016`/`1016b`. A delegate draft of this file (2026-10-16 09:58, never committed) was rejected
because it loosened every bar; this text replaces it. Bars here are the packet's bars, word for word where the
packet fixed them. This would be the first live seam retired from production chat.

## §1 What flips

| flag | old default | new default | anchor (`research/runners/brain_chat_tui.py`) |
|---|---|---|---|
| `BRAIN_WORD_ROLE_PARSE` | OFF | **ON** | `_WORD_ROLE_PARSE_DEFAULT_ON` (line 279) |
| `BRAIN_WORD_ROLE_COMMIT` | OFF | **ON** | `_WORD_ROLE_COMMIT_DEFAULT_ON` (line 315) |

Both anchors flip in ONE commit, `9919725a9` on `research/wrc-flip-packet` (the commit flag is read only when
the parse flag is on). `BRAIN_WORD_ROLE_OP` and `BRAIN_WORD_ROLE_COMMIT_OP` stay 0. An explicit `=0` on both
flags is the escape: production then parses with the positional BridgeParser and the count rule, as before the
flip. The only production reader is `brain_chat_tui` (`_neural_question_parse`); webapp chat goes through
ChatBrain, so production chat is the intended blast radius.

**Lesions (separate from the flags):** PARSE lesion `BRAIN_WORD_ROLE_LESION=1` closes the word -> role gate;
COMMIT lesion `BRAIN_WORD_ROLE_COMMIT_LESION=1` closes the role -> commit-gate transmission. A lesion arm runs
at the flipped default with only that one env var added.

**Gate revision F:** the tip of `research/wrc-flip-packet` after this prereg (merged from main) and the
Amendment-E code (§2, E) are on it. Every gate line pins F; gate seeds 42, 43, 44, 100, 101, 102 only.

## §2 Bars, fixed before any gate-seed data

**F1 at the default (reply load-bearing).** Both LBF rows (`research/runners/lbf_rows/word_role_parse.py`,
`research/runners/lbf_rows/word_role_commit.py`) at the flipped default (no REQUIRED_ENV override): reply-level
load-bearing on **6 of 6 seeds**, meaning in every DEFINED cell the intact reply differs from that row's lesion
reply (reply fields per `docs/TERMS.md` `load-bearing`). A DEFINED cell that is not reply-load-bearing fails the
seed. A cell failing engagement (organ route `word_role_commit`, no error, `commit_lesioned` matching the arm)
is UNDEFINED and never counts.

**F2 identity (default IS the capability).** `default` (no flags set) equals `on:both` (both flags `=1`
explicitly) on every script turn of iw1 and iw2 at the flip revision, byte-identical apart from the runners'
existing volatile-key drops. **Already measured, IDENTICAL on both runners** at `ce2e028`:
`research/findings/raw/_iw1/identity/ce2e028/compare_default_onwr.json`,
`research/findings/raw/_iw2/identity/ce2e028/compare_default_onwr.json`.

**F3 identity (the escape reproduces pre-flip).** `off:word-role` (ONLY the two word-role flags `=0`, every
other flag at its default) at F equals `default` at P' on every script turn (sha_canon), where
P' = `b1c11a0887b3c17ff7198674e9c33f162a3d58b9` is main with shared-LTM ON and both word-role anchors still OFF.
**Already measured, IDENTICAL on both runners** (run pin `8407f2193`):
`research/findings/raw/_iw1/identity/8407f21/compare_offwr_vs_pprime.json`,
`research/findings/raw/_iw2/identity/8407f21/compare_offwr_vs_pprime.json`. The first F3, against
P = `96e5198f5`, is UNDEFINED and does not count (confounded by shared-LTM default ON, RULING `wrc-f3-verdict-1015`).

**F4 off-target (target specificity).** The nine listed statement-form turns, exact strings fixed in
2026-10-14 (packet amendment A): "the cat eats fish" / "the cat precedes the ball" / "the ball precedes the
worm" / "Sally puts the marble in the basket." / "Sally leaves the room." / "Anne moves the marble to the box." /
"the wolf hunts the deer" / "the dog and the cat walked in" / "the cat and the dog walked in". They sit in the
iw1 and iw2 scripts, so both runners run three arms each at F: intact-at-default, PARSE lesion, COMMIT lesion.
Bar: every listed reply (`answer` / `abstained` / `recalled_svo`) is identical to intact-at-default under BOTH
lesions, on **6 of 6 seeds**. One changed listed reply fails the bar (the lesion leaks outside the parse path).
**Precondition (amendment E):** the intact-at-default arm persists each listed turn's parse route as `wr_route`
(`engaged` / `declined` / `not-reached`, from the per-turn `word_role_trace`; readout
`research/runners/wr_route.py`). The `for` field is NOT accepted as this readout. A listed turn whose route reads
`engaged` on any seed makes F4 UNDEFINED plus a NEEDS line, never a silent drop.

**Answer rate (FLIP-HURTS-ANSWERS).** `tools/answer_rate.py` paired layout, `--on-tag wrcflip --off-tag
wrcflip-alloff`, once with `--faculty word-role-parse` and once with `--faculty word-role-commit`. The on arm runs
at the flipped default; the alloff arm forces both flags `=0` at F. `flags: FLIP-HURTS-ANSWERS` on ANY measured
seed is reported beside the verdict, changes no PASS/FAIL by itself, and requires a NEEDS line and a Claude
ruling before the flip stands. Unmeasured seeds are listed, never read as "no harm".

**B1, scoped (battery).** At-risk rows are those whose reply differs between `default` and `off:word-role`
at the SAME revision F, dev seed 7, on every iw1 and iw2 turn. The packet's earlier reading (zero at-risk rows,
29 of 29 turns unchanged) compared two revisions and its source files are not committed, so it is not used.
**Pre-gate step:** run iw1 and iw2 `default` and `off:word-role` at F on dev seed 7 (4 lines), commit the four
outputs with sidecars, and list every at-risk row before any gate seed runs. Zero rows: B1 is met by the F2/F3
identity record. Any row: on the six gate seeds that row's n_load_bearing at the default must be >= the alloff
arm's at F; a drop reverts the flip (battery prereg section 6).

**Reported, never gated:** the dual-role decline rate at the default (right / wrong / declined per seed, plus
the exclusion-lesion co-commit count). The comparison point is the finding's count-mode-era figure (17 of 24
right, 0 wrong, 7 declined).

**Bar sanity (F1 is reachable).** The commit stage's own gate recorded reply load-bearing on 6 of 6 seeds at
explicit ON, with 6 of 6 no-fact controls ok: `research/findings/raw/_s2p/word-role-commit/c64f2c7/verdicts/R.json`
(finding `research/findings/2026-10-14-word-role-commit-gate.md`). The parse stage: O and R PASS over 6 seeds at
op 0, `research/findings/raw/_s2p/word-role-parse/9ea7411/verdicts/O.json` and
`research/findings/raw/_s2p/word-role-parse/9ea7411/verdicts/R.json` (finding
`research/findings/2026-10-06-word-role-parse-gate.md`). F1 here re-measures both at the default, at F.

## §3 Job shape

Pool lines, `SIM_BACKEND=numpy`, `max_age_h` stamped, sidecars required, one revision F for every line:
F1, 2 rows x 6 seeds; answer rate, 2 tags x 6 seeds (the on-tag may reuse F1's intact output only if the scorer
reads it unchanged); F4, 3 arms x 2 runners x 6 seeds; decline read-out, 6 seeds. Memory from the smoke peak
(`mem_gb` = ceil(1.3 x peak)). The emitted TSV's line count and arm list are stated in the ASK before queueing.
Above about 40 projected job-hours, an [aws-run] ASK is filed and Claude runs the AWS share (method of Stage-1
attempt 2).

## §4 Verdict and failure handling

PASS iff F1, F2, F3 and F4 all pass and no FLIP-HURTS-ANSWERS stands without a Claude ruling. Any bar failing at
the gate stage: both defaults REVERT to OFF with a finding (one flag may stand alone only by Claude ruling); no
retuning on gate seeds. A missing or UNDEFINED cell is re-run once at the same revision, logged before its
result is read; a still-UNDEFINED cell never counts, and a bar that cannot reach 6 defined seeds is UNDEFINED
plus NEEDS. The flip merges to main only after PASS.

## §5 What this does not claim

- No result: the F2/F3 records marked "already measured" are pre-gate dev-seed identity records; the gate RESULT
  is filed once the shards exist.
- The residual host steps named in `research/findings/2026-10-14-word-role-commit-gate.md` ("Not yet a live
  seam") remain at ON: the commit read (`spiking_commit.py:181-196`, threshold-invariant, not deciding), the
  slot -> word mapping, the a==v guard, self-alias resolution and ROLE_MIN folded into W_RG.
- No host-decision-share or ownership claim (`docs/plans/2026-09-25-prove-who-owns-the-computation-PLAN.md` is
  separate); no scale or latency claim; every reply-level read-out stays within the honesty boundary.
